from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl, write_jsonl
from longcontext.schema import Choice, Evidence, LCQASample


DEFAULT_MODEL = "gpt-5.5"


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def build_prompt(sample: LCQASample, max_context_chars: int | None) -> str:
    context = sample.document.context
    truncated = False
    if max_context_chars is not None and len(context) > max_context_chars:
        half = max_context_chars // 2
        context = (
            context[:half]
            + "\n\n[... context truncated for smoke test ...]\n\n"
            + context[-(max_context_chars - half) :]
        )
        truncated = True

    parts = [
        "You are a careful long-context QA teacher model.",
        "Answer strictly from the provided document. If the document is insufficient, say so.",
        "Return a compact JSON object with keys: answer, label, evidence, notes, "
        "needs_long_context, quality_score, evidence_spread.",
        "",
        f"Sample ID: {sample.id}",
        f"Length bucket: {sample.length.length_bucket}",
        f"Input tokens estimate: {sample.length.input_tokens}",
        f"Context was truncated for this run: {truncated}",
        "",
        "Document:",
        context,
        "",
        "Question:",
        sample.input.question,
    ]
    if sample.input.instruction:
        parts.extend(["", "Instruction:", sample.input.instruction])
    if sample.input.choices:
        parts.extend(["", "Choices:"])
        parts.extend(format_choice(choice) for choice in sample.input.choices)
    return "\n".join(parts)


def format_choice(choice: Choice) -> str:
    return f"{choice.label}. {choice.text}"


def parse_teacher_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"answer": text, "teacher_parse_error": "not_json"}
    if not isinstance(parsed, dict):
        return {"answer": text, "teacher_parse_error": "json_not_object"}
    return parsed


def stringify_answer(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def parse_evidence(value: Any) -> list[Evidence]:
    if not isinstance(value, list):
        return []

    evidence: list[Evidence] = []
    for item in value:
        try:
            if isinstance(item, str):
                evidence.append(Evidence(text=item))
            elif isinstance(item, dict) and item.get("text"):
                evidence.append(Evidence.model_validate(item))
        except Exception:
            continue
    return evidence


def call_openai(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float | None,
    timeout_seconds: float | None,
) -> tuple[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
    response = client.responses.create(**kwargs)
    return response.output_text, response


def label_row(
    row: dict[str, Any],
    *,
    client: Any,
    model: str,
    max_output_tokens: int,
    temperature: float | None,
    timeout_seconds: float | None,
    max_attempts: int,
    backoff_seconds: float,
    max_context_chars: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    sample = LCQASample.model_validate(row)
    prompt = build_prompt(sample, max_context_chars)
    teacher_meta: dict[str, Any] = {
        "teacher_model": model,
        "teacher_status": "dry_run" if dry_run else "pending",
        "input_tokens": sample.length.input_tokens,
        "length_bucket": sample.length.length_bucket,
        "max_context_chars": max_context_chars,
    }

    if dry_run:
        sample.quality.metadata["teacher"] = teacher_meta
        return sample.model_dump(mode="json")

    started = time.perf_counter()
    last_error: Exception | None = None
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                text, response = call_openai(
                    client,
                    model=model,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                )
                teacher_meta["attempts"] = attempt
                break
            except Exception as exc:
                last_error = exc
                teacher_meta["attempts"] = attempt
                if attempt >= max_attempts:
                    raise
                time.sleep(backoff_seconds)
        else:
            raise RuntimeError("Teacher labeling failed without an exception.") from last_error

        latency_seconds = time.perf_counter() - started
        parsed = parse_teacher_text(text)
        answer = stringify_answer(parsed.get("answer") if "answer" in parsed else text)
        label = parsed.get("label")
        sample.output.answer = answer
        sample.output.label = str(label).strip() if label else sample.output.label
        sample.output.answer_type = "generated"
        parsed_evidence = parse_evidence(parsed.get("evidence"))
        if parsed_evidence:
            sample.evidence = parsed_evidence
        sample.quality.status = "labeled"
        sample.quality.annotator = model
        needs_long_context = parsed.get("needs_long_context")
        quality_score = parsed.get("quality_score")
        sample.quality.needs_long_context = (
            needs_long_context if isinstance(needs_long_context, bool) else None
        )
        try:
            sample.quality.quality_score = (
                float(quality_score) if quality_score is not None else None
            )
        except (TypeError, ValueError):
            sample.quality.quality_score = None
        evidence_spread = parsed.get("evidence_spread")
        sample.quality.evidence_spread = str(evidence_spread) if evidence_spread else None
        teacher_meta.update(
            {
                "teacher_status": "success",
                "latency_seconds": round(latency_seconds, 3),
                "response_id": getattr(response, "id", None),
                "output_chars": len(text),
                "notes": parsed.get("notes"),
                "parse_error": parsed.get("teacher_parse_error"),
                "teacher_response": text,
            }
        )
    except Exception as exc:
        teacher_meta.update(
            {
                "teacher_status": "failed",
                "teacher_error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": round(time.perf_counter() - started, 3),
            }
        )
    sample.quality.metadata["teacher"] = teacher_meta
    return sample.model_dump(mode="json")


def iter_labeled_rows(args: argparse.Namespace, config: dict[str, Any]):
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {}
    base_url = (
        args.base_url
        or config.get("base_url")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_ENDPOINT")
    )
    base_url = normalize_base_url(base_url)
    if base_url:
        client_kwargs["base_url"] = base_url
    client = None if args.dry_run else OpenAI(**client_kwargs)

    model = args.model or config.get("teacher_model") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    max_output_tokens = args.max_output_tokens or int(config.get("max_output_tokens", 4096))
    timeout_seconds = args.timeout_seconds or config.get("timeout_seconds")
    retry_config = config.get("retry") or {}
    max_attempts = int(getattr(args, "max_attempts", None) or retry_config.get("max_attempts", 1))
    backoff_seconds = float(
        getattr(args, "backoff_seconds", None) or retry_config.get("backoff_seconds", 0)
    )
    temperature = args.temperature
    if temperature is None and "temperature" in config:
        temperature = float(config["temperature"])

    stats = {"processed": 0, "success": 0, "failed": 0, "skipped_training_ineligible": 0}
    for row in tqdm(read_jsonl(args.input), desc="label", unit="sample"):
        sample = LCQASample.model_validate(row)
        if sample.quality.training_eligible is False and not args.include_training_ineligible:
            stats["skipped_training_ineligible"] += 1
            continue
        if args.limit is not None and stats["processed"] >= args.limit:
            break
        stats["processed"] += 1
        labeled = label_row(
            row,
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            max_context_chars=args.max_context_chars,
            dry_run=args.dry_run,
        )
        teacher_status = (
            labeled.get("quality", {}).get("metadata", {}).get("teacher", {}).get("teacher_status")
        )
        if teacher_status == "success":
            stats["success"] += 1
        elif teacher_status == "failed":
            stats["failed"] += 1
        yield labeled
    print(
        f"labeling complete; processed={stats['processed']}, "
        f"success={stats['success']}, failed={stats['failed']}, "
        f"skipped_training_ineligible={stats['skipped_training_ineligible']}"
    )
    if args.fail_on_error and stats["failed"]:
        raise SystemExit(f"Teacher labeling failed for {stats['failed']} sample(s).")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Label LCQA JSONL with OpenAI Responses API.")
    parser.add_argument("input", help="Input LCQA JSONL.")
    parser.add_argument("output", help="Output labeled LCQA JSONL.")
    parser.add_argument("--config", default="configs/labeling/openai_teacher_example.yaml")
    parser.add_argument("--model", help=f"OpenAI model name. Default: {DEFAULT_MODEL}.")
    parser.add_argument("--base-url", help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--backoff-seconds", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        help="Truncate context for smoke tests. Omit for full-context labeling.",
    )
    parser.add_argument("--include-training-ineligible", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate plumbing without API calls.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any API call fails.")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run is used.")

    config = load_config(args.config)
    write_jsonl(args.output, iter_labeled_rows(args, config))
    print(f"Wrote labeled LCQA data to {args.output}")


if __name__ == "__main__":
    main()

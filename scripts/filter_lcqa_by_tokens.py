from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longcontext.io import read_jsonl
from longcontext.schema import LCQASample


def build_text(sample: LCQASample, token_field: str) -> str:
    if token_field == "question_text":
        return sample.input.question
    if token_field == "context":
        return sample.document.context
    if token_field == "answer":
        return sample.output.answer
    if token_field in {"full_question", "prompt"}:
        parts = [sample.document.context, sample.input.question]
        if sample.input.choices:
            parts.extend(f"{choice.label}. {choice.text}" for choice in sample.input.choices)
        return "\n\n".join(part for part in parts if part)
    raise ValueError(f"Unsupported token field: {token_field}")


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def update_lengths(
    sample: LCQASample,
    tokenizer_name: str,
    tokenizer,
    token_field: str,
    compute_all_lengths: bool,
) -> LCQASample:
    sample.length.tokenizer = tokenizer_name
    if compute_all_lengths or token_field == "context":
        sample.length.context_tokens = count_tokens(tokenizer, sample.document.context)
    if compute_all_lengths or token_field == "question_text":
        sample.length.question_tokens = count_tokens(tokenizer, sample.input.question)
    if compute_all_lengths or token_field == "answer":
        sample.length.answer_tokens = count_tokens(tokenizer, sample.output.answer)
    if compute_all_lengths or token_field in {"full_question", "prompt"}:
        sample.length.prompt_tokens = count_tokens(tokenizer, build_text(sample, "prompt"))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter LCQA JSONL by token length.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--token-field",
        choices=("full_question", "question_text", "context", "prompt", "answer"),
        default="full_question",
        help=(
            "Field used for filtering. Default is the full training question/user input: "
            "context + question + choices. 'prompt' is kept as an alias."
        ),
    )
    parser.add_argument("--min-tokens", type=int, default=32_000)
    parser.add_argument("--max-tokens", type=int, default=900_000)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N rows.")
    parser.add_argument(
        "--compute-all-lengths",
        action="store_true",
        help="Also fill context/question/prompt/answer token counts. Slower for long contexts.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: transformers\n"
            "Install project dependencies in the active Python environment:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    total = 0
    kept = 0
    min_seen = None
    max_seen = None
    token_sum = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in read_jsonl(args.input):
            if args.limit is not None and total >= args.limit:
                break
            total += 1
            sample = LCQASample.model_validate(row)
            token_count = count_tokens(tokenizer, build_text(sample, args.token_field))
            sample = update_lengths(
                sample,
                args.tokenizer,
                tokenizer,
                token_field=args.token_field,
                compute_all_lengths=args.compute_all_lengths,
            )
            min_seen = token_count if min_seen is None else min(min_seen, token_count)
            max_seen = token_count if max_seen is None else max(max_seen, token_count)
            token_sum += token_count

            if args.min_tokens <= token_count <= args.max_tokens:
                sample.quality.status = "filtered"
                f.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
                kept += 1

    stats = {
        "input": args.input,
        "output": args.output,
        "tokenizer": args.tokenizer,
        "token_field": args.token_field,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "min_seen": min_seen,
        "max_seen": max_seen,
        "avg_seen": token_sum / total if total else None,
    }

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

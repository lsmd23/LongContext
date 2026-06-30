from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from longcontext.length import get_length_bucket
from longcontext.schema import LCQASample


CONTEXT_FIELD_HINTS = (
    "context",
    "document",
    "documents",
    "passage",
    "passages",
    "article",
    "articles",
    "text",
    "content",
    "body",
    "source",
    "sources",
    "background",
    "paragraph",
    "paragraphs",
    "paper",
    "book",
    "code",
    "input",
    "prompt",
)

QUESTION_FIELD_HINTS = (
    "question",
    "query",
    "instruction",
    "task",
    "problem",
    "prompt",
    "input",
    "request",
)

ANSWER_FIELD_HINTS = (
    "answer",
    "answers",
    "response",
    "output",
    "target",
    "targets",
    "label",
    "labels",
    "summary",
    "completion",
)

TITLE_FIELD_HINTS = ("title", "name", "heading")
ID_FIELD_HINTS = ("id", "_id", "uid", "uuid", "guid", "qid", "example_id")

BENCHMARK_DATASETS = (
    "longbench",
    "infinitebench",
    "loogle",
    "needle",
    "ruler",
    "babilong",
    "longeval",
)


@dataclass
class FieldMapping:
    context_field: str | None = None
    question_field: str | None = None
    answer_field: str | None = None
    title_field: str | None = None
    id_field: str | None = None


@dataclass
class DiscoveryDecision:
    status: str
    reason: str
    candidate_type: str
    field_mapping: FieldMapping
    input_tokens: int | None = None
    length_bucket: str | None = None


def flatten_text(value: Any, max_items: int = 20) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [flatten_text(item, max_items=max_items) for item in value[:max_items]]
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key, item in list(value.items())[:max_items]:
            text = flatten_text(item, max_items=max_items)
            if text:
                parts.append(f"{key}: {text}")
        return "\n\n".join(parts)
    return str(value)


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def field_score(name: str, hints: Iterable[str]) -> int:
    normalized = normalize_field_name(name)
    score = 0
    for rank, hint in enumerate(hints):
        hint_norm = normalize_field_name(hint)
        if normalized == hint_norm:
            score += 100 - rank
        elif hint_norm in normalized:
            score += 50 - rank
    return score


def text_size_score(value: Any) -> int:
    return min(len(flatten_text(value)), 100_000)


def choose_field(row: dict[str, Any], hints: Iterable[str], used: set[str] | None = None) -> str | None:
    used = used or set()
    candidates = []
    for key, value in row.items():
        if key in used:
            continue
        text = flatten_text(value)
        if not text:
            continue
        score = field_score(key, hints)
        if score <= 0:
            continue
        candidates.append((score, text_size_score(value), key))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def choose_context_field(row: dict[str, Any], used: set[str] | None = None) -> str | None:
    used = used or set()
    hinted_candidates = []
    for key, value in row.items():
        if key in used:
            continue
        text = flatten_text(value)
        if not text:
            continue
        score = field_score(key, CONTEXT_FIELD_HINTS)
        if score <= 0:
            continue
        size = len(text)
        # Context selection should prefer real long text over short provenance fields such as "source".
        hinted_candidates.append((min(size, 100_000) + score * 100, size, score, key))
    if hinted_candidates:
        hinted_candidates.sort(reverse=True)
        best_score, best_size, _, best_key = hinted_candidates[0]
        if best_size >= 200 or best_score >= 20_000:
            return best_key

    text_fields = [
        (text_size_score(value), key)
        for key, value in row.items()
        if key not in used and len(flatten_text(value)) >= 1_000
    ]
    if not text_fields:
        return None
    text_fields.sort(reverse=True)
    return text_fields[0][1]


def infer_field_mapping(row: dict[str, Any]) -> FieldMapping:
    id_field = choose_field(row, ID_FIELD_HINTS)
    title_field = choose_field(row, TITLE_FIELD_HINTS)
    question_field = choose_field(row, QUESTION_FIELD_HINTS, used={id_field} if id_field else set())
    used = {field for field in (id_field, title_field, question_field) if field}
    context_field = choose_context_field(row, used=used)
    used = {field for field in (id_field, title_field, question_field, context_field) if field}
    answer_field = choose_field(row, ANSWER_FIELD_HINTS, used=used)
    return FieldMapping(
        context_field=context_field,
        question_field=question_field,
        answer_field=answer_field,
        title_field=title_field,
        id_field=id_field,
    )


def stable_id(*parts: str) -> str:
    payload = "\n".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()[:16]


def benchmark_contamination_risk(dataset_name: str, tags: list[str] | None = None) -> str:
    haystack = " ".join([dataset_name, *(tags or [])]).lower()
    if any(name in haystack for name in BENCHMARK_DATASETS):
        return "high"
    if any(word in haystack for word in ("benchmark", "evaluation", "eval", "leaderboard", "test")):
        return "medium"
    return "low"


def task_family_from_metadata(dataset_name: str, tags: list[str] | None, row: dict[str, Any]) -> str:
    haystack = " ".join([dataset_name, *(tags or []), *row.keys()]).lower()
    if "summar" in haystack:
        return "summarization"
    if "code" in haystack or "program" in haystack:
        return "code"
    if "math" in haystack:
        return "math"
    if "instruction" in haystack or "text2text" in haystack:
        return "instruction"
    return "qa"


def classify_candidate(mapping: FieldMapping, answer_text: str) -> tuple[str, str]:
    if mapping.context_field and mapping.question_field and answer_text:
        return "context_question_answer", "has context, question/instruction, and reference answer"
    if mapping.context_field and mapping.question_field:
        return "context_question_no_answer", "has context and question/instruction; answer is absent"
    if mapping.context_field:
        return "long_context_only", "has long context but no question/instruction"
    return "unusable", "could not infer a context field"


def build_question(row: dict[str, Any], mapping: FieldMapping) -> str:
    if not mapping.question_field:
        return ""
    return flatten_text(row.get(mapping.question_field))


def build_context(row: dict[str, Any], mapping: FieldMapping) -> str:
    if not mapping.context_field:
        return ""
    return flatten_text(row.get(mapping.context_field))


def build_answer(row: dict[str, Any], mapping: FieldMapping) -> tuple[str, list[str]]:
    if not mapping.answer_field:
        return "", []
    value = row.get(mapping.answer_field)
    if isinstance(value, list):
        refs = [flatten_text(item) for item in value if flatten_text(item)]
        return "\n".join(refs), refs
    text = flatten_text(value)
    return text, [text] if text else []


def make_lcqa_sample(
    row: dict[str, Any],
    dataset_name: str,
    subset: str | None,
    split: str | None,
    row_index: int,
    tokenizer_name: str,
    tokenizer,
    tags: list[str] | None = None,
    license_name: str | None = None,
) -> tuple[LCQASample | None, DiscoveryDecision]:
    mapping = infer_field_mapping(row)
    context = build_context(row, mapping)
    question = build_question(row, mapping)
    answer, reference_answers = build_answer(row, mapping)
    candidate_type, reason = classify_candidate(mapping, answer)

    if not context:
        return None, DiscoveryDecision("rejected", reason, candidate_type, mapping)
    if not question:
        return None, DiscoveryDecision("review_later", reason, candidate_type, mapping)

    context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
    input_tokens = context_tokens + question_tokens
    answer_tokens = len(tokenizer.encode(answer, add_special_tokens=False)) if answer else 0
    length_bucket = get_length_bucket(input_tokens)

    raw_id = flatten_text(row.get(mapping.id_field)) if mapping.id_field else ""
    source_id = raw_id or stable_id(dataset_name, subset or "", split or "", str(row_index), question[:500])
    local_dataset = dataset_name.replace("/", "__")
    sample_id = f"hf_discovery.{local_dataset}.{subset or 'default'}.{source_id}"
    family = task_family_from_metadata(dataset_name, tags, row)
    risk = benchmark_contamination_risk(dataset_name, tags)

    sample = LCQASample(
        id=sample_id,
        source={
            "dataset": dataset_name,
            "subset": subset,
            "split": split,
            "original_id": raw_id or None,
            "license": license_name,
            "url": f"https://huggingface.co/datasets/{dataset_name}",
        },
        task={
            "family": family,
            "type": f"long_context_{family}",
            "subtype": candidate_type,
        },
        document={
            "title": flatten_text(row.get(mapping.title_field)) if mapping.title_field else None,
            "context": context,
            "context_type": "document",
            "metadata": {
                "discovery_field_mapping": mapping.__dict__,
                "hf_tags": tags or [],
                "candidate_type": candidate_type,
            },
        },
        input={"question": question, "choices": None},
        output={
            "answer": answer,
            "answer_type": "free_form",
            "label": None,
            "reference_answers": reference_answers,
        },
        length={
            "tokenizer": tokenizer_name,
            "context_tokens": context_tokens,
            "question_tokens": question_tokens,
            "input_tokens": input_tokens,
            "prompt_tokens": input_tokens,
            "answer_tokens": answer_tokens,
            "length_bucket": length_bucket,
        },
        quality={
            "status": "raw",
            "annotator": "hf_discovery",
            "verified": False,
            "needs_long_context": input_tokens >= 32_768,
            "contamination_risk": risk,
            "metadata": {
                "candidate_type": candidate_type,
                "discovery_reason": reason,
                "has_reference_answer": bool(answer),
            },
        },
        raw=row,
    )
    return sample, DiscoveryDecision("candidate", reason, candidate_type, mapping, input_tokens, length_bucket)


def decision_to_json(decision: DiscoveryDecision, dataset_name: str, subset: str | None, row_index: int) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "subset": subset,
        "row_index": row_index,
        "status": decision.status,
        "reason": decision.reason,
        "candidate_type": decision.candidate_type,
        "field_mapping": decision.field_mapping.__dict__,
        "input_tokens": decision.input_tokens,
        "length_bucket": decision.length_bucket,
    }


def sample_preview(row: dict[str, Any], max_chars: int = 2_000) -> str:
    text = json.dumps(row, ensure_ascii=False, default=str)
    return text[:max_chars]

from __future__ import annotations

import re

from longcontext.schema import LCQASample


BENCHMARK_ALIASES = {
    "longbench": {
        "longbench",
        "zaiorglongbench",
        "zailongbench",
        "zai_longbench",
    },
    "longbenchv2": {
        "longbenchv2",
        "longbench_v2",
        "zaiorglongbenchv2",
        "zailongbenchv2",
        "zai_longbench_v2",
    },
    "infinitebench": {
        "infinitebench",
        "xinrongzhang2022infinitebench",
    },
    "loogle": {
        "loogle",
    },
    "ruler": {
        "ruler",
    },
    "needle": {
        "needle",
        "needlehaystack",
        "needleinhaystack",
    },
}


def normalize_source_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def expand_heldout_benchmarks(values: list[str] | None) -> set[str]:
    aliases: set[str] = set()
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            normalized = normalize_source_name(item)
            aliases.add(normalized)
            aliases.update(normalize_source_name(alias) for alias in BENCHMARK_ALIASES.get(normalized, set()))
    return aliases


def sample_source_keys(sample: LCQASample) -> set[str]:
    return {
        key
        for key in (
            normalize_source_name(sample.source.dataset),
            normalize_source_name(sample.source.url),
            normalize_source_name(sample.id),
        )
        if key
    }


def heldout_source_match(sample: LCQASample, heldout_aliases: set[str]) -> str | None:
    for source_key in sample_source_keys(sample):
        for alias in heldout_aliases:
            if source_key == alias or alias in source_key:
                return alias
    return None


def mark_training_eligibility(sample: LCQASample, heldout_aliases: set[str]) -> LCQASample:
    matched_alias = heldout_source_match(sample, heldout_aliases)
    metadata = sample.quality.metadata
    metadata["heldout_benchmark_aliases"] = sorted(heldout_aliases)

    if matched_alias:
        sample.quality.training_eligible = False
        sample.quality.training_exclusion_reason = (
            f"source_matches_heldout_benchmark:{matched_alias}"
        )
        sample.quality.contamination_risk = "high"
        metadata["heldout_benchmark_match"] = matched_alias
    else:
        sample.quality.training_eligible = True
        sample.quality.training_exclusion_reason = None
        metadata["heldout_benchmark_match"] = None
    return sample

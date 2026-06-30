from __future__ import annotations

from collections import OrderedDict


BUCKET_ORDER = (
    "<32K",
    "32K-64K",
    "64K-128K",
    "128K-256K",
    "256K-512K",
    "512K-900K",
    ">900K",
)

MAIN_LONG_CONTEXT_BUCKETS = (
    "32K-64K",
    "64K-128K",
    "128K-256K",
    "256K-512K",
    "512K-900K",
)


def get_length_bucket(input_tokens: int) -> str:
    if input_tokens < 32_768:
        return "<32K"
    if input_tokens < 65_536:
        return "32K-64K"
    if input_tokens < 131_072:
        return "64K-128K"
    if input_tokens < 262_144:
        return "128K-256K"
    if input_tokens < 524_288:
        return "256K-512K"
    if input_tokens <= 900_000:
        return "512K-900K"
    return ">900K"


def empty_bucket_counts() -> OrderedDict[str, int]:
    return OrderedDict((bucket, 0) for bucket in BUCKET_ORDER)


def is_main_long_context_bucket(bucket: str) -> bool:
    return bucket in MAIN_LONG_CONTEXT_BUCKETS

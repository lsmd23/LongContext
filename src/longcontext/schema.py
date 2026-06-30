from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    dataset: str
    subset: str | None = None
    split: str | None = None
    original_id: str | None = None
    doc_id: str | None = None
    license: str | None = None
    url: str | None = None


class Task(BaseModel):
    family: str
    type: str
    subtype: str | None = None
    domain: str | None = None
    sub_domain: str | None = None
    difficulty: str | None = None
    dependency: str | None = None


class Document(BaseModel):
    context: str
    title: str | None = None
    context_type: str | None = "document"
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Choice(BaseModel):
    label: str
    text: str


class Input(BaseModel):
    question: str
    instruction: str | None = None
    choices: list[Choice] | None = None


class Output(BaseModel):
    answer: str
    answer_type: Literal["free_form", "multiple_choice", "extractive", "generated"] = "free_form"
    label: str | None = None
    reference_answers: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    text: str
    start_char: int | None = None
    end_char: int | None = None
    source: str | None = None


class Length(BaseModel):
    tokenizer: str | None = None
    context_tokens: int | None = None
    question_tokens: int | None = None
    input_tokens: int | None = None
    prompt_tokens: int | None = None
    answer_tokens: int | None = None
    length_bucket: str | None = None


class Quality(BaseModel):
    status: Literal["raw", "filtered", "labeled", "verified", "rejected"] = "raw"
    annotator: str | None = None
    verified: bool = False
    notes: str | None = None
    needs_long_context: bool | None = None
    quality_score: float | None = None
    evidence_spread: str | None = None
    contamination_risk: Literal["low", "medium", "high"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Training(BaseModel):
    render_template: str | None = None
    messages: list[dict[str, str]] | None = None


class LCQASample(BaseModel):
    id: str
    schema_version: Literal["lcqa.v0.1"] = "lcqa.v0.1"
    source: Source
    task: Task
    document: Document
    input: Input
    output: Output
    evidence: list[Evidence] = Field(default_factory=list)
    length: Length = Field(default_factory=Length)
    quality: Quality = Field(default_factory=Quality)
    training: Training = Field(default_factory=Training)
    raw: dict[str, Any] = Field(default_factory=dict)

from __future__ import annotations

from longcontext.schema import LCQASample


def render_context_qa(sample: LCQASample) -> dict:
    user_parts = [
        "Please answer the question based on the given document.",
        "",
        "Document:",
        sample.document.context,
        "",
        "Question:",
        sample.input.question,
    ]

    if sample.input.choices:
        user_parts.extend(["", "Options:"])
        user_parts.extend(f"{choice.label}. {choice.text}" for choice in sample.input.choices)

    return {
        "id": sample.id,
        "messages": [
            {"role": "user", "content": "\n".join(user_parts)},
            {"role": "assistant", "content": sample.output.label or sample.output.answer},
        ],
        "metadata": {
            "source_id": sample.id,
            "dataset": sample.source.dataset,
            "template": "context_qa_v1",
            "prompt_tokens": sample.length.prompt_tokens,
            "answer_tokens": sample.length.answer_tokens,
        },
    }

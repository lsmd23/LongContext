# LCQA v0.1

LCQA 是本项目的 normalized 中间格式。它不是直接训练格式，而是用于统一管理不同来源的长上下文 QA 数据。

最小字段：

```json
{
  "id": "dataset.subset.sample_001",
  "schema_version": "lcqa.v0.1",
  "source": {
    "dataset": "dataset_name",
    "subset": "subset_name",
    "split": "train",
    "original_id": "sample_001",
    "doc_id": "doc_001"
  },
  "task": {
    "family": "qa",
    "type": "long_context_qa",
    "subtype": null
  },
  "document": {
    "title": "Document title",
    "context": "Long context..."
  },
  "input": {
    "question": "Question?",
    "choices": null
  },
  "output": {
    "answer": "Answer.",
    "answer_type": "free_form",
    "label": null
  },
  "evidence": [],
  "length": {
    "tokenizer": null,
    "context_tokens": null,
    "question_tokens": null,
    "prompt_tokens": null,
    "answer_tokens": null
  },
  "quality": {
    "status": "raw",
    "verified": false
  }
}
```

训练前再从 LCQA 渲染为 SFT `messages` 格式。

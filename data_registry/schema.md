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
    "input_tokens": null,
    "prompt_tokens": null,
    "answer_tokens": null,
    "length_bucket": null
  },
  "quality": {
    "status": "raw",
    "verified": false,
    "needs_long_context": null,
    "quality_score": null,
    "evidence_spread": null,
    "contamination_risk": null,
    "training_eligible": null,
    "training_exclusion_reason": null,
    "metadata": {}
  }
}
```

训练前再从 LCQA 渲染为 SFT `messages` 格式。

## 训练可用性标记

筛选阶段需要根据当前 held-out evaluation benchmark 标注样本是否可用于训练或 teacher 标注：

- `quality.training_eligible = true`：当前未命中 held-out benchmark 来源，可以进入后续训练候选池。
- `quality.training_eligible = false`：样本来源与实验 benchmark 同源，不能用于训练或 teacher 标注。
- `quality.training_exclusion_reason`：不可用原因，例如 `source_matches_heldout_benchmark:longbenchv2`。

默认 held-out benchmark 是 `longbench_v2`。如果实验 benchmark 改为其他集合，需要在筛选命令中传入对应参数，例如：

```bash
python scripts/filter_all_lcqa.py --heldout-benchmark infinitebench
```

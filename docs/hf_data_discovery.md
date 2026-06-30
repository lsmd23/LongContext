# Hugging Face 数据发现

本项目保留已有 LCQA 结构，不把 answer 作为候选样本的必要条件。新发现流程只做轻量搜索和 streaming 抽样，判断样本是否能形成：

```text
long context + question/query/instruction
```

## 搜索并抽样

```bash
python scripts/discover_hf_candidates.py \
  --search-terms "long document,document QA,summarization,instruction" \
  --max-datasets-per-term 5 \
  --sample-rows 20
```

也可以显式指定数据集：

```bash
python scripts/discover_hf_candidates.py \
  --datasets "dataset_a,dataset_b" \
  --sample-rows 50
```

默认输出到：

```text
data/discovery/hf_candidates/candidates_32k_900k.lcqa.jsonl
data/discovery/hf_candidates/samples_with_tokens.lcqa.jsonl
data/discovery/hf_candidates/discovery_decisions.jsonl
data/discovery/hf_candidates/candidate_sources.jsonl
data/discovery/hf_candidates/summary.json
data/normalized/<dataset>/<subset>/<split>.lcqa.jsonl
```

其中 `data/normalized/...` 与现有 LooGLE/ZAI benchmark 的 normalized 数据组织方式保持一致，可直接进入后续筛选流水。

## 字段识别

脚本会用启发式识别：

```text
context/document/passage/article/text/content/input/prompt
question/query/instruction/task/problem
answer/response/output/target/label/summary
```

有 answer 时保留为 `reference_answers`；没有 answer 但有 context 和 question/instruction 时仍可进入候选池，后续由 Teacher Model 重新标注 response。

## 风险标记

`LongBench`、`LongBench-v2`、`InfiniteBench`、`LooGLE` 等 benchmark 类数据会写入：

```json
{
  "quality": {
    "contamination_risk": "high"
  }
}
```

这类数据默认可用于评测、格式参考和分析，不应直接混入训练集。

## 接入后续筛选

Discovery 结束后，可以直接复用现有批量筛选脚本：

```bash
python scripts/filter_all_lcqa.py \
  --normalized-root data/normalized \
  --filtered-root data/filtered \
  --progress
```

这会为每个 discovery source 生成：

```text
filtered_32k_900k.jsonl
samples_with_tokens.jsonl
length_bucket_stats.csv
bucket_review_samples.md
filter_stats.json
```

如果只想生成 discovery 分析文件，不写入 normalized 流水目录，可以加：

```bash
python scripts/discover_hf_candidates.py --no-normalized-output
```

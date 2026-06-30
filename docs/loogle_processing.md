# LooGLE 处理

当前分三步：

```text
raw LooGLE JSONL
  -> LCQA normalized JSONL
  -> tokenized and bucketed LCQA JSONL
  -> 32K-900K filtered LCQA JSONL
```

这里的长度统计口径按长上下文训练范式理解为完整模型输入：

```text
input_tokens = context_tokens + question_tokens
question_text = instruction + question + choices
```

LCQA 中仍然把 `document.context` 和 `input.question` 分开保存，方便后续转换、质检和渲染；筛选时默认使用 `input_tokens`，并写入 `length.length_bucket`。

## 1. 统一格式

本机 smoke：

```bash
python scripts/normalize_loogle.py \
  --subset longdep_qa \
  --limit 20 \
  --output-root data/smoke/normalized/loogle
```

全量处理：

```bash
python scripts/normalize_loogle.py --subset all
```

默认输入：

```text
data/raw/loogle/<subset>/test.jsonl
```

默认输出：

```text
data/normalized/loogle/<subset>/test.lcqa.jsonl
```

## 2. Token 统计、分桶和 32K-900K 筛选

本机 smoke：

```bash
python scripts/filter_lcqa_by_tokens.py \
  data/smoke/normalized/loogle/longdep_qa/test.lcqa.jsonl \
  data/smoke/filtered/loogle/longdep_qa/filtered_32k_900k.jsonl \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --min-tokens 32768 \
  --max-tokens 900000 \
  --limit 5 \
  --stats-output data/smoke/filtered/loogle/longdep_qa/filter_stats.json
```

该命令会在输出目录下同时生成：

```text
samples_with_tokens.jsonl
filtered_32k_900k.jsonl
length_bucket_stats.csv
bucket_review_samples.md
```

全量处理：

```bash
python scripts/filter_lcqa_by_tokens.py \
  data/normalized/loogle/longdep_qa/test.lcqa.jsonl \
  data/filtered/loogle/longdep_qa/filtered_32k_900k.jsonl \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --min-tokens 32768 \
  --max-tokens 900000 \
  --stats-output data/filtered/loogle/longdep_qa/filter_stats.json
```

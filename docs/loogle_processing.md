# LooGLE 处理

当前分两步：

```text
raw LooGLE JSONL
  -> LCQA normalized JSONL
  -> token filtered LCQA JSONL
```

这里的 `Question token` 按长上下文训练范式理解为完整用户输入：

```text
full_question = context + question + choices
```

LCQA 中仍然把 `document.context` 和 `input.question` 分开保存，方便后续转换、质检和渲染；筛选时默认使用 `full_question`。

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

## 2. 按完整 Question token 筛选

本机 smoke：

```bash
python scripts/filter_lcqa_by_tokens.py \
  data/smoke/normalized/loogle/longdep_qa/test.lcqa.jsonl \
  data/smoke/filtered/loogle/longdep_qa/test.full_question32k_900k.lcqa.jsonl \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --min-tokens 32000 \
  --max-tokens 900000 \
  --limit 5 \
  --stats-output data/smoke/filtered/loogle/longdep_qa/test.full_question32k_900k.stats.json
```

全量处理：

```bash
python scripts/filter_lcqa_by_tokens.py \
  data/normalized/loogle/longdep_qa/test.lcqa.jsonl \
  data/filtered/loogle/longdep_qa/test.full_question32k_900k.lcqa.jsonl \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --min-tokens 32000 \
  --max-tokens 900000 \
  --stats-output data/filtered/loogle/longdep_qa/test.full_question32k_900k.stats.json
```

如果只想统计原始短问题文本，可显式指定：

```bash
python scripts/filter_lcqa_by_tokens.py input.lcqa.jsonl output.lcqa.jsonl --token-field question_text
```

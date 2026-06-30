# Benchmark 数据处理

支持以下 Hugging Face 数据源：

```text
zai-org/LongBench
zai-org/LongBench-v2
xinrongzhang2022/InfiniteBench
```

## 下载

```bash
python scripts/download_zai_benchmarks.py --dataset all
```

默认输出：

```text
data/raw/zai_longbench/<subset>/test.jsonl
data/raw/zai_longbench_v2/main/test.jsonl
data/raw/infinitebench/<task>/test.jsonl
```

## 归一化

```bash
python scripts/normalize_benchmarks.py --dataset all
```

默认输出：

```text
data/normalized/<dataset>/<subset>/test.lcqa.jsonl
```

## 分桶筛选

```bash
python scripts/filter_all_lcqa.py --normalized-root data/normalized --filtered-root data/filtered --progress
```

每个子集会输出：

```text
filtered_32k_900k.jsonl
samples_with_tokens.jsonl
length_bucket_stats.csv
bucket_review_samples.md
filter_stats.json
```

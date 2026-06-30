# 数据流水

```text
HF/source data
  -> raw
  -> normalize to LCQA
  -> tokenize and filter
  -> teacher labeling
  -> quality check
  -> render SFT
  -> train/eval
```

## Hugging Face Discovery 分支

广泛遍历 Hugging Face 时，discovery 阶段会直接用 streaming 小样本构造 LCQA 候选，并额外写入 normalized-compatible 目录：

```text
HF search / explicit dataset list
  -> streaming sample
  -> field discovery
  -> token bucket
  -> data/normalized/<dataset>/<subset>/<split>.lcqa.jsonl
  -> data/filtered/<dataset>/<subset>/
  -> teacher labeling candidate pool
```

命令：

```bash
python scripts/discover_hf_candidates.py
python scripts/filter_all_lcqa.py --normalized-root data/normalized --filtered-root data/filtered
```

`data/discovery/hf_candidates/candidate_sources.jsonl` 记录产生候选的数据源、config、split、license、tags 和污染风险，后续可以用来决定是否扩大下载/抽样规模。

## 原则

- raw 数据只保存原样，不做覆盖式修改。
- normalized 数据一行一个 QA 样本。
- SFT 数据只保存模型训练需要的 `messages` 和必要 metadata。
- 大文件不进入 Git。

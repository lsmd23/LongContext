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

## 原则

- raw 数据只保存原样，不做覆盖式修改。
- normalized 数据一行一个 QA 样本。
- SFT 数据只保存模型训练需要的 `messages` 和必要 metadata。
- 大文件不进入 Git。

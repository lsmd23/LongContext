# LongContext

长上下文 QA 数据构造、Teacher 标注、SFT 数据渲染与评测流水的最小代码框架。

## 目标

- 统一为 LCQA normalized 格式。
- 按 tokenizer 统计长度并筛选长上下文样本。
- 使用 Teacher Model 生成或改写 response。
- 渲染为 SFT `messages` 格式。
- 对接长上下文评测流水。

## 数据生命周期

```text
raw
  -> normalized
  -> filtered normalized
  -> labeled normalized
  -> sft
  -> train/eval
```

## 推荐目录

本仓库只保存代码、配置、文档和小样例。大数据、模型权重、评测输出不要进入 Git。

```text
configs/          # 数据、标注、训练、评测配置
data_registry/    # 数据源登记模板与格式说明
docs/             # 环境、流水、调研笔记
experiments/      # 实验记录，不放大文件
scripts/          # 命令行脚本入口
src/longcontext/  # Python 包
```

## 快速开始

```bash
conda env create -f environment.yml
conda activate longcontext
```

复制 `.env.example` 为 `.env`，按机器修改数据目录。

```bash
python scripts/validate_lcqa.py path/to/normalized.lcqa.jsonl
python scripts/render_sft.py path/to/normalized.lcqa.jsonl outputs/sft.jsonl
```

# LooGLE 下载

LooGLE 的 Hugging Face 地址：

```text
https://huggingface.co/datasets/bigai-nlco/LooGLE
```

当前下载脚本只保存 raw 数据，不做字段转换。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 下载全部 subset

```bash
python scripts/download_loogle.py --subset all
```

默认输出到：

```text
data/raw/loogle/<subset>/test.jsonl
data/raw/loogle/<subset>/test.metadata.json
```

## 只下载一个 subset

```bash
python scripts/download_loogle.py --subset longdep_qa
```

## 指定数据根目录

```bash
python scripts/download_loogle.py --subset longdep_qa --output-root D:/data/longcontext
```

也可以在 `.env` 中设置：

```text
LONGCONTEXT_DATA_ROOT=D:/data/longcontext
```

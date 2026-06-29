# 环境搭建

## 本机

本机主要负责代码编辑、小样本调试、配置和实验记录。

推荐使用 conda 独立环境，避免污染 `base`。

```bash
conda env create -f environment.yml
conda activate longcontext
```

如果环境已经存在：

```bash
conda activate longcontext
pip install -r requirements.txt
pip install -e .
```

## 开发机

开发机主要负责数据下载、token 统计、Teacher 标注、训练和评测。

建议目录：

```text
/home/$USER/projects/LongContext
/data/longcontext/raw
/data/longcontext/normalized
/data/longcontext/labeled
/data/longcontext/sft
/data/longcontext/outputs
/data/longcontext/checkpoints
```

代码同步以 Git 为主，SFTP 只用于查看日志或临时取小文件。

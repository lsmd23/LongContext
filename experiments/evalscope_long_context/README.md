# EvalScope Long Context 实验框架

基于 [EvalScope](https://github.com/modelscope/evalscope) 的 Long Context 评测实验模块。当前阶段只搭建框架与 smoke test 通路，不接入训练、不改动数据构造主流程。

## 目标

1. 通过 OpenAI-compatible API 调用远程模型。
2. 通过 YAML 配置切换模型名、API 地址、API key 环境变量、benchmark、subset、limit。
3. 运行 `longbench_v2` 的 `limit=5` 或 `limit=10` smoke test。
4. 保存 EvalScope 原始输出、统一指标、失败样本、运行环境信息。
5. 支持后续 baseline 与训练后模型的 before/after 对比（`summarize_runs.py`）。

## LongBench-v2 数据污染约束

**LongBench-v2 是 held-out evaluation benchmark，不得用于训练。**

```text
LongBench-v2 is held-out evaluation data.
Do not use its context, question, choices, answer, rewritten samples, translated samples,
or teacher-labeled variants for training.
```

训练数据筛选阶段应排除以下评测集相关样本：

```text
LongBench
LongBench-v2
InfiniteBench
LooGLE
RULER
Needle / needle-in-a-haystack variants
```

本模块只读取 benchmark 做评测，不向 `data/normalized`、`data/filtered` 或 SFT 导出路径写入训练数据。

## 目录结构

```text
experiments/evalscope_long_context/
  README.md
  configs/
    smoke_longbench_v2.yaml      # smoke test（limit=5）
    baseline_longbench_v2.yaml   # 完整/大规模 baseline
    compare_template.yaml        # before/after 对比模板
  scripts/
    run_evalscope_smoke.sh       # shell 入口
    run_evalscope_from_config.py # 主调度脚本
    collect_run_metadata.py
    parse_evalscope_outputs.py
    summarize_runs.py
  templates/
  runs/                          # 本地实验输出（已 gitignore）
```

## 环境安装

EvalScope 暂不写入项目根 `requirements.txt`，请在独立 conda 环境中安装：

```bash
conda activate longcontext
pip install evalscope pyyaml
```

## 环境变量

API key 只从环境变量读取，**不得**写入配置文件。

```bash
export MODEL_API_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_API_KEY="your_key"
```

Windows PowerShell：

```powershell
$env:MODEL_API_URL = "https://your-openai-compatible-endpoint/v1"
$env:MODEL_API_KEY = "your_key"
```

编辑 `configs/smoke_longbench_v2.yaml` 中的 `model.name` 为实际模型名。

## Smoke Test 最小运行命令

```bash
bash experiments/evalscope_long_context/scripts/run_evalscope_smoke.sh
```

或直接调用 Python（Windows / 无 bash 时推荐）：

```bash
python experiments/evalscope_long_context/scripts/run_evalscope_from_config.py \
  --config experiments/evalscope_long_context/configs/smoke_longbench_v2.yaml
```

## 输出目录说明

每次运行创建独立目录，例如：

```text
experiments/evalscope_long_context/runs/20260630_153000_longbench_v2_smoke/
  config.resolved.yaml
  run_manifest.yaml
  command.txt
  env.txt
  evalscope_raw/          # EvalScope 原始输出
  parsed/
    metrics.json
    predictions.jsonl
    failed_cases.jsonl
    latency.jsonl
  reports/
    summary.md
    blocking_issue.md     # 仅在失败时生成
  logs/
    stdout.log
    stderr.log
```

## 跨 run 汇总

```bash
python experiments/evalscope_long_context/scripts/summarize_runs.py \
  --runs-root experiments/evalscope_long_context/runs
```

生成 `runs/summary.csv` 与 `runs/summary.md`。

## 常见 Blocking Issue

| 现象 | 原因 | 下一步 |
| --- | --- | --- |
| `MODEL_API_URL is not set` | 未设置 API 地址 | 设置环境变量后重跑 |
| `MODEL_API_KEY is not set` | 未设置 API key | 设置环境变量后重跑 |
| `evalscope command not found` | 未安装 EvalScope | `pip install evalscope` |
| EvalScope 非零退出 | API 不兼容 / 模型名错误 / 超时 | 查看 `logs/stderr.log` |
| `parse_status=failed` | 输出目录结构与 EvalScope 版本不匹配 | 检查 `evalscope_raw/reports/` |

失败时会在 `reports/blocking_issue.md` 留下结构化记录，不会静默失败。

## Baseline / Before-After 对比

1. **Baseline**：使用 `configs/baseline_longbench_v2.yaml`（`limit: null` 表示全量）。
2. **对比**：复制 `configs/compare_template.yaml`，填入 baseline 与 trained run id。
3. **汇总**：运行 `summarize_runs.py` 生成 `summary.csv`，按 model / accuracy / failed / latency 对比。

## 开发机部署（新加坡 CPU）

```text
host: 43.163.98.45
user: ubuntu
workspace: /home/ubuntu/lisunmuduo/LongContext/
conda: /home/ubuntu/lisunmuduo/miniconda3/
```

原则：

1. 不修改公共 conda base 环境。
2. 在 `longcontext` 或专用实验环境中安装 EvalScope。
3. 长时间任务使用 tmux。
4. 输出写入 `experiments/evalscope_long_context/runs/`，不提交大文件。

## 相关文档

- [框架执行方案](../evalscope_long_context_framework_plan.md)
- [EvalScope LongBench-v2 文档](https://evalscope.readthedocs.io/en/latest/benchmarks/longbench_v2.html)

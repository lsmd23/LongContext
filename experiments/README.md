# Experiments

## 当前实验模块

### EvalScope Long Context 评测框架

- **模块目录**: [evalscope_long_context/](./evalscope_long_context/)
- **执行方案**: [evalscope_long_context_framework_plan.md](./evalscope_long_context_framework_plan.md)
- **状态**: 框架已落地（Phase A）；smoke test 待开发机 + 真实 API 验证（Phase C）

快速启动 smoke test：

```bash
conda activate longcontext
pip install evalscope pyyaml

export MODEL_API_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_API_KEY="your_key"

bash experiments/evalscope_long_context/scripts/run_evalscope_smoke.sh
```

详见 [evalscope_long_context/README.md](./evalscope_long_context/README.md)。

## 通用实验目录规范

每个实验建议单独建目录：

```text
exp_000_smoke_test/
  config.yaml
  notes.md
  metrics.json
```

实验记录里写清楚代码 commit、数据版本、tokenizer、teacher model、训练参数和评测命令。

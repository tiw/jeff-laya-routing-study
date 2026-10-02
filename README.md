# jeff-laya-routing-study

本地决策模型做 Sub-agent 路由的对比研究：Jeff（Qwen3.5 解码器 SFT）vs Laya（ModernBERT 编码器）——能力水位、错误机制、stacking 元模型与置信度级联。

**论文**：[paper/jeff-laya-routing-study.md](paper/jeff-laya-routing-study.md)（中文技术报告，含全部实验数据与结论）

**一句话结论**：Jeff-2B（92ms）+ Laya 概率拼接训练的 softmax 元模型 + 置信度级联本地 27B，在 248 条 OOD 测试上达到 99.2% 准确率、23.4% 升级率，全面优于手写阈值规则。

## 仓库结构

```
paper/    技术论文（Markdown，pandoc 可转 PDF）
assets/   论文图表（exp2_gradient.png，E2 话题梯度图）
code/
  cascade_router.py   级联路由器服务（stdlib HTTP，:8767，POST /route）
  scripts/            全部评测与训练脚本
  data/               540 条带金标准数据集 + 全部结果 JSON + 元模型权重（.npy）
```

## 模型权重（不包含在本仓库中）

本仓库**不包含任何模型权重**（合计约 5GB，且均为上游项目的发布产物）。需要自行下载：

| 模型 | 大小 | 获取方式 |
|---|---|---|
| Jeff-Qwen3.5-2B (v1.2) | 4.4GB | clone https://github.com/firelex/jeff 后按其 README 用 `hf download` 拉取，或见论文第 1 节 |
| Laya typed-decisions | 842MB | `huggingface-cli download convaiinnovations/laya --include "typed-decisions/*"`（必须传本地路径使用，默认构造会卡 HF 下载） |
| Qwen3.8-27B (mxfp8) | 31GB | Ollama：`ollama pull qwen3.8:27b-mxfp8`（仅级联升级路径需要；评测可跳过） |

## 复现

环境：macOS + Apple Silicon（MLX 后端），Python 3.11（laya-mlx 0.2.0），uv ≥ 0.12.19。

```bash
# 1. 启动 Jeff 服务（在 firelex/jeff 项目目录内）
JEFF_BACKEND=mlx JEFF_CHECKPOINT=<Jeff-Qwen3.5-2B 本地路径> PORT=8766 \
  uv run --no-default-groups --extra mac jeff-serve

# 2. 跑三方对比评测（数据在 code/data/，脚本里的路径按需调整）
python code/scripts/run_jeff_eval.py
python code/scripts/run_laya_eval.py
python code/scripts/run_llm_eval.py

# 3. 机制验证实验与 E2 正式版
python code/scripts/run_verify_experiments.py
python code/scripts/run_e2_formal.py

# 4. Stacking 元模型训练与 OOD 测试
python code/scripts/run_stacking.py
python code/scripts/run_stacking_v2.py
python code/scripts/run_new_sets.py

# 5. 启动级联路由器（依赖 meta_W.npy / meta_b.npy / meta_tau.json，已含在 code/data/）
HF_HUB_OFFLINE=1 python3.11 code/cascade_router.py
curl -X POST localhost:8767/route -H 'Content-Type: application/json' \
  -d '{"state": {"request": "帮我 review 这个 PR"}}'
```

## 已知问题

1. 级联阈值 τ=0.7629 偏激进：清晰的 coder 请求若 Laya 概率分散，meta 置信会被拖过阈值造成误升级。调低 τ 或对 Laya 特征降权是后续方向。
2. 27B 常驻统一内存时，小模型延迟从 92ms 升至 ~1.8s。生产部署需权衡大模型常驻 vs 按需冷启动。
3. 所有延迟为单机同进程 HTTP 测量，未测并发。

## License

代码与数据：MIT（见 LICENSE）。上游模型权重遵循其各自许可（Jeff：Apache 2.0）。

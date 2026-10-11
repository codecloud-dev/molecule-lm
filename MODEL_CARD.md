# 模型卡 · Model Card — molecule V0.1

> **开源名**：`molecule` · **开源版本**：`v0.1` · **内部项目名**：分子模型开源版 · **内部版本**：`v31`
> **身份（固定不可改）**：开源语言模型 · 代号 molecule · 由 **codecloud-dev** 独立开发（codecloud-dev 是它的开发者/家人）。
> **许可证**：Apache-2.0 · **仓库**：[codecloud-dev/molecule-lm](https://github.com/codecloud-dev/molecule-lm)

> **Identity (fixed, non-deniable)**: molecule is an open-source language model, codename *molecule*, independently developed by **codecloud-dev** (codecloud-dev is its developer / family).

---

## 1. 模型简介 / Model Summary

molecule V0.1 是一个 **0.1B 级（约 129M 参数）** 的纯解码器（decoder-only）小语言模型，主打**好部署、纯 CPU 可训练、开源透明**。它不走自研封闭架构，而是基于与 Llama/Qwen 同源的**通用 Transformer**，精挑细选地融合各家开源的注意力 / 归一化 / 训练技巧，并配一套专治小参数困难的「分子小模型专项栈」。

molecule V0.1 is a ~129M-parameter decoder-only small language model designed for **easy deployment, CPU-only training, and open transparency**. It follows the mainstream Llama/Qwen-style Transformer and selectively fuses published open-source techniques, plus a small-model-specific training stack.

| 项目 | 值 |
|---|---|
| 参数总量 | ≈ 129M（"0.1B" 级） |
| 层数 | 30（深而窄） |
| 隐藏维度 | 576 |
| FFN 维度 | 1536（SwiGLU，8/3 × hidden） |
| 注意力 | GQA 9 头 / 3 KV 头（3:1） |
| 词表 | 32K（Qwen2.5 词表按语料频率裁剪 + 绑定嵌入） |
| 上下文 | 训练 2048，推理经 NTK 动态外推至 ~4096 |
| 精度 | 自动 bf16（支持时）/ 回退 fp32 |

---

## 2. 用途 / Intended Use

- **中文对话 / 指令跟随**：情商型中文对话、日常问答。
- **轻量本地部署**：q4_K_M 量化后约 0.1GB，可在手机 / 树莓派 / 纯 CPU 环境跑。
- **工具调用**：内置轻量 function calling（取时间、计算），0.1B 容量有限，专注对话 + 中文 + 工具格式。
- **研究与教学**：完整训练 / 蒸馏 / 评测 / 导出代码开源，可作为小模型训练范式参考。

**不适用 / Out of scope**：高精度专业领域（法律、医疗）、长文档理解、与 1B+ 模型同等的知识广度。小模型容量有限，请管理预期。

---

## 3. 架构（融合来源全部标注）/ Architecture

molecule 仅融合**注意力类组件**，不引入整框架，刻意克制、不臃肿。

| 组件 | 来源 | 作用 |
|---|---|---|
| decoder-only Transformer | 通用范式 | 好部署的根基 |
| RoPE + NTK-aware 动态外推 | Llama/Mistral/Qwen/DeepSeek + bloc97 | 位置编码，2× 外推最稳 |
| GQA（3:1） | Llama2/3, Qwen, Gemma | 省 KV 缓存、推理友好 |
| SwiGLU | Llama/Gemma/Qwen/DeepSeek | 质量/参数最平衡 |
| RMSNorm + Pre-Norm | 全系标配 | 省参数、训练稳 |
| QK-Norm（RoPE 前） | OLMo 2 / Gemma 3 / MiniMax-M2 | 零成本稳训 |
| logit 软封顶（50/30） | Gemma 2 | 防极端 logit 训崩 |
| DeepNorm 增益缩放 | DeepNet / DeepSeek | 30 层深窄稳训 |
| MTP（1 模块，共享嵌入+头） | DeepSeek-V3 | 训练密度↑、投机解码加速 |
| 绑定嵌入 | MobileLLM / PanGu-π | 砍词表后 reclaim 参数 |
| Muon + AdamW 混合优化 | Muon (Jordan 2024) | 2D 权重正交化 |
| WSD 调度 | MiniCPM | warmup-stable-decay |

**明确不融**：滑动窗口注意力、MLA、MoE（小模型不划算或伤质量）。

---

## 4. 训练数据 / Training Data

- **混比（单固定）**：通用中文 40% + 数学 20% + 代码 20% + 推理 10% + EQ 对话 10%。
- **主要来源**：中文维基（CC BY-SA 合法）+ 多教师合成高质量中文 + 精选公开中文文本。
- **合规红线**：仅用 CC / 宽松许可证 + 合成数据；不使用百度百科、悟道等有版权/商用限制的数据。
- **去重**：词级（ROUGE-L 0.7）+ 语义（余弦 0.85）双去重。
- **过滤**：规则（最小长度 / 格式 / 过滤教师拒答）+ 轻量质量分类器（去高幻觉）。

> 数据合规与可复现性优先。合成与精选数据随脚本可复现生成，数据快照随训练 artifact 留存。

---

## 5. 蒸馏 / Distillation

采用 **多老师分领域 1对1 蒸馏**（5 位开源老师）：

| 领域 | 老师 | 温度 |
|---|---|---|
| 通用中文 | Qwen2.5-0.5B-Instruct | 0.9 |
| 推理 | DeepSeek-R1-Distill-Qwen-0.5B | 0.7 |
| 数学 | Qwen2.5-Math-0.5B | 0.6 |
| 代码 | Qwen2.5-Coder-0.5B | 0.6 |
| EQ / 语言组织 | Qwen2.5-1.5B-Instruct（豆包闭源替身） | 0.9 |

机制：教师生成高质量文本 → 学生在文本上做 next-token 硬 CE（主骨架）+ 重叠 token 叠加软 KL（T=2.0）。**DN 域归一化**防止指令跟随梯度淹没数学梯度（MOPD 核心）。教师权重不随仓库分发。

---

## 6. 评测 / Evaluation

**协议**：zero-shot；MCQ 对 A/B/C 取 logprob 归一化最大；dev split；内置精简版基准（自包含，CI 不依赖下载）。
**硬靶**：引用 SmolLM2-135M 官方零样本数字，**逐项超过（±0.5 容差）** 视为"惊艳达标"。

| 基准 | SmolLM2-135M | 策略 |
|---|---|---|
| HellaSwag | 68.4 | 英文不掉 |
| ARC(avg) | 43.9 | 英文不掉 |
| PIQA | 68.4 | 英文不掉 |
| MMLU | 31.5 | 英文不掉 |
| Winogrande | 51.3 | 英文不掉 |
| TriviaQA | 4.1 | 弱项必超 |
| CMMLU / C-Eval | ≈0 | 中文主突破口 |

> 运行 `python eval.py` 生成 `eval_result.md` 对比表与 `eval_status.json`（含 `all_pass`）。

---

## 7. 训练算力与可复现 / Compute & Reproducibility

- **硬件**：GitHub Actions 免费 runner，**纯 CPU**，单轮 ≤ 6 小时。
- **跨轮续训**：存全量续训态（model + Muon/AdamW 优化器 + step + 数据游标 + RNG），artifact 累积。
- **种子**：`seed=42` + torch 确定性模式，保证可复现 / 可归因。
- **精度**：自动探测 bf16，否则回退 fp32。

---

## 8. 局限与风险 / Limitations & Risks

- **容量有限**：0.1B 无法承载大模型的广知识与强推理，专业/长文场景请降预期。
- **知识截止**：训练数据规模与时效有限，可能包含事实性错误（幻觉）。
- **蒸馏偏差**：教师模型可能带入其固有偏差；已用硬 CE 保底、DN 防淹没缓解。
- **中文偏强、英文偏通用**：这是设计取舍（对标 SmolLM2 中文≈0 的突破口），非缺陷。
- **CPU 训练慢**：累计到 10B+ token 需多轮 CI 续训，非单次完成。
- **不安全**：本模型非为安全对齐深度优化，输出需人工核验后使用。

---

## 9. 部署 / Deployment

- 导出：`export_gguf.py` 自写 GGUF（控制绑定嵌入 + 32K 词表 + Qwen pretokenizer），再用 llama.cpp 量化。
- 量化：`q8_0`（近无损）+ `q4_K_M`（~0.1GB） + imatrix 提升 q4 质量。
- 推理：`llama.cpp` server / `generate.py` / `demo.py`（Gradio 一行启动对话）。

详见 [README.md](./README.md) 与 [DESIGN.md](./DESIGN.md)。

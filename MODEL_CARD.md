# 模型卡 · Model Card — molecule

> **开源名**：`molecule` · **开发者**：codecloud-dev（独立开发、开源）· **许可证**：Apache-2.0
> **仓库**：[codecloud-dev/molecule-lm](https://github.com/codecloud-dev/molecule-lm)
> **当前版本**：见仓库 `VERSION` 文件与各 GitHub Release 标签（从 `v0.1` 起，封顶 `V1.0`）。

> **Identity**: molecule is an open-source language model, codename *molecule*, independently developed and open-sourced by **codecloud-dev**.

---

## 1. 模型简介 / Model Summary

molecule 是一个 **0.1B 级（约 129M 参数）** 的纯解码器（decoder-only）小语言模型，主打**极小体积、好部署、纯 CPU 可训练、开源透明**。它基于与 Llama/Qwen 同源的通用 Transformer，精挑细选地融合各家开源的注意力 / 归一化 / 训练技巧，并配一套专治小参数困难的「小模型专项栈」。

molecule is a ~129M-parameter decoder-only small language model designed for **tiny footprint, easy deployment, CPU-only training, and open transparency**. It follows the mainstream Llama/Qwen-style Transformer and selectively fuses published open-source techniques, plus a small-model-specific training stack.

**主目标（部署场景）**：在**任意能运行 llama.cpp 的弱设备**——手机、树莓派、老旧 / 低配 CPU——上本地运行一个能对话的中文小模型。q4 量化后约 0.1GB，不联网、不依赖外部服务。当前版本**不追求极致推理速度**；以速度为核心的 Flash 优化版本列为后续路线。

| 项目 | 值 |
|---|---|
| 参数总量 | ≈ 129M（"0.1B" 级） |
| 层数 | 30（深而窄） |
| 隐藏维度 | 576 |
| FFN 维度 | 1536（SwiGLU，8/3 × hidden） |
| 注意力 | GQA 9 头 / 3 KV 头（3:1） |
| 词表 | 32K（Qwen2.5 词表按语料频率裁剪 + 绑定嵌入） |
| 上下文 | 训练 256（CI 资源约束），架构支持最长 4096（NTK 动态外推） |
| 精度 | 自动 bf16（支持时）/ 回退 fp32 |
| 量化体积 | q4_K_M ≈ 0.1GB |

---

## 2. 用途 / Intended Use

- **中文对话 / 指令跟随**：轻量中文对话、日常问答。
- **轻量本地部署**：q4_K_M 量化后约 0.1GB，可在手机 / 树莓派 / 纯 CPU 环境本地运行，不联网。
- **工具调用**：内置轻量 function calling（取时间、计算），0.1B 容量有限，专注对话 + 中文 + 工具格式。
- **研究与教学**：完整训练 / 蒸馏 / 评测 / 导出代码开源，可作为小模型训练范式参考。

**不适用 / Out of scope**：高精度专业领域（法律、医疗）、长文档理解、与 1B+ 模型同等的知识广度。小模型容量有限，请管理预期。

---

## 3. 架构（融合来源全部标注）/ Architecture

molecule 仅融合**注意力类组件**，不引入整框架，刻意克制、不臃肿。

| 组件 | 来源 | 作用 |
|---|---|---|
| decoder-only Transformer | 通用范式 | 好部署的根基 |
| RoPE + NTK-aware 动态外推 | Llama/Mistral/Qwen/DeepSeek + bloc97 | 位置编码，短训长推最稳 |
| GQA（3:1） | Llama2/3, Qwen, Gemma | 省 KV 缓存、推理省内存 |
| SwiGLU | Llama/Gemma/Qwen/DeepSeek | 质量/参数最平衡 |
| RMSNorm + Pre-Norm | 全系标配 | 省参数、训练稳 |
| QK-Norm（RoPE 前） | OLMo 2 / Gemma 3 / MiniMax-M2 | 零成本稳训 |
| logit 软封顶（50/30） | Gemma 2 | 防极端 logit 训崩 |
| DeepNorm 增益缩放 | DeepNet / DeepSeek | 30 层深窄稳训 |
| MTP（1 模块，共享嵌入+头） | DeepSeek-V3 | 训练密度↑（**当前为辅助训练目标，未做投机解码**） |
| 绑定嵌入 | MobileLLM / PanGu-π | 砍词表后 reclaim 参数 |
| Muon + AdamW 混合优化 | Muon (Jordan 2024) | 2D 权重正交化 |
| WSD 调度 | MiniCPM | warmup-stable-decay |

**明确不融**：滑动窗口注意力、MLA、MoE（小模型不划算或伤质量）。

---

## 4. 训练数据 / Training Data

- **混比（单固定）**：通用中文 40% + 数学 20% + 代码 20% + 推理 10% + EQ 对话 10%。
- **实际使用的来源**（如实披露）：
  - 仓库内置 `seed_corpus.py` **自写种子语料**（中文为主的多领域真实文本），从零训练保底；
  - **教师模型生成的合成文本**（见第 5 节蒸馏），按领域并入训练；
  - 数据快照随训练 artifact 留存，脚本可复现。
- **合规**：仅使用宽松许可证 + 自写/合成数据；不使用有版权/商用限制的数据源。

> 注：早期文档曾误称"主要来源中文维基"，与实现不符——此处按实际实现如实更正：**未使用维基等外部语料**，数据来自上述自写 + 合成来源。

---

## 5. 蒸馏 / Distillation

采用 **多老师分领域 1对1 生成式数据蒸馏**（5 位**开源**老师，均为 Apache-2.0 / MIT，**无需 HuggingFace token**；CI 需能联网访问 HF 即可）：

| 领域 | 老师 | 许可证 | 温度 |
|---|---|---|---|
| 通用中文 | Qwen2.5-0.5B-Instruct | Apache-2.0 | 0.9 |
| 推理 | DeepSeek-R1-Distill-Qwen-0.5B | MIT | 0.7 |
| 数学 | Qwen2.5-Math-0.5B | Apache-2.0 | 0.6 |
| 代码 | Qwen2.5-Coder-0.5B | Apache-2.0 | 0.6 |
| EQ / 语言组织 | Qwen2.5-1.5B-Instruct | Apache-2.0 | 0.9 |

**机制（当前实际落地）**：教师生成高质量文本 → 学生在文本上做 next-token **硬 CE** 训练（主骨架）+ **MTP 辅助损失（0.1）**。词表天然对齐我们的 32K 裁剪空间。
**预留未启用**：软标签 KL 蒸馏（`loss_soft_kl` 当前为 0，需教师在线产出 soft logits，纯 CPU 不现实，列为后续 GPU 路线）；域归一化（DN）损失级加权为预留设计，当前数据按领域配比混合采样实现域平衡，损失级 DN 未接。

教师权重**不随仓库分发**。

---

## 6. 评测 / Evaluation

> ⚠️ **诚实声明**：仓库内置 `benchmarks.py` 为**少量自写 MCQ 冒烟题**，**非** HellaSwag / ARC / MMLU / CMMLU / C-Eval 等官方基准的正式 split。下表数字**仅用于训练趋势参考，不可与外模型官方分数直接比较**。正式评测请接入 lm-evaluation-harness + 各基准官方 split。

**协议**：zero-shot；MCQ 对 A/B/C 取 logprob 归一化最大；内置精简冒烟集（自包含，CI 不依赖下载）。

| 冒烟集 | 类别 | 说明 |
|---|---|---|
| hellaswag / arc / piqa / mmlu / winogrande / triviaqa | 英文常识/知识 | 趋势参考 |
| cmmlu / ceval | 中文常识（自写题） | 中文能力趋势参考 |

> 运行 `python eval.py` 生成 `eval_result.md` 与 `eval_status.json`。

---

## 7. 训练算力与可复现 / Compute & Reproducibility

- **硬件**：GitHub Actions 免费 runner，**纯 CPU**，单轮 ≤ 6 小时。
- **实际训练配置（CI）**：序列长度 256、global batch 约 16K token、单轮上限 400 步（资源约束下的取值，非架构上限；架构设计目标上下文为 2048、最长外推 4096）。
- **跨轮续训**：存全量续训态（model + Muon/AdamW 优化器 + step + 数据游标 + RNG），artifact 累积。
- **种子**：`seed=42` + torch 确定性模式，保证可复现 / 可归因。
- **精度**：自动探测 bf16，否则回退 fp32。

---

## 8. 局限与风险 / Limitations & Risks

- **容量有限**：0.1B 无法承载大模型的广知识与强推理，专业/长文场景请降预期。
- **知识截止与幻觉**：训练数据规模与时效有限，可能包含事实性错误，输出需人工核验。
- **评测口径**：内置冒烟集为自写题，**非官方基准分数**，请勿据此做跨模型对比或商业宣传。
- **蒸馏偏差**：教师模型可能带入其固有偏差；已用硬 CE 保底缓解。
- **算力规模**：单轮纯 CPU 仅能训练数千万 token 量级，与大规模训练差距显著，靠多轮 CI 续训逐步累积。
- **安全**：本模型非为安全对齐深度优化，输出需人工核验后使用。
- **教师许可**：蒸馏使用的教师模型均为 Apache-2.0 / MIT 开源许可；若用于衍生分发，请自行核验各教师最新许可条款。

---

## 9. 部署 / Deployment

- 导出：`export_gguf.py` 自写 GGUF（控制绑定嵌入 + 32K 词表 + Qwen pretokenizer），再用 llama.cpp 量化。
- 量化：`q8_0`（近无损）+ `q4_K_M`（~0.1GB） + imatrix 提升 q4 质量。
- 推理：`llama.cpp` server / `generate.py` / `demo.py`（Gradio 一行启动对话）/ 可接入 Ollama。

详见 [README.md](./README.md) 与 [DESIGN.md](./DESIGN.md)。

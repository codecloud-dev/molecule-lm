# molecule

> **开源名**：`molecule` · **许可证**：Apache-2.0 · **训练**：GitHub Actions 纯 CPU，单轮 ≤6h、可跨轮续训
> **开发者**：由 codecloud-dev 独立开发、开源
> **定位**：一个 **0.1B 级、纯 CPU 可训、极小体积、好部署、开源**的自研小模型。

molecule 的目标很单纯：**让一个能跑 llama.cpp 的弱设备（手机、树莓派、老旧/低配 CPU）也能本地跑起一个能对话的中文小模型**。它体积小（q4 量化后约 0.1GB）、完全开源、不依赖任何外部服务。架构走通用 decoder-only Transformer（与 Llama/Qwen 同源，好部署），把各家开源的注意力 / 归一化 / 训练技巧**精挑细选地融合**，并配一套**专治小参数困难的「小模型专项栈」**。

当前版本**不追求极致推理速度**（够用即可）；后续会单独做一个以速度为核心的 Flash 优化版本。

---

## 为什么做 molecule

小模型（0.1B）和 7B/70B 玩法完全不同：参数少、容量小、嵌入占比爆炸、易遗忘。**光靠通用 Transformer 不够**，所以 molecule 在"通用骨架 + 精挑融合"之外，专门做了小参数专项栈，详见 [DESIGN.md](./DESIGN.md)。

它的差异化不是跑分，而是**可移植性 + 开放性**：

| 维度 | molecule 的取舍 |
|---|---|
| 体积 | q4_K_M 量化后约 0.1GB，手机/树莓派直接放下 |
| 部署 | llama.cpp / Ollama 一行命令即可本地跑，不联网 |
| 训练 | 纯 CPU（GitHub Actions 免费 runner），单轮 ≤6h，跨轮续训累积 |
| 透明 | 架构、数据来源、训练算力、评测口径全部公开可查 |

---

## 融合架构（来源全部标注）

|                          | 来源                      | 作用                 |
| ------------------------ | ----------------------- | ------------------ |
| decoder-only Transformer | GPT/Llama 通用            | 通用骨架，好部署           |
| RoPE + NTK-aware 动态外推    | Llama/Qwen + NTK        | 训短推长外推最稳 |
| GQA（9 头 / 3 KV 头 = 3:1）  | Llama2/3、Qwen、MobileLLM | KV 省、推理省内存         |
| SwiGLU（FFN = 2.67×）      | Llama/Gemma/Qwen        | 质量/参数平衡            |
| RMSNorm + Pre-Norm       | 全系标配                    | 省参、稳训              |
| QK-Norm（RoPE 前）          | OLMo 2 / Gemma 3        | 稳训、允许更高 lr         |
| logit 软封顶（50/30）         | Gemma 2                 | 防训崩/掉点             |
| DeepNorm 增益缩放            | DeepNet                 | 30 层深窄稳训           |
| MTP 多令牌预测（共享头）           | DeepSeek-V3             | 训练密度↑（当前为辅助训练目标）       |
| 绑定嵌入（输入=输出）              | MobileLLM / PanGu-π     | reclaim 参数给网络主体    |

**维度对齐 MobileLLM-125M 实测黄金配置**：hidden 576 / FFN 1536 / 30 层 / 9 头 / 3 KV 头，参数 **约 129M**（"0.1B" 级）。

---

## 训练数据

- **自包含基线**：仓库内置 `seed_corpus.py` 种子语料（中文为主的多领域真实文本），**无需任何外部模型/网络即可从零训练**——这是保底路径。
- **蒸馏增强（CI 内纯 CPU 生成）**：由独立工作流 `distill.yml` 调用 5 位**开源**教师模型（均为 Apache-2.0 / MIT，无需 token）分领域生成高质量文本，学生在该文本上做 next-token 训练。机制详见 [DESIGN.md §4](./DESIGN.md)。
- **合规**：仅使用宽松许可证 + 自写/合成数据。
- 数据混比（单固定）：通用中文 40% / 数学 20% / 代码 20% / 推理 10% / EQ 对话 10%。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 训练（纯 CPU；首轮用内置种子语料，保证自包含必绿；可跨轮续训）
python train.py --max_steps 400 --seq_len 256 --resume

# 2) 评测（轻量自建冒烟集，趋势参考，非官方分数）
python eval.py

# 3) 导出 HF 模型目录（供 llama.cpp 转换）
python export_gguf.py

# 4) 推理 / 对话 / 工具调用
python generate.py --prompt "你好，你是谁？"
python generate.py --chat
python demo.py          # Gradio 网页对话
```

---

## 部署（好部署）

```bash
# 导出后由 llama.cpp 转换并量化（CI 自动完成）
git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py hf_model --outfile molecule-f16.gguf
./llama.cpp/llama-quantize molecule-f16.gguf molecule-q4_K_M.gguf q4_K_M
./llama.cpp/llama-server -m molecule-q4_K_M.gguf
```

产出：`safetensors` + `q8_0` + `q4_K_M`（~0.1GB，CPU/手机可跑）+ `imatrix` 提升 q4 质量。
亦可一行接入 Ollama（见各 Release 说明）。

---

## 评测（轻量冒烟集，非官方分数）

> ⚠️ **诚实声明**：仓库内置的 `benchmarks.py` 是**少量自写 MCQ 冒烟题**（每类 2–3 道），**不是** HellaSwag / ARC / MMLU / CMMLU / C-Eval 等官方基准的正式 split，结果**仅用于训练趋势参考，不可与外模型官方分数直接比较**。若需正式评测，请接入 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 与各基准官方 split。

内置冒烟集覆盖：hellaswag / arc / piqa / mmlu / winogrande / triviaqa / cmmlu / ceval（中文两类为自写常识题）。运行 `python eval.py` 生成 `eval_result.md` 对比表与 `eval_status.json`。

---

## 版本与发布

- 每次**训练成功**都会发布一个真实版本（GitHub Release），版本号由仓库根 `VERSION` 文件驱动：`0.1 → 0.2 → … → 0.9 → 1.0`。
- **封顶 V1.0**：到达 `1.0` 后不再递增（绝不出现 1.1）；`v1.0` 作为「滚动最佳」反复重建。
- 每个版本附带：`molecule-f16.gguf` / `molecule-q8_0.gguf` / `molecule-q4_K_M.gguf` + 当版评测说明。
- 发版后自动触发下一代训练以**续训累积**；仓库根放 `STOP_AUTOTRAIN` 文件即停止自动续训。

---

## 文档

- [DESIGN.md](./DESIGN.md)：架构与训练设计的实现蓝图（融合来源、数据、蒸馏、部署）。
- [MODEL_CARD.md](./MODEL_CARD.md)：模型卡（架构/数据/评测/局限/透明声明）。
- [NOTICE](./NOTICE)：融合来源与 Qwen2.5 分词器归属。
- [PITFALLS.md](./PITFALLS.md)：开发踩坑日记（CI / 代码 gotcha，动手前必读）。

## 许可证

Apache-2.0。融合的技巧均为公开论文/开源实现的方法（非代码复制）；分词器复用 Qwen2.5（Apache-2.0），归属见 NOTICE。

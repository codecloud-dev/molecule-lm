# molecule · V0.1

> **开源名**：`molecule` · **开源版本**：`v0.1` · **内部项目名**：分子模型开源版 · **内部版本**：`v31`
> **许可证**：Apache-2.0 · **训练**：GitHub Actions 纯 CPU，单轮 ≤6h、可跨轮续训
> **身份（固定）**：开源语言模型 · 代号 molecule · 由 codecloud-dev 独立开发（codecloud-dev 是它的开发者/家人）

molecule 是一个 **0.1B 级、纯 CPU 可训、好部署、开源**的自研小模型。架构走通用 decoder-only Transformer（与 Llama/Qwen 同源，好部署），把各家开源的注意力/归一化/训练技巧**精挑细选地融合**，并配一套**专治小参数困难的「分子小模型专项栈」**。目标：**六个基准逐项超过 SmolLM2-135M**（它中文≈0，是我们主突破口）。

---

## 为什么做 molecule

小模型（0.1B）和 7B/70B 玩法完全不同：参数少、容量小、嵌入占比爆炸、易遗忘。**光靠通用 Transformer 不够**，所以 molecule 在"通用骨架 + 精挑融合"之外，专门做了小参数专项栈，详见 [DESIGN.md](./DESIGN.md)（25 轮讨论的较真档案）。

## 融合架构（来源全部标注）

| 组件 | 来源 | 作用 |
|---|---|---|
| decoder-only Transformer | GPT/Llama 通用 | 通用骨架，好部署 |
| RoPE + NTK-aware 动态外推 | Llama/Qwen + NTK | 训 2048、推 4096 外推最稳 |
| GQA（9 头 / 3 KV 头 = 3:1） | Llama2/3、Qwen、MobileLLM | KV 省、推理快 |
| SwiGLU（FFN = 2.67×） | Llama/Gemma/Qwen | 质量/参数平衡 |
| RMSNorm + Pre-Norm | 全系标配 | 省参、稳训 |
| QK-Norm（RoPE 前） | OLMo 2 / Gemma 3 | 稳训、允许更高 lr |
| logit 软封顶（50/30） | Gemma 2 | 防训崩/掉点 |
| DeepNorm 增益缩放 | DeepNet | 30 层深窄稳训 |
| MTP 多令牌预测（共享头） | DeepSeek-V3 | 训练密度↑、推理投机加速 |
| 绑定嵌入（输入=输出） | MobileLLM / PanGu-π | reclaim 参数给网络主体 |

**维度对齐 MobileLLM-125M 实测黄金配置**：hidden 576 / FFN 1536 / 30 层 / 9 头 / 3 KV 头，参数 ≈ **129M**（"0.1B" 级）。

## 训练数据（中文为主 + 多老师分领域蒸馏）

5 位开源老师分领域 1对1 蒸馏：通用(Qwen2.5-0.5B-Instruct) / 推理(DeepSeek-R1-Distill-Qwen-0.5B) / 数学(Qwen2.5-Math-0.5B) / 代码(Qwen2.5-Coder-0.5B) / 情商(Qwen2.5-1.5B-Instruct，对标豆包的高情商)。混比：通用40 / 数学20 / 代码20 / 推理10 / EQ10。合规只用 CC/宽松 + 合成数据。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 训练（纯 CPU；首轮用内置种子语料，保证自包含必绿；可跨轮续训）
python train.py --max_steps 2000 --resume

# 2) 评测（zero-shot logprob，与 SmolLM2-135M 硬靶对比）
python eval.py

# 3) 导出 HF 模型目录（供 llama.cpp 转换）
python export_gguf.py

# 4) 推理 / 对话 / 工具调用
python generate.py --prompt "你好，你是谁？"
python generate.py --chat
python demo.py          # Gradio 网页对话
```

## 部署（好部署）

```bash
# 导出后由 llama.cpp 转换并量化（CI 自动完成）
git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py hf_model --outfile molecule-f16.gguf
./llama.cpp/llama-quantize molecule-f16.gguf molecule-q4_K_M.gguf q4_K_M
./llama.cpp/llama-server -m molecule-q4_K_M.gguf
```

产出：`safetensors` + `q8_0` + `q4_K_M`（~0.1GB，CPU/手机可跑）+ `imatrix` 提升 q4 质量。

## 评测目标（惊艳达标线）

| 基准 | SmolLM2-135M | 我们目标 |
|---|---|---|
| HellaSwag / ARC / PIQA / MMLU / Winogrande | 68.4 / 43.9 / 68.4 / 31.5 / 51.3 | 英文不能掉 |
| TriviaQA | 4.1 | 弱项，必超 |
| CMMLU / C-Eval | ≈0 | 中文主突破口 |

六项**逐项超过 SmolLM2（±0.5 容差）**即视为"惊艳达标"，自动建 GitHub Release。

## 文档

- [DESIGN.md](./DESIGN.md)：25 轮讨论锁死的较真档案（从训练到部署全细节）
- [MODEL_CARD.md](./MODEL_CARD.md)：模型卡（架构/数据/评测/局限/透明声明）
- [NOTICE](./NOTICE)：融合来源与 Qwen2.5 分词器归属

## 许可证

Apache-2.0。融合的技巧均为公开论文/开源实现的方法（非代码复制）；分词器复用 Qwen2.5（Apache-2.0），归属见 NOTICE。

# molecule · 设计档案（实现蓝图）

> **开源名**：`molecule` · **许可证**：Apache-2.0 · **公开仓库**：`codecloud-dev/molecule-lm`
> **训练**：GitHub Actions 纯 CPU，单轮 ≤6h
> 本文件是从训练到部署的实现蓝图，任何改动需回到这里对齐。

---

## 0. 一句话定位

做一个 **0.1B 级、纯 CPU 可训、极小体积、好部署、开源**的自研小模型。架构走通用 decoder-only Transformer（与 Llama/Qwen 同源，好部署），精挑细选地融合各家开源的注意力/归一化/训练技巧，配一套**专治小参数困难的「小模型专项栈」**。

**主目标（部署场景）**：让一个能跑 llama.cpp 的弱设备——手机、树莓派、老旧/低配 CPU——也能本地跑起一个能对话的中文小模型。体积小、完全开源、不联网、不依赖外部服务。**当前版本不追求极致推理速度**（够用即可）；以速度为核心的 Flash 优化版本列为后续路线。

**验收口径（诚实）**：不做"超过某模型"的硬性承诺。用内置轻量冒烟集做训练趋势监控；如需对外对比，用 lm-evaluation-harness + 官方 split 实测，并如实标注。

---

## 1. 核心架构（融合来源全部标注）

| 组件 | 来源（开源实现） | 决策 | 理由 |
|---|---|---|---|
| decoder-only Transformer 骨架 | GPT/Llama 通用范式 | ✅ | 通用架构，好部署 |
| **RoPE 旋转位置编码 + NTK-aware 动态外推** | Llama/Mistral/Qwen/DeepSeek + NTK-aware scaling | ✅ | 替代绝对位置；训短、推长动态放大 base，保高频局部分辨率 |
| **GQA 分组查询注意力（9 头 / 3 KV 头 = 3:1）** | Llama2/3、Qwen、Gemma、MobileLLM | ✅ | KV 缓存省、质量几乎不降，小模型推理友好 |
| **SwiGLU 激活（FFN = 2.67× hidden）** | Llama/Gemma/Qwen/DeepSeek | ✅ | 8/3 标准预算，质量/参数最平衡 |
| **RMSNorm + Pre-Norm** | 全系标配 | ✅ | 省参数、训练稳 |
| **QK-Norm（RoPE 前对 Q/K 各加 RMSNorm）** | OLMo 2 / Gemma 3 / MiniMax-M2 | ✅ | 几乎零成本显著稳训，允许更高学习率 |
| **logit 软封顶（注意力 50.0 / 输出 30.0）** | Gemma 2 | ✅ | 防极端 logit 致训崩/掉点；CPU 用 eager 注意力无 FlashAttention 兼容问题 |
| **DeepNorm 增益缩放（30 层深窄稳训）** | DeepNet / DeepSeek | ✅ | 残差/注意力输出加增益因子，深窄不梯度爆炸 |
| **MTP 多令牌预测（1 模块，共享嵌入+输出头）** | DeepSeek-V3 | ✅ | 训练密度↑；**当前仅作辅助训练目标，未实现投机解码** |
| **绑定嵌入（输入=输出 embedding）** | MobileLLM / PanGu-π | ✅ | 砍词表到 32K 后 reclaim ~20% 参数给网络主体 |
| ❌ 滑动窗口注意力 | Mistral/Gemma | 不融 | 短上下文小模型用不上 |
| ❌ MLA 低秩 KV 压缩 | DeepSeek | 不融 | 0.1B 不划算 |
| ❌ MoE | DeepSeek/Qwen | 坚决不融 | 小模型加了伤质量 |

### 1.1 维度（对齐 MobileLLM-125M 实测黄金配置）
- `hidden_size = 576`
- `intermediate_size = 1536`（= 8/3 × 576，SwiGLU 标准预算）
- `num_hidden_layers = 30`（深而窄）
- `num_attention_heads = 9`，`num_key_value_heads = 3`（GQA 3:1）
- `head_dim = 64`（576/9）
- `vocab_size = 32768`（32K，由 Qwen2.5 词表按语料频率裁剪）
- 全无 bias（Llama/Qwen 式）
- 参数总量 **≈ 129M**（"0.1B" 级）

### 1.2 训练数值
- 精度：**自动探测**——程序先查 CPU 是否支持 bf16/AMX，支持才用 bf16 混合精度，**否则回退纯 fp32**。
- 上下文 `max_position_embeddings = 4096`，**实际训练序列 `seq_len = 256`**（CI runner 内存约束下的取值；架构支持最长 4096 外推）。
- 随机种子 `seed = 42` + torch 确定性模式，保证可复现/可归因。

---

## 2. 分词器（复用 Qwen2.5 + 砍表 + 绑定）
- **复用 Qwen2.5 分词器**（中文已优化、与多老师同词表空间蒸馏顺、生态通用）。
- 从 Qwen 原版词表**按实际训练语料词频保留最高频 32K**；模型本身仍完全自研。
- **绑定输入/输出嵌入**（reclaim 参数）。
- 特殊符复用 Qwen 的 `<|endoftext|>/<|im_start|>/<|im_end|>` 等。
- 未登录字符用 **byte-level fallback**（Qwen 自带 byte token，永不乱码）。
- 对话模板 **ChatML**（Qwen 格式）。

---

## 3. 训练数据

- **混比（单固定）**：通用中文 40% + 数学 20% + 代码 20% + 推理 10% + EQ 对话 10%。
- **实际使用的来源**（如实）：
  - 仓库内置 `seed_corpus.py` **自写种子语料**（中文为主的多领域真实文本）——从零训练保底，**无需任何外部模型/网络**；
  - **教师模型生成的合成文本**（见第 4 节），按领域并入训练；
  - 数据快照随训练 artifact 留存，脚本可复现。
- **合规**：仅使用宽松许可证 + 自写/合成数据。
- 数据可复现：合成+精选数据随脚本可复现生成。

> 说明：未使用维基百科等外部语料；此前文档中"主要来源中文维基"的表述与实现不符，已更正。

---

## 4. 多老师分领域 1对1 生成式数据蒸馏（5 位**开源**老师）

| 领域 | 老师模型 | 许可证 | 参数量 | 温度 |
|---|---|---|---|---|
| 通用中文 | Qwen2.5-0.5B-Instruct | Apache-2.0 | 0.5B | 0.9（偏多样） |
| 推理/长链思考 | DeepSeek-R1-Distill-Qwen-0.5B | MIT | 0.5B | 0.7（偏准确） |
| 数学 | Qwen2.5-Math-0.5B | Apache-2.0 | 0.5B | 0.6（偏准确） |
| 代码 | Qwen2.5-Coder-0.5B | Apache-2.0 | 0.5B | 0.6（偏准确） |
| 语言组织/EQ | Qwen2.5-1.5B-Instruct | Apache-2.0 | 1.5B | 0.9（偏多样暖） |

- **运行位置**：由独立工作流 `distill.yml` 在 GitHub Actions 免费 runner（**纯 CPU**）上生成。5 个域走 matrix 并行（各跑一个教师，≈5 倍提速），HuggingFace 模型目录用 `actions/cache` 缓存避免重复下载。**教师均为开源许可（Apache-2.0 / MIT），无需配置 HF_TOKEN；CI 仅需能联网访问 HuggingFace**。
- **蒸馏机制（生成式数据蒸馏，当前实际落地）**：教师生成**高质量合成文本**，学生在该文本上做 next-token **硬 CE** 训练（词表天然对齐我们的 32K 裁剪空间）。这是当前落地的蒸馏方式（Phi/Gemma 同路线），不依赖教师在线推理。
- **软蒸馏 KL（预留，未启用）**：`loss_soft_kl=0.0`。真正的 soft-label KL 需教师在线产出 soft logits，纯 CPU 不现实，列为后续 GPU 路线；届时温度 T=2.0、权重 0.5。
- **域平衡**：数据按上述领域配比**混合采样**实现域平衡（硬 CE 路径天然支持）。
- **断点续跑累积**：`distill.py` 按 domain 记录条数，软限到点优雅停，跨 run 续跑；产物 `teacher_*.jsonl` 保留 90 天，可多次跑累积语料。

---

## 5. 训练配方

- **优化器**：Muon（2D 权重）+ AdamW（嵌入/bias/1D）混合；WSD 调度（warmup 6%、stable 90%、decay 末 10%）。
- **超参基线**：lr=5e-4（AdamW 部分）/ Muon 部分按其特征 lr；weight_decay=0.05；grad_clip=1.0。
- **梯度累积**：按 CPU 内存自动算最大 microbatch，累积到目标 global batch（塞满算力）。
- **总损失**：硬 CE 1.0（主骨架）/ MTP 0.1（轻量辅助）。软蒸馏 KL 当前为 0——生成式数据蒸馏已通过 hard CE 吸收教师知识；soft KL 预留为 GPU 路线。
- **续训机制**：存**全量续训态**——model state_dict + Muon/AdamW optimizer state + step 计数 + 数据游标 + RNG 种子；artifact 按 run 编号；下一轮 load 继续，累计 token。
- **保存频率**：每 2000 step 存一次 + 每轮末强制存。
- **首轮预算（CI 实际）**：序列 256、global batch ~16K token、单轮上限 400 步（受 runner 内存/6h 约束），单轮训练数千万 token 量级。架构设计目标为更长序列与更高吞吐，留待资源放开或 Flash 版本。

---

## 6. 评测（轻量冒烟集，趋势参考）

> ⚠️ **诚实声明**：`benchmarks.py` 为**少量自写 MCQ 冒烟题**，**非** HellaSwag/ARC/MMLU/CMMLU/C-Eval 的官方 split。结果**仅用于训练趋势监控**，**不可与外模型官方分数直接比较**。正式评测请接入 lm-evaluation-harness + 官方 split。

- **协议**：zero-shot；MCQ 对 A/B/C 算 logprob 归一化取最大（CPU 友好、无生成随机性）；内置精简冒烟集（自包含，CI 不依赖下载）。
- **覆盖**：hellaswag / arc / piqa / mmlu / winogrande / triviaqa / cmmlu / ceval（中文两类为自写常识题）。
- **评测频率**：每轮末跑内置冒烟集出趋势表；正式 benchmark 作为后续增强项。

---

## 7. 部署（好部署）

- **导出**：自写 GGUF 导出（`gguf` python 库），完全控制 tied embeddings（`output.weight` 复用 `token_embd.weight`）和 32K 自定义词表 + Qwen pre-tokenizer 写入；再用 llama.cpp 的 `llama-quantize` 出 `q8_0`（近无损）+ `q4_K_M`（~0.1GB，手机/CPU 可跑）。
- **imatrix**：生成重要性矩阵后再量化，提升 q4 质量。
- **server 示例**：提供 llama.cpp server 启动脚本 + 推理示例；可一行接入 Ollama。
- **Gradio demo**：附本地一行启动对话 demo。
- **工具调用**：支持轻量 function calling（0.1B 容量有限，专注对话+中文+工具格式）。

---

## 8. 版本与发布

- **版本号**：仓库根 `VERSION` 文件驱动，从 `0.1` 起逐次递增（`0.1 → 0.2 → … → 0.9 → 1.0`）。
- **每次训练成功都发布一个真实版本**（GitHub Release，非 draft），附 `molecule-f16.gguf` / `q8_0` / `q4_K_M` + 当版评测说明。
- **封顶 V1.0**：到达 `1.0` 后不再递增（绝不出现 1.1）；`v1.0` 作为「滚动最佳」反复重建。
- **自动续训**：发版后自动触发下一代训练以续训累积；仓库根放 `STOP_AUTOTRAIN` 文件即停止自动续训。
- 仅 GitHub Release，不自动推 HuggingFace（如需多平台分发，后续手动同步）。

---

## 9. 风险与对策

- **CPU 6h 跑不完 / 内存受限**：跨轮续训 + checkpoint 累积；加速四招（预分词 mmap / packing / torch.compile / 最大 batch + 线程调优）；当前 CI 取 seq_len 256 等约束值。
- **bf16 硬件不支持**：程序自动探测，回退 fp32。
- **小模型嵌入占比爆炸**：砍词表到 32K + 绑定嵌入。
- **数据合规**：只用宽松许可 + 自写/合成。
- **评测口径误导**：内置冒烟集明确标注"非官方 split、仅趋势参考"，禁止据此做跨模型对比或商业宣传。
- **蒸馏偏差**：始终保留硬 CE 保底。

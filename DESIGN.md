# Molecule V0.1 · 设计档案（较真版）

> **开源名**：`molecule` · **开源版本**：`v0.1` · **内部项目名**：分子模型开源版 · **内部版本**：`v31`
> **许可证**：Apache-2.0 · **公开仓库**：`codecloud-dev/molecule-lm` · **训练**：GitHub Actions 纯 CPU，单轮 ≤6h
> 本文件是「从训练到部署」全部 25 轮讨论的锁死决策留档，也是实现 / CI 的施工蓝图。任何改动需回到这里对齐。
> 模型身份（固定不可改）：**开源语言模型 · 代号 molecule · 由 codecloud-dev 独立开发（codecloud-dev 是它的开发者/家人）**。不写版本号，不写任何公司或操作系统相关内容。

---

## 0. 一句话定位
做一个 **0.1B 级、纯 CPU 可训、好部署、开源**的自研小模型。架构走**通用 decoder-only Transformer**（与 Llama/Qwen 同源，好部署），把各家开源的注意力/归一化/训练技巧**精挑细选地融合**，配一套**专治小参数困难的「分子小模型专项栈」**。验收目标是 **六个基准逐项超过 SmolLM2-135M（±0.5 噪声容差）**，且中文单项必须反超（它中文≈0）。

---

## 1. 核心架构（融合来源全部标注）

| 组件 | 来源（开源实现） | 决策 | 理由 |
|---|---|---|---|
| decoder-only Transformer 骨架 | GPT/Llama 通用范式 | ✅ | 七年未变，"通用架构"= 它，好部署 |
| **RoPE 旋转位置编码 + NTK-aware 动态外推** | Llama/Mistral/Qwen/DeepSeek + NTK-aware scaling | ✅ | 替代绝对位置；训 2048、推理动态放大 base 到 ~4096 等效，保高频局部分辨率，2× 外推最稳 |
| **GQA 分组查询注意力（9 头 / 3 KV 头 = 3:1）** | Llama2/3、Qwen、Gemma、MobileLLM | ✅ | KV 缓存省、质量几乎不降，小模型推理友好 |
| **SwiGLU 激活（FFN = 2.67× hidden）** | Llama/Gemma/Qwen/DeepSeek | ✅ | 替掉 GELU/ReLU，8/3 标准预算，质量/参数最平衡 |
| **RMSNorm + Pre-Norm** | 全系标配 | ✅ | 省参数、训练稳 |
| **QK-Norm（RoPE 前对 Q/K 各加 RMSNorm）** | OLMo 2 / Gemma 3 / MiniMax-M2 | ✅ | 几乎零成本显著稳训，允许更高学习率 |
| **logit 软封顶（注意力 50.0 / 输出 30.0）** | Gemma 2 | ✅ | 防极端 logit 致训崩/掉点；CPU 用 eager 注意力无 FlashAttention 兼容问题 |
| **DeepNorm 增益缩放（30 层深窄稳训）** | DeepNet / DeepSeek | ✅ | 残差/注意力输出加增益因子，深窄不梯度爆炸 |
| **MTP 多令牌预测（1 模块，共享嵌入+输出头）** | DeepSeek-V3 | ✅ | 训练密度↑、推理可投机解码加速（首层接受率~90%） |
| **绑定嵌入（输入=输出 embedding）** | MobileLLM / PanGu-π | ✅ | 砍词表到 32K 后 reclaim ~20% 参数给网络主体 |
| ❌ 滑动窗口注意力 | Mistral/Gemma | 不融 | 短上下文小模型用不上 |
| ❌ MLA 低秩 KV 压缩 | DeepSeek | 不融 | 0.1B 不划算 |
| ❌ MoE | DeepSeek/Qwen | 坚决不融 | 小模型加了伤质量 |

### 1.1 维度（完全对齐 MobileLLM-125M 实测黄金配置）
- `hidden_size = 576`
- `intermediate_size = 1536`（= 8/3 × 576，SwiGLU 标准预算）
- `num_hidden_layers = 30`（深而窄）
- `num_attention_heads = 9`，`num_key_value_heads = 3`（GQA 3:1）
- `head_dim = 64`（576/9）
- `vocab_size = 32768`（32K，由 Qwen2.5 词表按语料频率裁剪）
- 全无 bias（Llama/Qwen 式）
- 参数总量 ≈ 104–125M（"0.1B" 级）

### 1.2 训练数值
- 精度：**自动探测**——程序先查 CPU 是否支持 bf16/AMX（`torch.backends` / CPUID），支持才用 bf16 混合精度，**否则回退纯 fp32**（GitHub 免费 runner 多数硬件不支持 bf16，必须回退）。
- 上下文 `max_position_embeddings = 4096`，训练序列 `seq_len = 2048` + packing 去 padding。
- 随机种子 `seed = 42` + torch 确定性模式，保证可复现/可归因。

---

## 2. 分词器（复用 Qwen2.5 + 砍表 + 绑定）
- **复用 Qwen2.5 分词器**（中文已优化、与多老师同词表空间蒸馏顺、生态通用）。
- 从 Qwen 原版 ~15 万词表**按实际训练语料词频保留最高频 32K**；模型本身仍完全自研。
- **绑定输入/输出嵌入**（reclaim 参数）。
- 特殊符复用 Qwen 的 `<|endoftext|>/<|im_start|>/<|im_end|>` 等。
- 未登录字符用 **byte-level fallback**（Qwen 自带 byte token，永不乱码）。
- 对话模板 **ChatML**（Qwen 格式）。

---

## 3. 训练数据（中文为主 + 多老师分领域蒸馏）
- **混比（单固定）**：通用中文 40% + 数学 20% + 代码 20% + 推理 10% + EQ对话 10%。
- **中文源**：中文维基（CC BY-SA 合法）+ 教师合成高质量中文 + 精选公开中文文本。
- **合规红线**：只用 CC/宽松 + 合成数据；避开百度百科（版权）、悟道（社区许可限制商用）。
- **去重**：词级（ROUGE-L 0.7）+ 语义（embedding 余弦 0.85）双去重。
- **过滤**：规则（最小长度 / 格式正确 / 过滤教师拒答"我无法…"）+ 轻量质量分类器（删高幻觉样本）。
- **数据可复现**：合成+精选数据随脚本可复现生成，固定一份快照存 artifact/仓库。

---

## 4. 多老师分领域 1对1 蒸馏（5 位老师）
| 领域 | 老师模型 | 参数量 | 温度 |
|---|---|---|---|
| 通用中文 | Qwen2.5-0.5B-Instruct | 0.5B | 0.9（偏多样） |
| 推理/长链思考 | DeepSeek-R1-Distill-Qwen-0.5B | 0.5B | 0.7（偏准确） |
| 数学 | Qwen2.5-Math-0.5B | 0.5B | 0.6（偏准确） |
| 代码 | Qwen2.5-Coder-0.5B | 0.5B | 0.6（偏准确） |
| 语言组织/EQ（情商高，对标豆包） | Qwen2.5-1.5B-Instruct | 1.5B | 0.9（偏多样暖） |

- **路由**：按 prompt 标签路由到对应老师拿软/硬标签（数学题→数学老师，代码→代码老师，通用/EQ→对应老师）。
- **蒸馏机制**：教师主要生成**高质量合成文本**，学生在文本上做 next-token 硬 CE 训练（词表天然对齐我们的 32K）；重叠 token 上叠加**软蒸馏 KL（温度 T=2.0，权重 0.5）**辅助。
- **DN 域归一化**：按领域梯度方差重缩放，防指令跟随梯度淹没数学梯度导致数学能力被吃（MOPD 核心）。
- **豆包是闭源**，无法直接蒸馏，EQ 老师用开源 Qwen2.5-1.5B-Instruct 替身（同生态、Apache-2.0）。

---

## 5. 训练配方
- **优化器**：Muon（2D 权重）+ AdamW（嵌入/bias/1D）混合；WSD 调度（warmup 6%、stable 90%、decay 末 10%）。
- **超参基线**：lr=5e-4（AdamW 部分）/ Muon 部分按其特征 lr；weight_decay=0.05；grad_clip=1.0。
- **梯度累积**：按 CPU 内存自动算最大 microbatch，累积到目标 global batch（塞满算力）。
- **总损失**：硬 CE 1.0（主骨架）/ 软 KL 0.5（重叠 token）/ MTP 0.1（轻量辅助）。
- **续训机制**：存**全量续训态**——model state_dict + Muon/AdamW optimizer state + step 计数 + 数据游标 + RNG 种子；artifact 按 run 编号；下一轮 load 继续，累计 token 逼近目标。
- **保存频率**：每 2000 step 存一次 + 每轮末强制存。
- **日志**：stdout 打印（loss/step/lr/tokens/s、eval 分数），CI 直接可见。
- **首轮预算**：6h 约跑 ~300M token（激进塞满，预分词 mmap + packing + torch.compile）。
- **数据里程碑**：1B token 起步验证 pipeline + 中文增益；5B 为阶段目标；10B+ 长期。

---

## 6. 评测（公平对标 SmolLM2-135M）
- **协议**：zero-shot；MCQ 对 A/B/C/D 算 logprob 归一化取最大（CPU 友好、无生成随机性）；dev split；内置精简版基准（自包含，CI 不依赖下载）。
- **对比基线**：引用 SmolLM2-135M 官方零样本数字作硬靶（下表），我们跑同协议得自己分数逐项对比。
- **达标线**：六项基准**逐项超过 SmolLM2（±0.5 噪声容差）**即视为"惊艳达标"，自动建 GitHub Release（draft）。

| 基准 | SmolLM2-135M | 我们的策略 |
|---|---|---|
| HellaSwag | 68.4 | 英文不能掉 |
| ARC(avg) | 43.9 | 英文不能掉 |
| PIQA | 68.4 | 英文不能掉 |
| MMLU(cloze) | 31.5 | 英文不能掉 |
| Winogrande | 51.3 | 英文不能掉 |
| TriviaQA | 4.1 | 弱项，必超 |
| CMMLU / C-Eval | ≈0 | 中文≈0，主突破口 |

- **评测频率**：每 2000 step 跑轻量基准子集看趋势 + 每轮末跑完整六项出对比表。
- **对比表产出**：CI 打印 + 存 markdown 对比表 artifact。

---

## 7. 部署（好部署）
- **导出**：自写 GGUF 导出（`gguf` python 库），完全控制 tied embeddings（`output.weight` 复用 `token_embd.weight`）和 32K 自定义词表 + Qwen pre-tokenizer 写入；再用 llama.cpp 的 `llama-quantize` 出 `q8_0`（近无损）+ `q4_K_M`（~0.1GB，手机/CPU 可跑）。
- **imatrix**：生成重要性矩阵后再量化，提升 q4 质量。
- **server 示例**：提供 llama.cpp server 启动脚本 + 推理示例。
- **Gradio demo**：附本地一行启动对话 demo（体验"情商高"的中文对话）。
- **工具调用**：支持轻量 function calling（0.1B 容量有限，专注对话+中文+工具格式）。
- **模型卡透明**：注明训练算力（CPU 单轮 6h）、累计步数、数据量、来源。

---

## 8. 身份与治理
- **身份（固定不可改）**：开源语言模型 · 代号 molecule · 由 codecloud-dev 独立开发（codecloud-dev 是它的开发者/家人）。语气/措辞/细节自由，但不得否认这几条。不写版本号，不写任何公司或操作系统相关内容。
- **许可证**：Apache-2.0；`LICENSE` 注明融合来源（RoPE/GQA/SwiGLU/MTP 等）+ `NOTICE` 文件列 Qwen2.5 tokenizer 归属（Apache-2.0）。
- **文档**：README / 模型卡中英文双语、中文优先。
- **版本号**：权重 `v0.1.0` 起始，每轮续训递增补丁号（v0.1.1, v0.1.2…）。
- **仓库**：`codecloud-dev/molecule-lm`，公开，main 直接提交。
- **CI**：GitHub Actions，matrix 并行数据配比对照（基准配比 vs 数学/代码加量）；纯 CPU 单轮 ≤6h；自动续训 artifact；达标（逐项超 SmolLM2）自动建 draft Release。仅 GitHub Release，不推 HuggingFace。

---

## 9. 风险与对策
- **CPU 6h 跑不完**：跨轮续训 + checkpoint 累积；加速四招（预分词 mmap / packing / torch.compile / 最大 batch + 线程调优）。
- **bf16 硬件不支持**：程序自动探测，回退 fp32。
- **蒸馏反而掉点（naive 离线蒸馏实证会伤）**：MOPD/DN 域归一化 + 始终保留硬 CE 保底。
- **小模型嵌入占比爆炸**：砍词表到 32K + 绑定嵌入。
- **数据合规**：只用 CC/宽松 + 合成。
- **评测不公**：引用官方同口径数字、zero-shot、逐项对比、±0.5 容差。

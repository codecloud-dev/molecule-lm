#!/usr/bin/env python3
# 生成 molecule 每个累积版本的 Release 说明（notes.md）。
# 由 package.yml 在每次训练成功后调用；VERSION 文件驱动版本号。
import json
import os

VERSION = open("VERSION").read().strip()
TAG = "v" + VERSION

# 固定身份句：不写版本号、不写公司
IDENTITY = "开源语言模型·代号 molecule·由 codecloud-dev 独立开发（开发者/家人）"

eval_note = "（暂无评测产物）"
p = "eval/eval_status_baseline.json"
if os.path.exists(p):
    try:
        d = json.load(open(p))
        eval_note = (
            f"baseline all_pass={d.get('all_pass')}；"
            "详见 artifact eval-baseline / eval_result_baseline.md"
        )
    except Exception as e:  # noqa: BLE001
        eval_note = f"（评测产物读取失败：{e}）"

notes = f"""# molecule {TAG}

{IDENTITY}

## 版本 {TAG}（累积训练发布）

纯 CPU 训练的开源语言模型，单次训练 ≤6h，跨轮续训累积 checkpoints。本版本基于 VERSION={VERSION} 自动发布。

- **架构**：decoder-only Transformer（约 129M），融合 RoPE-NTK / SwiGLU / RMSNorm+Pre-Norm / QK-Norm / logit soft-capping / DeepNorm / MTP（1 模块）/ 权重共享。
- **分词**：Qwen2.5 tokenizer 裁 32K + 字节级兜底。
- **训练**：Muon+AdamW 混合、WSD、bf16、seed=42、全量 resume。
- **蒸馏**：CI 内 5 个开源教师模型（general / reason / math / code / eq）分领域生成式数据蒸馏。
- **能力**：支持工具调用。

### 诚实现状
早期累积版本，训练算力有限（纯 CPU、单次 ≤6h），与 SmolLM2-135M 等成熟小模型相比仍有差距，随续训累积持续提升。{eval_note}

### 权重
- `molecule-q4_K_M.gguf`（推荐，体积小、够用）
- `molecule-q8_0.gguf`
- `molecule-f16.gguf`

### 快速使用
```bash
# llama.cpp
./llama-cli -m molecule-q4_K_M.gguf -p "你好"
```
（亦可自行转换为 Ollama / llama.cpp server 等。）
"""

with open("notes.md", "w", encoding="utf-8") as f:
    f.write(notes)

print("notes.md written for", TAG)

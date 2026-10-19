---
license: apache-2.0
language:
  - zh
  - en
---

# 模型卡 · molecule

> **开源名**：`molecule` · **开发者**：codecloud-dev（独立开发、开源）· **许可证**：Apache-2.0
> **仓库**：[codecloud-dev/molecule-lm](https://github.com/codecloud-dev/molecule-lm)
> **当前版本**：见仓库 `VERSION` 文件与各 GitHub Release 标签（从 `v0.1` 起，封顶 `V1.0`）

## 模型简介

molecule 是一个 **0.1B 级（约 129M 参数）** 的轻量开源语言模型，主打**在资源受限的设备上本地运行**。它体积小、完全开源、不依赖任何外部服务，主要面向**中文对话与轻量文本生成**场景——手机、树莓派、老旧 / 低配 CPU 都能跑。

主要特点：

- **体积小**：q4_K_M 量化后约 0.1GB，手机 / 树莓派可直接放下。
- **好部署**：导出为 GGUF，可被 llama.cpp / Ollama 等常见推理引擎直接加载，纯本地、不联网。
- **开源透明**：训练、数据来源、评测口径全部公开可查。
- **中文友好**：以中文对话与轻量指令跟随为主要场景。

| 项目 | 值 |
| --- | --- |
| 参数总量 | 约 129M（"0.1B" 级） |
| 类型 | 纯解码器（decoder-only）语言模型 |
| 词表 | 32K |
| 量化体积 | q4_K_M 约 0.1GB |
| 上下文 | 支持最长 4096 token |

## 用途

- **中文对话 / 指令跟随**：轻量中文对话、日常问答。
- **弱设备本地部署**：q4_K_M 量化后约 0.1GB，可在手机 / 树莓派 / 纯 CPU 环境本地运行，不联网。
- **轻量工具调用**：内置轻量 function calling（取时间、计算）。
- **研究与教学**：完整训练 / 蒸馏 / 评测 / 导出代码开源，可作为小模型训练范式参考。

**不适用**：高精度专业领域（法律、医疗）、长文档理解、与 1B+ 模型同等的广知识与强推理。小模型容量有限，请管理预期。

## 怎么用

### 方式一：直接下载 GGUF（推荐，无需训练）

从 GitHub Releases 取一个发布版 GGUF（推荐 `q4_K_M`），用 llama.cpp 直接对话：

```bash
llama-cli -m molecule-q4_K_M.gguf -p "你好，介绍一下你自己"
# 或起服务：llama-server -m molecule-q4_K_M.gguf
```

### 方式二：本地推理 / 对话 / 网页

```bash
pip install -r requirements.txt

python generate.py --prompt "你好，你是谁？"   # 单次生成
python generate.py --chat                      # 多轮对话
python demo.py                                 # Gradio 网页对话
```

更多用法见 [README.md](./README.md)。

## 评测

> ⚠️ **诚实声明**：仓库内置 `benchmarks.py` 为**少量自写冒烟题**，**非** HellaSwag / ARC / MMLU / CMMLU / C-Eval 等官方基准的正式 split。下表数字**仅用于训练趋势参考，不可与外模型官方分数直接比较**。正式评测请接入 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) + 各基准官方 split。

运行 `python eval.py` 生成当版评测趋势表与状态文件。

## 局限

- **容量有限**：0.1B 无法承载大模型的广知识与强推理，专业 / 长文场景请降预期。
- **知识截止与幻觉**：训练数据规模与时效有限，可能包含事实性错误，输出需人工核验。
- **评测口径**：内置冒烟集为自写题，**非官方基准分数**，请勿据此做跨模型对比或商业宣传。
- **安全**：本模型非为安全对齐深度优化，输出需人工核验后使用。

## 许可证

Apache-2.0。分词器复用 Qwen2.5（Apache-2.0），归属见 [NOTICE](./NOTICE)。

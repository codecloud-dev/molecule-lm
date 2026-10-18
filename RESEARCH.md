# 开源语言模型仓库调研（结构 / 呈现 / 传播）

> 目的：调研主流开源小模型仓库「怎么写、怎么好用、怎么有流量」，据此打磨 molecule-lm 的公开呈现与工程结构。
> 范围：仅公开仓库结构与文档实践，**不涉及任何内部信息**；本文件可随仓库公开。

## 1. 调研对象与要点

| 仓库 | 体量 / 定位 | 结构亮点 | 文档亮点 | 可借鉴 |
| --- | --- | --- | --- | --- |
| **SmolLM2**（HuggingFaceTB） | 135M / 360M / 1.7B，on-device 轻量 | 顶级目录按用途拆分：`pre-training/`、`finetuning/`、`evaluation/`、`local_inference/`、`distilabel_pipelines/`、`smol_tools/` | 顶级 README 含目录导航 + `Usage`（transformers 一行载入）+ 每尺寸独立 model card；Apache-2.0 | 按用途拆分目录、每尺寸独立 model card、开箱即用的载入示例 |
| **tiny-qwen**（社区复现） | 极简 PyTorch 复现，仅 ~190★ 但结构干净 | `model/` `train/` `data/` + `run.py` 交互聊天 + `requirements.txt` | 中英双语 README、Quick Start、代码样例 | 「极简 + 可读 + 开箱即用」即便小项目也能涨 star |
| **Qwen / DeepSeek / Phi** | 全家桶、工业级 | 统一 `README.md` + `README_CN.md`、量化/推理示例、生态链接 | 清晰的能力边界声明、版本与变更记录 | 双语 README、能力边界诚实声明、量化部署指引 |

## 2. 共同规律（高可用 + 高传播仓库的共性）

1. **一眼能跑**：README 前几屏就有「安装 → 载入 → 一行推理」的复制即用的代码块。
2. **目录即导航**：代码按职责（训练 / 推理 / 评测 / 数据）拆分，不堆在一个文件。
3. **能力边界诚实**：明确「能做什么、不做什么、已知局限」，反而更可信、更少被骂。
4. **双语 + 轻量文档**：中文项目配 `README.md` + `README_CN.md` 覆盖面更广（见 Qwen）。
5. **量化/弱设备友好**：显式给出 GGUF / llama.cpp / MLX / 4-bit 体积与内存，on-device 定位清晰（SmolLM2 强项）。
6. **独立 model card**：每个发布版本有独立说明（架构、数据、评测、许可），而非只塞进一个长 README。

## 3. molecule-lm 现状对照

| 维度 | 现状 | 与标杆差距 | 计划 |
| --- | --- | --- | --- |
| 开箱即用载入 | `generate.py` / `export_gguf.py` 已提供，README 有示例 | 示例分散，未聚焦「3 步跑起来」 | README 强化「安装→载入→推理」3 步块 |
| 目录拆分 | `train.py` `eval.py` `model.py` `distill.py` `tokenizer.py` 已按职责拆分 | 已基本对齐 | 保持，新增代码仍按职责落位 |
| 能力边界 | `MODEL_CARD` / `DESIGN` 已诚实声明局限 | 已较好 | 维持诚实，新增能力即补局限 |
| 弱设备定位 | 已定「弱设备可跑、体积小、可移植优先」，GGUF 量化已落地 | 已对齐 SmolLM2 强项 | README 显式给出体积/内存 |
| 版本数据 | 每版落 `releases/vX.Y.md` + `benchmarks/vX.Y.json` | 已超越多数小项目 | 维持，release 页附链接 |
| 双语 | 当前仅中文 README | 差于 Qwen 实践 | 后续补 `README_EN.md`（非当前重点） |

## 4. 结论（落地优先级）

- **高优先（已在做）**：弱设备可跑 + GGUF 量化 + 每版数据入库 + 诚实能力边界——这些正是 SmolLM2 高传播的核心，我们已对齐。
- **中优先**：README「3 步跑起来」前置、体积/内存数字显式化、目录导航清晰——零成本、直接涨可用性。
- **低优先（暂缓）**：双语 README、官网——你明确表示官网自己来做，双语可后置。

> 注：本仓库不追求「对标/超越某具体模型」的对外表述；CI 内部保留基线趋势比对仅作训练监控，不对外出胜负声明。

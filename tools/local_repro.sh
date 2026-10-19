#!/bin/bash
# molecule 本地离线复现（开发者本地用，不联网、不依赖 checkpoint）。
#
# 用法：
#   bash tools/local_repro.sh
#
# 它会跑三个离线自测，全程确定性离线（HF_HUB_OFFLINE=1 防止联网挂起），约几秒：
#   tests/test_offline.py            模型前向 / 字节级分词中英文往返 / KV 缓存一致性
#   tests/test_export_gguf.py        离线导出合法 GGUF（llama.cpp 可加载）—— 弱设备可跑前提
#   tests/test_generate_offline.py   离线聊天路径（build_messages -> generate_text）—— 无网也能推理
#
# 目的：改完代码先本地秒级验证，别把每个小改动丢进 6h CI 试错（见 PITFALLS.md 铁律 1）。
#
# 关于「短训练」：本机 CPU + Muon + 129M 参数下，哪怕 2 步也要数分钟且不稳，
# 不适合做秒级冒烟，故不放在这里。真正训练请直接跑 train.py（CI 或联网机器），
# 它能联网加载 Qwen2.5 分词器；沙箱/离线时会自动退到 byte-level 兜底（仅正确性冒烟用）。

set -u
cd "$(dirname "$0")/.." || exit 1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

fail=0
for t in tests/test_offline.py tests/test_export_gguf.py tests/test_generate_offline.py; do
  echo "===== $t ====="
  python3 "$t"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "✗ $t 失败（rc=$rc）"
    fail=1
  fi
  echo
done

if [ $fail -ne 0 ]; then
  echo "本地复现（离线自测）存在失败，请先排查再提交/推送。"
  exit 1
fi
echo "本地复现（离线自测全过）✓  改完代码再进 6h CI。"

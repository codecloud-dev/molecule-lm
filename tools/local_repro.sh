#!/bin/bash
# molecule 本地离线复现（开发者本地用，不联网、不依赖 checkpoint）。
#
# 用法：
#   bash tools/local_repro.sh
#
# 它会跑 tests/test_offline.py，覆盖四个关键正确性点（约几秒）：
#   1. 模型前向：logits 形状正确且数值有限（含 MTP）。
#   2. 字节级兜底分词器：中英文/emoji 往返一致（离线无需下载 Qwen）。
#   3. 空 bin 兜底：小语料也能产出非空 .bin（防首轮 mmap 崩溃）。
#   4. KV 缓存生成：forward_cached 首步与全量前向一致。
#
# 目的：改完代码先本地秒级验证，别把每个小改动丢进 6h CI 试错（见 PITFALLS.md 铁律 1）。
#
# 关于「短训练」：本机 CPU + Muon + 129M 参数下，哪怕 2 步也要数分钟且不稳，
# 不适合做秒级冒烟，故不放在这里。真正训练请直接跑 train.py（CI 或联网机器），
# 它能联网加载 Qwen2.5 分词器；沙箱/离线时会自动退到 byte-level 兜底（仅正确性冒烟用）。

set -u
cd "$(dirname "$0")/.." || exit 1

echo "===== molecule 本地离线自测 tests/test_offline.py ====="
python3 tests/test_offline.py
rc=$?
if [ $rc -ne 0 ]; then
  echo "离线自测失败（rc=$rc），请先排查再提交/推送。"
  exit $rc
fi
echo
echo "本地复现（离线自测）通过 ✓  改完代码再进 6h CI。"

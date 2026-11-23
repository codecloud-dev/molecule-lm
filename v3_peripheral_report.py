#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 外围总览看板（只读；不碰 train_v3.py / checkpoint / clock / 续训）

把三类外围产物合成一张人话看板：
  1. 训练进度：data/v3_clock.json + data/v3_train_log.jsonl
  2. 评测分层：eval_v3_metrics.json（锚点/阶段/两级公开集/污染/生成质量）
  3. 发版闸门：release_gate_report.json（24 项硬门槛 + 联合门槛）
输出 data/v3_peripheral_dashboard.md，供 CI 留档，不影响原有 data/v3_dashboard.md。
"""

import argparse
import json
import os


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def pct(v):
    try:
        return "%.1f%%" % float(v)
    except Exception:
        return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", default="data/v3_clock.json")
    ap.add_argument("--metrics", default="eval_v3_metrics.json")
    ap.add_argument("--gate", default="release_gate_report.json")
    ap.add_argument("--train-log", default="data/v3_train_log.jsonl")
    ap.add_argument("--out", default="data/v3_peripheral_dashboard.md")
    args = ap.parse_args()

    clock = load(args.clock, {}) or {}
    metrics = load(args.metrics, {}) or {}
    gate = load(args.gate, {}) or {}
    mins = float(clock.get("total_minutes", 0) or 0)
    hours = mins / 60.0
    progress = min(100.0, 100.0 * hours / 480.0)
    filled = int(progress / 2.5)
    bar = "#" * filled + "-" * (40 - filled)

    last = {}
    try:
        with open(args.train_log, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
    except Exception:
        pass

    lines = []
    lines.append("# molecule v3 外围总览（训练主链路零改动）")
    lines.append("")
    lines.append("> 开源语言模型·代号 molecule·由 codecloud-dev 独立开发")
    lines.append("")
    lines.append("## 1. 训练进度（20 天窗口）")
    lines.append("")
    lines.append("`[%s] %.1f%%`（%.1f h / 480 h，%s 轮）" % (
        bar, progress, hours, clock.get("runs", 0)))
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 累计 token | %.3f B |" % (float(clock.get("total_tokens", 0) or 0) / 1e9))
    lines.append("| 全局步 | %s |" % last.get("gstep", last.get("step", "—")))
    lines.append("| 最近训练 PPL | %s |" % last.get("ppl", "—"))
    lines.append("| 课程阶段 | %s |" % last.get("phase", "—"))
    lines.append("| 最近吞吐 | %s tok/s |" % last.get("tok_s", "—"))

    lines.append("")
    lines.append("## 2. 评测分层（锚点 / 阶段 / 两级公开集）")
    lines.append("")
    if metrics:
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        for k, label in [
            ("stage", "评测阶段"), ("overall_acc", "总准确率"),
            ("acc_anchor", "固定锚点题"), ("anchor_delta", "锚点回归"),
            ("acc_stage", "当前阶段题"), ("acc_classic_public", "经典公开集"),
            ("acc_clean_public", "去污染公开卷"), ("contamination_suspect_rate", "疑似污染题占比"),
            ("repetition_rate", "重复率"), ("eos_rate", "EOS 收尾率"),
            ("distinct_1", "Distinct-1"), ("ppl", "检查点平滑 PPL"),
        ]:
            v = metrics.get(k, "—")
            lines.append("| %s | %s |" % (label, pct(v) if k not in ("stage", "anchor_delta", "ppl", "repetition_rate", "eos_rate", "distinct_1") else v))
    else:
        lines.append("（本轮无评测产物；非评测轮或评测失败）")

    lines.append("")
    lines.append("## 3. 发版闸门")
    lines.append("")
    if gate:
        status = gate.get("status", "—")
        cn = {"PASS": "通过", "BLOCKED": "拦截", "FORCED": "720h 强制兜底", "NOT_DUE": "未到时间"}.get(status, status)
        lines.append("**%s** ｜ 硬门槛 %s 项 ｜ 未过 %s 项" % (
            cn, gate.get("n_gates", "—"), gate.get("n_failed", "—")))
        if gate.get("failed_keys"):
            lines.append("")
            lines.append("未过项：`%s`" % "`, `".join(gate["failed_keys"]))
    else:
        lines.append("（本轮无闸门报告）")

    lines.append("")
    lines.append("## 4. 自动告警（大白话）")
    lines.append("")
    alerts = []
    if gate.get("status") == "BLOCKED":
        alerts.append("发版闸门拦截：质量未达标，当前不会发 v3.0，训练继续。")
    rep = metrics.get("repetition_rate")
    if rep is not None and float(rep) > 0.18:
        alerts.append("重复率 %.3f 偏高：有复读/模式坍缩嫌疑，低 PPL 也不能信。" % float(rep))
    eos = metrics.get("eos_rate")
    if eos is not None and float(eos) < 0.30:
        alerts.append("EOS 收尾率 %s：模型经常不会正常结束回答。" % pct(eos))
    delta = metrics.get("anchor_delta")
    if delta is not None and float(delta) < -10:
        alerts.append("锚点题相对上次下降 %s 个百分点：疑似退化。" % delta)
    classic, clean = metrics.get("acc_classic_public"), metrics.get("acc_clean_public")
    if classic is not None and clean is not None and float(classic) - float(clean) > 20:
        alerts.append("经典公开集比去污染卷高 %.1f 个百分点：公开集可能被训练语料污染。" % (
            float(classic) - float(clean)))
    if not alerts:
        alerts.append("（无告警）")
    for a in alerts:
        lines.append("- %s" % a)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("外围总览已存 %s" % args.out)


if __name__ == "__main__":
    main()

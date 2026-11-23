#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""molecule 离线 MetaRSI 训练调度器（外围，只读建议；训练主链路零改动）

修复记录：
  1. 历史不再只存在 config_v3_rsi.json（跨 CI runner 会丢）。改为优先从 data/v3_train_log.jsonl
     读取跨轮累积轨迹（该文件由 persist_v3.py 随 checkpoint 滚动发布，天然跨 runner 保留）。
  2. 最优 checkpoint 不再拿“本轮唯一一条记录”当最优。每次从完整历史里按【平滑 loss】取全局最优。
  3. 平滑口径修正：CI 传 --ppl（checkpoint 里 exp(run_lm) 的平滑 PPL），这里用 ln(ppl) 作 loss；
     不再拿瞬时 loss 误判发散或最优。保留 --loss 仅作向后兼容。
  4. 状态文件只存 lr / phase / best 摘要，history 永远以 train_log 为准，避免重复记账。

用法：
  python metarsi.py update --step 120 --ppl 9.42
  python metarsi.py suggest
  python metarsi.py best
  python metarsi.py reset

阶段阈值：v3.0 目标 PPL<10；本脚本只给建议，不直接改训练配置。
"""

import argparse
import json
import math
import os

STATE = "config_v3_rsi.json"
HISTORY = "data/v3_train_log.jsonl"
RSI_HISTORY = "data/v3_rsi_history.jsonl"
TARGET_PPL_V30 = 10.0
PLATEAU_WINDOW = 8
PLATEAU_EPS = 0.01
LR_DECAY = 0.5


def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE, encoding="utf-8") as f:
                s = json.load(f)
            return {"phase": s.get("phase", "Phase1"), "lr": float(s.get("lr", 3e-4)),
                    "best": s.get("best")}
        except Exception:
            pass
    return {"phase": "Phase1", "lr": 3e-4, "best": None}


def save_state(s):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump({"phase": s["phase"], "lr": s["lr"], "best": s.get("best")},
                  f, ensure_ascii=False, indent=2)


def _rows_from(path):
    rows = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                step = r.get("gstep", r.get("step"))
                lm = r.get("lm") if r.get("lm") is not None else r.get("loss")
                if lm is None and r.get("ppl"):
                    lm = math.log(float(r["ppl"]))
                if step is None or lm is None:
                    continue
                rows.append({"step": int(step), "loss": float(lm),
                             "ppl": round(math.exp(min(float(lm), 20)), 3),
                             "lr": r.get("lr", "—")})
    except Exception:
        pass
    return rows


def read_history():
    """训练日志(跨轮主源) + RSI 补记(本轮新点) 合并，按 step 去重；失败回退旧 state。"""
    rows = _rows_from(HISTORY) + _rows_from(RSI_HISTORY)
    if not rows:
        try:
            s = json.load(open(STATE, encoding="utf-8"))
            for r in s.get("history", []):
                rows.append({"step": int(r["step"]), "loss": float(r["loss"]),
                             "ppl": round(math.exp(min(float(r["loss"]), 20)), 3),
                             "lr": r.get("lr", "—")})
        except Exception:
            pass
    by_step = {}
    for r in rows:
        by_step[r["step"]] = r       # 同 step 保留最后一条（RSI 补记覆盖训练日志）
    return [by_step[k] for k in sorted(by_step)]


def write_rsi_history(rows):
    os.makedirs(os.path.dirname(os.path.abspath(RSI_HISTORY)), exist_ok=True)
    with open(RSI_HISTORY, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def upsert(history, step, loss):
    row = {"step": int(step), "loss": float(loss),
           "ppl": round(math.exp(min(float(loss), 20)), 3), "lr": "—"}
    by_step = {r["step"]: r for r in history}
    by_step[row["step"]] = row
    return [by_step[k] for k in sorted(by_step)]


def best_of(history):
    if not history:
        return None
    b = min(history, key=lambda r: r["loss"])
    return {"step": b["step"], "loss": b["loss"], "ppl": b["ppl"]}


def cmd_update(step, loss, ppl=None):
    if ppl is not None and math.isfinite(ppl) and ppl > 0:
        loss = math.log(ppl)         # checkpoint 的 ppl = exp(run_lm)，取回平滑 loss
    h = read_history()
    h = upsert(h, step, loss)
    write_rsi_history(h)
    s = load_state()
    s["best"] = best_of(h)
    save_state(s)
    print("[rsi] recorded step=%d smooth_loss=%.4f ppl=%.3f" % (step, loss, math.exp(min(loss, 20))))
    print("      全局最优（跨轮历史 %d 点）: step=%d smooth_loss=%.4f" % (
        len(h), s["best"]["step"], s["best"]["loss"]))


def cmd_suggest():
    s = load_state()
    h = read_history()
    if len(h) < 2:
        print("[rsi] 历史数据不足（%d 点），先积累几步再建议。" % len(h))
        return
    last = h[-1]
    ppl = last["ppl"]
    best = best_of(h)
    s["best"] = best

    recent = h[-PLATEAU_WINDOW:]
    improved = (recent[0]["loss"] - recent[-1]["loss"]) / max(abs(recent[0]["loss"]), 1e-6)
    is_plateau = improved < PLATEAU_EPS and len(h) >= PLATEAU_WINDOW

    if ppl <= TARGET_PPL_V30:
        rec, lines = "freeze", ["v3.0 已达 PPL<10 目标，建议冻结并按计划导出/发版。"]
    elif is_plateau:
        s["lr"] = max(s["lr"] * LR_DECAY, 5e-5)
        rec, lines = "decay-lr", [
            "近 %d 个记录点改进 %.4f < 阈值 %.3f，疑似平台。" % (len(recent), improved, PLATEAU_EPS),
            "建议学习率衰减至 %.2e 后继续；若再平台一次则考虑推进阶段/换数据配比。" % s["lr"]]
    else:
        rec, lines = "continue", ["平滑 loss 仍在下降（近 %d 点改进 %.4f），保持当前节奏继续。" % (
            len(recent), improved)]
    save_state(s)

    print("==================== MetaRSI 调度建议 ====================")
    print("阶段: %s | 当前 step=%d smooth_loss=%.4f ppl=%.3f | lr=%.2e"
          % (s["phase"], last["step"], last["loss"], ppl, s["lr"]))
    for ln in lines:
        print("  · " + ln)
    print("  >> 建议动作: %s" % rec)
    print("  >> 全局最优 checkpoint: step=%d smooth_loss=%.4f ppl=%.3f"
          % (best["step"], best["loss"], best["ppl"]))

    with open("molecule_rsi_status.md", "w", encoding="utf-8") as f:
        f.write("# molecule 训练调度状态（MetaRSI 离线 · 跨轮平滑口径）\n\n")
        f.write("- 阶段: %s\n- 当前: step=%d, smooth_loss=%.4f, ppl=%.3f, lr=%.2e\n"
                % (s["phase"], last["step"], last["loss"], ppl, s["lr"]))
        f.write("- 全局最优 ckpt: step=%d, smooth_loss=%.4f, ppl=%.3f\n\n"
                % (best["step"], best["loss"], best["ppl"]))
        f.write("## 最近 %d 个记录点\n\n" % len(recent))
        for r in recent:
            f.write("- step=%d smooth_loss=%.4f ppl=%.3f\n" % (r["step"], r["loss"], r["ppl"]))
        f.write("\n## 建议\n\n- 动作: **%s**\n" % rec)
        for ln in lines:
            f.write("- %s\n" % ln)
    print("\n[rsi] 状态已写入 molecule_rsi_status.md")


def cmd_best():
    b = best_of(read_history()) or load_state().get("best")
    if b:
        print("[rsi] 全局最优: step=%d smooth_loss=%.4f ppl=%.3f" % (b["step"], b["loss"], b["ppl"]))
    else:
        print("[rsi] 暂无记录")


def cmd_reset():
    for p in (STATE, "molecule_rsi_status.md"):
        if os.path.exists(p):
            os.remove(p)
    print("[rsi] 状态已清空（训练日志历史保留）")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    u = sub.add_parser("update")
    u.add_argument("--step", type=int, required=True)
    u.add_argument("--loss", type=float, default=None, help="向后兼容；瞬时 loss，不推荐")
    u.add_argument("--ppl", type=float, default=None, help="推荐：checkpoint 的平滑 PPL")
    sub.add_parser("suggest")
    sub.add_parser("best")
    sub.add_parser("reset")
    args = ap.parse_args()
    if args.cmd == "update":
        if args.loss is None and args.ppl is None:
            u.error("需要 --ppl（推荐）或 --loss")
        cmd_update(args.step, args.loss or 0.0, args.ppl)
    elif args.cmd == "suggest":
        cmd_suggest()
    elif args.cmd == "best":
        cmd_best()
    elif args.cmd == "reset":
        cmd_reset()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

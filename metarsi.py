#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""molecule 离线 MetaRSI 训练调度器（设计文档 §7.4：纯赚零赔，零用户侧影响）。

只在你自己的训练环境跑，不接触推理引擎、不碰发出去的模型。干的事：
  - 看 loss/PPL 走势，建议是否衰减学习率 / 推进阶段 / 判定达标冻结
  - 记录并自动挑选最优 checkpoint（按 PPL 最低）
  - 输出 molecule_rsi_status.md 给你（人类）过目决策，不直接改训练主流程

用法：
  python metarsi.py update --step 120 --loss 7.42        # 记录一步
  python metarsi.py suggest                                # 给出调度建议
  python metarsi.py best                                     # 列出最优 ckpt
  python metarsi.py reset                                    # 清空状态

阶段阈值（与 §10/§12 对齐）：
  v3.0 目标 PPL<10 即达标冻结；v3.5 起 Phase2(PPL<6)… 这里只管 v3.0 区间，超阈值自动提示进入下一阶段规划。
"""

import argparse
import json
import os

STATE = "config_v3_rsi.json"
TARGET_PPL_V30 = 10.0
PPL_OF_LOSS = lambda l: 2.718281828 ** l
PLATEAU_WINDOW = 8          # 连续 N 步改进 < eps 视为平台
PLATEAU_EPS = 0.01
LR_DECAY = 0.5


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"phase": "Phase1", "history": [], "best": None, "lr": 3e-4}


def save_state(s):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def cmd_update(step, loss):
    s = load_state()
    s["history"].append({"step": step, "loss": loss,
                         "ppl": round(PPL_OF_LOSS(loss), 3), "lr": s["lr"]})
    if s["best"] is None or loss < s["best"]["loss"]:
        s["best"] = {"step": step, "loss": loss, "ppl": round(PPL_OF_LOSS(loss), 3)}
    save_state(s)
    print("[rsi] recorded step=%d loss=%.4f ppl=%.3f" % (step, loss, PPL_OF_LOSS(loss)))
    print("      当前最优: step=%d loss=%.4f" % (s["best"]["step"], s["best"]["loss"]))


def cmd_suggest():
    s = load_state()
    h = s["history"]
    if len(h) < 2:
        print("[rsi] 数据不足，先积累几步再建议。")
        return
    last = h[-1]
    ppl = PPL_OF_LOSS(last["loss"])

    # 平台检测
    recent = h[-PLATEAU_WINDOW:]
    improved = (recent[0]["loss"] - recent[-1]["loss"]) / max(abs(recent[0]["loss"]), 1e-6)
    is_plateau = improved < PLATEAU_EPS and len(h) >= PLATEAU_WINDOW

    lines = []
    rec = "continue"
    if ppl <= TARGET_PPL_V30:
        rec, lines = "freeze", ["v3.0 已达 PPL<10 目标，建议冻结并按计划导出/发版。"]
    elif is_plateau:
        s["lr"] = max(s["lr"] * LR_DECAY, 5e-5)
        rec, lines = "decay-lr", [
            "近 %d 步改进 %.4f < 阈值 %.3f，疑似平台。" % (PLATEAU_WINDOW, improved, PLATEAU_EPS),
            "建议学习率衰减至 %.2e 后继续；若再平台一次则考虑推进阶段/换数据配比。" % s["lr"]]
    else:
        lines = ["loss 仍在稳定下降（近 %d 步改进 %.4f），保持当前节奏继续。" % (PLATEAU_WINDOW, improved)]

    save_state(s)
    print("==================== MetaRSI 调度建议 ====================")
    print("阶段: %s | 当前 step=%d loss=%.4f ppl=%.3f | lr=%.2e"
          % (s["phase"], last["step"], last["loss"], ppl, last["lr"]))
    for ln in lines:
        print("  · " + ln)
    print("  >> 建议动作: %s" % rec)
    print("  >> 最优 checkpoint: step=%d loss=%.4f ppl=%.3f"
          % (s["best"]["step"], s["best"]["loss"], s["best"]["ppl"]))

    with open("molecule_rsi_status.md", "w", encoding="utf-8") as f:
        f.write("# molecule 训练调度状态（MetaRSI 离线）\n\n")
        f.write("- 阶段: %s\n- 当前: step=%d, loss=%.4f, ppl=%.3f, lr=%.2e\n"
                % (s["phase"], last["step"], last["loss"], ppl, last["lr"]))
        f.write("- 最优 ckpt: step=%d, loss=%.4f, ppl=%.3f\n\n" % (s["best"]["step"], s["best"]["loss"], s["best"]["ppl"]))
        f.write("## 最近 %d 步轨迹\n\n" % PLATEAU_WINDOW)
        for r in recent:
            f.write("- step=%d loss=%.4f ppl=%.3f lr=%.2e\n" % (r["step"], r["loss"], r["ppl"], r["lr"]))
        f.write("\n## 建议\n\n- 动作: **%s**\n" % rec)
        for ln in lines:
            f.write("- %s\n" % ln)
    print("\n[rsi] 状态已写入 molecule_rsi_status.md")


def cmd_best():
    s = load_state()
    if s["best"]:
        print("[rsi] 最优: step=%d loss=%.4f ppl=%.3f" % (s["best"]["step"], s["best"]["loss"], s["best"]["ppl"]))
    else:
        print("[rsi] 暂无记录")


def cmd_reset():
    if os.path.exists(STATE):
        os.remove(STATE)
    print("[rsi] 状态已清空")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    u = sub.add_parser("update"); u.add_argument("--step", type=int, required=True); u.add_argument("--loss", type=float, required=True)
    sub.add_parser("suggest")
    sub.add_parser("best")
    sub.add_parser("reset")
    args = ap.parse_args()
    if args.cmd == "update":
        cmd_update(args.step, args.loss)
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

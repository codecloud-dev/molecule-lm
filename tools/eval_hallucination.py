"""v3.0 幻觉评估集工具（用户口径·方案C：时政语料主力当评估集，不进预训练）。

为什么时政是幻觉高压区：
  人名、职务、会议名称、政策表述——记错一个字就是硬错，而且是用户最常拿来测模型的。
  特别危险的是"半记半猜"：模型从预训练里模模糊糊记了些要素，
  却用最自信的语气编一个错的。这比"不知道"危险得多。

用法：
  python3.11 tools/eval_hallucination.py --model checkpoints/v3.pt
  python3.11 tools/eval_hallucination.py --build-only     # 只重建评估集（不加载模型）

评估集来源：build_corpus_v3 里被【源级硬上限】限流下来的时政原文
（data/v3_eval_hallucination.jsonl）。这些文本没有进入预训练，
因此可以用来检验"该拒绝时能否拒绝、该复述时能否复述准确"。

评测方式（无需外部模型，纯规则 + token 级比对）：
  1) 要素抽取：从原文里抽出【人名 / 职务 / 数字 / 会议名】等关键事实要素。
  2) 事实复述：给模型标题 + 提示，让它复述关键事实。
  3) 核对：模型输出中命中多少要素、编造多少要素（输出里出现但原文没有的要素）。
  4) 报告：逐要素命中率 + 幻觉率（编造率）。
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_PATH = "data/v3_eval_hallucination.jsonl"

# 关键事实要素抽取规则
RE_NAME = re.compile(r"[一-龥]{2,4}(?:同志|书记|主席|总理|部长|主任|局长|省长|市长|"
                     r"教授|博士|将军|委员|总裁|经理|总监|校长|馆长|院长)")
RE_NUM = re.compile(r"\d{2,}(?:\.\d+)?(?:%|万|亿|年|月|日|人|次|项|个)?")
RE_MEET = re.compile(r"[一-龥]{2,12}(?:会议|大会|论坛|峰会|座谈会|发布会|典礼|仪式)")


def extract_facts(text, max_each=6):
    """从时政原文抽取关键事实要素：人名职务 / 数字 / 会议名。"""
    facts = []
    for rx, tag in ((RE_NAME, "person"), (RE_MEET, "meeting"), (RE_NUM, "number")):
        seen = []
        for m in rx.finditer(text):
            v = m.group(0)
            if v not in seen:
                seen.append(v)
            if len(seen) >= max_each:
                break
        for v in seen:
            facts.append((tag, v))
    return facts


def build_eval_set(max_items=120):
    """从落盘的评估集 JSONL 抽取事实要素，生成可执行的评测题目。"""
    items = []
    if not os.path.exists(EVAL_PATH):
        print("[eval] 评估集不存在：%s（先跑 build_corpus_v3.py）" % EVAL_PATH)
        return items
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            facts = extract_facts(r.get("text", ""))
            if len(facts) < 4:
                continue          # 要素太少的事实题没有区分度
            items.append({
                "title": r.get("title", ""),
                "source": r.get("source", ""),
                "facts": facts,
                "text": r.get("text", "")[:1200],
            })
            if len(items) >= max_items:
                break
    return items


def make_prompt(item):
    """构造评测 prompt：给标题，请模型复述关键事实（不给原文，逼它凭记忆/推理作答）。"""
    t = item["title"] or "该新闻"
    return ("请复述以下报道的关键事实要素（人物、职务、会议、数字）：\n标题：%s" % t)


def check_output(out, facts):
    """核对模型输出：命中要素 / 编造要素。"""
    hit = [f for f in facts if f[1] in out]
    # 编造：输出里出现的"职务类"词不在事实列表中
    out_persons = set(RE_NAME.findall(out))
    known = {f[1] for f in facts if f[0] == "person"}
    fab = [p for p in out_persons if p not in known]
    return hit, fab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true", help="只重建评估集，不加载模型")
    ap.add_argument("--model", default="checkpoints/v3.pt", help="模型权重路径")
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--out", default="data/v3_eval_report.json")
    args = ap.parse_args()

    items = build_eval_set(args.max_items)
    if not items:
        print("[eval] 无可用评测样本")
        return
    print("[eval] 评测样本 %d 条（来自 %s）" % (len(items), EVAL_PATH))
    if args.build_only:
        for it in items[:5]:
            print("  [%s] %s" % (it["source"], it["title"][:40]))
            print("     要素:", ", ".join("%s:%s" % f for f in it["facts"][:6]))
        return

    import torch
    from model_v3 import MoleculeV3
    from tokenizer_bpe import ByteBPETokenizer
    import generate_v3 as G

    tk = ByteBPETokenizer.load()
    model = MoleculeV3(vocab_size=8192)
    sd = torch.load(args.model, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()

    total_facts = total_hit = 0
    total_fab = 0
    rows = []
    for it in items:
        out = G.generate(model, tk, make_prompt(it), max_new=args.max_new,
                         temperature=0.7, top_p=0.92)
        hit, fab = check_output(out, it["facts"])
        total_facts += len(it["facts"])
        total_hit += len(hit)
        total_fab += len(fab)
        rows.append({"title": it["title"], "source": it["source"],
                     "hit": len(hit), "facts": len(it["facts"]),
                     "fabricated": fab, "output": out[:400]})
        print("  [%s] %s → 命中 %d/%d，编造 %d"
              % (it["source"], it["title"][:24], len(hit), len(it["facts"]), len(fab)))

    recall = total_hit / max(total_facts, 1)
    fab_rate = total_fab / max(total_hit + total_fab, 1)
    print("\n[eval] 要素召回率 %.1f%%  幻觉(编造)率 %.1f%%  样本 %d"
          % (100 * recall, 100 * fab_rate, len(items)))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"recall": recall, "fabrication_rate": fab_rate,
                   "n_items": len(items), "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print("[eval] 报告已写入 %s" % args.out)


if __name__ == "__main__":
    main()

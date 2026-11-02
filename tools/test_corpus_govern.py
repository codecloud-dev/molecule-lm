"""corpus_govern 本地快速自测（1-2 分钟内跑完，不联网）。

验证四件事：
  1) 跨源近重复能被检出并裁掉（同一篇通稿在三个源各发一遍 → 只留一份）
  2) 精确重复（转载）能被检出
  3) token 口径的占比计算是对的（域权重 × 域内占比）
  4) 单源上限 / 规范书面语上限的裁剪求解器能让上限真正达标
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import corpus_govern as G


def _mk(doc_id, src, dom, regime, body):
    return {"id": doc_id, "src": src, "dom": dom, "regime": regime, "body": body}


def test_dedupe_cross_source():
    print("=" * 70)
    print("[T1] 跨源近重复")
    body = ("新华社北京三月一日电 国务院总理今日主持召开国务院常务会议，"
            "会议指出，要深入贯彻落实新发展理念，坚定不移推动高质量发展。"
            "会议强调，各地区各部门要提高认识，扎实推进相关工作，"
            "不断增强人民群众的获得感幸福感安全感。会议还就当前重点问题作出部署。") * 4
    docs = [
        _mk("a1", "人民网", "zh_general", "official", body),
        _mk("a2", "新华网", "zh_general", "official", body),          # 完全相同的转载
        _mk("a3", "央视网", "zh_general", "official", body[:len(body) - 30]),  # 略短转载
        _mk("b1", "中国新闻网", "zh_general", "news",
            "上海暴雨引发市民关注，气象部门发布预警信号，提醒市民减少外出，"
            "地铁部分线路调整运营方案，公交集团增加接驳车辆运力。"),
        _mk("c1", "第一财经", "zh_general", "news",
            "A股三大指数今日集体收涨，创业板指领涨两市，半导体与新能源板块表现活跃，"
            "多家券商机构认为估值修复行情有望延续，但需警惕外围市场波动风险。"),
    ]
    kept, rep = G.minhash_lsh_dedupe(docs, verbose=True)
    print("  保留 %d 篇，源分布：%s" % (len(kept), sorted(d["src"] for d in kept)))
    assert len(kept) == 3, "应只剩 3 篇（通稿三份合并 + 两篇独立新闻）"
    # 三份通稿合并成一份，代表 = 正文最全者（人民网那份最长）
    assert sum(1 for d in kept if d["src"] in ("人民网", "新华网", "央视网")) == 1, \
        "通稿三份必须合并成一份"
    assert len([d for d in kept if d["src"] == "人民网"]) == 1
    print("  ✓ 跨源转载已合并")


def test_template_profile():
    print("=" * 70)
    print("[T2] 高频模板度量")
    head = "新华社北京三月一日电 国务院常务会议指出"
    texts = [head + "会议强调要" + "推动发展" * 30 for _ in range(12)]
    texts += ["某地举办文化节，吸引了大量游客前来参观，配套服务措施同步完善。" * 4] * 8
    prof = G.template_profile(texts, n=8)
    print("  最高频段首：%s (x%d, 占比 %.0f%%)"
          % (prof["head_top"][0][0], prof["head_top"][0][1], prof["head_top_share"] * 100))
    assert prof["head_top"][0][1] == 12
    assert prof["head_top_share"] > 0.5
    print("  ✓ 模板污染可被度量")


def test_shares_and_caps():
    print("=" * 70)
    print("[T3] token 口径占比 + 配额上限")
    dom_weights = G.effective_domain_weights([
        {"frac": 0.50, "mix": {"zh_general": 0.60, "en_general": 0.10,
                               "code": 0.20, "book": 0.10}},
        {"frac": 0.40, "mix": {"zh_general": 0.30, "code": 0.30,
                               "math_reason": 0.25, "book": 0.10,
                               "encyclopedia": 0.05}},
        {"frac": 0.10, "mix": None},
    ])
    print("  域权重：%s" % {k: round(v, 4) for k, v in dom_weights.items()})
    assert abs(sum(dom_weights.values()) - 1.0) < 1e-9

    # 造一个"公文体严重超标"的池子：zh_general 里大量是官方稿
    docs, tok = [], {}
    for i in range(40):
        did = "off%d" % i
        docs.append(_mk(did, "人民网", "zh_general", "official", "会议指出要抓落实。" * 200))
        tok[did] = 1000
    for i in range(40):
        did = "edu%d" % i
        docs.append(_mk(did, "中文科普", "zh_general", "educational", "光合作用的原理是…" * 200))
        tok[did] = 1000
    for i in range(20):
        did = "code%d" % i
        docs.append(_mk(did, "the-stack", "code", "technical", "def f(): pass" * 200))
        tok[did] = 1000

    sh = G.compute_shares(docs, tok, dom_weights)
    print("  裁剪前：规范书面语训练流占比 %.2f%%"
          % (sh["formal_eff_share"] * 100))
    assert sh["formal_eff_share"] > 0.10, "构造的池子应当超标"

    kept, rep = G.enforce_caps(docs, tok, dom_weights, source_cap=0.005,
                                formal_cap=0.10, verbose=True)
    print("  裁剪后：规范书面语 %.3f%%，单源最大 %.3f%%"
          % (rep["final_formal_eff_share"] * 100,
             max(rep["final_src_eff_share"].values()) * 100))
    assert rep["final_formal_eff_share"] <= 0.10 + 1e-6, "规范书面语上限未达标"
    # 单源 0.5% 在"域权重 43.67% + 域内仅 2 个源"结构下数学不可达：
    # 下限 = w_dom / n_src = 0.4367/2 = 21.8% >> 0.5%
    # 正确行为是【如实报告不可达 + 保留语料】，而不是把池子砍空。
    assert not rep["source_cap_feasible"], "此结构下单源上限本就不该可达"
    assert rep["min_achievable_src_share"] > 0.005
    assert rep["n_after"] > 0, "绝不能把语料砍空"
    print("  ✓ 上限不可达被如实报告（数学下限 %.2f%%），语料未被砍空（%d→%d 篇）"
          % (rep["min_achievable_src_share"] * 100, rep["n_before"], rep["n_after"]))


def test_caps_with_many_sources():
    print("=" * 70)
    print("[T4] 源足够多时，单源上限可以真正达标")
    dom_weights = G.effective_domain_weights([
        {"frac": 1.0, "mix": {"zh_general": 1.0}},
    ])
    docs, tok = [], {}
    # 250 个独立源，每个 1000 token → 每个占 0.4%（<0.5%）→ 上限天然达标
    for i in range(250):
        did = "s%d" % i
        docs.append(_mk(did, "源%03d" % i, "zh_general", "educational",
                        "科普内容 %d：" % i + "这是一段独特的科普文本。" * 100))
        tok[did] = 1000
    kept, rep = G.enforce_caps(docs, tok, dom_weights, source_cap=0.005,
                                formal_cap=0.10, verbose=True)
    print("  可达性：%s，单源最高 %.3f%%"
          % (rep["source_cap_feasible"],
             max(rep["final_src_eff_share"].values()) * 100))
    assert rep["source_cap_feasible"], "100 个独立源时 0.5% 上限应当可达"
    worst = max(v for s, v in rep["final_src_eff_share"].items()
                if s not in G.EXEMPT_SOURCES)
    assert worst <= 0.005 + 1e-9, "单源上限未达标: %.5f" % worst
    print("  ✓ 单源上限真正达标（最差 %.3f%%）" % (worst * 100))


def test_dilution_makes_formal_cap_reachable():
    """【核心机制】域内 formal 占比 f<1 时，删 formal 能让 f 下降 → 上限可达。"""
    print("=" * 70)
    print("[T6] 有稀释剂时，formal 上限可以真正达标")
    dom_weights = G.effective_domain_weights([
        {"frac": 1.0, "mix": {"zh_general": 1.0}},
    ])
    docs, tok = [], {}
    # 50 篇 formal（每篇 1000 token）+ 900 篇非 formal（每篇 1000 token）
    # → 域内 f = 50000/950000 = 5.3% → 天然达标
    for i in range(50):
        did = "f%d" % i
        docs.append(_mk(did, "人民网", "zh_general", "official",
                        "会议指出，要深入贯彻落实新发展理念，扎实推进相关工作。"))
        tok[did] = 1000
    for i in range(900):
        did = "n%d" % i
        docs.append(_mk(did, "科普源%d" % i, "zh_general", "educational",
                        "这是一段独特的科普讲解内容，讲的是自然现象与原理。"))
        tok[did] = 1000
    kept, rep = G.enforce_caps(docs, tok, dom_weights, source_cap=0.005,
                                formal_cap=0.10, verbose=True)
    assert rep["final_formal_eff_share"] <= 0.10 + 1e-6, "formal 上限未达标"
    print("  ✓ 稀释剂存在时 formal 上限达标（%.2f%%）"
          % (rep["final_formal_eff_share"] * 100))


def test_pure_formal_domain_reported_not_destroyed():
    """【核心机制】域内 100% formal 时，删 formal 不改变 f → 必须如实报告"需稀释"，
    且绝不能把语料砍空。"""
    print("=" * 70)
    print("[T7] 纯 formal 域：如实报告不可达，不砍空语料")
    dom_weights = G.effective_domain_weights([
        {"frac": 1.0, "mix": {"zh_general": 1.0}},
    ])
    docs, tok = [], {}
    for i in range(20):
        did = "p%d" % i
        docs.append(_mk(did, "人民网", "zh_general", "official",
                        "会议强调，各地区各部门要提高认识，全面落实相关工作要求。"))
        tok[did] = 1000
    kept, rep = G.enforce_caps(docs, tok, dom_weights, source_cap=0.005,
                                formal_cap=0.10, verbose=True)
    assert rep["n_after"] > 0, "绝不能把语料砍空"
    assert rep["final_formal_eff_share"] > 0.10, "纯 formal 域的 formal 占比应为 100%"
    assert rep["pure_formal_doms"], "应识别出纯 formal 域"
    print("  ✓ 纯 formal 域被识别：%s，语料保留 %d 篇"
          % ([p["dom"] for p in rep["pure_formal_doms"]], rep["n_after"]))


def test_dom_concentration_balance():
    """【域内均衡】单一数据集霸占域内 token 时，削它 → 域内份额下降，
    且 formal 上限不能被顶破。"""
    print("=" * 70)
    print("[T8] 域内集中度均衡")
    dom_weights = G.effective_domain_weights([
        {"frac": 1.0, "mix": {"zh_general": 1.0}},
    ])
    docs, tok = [], {}
    # 一个巨型通用源 + 20 个各有 1 篇的小源
    for i in range(200):
        did = "big%d" % i
        docs.append(_mk(did, "SkyPile", "zh_general", "educational",
                        "这是一段通用的中文网页文本，内容各不相同，讲的是各种话题。"))
        tok[did] = 1000
    for i in range(20):
        did = "sm%d" % i
        docs.append(_mk(did, "源%02d" % i, "zh_general", "educational",
                        "小源 %d 的独立中文内容，讲的是独特的话题与细节。" % i))
        tok[did] = 1000
    sh = G.compute_shares(docs, tok, dom_weights)
    before = sh["grp_eff_share"]["SkyPile"]
    assert before > 0.85, "构造时应为单源霸占"
    kept, rep = G.enforce_caps(docs, tok, dom_weights, source_cap=0.005,
                                formal_cap=0.10, dom_max_group_share=0.60,
                                verbose=True)
    after = rep["final_grp_eff_share"]["SkyPile"]
    print("  域内均衡：SkyPile %.2f%% → %.2f%%" % (before * 100, after * 100))
    assert after < before - 0.1, "域内集中度应明显下降"
    assert rep["final_formal_eff_share"] <= 0.10 + 1e-6, "formal 上限被顶破"
    assert rep["n_after"] > 20, "小源不该被误删"
    print("  ✓ 域内均衡生效，且 formal 上限未被顶破")


def test_wide_cap_encyclopedic():
    """【第 2 条动作】规范书面语 + 百科（宽口径）合计必须有硬上限，
    且 encyclopedia 域（课程指定预算）不可被砍。"""
    print("=" * 70)
    print("[T9] 宽口径（formal + 百科）上限")
    # 【物理可达性】encyclopedia 域的域权重 = 课程指定预算，
    # 它本身全部是 encyclopedic → 宽口径至少 = 该域权重。
    # 真实配比里 encyclopedia 域权重 3.67%，所以这里取 5%，
    # 保证上限 12% 在数学上可达（否则测的是"不可达"而非"裁剪逻辑"）。
    dom_weights = G.effective_domain_weights([
        {"frac": 1.0, "mix": {"zh_general": 0.95, "encyclopedia": 0.05}},
    ])
    docs, tok = [], {}
    # encyclopedia 域：20 篇百科
    for i in range(20):
        did = "e%d" % i
        docs.append(_mk(did, "百科源", "encyclopedia", "encyclopedic",
                        "北京，简称京，是中华人民共和国的首都。"))
        tok[did] = 1000
    # zh_general 域：大量"百科腔"通用网页（被误判成 encyclopedic）
    for i in range(200):
        did = "z%d" % i
        docs.append(_mk(did, "网页源", "zh_general", "encyclopedic",
                        "某某事物，是某领域的一种重要现象，位于江苏省某市。"))
        tok[did] = 1000
    for i in range(100):
        did = "g%d" % i
        docs.append(_mk(did, "通用源", "zh_general", "general",
                        "今天天气不错，街上人来人往，说了些日常琐事。"))
        tok[did] = 1000
    sh = G.compute_shares(docs, tok, dom_weights)
    print("  裁剪前宽口径 %.2f%%" % (sh["formal_wide_eff_share"] * 100))
    assert sh["formal_wide_eff_share"] > 0.12, "构造时应超标"
    kept, rep = G.enforce_caps(docs, tok, dom_weights, source_cap=0.005,
                                formal_cap=0.10, wide_cap=0.12, verbose=True)
    assert rep["final_formal_wide_eff_share"] <= 0.12 + 1e-6, "宽口径上限未达标"
    # encyclopedia 域必须保留
    n_ency = sum(1 for d in kept if d["dom"] == "encyclopedia")
    assert n_ency > 0, "encyclopedia 域（课程指定预算）不该被砍空"
    print("  ✓ 宽口径达标 %.2f%%，encyclopedia 域保留 %d 篇"
          % (rep["final_formal_wide_eff_share"] * 100, n_ency))


def test_report_render():
    print("=" * 70)
    print("[T5] 报告渲染")
    dom_weights = G.effective_domain_weights([
        {"frac": 1.0, "mix": {"zh_general": 1.0}},
    ])
    docs = [_mk("x1", "人民网", "zh_general", "official", "会议指出。")]
    tok = {"x1": 100}
    sh = G.compute_shares(docs, tok, dom_weights)
    txt = G.render_report(sh)
    print(txt)
    assert "训练 token 流" in txt
    print("  ✓ 报告可渲染")


if __name__ == "__main__":
    test_dedupe_cross_source()
    test_template_profile()
    test_shares_and_caps()
    test_caps_with_many_sources()
    test_dilution_makes_formal_cap_reachable()
    test_pure_formal_domain_reported_not_destroyed()
    test_dom_concentration_balance()
    test_wide_cap_encyclopedic()
    test_report_render()
    print("=" * 70)
    print("corpus_govern 自测全部通过")

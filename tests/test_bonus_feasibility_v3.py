# -*- coding: utf-8 -*-
"""V3 定向测试：派奖真实生效、金额精度、四格精确收益与盈亏平衡边界。

对应 `docs/project_notes/PROFIT_V3_BONUS_FEASIBILITY_PLAN.md` 的 E1/E2 验收。

运行： python -m pytest tests/test_bonus_feasibility_v3.py -q
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal
from math import comb

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                                            # noqa: E402
from tools import bonus_feasibility_v3 as bf                        # noqa: E402
from tools import profit_opportunity as po                          # noqa: E402
from tools.five_plus12_math import BACK_HIT_TICKETS                 # noqa: E402

SNAPSHOT_OK = pr.get_snapshot().ok
TARGET = 26117
D = Decimal


def _pair(kind="assumed_pair", first=5_000_000, second=100_000):
    return {"kind": kind, "first": first, "second": second, "source": "test"}


# ---------------------------------------------------------------------------
# E1-1 先证明缺口：派奖必须**真的**改变金额
# ---------------------------------------------------------------------------

def test_bonus_actually_changes_amounts_not_just_a_flag():
    """回归护栏：修复前 bonus_applied=True 但金额与关闭时完全相同。"""
    t_off = bf.rule_cell_table("A")
    t_on = bf.rule_cell_table("C")
    assert t_on[(3, 2)] > t_off[(3, 2)]          # 五等奖确实变高
    assert t_on[(5, 0)] > t_off[(5, 0)]          # 三等奖确实变高
    # 期望回收必须随之变化
    e_off = bf.evaluate_rule_cell("A", _pair())["expected_gross"]
    e_on = bf.evaluate_rule_cell("C", _pair())["expected_gross"]
    assert e_on > e_off


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_profit_opportunity_now_applies_bonus():
    """修复后：estimate_opportunity 打开派奖必须改变 E[gross]。"""
    snap = pr.get_snapshot()
    s = po.make_floating_scenario("t", 5_000_000, 100_000, [25001], "t")
    ctx = {"issue": TARGET, "as_of": "2026-10-13", "n_groups": 1,
           "pool_before": snap.pool_before(TARGET), "bonus_active": True,
           "bonus_certainty": "official", "bonus_eligibility": True}
    r_on = po.estimate_opportunity("2026-10-13", ctx, prize_scenarios=[s], snapshot=snap)
    r_off = po.estimate_opportunity("2026-10-13", dict(ctx, bonus_active=False),
                                    prize_scenarios=[s], snapshot=snap)
    assert r_on["bonus_applied"] is True and r_off["bonus_applied"] is False
    assert r_on["scenarios"][0]["expected_gross"] > r_off["scenarios"][0]["expected_gross"]


# ---------------------------------------------------------------------------
# E1-2 四格目标值（税前回收）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell,expected", [("A", 675), ("B", 875),
                                           ("C", 1125), ("D", 1470)])
def test_front3_payout_by_cell(cell, expected):
    """单组前区恰好中 3 个的税前回收：A=675、B=875、C=1125、D=1470。"""
    t = bf.rule_cell_table(cell)
    assert bf.group_payout(t, 3) == D(expected)


def test_cell_C_keeps_the_half_yuan():
    """C = 225 + 20×22.5 + 45×10；**不得**因 22.5 截断为 22 而写成 1115。"""
    t = bf.rule_cell_table("C")
    assert t[(3, 2)] == D("225")          # 150 + 75
    assert t[(3, 1)] == D("22.5")         # 15 + 7.5
    assert t[(3, 0)] == D("10")           # 5 + 5
    assert bf.group_payout(t, 3) == D(225) + 20 * D("22.5") + 45 * D(10) == D(1125)
    assert bf.group_payout(t, 3) != D(1115)          # 旧（截断）口径


def test_front2_retains_half_yuan():
    """前区中 2 个：六等奖 22.5 + 七等奖 10 → 保留半元。"""
    t = bf.rule_cell_table("C")
    assert t[(2, 2)] == D("22.5")
    assert bf.group_payout(t, 2) == D("22.5") + 20 * D(10) + 45 * D(0) == D("222.5")


def test_official_pernote_floor_is_a_labelled_sensitivity_only():
    """逐注向下取整只作敏感性对照，且必须给出与精确口径的差。"""
    exact = bf.rule_cell_table("C", rounding="exact")
    floor = bf.rule_cell_table("C", rounding="official_pernote_floor")
    assert exact[(3, 1)] == D("22.5") and floor[(3, 1)] == D("22")
    assert bf.group_payout(exact, 3) - bf.group_payout(floor, 3) == D(10)


def test_unknown_rounding_is_rejected():
    with pytest.raises(bf.BonusFeasibilityError):
        bf.rule_cell_table("C", rounding="whatever")


# ---------------------------------------------------------------------------
# E1-3 派奖边界：不误加、只加一次、无资格不加
# ---------------------------------------------------------------------------

def test_floating_levels_never_get_fixed_prize_bonus():
    """一、二等奖**不参与**固定奖派奖：C/D 的一二等奖金额必须与浮动奖输入一致。"""
    pair = _pair(first=7_777_777, second=321_098)
    for cell in ("A", "B", "C", "D"):
        t = bf.rule_cell_table(cell, floating_pair=pair)
        assert t[(5, 2)] == D("7777777")
        assert t[(5, 1)] == D("321098")


def test_bonus_applied_exactly_once():
    """同一金额只加奖一次：再加一次会改变结果。"""
    once = bf.rule_cell_table("C")
    twice = bf.apply_bonus_to_table(once)
    assert twice[(3, 2)] != once[(3, 2)]
    assert twice[(3, 2)] == D("225") * D("1.5")   # 加两次 = 1.5 倍
    assert once[(3, 2)] == D("225")


def test_bonus_off_cells_equal_base_tier_amounts():
    tA, tB = bf.rule_cell_table("A"), bf.rule_cell_table("B")
    for cond, amt in pr.BASIC_AMOUNT["era3"].items():
        for c in pr.LEVEL_CONDITIONS["era3"][cond]:
            assert tA[c] == D(int(amt))
    for cond, amt in pr.ERA3_HIGH_AMOUNT.items():
        for c in pr.LEVEL_CONDITIONS["era3"][cond]:
            assert tB[c] == D(int(amt))


def test_bonus_rates_match_verified_2026_rule():
    """2026 已核实：三至六等奖 +50%，七等奖 +100%。"""
    assert bf.BONUS_RATE_2026 == {
        "三等奖": D("0.5"), "四等奖": D("0.5"), "五等奖": D("0.5"),
        "六等奖": D("0.5"), "七等奖": D("1.0")}
    # 一等奖、二等奖不在其中
    assert "一等奖" not in bf.BONUS_RATE_2026
    assert "二等奖" not in bf.BONUS_RATE_2026


# ---------------------------------------------------------------------------
# E1-4 恒等式与概率
# ---------------------------------------------------------------------------

def test_back_hit_identity_holds_and_is_result_independent():
    """后区 1/20/45 恒等式：66 注的命中分布与开奖结果无关。"""
    assert BACK_HIT_TICKETS == {2: 1, 1: 20, 0: 45}
    assert sum(BACK_HIT_TICKETS.values()) == 66 == comb(12, 2)
    # 组回收 = 三项之和，与"哪一期/哪个开奖结果"无关
    t = bf.rule_cell_table("B")
    for k in range(6):
        manual = (1 * t.get((k, 2), 0) + 20 * t.get((k, 1), 0) + 45 * t.get((k, 0), 0))
        assert bf.group_payout(t, k) == manual


def test_probabilities_sum_to_one_and_match_hypergeometric():
    p = bf.front_hit_probabilities()
    assert set(p) == set(range(6))
    assert sum(p.values()) == D(1)
    for k in range(6):
        exact = D(comb(5, k) * comb(30, 5 - k)) / D(comb(35, 5))
        assert p[k] == exact


def test_group_first_prize_probability_not_divided_by_66():
    """后区全包 ⇒ P(一等奖) = P(前区5中)，**不除以 66**。"""
    p = bf.front_hit_probabilities()
    assert p[5] == D(1) / D(comb(35, 5))
    assert D(1) / p[5] == D(324632)
    # 四格中 k=5 的期望贡献必须按整组算（1 注一等奖 + 20 注二等奖 + 45 注 5+0）
    r = bf.evaluate_rule_cell("C", _pair(first=1_000_000, second=50_000))
    o5 = [o for o in r["outcomes"] if o["front_hit"] == 5][0]
    t = bf.rule_cell_table("C", floating_pair=_pair(first=1_000_000, second=50_000))
    assert o5["gross"] == D(1_000_000) + 20 * D(50_000) + 45 * t[(5, 0)]


# ---------------------------------------------------------------------------
# E2 精确四格
# ---------------------------------------------------------------------------

def test_evaluate_rule_cell_units_and_cost():
    """税后费用单位一致：period_profit = gross − tax − 132。"""
    r = bf.evaluate_rule_cell("D", _pair())
    assert r["period_cost"] == D(132)
    for o in r["outcomes"]:
        assert o["period_profit"] == o["gross"] - o["tax"] - D(132)


def test_evaluate_rule_cell_extra_costs_are_added():
    r = bf.evaluate_rule_cell("A", _pair(), costs={"execution_fee": 8})
    assert r["period_cost"] == D(140)
    assert r["outcomes"][0]["period_profit"] == r["outcomes"][0]["gross"] - D(0) - D(140)


def test_tax_segments_are_enumerated_not_assumed_monotone():
    """税档分段必须逐档给出，且不假定全局单调。"""
    be = bf.solve_break_even_region("C")
    segs = {s["front_hit"]: s for s in be["tax_segments"]}
    assert set(segs) == set(range(6))
    # 本规则下 k=0..3 不课税、k=4 课税、k=5 恒课税
    assert all(segs[k]["taxed"] is False for k in range(4))
    assert segs[4]["taxed"] is True
    assert segs[5]["taxed"] is True


def test_tax_is_on_full_amount_above_threshold():
    assert bf.apply_tax(D(10_000)) == D(0)
    assert bf.apply_tax(D(10_001)) == D(10_001) * D("0.20")
    assert bf.apply_tax(D(10_001)) != (D(10_001) - D(10_000)) * D("0.20")


# ---------------------------------------------------------------------------
# E2 盈亏平衡边界
# ---------------------------------------------------------------------------

def test_break_even_boundary_is_a_line_in_a_plus_20b():
    """E[net] 对 (a + 20b) 严格线性 ⇒ 边界是一条直线。"""
    be = bf.solve_break_even_region("A")
    K = be["break_even_line"]["a_plus_20b"]
    assert K is not None and K > 0
    # 在边界上取一点，E[net] 必须为 0
    a = K / 2
    b = (K - a) / 20
    r = bf.evaluate_rule_cell("A", _pair(first=a, second=b))
    assert abs(r["expected_net_profit"]) < D("1e-9")


def test_break_even_region_reports_legal_domain_and_reachability():
    be = bf.solve_break_even_region("A")
    ld = be["legal_domain"]
    assert ld["first_cap"] == D(10_000_000) and ld["second_cap"] == D(5_000_000)
    assert isinstance(be["reachable_within_caps"], bool)
    assert "能否同时实现未核实" in be["envelope_at_caps"]["caveat"] or \
           "未核实" in be["envelope_at_caps"]["caveat"]


def test_cells_ordering_and_bonus_effect():
    """派奖与上浮档都提高回收；D 最高，A 最低。"""
    vals = {c: bf.evaluate_rule_cell(c, _pair())["payout_at_front3"]
            for c in "ABCD"}
    assert vals["A"] < vals["B"] < vals["D"]
    assert vals["A"] < vals["C"] < vals["D"]
    assert vals["D"] - vals["A"] == D(1470) - D(675)


# ---------------------------------------------------------------------------
# E2 浮动奖输入必须带标签；配对保持；上限标记
# ---------------------------------------------------------------------------

def test_unlabelled_floating_fallback_is_rejected():
    with pytest.raises(bf.BonusFeasibilityError):
        bf.evaluate_rule_cell("A", {"first": 5_000_000, "second": 100_000})
    with pytest.raises(bf.BonusFeasibilityError):
        bf.validate_floating_pair({"kind": "fallback", "first": 1, "second": 1})
    with pytest.raises(bf.BonusFeasibilityError):
        bf.rule_cell_table("A", floating_pair={"kind": "assumed_pair", "first": 1})


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_pairs_from_history_preserve_same_issue_joint_relation():
    """配对应来自**同一期**，不是两个独立分位数拼起来。"""
    snap = pr.get_snapshot()
    pairs = bf.pairs_from_history(snap, TARGET, window=500)
    assert pairs
    d = snap.draws.set_index("issue")
    for name, p in pairs.items():
        iss = p["source_issue"]
        assert p["first"] == D(str(d.loc[iss, "amt_一等奖"]))
        assert p["second"] == D(str(d.loc[iss, "amt_二等奖"]))
        assert p["kind"] == "historical_pair"


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_typical_pairs_label_edge_quantiles_as_not_joint():
    snap = pr.get_snapshot()
    tp = bf.typical_pairs(snap, TARGET, window=500)
    assert "不是同一期" in tp["max_pair"]["note"] or "不是一次实际开奖" in tp["max_pair"]["note"]
    assert all(v["kind"] == "historical_pair" for v in tp.values())


def test_rule_envelope_is_labelled_and_caveated():
    env = bf.rule_envelope_pair()
    assert env["kind"] == "rule_envelope"
    assert env["first"] == D(10_000_000) and env["second"] == D(5_000_000)
    assert "未核实" in env["note"]


def test_floating_above_cap_is_flagged():
    fp = bf.validate_floating_pair(
        {"kind": "assumed_pair", "first": 20_000_000, "second": 6_000_000})
    assert fp.get("exceeds_first_cap") is True
    assert fp.get("exceeds_second_cap") is True


def test_rule_evidence_is_recorded():
    r = bf.evaluate_rule_cell("C", _pair(), rule_evidence={"doc": "2026七奖级规则",
                                                           "bonus": "26050-26066"})
    assert r["rule_evidence"]["bonus"] == "26050-26066"
    assert r["labels"]["is_rule_scenario"] is True
    assert "26066" in r["labels"]["note"]


def test_unknown_cell_is_rejected():
    with pytest.raises(bf.BonusFeasibilityError):
        bf.evaluate_rule_cell("Z", _pair())
    with pytest.raises(bf.BonusFeasibilityError):
        bf.rule_cell_table("Z")


# ---------------------------------------------------------------------------
# E1-5 官方"7 元"是展示取整 —— 用总额/注数反推真实支付额
# ---------------------------------------------------------------------------

def test_official_stake_amount_7_is_a_display_rounding():
    """官方 stakeAmount=7 只是展示取整；总额/注数显示真实支付为 7.5。"""
    import pandas as pd
    p = os.path.join(ROOT, "outputs", "2026-10-13_5plus12_profitability_verification",
                     "data", "official_dlt_prize_levels.csv")
    if not os.path.exists(p):
        pytest.skip("缺少官方逐期奖级明细")
    lv = pd.read_csv(p)
    s = lv[lv.level == "六等奖派奖"].copy()
    s["implied"] = s.total_amount / s.stake_count
    base = s[s.stake_amount == 7.0]
    assert len(base) > 0
    # 展示值是 7，但真实支付额是 7.5
    assert all(abs(x - 7.5) < 1e-4 for x in base["implied"])
    assert not any(abs(x - 7.0) < 1e-6 for x in base["implied"])
    # 上浮档 18×50% = 9，恰为整数
    high = s[s.stake_amount == 9.0]
    assert all(abs(x - 9.0) < 1e-9 for x in high["implied"])
    # 因此规则精确值 7.5 与官方实际支付一致，C 格 1125 成立
    assert D("15") * D("0.5") == D("7.5")


# ---------------------------------------------------------------------------
# E1-6 官方接口的半元金额必须保留（截断缺陷的回归护栏）
# ---------------------------------------------------------------------------

def test_fetch_tool_num_preserves_half_yuan():
    """官方接口以字符串公布 '7.50'；解析器**不得**截断成 7。"""
    from tools.fetch_official_dlt import _num
    assert _num("7.50") == 7.5
    assert _num("4,521,202.50") == 4521202.5
    assert _num("2.50") == 2.5
    assert _num("10,000,000") == 10_000_000
    assert _num("---") is None and _num("") is None
    # 护栏：旧实现 int(float(s)) 的行为
    assert _num("7.50") != 7


def test_corrected_snapshot_keeps_official_half_yuan_bonus():
    """修正后的快照必须保留官方公布的 7.50 / 2.50。"""
    import pandas as pd
    p = os.path.join(ROOT, "outputs", "2026-10-13_5plus12_v3_bonus",
                     "data", "official_dlt_prize_levels.csv")
    if not os.path.exists(p):
        pytest.skip("缺少修正后的快照")
    lv = pd.read_csv(p)
    s = lv[(lv.issue == 26050) & (lv.level == "六等奖派奖")]
    assert len(s) == 1
    assert float(s.iloc[0].stake_amount) == 7.5
    assert float(s.iloc[0].total_amount) == 4521202.5
    # 历史活动期的八/九等奖派奖同样是半元
    for iss, level, val in ((24030, "八等奖派奖", 7.5), (21039, "九等奖派奖", 2.5)):
        r = lv[(lv.issue == iss) & (lv.level == level)]
        if len(r):
            assert float(r.iloc[0].stake_amount) == val


def test_cell_C_matches_official_actual_payment_not_truncated_field():
    """C 格 R(3)=1125 与官方**实际支付**一致；1115 只是展示字段被截断的产物。"""
    import pandas as pd
    p = os.path.join(ROOT, "outputs", "2026-10-13_5plus12_v3_bonus",
                     "data", "official_dlt_prize_levels.csv")
    if not os.path.exists(p):
        pytest.skip("缺少修正后的快照")
    lv = pd.read_csv(p)
    b = lv[(lv.issue == 26050) & (lv.level == "六等奖派奖")].iloc[0]
    assert float(b.stake_amount) / 15.0 == pytest.approx(0.5)     # 规则 50%
    # 用官方 value 重建 26050 的 R(3)：五等奖 150+75、六等奖 15+7.5、七等奖 5+5
    r3 = (150 + 75) + 20 * (15 + float(b.stake_amount)) + 45 * (5 + 5)
    assert r3 == 1125.0
    assert r3 != 1115.0

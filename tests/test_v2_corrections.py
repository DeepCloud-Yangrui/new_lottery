# -*- coding: utf-8 -*-
"""V2 定向回归测试：针对本轮修正的每一个错误。

运行： python -m pytest tests/test_v2_corrections.py -q

覆盖：
  E1 完整分布 sd 必须含稀有大奖（撤回"样本 sd 推功效"）
  E2 bootstrap 支持集有限，不可能生成未见结果
  E3 高奖池反事实必须改变真实 (前区,后区) 奖金，而不是塞字符串键
  E4 as_of 必须拒绝决策之后才公开的情景
  E5 as_of 必须拒绝来源晚于决策时点的奖池
  E6 预算模拟的三个指标必须分开，且超限恒为 0
  E7 预算模拟默认不允许动态减仓
  E8 奖级分解必须与机会估计使用同一浮动奖情景
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                                     # noqa: E402
from tools import profit_opportunity as po                   # noqa: E402
from tools import sim_budget_v2 as sb                        # noqa: E402
from tools.five_plus12_math import (front_hit_distribution,   # noqa: E402
                                    group_payout_by_front_hit, no_jackpot_table)
from tools.profit_accounting import TAX_2024_MERGED          # noqa: E402
from tools.analysis_power_v2 import exact_period_distribution  # noqa: E402
from tools.build_opportunity_map import (_apply_high_tier,    # noqa: E402
                                         _assert_high_tier_differs)

SNAPSHOT_OK = pr.get_snapshot().ok
TARGET = 26117


# ---------------------------------------------------------------------------
# E1 完整分布 vs 样本 sd
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_full_distribution_sd_is_dominated_by_jackpot():
    """完整单期 sd 必须由大奖支配，且远大于"去大奖"口径的 sd。"""
    snap = pr.get_snapshot()
    base = pr.base_table(TARGET, pool_before=snap.pool_before(TARGET))
    d_full = exact_period_distribution(base, n_groups=1)
    d_nj = exact_period_distribution(no_jackpot_table(base), n_groups=1)

    def sd(dist):
        mu = sum(k * v for k, v in dist.items())
        return (sum(v * (k - mu) ** 2 for k, v in dist.items())) ** 0.5

    assert sd(d_full) > 5000          # 量级：万元级，不是百元级
    assert sd(d_full) / sd(d_nj) > 20  # 大奖项贡献了绝大部分方差
    # 旧口径（样本 sd=106.4）必须显著低于完整分布 sd
    assert sd(d_full) / 106.4 > 50


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_periods_needed_for_breakeven_effect_is_far_beyond_available():
    """检出盈亏平衡量级（66 元/期）所需期数远超现有窗口。"""
    snap = pr.get_snapshot()
    base = pr.base_table(TARGET, pool_before=snap.pool_before(TARGET))
    dist = exact_period_distribution(base, n_groups=1)
    mu = sum(k * v for k, v in dist.items())
    sd = (sum(v * (k - mu) ** 2 for k, v in dist.items())) ** 0.5
    n_needed = ((1.6449 + 0.8416) ** 2) * sd ** 2 / 66.0 ** 2
    assert n_needed > 100_000          # 远超 500 期
    assert n_needed > 100 * 500


# ---------------------------------------------------------------------------
# E2 bootstrap 支持集
# ---------------------------------------------------------------------------

def test_bootstrap_cannot_generate_unseen_outcomes():
    """从"无大奖"样本重采样，永远不可能生成大奖。"""
    sample = np.array([-132.0, 5.0, -17.0, 675.0, -132.0])   # 无大奖
    rng = np.random.default_rng(0)
    draws = rng.choice(sample, size=(50_000, len(sample)), replace=True)
    totals = draws.sum(axis=1)
    # 上界 = 每一期都抽到样本最大值（重抽样是独立有放回的）
    assert totals.max() == len(sample) * sample.max()
    assert totals.max() < 10_000               # 不可能出现万元级结果
    assert set(np.unique(draws)).issubset(set(sample))


# ---------------------------------------------------------------------------
# E3 高奖池反事实的键类型
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_high_tier_counterfactual_changes_real_conditions():
    """上浮档必须改写 (前区命中,后区命中) 的金额。"""
    snap = pr.get_snapshot()
    base = pr.base_table(TARGET, pool_before=0)          # 强制基础档
    high = _apply_high_tier(base, TARGET)
    changed = _assert_high_tier_differs(base, high, TARGET)
    assert changed, "上浮档没有改变任何 (前区,后区) 的奖金"
    # 字典键必须全部是 (int,int) 元组
    assert all(isinstance(k, tuple) and len(k) == 2 for k in high)
    # 三等奖 = 5+0 或 4+2 -> 应变为 6666
    assert high[(5, 0)] == pr.ERA3_HIGH_AMOUNT["三等奖"] == 6666
    assert high[(4, 2)] == 6666
    assert base[(5, 0)] == pr.BASIC_AMOUNT["era3"]["三等奖"] == 5000
    # 且必须真的改变单组回收
    assert group_payout_by_front_hit(high)[3] != group_payout_by_front_hit(base)[3]


def test_high_tier_update_by_level_name_would_be_a_bug():
    """回归护栏：直接 update(ERA3_HIGH_AMOUNT) 会引入字符串键。"""
    tbl = {(3, 2): 150}
    bad = dict(tbl)
    bad.update(pr.ERA3_HIGH_AMOUNT)
    assert any(isinstance(k, str) for k in bad)          # 这就是那个 bug
    good = _apply_high_tier(tbl, TARGET)
    assert all(isinstance(k, tuple) for k in good)


# ---------------------------------------------------------------------------
# E4/E5 as_of 校验
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_as_of_rejects_scenario_published_after_decision():
    snap = pr.get_snapshot()
    # 用一个远早于今天的 as_of，但情景来自更晚的期次
    late = po.make_floating_scenario("late", 5_000_000, 100_000,
                                     [26000, 26010], "t")
    with pytest.raises(po.OpportunityInputError) as e:
        po.estimate_opportunity(
            "2024-01-01",
            {"issue": TARGET, "as_of": "2024-01-01", "n_groups": 1},
            prize_scenarios=[late], snapshot=snap)
    assert "未来信息" in str(e.value) or "晚于" in str(e.value)


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_as_of_rejects_pool_from_future_issue():
    snap = pr.get_snapshot()
    ok = po.make_floating_scenario("ok", 5_000_000, 100_000, [25000], "t")
    with pytest.raises(po.OpportunityInputError):
        po.estimate_opportunity(
            "2024-01-01",
            {"issue": TARGET, "as_of": "2024-01-01", "n_groups": 1,
             "pool_before": 1.0, "pool_before_source_issue": 26116},
            prize_scenarios=[ok], snapshot=snap)


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_as_of_accepts_legitimate_inputs_and_returns_timeline():
    snap = pr.get_snapshot()
    s = po.scenario_from_history(snap, TARGET, mask="all", q=0.5, window=200)
    r = po.estimate_opportunity(
        "2026-10-13",
        {"issue": TARGET, "as_of": "2026-10-13", "n_groups": 1,
         "pool_before": snap.pool_before(TARGET),
         "pool_before_source_issue": 26116},
        prize_scenarios=[s], snapshot=snap)
    tl = r["as_of_timeline"]
    assert tl["as_of"] == "2026-10-13"
    assert tl["pool_before"]["source_issue"] == 26116
    assert all(x["published_at"] <= tl["as_of"] for x in tl["scenarios"])


def test_unparseable_as_of_is_rejected():
    with pytest.raises(po.OpportunityInputError):
        po.estimate_opportunity("not-a-date",
                                {"issue": TARGET, "as_of": "not-a-date"},
                                prize_scenarios=[po.make_floating_scenario(
                                    "s", 1, 1, [25000], "t", published_at="2023-01-01")])


# ---------------------------------------------------------------------------
# E6/E7 预算模拟
# ---------------------------------------------------------------------------

def _rows(costs_profits):
    return [{"issue": 26000 + i, "n_groups": 1, "cost": c, "profit": p}
            for i, (c, p) in enumerate(costs_profits)]


def test_three_indicators_are_separate_and_over_limit_is_zero():
    rows = _rows([(132, -132)] * 10)          # 每期全亏
    r = sb.simulate_policy(rows, loss_limit=500)
    # 396 之后只剩 104 < 132 -> 第 4 期起因额度不足停止
    assert r["bet_periods"] == 3
    assert r["budget_stop"]["stopped"] is True
    assert r["budget_stop"]["stopped_at_issue"] == 26003
    assert r["final_realized_net"] == -396
    # 396 < 500 -> 没有"触及上限"
    assert r["touched_limit"] is False
    # 设计上超限恒为 0
    assert r["over_limit_violations"] == 0


def test_touched_limit_can_be_true_without_violation():
    # 132+132+236 = 500 -> 恰好触及限额，但不超过
    rows = _rows([(132, -132), (132, -132), (236, -236)])
    r = sb.simulate_policy(rows, loss_limit=500)
    assert r["touched_limit"] is True          # 累计正好 500
    assert r["over_limit_violations"] == 0     # 但没有超过
    assert r["final_realized_net"] == -500


def test_over_limit_never_happens_under_check_first_rule():
    """随机压力测试：先检查后参与 -> 超限必须恒为 0。"""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 40))
        rows = [{"issue": 26000 + i, "n_groups": 1, "cost": 132,
                 "profit": int(rng.integers(-132, 4000))} for i in range(n)]
        r = sb.simulate_policy(rows, loss_limit=int(rng.choice([500, 1000])))
        assert r["over_limit_violations"] == 0
        assert -r["final_realized_net"] <= 1000


def test_budget_sim_does_not_reduce_groups_by_default():
    rows = _rows([(396, -396), (396, -396)])
    r = sb.simulate_policy(rows, loss_limit=500)     # 396 ok, 再 396 超
    assert r["bet_periods"] == 1
    assert r["budget_stop"]["stopped_at_issue"] == 26001
    assert r["reduced_groups_used"] is False
    # 显式开启减仓（违反预登记规则）才允许继续
    r2 = sb.simulate_policy(rows, loss_limit=500, allow_reduce_groups=True)
    assert r2["reduced_groups_used"] is True


def test_stopped_policy_has_zero_flow_after_stop():
    rows = _rows([(132, -132)] * 6)
    r = sb.simulate_policy(rows, loss_limit=500)
    tail = [p for p in r["path"] if p["issue"] > r["budget_stop"]["stopped_at_issue"]]
    assert tail and all(p["cost"] == 0 and p["profit"] == 0 for p in tail)


# ---------------------------------------------------------------------------
# E8 奖级分解与机会估计同源
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_level_decomposition_matches_single_group_expectation():
    """同情景下，奖级分解之和必须等于机会估计的期望回收。"""
    snap = pr.get_snapshot()
    s = po.scenario_from_history(snap, TARGET, mask="all", q=0.50, window=200)
    tbl = pr.base_table(TARGET, pool_before=snap.pool_before(TARGET))
    tbl[(5, 2)] = s["first_prize"]
    tbl[(5, 1)] = s["second_prize"]
    pf = front_hit_distribution()
    R = group_payout_by_front_hit(tbl)
    e_from_R = sum(pf[f] * R[f] for f in range(6))

    agg = {}
    for f in range(6):
        for b, cnt in ((2, 1), (1, 20), (0, 45)):
            v = tbl.get((f, b), 0)
            if v > 0:
                lv = pr.level_of(TARGET, f, b) or "未中奖"
                agg[lv] = agg.get(lv, 0.0) + pf[f] * cnt * v
    assert sum(agg.values()) == pytest.approx(e_from_R, rel=1e-12)

    r = po.estimate_opportunity(
        "2026-10-13",
        {"issue": TARGET, "as_of": "2026-10-13", "n_groups": 1,
         "pool_before": snap.pool_before(TARGET)},
        prize_scenarios=[s], snapshot=snap)
    assert r["scenarios"][0]["expected_gross"] == pytest.approx(e_from_R, rel=1e-9)


def test_fallback_floating_is_marked_as_assumption_not_official():
    """固定 fallback 只能作为显式假设标注，不得冒充官方或历史实测。"""
    src = open(os.path.join(ROOT, "tools", "build_opportunity_map.py"),
               encoding="utf-8").read()
    assert "显式假设" in src
    assert "不是官方预期值" in src


# ---------------------------------------------------------------------------
# E9 一组 5+12 的一等奖概率：后区全包，不得再除以 66
# ---------------------------------------------------------------------------

def test_group_first_prize_probability_is_front_hit_probability():
    """一组 5+12 后区全包 ⇒ P(一等奖) = P(前区5中) = 1/C(35,5)，**不除以 66**。"""
    from math import comb
    p_group = 1 / comb(35, 5)
    assert p_group == pytest.approx(3.0804110500505186e-06, rel=1e-12)
    assert 1 / p_group == comb(35, 5) == 324632
    # 单注口径（除以 66）是另一种投注结构，只能作对照
    assert p_group / 66 == pytest.approx(4.667289469773513e-08, rel=1e-12)
    assert p_group / (p_group / 66) == pytest.approx(66.0, rel=1e-12)


def test_500_period_first_prize_probability_is_about_0_1539_percent():
    """500 期内至少一次一等奖 ≈ 0.1539%（旧写法只有它的 1/66）。"""
    from math import comb
    p = 1 / comb(35, 5)
    v = 1 - (1 - p) ** 500
    assert v == pytest.approx(0.0015390223857854046, rel=1e-9)
    assert 0.00153 < v < 0.00155
    # 旧（错误）口径：除以 66 后再算，只有约 0.0023%
    v_bad = 1 - (1 - p / 66) ** 500
    assert v_bad == pytest.approx(2.3336175581922802e-05, rel=1e-9)
    assert v / v_bad == pytest.approx(66.0, rel=0.01)


def test_power_module_reports_correct_probability_field():
    """analysis_power_v2 的输出字段必须用组口径，并把单注口径标为对照。"""
    import json
    p = os.path.join(ROOT, "outputs", "2026-10-13_5plus12_correction_r3",
                     "corrections", "power_v3.json")
    if not os.path.exists(p):
        pytest.skip("缺少 power_v3.json")
    d = json.load(open(p, encoding="utf-8"))
    pj = d["p_observing_first_prize"]
    assert pj["per_period_group_5plus12"] == pytest.approx(3.0804110500505186e-06, rel=1e-12)
    assert pj["over_n_periods_group"] == pytest.approx(0.0015390223857854046, rel=1e-9)
    assert "CONTRAST_ONLY" in " ".join(pj.keys())
    assert pj["over_n_periods_group"] > 60 * pj["over_n_periods_single_note_CONTRAST_ONLY"]


def test_power_module_labels_normal_approximation_as_example_only():
    """正态近似字段必须显式标为示例；可靠功效写"未确定"。"""
    import json
    p = os.path.join(ROOT, "outputs", "2026-10-13_5plus12_correction_r3",
                     "corrections", "power_v3.json")
    if not os.path.exists(p):
        pytest.skip("缺少 power_v3.json")
    d = json.load(open(p, encoding="utf-8"))
    assert d["reliable_power"]["status"] == "未确定"
    for row in d["full_distribution_power"]:
        assert "NORMAL_APPROX_EXAMPLE_ONLY_min_detectable_effect" in row
        assert "NORMAL_APPROX_EXAMPLE_ONLY_periods_for_66yuan_effect" in row
        assert row["status"] == "示例，不是结论"
        # 不应再存在未标注的旧字段名
        assert "min_detectable_effect_normal_approx" not in row


def test_docs_do_not_state_460k_or_2000yuan_as_conclusions():
    """当前文档中不得把"约46万期""可排除2000元/期"当作确定结论。"""
    import io as _io
    for rel in ("docs/project_notes/PROFIT_V2_D4_RESULTS.md",
                "docs/project_notes/PROFIT_V2_D5_BUDGET.md",
                "docs/project_notes/PROFIT_V2_WEEK1_VERDICT.md",
                "ROADMAP.md", "TASKS.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        txt = _io.open(p, encoding="utf-8").read()
        for bad in ("46 万期", "46万期"):
            if bad in txt:
                # 只允许出现在"撤回/示例"语境中
                idx = txt.find(bad)
                ctx = txt[max(0, idx - 200):idx + 120]
                assert ("撤回" in ctx or "示例" in ctx or "近似" in ctx or "不" in ctx), \
                    "{} 把 {} 当作结论使用".format(rel, bad)

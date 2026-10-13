# -*- coding: utf-8 -*-
"""V2 T1/D1 定向测试：概率论、盈亏平衡、单位、二项尾概率、显著性口径。

运行： python -m pytest tests/test_profit_math_v2.py -q

本文件对应 PLAN.md 的 T1 验收项：
  * 概率归一
  * 盈亏平衡不可达 / 单位一致 / 退化保护
  * 二项尾概率与置信上限（含 PLAN 指定的两个验算点）
  * 税前多组期望线性
  * 禁止把"模拟均值标准误"用于单个策略的显著性结论
"""
from __future__ import annotations

import os
import sys
from math import comb
from itertools import combinations

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                                    # noqa: E402
from tools import five_plus12_math as fpm                   # noqa: E402

SNAPSHOT_OK = pr.get_snapshot().ok


# ---------------------------------------------------------------------------
# 1. 概率归一与恒等式
# ---------------------------------------------------------------------------

def test_front_hit_probability_normalized_exactly():
    pf = fpm.front_hit_distribution()
    assert set(pf) == set(range(6))
    assert sum(pf.values()) == pytest.approx(1.0, abs=1e-15)
    # 与精确组合数一致
    for k in range(6):
        assert pf[k] == pytest.approx(comb(5, k) * comb(30, 5 - k) / comb(35, 5), rel=0, abs=1e-18)
    assert sum(pf.values()) == 1.0 or abs(sum(pf.values()) - 1.0) < 1e-15


def test_joint_distribution_normalized_and_back_identity():
    j = fpm.joint_distribution()
    assert sum(j.values()) == pytest.approx(1.0, abs=1e-15)
    # 后区 1/20/45 恒等式
    for f in range(6):
        assert (j[(f, 2)] + j[(f, 1)] + j[(f, 0)]) == pytest.approx(
            fpm.front_hit_distribution()[f], rel=1e-12, abs=1e-18)
    assert j[(0, 2)] * 66 == pytest.approx(fpm.front_hit_distribution()[0] * 1, rel=1e-12)


def test_single_group_payout_equals_66_single_bets():
    t = pr.base_table(25000)
    pf = fpm.front_hit_distribution()
    R = fpm.group_payout_by_front_hit(t)
    assert sum(pf[f] * R[f] for f in range(6)) == pytest.approx(
        66 * fpm.single_bet_ev(t), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. 期望的线性与"选号结构无关"（精确枚举）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_marginal_hit_distribution_independent_of_chosen_numbers():
    """对任意固定 5 码 C，|C ∩ D| 的分布都相同（枚举全部开奖组合）。"""
    allD = list(combinations(range(1, 36), 5))
    expected = [comb(5, k) * comb(30, 5 - k) for k in range(6)]
    for C in ([1, 2, 3, 4, 5], [31, 32, 33, 34, 35], [1, 2, 33, 34, 35]):
        C = frozenset(C)
        hist = [0] * 6
        for D in allD:
            hist[len(C & frozenset(D))] += 1
        assert hist == expected, C


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_total_expectation_linear_in_groups():
    """税前多组总期望 = 组数 × 单组期望（精确，不要求组间独立）。"""
    t = pr.get_prize_table(26116, "official")
    R = fpm.group_payout_by_front_hit(t)
    pf = fpm.front_hit_distribution()
    E1 = sum(pf[f] * R[f] for f in range(6))
    for n in (1, 2, 3, 5):
        m = fpm.exact_multi_group_moments(t, n_groups=n, pool_size=None)
        assert m["E_total"] == pytest.approx(n * E1, rel=1e-12)
    # 候选池结构不改变期望
    for ps in (9, 12, 15):
        m = fpm.exact_multi_group_moments(t, n_groups=3, pool_size=ps)
        assert m["E_total"] == pytest.approx(3 * E1, rel=1e-12)


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_pool_reduces_or_keeps_variance_bounded():
    """候选池只改变**方差**，且量级有限（不是"结构必然更差"）。"""
    t = pr.get_prize_table(26116, "official")
    a = fpm.exact_multi_group_moments(t, 3, None)
    b = fpm.exact_multi_group_moments(t, 3, 9)
    assert b["E_total"] == pytest.approx(a["E_total"], rel=1e-12)
    # 方差差异应远小于期望本身（这里只断言它是同量级内的调整）
    assert abs(b["sd_total"] - a["sd_total"]) / a["sd_total"] < 0.5


# ---------------------------------------------------------------------------
# 3. 盈亏平衡：单位、不可达、退化
# ---------------------------------------------------------------------------

def _toy_table(r3, r4, cost_low=5):
    return {(0, 2): cost_low, (1, 2): cost_low, (2, 2): 15, (2, 1): 5,
            (3, 2): r3, (3, 1): 15, (3, 0): 5, (4, 1): 300, (4, 0): 150,
            (4, 2): r4, (5, 0): 5000, (5, 1): 100000, (5, 2): 5_000_000}


def test_breakeven_normal_case_in_unit_interval():
    b = fpm.breakeven_scenarios(_toy_table(150, 5000), cost_group=132)
    q = b["S1_block_transfer"]["q_break_even"]
    assert 0 < q < 1
    assert b["S1_block_transfer"]["verdict"] == "可达"
    # 与手算公式一致：q* = (cost - L)/(H - L)
    assert q == pytest.approx((132 - b["L_mean_low_hits"]) /
                              (b["H_mean_high_hits"] - b["L_mean_low_hits"]), rel=1e-12)


def test_breakeven_below_zero_when_low_hits_already_cover_cost():
    """低命中组条件均值已超过成本时，解为负 -> 标记"已满足"，而不是给出负数门槛。"""
    b = fpm.breakeven_scenarios(_toy_table(150, 5000), cost_group=10)
    s1 = b["S1_block_transfer"]
    assert s1["q_break_even"] is not None
    assert s1["q_break_even"] <= 0
    assert "已" in s1["verdict"]          # "已在当前假设下满足"
    assert b["expected_roi"] > 0


def test_breakeven_unreachable_flagged_and_bounded():
    """即使把所有质量堆到高命中组也无法达到成本 -> 必须标记为不可达。"""
    b = fpm.breakeven_scenarios(_toy_table(150, 5000), cost_group=10 ** 9)
    s1 = b["S1_block_transfer"]
    assert s1["q_break_even"] > 1
    assert "不可达" in s1["verdict"]
    for k in ("S2_only_front3", "S3_only_front4"):
        v = b[k]["verdict"]
        assert ("不可达" in v) or (b[k]["p_break_even"] is not None and b[k]["p_break_even"] > 1)


def test_breakeven_scenario_K_requires_larger_shift_than_S1():
    """只增加"恰好 3 中"所需的概率，必须高于整块搬运所需的总概率。"""
    t = _toy_table(150, 5000)
    b = fpm.breakeven_scenarios(t, cost_group=132)
    q_s1 = b["S1_block_transfer"]["q_break_even"]
    p_s2 = b["S2_only_front3"]["p_break_even"]
    pf = fpm.front_hit_distribution()
    assert p_s2 > pf[3]
    assert b["S2_only_front3"]["p_multiple"] > b["S1_block_transfer"]["q_multiple"] * 0.999
    assert q_s1 > 0


def test_breakeven_unit_consistency_across_group_counts():
    """盈亏平衡是**单组**口径；组数只按比例放大成本与回收，不改变结论。"""
    t = _toy_table(150, 5000)
    b1 = fpm.breakeven_scenarios(t, cost_group=132)
    # 成本翻倍时，需要的偏移量必须同步变大（S2 口径下可直接比较）
    b2 = fpm.breakeven_scenarios(t, cost_group=264)
    assert b2["S2_only_front3"]["p_break_even"] > b1["S2_only_front3"]["p_break_even"]
    assert b1["cost_group"] == 132 and b2["cost_group"] == 264


def test_breakeven_reports_payout_ratio_not_probability_multiple():
    """必须显式区分"回收倍数"与"命中率倍数"，避免 132/47.99 被读成概率倍数。"""
    b = fpm.breakeven_scenarios(_toy_table(150, 5000), cost_group=132)
    assert "payout_ratio_needed" in b
    assert "note_payout_ratio" in b
    assert isinstance(b["note_payout_ratio"], str)
    assert b["payout_ratio_needed"] == pytest.approx(132 / b["expected_payout"], rel=1e-12)
    q_mult = b["S1_block_transfer"]["q_multiple"]
    assert q_mult is not None
    assert abs(q_mult - b["payout_ratio_needed"]) > 1e-6   # 两者不相等，不能混用


# ---------------------------------------------------------------------------
# 4. 二项尾概率与置信上限（含 PLAN 指定的两个验算点）
# ---------------------------------------------------------------------------

def _p_front_ge3():
    return 1 - sum(comb(5, k) * comb(30, 5 - k) / comb(35, 5) for k in range(3))


def test_p_front_ge3_exact_value():
    assert _p_front_ge3() == pytest.approx(0.013864930136277653, rel=1e-12)


def test_binomial_tail_plan_checkpoint_345():
    """PLAN 指定的验算点：真实命中率 3% 时，74 期观测到 <=1 次的概率约 34.5%。"""
    v = stats.binom.cdf(1, 74, 0.03)
    assert v == pytest.approx(0.345244910053285, abs=1e-12)
    assert 0.34 < v < 0.35


def test_binomial_upper_confidence_bound_plan_checkpoint_625():
    """PLAN 指定的验算点：74 期观测 1 次，单侧 95% 上限约 6.25%。"""
    ub = brentq(lambda p: stats.binom.cdf(1, 74, p) - 0.05, 1e-6, 0.5)
    assert 0.0620 < ub < 0.0630
    # 该上限必须**包含**零假设下的 1.3865%（即无法排除随机）
    assert ub > _p_front_ge3()


def test_observed_1_in_74_is_consistent_with_null():
    """1/74 与零假设完全相容 —— 不能写成"命中率回到了随机水平"之外的任何强结论。"""
    p0 = _p_front_ge3()
    assert stats.binom.cdf(1, 74, p0) == pytest.approx(0.7261321321458554, abs=1e-9)
    # 单侧 p 值（检验"优于随机"）应约为 1 - 0.726 = 0.274
    assert stats.binom.sf(0, 74, p0) == pytest.approx(1 - (1 - p0) ** 74, rel=1e-12)


def test_three_percent_true_rate_is_not_rejected_by_74_periods():
    """74 期**无法**排除 3% 的真实命中率（这是"未拒绝"而非"证明")。"""
    assert stats.binom.cdf(1, 74, 0.03) > 0.05      # 不能拒绝
    # 而要能达到 80% 功效去检测 p=0.03（单侧 5%），至少需要数百期
    def power(n):
        crit = stats.binom.ppf(0.95, n, _p_front_ge3())
        return 1 - stats.binom.cdf(crit, n, 0.03)
    assert power(74) < 0.5
    assert power(1000) > power(74)


# ---------------------------------------------------------------------------
# 5. 显著性口径：禁止把模拟均值标准误当成单个策略的检验
# ---------------------------------------------------------------------------

def test_mc_se_is_not_a_hypothesis_test():
    """(策略值 − 对照均值) / (sd/√n_trials) 会随 n_trials 无限放大。"""
    val, mean, sd = -0.74, -0.64, 0.05
    z1 = (val - mean) / (sd / np.sqrt(100))
    z2 = (val - mean) / (sd / np.sqrt(100_000))
    assert abs(z2) > 10 * abs(z1)              # 同一个事实，放大 31.6 倍
    # 正确做法：看距离分布的位置（百分位）或直接看金额差
    assert abs(val - mean) == pytest.approx(0.10, abs=1e-12)


def test_conditional_gap_standard_error_is_period_level():
    """候选池效应的正确标准误来自"期"级别的条件期望波动，而不是 n_trials。"""
    import json
    p = os.path.join(ROOT, "outputs", "2026-10-13_5plus12_correction_r2",
                     "corrections", "pool_conditional_v2_withpools.json")
    if not os.path.exists(p):
        pytest.skip("缺少条件分析产物")
    d = json.load(open(p, encoding="utf-8"))
    s = d["windows"]["oos_26043_26116"]["summary_no_jackpot"]
    # 条件期望在期级别的波动远大于 0，因此标准误是"分量级"的
    assert s["mean_sd_g"] > 50
    assert s["se_of_window_mean_gap"] > 5
    # 已实现的候选池-开奖重叠与零假设一致（9*5/35）
    assert s["realised_mean_m"] == pytest.approx(9 * 5 / 35, abs=0.5)
    # 修正后的 z 必须在噪声范围内，而不是旧的 −9 ~ −11
    assert abs(s["realised_gap_z_conditional"]) < 2


def test_summary_marks_old_structure_z_as_retracted():
    import json
    p = os.path.join(ROOT, "outputs", "2026-10-13_5plus12_correction_r2",
                     "corrections", "SUMMARY.json")
    if not os.path.exists(p):
        pytest.skip("缺少纠错后的 SUMMARY")
    d = json.load(open(p, encoding="utf-8"))
    st = d.get("structure_test", {})
    assert st.get("status") == "RETRACTED_AND_REPLACED"
    assert "z" not in st or st.get("z") is None
    assert "corrected_no_jackpot" in st

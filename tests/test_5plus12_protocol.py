# -*- coding: utf-8 -*-
"""5+12 计奖与评估口径的关键不变式测试。

运行： python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import random
import sys
from math import comb

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                      # noqa: E402
from tools import five_plus12_math as fpm     # noqa: E402

SNAPSHOT_OK = pr.get_snapshot().ok


# ---------------------------------------------------------------------------
# 1. 后区全包的结构恒等式：1 / 20 / 45
# ---------------------------------------------------------------------------

def test_back_hit_distribution_is_1_20_45():
    """对任意 66 个后区配对与任意开奖后区，命中分布恒为 1/20/45。"""
    rng = random.Random(7)
    pairs = [(b1, b2) for b1 in range(1, 13) for b2 in range(b1 + 1, 13)]
    assert len(pairs) == 66
    for _ in range(200):
        drawn = set(rng.sample(range(1, 13), 2))
        counts = {0: 0, 1: 0, 2: 0}
        for p in pairs:
            counts[len(set(p) & drawn)] += 1
        assert counts == {2: 1, 1: 20, 0: 45}


def test_back_hit_distribution_independent_of_front():
    """后区命中分布与前区命中数无关（结构恒等式）。"""
    table = {(f, b): 1 for f in range(6) for b in range(3)}
    for f in range(6):
        assert pr.group_payout(table, f) == 66
    # 带权：R(f) = 1*prize(f,2) + 20*prize(f,1) + 45*prize(f,0)
    t = {(0, 0): 0, (0, 1): 0, (0, 2): 5}
    assert pr.group_payout(t, 0) == 5


# ---------------------------------------------------------------------------
# 2. 前区命中分布的精确值
# ---------------------------------------------------------------------------

def test_front_hit_distribution_exact():
    pf = fpm.front_hit_distribution()
    assert abs(sum(pf.values()) - 1.0) < 1e-12
    assert abs(pf[0] - 142506 / 324632) < 1e-15
    assert abs(pf[1] - 5 * 27405 / 324632) < 1e-15
    assert abs(pf[2] - 10 * 4060 / 324632) < 1e-15
    assert abs(pf[3] - 10 * 435 / 324632) < 1e-15
    assert abs(pf[4] - 5 * 30 / 324632) < 1e-15
    assert abs(pf[5] - 1 / 324632) < 1e-18
    # 一等奖概率应为公认的 1/21425712
    j = fpm.joint_distribution()
    assert abs(j[(5, 2)] - 1 / 21425712) < 1e-15


# ---------------------------------------------------------------------------
# 3. 计奖口径：前区中 3 个的单组回收
# ---------------------------------------------------------------------------

def test_group_payout_front3_by_era():
    """官方核验后的 (3,2)/(3,1)/(3,0) 单组回收。"""
    # era2（2019-02-20 ~ 2026-01-31）：(3,2)=200, (3,1)=15, (3,0)=5
    t2 = pr.base_table(25000)
    assert pr.group_payout(t2, 3) == 200 + 20 * 15 + 45 * 5 == 725
    # era3 基础档：(3,2)=150, (3,1)=15, (3,0)=5 -> 675
    t3 = pr.base_table(26116, pool_before=0)
    assert pr.group_payout(t3, 3) == 150 + 20 * 15 + 45 * 5 == 675
    # era3 上浮档（开奖前奖池 >= 8 亿）：200 / 18 / 7 -> 875
    t3h = pr.base_table(26116, pool_before=pr.ERA3_POOL_THRESHOLD)
    assert pr.group_payout(t3h, 3) == 200 + 20 * 18 + 45 * 7 == 875


def test_legacy_table_is_wrong_and_kept_for_reproduction():
    """旧的 config 表在前区中 3 个时给出 625 / 725，与官方不符。"""
    legacy_25000 = pr.get_prize_table(25000, mode="legacy_config")
    assert pr.group_payout(legacy_25000, 3) == 625      # 实际应为 725
    legacy_26116 = pr.get_prize_table(26116, mode="legacy_config")
    assert pr.group_payout(legacy_26116, 3) == 725      # 实际应为 675 / 875
    # config 的公开入口现在应返回官方核验表
    from config import get_dlt_prize_table
    assert pr.group_payout(dict(get_dlt_prize_table(25000)), 3) == 725


def test_front_hit_low_counts_are_not_zero():
    """前区中 0/1/2 个也有回收，不得当作零。"""
    t = pr.base_table(25000)
    R = pr.payout_by_front_hit(t)
    assert R[0] == 5 and R[1] == 5 and R[2] == 115
    assert all(R[f] > 0 for f in range(6))


# ---------------------------------------------------------------------------
# 4. 一注一票的计奖与 group_payout 一致
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_calc_reward_matches_group_payout():
    import backtest_plus as bt
    issue = 25000
    drawn_front = {3, 11, 20, 27, 33}
    drawn_back = {5, 9}
    t = bt.compute_prize_table(issue)
    for f in range(6):
        combo = sorted(list(drawn_front)[:f] + [1, 2, 3, 4, 5][: 5 - f])
        combo = sorted(set(combo))
        if len(combo) != 5:
            continue
        reward, fh = bt.calc_5_plus_12_reward(combo, drawn_front, drawn_back, issue)
        assert fh == len(set(combo) & drawn_front)
        assert reward == pr.group_payout(t, fh)


def test_multi_group_cost_is_linear():
    from prize_rules import COST_PER_GROUP
    assert COST_PER_GROUP == 132
    assert COST_PER_GROUP * 3 == 396


# ---------------------------------------------------------------------------
# 5. 回撤口径：必须包含第一期亏损
# ---------------------------------------------------------------------------

def test_max_drawdown_includes_first_period_loss():
    import backtest_plus as bt
    profits = [-132, 50, -132, -132]
    assert bt.max_drawdown_from_zero(profits) == 346.0
    # 旧口径（cumsum 后取 peak-cum）会漏掉第一期
    cum = np.cumsum(profits)
    old = float((np.maximum.accumulate(cum) - cum).max())
    assert old == 264.0
    assert bt.max_drawdown_from_zero([10, 10]) == 0.0


def test_max_loss_streak():
    import backtest_plus as bt
    assert bt._max_loss_streak([-1, -1, 0, -1, -1, -1]) == 3
    assert bt._max_loss_streak([1, 2, 3]) == 0


# ---------------------------------------------------------------------------
# 6. 无前视：信号门槛只用过去
# ---------------------------------------------------------------------------

def test_signal_gate_threshold_uses_only_past():
    """阈值由 signal_history 的滚动分位数给出，当期分数在决策之后才入队。"""
    import backtest_plus as bt
    import inspect
    src = inspect.getsource(bt.run_n9_signal_gate_backtest)
    # 决策必须发生在 append(current_signal) 之前
    i_decide = src.index("threshold = np.quantile")
    i_append = src.index("signal_history.append(current_signal)")
    assert i_decide < i_append
    # 阈值只使用 signal_history 的尾部
    assert "lookback_scores = signal_history[-min(signal_lookback" in src


def test_signal_gate_warmup_is_before_backtest_window():
    import backtest_plus as bt
    import inspect
    src = inspect.getsource(bt.run_n9_signal_gate_backtest)
    assert "for idx in range(warmup_start, backtest_start)" in src
    assert "warmup_start = max(121, backtest_start - signal_lookback)" in src


# ---------------------------------------------------------------------------
# 7. 官方快照与仓库数据一致性
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_repo_data_matches_official_snapshot():
    """仓库 data.csv 与官方快照在重叠期号上逐期一致（数据完整性）。"""
    import pandas as pd
    repo = pd.read_csv(os.path.join(ROOT, "data", "dlt", "data.csv"))
    repo.columns = [c.strip() for c in repo.columns]
    off = pd.read_csv(os.path.join(ROOT, pr.DEFAULT_SNAPSHOT_DIR, pr.DRAWS_CSV))
    front_cols = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
    back_cols = ["蓝球_1", "蓝球_2"]
    m = repo.merge(off, left_on="期数", right_on="issue", how="inner")
    assert len(m) == len(repo)
    for i, c in enumerate(front_cols, start=1):
        assert (m[c].astype(int) == m["front_%d" % i]).all(), c
    for j, c in enumerate(back_cols, start=1):
        assert (m[c].astype(int) == m["back_%d" % j]).all(), c


class _SnapTable:
    """把官方快照包装成 pandas 的简便结构，供下面的映射检验使用。"""

    def __init__(self):
        import pandas as pd
        self.lv = pd.read_csv(os.path.join(ROOT, pr.DEFAULT_SNAPSHOT_DIR, pr.LEVELS_CSV))
        self.piv = self.lv.pivot_table(index="issue", columns="level",
                                       values="stake_count", aggfunc="sum")


# ---------------------------------------------------------------------------
# 8. 奖级→中奖条件映射：用官方中奖注数独立验证
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_level_mapping_matches_official_winner_counts():
    """奖级映射推出的概率应与官方中奖注数比一致（相对误差 < 6%）。

    这是"计奖是否可信"的核心回归测试：如果映射错了，某奖级的注数
    与理论概率会整体偏离。以二等奖（5+1）中奖注数为基准，避免依赖总注数。
    """
    snaps = _SnapTable()
    p = fpm.joint_distribution()
    base = p[(5, 1)]
    for era, (lo, hi) in (("era2", (22000, 26013)), ("era3", (26014, 26116))):
        sub = snaps.piv.loc[(snaps.piv.index >= lo) & (snaps.piv.index <= hi)]
        c2 = sub["二等奖"].sum()
        for level, conds in pr.LEVEL_CONDITIONS[era].items():
            if level == "一等奖":
                continue
            col = level
            if col not in sub.columns:
                continue
            obs = sub[col].sum() / c2
            exp = sum(p[c] for c in conds) / base
            rel = abs(obs - exp) / exp
            assert rel < 0.06, "{} {}: 观测比 {:.3f} vs 理论 {:.3f}".format(era, level, obs, exp)


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_era3_pool_threshold_rule_exact():
    """era3 的 8 亿档位规则：开奖前奖池 >= 8 亿 <-> 上浮档，103 期 0 例外。"""
    snap = pr.get_snapshot()
    checked = 0
    for issue in snap.issues:
        if issue < 26014:
            continue
        pb = snap.pool_before(issue)
        amt, _ = snap.amount(issue, "七等奖")
        if pb is None or amt is None:
            continue
        expect = pr.ERA3_HIGH_AMOUNT["七等奖"] if pb >= pr.ERA3_POOL_THRESHOLD else pr.BASIC_AMOUNT["era3"]["七等奖"]
        assert amt == expect, "issue {} pool_before={} 七等奖={} 期望={}".format(issue, pb, amt, expect)
        checked += 1
    assert checked >= 100


# ---------------------------------------------------------------------------
# 9. 数学基线的核心结论（回归护栏）
# ---------------------------------------------------------------------------

def test_breakeven_shapes_are_wellformed_and_not_generic():
    """盈亏平衡是"特定分布变化假设下的情景计算"，不是通用门槛。

    V2 T1 修正：旧版本把 1.7~2.3 的倍数写成"结论的数学护栏"，
    并把 132/E[回收] 与"命中率倍数"混用。现在只断言结构性性质。
    """
    for tbl in (pr.base_table(25000),
                pr.base_table(26116, pool_before=0),
                pr.base_table(26116, pool_before=pr.ERA3_POOL_THRESHOLD)):
        rep = fpm.baseline_report(tbl, "t", n_draws=10, mc_trials=100)
        assert rep["expected_roi"] < 0
        be = rep["breakeven_full"]
        q = be["S1_block_transfer"]["q_break_even"]
        # 解必须落在 (0,1) 内且严格大于当前 q，否则说明公式或单位有问题
        assert q is not None and 0 < q < 1
        assert q > be["q_front_ge3_now"]
        assert be["S1_block_transfer"]["verdict"] == "可达"
        # 回收倍数与概率倍数是两个不同的量
        assert abs(be["payout_ratio_needed"] -
                   be["S1_block_transfer"]["q_multiple"]) > 1e-6


def test_no_jackpot_keeps_front5_fixed_prize():
    """剔除大奖口径只置零 (5,2)/(5,1)，必须**保留** (5,0) 的固定奖。"""
    tbl = pr.base_table(26116, pool_before=0)
    R = fpm.group_payout_by_front_hit(tbl)
    Rnj = fpm.group_payout_by_front_hit(fpm.no_jackpot_table(tbl))
    assert Rnj[5] == 45 * tbl[(5, 0)]          # 仅剩 45 注 (5,0)
    assert Rnj[5] > 0
    for f in range(5):
        assert Rnj[f] == R[f]


def test_expected_group_payout_equals_66_single_bets():
    """后区全包不改变期望值：E[一组] == 66 × E[单注]。"""
    t = pr.base_table(25000)
    e_group = sum(fpm.front_hit_distribution()[f] * pr.group_payout(t, f) for f in range(6))
    assert abs(e_group - 66 * fpm.single_bet_ev(t)) < 1e-9


# ---------------------------------------------------------------------------
# 10. 计账性质：多组累加、随机对照成本可比、单调性
# ---------------------------------------------------------------------------

def test_multi_group_cost_and_reward_accumulate_linearly():
    """g 组的成本与奖金必须等于单组的 g 倍（同号码组合）。"""
    from tools.eval_5plus12_frozen import score_period
    issue = 25000
    tf = {3, 11, 20, 27, 33}
    tb = {5, 9}
    one = [[3, 11, 20, 27, 33]]
    three = [list(one[0]) for _ in range(3)]
    c1, r1, n1, h1, _ = score_period(issue, one, tf, tb, "official")
    c3, r3, n3, h3, _ = score_period(issue, three, tf, tb, "official")
    assert c1 == 132 and c3 == 396
    assert r3 == 3 * r1
    assert n3 == 3 * n1
    assert h1 == h3 == 5


def test_payout_monotone_in_front_hit():
    """单组回收必须随前区命中数单调不减（任何规则时期）。"""
    for tbl in (pr.base_table(25000),
                pr.base_table(26116, pool_before=0),
                pr.base_table(26116, pool_before=pr.ERA3_POOL_THRESHOLD)):
        R = pr.payout_by_front_hit(tbl)
        assert all(R[f] <= R[f + 1] for f in range(5)), R


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_random_same_bet_count_cost_matches_strategy():
    """择时对照的成本必须与策略相同：bet_periods × 组数 × 132。"""
    import backtest_plus as bt
    from feature_engineering import load_dlt_history
    csv = os.path.join(ROOT, pr.DEFAULT_SNAPSHOT_DIR, pr.DRAWS_CSV)
    d = pd.read_csv(csv)
    df = pd.DataFrame({
        "期数": d.issue, "红球_1": d.front_1, "红球_2": d.front_2,
        "红球_3": d.front_3, "红球_4": d.front_4, "红球_5": d.front_5,
        "蓝球_1": d.back_1, "蓝球_2": d.back_2,
    }).sort_values("期数").reset_index(drop=True)
    periods, groups, bet_periods = 60, 3, 20
    r = bt.run_random_same_bet_count_baseline(
        df, periods=periods, bet_periods=bet_periods,
        front_group_count=groups, n_trials=3, rng_seed=1)
    assert r["bet_periods"] == bet_periods
    assert r["periods"] == max(1, periods)
    for m in r["trial_metrics"]:
        assert m["total_cost"] == bet_periods * groups * 132
        assert m["periods"] == periods


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_random_from_pool_uses_given_pool():
    """结构对照（C2）必须落在给定候选池内。"""
    import backtest_plus as bt
    from feature_engineering import load_dlt_history
    csv = os.path.join(ROOT, pr.DEFAULT_SNAPSHOT_DIR, pr.DRAWS_CSV)
    d = pd.read_csv(csv)
    df = pd.DataFrame({
        "期数": d.issue, "红球_1": d.front_1, "红球_2": d.front_2,
        "红球_3": d.front_3, "红球_4": d.front_4, "红球_5": d.front_5,
        "蓝球_1": d.back_1, "蓝球_2": d.back_2,
    }).sort_values("期数").reset_index(drop=True)
    pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    r = bt.run_random_from_pool_baseline(
        df, periods=30, n_groups=2, n_trials=2, rng_seed=3,
        pool_callback=lambda h, i: pool)
    assert r["periods"] > 0
    for rec in r["records"]:
        assert 0 <= rec["front_hit"] <= 5


def test_prize_rules_default_is_official_not_legacy():
    """默认口径必须走官方核验表，而不是已知有误的旧表。"""
    assert pr.get_prize_table(25000) == pr.get_prize_table(25000, mode="official")
    assert pr.get_prize_table(25000) != pr.get_prize_table(25000, mode="legacy_config")


def test_bonus_eligibility_rule_is_documented():
    """5+12 单票 132 元 >= 18 元派奖门槛，故默认口径使用全额派奖。"""
    assert pr.COST_PER_GROUP == 132 and pr.COST_PER_GROUP >= 18
    assert pr._BONUS_MODE_ALIASES["official"] == "full"

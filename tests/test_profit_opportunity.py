# -*- coding: utf-8 -*-
"""V2 T3/D3 定向测试：开奖前机会估计、未来信息拒绝、资格降级、缺口。

运行： python -m pytest tests/test_profit_opportunity.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                                     # noqa: E402
from tools import profit_opportunity as po                   # noqa: E402

SNAPSHOT_OK = pr.get_snapshot().ok
TARGET = 26116


def _base_context(**kw):
    ctx = {"issue": TARGET, "as_of": TARGET, "n_groups": 1, "pool_size": None,
           "pool_before": 733_972_024.0}
    ctx.update(kw)
    return ctx


# ---------------------------------------------------------------------------
# 1. 未来信息必须被拒绝
# ---------------------------------------------------------------------------

def test_rejects_future_outcome_fields_in_context():
    for bad in ({"actual_front": [1, 2, 3, 4, 5]},
                {"amt_一等奖": 10_000_000},
                {"cnt_二等奖": 120},
                {"winning_numbers": "1 2 3 4 5 6 7"},
                {"draw_result": "..."}):
        with pytest.raises(po.OpportunityInputError):
            po.estimate_opportunity(TARGET, _base_context(**bad),
                                    prize_scenarios=[
                                        po.make_floating_scenario("s", 1, 1, [TARGET - 1], "t")])


def test_rejects_scenario_built_from_target_or_future_issue():
    """情景来源期次必须全部早于目标期。"""
    for src in ([TARGET], [TARGET - 1, TARGET], [TARGET + 1]):
        with pytest.raises(po.OpportunityInputError):
            po.estimate_opportunity(TARGET, _base_context(),
                                    prize_scenarios=[
                                        po.make_floating_scenario("bad", 1, 1, src, "t")])


def test_rejects_scenario_without_declared_sources():
    with pytest.raises(po.OpportunityInputError):
        po.estimate_opportunity(TARGET, _base_context(),
                                prize_scenarios=[
                                    {"label": "x", "first_prize": 1, "second_prize": 1}])


def test_rejects_non_normalized_hit_distribution():
    with pytest.raises(po.OpportunityInputError):
        po.estimate_opportunity(
            TARGET, _base_context(), hit_distribution={0: 0.5, 1: 0.5},
            prize_scenarios=[po.make_floating_scenario("s", 1, 1, [TARGET - 1], "t")])
    with pytest.raises(po.OpportunityInputError):
        po.estimate_opportunity(
            TARGET, _base_context(),
            hit_distribution={0: 1.2, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            prize_scenarios=[po.make_floating_scenario("s", 1, 1, [TARGET - 1], "t")])


# ---------------------------------------------------------------------------
# 2. 均匀基线
# ---------------------------------------------------------------------------

def test_default_hit_distribution_is_uniform_hypergeometric():
    r = po.estimate_opportunity(
        TARGET, _base_context(),
        prize_scenarios=[po.make_floating_scenario("s", 5_000_000, 100_000,
                                                   [TARGET - 1], "t")])
    assert r["hit_distribution_source"] == "uniform_hypergeometric"
    from math import comb
    for k in range(6):
        assert r["hit_distribution"][str(k)] == pytest.approx(
            comb(5, k) * comb(30, 5 - k) / comb(35, 5), rel=1e-12)


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_provided_hit_distribution_is_validated_and_used():
    hd = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}   # 假设"必中 3 个"
    r = po.estimate_opportunity(
        TARGET, _base_context(), hit_distribution=hd,
        prize_scenarios=[po.make_floating_scenario("s", 5_000_000, 100_000,
                                                   [TARGET - 1], "t")])
    assert r["hit_distribution_source"] == "provided"
    # 必中 3 个时，一期回收就是 R(3)（26116 基础档 = 675）
    assert r["scenarios"][0]["expected_gross"] == pytest.approx(675, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. 未知派奖资格必须降级
# ---------------------------------------------------------------------------

def test_bonus_downgraded_when_certainty_not_official():
    """bonus_certainty != official 时不得把派奖计入机会估计。"""
    r = po.estimate_opportunity(
        TARGET, _base_context(bonus_active=True, bonus_certainty="uncertain"),
        prize_scenarios=[po.make_floating_scenario("s", 1_000_000, 50_000,
                                                   [TARGET - 1], "t")])
    assert r["bonus_applied"] is False
    assert "降级" in r["fixed_prize_source"]


def test_bonus_applied_only_when_active_and_official_and_eligible():
    r = po.estimate_opportunity(
        TARGET, _base_context(bonus_active=True, bonus_certainty="official",
                              bonus_eligibility=True),
        prize_scenarios=[po.make_floating_scenario("s", 1_000_000, 50_000,
                                                   [TARGET - 1], "t")])
    assert r["bonus_applied"] is True


# ---------------------------------------------------------------------------
# 4. 结论形态：全非正 -> 无候选
# ---------------------------------------------------------------------------

def test_no_candidate_when_all_scenarios_non_positive():
    r = po.estimate_opportunity(
        TARGET, _base_context(),
        prize_scenarios=[
            po.make_floating_scenario("悲观", 2_000_000, 20_000, [TARGET - 1], "t"),
            po.make_floating_scenario("中等", 5_000_000, 100_000, [TARGET - 1], "t"),
            po.make_floating_scenario("乐观", 10_000_000, 200_000, [TARGET - 1], "t"),
        ])
    assert r["verdict"] == "no_positive_scenario"
    assert all(s["expected_profit"] <= 0 for s in r["scenarios"])


def test_verdict_is_fragile_when_only_optimistic_scenario_is_positive():
    """假设"必中 3 个"才能转正 -> 必须判为 fragile，不能判为稳健。"""
    hd = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}
    r = po.estimate_opportunity(
        TARGET, _base_context(), hit_distribution=hd,
        prize_scenarios=[
            po.make_floating_scenario("悲观", 2_000_000, 20_000, [TARGET - 1], "t"),
            po.make_floating_scenario("中等", 5_000_000, 100_000, [TARGET - 1], "t"),
            po.make_floating_scenario("乐观", 10_000_000, 200_000, [TARGET - 1], "t"),
        ])
    assert r["verdict"] in ("positive_but_fragile", "positive_under_all_scenarios")
    assert r["scenarios"][0]["expected_profit"] > 0


# ---------------------------------------------------------------------------
# 5. 税与分摊敏感性
# ---------------------------------------------------------------------------

def test_tax_applied_on_merged_period_income():
    """一期多组必须合并判定起征点。"""
    hd = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}
    # 每组中 3 个 -> R(3)=675；4 组 -> 2700，仍 ≤10000
    r4 = po.estimate_opportunity(
        TARGET, _base_context(n_groups=4), hit_distribution=hd,
        prize_scenarios=[po.make_floating_scenario("s", 1, 1, [TARGET - 1], "t")])
    assert r4["scenarios"][0]["expected_gross"] == pytest.approx(4 * 675, rel=1e-6)
    # 30 组 -> 20250 > 10000，应计 20% 税
    r30 = po.estimate_opportunity(
        TARGET, _base_context(n_groups=30), hit_distribution=hd,
        prize_scenarios=[po.make_floating_scenario("s", 1, 1, [TARGET - 1], "t")])
    assert r30["scenarios"][0]["expected_gross"] == pytest.approx(30 * 675, rel=1e-6)
    assert r30["scenarios"][0]["expected_profit"] < 30 * 675 - 30 * 132


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_dilution_sensitivity_uses_only_counts_no_collision_model():
    snap = pr.get_snapshot()
    d = snap.draws
    counts = d[(d.issue < TARGET)].tail(200)["cnt_二等奖"].tolist()
    out = po.dilution_factor(counts)
    assert out["n"] > 0
    assert 0.9 < out["median_factor"] < 1.0
    assert "未建立" in out["note"]


def test_dilution_factor_handles_missing_data():
    out = po.dilution_factor([])
    assert out["n"] == 0
    assert "缺少" in out["note"]


# ---------------------------------------------------------------------------
# 6. 缺失信息必须被列出
# ---------------------------------------------------------------------------

def test_missing_information_is_reported():
    r = po.estimate_opportunity(
        TARGET, _base_context(),
        prize_scenarios=[po.make_floating_scenario("s", 1_000_000, 50_000,
                                                   [TARGET - 1], "t")])
    assert any("销售额" in m for m in r["missing_information"])
    assert any("中奖注数" in m for m in r["missing_information"])


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_scenario_from_history_never_uses_target_or_future():
    snap = pr.get_snapshot()
    s = po.scenario_from_history(snap, TARGET, mask="all", q=0.5, window=200)
    assert max(s["source_issues"]) < TARGET
    r = po.estimate_opportunity(TARGET, _base_context(), prize_scenarios=[s])
    assert r["scenarios"][0]["source_issues_max"] < TARGET


# ---------------------------------------------------------------------------
# 7. 未来期次的开奖前奖池（档位判定不得前视）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_pool_before_for_undrawn_future_issue_uses_last_published_pool():
    """尚未开奖的期次，其"开奖前奖池"就是快照最新一期的开奖后奖池。"""
    snap = pr.get_snapshot()
    last = snap.issues[-1]
    nxt = last + 1
    assert snap.pool_before(nxt) == snap.pool_after[last]
    # 必须仍能在开奖前确定固定奖档位
    pb = snap.pool_before(nxt)
    expect_high = pb >= pr.ERA3_POOL_THRESHOLD
    t_high = pr.base_table(nxt, pool_before=pb)
    t_base = pr.base_table(nxt, pool_before=0)
    # (5,0) 属三等奖：上浮档 6666，基础档 5000
    assert (t_high[(5, 0)] == pr.ERA3_HIGH_AMOUNT["三等奖"]) == expect_high
    assert t_base[(5, 0)] == pr.BASIC_AMOUNT["era3"]["三等奖"]


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_opportunity_uses_correct_tier_for_future_issue():
    snap = pr.get_snapshot()
    nxt = snap.issues[-1] + 1
    pb = snap.pool_before(nxt)
    r = po.estimate_opportunity(
        "2026-10-13",
        {"issue": nxt, "as_of": "2026-10-13", "n_groups": 1, "pool_size": None,
         "pool_before": pb, "bonus_active": False, "bonus_certainty": "official"},
        prize_scenarios=[po.make_floating_scenario("s", 5_000_000, 100_000,
                                                   [nxt - 1], "t")],
        snapshot=snap)
    assert r["pool_before"] == pb
    assert r["tax_certainty"] in ("official", "uncertain", "unknown")

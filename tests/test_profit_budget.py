# -*- coding: utf-8 -*-
"""V2 T4 定向测试：亏损预算约束（PLAN §5）。

运行： python -m pytest tests/test_profit_budget.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import profit_budget as pb       # noqa: E402


# ---------------------------------------------------------------------------
# 1. 边界
# ---------------------------------------------------------------------------

def test_boundary_exactly_at_limit_is_allowed():
    """累计亏损 + 未结算 + 拟投入 == 限额 -> 允许（≤ 而非 <）。"""
    assert pb.can_allocate(368, 0, 132, 500) is True      # 368+132 = 500
    assert pb.can_allocate(369, 0, 132, 500) is False     # 501 > 500


def test_boundary_zero_everything():
    assert pb.can_allocate(0, 0, 0, 500) is True
    assert pb.can_allocate(500, 0, 1, 500) is False


def test_negative_inputs_are_clamped_not_credited():
    """已实现盈利（负的 loss_to_date）不得当作额外额度。"""
    assert pb.can_allocate(-900, 0, 132, 500) is True
    assert pb.can_allocate(-900, 0, 501, 500) is False    # 不能拿 900 盈利去加仓


# ---------------------------------------------------------------------------
# 2. 未结算暴露占用额度
# ---------------------------------------------------------------------------

def test_unsettled_exposure_is_committed():
    t = pb.BudgetTracker(loss_limit=500)
    assert t.can_allocate(396) is True
    assert t.reserve("A", 396) is True
    assert t.unsettled_cost == 396
    # 未结算已占用 396，再买 1 组 132 会超（396+132=528>500）
    assert t.can_allocate(132) is False
    assert t.reserve("B", 132) is False
    assert t.max_affordable_groups() == 0          # 500-396=104 < 132


def test_settlement_releases_exposure_and_updates_loss():
    t = pb.BudgetTracker(loss_limit=500)
    t.reserve("A", 132); t.reserve("B", 132); t.reserve("C", 132)
    assert t.unsettled_cost == 396
    t.settle("A", -132)
    t.settle("B", -132)
    t.settle("C", -132)
    assert t.unsettled_cost == 0
    assert t.realized_net == -396
    assert t.loss_to_date == 396
    assert t.remaining() == 104
    assert t.max_affordable_groups() == 0

    t.settle("A", 0)                                # 幂等/重复结算不应重复计
    assert t.realized_net == -396


def test_worst_case_uses_full_ticket_cost():
    """最坏新增损失按全部票款计，不预支可能回收的小奖。"""
    t = pb.BudgetTracker(loss_limit=500)
    assert t.can_allocate(396) is True      # 即使期望会回收一部分，也按全额占用
    assert t.can_allocate(528) is False


# ---------------------------------------------------------------------------
# 3. 换策略不重置
# ---------------------------------------------------------------------------

def test_switching_strategy_does_not_reset_loss():
    t = pb.BudgetTracker(loss_limit=500)
    t.reserve("s1_26043", 132); t.settle("s1_26043", -132)
    t.reserve("s1_26044", 132); t.settle("s1_26044", -132)
    assert t.loss_to_date == 264
    # 换到另一个策略/另一种组数 —— 累计亏损必须延续
    assert t.can_allocate(264) is False     # 264+264 = 528 > 500
    assert t.can_allocate(236) is True      # 264+236 = 500

    assert t.reserve("s2_26045", 236) is True
    assert pb.loss_to_date_from(t.realized_net) == 264   # reserve 不改动已实现盈亏


def test_no_period_or_daily_reset():
    """累计亏损跨期累加，不因换期重置；第 4 次投注必须被限额拦住。"""
    t = pb.BudgetTracker(loss_limit=500)
    for i in range(3):
        assert t.reserve("t{}".format(i), 132) is True
        t.settle("t{}".format(i), -132)
    assert t.loss_to_date == 396
    # 396 + 132 = 528 > 500 -> 第 4 个单元被拒绝（这正是限额生效的证据）
    assert t.can_allocate(132) is False
    assert t.reserve("t3", 132) is False
    t.settle("t3", -132)                    # 未登记的票不可结算
    assert t.loss_to_date == 396
    assert t.realized_net == -396


# ---------------------------------------------------------------------------
# 4. 与 0/1/2/3 组政策的衔接
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("groups,expected",
                         [(0, True), (1, True), (2, True), (3, True), (4, False)])
def test_group_counts_against_500_limit(groups, expected):
    """500 元在零回收压力下只能容纳 3 个 132 元单元（396 ≤ 500 < 528）。"""
    cost = 132 * groups
    assert pb.can_allocate(0, 0, cost, 500) is expected


@pytest.mark.parametrize("limit,expect_groups", [(500, 3), (1000, 7)])
def test_max_affordable_groups_matches_plan_text(limit, expect_groups):
    """PLAN §5：500 元容纳 3 个单元，1000 元容纳 7 个。"""
    assert pb.max_affordable(0, 0, limit, 132) == expect_groups


def test_zero_groups_is_a_valid_choice_with_zero_profit():
    """0 组（跳过）是合法决策：利润 0，ROI 未定义。"""
    t = pb.BudgetTracker(loss_limit=500)
    assert t.can_allocate(0) is True
    assert pb.max_affordable(0, 0, 0, 132) == 0
    assert t.unsettled_cost == 0 and t.committed == 0


def test_no_candidate_does_not_force_participation():
    """预算允许 ≠ 必须参与；限额不足时必须给出"不投"这一合法结论。"""
    t = pb.BudgetTracker(loss_limit=100)
    assert t.can_allocate(132) is False
    assert t.max_affordable_groups() == 0
    assert t.reserve("x", 132) is False
    assert t.realized_net == 0            # 未参与 -> 无盈亏

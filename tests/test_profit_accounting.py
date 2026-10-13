# -*- coding: utf-8 -*-
"""V2 T2/D2 定向测试：完整净收益账本、税务边界、账户隔离、来源标签。

运行： python -m pytest tests/test_profit_accounting.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                                     # noqa: E402
from tools import profit_accounting as pa                    # noqa: E402
from tools.five_plus12_math import COST_PER_GROUP            # noqa: E402

SNAPSHOT_OK = pr.get_snapshot().ok


def _tbl(**kw):
    """构造只含指定 (f,b) 奖金的最小表。键形如 ``f3_b2=(3,2)``。"""
    out = {}
    for k, v in kw.items():
        head, back = k.split("_b")
        out[(int(head.lstrip("f")), int(back))] = v
    return out


# ---------------------------------------------------------------------------
# 1. 税务：10000 边界与"全额计税"
# ---------------------------------------------------------------------------

def test_tax_threshold_boundary_exactly_10000():
    """一次中奖收入恰为 10000 元：暂免。10001 元：全额计税。"""
    t = _tbl(f3_b2=10_000)
    r = pa.settle_period(99999, [3], rule_snapshot={"table": t, "label": "test"},
                         tax_policy=pa.TAX_2024_MERGED)
    assert r["gross"] == 10_000 and r["tax"] == 0

    t2 = _tbl(f3_b2=10_001)
    r2 = pa.settle_period(99999, [3], rule_snapshot={"table": t2, "label": "test"},
                          tax_policy=pa.TAX_2024_MERGED)
    assert r2["gross"] == 10_001
    assert r2["tax"] == int(10_001 * 0.20) == 2000


def test_tax_is_on_full_amount_not_only_the_excess():
    """超过起征点时按**全额**计税，不是只对超过 10000 的部分计税。"""
    t = _tbl(f3_b2=50_000)
    r = pa.settle_period(99999, [3], rule_snapshot={"table": t, "label": "test"},
                         tax_policy=pa.TAX_2024_MERGED)
    assert r["tax"] == 10_000                      # 50_000 × 20%
    assert r["tax"] != int((50_000 - 10_000) * 0.20)   # 不是 8_000


def test_net_cost_profit_identity():
    t = _tbl(f3_b2=50_000)
    r = pa.settle_period(99999, [3], rule_snapshot={"table": t, "label": "test"},
                         tax_policy=pa.TAX_2024_MERGED)
    assert r["net"] == r["gross"] - r["tax"]
    assert r["profit"] == r["net"] - r["cost"]
    assert r["cost"] == COST_PER_GROUP * 1


# ---------------------------------------------------------------------------
# 2. 跨组合并计税
# ---------------------------------------------------------------------------

def test_same_issue_groups_are_merged_for_tax():
    """同一账户同一期的全部组必须**合并**判定一次中奖收入。"""
    t = _tbl(f3_b2=6_000)
    merged = pa.settle_period(99999, [3, 3], rule_snapshot={"table": t, "label": "t"},
                              tax_policy=pa.TAX_2024_MERGED)
    assert merged["gross"] == 12_000
    assert merged["tax"] == 2_400                     # 合并后超 10000，全额计税

    single = pa.settle_period(99999, [3], rule_snapshot={"table": t, "label": "t"},
                              tax_policy=pa.TAX_2024_MERGED)
    assert single["tax"] == 0                         # 单组未超起征点
    assert merged["tax"] > 2 * single["tax"]           # 合并不是简单相加


def test_mutually_exclusive_strategies_are_separate_accounts():
    """互斥策略必须使用不同账户，奖金不合并计税。"""
    t = _tbl(f3_b2=6_000)
    a = pa.SimAccount("A", snapshot=None)
    b = pa.SimAccount("B", snapshot=None)
    ra = a.settle(99999, [3], rule_snapshot={"table": t, "label": "t"})
    rb = b.settle(99999, [3], rule_snapshot={"table": t, "label": "t"})
    assert ra["tax"] == 0 and rb["tax"] == 0          # 各自未达起征点
    assert a.totals()["gross"] == 6_000 and b.totals()["gross"] == 6_000


def test_cross_period_winnings_are_not_merged():
    """不同期的奖金**不**合并（一次中奖收入按期界定）。"""
    t = _tbl(f3_b2=6_000)
    acc = pa.SimAccount("A", snapshot=None)
    acc.settle(99999, [3], rule_snapshot={"table": t, "label": "t"})
    acc.settle(99998, [3], rule_snapshot={"table": t, "label": "t"})
    assert acc.totals()["tax"] == 0
    assert [r["tax"] for r in acc.ledger()] == [0, 0]


# ---------------------------------------------------------------------------
# 3. 历史税制与来源标签
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_current_issue_uses_official_tax_policy():
    snap = pr.get_snapshot()
    pol, cert, why = pa.resolve_tax_policy(26116, snapshot=snap)
    assert cert == "official" and pol.threshold == 10_000 and pol.rate == pytest.approx(0.20)


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_historical_issue_tax_certainty_is_flagged_uncertain():
    """2024-09-01 之前的期次：历史税制未取得条款级依据，必须标记。"""
    snap = pr.get_snapshot()
    pol, cert, why = pa.resolve_tax_policy(23001, snapshot=snap)   # 2023 年
    assert cert == "uncertain"
    assert "未" in why or "未经" in why
    r = pa.settle_period(23001, [3], snapshot=snap)
    assert r["tax_certainty"] == "uncertain"


def test_issue_without_draw_date_is_unknown_certainty():
    pol, cert, why = pa.resolve_tax_policy(99999, snapshot=None)
    assert cert == "unknown"


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_official_issue_is_labeled_actual_not_estimate():
    snap = pr.get_snapshot()
    r = pa.settle_period(25001, [3], snapshot=snap)
    assert r["labels"]["prize_source"] == pa.SRC_OFFICIAL
    assert r["labels"]["is_estimate"] is False
    assert r["labels"]["warning"] == ""


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_missing_snapshot_falls_back_to_base_and_is_labeled_estimate():
    """官方快照未覆盖的期次必须降级为基础档**估计**，且不得标为官方实测。"""
    snap = pr.get_snapshot()
    future = max(snap.issues) + 1
    r = pa.settle_period(future, [3], snapshot=snap)
    assert r["labels"]["prize_source"] == pa.SRC_BASE_ESTIMATE
    assert r["labels"]["is_estimate"] is True
    assert "不得" in r["labels"]["warning"]


def test_legacy_table_is_labeled_static_not_official():
    r = pa.settle_period(25001, [3], rule_snapshot="legacy_config")
    assert r["labels"]["prize_source"] == pa.SRC_LEGACY_STATIC
    assert r["labels"]["is_estimate"] is True


# ---------------------------------------------------------------------------
# 4. 费用与派奖资格
# ---------------------------------------------------------------------------

def test_extra_costs_are_added_and_itemised():
    t = _tbl(f3_b2=1_000)
    r = pa.settle_period(99999, [3, 3, 3], rule_snapshot={"table": t, "label": "t"},
                         tax_policy=pa.TAX_2024_MERGED,
                         costs={"execution_fee": 7, "other": 3})
    assert r["cost_breakdown"]["bet"] == 3 * 132
    assert r["cost_breakdown"]["execution_fee"] == 7
    assert r["cost"] == 3 * 132 + 10
    assert r["profit"] == r["net"] - r["cost"]


@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_bonus_eligibility_affects_table_and_label():
    snap = pr.get_snapshot()
    issue = 26051                                   # 2026 派奖期 + 奖池 >= 8 亿
    on = pa.settle_period(issue, [3], snapshot=snap, bonus_eligible=True)
    off = pa.settle_period(issue, [3], snapshot=snap, bonus_eligible=False)
    assert on["gross"] > off["gross"]
    assert off["labels"]["bonus_eligible"] is False
    assert off["labels"]["prize_source"].endswith("+no_bonus")
    assert on["bonus_component"] is not None and on["bonus_component"] > 0
    assert on["gross_no_bonus"] == off["gross"]


# ---------------------------------------------------------------------------
# 5. 账户汇总与风险指标
# ---------------------------------------------------------------------------

def test_totals_drawdown_from_zero_and_loss_streak():
    t = _tbl(f3_b2=1_000)
    acc = pa.SimAccount("A", snapshot=None)
    for hits in ([0], [0], [3], [0]):               # 回收只有第 3 期有
        acc.settle(99999, hits, rule_snapshot={"table": t, "label": "t"},
                   tax_policy=pa.TAX_2024_MERGED)
    tot = acc.totals()
    assert tot["n_periods"] == 4
    assert tot["profit"] == 1_000 - 4 * 132
    # 回撤必须从 0 起算，包含第 1 期亏损
    assert tot["max_drawdown_from_zero"] == 2 * 132
    assert tot["max_consecutive_loss"] == 2          # 前两期；第 3 期盈利中断


def test_unknown_project_costs_are_listed_as_gap_not_zero():
    acc = pa.SimAccount("A", snapshot=None)
    acc.add_external_cost(0, "研发工时", note="未结算", known=False)
    acc.settle(99999, [3], rule_snapshot={"table": _tbl(f3_b2=1_000), "label": "t"},
               tax_policy=pa.TAX_2024_MERGED)
    tot = acc.totals()
    assert tot["project_unknown_cost_items"]          # 非空：缺口被列出
    assert tot["project_known_external_cost"] == 0
    assert "未结算" in tot["project_profit_gap"] or "研发工时" in tot["project_profit_gap"]


def test_roi_undefined_when_no_cost():
    acc = pa.SimAccount("A", snapshot=None)
    r = acc.settle(99999, [], rule_snapshot={"table": _tbl(f3_b2=1_000), "label": "t"},
                   tax_policy=pa.TAX_2024_MERGED)
    assert r["cost"] == 0 and r["roi"] is None and r["profit"] == 0


# ---------------------------------------------------------------------------
# 6. 计奖实现的重复实现一致性（PLAN §2 风险 4）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SNAPSHOT_OK, reason="缺少官方奖金快照")
def test_duplicate_prize_implementations_agree():
    """prize_rules.py 与 tools/dlt_prize_rules.py 必须给出一致的奖金表。"""
    from tools import dlt_prize_rules as dpr
    assert dpr.LEVEL_CONDITIONS == pr.LEVEL_CONDITIONS
    assert dpr.BASE_AMOUNT == pr.BASIC_AMOUNT
    assert dpr.ERA3_POOL_THRESHOLD == pr.ERA3_POOL_THRESHOLD
    assert dpr.ERA3_HIGH_AMOUNT == pr.ERA3_HIGH_AMOUNT

    snap = pr.get_snapshot()
    od = dpr.load_official(os.path.join(
        ROOT, "outputs", "2026-10-13_5plus12_profitability_verification"))
    for issue in (25001, 26001, 26014, 26050, 26051, 26116):
        if not snap.has(issue):
            continue
        for bonus_mode, our_mode in (("full", "official"),
                                     ("none", "official_no_bonus")):
            a = pr.get_prize_table(issue, mode=our_mode)
            b = od.issue_table(issue, bonus_mode)
            assert a == b, (issue, bonus_mode)

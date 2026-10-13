# -*- coding: utf-8 -*-
"""完整净收益账本（V2 T2 / D2）。

三本账（PLAN §4.1）
------------------
1. **税前回收 `gross`**：所有奖级 + 符合资格的派奖。
2. **投注净利润 `net_profit`**：同一期**同一模拟账户**全部组奖金合计，
   扣适用税款，再减 132×组数与实际执行费用。
3. **项目净利润**：累计投注净利润再减实际研发/数据/运行现金支出；
   未知费用**列缺口**，不虚构、不默认为零。

严格分离
--------
* 本模块只做**开奖后结算**。开奖前的机会估计见 `tools/profit_opportunity.py`。
* 每期必须记录**奖金来源标签**：官方逐期实测 / 规则基础档估计 / 历史静态表。
  估算回退**不得**标为官方实测。

税务口径（已核验，见 `sources/TAX_10044631.html.html`）
------------------------------------------------------
《财政部 税务总局 民政部 体育总局公告 2024 年第 12 号》，**自 2024-09-01 起执行**：

* 「电脑彩票以同一人在同一期同一游戏中获得的全部奖金为一次中奖收入」；
* 「彩票一次中奖收入不超过 10000 元暂免征收个人所得税」（1998/1994 年政策延续）；
* 「彩票中奖收入属于偶然所得，以**每次收入额为应纳税所得额**」 → 超过 10000 元时
  **全额**计税，不是只对超额部分计税；
* 偶然所得税率 20%。

**历史期次**：2024-09-01 之前的期次，"一次中奖收入"如何界定（按票 / 按期合并）
在本轮未取得条款级依据，标记为 `uncertain`，并按两种口径分别出具。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr
from tools.five_plus12_math import BACK_HIT_TICKETS, COST_PER_GROUP

# 来源标签（写入账本，供审计）
SRC_OFFICIAL = "official_actual"          # 官方逐期公布的单注奖金（实测）
SRC_BASE_ESTIMATE = "base_estimate"       # 规则基础档（估计，官方未覆盖）
SRC_LEGACY_STATIC = "legacy_static"       # 旧静态表（已知有误，仅供复现）
SRC_EXPLICIT = "explicit_table"           # 调用方显式提供的表

TAX_SOURCE_2024 = (
    "财政部 税务总局 民政部 体育总局公告2024年第12号（2024-08-15 发布，"
    "自 2024-09-01 起执行）"
)
TAX_EFFECTIVE_DATE = "2024-09-01"


# ---------------------------------------------------------------------------
# 税务政策
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaxPolicy:
    """一次中奖收入的免税额与税率。"""
    name: str
    threshold: int = 10_000
    rate: float = 0.20
    combine: str = "same_person_same_issue_same_game"
    certainty: str = "official"        # official | uncertain | assumed
    source: str = TAX_SOURCE_2024
    note: str = ""

    def tax_on(self, gross: int) -> int:
        """按"全额计税、10000 元起征"计算税额（向下取整到元）。"""
        if gross <= self.threshold:
            return 0
        return int(gross * self.rate)


NO_TAX = TaxPolicy(name="no_tax_legacy", threshold=10 ** 18, rate=0.0,
                   certainty="assumed",
                   source="显式关闭税（仅用于复现旧税前口径）",
                   note="这不是法定口径，只用于对照。")

TAX_2024_MERGED = TaxPolicy(name="cn_lottery_2024_merged", certainty="official")

TAX_PRE_2024_PER_TICKET = TaxPolicy(
    name="cn_lottery_pre2024_perticket_ASSUMED", certainty="uncertain",
    source="未取得条款级依据（2024 年第 12 号公告自 2024-09-01 起执行）",
    note="2024-09-01 之前的期次，'一次中奖收入'是否按期合并未经核验；"
         "此口径按每张票（=每组 66 注）分别判定起征点，属假设。")

TAX_PRE_2024_MERGED = TaxPolicy(
    name="cn_lottery_pre2024_merged_UNCERTAIN", certainty="uncertain",
    source="未取得条款级依据",
    note="与现行口径同形，但历史适用性未经核验。")


def _issue_dates(snapshot):
    """{issue: 'YYYY-MM-DD'}，来自官方快照；缺失则返回空。"""
    if snapshot is None or not getattr(snapshot, "ok", False):
        return {}
    d = getattr(snapshot, "dates", None)
    if d:
        return d
    try:                                        # 兼容旧版快照对象
        return {int(r.issue): str(r.draw_date) for r in snapshot.draws.itertuples()}
    except Exception:
        return {}


def resolve_tax_policy(issue: int, snapshot=None, as_of: Optional[str] = None):
    """按**兑奖时点**（以开奖日为代理）返回适用税制。

    返回 (policy, certainty, rationale)。
    """
    if as_of is None and snapshot is not None and getattr(snapshot, "ok", False):
        as_of = _issue_dates(snapshot).get(int(issue))
    if as_of is None:
        return (TAX_PRE_2024_MERGED, "unknown",
                "该期无官方开奖日期，无法判定适用税制；按 2024 年前的合并口径出具，"
                "并标记为未核实。")
    if str(as_of) >= TAX_EFFECTIVE_DATE:
        return (TAX_2024_MERGED, "official", TAX_SOURCE_2024)
    return (TAX_PRE_2024_MERGED, "uncertain",
            "开奖日 {} 早于 {}，'一次中奖收入'的界定未经条款级核验；"
            "另行按每组分别判定起征点的假设口径给出对照。".format(
                as_of, TAX_EFFECTIVE_DATE))


# ---------------------------------------------------------------------------
# 单期结算
# ---------------------------------------------------------------------------

def _table_and_label(issue: int, rule_snapshot, snapshot):
    """返回 (table, label, is_estimate, by_level_map)。"""
    if isinstance(rule_snapshot, dict) and "table" in rule_snapshot:
        return (dict(rule_snapshot["table"]),
                rule_snapshot.get("label", SRC_EXPLICIT),
                bool(rule_snapshot.get("is_estimate", True)),
                rule_snapshot.get("by_level"))
    mode = rule_snapshot or "official"
    # 服务器交接勘误（V3_HANDOFF_REVIEW_2026-10-14.md）：显式传入的 snapshot
    # 必须一路传到计奖表，否则会静默回落到 data/dlt/ 下的旧默认快照。
    tbl = pr.get_prize_table(int(issue), mode=mode, snapshot=snapshot)
    if mode == "legacy_config":
        return tbl, SRC_LEGACY_STATIC, True, None
    if mode.startswith("official") and snapshot is not None and getattr(snapshot, "ok", False):
        if snapshot.has(int(issue)):
            return tbl, SRC_OFFICIAL, False, None
    return tbl, SRC_BASE_ESTIMATE, True, None


def settle_period(issue: int,
                  group_hits: Sequence[int],
                  rule_snapshot="official",
                  tax_policy: Optional[TaxPolicy] = None,
                  costs: Optional[Dict[str, int]] = None,
                  snapshot=None,
                  account_id: str = "default",
                  bonus_eligible: bool = True,
                  group_back_hits: Optional[Sequence[Dict[int, int]]] = None,
                  ) -> dict:
    """结算**一个模拟账户在某一期**的全部组。

    Args:
        issue: 目标期号。
        group_hits: 每组的**前区命中数**（0..5）。后区按 1/20/45 恒等式展开。
        rule_snapshot: "official" / "base" / "legacy_config"，
            或 {"table": {...}, "label": ..., "is_estimate": bool}。
        tax_policy: 见 ``TaxPolicy``；None 时按 ``resolve_tax_policy`` 自动判定。
        costs: 额外费用，如 {"execution_fee": 0, "other": 0}；
            票款按 132×组数**自动**计入，不需要也不应重复填写。
        account_id: 模拟账户标识。**互斥策略必须使用不同账户**，
            不同账户的奖金不合并计税。
        bonus_eligible: 本账户当期的票是否满足派奖参与门槛
            （单张单期 ≥ 18 元；一组 5+12 = 132 元满足）。
            为 False 时按"不计派奖"的下界口径出账，并标记。

    Returns:
        dict，包含 gross/tax/net/cost/profit、按奖级分解、以及来源与估算标签。
    """
    snapshot = snapshot if snapshot is not None else pr.get_snapshot()
    issue = int(issue)
    hits = [int(h) for h in group_hits]
    n_groups = len(hits)

    if tax_policy is None:
        tax_policy, tax_certainty, tax_rationale = resolve_tax_policy(
            issue, snapshot=snapshot)
    else:
        tax_certainty, tax_rationale = tax_policy.certainty, tax_policy.note

    # 派奖资格 -> 影响取表口径
    if not bonus_eligible:
        mode = "official_no_bonus"
    else:
        mode = rule_snapshot
    table, label, is_estimate, _ = _table_and_label(issue, mode, snapshot)
    if not bonus_eligible and isinstance(rule_snapshot, str) and rule_snapshot.startswith("official"):
        label = label + "+no_bonus"
        is_estimate = is_estimate

    # 不计派奖的对照表（用于分离 base / bonus 贡献）
    if snapshot is not None and getattr(snapshot, "ok", False) and snapshot.has(issue):
        table_nb = pr.get_prize_table(issue, mode="official_no_bonus", snapshot=snapshot)
        has_bonus_info = True
    else:
        table_nb = table
        has_bonus_info = False

    by_level: Dict[str, int] = {}
    by_level_nb: Dict[str, int] = {}
    gross = 0
    gross_nb = 0

    for f in hits:
        for b, cnt in BACK_HIT_TICKETS.items():
            v = int(table.get((f, b), 0)) * cnt
            v_nb = int(table_nb.get((f, b), 0)) * cnt
            gross += v
            gross_nb += v_nb
            lv = pr.level_of(issue, f, b) or "未中奖"
            by_level[lv] = by_level.get(lv, 0) + v
            by_level_nb[lv] = by_level_nb.get(lv, 0) + v_nb

    bet_cost = COST_PER_GROUP * n_groups
    extra = dict(costs or {})
    extra_total = sum(int(v) for v in extra.values())
    total_cost = bet_cost + extra_total

    tax = tax_policy.tax_on(gross)
    net = gross - tax
    profit = net - total_cost

    return {
        "account_id": account_id,
        "issue": issue,
        "n_groups": n_groups,
        "group_hits": hits,
        # --- 第 1 本账：税前回收 ---
        "gross": gross,
        "gross_by_level": by_level,
        "gross_no_bonus": gross_nb,
        "bonus_component": gross - gross_nb if has_bonus_info else None,
        "has_bonus_info": has_bonus_info,
        # --- 第 2 本账：投注净利润 ---
        "tax": tax,
        "tax_policy": tax_policy.name,
        "tax_threshold": tax_policy.threshold,
        "tax_rate": tax_policy.rate,
        "tax_certainty": tax_certainty,
        "tax_rationale": tax_rationale,
        "net": net,
        "cost": total_cost,
        "cost_breakdown": {"bet": bet_cost, **extra},
        "profit": profit,
        "roi": (profit / total_cost) if total_cost else None,
        # --- 标签与来源 ---
        "labels": {
            "prize_source": label,
            "is_estimate": bool(is_estimate),
            "bonus_eligible": bool(bonus_eligible),
            "official_snapshot_available": bool(
                snapshot is not None and getattr(snapshot, "ok", False)
                and snapshot.has(issue)),
            "warning": ("奖金表回退为规则基础档估计，**不得**当作官方实测"
                        if is_estimate else ""),
        },
    }


# ---------------------------------------------------------------------------
# 模拟账户：累计账本
# ---------------------------------------------------------------------------

class SimAccount:
    """一个模拟账户的累计账本。

    互斥策略（同一笔预算不能同时投注的方案）必须使用**不同账户**：
    它们的奖金不合并计税，也不共享亏损限额。
    """

    def __init__(self, account_id: str, loss_limit: Optional[int] = None,
                 snapshot=None, tax_policy=None):
        self.account_id = account_id
        self.loss_limit = loss_limit
        self.snapshot = snapshot if snapshot is not None else pr.get_snapshot()
        self.tax_policy = tax_policy
        self.rows: List[dict] = []
        self.external_costs: List[dict] = []

    def settle(self, issue, group_hits, rule_snapshot="official", costs=None,
               bonus_eligible=True, tax_policy=None, **kw) -> dict:
        row = settle_period(issue, group_hits, rule_snapshot=rule_snapshot,
                            tax_policy=tax_policy or self.tax_policy, costs=costs,
                            snapshot=self.snapshot, account_id=self.account_id,
                            bonus_eligible=bonus_eligible, **kw)
        self.rows.append(row)
        return row

    def add_external_cost(self, amount: int, kind: str, note: str = "",
                          known: bool = True):
        """记录**项目级**现金支出（研发/数据/运行）。未知则 known=False 并计入缺口。"""
        self.external_costs.append({"amount": int(amount), "kind": kind,
                                    "note": note, "known": bool(known)})

    # -- 汇总 ----------------------------------------------------------
    def totals(self) -> dict:
        gross = sum(r["gross"] for r in self.rows)
        tax = sum(r["tax"] for r in self.rows)
        cost = sum(r["cost"] for r in self.rows)
        profit = sum(r["profit"] for r in self.rows)
        cum, peak, mdd, streak, worst = [], 0, 0, 0, 0
        cur = 0
        for r in self.rows:
            cur += r["profit"]
            cum.append(cur)
            peak = max(peak, cur)
            mdd = max(mdd, peak - cur)
            if r["profit"] < 0:
                streak += 1
                worst = max(worst, streak)
            else:
                streak = 0
        known_ext = sum(c["amount"] for c in self.external_costs if c["known"])
        unknown_ext = [c for c in self.external_costs if not c["known"]]
        return {
            "account_id": self.account_id,
            "n_periods": len(self.rows),
            "n_groups_total": sum(r["n_groups"] for r in self.rows),
            "gross": gross,
            "tax": tax,
            "net": gross - tax,
            "bet_cost": cost,
            "profit": profit,
            "roi": (profit / cost) if cost else None,
            "max_drawdown_from_zero": mdd,
            "max_consecutive_loss": worst,
            "cumulative": cum,
            "project_known_external_cost": known_ext,
            "project_unknown_cost_items": unknown_ext,
            "project_profit_known_only": profit - known_ext,
            "project_profit_gap": ("未列入的费用项目：{}".format(
                [c["kind"] for c in unknown_ext]) if unknown_ext else ""),
            "profit_positive_periods": sum(1 for r in self.rows if r["profit"] > 0),
            "periods": len(self.rows),
        }

    def ledger(self) -> List[dict]:
        return self.rows

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"totals": self.totals(),
                       "external_costs": self.external_costs,
                       "rows": self.rows}, f, ensure_ascii=False, indent=2)

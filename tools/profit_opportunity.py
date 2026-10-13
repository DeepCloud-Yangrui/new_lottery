# -*- coding: utf-8 -*-
"""开奖前奖金机会估计与盈利缺口（V2 T3 / D3）。

硬约束（PLAN §4.2）
-------------------
1. **只允许使用 `as_of` 之前已公开的信息**。目标期的实际中奖注数、实际单注奖金、
   开奖号码一律不得进入机会估计，只用于事后结算
   （见 ``tools/profit_accounting.py``）。
2. 浮动奖（一、二等奖）**不得**用目标期的未来实际值，也不得用固定大奖常数选期。
   本模块要求调用方给出**情景**，并且每个情景必须声明它是由**哪些已开奖期次**
   构造的；如果任一来源期次 >= 目标期号，直接拒绝。
3. 缺少销售额 / 中奖注数数据时，只做**明确的分摊敏感性与盈亏平衡阈值**，
   不编造"防撞号"模型。

估计内容
--------
给定
  * 前区命中分布（默认 = 均匀基线的精确超几何），
  * 若干浮动奖情景（每个情景带来一组 (一等奖单注, 二等奖单注)），
  * 适用税制与费用，
计算**该期净收益的完整分布**（不是只算均值）：

* 固定奖部分由 `prize_rules` 按目标期号 + 开奖前奖池档位给出（开奖前可得）；
* 多组使用**同一期共同开奖**的相关结构（``pool_size`` 指定是否共用候选池）；
* 按每期**合并**后的奖金判定税额（≤10000 暂免，>10000 全额 20%）；
* 输出每个情景的 E[净收益]、区间、P(净收益>0)、以及盈亏平衡所需的命中率偏移。

结论形态
--------
``verdict`` 只会是以下之一：
  * ``no_positive_scenario``：所有情景的期望净收益都 ≤ 0 → **无候选**；
  * ``positive_but_fragile``：存在正的情景，但依赖单一情景或极端尾部；
  * ``positive_under_all_scenarios``：所有情景都为正（极罕见，需额外审查假设）。
"""
from __future__ import annotations

import math
from math import comb
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr
from tools.five_plus12_math import (
    BACK_HIT_TICKETS, COST_PER_GROUP, front_hit_distribution,
    group_payout_by_front_hit,
)
from tools.profit_accounting import TaxPolicy, TAX_2024_MERGED, resolve_tax_policy

# ---------------------------------------------------------------------------
# 允许进入机会估计的字段（白名单）
# ---------------------------------------------------------------------------
ALLOWED_CONTEXT_KEYS = {
    "issue",                    # 目标期号
    "as_of",                    # 决策时点（期号或 ISO 日期）
    "pool_before",              # 目标期**开奖前**奖池余额（上一期开奖后公布）
    "bonus_rounding",           # 派奖金额精度："exact"（默认，保留半元）或
                                # "official_pernote_floor"（逐注向下取整，敏感性）
    "pool_before_source_issue",  # 该奖池来自哪一期的开奖（用于核对公开时间）
    "pool_before_published_at",  # 该值的公开日期（ISO），必须 <= as_of
    "rule_era",                 # 'era2' / 'era3'
    "bonus_active",             # 目标期是否处于派奖活动期内（活动期次表必须事先公布）
    "bonus_certainty",          # official / uncertain / assumed
    "bonus_eligibility",        # 本账户票面是否满足派奖门槛
    "n_groups",                 # 拟投注组数
    "pool_size",                # 多组是否共用候选池（None=全号池）
    "sales_history",            # 历史销售额（用于分摊敏感性，非目标期实际值）
    "historical_winner_counts",  # 历史各奖级中奖注数（用于分摊敏感性）
    "notes",
}

# 明确禁止的字段形态（目标期的未来信息）
FORBIDDEN_KEY_PATTERNS = (
    "actual_", "realized_", "draw_result", "winning_numbers",
    "amt_", "cnt_",
)


class OpportunityInputError(ValueError):
    """输入包含未来信息或不可用字段。"""


def _norm_date(x, snapshot=None):
    """把 ISO 日期或期号归一化为 'YYYY-MM-DD'；无法解析返回 None。"""
    if x is None:
        return None
    s = str(x).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if s.isdigit() and snapshot is not None and getattr(snapshot, "ok", False):
        return getattr(snapshot, "dates", {}).get(int(s))
    return None


def _issue_date(issue, snapshot):
    if snapshot is None or not getattr(snapshot, "ok", False):
        return None
    return getattr(snapshot, "dates", {}).get(int(issue))


def validate_as_of(context, scenarios, snapshot, target_issue):
    """**真正的 as_of 校验**：决策之后才公开的信息一律拒绝。

    规则：
      1. ``context['as_of']`` 必须能解析为日期（ISO 或期号）；
      2. 每个浮动奖情景的 ``source_issues`` 必须**全部早于目标期**，
         且其开奖日期必须 **<= as_of**（否则该情景在决策时还不存在）；
      3. ``pool_before`` 必须声明来源期次 ``pool_before_source_issue``，
         该期开奖日期必须 **<= as_of**；
      4. 若提供 ``pool_before_published_at``，必须 **<= as_of**。

    返回一份可审计的时间线；违反则抛出 ``OpportunityInputError``。
    """
    as_of = _norm_date(context.get("as_of"), snapshot)
    if as_of is None:
        raise OpportunityInputError(
            "as_of 无法解析为日期（应为 ISO 日期或期号）：{!r}".format(
                context.get("as_of")))

    timeline = {"as_of": as_of, "scenarios": [], "pool_before": None}

    for s in scenarios:
        src = [int(x) for x in s.get("source_issues", [])]
        pub = _norm_date(s.get("published_at"), snapshot)
        if pub is None:
            pub = _issue_date(max(src), snapshot) if src else None
        if pub is None:
            raise OpportunityInputError(
                "情景 {} 未声明 published_at，且无法由来源期次推出开奖日期；"
                "无法证明它在 as_of 之前可用".format(s.get("label")))
        if pub > as_of:
            raise OpportunityInputError(
                "情景 {} 的公开日期 {} 晚于 as_of {}（未来信息）".format(
                    s.get("label"), pub, as_of))
        newest = max(src) if src else None
        if newest is not None and newest >= int(target_issue):
            raise OpportunityInputError(
                "情景 {} 含目标期或之后的来源期次 {}".format(s.get("label"), newest))
        timeline["scenarios"].append({"label": s.get("label"), "published_at": pub,
                                      "newest_source_issue": newest})

    pb_src = context.get("pool_before_source_issue")
    if pb_src is not None:
        d = _issue_date(pb_src, snapshot)
        if d is None:
            raise OpportunityInputError(
                "pool_before_source_issue={} 无开奖日期，无法核对公开时间".format(pb_src))
        if d > as_of:
            raise OpportunityInputError(
                "pool_before 来源期次 {} 的开奖日期 {} 晚于 as_of {}".format(
                    pb_src, d, as_of))
        timeline["pool_before"] = {"source_issue": int(pb_src), "published_at": d}
    ppb = _norm_date(context.get("pool_before_published_at"), snapshot)
    if ppb is not None:
        if ppb > as_of:
            raise OpportunityInputError(
                "pool_before_published_at {} 晚于 as_of {}".format(ppb, as_of))
        timeline["pool_before_published_at"] = ppb
    return timeline


# ---------------------------------------------------------------------------
# 浮动奖情景
# ---------------------------------------------------------------------------

def make_floating_scenario(label: str,
                           first_prize: int,
                           second_prize: int,
                           source_issues: Sequence[int],
                           basis: str,
                           is_estimate: bool = True,
                           published_at: Optional[str] = None) -> dict:
    """构造一个"开奖前"可用的浮动奖情景。

    Args:
        source_issues: 该情景由哪些**已开奖**期次构造（必须全部早于目标期）。
        basis: 构造方式说明（如 "past 500 issues, tier=pool>=8e8, P50"）。
    """
    return {
        "label": label,
        "first_prize": int(first_prize),
        "second_prize": int(second_prize),
        "source_issues": [int(x) for x in source_issues],
        "basis": basis,
        "is_estimate": bool(is_estimate),
        # 该情景最早只能在其**最后一个来源期次开奖之后**才可用
        "published_at": published_at,
    }


def scenario_from_history(snapshot, target_issue: int, mask: str = "all",
                          q: float = 0.5, window: int = 500) -> dict:
    """用**目标期之前**的官方历史构造一个浮动奖情景（分位数）。

    绝不允许使用 target_issue 及之后的数据。
    """
    if snapshot is None or not getattr(snapshot, "ok", False):
        raise OpportunityInputError("缺少官方快照，无法构造历史浮动奖情景")
    d = snapshot.draws
    sub = d[(d.issue < int(target_issue))].tail(window)
    if mask == "pool_ge_8e8":
        pb = [snapshot.pool_before(int(i)) for i in sub.issue.tolist()]
        keep = [i for i, p in zip(sub.issue.tolist(), pb)
                if p is not None and p >= pr.ERA3_POOL_THRESHOLD]
        sub = sub[sub.issue.isin(keep)]
    if len(sub) == 0:
        raise OpportunityInputError("历史窗口为空，无法构造情景")
    a1 = float(sub["amt_一等奖"].quantile(q))
    a2 = float(sub["amt_二等奖"].quantile(q))
    newest = int(sub.issue.max())
    pub = _issue_date(newest, snapshot)
    return make_floating_scenario(
        "hist_q{:.2f}_{}_last{}".format(q, mask, window),
        round(a1), round(a2), sub.issue.tolist(),
        "目标期之前的 {} 期官方数据，mask={}，取分位数 q={:.2f}".format(
            len(sub), mask, q),
        published_at=pub)


# ---------------------------------------------------------------------------
# 分摊敏感性（不做"防撞号"模型）
# ---------------------------------------------------------------------------

def dilution_factor(winner_counts: Sequence[int]) -> dict:
    """我方多中 1 注时，该奖级单注奖金的稀释系数。

    若某期有 N 注中奖、单注奖金 A，则多 1 注后单注变为 ``A × N/(N+1)``
    （奖级总额固定时）。这里只做**敏感性**，不预测中奖注数。
    """
    ns = [int(n) for n in winner_counts if n is not None and n >= 1]
    if not ns:
        return {"n": 0, "note": "缺少中奖注数数据，无法计算分摊敏感性"}
    facs = [n / (n + 1.0) for n in ns]
    facs.sort()
    return {
        "n": len(ns),
        "min_factor": facs[0], "median_factor": facs[len(facs) // 2],
        "max_factor": facs[-1],
        "interpretation": "系数越小说明中奖注数越少、加 1 注的稀释越明显",
        "note": "仅分摊敏感性；未建立任何'避开热门号码'的模型。",
    }


# ---------------------------------------------------------------------------
# 核心：机会估计
# ---------------------------------------------------------------------------

def _validate_context(context: dict, target_issue: int):
    bad_keys = [k for k in context
                if k not in ALLOWED_CONTEXT_KEYS
                and any(p in k for p in FORBIDDEN_KEY_PATTERNS)]
    unknown = [k for k in context if k not in ALLOWED_CONTEXT_KEYS]
    rejected = []
    if bad_keys:
        rejected.append({"reason": "字段形态属于开奖后信息", "keys": bad_keys})
    if unknown:
        rejected.append({"reason": "未在白名单内的字段（需显式扩展白名单）",
                         "keys": unknown})
    if rejected:
        raise OpportunityInputError(
            "机会估计拒绝以下输入：{}".format(rejected))
    if "issue" in context and int(context["issue"]) != int(target_issue):
        raise OpportunityInputError("context.issue 与 target_issue 不一致")


def _validate_scenarios(scenarios: Sequence[dict], target_issue: int):
    if not scenarios:
        raise OpportunityInputError("至少需要一个浮动奖情景")
    for s in scenarios:
        src = [int(x) for x in s.get("source_issues", [])]
        if not src:
            raise OpportunityInputError(
                "情景 {} 未声明来源期次；无法证明它不使用未来数据".format(s.get("label")))
        future = [x for x in src if x >= int(target_issue)]
        if future:
            raise OpportunityInputError(
                "情景 {} 使用了目标期及之后的期次 {}（未来信息）".format(
                    s.get("label"), future[:5]))


def _hit_dist_from_uniform():
    return front_hit_distribution()


def _validate_hit_distribution(hd: Optional[dict]):
    if hd is None:
        return _hit_dist_from_uniform(), "uniform_hypergeometric"
    vals = {int(k): float(v) for k, v in hd.items()}
    if set(vals) != set(range(6)):
        raise OpportunityInputError("命中分布必须给出 f=0..5 全部概率")
    if abs(sum(vals.values()) - 1.0) > 1e-9:
        raise OpportunityInputError("命中分布未归一（sum={}）".format(sum(vals.values())))
    if any(v < 0 for v in vals.values()):
        raise OpportunityInputError("命中分布存在负概率")
    return vals, "provided"


def _outcome_distribution(R: Dict[int, int], n_groups: int,
                          pool_size: Optional[int], hd: Dict[int, float]):
    """返回 {净税前回收(一期全部组): 概率}，使用共同开奖的相关结构。"""
    outcomes = [(R[f], hd[f]) for f in range(6) if hd[f] > 0]
    if pool_size is None:
        dist = {0: 1.0}
        for _ in range(n_groups):
            nd = {}
            for a, pa in dist.items():
                for v, pv in outcomes:
                    nd[a + v] = nd.get(a + v, 0.0) + pa * pv
            dist = nd
        return dist, {"structure": "independent_groups_full_pool"}
    p = int(pool_size)
    ms = list(range(0, min(p, 5) + 1))
    pm = [comb(p, x) * comb(35 - p, 5 - x) / comb(35, 5) for x in ms]
    tot = sum(pm)
    pm = [x / tot for x in pm]
    final: Dict[int, float] = {}
    detail = {}
    for mi, m in enumerate(ms):
        ks = list(range(0, min(m, 5) + 1))
        pk = [comb(m, k) * comb(p - m, 5 - k) / comb(p, 5) for k in ks]
        ssum = sum(pk)
        pk = [v / ssum for v in pk]
        dist = {0: 1.0}
        for _ in range(n_groups):
            nd = {}
            for a, pa in dist.items():
                for ki, k in enumerate(ks):
                    nd[a + R[k]] = nd.get(a + R[k], 0.0) + pa * pk[ki]
            dist = nd
        for v, pr_v in dist.items():
            final[v] = final.get(v, 0.0) + pm[mi] * pr_v
        detail[str(m)] = {"prob": pm[mi],
                          "E_R_given_m": sum(pk[i] * R[ks[i]] for i in range(len(ks)))}
    return final, {"structure": "shared_pool_correlated", "by_m": detail}


def estimate_opportunity(as_of,
                         context: dict,
                         hit_distribution=None,
                         prize_scenarios: Sequence[dict] = (),
                         tax_policy: Optional[TaxPolicy] = None,
                         costs: Optional[dict] = None,
                         snapshot=None) -> dict:
    """开奖前的净收益机会估计。

    Args:
        as_of: 决策时点（期号或 ISO 日期字符串）。仅用于记录与税制判定。
        context: 见 ``ALLOWED_CONTEXT_KEYS``。
        hit_distribution: 可选的 {f: p}；默认均匀超几何（基线）。
        prize_scenarios: 由 ``make_floating_scenario`` / ``scenario_from_history``
            构造；**必须**声明来源期次且全部早于目标期。
        tax_policy: 默认按 ``resolve_tax_policy`` 判定。
        costs: 额外费用（票款按 132×组数自动计入）。

    Returns:
        dict：情景明细、区间、缺口、缺失信息、被拒绝项、结论。
    """
    snapshot = snapshot if snapshot is not None else pr.get_snapshot()
    target_issue = int(context["issue"])
    _validate_context(context, target_issue)
    scenarios = list(prize_scenarios)
    _validate_scenarios(scenarios, target_issue)
    timeline = validate_as_of(context, scenarios, snapshot, target_issue)
    hd, hd_label = _validate_hit_distribution(hit_distribution)

    n_groups = int(context.get("n_groups", 1))
    pool_size = context.get("pool_size", None)
    bonus_active = bool(context.get("bonus_active", False))
    bonus_cert = context.get("bonus_certainty", "uncertain")
    bonus_elig = bool(context.get("bonus_eligibility", True))
    pool_before = context.get("pool_before", None)
    if pool_before is None and getattr(snapshot, "ok", False):
        pool_before = snapshot.pool_before(target_issue)

    if tax_policy is None:
        tax_policy, tax_cert, tax_why = resolve_tax_policy(
            target_issue, snapshot=snapshot, as_of=context.get("as_of"))
    else:
        tax_cert, tax_why = tax_policy.certainty, tax_policy.note

    extra = dict(costs or {})
    extra_total = sum(int(v) for v in extra.values())
    period_cost = COST_PER_GROUP * n_groups + extra_total

    # 固定奖表：按目标期 + 开奖前奖池档位（开奖前可得）
    base = pr.base_table(target_issue, pool_before=pool_before)
    # 派奖：只有**事先公布的活动期**才允许计入；不确定则降级并标记
    bonus_applied = bonus_active and bonus_elig and bonus_cert == "official"
    # 本轮修复（V3 E1）：此前只计算 bonus_applied **标记**，从未把派奖金额写进
    # 奖金表——打开与关闭派奖的 E[gross] 完全相同。现真正叠加。
    bonus_rounding = context.get("bonus_rounding", "exact")
    if bonus_applied:
        from tools.bonus_feasibility_v3 import apply_bonus_to_table
        # 本模块是 float 口径的机会估计器；**精确 Decimal 口径**见
        # tools/bonus_feasibility_v3.py（V3 四格实验的主口径）。
        # 22.5 这类半元在 float 下可精确表示，不会引入舍入误差。
        base = {k: float(v) for k, v in
                apply_bonus_to_table(base, rounding=bonus_rounding).items()}
    fixed_source = "rule_base_table(target, pool_before={})".format(pool_before)
    if bonus_active and bonus_cert != "official":
        fixed_source += " + bonus(降级：资格未核实)"

    missing: List[str] = []
    if context.get("sales_history") is None:
        missing.append("目标期销售额（开奖前不可知，只能做情景）")
    if context.get("historical_winner_counts") is None:
        missing.append("各奖级中奖注数（无法精算浮动奖分摊，只能用历史情景）")
    if not getattr(snapshot, "ok", False):
        missing.append("官方快照（无法用历史构造浮动奖情景）")

    per_scenario = []
    for s in scenarios:
        table = dict(base)
        table[(5, 2)] = int(s["first_prize"])
        table[(5, 1)] = int(s["second_prize"])
        if not bonus_applied:
            # 不计派奖口径：用规则基础表已经不含派奖，这里只记录标记
            pass
        R = group_payout_by_front_hit(table)
        E_R = sum(hd[f] * R[f] for f in range(6))

        dist, struct = _outcome_distribution(R, n_groups, pool_size, hd)
        items = sorted(dist.items())
        net_vals, probs = [], []
        for gross, p in items:
            tax = tax_policy.tax_on(int(round(gross)))
            net_vals.append(int(round(gross)) - tax - period_cost)
            probs.append(p)
        e_profit = sum(v * p for v, p in zip(net_vals, probs))
        # 分位数
        cum, q05, q50, q95 = 0.0, None, None, None
        for v, p in zip(net_vals, probs):
            cum += p
            if q05 is None and cum >= 0.05:
                q05 = v
            if q50 is None and cum >= 0.50:
                q50 = v
            if q95 is None and cum >= 0.95:
                q95 = v
        p_pos = sum(p for v, p in zip(net_vals, probs) if v > 0)

        per_scenario.append({
            "label": s["label"],
            "first_prize": table[(5, 2)], "second_prize": table[(5, 1)],
            "source_issues_max": max(s["source_issues"]),
            "basis": s.get("basis", ""),
            "is_estimate": s.get("is_estimate", True),
            "expected_gross": E_R * n_groups,
            "expected_profit": e_profit,
            "expected_roi": (e_profit / period_cost) if period_cost else None,
            "q05_profit": q05, "median_profit": q50, "q95_profit": q95,
            "prob_profit_positive": p_pos,
            "outcome_structure": struct["structure"],
        })

    probs_pos = [x["expected_profit"] for x in per_scenario]
    all_pos = all(v > 0 for v in probs_pos)
    any_pos = any(v > 0 for v in probs_pos)
    if not any_pos:
        verdict = "no_positive_scenario"
    elif all_pos and len(per_scenario) >= 3:
        verdict = "positive_under_all_scenarios"
    else:
        verdict = "positive_but_fragile"

    return {
        "as_of": as_of,
        "target_issue": target_issue,
        "context_used": {k: context[k] for k in context},
        "as_of_timeline": timeline,
        "hit_distribution_source": hd_label,
        "hit_distribution": {str(k): hd[k] for k in sorted(hd)},
        "n_groups": n_groups,
        "pool_size": pool_size,
        "period_cost": period_cost,
        "cost_breakdown": {"bet": COST_PER_GROUP * n_groups, **extra},
        "fixed_prize_source": fixed_source,
        "pool_before": pool_before,
        "bonus_active": bonus_active,
        "bonus_applied": bonus_applied,
        "bonus_certainty": bonus_cert,
        "bonus_rounding": bonus_rounding,
        "tax_policy": tax_policy.name,
        "tax_certainty": tax_cert,
        "tax_rationale": tax_why,
        "scenarios": per_scenario,
        "verdict": verdict,
        "missing_information": missing,
        "notes": [
            "本估计只使用 as_of 之前公开的信息；目标期的实际中奖注数、"
            "实际单注奖金与开奖号码不得进入本函数。",
            "浮动奖情景必须声明来源期次，且全部早于目标期，否则被拒绝。",
            "缺少销售/中奖注数时只做分摊敏感性，不建立'防撞号'模型。",
        ],
    }

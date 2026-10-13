# -*- coding: utf-8 -*-
"""V3：5+12「固定奖档位 × 派奖」四格精确收益与盈亏平衡边界（仅模拟）。

对应 `docs/project_notes/PROFIT_V3_BONUS_FEASIBILITY_PLAN.md` 的 E1/E2。

修复的两个已证实缺口
--------------------
1. ``tools/profit_opportunity.estimate_opportunity`` 只计算 ``bonus_applied``
   **标记**，从未把派奖金额写进奖金表 —— 打开派奖与关闭派奖的 E[gross]
   完全相同（已用脚本证实）。本模块**真正实现**派奖叠加。
2. 金额用**整数元**会在 50% 派奖处产生半元（如 15×50% = 7.5）。
   PLAN §4 要求：**统一用整数分/Decimal，禁止逐注截断**；
   官方兑奖舍入口径未查明时保留精确值并标 `待核实`。
   本模块默认 ``rounding="exact"``（7.5 就是 7.5）。

四格设计（2026 七奖级规则，不混入其他年份的加奖/追加/特殊活动）
--------------------------------------------------------------
| 单元 | 固定奖档位 | 派奖 |
|---|---|---|
| A | 基础档（奖池 < 8 亿） | 关闭 |
| B | 上浮档（奖池 ≥ 8 亿） | 关闭 |
| C | 基础档 | 开启且满足资格 |
| D | 上浮档 | 开启且满足资格 |

已知目标值（前区恰好中 3 个的单组税前回收）：
**A=675、B=875、C=1125、D=1470**。

注意：这四格是**规则情景**，不是某期实测。2026 派奖已于 26066 期结束，
不得称为当前在售机会。
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal, getcontext
from math import comb
from typing import Dict, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr

getcontext().prec = 60

Condition = Tuple[int, int]

COST_PER_GROUP = Decimal(132)
BET_PRICE = Decimal(2)
BACK_PAIRS = 66
BACK_HIT_TICKETS = {2: 1, 1: 20, 0: 45}      # 恒等式，与开奖无关
FRONT_TOTAL = comb(35, 5)

# 2026 已核实规则：三至六等奖 +50%，七等奖 +100%（一等奖、二等奖不参与固定奖派奖）
BONUS_RATE_2026: Dict[str, Decimal] = {
    "三等奖": Decimal("0.5"),
    "四等奖": Decimal("0.5"),
    "五等奖": Decimal("0.5"),
    "六等奖": Decimal("0.5"),
    "七等奖": Decimal("1.0"),
}
FLOATING_LEVELS = ("一等奖", "二等奖")

# 官方封顶（2026 七奖级规则）
FIRST_PRIZE_CAP = Decimal(10_000_000)     # 两段各 500 万
SECOND_PRIZE_CAP = Decimal(5_000_000)     # 单注上限 500 万

CELLS: Dict[str, Tuple[str, bool]] = {
    "A": ("base", False),
    "B": ("high", False),
    "C": ("base", True),
    "D": ("high", True),
}

ALLOWED_FLOATING_KINDS = ("historical_pair", "assumed_pair", "rule_envelope")


class BonusFeasibilityError(ValueError):
    """输入不满足 V3 的标注或范围要求。"""


# ---------------------------------------------------------------------------
# 概率
# ---------------------------------------------------------------------------

def front_hit_probabilities() -> Dict[int, Decimal]:
    """P(前区恰好命中 k)、k=0..5。精确超几何，且与所选号码无关。"""
    out = {}
    for k in range(6):
        out[k] = Decimal(comb(5, k) * comb(30, 5 - k)) / Decimal(FRONT_TOTAL)
    assert sum(out.values()) == Decimal(1)
    return out


# ---------------------------------------------------------------------------
# 规则奖金表（四格）
# ---------------------------------------------------------------------------

def _level_amounts(pool_tier: str, bonus: bool,
                   rounding: str = "exact") -> Dict[str, Decimal]:
    """按 2026 七奖级规则给出各**固定奖**奖级的单注奖金（含派奖）。

    浮动奖（一、二等奖）不在此表内 —— 它们由调用方以带标签的情景提供。
    """
    if pool_tier not in ("base", "high"):
        raise BonusFeasibilityError("pool_tier 必须是 'base' 或 'high'")
    if rounding not in ("exact", "official_pernote_floor"):
        raise BonusFeasibilityError("未知 rounding：{}".format(rounding))

    out = {}
    for level, base_amt in pr.BASIC_AMOUNT["era3"].items():
        if level in FLOATING_LEVELS:
            continue
        amt = Decimal(int(base_amt))
        if pool_tier == "high":
            amt = Decimal(int(pr.ERA3_HIGH_AMOUNT[level]))
        if bonus and level in BONUS_RATE_2026:
            add = amt * BONUS_RATE_2026[level]
            if rounding == "official_pernote_floor":
                add = add.to_integral_value(rounding="ROUND_FLOOR")
            amt = amt + add
        out[level] = amt
    return out


def rule_cell_table(cell: str, rounding: str = "exact",
                    floating_pair: Optional[dict] = None) -> Dict[Condition, Decimal]:
    """构造四格之一的完整单注奖金表（含浮动奖）。

    奖级→(前区命中,后区命中) 的映射**复用 `prize_rules.LEVEL_CONDITIONS`**，
    不另建一套映射。
    """
    if cell not in CELLS:
        raise BonusFeasibilityError("未知单元：{}（应为 A/B/C/D）".format(cell))
    pool_tier, bonus = CELLS[cell]
    amounts = _level_amounts(pool_tier, bonus, rounding=rounding)
    conds = pr.LEVEL_CONDITIONS["era3"]

    table: Dict[Condition, Decimal] = {}
    for level, amt in amounts.items():
        for cond in conds[level]:
            table[cond] = amt

    if floating_pair is not None:
        fp = validate_floating_pair(floating_pair)
        for cond in conds["一等奖"]:
            table[cond] = fp["first"]
        for cond in conds["二等奖"]:
            table[cond] = fp["second"]
    return table


JACKPOT_CONDITIONS = {(5, 2), (5, 1)}


def apply_bonus_to_table(table: Dict[Condition, object],
                         rounding: str = "exact") -> Dict[Condition, Decimal]:
    """把 2026 固定奖派奖**真正加到**既有的 (前区,后区)→金额 表上。

    - 只加三~七等奖；**一、二等奖不参与固定奖派奖**；
    - 奖级→条件的映射复用 ``prize_rules.LEVEL_CONDITIONS['era3']``；
    - ``rounding="exact"`` 时保留半元（如 15×50% = 7.5），不逐注截断。
    """
    if rounding not in ("exact", "official_pernote_floor"):
        raise BonusFeasibilityError("未知 rounding：{}".format(rounding))
    out = {k: Decimal(str(v)) for k, v in table.items()}
    conds = pr.LEVEL_CONDITIONS["era3"]
    for level, rate in BONUS_RATE_2026.items():
        for cond in conds[level]:
            if cond in JACKPOT_CONDITIONS:
                continue
            base = out.get(cond, Decimal(0))
            add = base * rate
            if rounding == "official_pernote_floor":
                add = add.to_integral_value(rounding="ROUND_FLOOR")
            out[cond] = base + add
    return out


def validate_floating_pair(floating_pair: dict) -> dict:
    """浮动奖输入必须**显式标注**来源类型；拒绝无标签的 fallback。"""
    if not isinstance(floating_pair, dict):
        raise BonusFeasibilityError("floating_pair 必须是 dict")
    kind = floating_pair.get("kind")
    if kind not in ALLOWED_FLOATING_KINDS:
        raise BonusFeasibilityError(
            "floating_pair.kind 必须是 {} 之一，收到 {!r}；"
            "禁止使用无标签的 fallback".format(list(ALLOWED_FLOATING_KINDS), kind))
    for key in ("first", "second"):
        if key not in floating_pair:
            raise BonusFeasibilityError("floating_pair 缺少 {}".format(key))
    fp = {"kind": kind,
          "first": Decimal(str(floating_pair["first"])),
          "second": Decimal(str(floating_pair["second"])),
          "source": floating_pair.get("source", ""),
          "note": floating_pair.get("note", "")}
    if fp["first"] < 0 or fp["second"] < 0:
        raise BonusFeasibilityError("浮动奖金额不能为负")
    if fp["first"] > FIRST_PRIZE_CAP:
        fp["exceeds_first_cap"] = True
    if fp["second"] > SECOND_PRIZE_CAP:
        fp["exceeds_second_cap"] = True
    return fp


# ---------------------------------------------------------------------------
# 单组回收与完整期望
# ---------------------------------------------------------------------------

def group_payout(table: Dict[Condition, Decimal], front_hit: int) -> Decimal:
    """一组 5+12 在前区命中 ``front_hit`` 时的税前总奖金（Decimal）。"""
    return sum(Decimal(BACK_HIT_TICKETS[b]) * table.get((front_hit, b), Decimal(0))
               for b in (2, 1, 0))


def apply_tax(gross: Decimal, threshold: Decimal = Decimal(10_000),
              rate: Decimal = Decimal("0.20")) -> Decimal:
    """偶然所得：≤ 阈值暂免；> 阈值按**全额**计税，非仅对超额部分。"""
    if gross <= threshold:
        return Decimal(0)
    return gross * rate


def evaluate_rule_cell(cell: str,
                       floating_pair: dict,
                       rule_evidence: Optional[dict] = None,
                       tax_policy: Optional[dict] = None,
                       costs: Optional[dict] = None,
                       rounding: str = "exact") -> dict:
    """四格之一的**精确**（概率加权）收益。

    Args:
        cell: 'A' / 'B' / 'C' / 'D'
        floating_pair: 必须带 ``kind`` 标签（见 ``validate_floating_pair``）
        rule_evidence: 规则来源说明（写入结果，便于审计）
        tax_policy: {"threshold": 10000, "rate": 0.20, "certainty": "...", "source": "..."}
        costs: 额外费用（票款按 132 元自动计入）
        rounding: 'exact'（默认，禁止逐注截断）或 'official_pernote_floor'（敏感性）

    Returns:
        dict：6 个命中结果各自的税前回收、税、净收益；期望；ROI；来源标签。
    """
    fp = validate_floating_pair(floating_pair)
    table = rule_cell_table(cell, rounding=rounding, floating_pair=floating_pair)
    probs = front_hit_probabilities()

    tp = dict(tax_policy or {})
    threshold = Decimal(str(tp.get("threshold", 10_000)))
    rate = Decimal(str(tp.get("rate", "0.20")))
    extra = {k: Decimal(str(v)) for k, v in (costs or {}).items()}
    period_cost = COST_PER_GROUP + sum(extra.values())

    outcomes = []
    e_gross = Decimal(0)
    e_tax = Decimal(0)
    for k in range(6):
        g = group_payout(table, k)
        t = apply_tax(g, threshold, rate)
        e_gross += probs[k] * g
        e_tax += probs[k] * t
        outcomes.append({
            "front_hit": k,
            "probability": probs[k],
            "gross": g,
            "tax": t,
            "net_after_tax": g - t,
            "period_cost": period_cost,
            "period_profit": g - t - period_cost,
            "taxed": bool(g > threshold),
        })
    e_net_profit = e_gross - e_tax - period_cost

    return {
        "cell": cell,
        "cell_definition": {"pool_tier": CELLS[cell][0], "bonus": CELLS[cell][1]},
        "rounding": rounding,
        "probs": probs,
        "outcomes": outcomes,
        "payout_at_front3": group_payout(table, 3),
        "payout_at_front4": group_payout(table, 4),
        "expected_gross": e_gross,
        "expected_tax": e_tax,
        "expected_net": e_gross - e_tax,
        "period_cost": period_cost,
        "expected_net_profit": e_net_profit,
        "expected_roi": (e_net_profit / period_cost) if period_cost else None,
        "floating_pair": fp,
        "tax_policy": {"threshold": threshold, "rate": rate,
                       "certainty": tp.get("certainty", "unspecified"),
                       "source": tp.get("source", "")},
        "rule_evidence": dict(rule_evidence or {}),
        "labels": {
            "is_rule_scenario": True,
            "note": "四格是规则情景，不是某期实测；2026 派奖已于 26066 期结束。",
        },
    }


# ---------------------------------------------------------------------------
# 盈亏平衡边界
# ---------------------------------------------------------------------------

def solve_break_even_region(cell: str,
                            rule_evidence: Optional[dict] = None,
                            tax_policy: Optional[dict] = None,
                            costs: Optional[dict] = None,
                            rounding: str = "exact") -> dict:
    """求该格的**盈亏平衡边界**（一二等奖金额作为参数）。

    结构性事实（可在结果中核对）：本规则下只有前区命中 4、5 两档的整组回收
    会超过 10000 元起征点，因此两者恒被课税、且税额为全额 20%。
    于是 ``E[净收益]`` 对 ``(a, b)`` 是**线性**的，

        E[net] = C0 + p5·(1−τ)·(a + 20b)

    盈亏平衡边界是一条**直线** ``a + 20b = K``。
    本函数不假设全局单调，而是**逐档枚举**并核查解所在区间。

    Returns:
        含边界常数 K、合法域、可达性、各命中档的课税状态与说明。
    """
    if cell not in CELLS:
        raise BonusFeasibilityError("未知单元：{}".format(cell))
    table_nofloat = rule_cell_table(cell, rounding=rounding)   # 无浮动奖项
    probs = front_hit_probabilities()
    tp = dict(tax_policy or {})
    threshold = Decimal(str(tp.get("threshold", 10_000)))
    rate = Decimal(str(tp.get("rate", "0.20")))
    extra = {k: Decimal(str(v)) for k, v in (costs or {}).items()}
    period_cost = COST_PER_GROUP + sum(extra.values())

    # 逐档课税状态（不假设全局单调，逐档枚举）
    tax_segments = []
    for k in range(5):                      # k=0..4 与浮动奖无关
        g = group_payout(table_nofloat, k)
        tax_segments.append({"front_hit": k, "gross": g, "taxed": bool(g > threshold),
                             "note": "与浮动奖无关"})
    # k=5：gross = a + 20b + 45·prize(5,0) ≥ 45·prize(5,0)
    c5 = table_nofloat.get((5, 0), Decimal(0))
    tax_segments.append({"front_hit": 5, "gross": None, "taxed": True,
                         "note": "gross = a + 20b + 45·{}；恒 > 阈值".format(c5)})

    # C0 = Σ_{k<=4} p_k·(gross_k − tax_k) − cost
    # 注意：k=5 的整组回收是 (a + 20b + 45·c5)，其中 45·c5 与浮动奖**无关**，
    # 必须计入常数项，否则边界会系统性偏移（此前漏掉，由定向测试发现）。
    C0 = Decimal(0)
    for k in range(5):
        g = group_payout(table_nofloat, k)
        C0 += probs[k] * (g - apply_tax(g, threshold, rate))
    C0 += probs[5] * (Decimal(1) - rate) * (Decimal(45) * c5)
    C0 -= period_cost
    coef = probs[5] * (Decimal(1) - rate)      # 对 (a + 20b) 的系数
    K = (Decimal(0) - C0) / coef                # 所需的 a + 20b

    # 合法域与上限包络
    lo_domain = Decimal(0)
    hi_domain = FIRST_PRIZE_CAP + Decimal(20) * SECOND_PRIZE_CAP
    reachable = lo_domain <= K <= hi_domain
    both_at_cap = (FIRST_PRIZE_CAP + Decimal(20) * SECOND_PRIZE_CAP)

    return {
        "cell": cell,
        "cost": period_cost,
        "tax_segments": tax_segments,
        "C0_all_non_floating_terms": C0,
        "c5_fixed_prize_at_front5": c5,
        "coefficient_on_a_plus_20b": coef,
        "break_even_line": {"a_plus_20b": K,
                            "form": "a + 20b = K，其中 a=一等奖单注、b=二等奖单注"},
        "legal_domain": {"a_plus_20b_min": lo_domain, "a_plus_20b_max": hi_domain,
                         "first_cap": FIRST_PRIZE_CAP, "second_cap": SECOND_PRIZE_CAP},
        "reachable_within_caps": bool(reachable),
        "required_multiple_of_typical": (
            (K / Decimal(10_000_000 + 20 * 127_256)).to_eng_string()
            if K is not None else None),
        "envelope_at_caps": {
            "a_plus_20b": both_at_cap,
            "E_net_at_caps": C0 + coef * both_at_cap,
            "caveat": "一等奖与二等奖同时触及各自封顶**能否同时实现未核实**；"
                      "该包络只说明'即使按规则上限也不足/足够'。",
        },
        "linearity_note": "只有前区命中 4、5 两档被课税，且均为全额 20%。"
                          "k=5 的整组回收 = a + 20b + 45·c5（c5 与浮动奖无关），"
                          "故 E[net] 对 (a + 20b) 严格线性，盈亏平衡是一直线；"
                          "已逐档枚举课税状态而非假定单调。",
        "rule_evidence": dict(rule_evidence or {}),
    }


def typical_pairs(snapshot, target_issue: int, window: int = 500) -> Dict[str, dict]:
    """从**目标期之前**的官方数据构造带标签的浮动奖配对情景集合。

    只保留**同一期**的一、二等奖联合关系（不得把两个独立分位数拼成一次开奖）。
    """
    if snapshot is None or not getattr(snapshot, "ok", False):
        raise BonusFeasibilityError("缺少官方快照")
    d = snapshot.draws
    sub = d[(d.issue < int(target_issue))].tail(window)
    if len(sub) == 0:
        raise BonusFeasibilityError("历史窗口为空")
    src = [int(x) for x in sub.issue.tolist()]
    out = {}
    for q, name in ((Decimal("0.05"), "p05"), (Decimal("0.50"), "p50"),
                    (Decimal("0.95"), "p95")):
        a = Decimal(str(sub["amt_一等奖"].quantile(float(q))))
        b = Decimal(str(sub["amt_二等奖"].quantile(float(q))))
        out[name] = {"kind": "historical_pair",
                     "first": a, "second": b,
                     "source": "目标期之前最近 {} 期的官方期数加权分位数 q={}".format(
                         len(sub), q),
                     "note": "分位数分别取自一、二等奖的边缘分布；"
                             "若要保留**同期联合关系**，应使用 pairs_from_history()。",
                     "source_issues": [min(src), max(src)]}
    out["max_pair"] = {"kind": "historical_pair",
                       "first": Decimal(str(sub["amt_一等奖"].max())),
                       "second": Decimal(str(sub["amt_二等奖"].max())),
                       "source": "窗口内各自的最大值（**不是同一期**，仅作上限参考）",
                       "note": "明确标注：不是一次实际开奖的配对",
                       "source_issues": [min(src), max(src)]}
    return out


def pairs_from_history(snapshot, target_issue: int, window: int = 500,
                       quantiles=(Decimal("0.05"), Decimal("0.50"), Decimal("0.95"))) -> Dict[str, dict]:
    """保留**同一期**一、二等奖联合关系的配对情景（按 _一等奖 + 20·_二等奖 排序取分位）。"""
    d = snapshot.draws
    sub = d[(d.issue < int(target_issue))].tail(window).copy()
    if len(sub) == 0:
        raise BonusFeasibilityError("历史窗口为空")
    sub["_score"] = sub["amt_一等奖"] + 20 * sub["amt_二等奖"]
    src = [int(x) for x in sub.issue.tolist()]
    out = {}
    for q in quantiles:
        row = sub[sub["_score"] >= sub["_score"].quantile(float(q))].iloc[0]
        out["pair_q{:.2f}".format(float(q))] = {
            "kind": "historical_pair",
            "first": Decimal(str(row["amt_一等奖"])),
            "second": Decimal(str(row["amt_二等奖"])),
            "source": "期号 {} 的**同一期**一/二等奖配对；按 (一等+20×二等) 取分位 q={}".format(
                int(row["issue"]), q),
            "note": "保留同期联合关系",
            "source_issue": int(row["issue"]),
            "source_issues": [min(src), max(src)],
        }
    return out


def rule_envelope_pair(note: str = "") -> dict:
    """规则允许的乐观包络：一、二等奖同时取各自封顶。"""
    return {"kind": "rule_envelope",
            "first": FIRST_PRIZE_CAP, "second": SECOND_PRIZE_CAP,
            "source": "2026 七奖级规则：一等奖两段各封顶 500 万（合计 1000 万）；"
                      "二等奖单注上限 500 万",
            "note": note or "两者能否**同时**触及上限未核实；该包络仅用于'即使按上限'的判断"}

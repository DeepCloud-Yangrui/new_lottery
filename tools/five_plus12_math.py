# -*- coding: utf-8 -*-
"""5+12 策略的精确数学基线（零假设：前区均匀随机、各期独立）。

"5+12" 定义
-----------
固定一组前区 5 码；后区 1~12 全包，展开 C(12,2)=66 注；
基本投注、不追加、不倍投。单组成本 = 66 × 2 = 132 元。

关键结构事实（必须说清楚，否则会得出错误结论）
--------------------------------------------
1. 一组 5+12 = 66 注**共用同一组前区号码**。因此 66 注的奖金高度相关：
   后区"全包"只是消除了后区不确定性，**不改变期望值**。
   E[一组 5+12 奖金] = 66 × E[单注奖金]。
2. 对固定前区，后区命中分布恒为 1 / 20 / 45（中 2 / 恰好中 1 / 中 0），
   与开奖结果无关，是组合结构决定的恒等式。
3. 因此单组回收只是"前区命中数"的函数：R(f) = Σ_b n_b × prize(f, b)。
   前区中 0~2 个也有回收（不是零），必须计入。

关于"多组"与"候选池"的严格表述（本轮修正）
------------------------------------------
对**任意**固定的前区 5 码组合 C，以及均匀随机的开奖 5 码 D：

    P(|C ∩ D| = k) = C(5,k)·C(30,5−k) / C(35,5)      —— 与 C 具体是什么无关

因此：

* **单组的命中数边际分布与"选哪些号码"无关**，也就与候选池大小无关；
* 多组的总回收期望 = 各组期望之和（**线性，不要求组间独立**），
  所以"9 候选池 + 3 组"与"全号池 + 3 组"在**同成本**下的
  无条件期望回收是**相同的**；
* 候选池与组合重叠改变的是**联合分布**（重合度、覆盖率、方差、回撤），
  而不是单组的无条件期望。

> 任何"候选池更小 ⇒ 期望 ROI 更低"的说法都需要额外假设（例如开奖与
> 我们的选号存在相关性），本项目没有任何此类证据。见
> ``docs/project_notes/P5_ERRATA_AND_CORRECTED_INFERENCE.md``。

本模块只做精确组合计算，不含任何模型。
"""
from __future__ import annotations

import json
import math
from math import comb
from typing import Dict, Iterable, Optional, Sequence, Tuple

N_FRONT = 35
N_BACK = 12
FRONT_PICK = 5
BACK_PICK = 2
BET_PRICE = 2
BACK_PAIRS = comb(N_BACK, BACK_PICK)          # 66
COST_PER_GROUP = BACK_PAIRS * BET_PRICE       # 132

# 后区全包下，给定后区命中数的注数（恒等式，与开奖无关）
BACK_HIT_TICKETS = {2: 1, 1: 20, 0: 45}

# 浮动大奖对应的 (前区命中, 后区命中)。"剔除大奖口径"只置零这两个组合，
# **保留 (5,0) 的固定奖**。
JACKPOT_CONDITIONS = {(5, 2), (5, 1)}

_FRONT_TOTAL = comb(N_FRONT, FRONT_PICK)      # 324632


def front_hit_distribution() -> Dict[int, float]:
    """P(前区命中 k)、k=0..5。精确超几何（与所选号码无关）。"""
    return {
        k: comb(FRONT_PICK, k) * comb(N_FRONT - FRONT_PICK, FRONT_PICK - k) / _FRONT_TOTAL
        for k in range(FRONT_PICK + 1)
    }


def joint_distribution() -> Dict[Tuple[int, int], float]:
    """P(前区命中 f, 后区命中 b)。"""
    pf = front_hit_distribution()
    pb = {b: n / BACK_PAIRS for b, n in BACK_HIT_TICKETS.items()}
    return {(f, b): pf[f] * pb[b] for f in range(6) for b in range(3)}


def group_payout_by_front_hit(table: Dict[Tuple[int, int], int]) -> Dict[int, int]:
    """R(f)：一组 5+12 在前区命中 f 个时的总奖金。"""
    return {
        f: sum(n * table.get((f, b), 0) for b, n in BACK_HIT_TICKETS.items())
        for f in range(6)
    }


def no_jackpot_table(table: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    """剔除大奖口径：只把 (5,2)、(5,1) 置零，**(5,0) 的固定奖保留**。

    规范依据：官方规则第二十六条"每注只兑付所中得最高奖级奖金"，
    因此 (5,0) 是一注独立的中奖情形，与一、二等奖无关。
    """
    return {c: v for c, v in table.items() if c not in JACKPOT_CONDITIONS}


def single_bet_ev(table: Dict[Tuple[int, int], int]) -> float:
    """单注（2 元）期望奖金。"""
    j = joint_distribution()
    return sum(p * table.get(c, 0) for c, p in j.items())


# ---------------------------------------------------------------------------
# 盈亏平衡：多种情景，明确假设，含退化保护
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    if b is None or abs(b) < 1e-12:
        return None
    return a / b


def breakeven_scenarios(table: Dict[Tuple[int, int], int],
                        cost_group: int = COST_PER_GROUP) -> dict:
    """盈亏平衡所需的"前区命中分布偏移"。

    **重要：这里给出的不是"任意策略通用门槛"，而是若干条明确假设下的
    情景计算。** 提高命中率的方式不同，所需幅度也不同。

    记号：
        pf[k] = P(前区命中 k)；q = P(前区 >= 3) = Σ_{k>=3} pf[k]
        L = E[R | 前区 < 3]（低命中组的条件均值）
        H = E[R | 前区 >= 3]（高命中组的条件均值）
        E[R] = (1-q)·L + q·H

    情景
    ----
    S1「整块搬运、组内形状不变」：
        把质量从 {0,1,2} 搬到 {3,4,5}，两组的**条件分布形状保持不变**
        （L、H 不变）。这是最常见的隐含假设，解
            q_be = (cost − L) / (H − L)
        等价说法：保持 P(0),P(1),P(2) 的**绝对**概率不变，
        把剩余质量按当前条件比例分给 {3,4,5}。

    S2「只增加恰好 3 中」（3 中专家）：
        保持 f ∈ {0,1,2,4,5} 的**绝对**概率不变，只把 P(3) 调大。

    S3「只增加恰好 4 中」（4 中专家）：
        保持 f ∈ {0,1,2,3,5} 的绝对概率不变，只把 P(4) 调大。

    退化保护：解小于 0 → 在当前假设下已经满足；解大于 1 → 该情景**不可达**。
    """
    R = group_payout_by_front_hit(table)
    pf = front_hit_distribution()
    q0 = sum(pf[k] for k in (3, 4, 5))
    L = _safe_div(sum(pf[k] * R[k] for k in (0, 1, 2)), 1 - q0)
    H = _safe_div(sum(pf[k] * R[k] for k in (3, 4, 5)), q0)
    E_R = sum(pf[k] * R[k] for k in range(6))

    def _verdict(x, unit="概率"):
        if x is None:
            return "退化（分母为 0）"
        if x < 0:
            return "已在当前假设下满足（无需提高）"
        if x > (1 if unit == "概率" else float("inf")):
            return "不可达（该情景无法达到盈亏平衡）"
        return "可达"

    out = {
        "cost_group": cost_group,
        "p_front_eq_k": {str(k): pf[k] for k in range(6)},
        "q_front_ge3_now": q0,
        "L_mean_low_hits": L,
        "H_mean_high_hits": H,
        "expected_payout": E_R,
        "expected_roi": E_R / cost_group - 1,
        # 回收倍数（**不是**概率倍数）
        "payout_ratio_needed": cost_group / E_R if E_R > 0 else None,
        "note_payout_ratio": "cost / E[回收] 是**回收倍数**，不能读成"
                             "'命中率需要提高这么多倍'。",
    }
    # S1
    q_be = _safe_div(cost_group - L, H - L)
    out["S1_block_transfer"] = {
        "assumption": "从 {0,1,2} 向 {3,4,5} 搬运质量，两组条件分布形状不变",
        "q_break_even": q_be,
        "q_multiple": _safe_div(q_be, q0),
        "verdict": _verdict(q_be, "概率"),
    }
    # S2 / S3
    for name, k in (("S2_only_front3", 3), ("S3_only_front4", 4)):
        others = [j for j in range(6) if j != k]
        rest = sum(pf[j] * R[j] for j in others)
        p_be = _safe_div(cost_group - rest, R[k])
        out[name] = {
            "assumption": "只调整 P(前区=={})，其余命中数的绝对概率不变".format(k),
            "p_break_even": p_be,
            "p_multiple": _safe_div(p_be, pf[k]),
            "verdict": _verdict(p_be, "概率"),
        }
    return out


# ---------------------------------------------------------------------------
# 多组矩的精确计算（同一期共同开奖）
# ---------------------------------------------------------------------------

def exact_multi_group_moments(table: Dict[Tuple[int, int], int],
                              n_groups: int = 3,
                              pool_size: Optional[int] = None) -> dict:
    """**精确**计算一期多组总回收的均值与方差（不使用蒙特卡洛）。

    模型：同一期的 n_groups 组共用同一次开奖 D。

    - ``pool_size=None``：每组独立均匀取自 1~35。
      给定 D 时各组条件独立，且每组 f 的边际分布与 D 无关，
      故组间**完全独立**：
          E = n·μ，Var = n·σ²
    - ``pool_size=p``：每组取自同一期固定的 p 个候选号。
      令 m = |池 ∩ D| ~ Hypergeometric(35, p, 5)，
      则 f | m ~ Hypergeometric(p, m, 5)，组间条件独立、
      但通过 m **正相关**：
          E = n·E_m[μ(m)]
          Var = n·E_m[σ²(m)] + n²·Var_m(μ(m))

    结论（可验证）：**两种结构的 E 完全相同**（因为 E_m[μ(m)] = μ），
    差别只在方差这一项上。
    """
    R = group_payout_by_front_hit(table)
    pf = front_hit_distribution()
    mu = sum(pf[k] * R[k] for k in range(6))
    var = sum(pf[k] * R[k] ** 2 for k in range(6)) - mu ** 2
    if pool_size is None:
        return {
            "pool_size": None, "n_groups": n_groups,
            "E_total": n_groups * mu, "Var_total": n_groups * var,
            "sd_total": math.sqrt(max(n_groups * var, 0.0)),
            "E_single": mu, "Var_single": var,
            "between_group_component": 0.0,
            "note": "全号池：给定开奖，各组条件独立；且边际与开奖无关，故组间无相关。",
        }
    p = int(pool_size)
    m_sup = list(range(0, min(p, 5) + 1))
    pm = [comb(p, x) * comb(35 - p, 5 - x) / comb(35, 5) for x in m_sup]
    tot = sum(pm)
    pm = [x / tot for x in pm]
    mu_m, var_m = [], []
    for x in m_sup:
        ks = list(range(0, min(x, 5) + 1))
        pk = [comb(x, k) * comb(p - x, 5 - k) / comb(p, 5) for k in ks]
        s = sum(pk)
        pk = [v / s for v in pk]
        mm = sum(pk[i] * R[ks[i]] for i in range(len(ks)))
        vv = sum(pk[i] * R[ks[i]] ** 2 for i in range(len(ks))) - mm ** 2
        mu_m.append(mm)
        var_m.append(vv)
    E_mu = sum(pm[i] * mu_m[i] for i in range(len(m_sup)))
    E_var = sum(pm[i] * var_m[i] for i in range(len(m_sup)))
    V_mu = sum(pm[i] * mu_m[i] ** 2 for i in range(len(m_sup))) - E_mu ** 2
    E_total = n_groups * E_mu
    Var_total = n_groups * E_var + n_groups ** 2 * V_mu
    return {
        "pool_size": p, "n_groups": n_groups,
        "E_total": E_total, "Var_total": Var_total,
        "sd_total": math.sqrt(max(Var_total, 0.0)),
        "E_single": E_mu, "Var_single": E_var,
        "between_group_component": n_groups ** 2 * V_mu,
        "within_group_component": n_groups * E_var,
        "m_distribution": {str(m_sup[i]): pm[i] for i in range(len(m_sup))},
        "E_R_given_m": {str(m_sup[i]): mu_m[i] for i in range(len(m_sup))},
        "note": "p 号候选池：组间通过 m=|池∩开奖| 正相关；"
                "E_total 与全号池相同，Var_total 多了 n²·Var_m(μ(m)) 这一项。",
    }


# ---------------------------------------------------------------------------
# 蒙特卡洛：多组必须共用同一期开奖
# ---------------------------------------------------------------------------

def simulate_profits(table: Dict[Tuple[int, int], int], n_draws: int,
                     n_groups: int = 1, pool_size: Optional[int] = None,
                     trials: int = 200_000, seed: int = 20261013,
                     cost_group: int = COST_PER_GROUP):
    """模拟 n_draws 期的累计盈亏。

    多组模型（本轮修正）
    --------------------
    **同一期的所有组共用同一次开奖**，因此组间收益天然相关。

    - ``pool_size=None``：每组从 1~35 独立均匀取 5 码。
      此时给定开奖 D，各组命中数条件独立，且**边际分布与 D 无关**，
      故组间实际相互独立（这来自"全号池"这一设定，不是普适结论）。
    - ``pool_size=p``：每组从同一期固定的 p 个候选号中取 5 码。
      命中数 f_i | m ~ Hypergeometric(p, m, 5)，其中 m = |候选池 ∩ 开奖|，
      m ~ Hypergeometric(35, p, 5)。组间通过 m 正相关。

    返回：每期盈亏数组的分布统计（含从 0 起算的回撤）。
    """
    import numpy as np

    R = np.array([group_payout_by_front_hit(table)[k] for k in range(6)], dtype=float)
    rng = np.random.default_rng(seed)
    cost = cost_group * n_groups

    if pool_size is None:
        pf = front_hit_distribution()
        probs = np.array([pf[k] for k in range(6)])
        hits = rng.choice(6, size=(trials, n_draws * n_groups), p=probs)
        total_reward = R[hits].reshape(trials, n_draws, n_groups).sum(axis=2)
    else:
        p = int(pool_size)
        # m = |候选池 ∩ 开奖| ~ Hypergeometric(35, p, 5)
        m_support = np.arange(0, min(p, 5) + 1)
        m_prob = np.array([comb(p, mm) * comb(35 - p, 5 - mm) / comb(35, 5)
                           for mm in m_support])
        m_prob = m_prob / m_prob.sum()
        m = rng.choice(m_support, size=trials * n_draws, p=m_prob)
        flat = np.zeros(trials * n_draws)
        chunk = 2_000_000
        for mm in m_support:
            idx = np.flatnonzero(m == mm)
            if idx.size == 0:
                continue
            kk = np.arange(0, min(mm, 5) + 1)
            kp = np.array([comb(mm, k) * comb(p - mm, 5 - k) / comb(p, 5)
                           for k in kk])
            kp = kp / kp.sum()
            for s0 in range(0, idx.size, chunk):
                sl = idx[s0:s0 + chunk]
                f = rng.choice(kk, size=(sl.size, n_groups), p=kp)
                flat[sl] = R[f].sum(axis=1)
        total_reward = flat.reshape(trials, n_draws)

    profit = total_reward - cost
    cum = np.concatenate([np.zeros((trials, 1)), np.cumsum(profit, axis=1)], axis=1)
    dd = (np.maximum.accumulate(cum, axis=1) - cum).max(axis=1)
    total = profit.sum(axis=1)
    return {
        "n_draws": n_draws, "n_groups": n_groups, "pool_size": pool_size,
        "trials": trials, "seed": seed, "period_cost": cost,
        "expected_total_profit": float(total.mean()),
        "prob_total_profit_positive": float((total > 0).mean()),
        "prob_ever_above_zero": float((cum.max(axis=1) > 0).mean()),
        "median_total_profit": float(np.median(total)),
        "p05_total_profit": float(np.percentile(total, 5)),
        "p95_total_profit": float(np.percentile(total, 95)),
        "sd_total_profit": float(total.std(ddof=1)),
        "max_drawdown_mean": float(dd.mean()),
        "max_drawdown_p95": float(np.percentile(dd, 95)),
        "max_drawdown_worst": float(dd.max()),
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def baseline_report(table: Dict[Tuple[int, int], int],
                    label: str = "baseline",
                    n_draws: int = 200,
                    n_groups: int = 1,
                    mc_trials: int = 200_000,
                    seed: int = 20261013,
                    pool_size: Optional[int] = None) -> dict:
    """生成数学基线报告。

    同时给出两套口径：
      - ``full``      ：完整奖金表
      - ``no_jackpot``：只置零 (5,2)、(5,1)，保留 (5,0) 固定奖
    多组时成本与回收**使用同一单位**（每期 n_groups × 132 元）。
    """
    pf = front_hit_distribution()
    R = group_payout_by_front_hit(table)
    Rnj = group_payout_by_front_hit(no_jackpot_table(table))

    def _ev(Rmap):
        return sum(pf[f] * Rmap[f] for f in range(6))

    cost = COST_PER_GROUP * n_groups
    E_R, E_Rnj = _ev(R), _ev(Rnj)

    # 固定奖贡献（不含一、二等奖的浮动部分，但仍保留 (5,0)）
    j = joint_distribution()
    contrib = {}
    for c, p in sorted(j.items()):
        v = p * table.get(c, 0)
        if v > 0:
            contrib["{}+{}".format(*c)] = v
    floating = sum(v for k, v in contrib.items() if k in ("5+2", "5+1"))

    out = {
        "label": label,
        "table_used": {"{}+{}".format(*k): v for k, v in sorted(table.items())},
        "cost_per_group": COST_PER_GROUP,
        "n_groups": n_groups,
        "period_cost": cost,
        "front_hit_probability": {str(k): pf[k] for k in range(6)},
        "group_payout_by_front_hit": {str(k): R[k] for k in range(6)},
        "group_payout_no_jackpot": {str(k): Rnj[k] for k in range(6)},
        # 完整口径
        "expected_group_payout": E_R,
        "expected_period_payout": E_R * n_groups,
        "expected_roi": E_R / COST_PER_GROUP - 1,
        # 剔除大奖口径（只置零 5+2 / 5+1）
        "expected_group_payout_no_jackpot": E_Rnj,
        "expected_roi_no_jackpot": E_Rnj / COST_PER_GROUP - 1,
        "component_ev": contrib,
        "floating_jackpot_share_of_ev": floating / E_R if E_R else None,
        "breakeven_full": breakeven_scenarios(table),
        "breakeven_no_jackpot": breakeven_scenarios(no_jackpot_table(table)),
        "mc": simulate_profits(table, n_draws=n_draws, n_groups=n_groups,
                               pool_size=pool_size, trials=mc_trials, seed=seed),
    }
    return out


def base_table_from_config(issue: int) -> Dict[Tuple[int, int], int]:
    """兼容旧接口：用 config 的奖金表（仅供对照，已知有误）。"""
    from config import get_dlt_prize_table
    return dict(get_dlt_prize_table(issue))


if __name__ == "__main__":
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--issue", type=int, default=26116)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dlt_prize_rules import load_official, base_table

    od = load_official(args.run_dir)
    results = []
    for name, tbl in [
        ("official_issue_{}".format(args.issue), od.issue_table(args.issue, "full")),
        ("official_issue_{}_no_bonus".format(args.issue), od.issue_table(args.issue, "none")),
        ("era3_base_table", base_table(26116)),
    ]:
        results.append(baseline_report(tbl, label=name, n_draws=args.n_draws))
    txt = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
    print(txt)

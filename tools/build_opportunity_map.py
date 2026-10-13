# -*- coding: utf-8 -*-
"""D3 交付：奖金机会地图与盈利缺口（V2 T3）。

产出
----
1. **奖级分解**：一组 5+12 在各档位/派奖状态下，每个奖级对回收的期望贡献。
2. **情景表**：用**目标期之前**的历史构造浮动奖情景（分位数），
   逐情景给出净收益与 ROI。
3. **机会地图**：在"提高前区命中率"的不同倍数下，各情景的净收益
   —— 用来回答"还差多少"。
4. **盈利缺口**：当前最好情景与 0 的差距，以及达到 0 所需的命中率偏移。
5. **结论**：若连"合理乐观"情景都是负的，则该分支停止（PLAN §6 D3 关卡）。

全部输入都满足"开奖前可得"：奖池用上一期开奖后公布的值，
浮动奖情景只用目标期之前的期次，派奖只在活动期次已事先公布时计入。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import prize_rules as pr
from tools.five_plus12_math import (
    BACK_HIT_TICKETS, COST_PER_GROUP, front_hit_distribution,
    group_payout_by_front_hit, breakeven_scenarios,
)
from tools.profit_accounting import TAX_2024_MERGED
from tools import profit_opportunity as po


def level_contribution(table, hd):
    """按奖级分解一组的期望回收。"""
    out = {}
    for f in range(6):
        if hd[f] == 0:
            continue
        for b, cnt in BACK_HIT_TICKETS.items():
            v = table.get((f, b), 0)
            if v <= 0:
                continue
            lv = pr.level_of(None, f, b) if False else None
            out.setdefault("{}".format((f, b)), 0.0)
            out["{}".format((f, b))] += hd[f] * cnt * v
    return out


def by_level(table, issue, hd):
    """按**奖级名**分解一组的期望回收（元/期）。"""
    agg = {}
    for f in range(6):
        for b, cnt in BACK_HIT_TICKETS.items():
            v = table.get((f, b), 0)
            if v <= 0:
                continue
            lv = pr.level_of(issue, f, b) or "未中奖"
            agg[lv] = agg.get(lv, 0.0) + hd[f] * cnt * v
    return agg


def scale_distribution(hd, q_mult):
    """S1 假设下按倍数 q_mult 放大 P(前区>=3)（组内条件分布形状不变）。"""
    hd = dict(hd)
    q0 = sum(hd[f] for f in (3, 4, 5))
    q1 = min(max(q0 * q_mult, 0.0), 1.0)
    cond = {f: hd[f] / q0 for f in (3, 4, 5)}
    low = {f: hd[f] / (1 - q0) for f in (0, 1, 2)}
    out = {}
    for f in (3, 4, 5):
        out[f] = q1 * cond[f]
    for f in (0, 1, 2):
        out[f] = (1 - q1) * low[f]
    return out


def _apply_high_tier(table, issue):
    """把上浮档金额按 (前区命中, 后区命中) 正确写回奖金表。"""
    out = dict(table)
    era = pr.era_of(issue)
    for level, amount in pr.ERA3_HIGH_AMOUNT.items():
        for cond in pr.LEVEL_CONDITIONS[era].get(level, ()):
            out[cond] = int(amount)
    return out


def _assert_high_tier_differs(base_tbl, high_tbl, issue):
    """自检：上浮档确实改变了 (前区命中,后区命中) 的奖金。"""
    changed = {k: (base_tbl.get(k), high_tbl.get(k))
               for k in high_tbl if base_tbl.get(k) != high_tbl.get(k)}
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-issue", type=int, default=26117,
                    help="尚未开奖的目标期号")
    ap.add_argument("--as-of", default="2026-10-13")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--n-groups", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap = pr.get_snapshot()
    if not snap.ok:
        raise SystemExit("缺少官方快照，无法构造开奖前情景")

    issue = int(args.target_issue)
    pool_before = snap.pool_before(issue)
    drawn = snap.has(issue)
    tier = "high" if (pool_before is not None and pool_before >= pr.ERA3_POOL_THRESHOLD) else "base"

    hd = front_hit_distribution()
    out = {
        "target_issue": issue,
        "as_of": args.as_of,
        "already_drawn": bool(drawn),
        "pool_before": pool_before,
        "fixed_prize_tier": tier,
        "goal": "含一、二等奖的税后净收益",
        "notes": [],
    }
    if drawn:
        out["notes"].append("目标期已开奖——本图仅作回溯演示，不得当作开奖前估计。")

    # ---- 1. 奖级分解 ----
    base_tbl = pr.base_table(issue, pool_before=pool_before)
    # 反事实：若奖池达到 8 亿。注意 ERA3_HIGH_AMOUNT 以**奖级名**为键，
    # 而奖金表以 (前区命中, 后区命中) 为键 —— 必须经 LEVEL_CONDITIONS 映射，
    # 否则 update() 只是往表里塞了无效的字符串键（此前的错误）。
    high_tbl = _apply_high_tier(base_tbl, issue)
    # 奖级分解必须使用**与机会估计相同**的浮动奖情景，
    # 并对每个情景标明浮动奖来自哪里；固定 fallback 只能作为"显式假设"单独列出。
    hist_scen = po.scenario_from_history(snap, issue, mask="all", q=0.50,
                                         window=args.window)
    out["level_decomposition"] = {}
    for name, tbl in (("base_tier", base_tbl), ("high_tier_counterfactual", high_tbl)):
        d = {k: round(v, 4) for k, v in
             sorted(by_level(tbl, issue, hd).items(), key=lambda kv: -kv[1])}
        # 同时给出用历史 q0.50 浮动奖情景替换后的分解（与情景表一致）
        tbl_s = dict(tbl)
        tbl_s[(5, 2)] = hist_scen["first_prize"]
        tbl_s[(5, 1)] = hist_scen["second_prize"]
        d_s = {k: round(v, 4) for k, v in
               sorted(by_level(tbl_s, issue, hd).items(), key=lambda kv: -kv[1])}
        out["level_decomposition"][name] = {
            "by_level_fallback_floating": d,
            "total_expected_gross_fallback_floating": round(sum(d.values()), 4),
            "by_level_hist_q0.50_floating": d_s,
            "total_expected_gross_hist_q0.50_floating": round(sum(d_s.values()), 4),
            "floating_source": {
                "fallback": "prize_rules.FLOATING_FALLBACK —— **显式假设**，"
                            "不是官方预期值，也不是历史实测值",
                "hist_q0.50": "目标期之前 {} 期的官方数据分位数（q=0.50），"
                              "来源期次 {}".format(len(hist_scen["source_issues"]),
                                                [min(hist_scen["source_issues"]),
                                                 max(hist_scen["source_issues"])]),
            },
        }
    out["high_tier_self_check"] = {
        "changed_conditions": {str(k): v for k, v in
                               _assert_high_tier_differs(base_tbl, high_tbl, issue).items()},
        "note": "自检：上浮档必须改变实际 (前区命中,后区命中) 的奖金，"
                "而不是只往表里加字符串键。",
    }

    # ---- 2. 情景表 ----
    qs = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
    scenarios = []
    for q in qs:
        s = po.scenario_from_history(snap, issue, mask="all", q=q, window=args.window)
        scenarios.append(s)
    # 上浮档专用情景（奖池 >= 8 亿的期次）
    try:
        scenarios.append(po.scenario_from_history(snap, issue, mask="pool_ge_8e8",
                                                  q=0.5, window=args.window))
    except po.OpportunityInputError:
        pass

    ctx = {"issue": issue, "as_of": args.as_of, "n_groups": args.n_groups,
           "pool_size": None, "pool_before": pool_before,
           "bonus_active": False, "bonus_certainty": "official",
           "bonus_eligibility": True}
    est = po.estimate_opportunity(args.as_of, ctx, hit_distribution=None,
                                  prize_scenarios=scenarios,
                                  tax_policy=TAX_2024_MERGED, snapshot=snap)
    out["scenario_estimates"] = {
        "n_groups": args.n_groups,
        "period_cost": est["period_cost"],
        "tax_policy": est["tax_policy"],
        "scenarios": est["scenarios"],
        "verdict": est["verdict"],
        "missing_information": est["missing_information"],
    }

    # ---- 3. 机会地图：提高前区命中率的倍数 ----
    mults = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    grid = []
    for mult in mults:
        hd2 = scale_distribution(hd, mult)
        row = {"q_multiple": mult,
               "p_front_ge3": sum(hd2[f] for f in (3, 4, 5))}
        for s in scenarios:
            tbl = dict(base_tbl)
            tbl[(5, 2)] = s["first_prize"]; tbl[(5, 1)] = s["second_prize"]
            R = group_payout_by_front_hit(tbl)
            # 税必须按**每个可能的期末结果**分别计算后再取期望：
            # 税收是非线性的（≤10000 免征、超过按全额 20%），
            # 对均值套用税额会系统性高估净收益。
            dist, _st = po._outcome_distribution(R, args.n_groups, None, hd2)
            e_net = 0.0
            for gross, p in dist.items():
                g = int(round(gross))
                e_net += p * (g - TAX_2024_MERGED.tax_on(g))
            row[s["label"]] = round(e_net - est["period_cost"], 2)
        grid.append(row)
    out["opportunity_map"] = {
        "hit_rate_multipliers": mults,
        "rows": grid,
        "note": "按 S1 假设（低/高命中组内条件分布形状不变）放大 P(前区>=3)。",
        "group_model": "均按**单组**计算（n_groups=1）：本表不涉及多组的联合分布。"
                       "若要做多组，必须用**实际选定的固定组合**去算其真实相关结构，"
                       "不能用'随机生成的组'模型冒充。",
    }

    # ---- 3b. 每个情景各自的盈亏平衡倍数 ----
    be_by_scenario = []
    for sc in scenarios:
        tbl = dict(base_tbl)
        tbl[(5, 2)] = sc["first_prize"]; tbl[(5, 1)] = sc["second_prize"]
        b = breakeven_scenarios(tbl, cost_group=est["period_cost"])
        be_by_scenario.append({
            "label": sc["label"],
            "first_prize": sc["first_prize"], "second_prize": sc["second_prize"],
            "q_break_even": b["S1_block_transfer"]["q_break_even"],
            "q_multiple": b["S1_block_transfer"]["q_multiple"],
            "verdict": b["S1_block_transfer"]["verdict"],
        })
    out["breakeven_by_scenario"] = {
        "rows": be_by_scenario,
        "note": "每个浮动奖情景各自求解 q*=(cost-L)/(H-L)；"
                "倍数 = q*/q_now。这是特定分布变化假设下的情景值。",
    }

    # ---- 4. 盈利缺口 ----
    best = max(est["scenarios"], key=lambda s: s["expected_profit"])
    be = breakeven_scenarios(base_tbl, cost_group=est["period_cost"])
    out["profit_gap"] = {
        "best_scenario_label": best["label"],
        "best_expected_profit": best["expected_profit"],
        "best_expected_roi": best["expected_roi"],
        "gap_to_zero": -best["expected_profit"],
        "q_multiple_needed_full_table": be["S1_block_transfer"]["q_multiple"],
        "q_multiple_range_across_scenarios": [
            min(x["q_multiple"] for x in be_by_scenario),
            max(x["q_multiple"] for x in be_by_scenario)],
        "q_break_even": be["S1_block_transfer"]["q_break_even"],
        "note": "gap_to_zero 是最优情景下每期仍缺的金额；"
                "q_multiple_needed_full_table 是达到盈亏平衡所需的 P(前区>=3) 倍数。",
    }

    # ---- 5. 结论 ----
    all_neg = all(s["expected_profit"] <= 0 for s in est["scenarios"])
    out["verdict"] = {
        "tested_scenarios_all_non_positive": bool(all_neg),
        "scope": "仅限本轮**实际测试过的 {} 个开奖前情景**×均匀基线命中分布"
                 "×1 组×132 元。".format(len(est["scenarios"])),
        "branch_decision": (
            "在本轮测试范围内停止该分支：所测情景无一为正，且最优情景仍差 {:.2f} 元/期。"
            .format(-best["expected_profit"]) if all_neg else
            "存在正情景，但需按 PLAN §8 判断是否构成可信证据"),
        "not_claimed": [
            "不声称'所有奖金条件都不可能盈利'——本轮只覆盖有限情景；",
            "不声称'所有 5+12 策略都不可能盈利'——命中分布只用了均匀基线；",
            "不声称未来规则下的结论——结论只对期号 ≤ 当前快照有效。",
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("目标期 {}（已开奖={}） 开奖前奖池={} 档位={}".format(
        issue, drawn, pool_before, tier))
    _dec = out["level_decomposition"]["base_tier"]
    print("奖级分解（base 档，历史 q0.50 浮动奖情景，元/期）：")
    for k, v in _dec["by_level_hist_q0.50_floating"].items():
        print("    {:<8s} {:>10.4f}".format(k, v))
    print("    {:<8s} {:>10.4f}".format("合计",
                                       _dec["total_expected_gross_hist_q0.50_floating"]))
    print("高奖池自检：改变的 (前区,后区) 数 = {}".format(
        len(out["high_tier_self_check"]["changed_conditions"])))
    print("\n情景净收益（每期，成本 {} 元）：".format(est["period_cost"]))
    for s in est["scenarios"]:
        print("    {:<34s} E[profit]={:>10.2f}  ROI={:>8.4f}  P(profit>0)={:.4f}".format(
            s["label"][:34], s["expected_profit"],
            s["expected_roi"] if s["expected_roi"] is not None else float("nan"),
            s["prob_profit_positive"]))
    print("\n机会地图（净收益，列=情景）：")
    hdr = [k for k in grid[0] if k not in ("q_multiple", "p_front_ge3")]
    print("    {:>5s} {:>9s} ".format("倍数", "P(>=3)") + " ".join("{:>9s}".format(h[:9]) for h in hdr))
    for r in grid:
        print("    {:>5.1f} {:>9.5f} ".format(r["q_multiple"], r["p_front_ge3"])
              + " ".join("{:>9.1f}".format(r[h]) for h in hdr))
    print("\n盈利缺口：", json.dumps(out["profit_gap"], ensure_ascii=False)[:400])
    print("结论：", out["verdict"]["branch_decision"])
    print("written:", args.out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""V2 D4 执行脚本：同预算下的政策比较（税后净收益）。

协议：``docs/project_notes/PROFIT_V2_D4_PROTOCOL.md``（先冻结、后执行）。

政策：P_skip / P1_a / P1_b / P2 / P3 + 对照 C1（全号池随机）/ C2（同候选池随机）。
计账：``tools/profit_accounting.settle_period``（同期合并计税）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtest_plus as bt                                   # noqa: E402
import prize_rules as pr                                     # noqa: E402
from feature_engineering import load_dlt_history             # noqa: E402
from inference_plus import maybe_load_rf_model                # noqa: E402
from tools.eval_5plus12_frozen import StrategyRunner          # noqa: E402
from tools.profit_accounting import (SimAccount, TAX_2024_MERGED,  # noqa: E402
                                     NO_TAX, resolve_tax_policy)
from tools.five_plus12_math import COST_PER_GROUP             # noqa: E402
from tools.profit_budget import BudgetTracker                 # noqa: E402

FRONT = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
BACK = ["蓝球_1", "蓝球_2"]
Z95, Z80 = 1.6449, 0.8416
BREAKEVEN_EFFECT = 66.0      # 元/期，来自 D3 的中位情景缺口


def analyze(rows, loss_limit=500, bootstrap=2000, seed=20261013,
            tax_on=True):
    """从逐期账本行计算全部预登记指标。"""
    profits = np.array([r["profit"] for r in rows], dtype=float)
    gross = np.array([r["gross"] for r in rows], dtype=float)
    costs = np.array([r["cost"] for r in rows], dtype=float)
    taxes = np.array([r["tax"] for r in rows], dtype=float)
    n = len(profits)
    out = {
        "periods": n,
        "groups_total": int(sum(r["n_groups"] for r in rows)),
        "gross": float(gross.sum()),
        "tax": float(taxes.sum()),
        "net": float(gross.sum() - taxes.sum()),
        "cost": float(costs.sum()),
        "profit": float(profits.sum()),
        "roi": float(profits.sum() / costs.sum()) if costs.sum() else None,
        "tax_share_of_gross": float(taxes.sum() / gross.sum()) if gross.sum() else None,
        "prob_profit_positive_period": float((profits > 0).mean()),
        "mean_profit_per_period": float(profits.mean()),
        "sd_profit_per_period": float(profits.std(ddof=1)) if n > 1 else 0.0,
    }
    pos = profits[profits > 0]
    if pos.size:
        s = np.sort(pos)[::-1]
        out["tail_contribution_top1"] = float(s[:1].sum() / pos.sum())
        out["tail_contribution_top3"] = float(s[:3].sum() / pos.sum())
        out["largest_single_period_profit"] = float(s[0])
    else:
        out["tail_contribution_top1"] = None
        out["tail_contribution_top3"] = None
        out["largest_single_period_profit"] = 0.0

    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    out["max_drawdown"] = float((peak - np.concatenate([[0.0], cum])).max())

    # 期级 bootstrap（只反映模拟不确定性，不增加开奖样本）
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(bootstrap, n)) if n else None
    if idx is not None:
        paths = np.cumsum(profits[idx], axis=1)
        out["bootstrap"] = {
            "B": bootstrap,
            "prob_cumulative_positive": float((paths[:, -1] > 0).mean()),
            "prob_touch_loss_limit": float(((paths < -loss_limit).any(axis=1)).mean()),
            "cumulative_p05": float(np.percentile(paths[:, -1], 5)),
            "cumulative_p50": float(np.percentile(paths[:, -1], 50)),
            "cumulative_p95": float(np.percentile(paths[:, -1], 95)),
            "note": "期级重抽样；不增加开奖样本，不代表未来概率。",
        }
    # 功效：该窗口在 α=.05 单侧、power=.80 下可检出的最小效应
    sd = out["sd_profit_per_period"]
    if sd > 0 and n > 0:
        out["power"] = {
            "target_effect_per_period": BREAKEVEN_EFFECT,
            "detectable_effect_at_power80": float((Z95 + Z80) * sd / np.sqrt(n)),
            "periods_needed_for_breakeven_effect": float(
                ((Z95 + Z80) ** 2) * sd ** 2 / BREAKEVEN_EFFECT ** 2),
            "periods_available": n,
            "note": "含重尾（一/二等奖），正态近似仅供量级参考。",
        }
    # 预算
    acc = BudgetTracker(loss_limit=loss_limit)
    worst = 0
    for p in profits:
        acc.realized_net += int(round(p))
        worst = min(worst, acc.realized_net)
    out["budget"] = {
        "loss_limit": loss_limit,
        "final_cumulative": acc.realized_net,
        "worst_cumulative": worst,
        "limit_breached": worst < -loss_limit,
        "max_affordable_groups_at_start": BudgetTracker(loss_limit=loss_limit).max_affordable_groups(),
    }
    return out


def build_policy_rows(data, indices, sr, policy, snapshot, tax_policy):
    acc = SimAccount(policy, snapshot=snapshot, tax_policy=tax_policy)
    for i in indices:
        issue = int(data.iloc[i]["期数"])
        drawn = set(int(data.iloc[i][c]) for c in FRONT)
        if policy == "P_skip":
            hits = []
        elif policy in ("P1_a", "P1_b"):
            mode = "stat_miss_penalty" if policy == "P1_a" else "stat_miss_bonus"
            g = sr.s1_groups("d4", data, i, top_n=10, play=1, score_mode=mode)
            hits = [len(set(x) & drawn) for x in g]
        elif policy == "P2":
            g = sr.s1_groups("d4", data, i, top_n=10, play=2,
                             score_mode="stat_miss_penalty")
            hits = [len(set(x) & drawn) for x in g]
        elif policy == "P3":
            g = sr.s2_groups("d4", data, i, "n9_mix_441", 3)
            hits = [len(set(x) & drawn) for x in g]
        else:
            raise ValueError(policy)
        if not hits:
            acc.rows.append({"account_id": policy, "issue": issue, "n_groups": 0,
                             "gross": 0, "tax": 0, "net": 0, "cost": 0, "profit": 0,
                             "gross_by_level": {}, "labels": {"prize_source": "n/a",
                                                              "is_estimate": False}})
        else:
            acc.settle(issue, hits)
    return acc.ledger()


def build_control_rows(data, indices, n_groups, snapshot, seed=7, n_trials=200,
                       pool_fn=None, policy="C", bonus_eligible=True):
    """同期、同组数的随机对照；返回 (rows, trials_metrics)。"""
    trial_rows = []
    for t in range(n_trials):
        rng = random.Random(seed + t * 7919)
        acc = SimAccount("{}_{}".format(policy, t), snapshot=snapshot,
                         tax_policy=TAX_2024_MERGED)
        for i in indices:
            issue = int(data.iloc[i]["期数"])
            drawn = set(int(data.iloc[i][c]) for c in FRONT)
            if pool_fn is None:
                groups = [sorted(rng.sample(range(1, 36), 5)) for _ in range(n_groups)]
            else:
                cand = pool_fn(i)
                if not cand or len(cand) < 5:
                    groups = []
                else:
                    groups = [sorted(rng.sample(cand, 5)) for _ in range(n_groups)]
            hits = [len(set(g) & drawn) for g in groups]
            if hits:
                acc.settle(issue, hits)
            else:
                acc.rows.append({"account_id": acc.account_id, "issue": issue,
                                 "n_groups": 0, "gross": 0, "tax": 0, "net": 0,
                                 "cost": 0, "profit": 0, "gross_by_level": {},
                                 "labels": {}})
        trial_rows.append(acc.ledger())
    return trial_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--hist", default=None)
    ap.add_argument("--periods", type=int, default=500)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--loss-limit", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-rows", default=None,
                    help="把逐期明细写入该目录，供后续（D5/D7）复用而不重训 RF")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    hist = args.hist or os.path.join(args.run_dir, "data", "dlt_hist_7001_26042.csv")
    data = load_dlt_history(hist)
    n = len(data)
    start = max(121, n - args.periods)
    indices = list(range(start, n))
    issue_lo, issue_hi = int(data.iloc[start]["期数"]), int(data.iloc[n - 1]["期数"])
    rf, meta = maybe_load_rf_model()
    sr = StrategyRunner(rf, meta)
    snap = pr.get_snapshot()

    result = {
        "protocol": "docs/project_notes/PROFIT_V2_D4_PROTOCOL.md",
        "window": {"index_from": start, "index_to": n - 1,
                   "issue_from": issue_lo, "issue_to": issue_hi,
                   "periods": len(indices)},
        "data_note": "该窗口已被历史实验反复使用；本轮为回顾性诊断，不宣称样本外。",
        "policies": {}, "controls": {}, "comparisons": {},
    }

    dumped = {}
    for pol in ("P_skip", "P1_a", "P1_b", "P2", "P3"):
        rows = build_policy_rows(data, indices, sr, pol, snap, TAX_2024_MERGED)
        dumped[pol] = rows
        m = analyze(rows, loss_limit=args.loss_limit, bootstrap=args.bootstrap)
        rows_nb = build_policy_rows(data, indices, sr, pol, snap, NO_TAX) \
            if pol != "P_skip" else rows
        m_nb = analyze(rows_nb, loss_limit=args.loss_limit, bootstrap=args.bootstrap)
        result["policies"][pol] = {
            "after_tax": m,
            "pre_tax_legacy": {k: m_nb[k] for k in
                               ("gross", "net", "profit", "roi", "tax")},
            "avg_groups": float(np.mean([r["n_groups"] for r in rows])),
        }
        print("{:<8s} periods={:>4d} cost={:>8.0f} gross={:>8.0f} tax={:>8.0f} "
              "profit={:>9.0f} roi={:>8.4f}".format(
                  pol, m["periods"], m["cost"], m["gross"], m["tax"], m["profit"],
                  m["roi"] if m["roi"] is not None else float("nan")))

    # 对照：C1（全号池随机）
    for name, g, ref in (("C1_1group", 1, "P1_a"), ("C1_3groups", 3, "P3")):
        trials = build_control_rows(data, indices, g, snap,
                                    n_trials=args.n_trials, policy=name)
        ms = [analyze(r, loss_limit=args.loss_limit, bootstrap=200) for r in trials]
        agg = {}
        for k in ("profit", "roi", "prob_profit_positive_period",
                  "max_drawdown", "tax"):
            v = np.array([m[k] for m in ms if m[k] is not None], dtype=float)
            agg[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                      "p05": float(np.percentile(v, 5)),
                      "p95": float(np.percentile(v, 95))}
        result["controls"][name] = {"n_trials": args.n_trials, "metrics": agg}
        refm = result["policies"][ref]["after_tax"]
        result["comparisons"]["{}_{}".format(ref, name)] = {
            "strategy_profit": refm["profit"],
            "control_mean_profit": agg["profit"]["mean"],
            "gap": refm["profit"] - agg["profit"]["mean"],
            "strategy_percentile": float(np.mean(
                [m["profit"] <= refm["profit"] for m in ms])),
            "note": "对照与策略同期、同组数、同成本。",
        }
        print("{:<8s} mean_profit={:>9.0f} sd={:>9.0f}".format(
            name, agg["profit"]["mean"], agg["profit"]["sd"]))

    # C2：同候选池随机（仅 3 组口径）
    def pool_fn(i):
        return sr.s2_pool("d4", data, i, "n9_mix_441").get("candidates", [])
    trials = build_control_rows(data, indices, 3, snap, n_trials=max(20, args.n_trials // 5),
                                policy="C2_441", pool_fn=pool_fn)
    ms = [analyze(r, loss_limit=args.loss_limit, bootstrap=200) for r in trials]
    v = np.array([m["profit"] for m in ms], dtype=float)
    result["controls"]["C2_441"] = {
        "n_trials": len(ms),
        "metrics": {"profit": {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}},
    }
    refm = result["policies"]["P3"]["after_tax"]
    result["comparisons"]["P3_C2_441"] = {
        "strategy_profit": refm["profit"], "control_mean_profit": float(v.mean()),
        "gap": refm["profit"] - float(v.mean()),
        "strategy_percentile": float(np.mean([m["profit"] <= refm["profit"] for m in ms])),
        "note": "隔离缩水贡献：候选池与策略相同，组内随机。",
    }
    print("{:<8s} mean_profit={:>9.0f} sd={:>9.0f}".format("C2_441", v.mean(), v.std(ddof=1)))

    # 同预算比较：P1_a 的 1/3 期数 vs P3 全部期数（总票款近似相等）
    k = len(indices) // 3
    rows_p3 = build_policy_rows(data, indices, sr, "P3", snap, TAX_2024_MERGED)
    rows_p1 = build_policy_rows(data, indices, sr, "P1_a", snap, TAX_2024_MERGED)
    cut3 = analyze(rows_p3[:k], loss_limit=args.loss_limit, bootstrap=args.bootstrap)
    sub1 = [r for r in rows_p1[:k]]
    # 1 组方案在相同总票款下可覆盖 3 倍期数 -> 用前 3k 期
    cut1 = analyze(rows_p1[:min(len(rows_p1), 3 * k)], loss_limit=args.loss_limit,
                   bootstrap=args.bootstrap)
    result["same_budget_comparison"] = {
        "basis": "总票款相等（1 组 × 3k 期 vs 3 组 × k 期）",
        "k_periods_3groups": k,
        "P3_3groups": {"cost": cut3["cost"], "profit": cut3["profit"], "roi": cut3["roi"]},
        "P1_a_1group": {"cost": cut1["cost"], "profit": cut1["profit"], "roi": cut1["roi"]},
        "note": "同预算比较用于避免把多组=多花钱=中奖率更高当成效率提升。",
    }
    print("同预算: P3(cost={:.0f}, profit={:.0f}) vs P1_a(cost={:.0f}, profit={:.0f})".format(
        cut3["cost"], cut3["profit"], cut1["cost"], cut1["profit"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("written:", args.out)

    if args.dump_rows:
        os.makedirs(args.dump_rows, exist_ok=True)
        slim = {pol: [{"issue": r["issue"], "n_groups": r["n_groups"],
                       "gross": r["gross"], "tax": r["tax"], "cost": r["cost"],
                       "profit": r["profit"]} for r in rs]
                for pol, rs in dumped.items()}
        pth = os.path.join(args.dump_rows, "rows_d4_policies.json")
        with open(pth, "w", encoding="utf-8") as f:
            json.dump({"window": result["window"], "rows": slim}, f,
                      ensure_ascii=False, indent=2)
        print("rows dumped:", pth)


if __name__ == "__main__":
    main()

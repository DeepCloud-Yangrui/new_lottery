# -*- coding: utf-8 -*-
"""P5 补充：回顾窗口的均匀随机对照（C1）与命中率的解析检验。

为什么需要
----------
`tools/eval_5plus12_frozen.py` 只在样本外窗口跑了随机对照。
回顾窗口（200/500/1000 期）的结论也需要可比的随机基线，
否则"策略 ROI −0.66"这个数字没有参照物。

本脚本只做均匀随机对照（不调用 RF，秒级完成）+ 解析二项检验：
策略的"至少一组中 3 个前区"的观测频率，与零假设
`P(前区 >= 3) = 1 - C(30,5)/C(35,5) - 5*C(30,4)/C(35,5) - 10*C(30,3)/C(35,5)`
（多组时按 `1-(1-p)^g`）做单侧二项检验。

用法：
  python tools/analysis_retro_controls.py --data <hist csv> --results <full_run.json> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from math import comb
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtest_plus as bt                      # noqa: E402
from feature_engineering import load_dlt_history  # noqa: E402

FRONT_COLS = bt.FRONT_COLS
BACK_COLS = bt.BACK_COLS
COST = 132


def p_front_ge3():
    C = comb(35, 5)
    return 1.0 - sum(comb(5, k) * comb(30, 5 - k) / C for k in range(3))


def run_uniform(data_asc, start_idx, end_idx, n_groups, n_trials, seed):
    import random
    pf = {}
    rows = []
    for t in range(n_trials):
        rng = random.Random(seed + t * 7919)
        rec = []
        tr = trnj = tc = 0
        h3 = 0
        for idx in range(start_idx, end_idx + 1):
            target = data_asc.iloc[idx]
            issue = int(target["期数"])
            tf = set(int(target[c]) for c in FRONT_COLS)
            tb = set(int(target[c]) for c in BACK_COLS)
            table = bt.compute_prize_table(issue)
            cost = COST * n_groups
            rew = 0
            rew_nj = 0
            max_hit = 0
            for _g in range(n_groups):
                combo = rng.sample(range(1, 36), 5)
                fh = len(set(combo) & tf)
                max_hit = max(max_hit, fh)
                for bh, n in ((2, 1), (1, 20), (0, 45)):
                    v = table.get((fh, bh), 0) * n
                    rew += v
                    if fh < 5:
                        rew_nj += v
            if max_hit >= 3:
                h3 += 1
            tc += cost
            tr += rew
            trnj += rew_nj
        rows.append({
            "roi": (tr - tc) / tc,
            "roi_no_jackpot": (trnj - tc) / tc,
            "hit3_rate": h3 / (end_idx - start_idx + 1),
            "total_profit": tr - tc,
        })
    out = {}
    for k in ("roi", "roi_no_jackpot", "hit3_rate", "total_profit"):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                  "p05": float(np.percentile(v, 5)),
                  "p95": float(np.percentile(v, 95))}
    out["_arrays"] = {k: np.array([r[k] for r in rows], dtype=float)
                      for k in ("roi", "roi_no_jackpot", "hit3_rate", "total_profit")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    bt.set_prize_mode("official")
    data = load_dlt_history(args.data)
    res = json.load(open(args.results, encoding="utf-8"))
    blk = res["windows"]["official"]

    p1 = p_front_ge3()
    out = {"p_front_ge3_single_group": p1, "windows": {}}

    for n in (200, 500, 1000):
        s = max(121, len(data) - n)
        e = len(data) - 1
        periods = e - s + 1
        entry = {"periods": periods,
                 "issue_start": int(data.iloc[s]["期数"]),
                 "issue_end": int(data.iloc[e]["期数"])}
        for g, key in ((1, "RETROlast{}_S1_main".format(n)),
                       (3, "RETROlast{}_S2_n9_mix_441".format(n))):
            ctl = run_uniform(data, s, e, g, args.n_trials, args.seed)
            strat = blk.get(key)
            if strat is None:
                continue
            # ⚠️ 统计口径警告：策略是**单一确定路径**，不是 n_trials 次随机试验的均值。
            # 因此 (策略值 − 对照均值) / (对照标准差/sqrt(n_trials)) 会随着 n_trials 增大
            # 而无限放大，**不是有效的检验统计量**。此处只把它作为描述性指标，
            # 有效的判据是「百分位排名」与「命中率的二项检验」。
            se = ctl["roi_no_jackpot"]["std"] / np.sqrt(args.n_trials)
            z = (strat["roi_no_jackpot"] - ctl["roi_no_jackpot"]["mean"]) / se if se else float("nan")
            arr = ctl["_arrays"]["roi_no_jackpot"]
            pct = float((arr <= strat["roi_no_jackpot"]).mean())
            arr3 = ctl["_arrays"]["hit3_rate"]
            pct3 = float((arr3 <= strat["hit3_or_more_rate"]).mean())
            # 命中率解析二项检验（单侧，检验"优于随机"）
            k = int(round(strat["hit3_or_more_rate"] * periods))
            p_null = 1 - (1 - p1) ** g
            pval = float(stats.binom.sf(k - 1, periods, p_null))
            entry[key] = {
                "strategy_roi_no_jackpot": strat["roi_no_jackpot"],
                "strategy_hit3_rate": strat["hit3_or_more_rate"],
                "strategy_total_profit": strat["total_profit"],
                "strategy_max_drawdown": strat["max_drawdown"],
                "C1_uniform_roi_no_jackpot": ctl["roi_no_jackpot"],
                "C1_uniform_roi": ctl["roi"],
                "C1_uniform_hit3_rate": ctl["hit3_rate"],
                "roi_no_jackpot_z_vs_C1_NOT_A_P_VALUE": float(z),
                "roi_no_jackpot_percentile_rank": pct,
                "hit3_percentile_rank": pct3,
                "hit3_p_null": p_null,
                "hit3_observed_count": k,
                "hit3_expected_count": periods * p_null,
                "hit3_one_sided_p_value": pval,
            }
        out["windows"]["last{}".format(n)] = entry

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""P5 冻结协议：信号门槛策略（S3）与择时对照（C3）评估。

复用 backtest_plus 的现有实现（P2-4）：
  - S3: run_n9_signal_gate_backtest
  - C3: run_random_same_bet_count_baseline（随机参与期次，同下注次数与组数）
  - C1: run_random_3x5plus12_baseline（相同参与期次，均匀随机选号）

计奖口径由 backtest_plus.set_prize_mode 控制（默认 official，官方核验表）。

用法：
  python tools/eval_gate_frozen.py --data <csv> --out <json> \
      --profile n9_mix_441 --group-counts 3 --quantile 0.7 --periods 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtest_plus as bt                          # noqa: E402
from feature_engineering import load_dlt_history    # noqa: E402
from inference_plus import maybe_load_rf_model      # noqa: E402


def _clean(d):
    return {k: v for k, v in d.items() if k != "records"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prize-mode", default="official")
    ap.add_argument("--profile", default="n9_mix_441")
    ap.add_argument("--group-counts", default="2,3")
    ap.add_argument("--quantiles", default="0.60,0.70,0.80")
    ap.add_argument("--periods", type=int, default=500)
    ap.add_argument("--lookback", type=int, default=200)
    ap.add_argument("--min-signal-history", type=int, default=50)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    bt.set_prize_mode(args.prize_mode)
    rf_model, rf_meta = maybe_load_rf_model()
    data_asc = load_dlt_history(args.data)
    print("数据 {} 期，期号 {}..{}".format(
        len(data_asc), int(data_asc["期数"].iloc[0]), int(data_asc["期数"].iloc[-1])))

    ranked_cache = {}
    out = {"profile": args.profile, "periods": args.periods,
           "prize_mode": args.prize_mode, "strategies": {}, "controls": {}}

    for gc in [int(x) for x in args.group_counts.split(",")]:
        for q in [float(x) for x in args.quantiles.split(",")]:
            r = bt.run_n9_signal_gate_backtest(
                data_asc, args.profile, periods=args.periods,
                front_group_count=gc, signal_quantile=q,
                signal_lookback=args.lookback,
                min_signal_history=args.min_signal_history,
                rf_model_static=rf_model, rf_meta=rf_meta,
                score_windows=(10, 30, 100), ranked_cache=ranked_cache)
            key = "{}_q{:.2f}_g{}".format(args.profile, q, gc)
            out["strategies"][key] = _clean(r)
            print("  {:<28s} bet={:>4d}/{:>4d} cost={:>8d} roi={:>8.4f} "
                  "roi_t={:>8.4f} any3/bet={:.4f}".format(
                      key, r["bet_periods"], r["total_periods"], r["total_cost"],
                      r["roi"], r["roi_truncated"],
                      r["any_group_hit3_or_more_rate_per_bet"]))

            c3 = bt.run_random_same_bet_count_baseline(
                data_asc, periods=args.periods,
                bet_periods=r["bet_periods"], front_group_count=gc,
                n_trials=args.n_trials, rng_seed=args.seed)
            out["controls"][key + "__C3_same_bet_count"] = _clean(c3)
            c1 = bt.run_random_3x5plus12_baseline(
                data_asc, periods=args.periods, n_trials=args.n_trials,
                rng_seed=args.seed) if gc == 3 else None
            if c1 is not None:
                out["controls"][key + "__C1_uniform_3groups"] = _clean(c1)

            def _mc_precision(val, mean, std):
                """仅表示"在固定样本下模拟均值算得有多准"，**不是显著性检验**。"""
                return (val - mean) / (std / np.sqrt(args.n_trials)) if std > 0 else float("nan")

            # 策略是**单一确定路径**，不是 n_trials 次随机试验的均值。
            # 因此 (策略值 − 对照均值) / MC均值标准误 会随 n_trials 增大而放大，
            # 不能作为显著性证据。这里保留为"描述性"字段并显式改名。
            out["strategies"][key]["vs_C3"] = {
                "mc_precision_only_roi_truncated": _mc_precision(
                    r["roi_truncated"], c3["roi_truncated_mean"], c3["roi_truncated_std"]),
                "mc_precision_only_any3_per_bet": _mc_precision(
                    r["any_group_hit3_or_more_rate_per_bet"],
                    c3["any3_per_bet_mean"], c3["any3_per_bet_std"]),
                "significance_note": "以上两个字段衡量模拟精度，不是泛化检验；"
                                     "策略与对照的差异幅度请直接看均值差。",
                "C3_roi_truncated_mean": c3["roi_truncated_mean"],
                "C3_roi_truncated_std": c3["roi_truncated_std"],
                "C3_any3_per_bet_mean": c3["any3_per_bet_mean"],
                "C3_any3_per_bet_std": c3["any3_per_bet_std"],
            }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("written:", args.out)


if __name__ == "__main__":
    main()

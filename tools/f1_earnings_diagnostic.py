# -*- coding: utf-8 -*-
"""F1 收益分层诊断（次要诊断，F1-3）。

**收益只作分层诊断，不用于选择模型**（协议：主指标为按期 Brier）。

关键口径（V3_HANDOFF_REVIEW_2026-10-14.md）：
* 必须**显式**使用 V3 修正快照
  ``OfficialSnapshot(draws_csv=..., levels_csv=...)`` 并传入支持 snapshot 的计奖入口；
  不得静默回落 ``data/dlt/official_dlt_*.csv``（旧公共快照含半元截断）。
* 每期固定一组 5+12（前区 Top-5 + 后区全包 66 注），成本 132 元；
  单张单期 132 元满足派奖门槛，按全额派奖计。

用法::

    python -m tools.f1_earnings_diagnostic \
        --run-dir outputs/f1_front_probability_<...> \
        --snapshot-dir outputs/2026-10-13_5plus12_v3_bonus/data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                                        # noqa: E402
from tools import profit_accounting as pa                       # noqa: E402
from tools.five_plus12_math import COST_PER_GROUP, no_jackpot_table  # noqa: E402

DEFAULT_SNAPSHOT_DIR = os.path.join("outputs", "2026-10-13_5plus12_v3_bonus", "data")
JACKPOT_CONDS = ((5, 2), (5, 1))


def load_predictions(run_dir: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    with open(os.path.join(run_dir, "predictions.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.setdefault(r["candidate"], []).append(r)
    for c in out:
        out[c].sort(key=lambda r: r["issue"])
    return out


def _drawdown(equity: np.ndarray) -> float:
    """资金曲线**从 0 起算**的最大回撤（含第一期亏损）。"""
    curve = np.concatenate([[0.0], equity])
    peak = np.maximum.accumulate(curve)
    return float(np.max(peak - curve))


def _max_consecutive_losses(profits: np.ndarray) -> int:
    best = cur = 0
    for p in profits:
        cur = cur + 1 if p < 0 else 0
        best = max(best, cur)
    return int(best)


def settle_candidate(records: List[Dict], snapshot, no_jackpot: bool = False) -> Dict:
    """逐期结算一个候选的 Top-5；返回汇总与逐期序列。"""
    gross_sum = tax_sum = cost_sum = 0
    profits: List[float] = []
    hits: List[int] = []
    by_level: Dict[str, int] = {}
    estimate_periods = 0

    for r in records:
        issue = int(r["issue"])
        y = np.asarray(r["y"], dtype=int)
        top5 = [int(t) - 1 for t in r["top5"]]
        k = int(y[top5].sum())
        hits.append(k)

        if no_jackpot:
            tbl = no_jackpot_table(pr.get_prize_table(issue, mode="official",
                                                      snapshot=snapshot))
            rule = {"table": tbl, "label": "official_v3_no_jackpot",
                    "is_estimate": False}
        else:
            rule = "official"

        res = pa.settle_period(issue, [k], rule_snapshot=rule,
                               snapshot=snapshot, costs={})
        gross_sum += int(res["gross"])
        tax_sum += int(res["tax"])
        cost_sum += int(res["cost"])
        profits.append(float(res["net"] - res["cost"]))
        if res.get("is_estimate"):
            estimate_periods += 1
        for lv, v in (res.get("gross_by_level") or {}).items():
            by_level[lv] = by_level.get(lv, 0) + int(v)

    profits_arr = np.asarray(profits, dtype=float)
    equity = np.cumsum(profits_arr)
    net = gross_sum - tax_sum
    return {
        "n_periods": len(records),
        "cost_total": cost_sum,
        "gross_total": gross_sum,
        "tax_total": tax_sum,
        "net_total": net,
        "profit_total": net - cost_sum,
        "roi": (net - cost_sum) / cost_sum if cost_sum else float("nan"),
        "roi_no_jackpot": None if not no_jackpot else (
            (net - cost_sum) / cost_sum if cost_sum else float("nan")),
        "max_drawdown": _drawdown(equity),
        "max_consecutive_loss_periods": _max_consecutive_losses(profits_arr),
        "mean_front_hit": float(np.mean(hits)) if hits else float("nan"),
        "k_distribution": {str(k): int(sum(1 for h in hits if h == k)) for k in range(6)},
        "gross_by_level": by_level,
        "estimate_periods": estimate_periods,
        "equity_final": float(equity[-1]) if len(equity) else 0.0,
        "period_profits": [float(x) for x in profits_arr],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="F1 收益分层诊断（仅模拟）")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    d = os.path.join(ROOT, args.snapshot_dir) if not os.path.isabs(args.snapshot_dir) \
        else args.snapshot_dir
    draws = os.path.join(d, "official_dlt_draws.csv")
    levels = os.path.join(d, "official_dlt_prize_levels.csv")
    if not (os.path.exists(draws) and os.path.exists(levels)):
        print("拒绝运行：未找到 V3 修正快照 {}（不使用旧快照代替）".format(d),
              file=sys.stderr)
        return 2

    snap = pr.OfficialSnapshot(draws_csv=draws, levels_csv=levels)
    if not snap.ok:
        print("拒绝运行：V3 快照无法解析", file=sys.stderr)
        return 2

    run_dir = os.path.join(ROOT, args.run_dir) if not os.path.isabs(args.run_dir) \
        else args.run_dir
    preds = load_predictions(run_dir)

    out = {
        "snapshot_used": {"draws_csv": os.path.relpath(draws, ROOT),
                          "levels_csv": os.path.relpath(levels, ROOT),
                          "issues": len(snap.issues),
                          "issue_min": int(min(snap.issues)),
                          "issue_max": int(max(snap.issues))},
        "cost_per_group": COST_PER_GROUP,
        "simulation_only": True,
        "candidates": {},
    }
    for cand, recs in sorted(preds.items()):
        uncov = [r["issue"] for r in recs if not snap.has(int(r["issue"]))]
        base = settle_candidate(recs, snap, no_jackpot=False)
        nj = settle_candidate(recs, snap, no_jackpot=True)
        base["roi_no_jackpot"] = nj["roi"]
        base["gross_no_jackpot_total"] = nj["gross_total"]
        base["jackpot_component"] = base["gross_total"] - nj["gross_total"]
        base["periods_without_official_snapshot"] = len(uncov)
        out["candidates"][cand] = base

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

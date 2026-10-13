# -*- coding: utf-8 -*-
"""生成账本样例：把冻结策略在指定窗口的结果按**完整净收益账本**结算。

用途
----
T2 要求"定向测试通过后保存账本样例"。本脚本产出一个可直接审计的三本账：
税前回收 → 扣除适用税款 → 投注净利润 → （项目净利润另列，未知费用计缺口）。

同时输出**按奖级分解**，供 T3 的奖金机会地图复用。

用法::

    python tools/build_ledger_sample.py --run-dir <快照目录> \
        --strategy s1 --issue-from 26043 --issue-to 26116 \
        --out outputs/<run>/corrections/ledger_s1_oos.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def s1_group_hits(data, indices, sr):
    """返回 {issue: [各组前区命中数]}（S1 = 现有 config 单组策略）。"""
    out = {}
    for idx in indices:
        issue = int(data.iloc[idx]["期数"])
        groups = sr.s1_groups("ledger", data, idx, top_n=10, play=1,
                              score_mode="stat_miss_penalty")
        drawn = set(int(data.iloc[idx][c]) for c in
                    ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"])
        out[issue] = [len(set(g) & drawn) for g in groups]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--strategy", default="s1", choices=["s1", "n9_441"])
    ap.add_argument("--issue-from", type=int, default=26043)
    ap.add_argument("--issue-to", type=int, default=26116)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tax", default="auto", choices=["auto", "merged", "none"],
                    help="auto=按期判定（2024-09-01 起为现行合并口径）")
    args = ap.parse_args()

    import prize_rules as pr
    from feature_engineering import load_dlt_history
    from inference_plus import maybe_load_rf_model
    from tools.profit_accounting import SimAccount, TAX_2024_MERGED, NO_TAX
    from tools.eval_5plus12_frozen import StrategyRunner

    data = load_dlt_history(os.path.join(args.run_dir, "data", "dlt_full_7001_26116.csv"))
    indices = [i for i in range(len(data))
               if args.issue_from <= int(data.iloc[i]["期数"]) <= args.issue_to]
    rf, meta = maybe_load_rf_model()
    sr = StrategyRunner(rf, meta)

    if args.strategy == "s1":
        hits = s1_group_hits(data, indices, sr)
        acc_id = "S1_main_5plus12"
    else:
        hits = {}
        for i in indices:
            issue = int(data.iloc[i]["期数"])
            groups = sr.s2_groups("ledger", data, i, "n9_mix_441", 3)
            drawn = set(int(data.iloc[i][c]) for c in
                        ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"])
            hits[issue] = [len(set(g) & drawn) for g in groups]
        acc_id = "S2_n9_mix_441"

    tax_policy = {"merged": TAX_2024_MERGED, "none": NO_TAX}.get(args.tax)
    acc = SimAccount(acc_id, snapshot=pr.get_snapshot(), tax_policy=tax_policy)
    for issue in sorted(hits):
        acc.settle(issue, hits[issue])

    # 项目级费用：未知项必须列缺口，不得默认为 0
    acc.add_external_cost(0, "研发现金支出", note="预算未确认", known=False)

    sample = {
        "account_id": acc_id,
        "window": [args.issue_from, args.issue_to],
        "strategy": args.strategy,
        "tax_setting": args.tax,
        "period_rows": acc.ledger(),
        "totals": acc.totals(),
        "external_costs": acc.external_costs,
        "notes": [
            "每期按『同一人同一期同一游戏全部奖金合并』判定一次中奖收入。",
            "一次中奖收入 ≤10000 元暂免；>10000 元按全额 20% 计税"
            "（公告2024年第12号，2024-09-01 起执行）。",
            "奖金表来源逐期标注；估算回退在 labels.warning 中标出。",
        ],
    }
    # 按奖级分解
    by_level = {}
    for r in acc.ledger():
        for lv, v in r["gross_by_level"].items():
            by_level[lv] = by_level.get(lv, 0) + v
    sample["gross_by_level_total"] = by_level
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    t = sample["totals"]
    print("账户 {} 窗口 {}..{}".format(acc_id, args.issue_from, args.issue_to))
    print("  期数={} 组数合计={}".format(t["n_periods"], t["n_groups_total"]))
    print("  税前回收={} 税={} 净={} 票款={} 净利润={} ROI={}".format(
        t["gross"], t["tax"], t["net"], t["bet_cost"], t["profit"],
        None if t["roi"] is None else round(t["roi"], 4)))
    print("  最大回撤={} 最长连亏={} 盈利期数={}/{}".format(
        t["max_drawdown_from_zero"], t["max_consecutive_loss"],
        t["profit_positive_periods"], t["periods"]))
    print("  项目净利润（仅计已知费用）={}；缺口={}".format(
        t["project_profit_known_only"], t["project_profit_gap"] or "无"))
    print("  按奖级分解：", json.dumps(by_level, ensure_ascii=False))
    print("written:", args.out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""D5：**真正执行**预算约束下的政策模拟（PLAN §5）。

预登记规则（在运行之前固定，不得事后调整）
------------------------------------------
1. **每期组数固定**，**不允许动态减少组数**。理由：自适应仓位是一种新的
   策略，属于本轮范围之外；允许它会把"预算约束"变成"择时仓位"。
2. **参与日期规则固定**：政策在窗口内**每期都尝试参与**，不择时。
3. **先检查、后参与**：每期开始前计算
   `已实现亏损 + 未结算暴露 + 本期成本 <= 限额`。
   不满足则该政策**永久停止**参与（首次不足的期号记为 `stopped_at_issue`），
   之后的期次一律记 0 组、0 收支。
   理由：跳过本期、等下一期再试，本质上是一种择时；本轮不允许.
4. **当期立即结算**（同一期同一账户合并计税）。
5. 三个指标**分开报告，不得混用**：

   * `budget_stop`        —— 因额度不足停止（含期号）
   * `touched_limit`     —— 停止前累计亏损**达到**限额（`>= limit`）
   * `over_limit`        —— 累计亏损**严格超过**限额（`> limit`）；
     按规则 3，该值**恒为 0**；若非 0 即说明预算检查存在缺陷。

用法::

    python tools/sim_budget_v2.py --rows <rows_d4_policies.json> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.profit_budget import can_allocate, max_affordable   # noqa: E402

RULES = {
    "group_count": "fixed_per_period（不允许动态减少组数）",
    "participation": "try_every_period_until_stopped（不择时）",
    "check_order": "check_budget_before_betting",
    "on_insufficient": "stop_permanently（不跳过后再试）",
    "settlement": "immediate",
    "indicators_separate": ["budget_stop", "touched_limit", "over_limit"],
}


def simulate_policy(rows, loss_limit, allow_reduce_groups=False, unit_cost=132):
    """在预算约束下逐期模拟一个政策。

    Args:
        rows: [{"issue", "n_groups", "cost", "profit", ...}]，按时间升序
        loss_limit: 亏损硬上限
        allow_reduce_groups: 必须为 False（预登记规则 1）；
            置 True 仅用于敏感性对照，且会在结果中标注为违反预登记规则。
    """
    realized_net = 0
    stopped_at = None
    touched = False
    over_limit = 0
    path = []
    bet_periods = 0
    skipped_periods = 0
    total_cost = 0
    total_profit = 0

    for r in rows:
        issue = int(r["issue"])
        base_cost = int(r["cost"])
        n_groups = int(r["n_groups"])

        if stopped_at is not None:
            skipped_periods += 1
            path.append({"issue": issue, "bet": False, "cost": 0, "profit": 0,
                         "realized_net": realized_net, "reason": "stopped"})
            continue

        loss_to_date = max(0, -realized_net)
        if can_allocate(loss_to_date, 0, base_cost, loss_limit):
            # 参与：按该期**实际**实现的收益结算
            realized_net += int(r["profit"])
            bet_periods += 1
            total_cost += base_cost
            total_profit += int(r["profit"])
            path.append({"issue": issue, "bet": True, "cost": base_cost,
                         "profit": int(r["profit"]),
                         "realized_net": realized_net, "reason": ""})
            if -realized_net >= loss_limit:
                touched = True
            if -realized_net > loss_limit:
                over_limit += 1
            continue

        # 额度不足
        if allow_reduce_groups:
            affordable = max_affordable(loss_to_date, 0, loss_limit, unit_cost)
            if 0 < affordable < n_groups:
                frac = affordable / n_groups
                profit = int(round(r["profit"] * frac))
                cost = affordable * unit_cost
                realized_net += profit
                bet_periods += 1
                total_cost += cost
                total_profit += profit
                path.append({"issue": issue, "bet": True, "cost": cost,
                             "profit": profit, "realized_net": realized_net,
                             "reason": "reduced_groups({})".format(affordable)})
                if -realized_net >= loss_limit:
                    touched = True
                if -realized_net > loss_limit:
                    over_limit += 1
                continue
        stopped_at = issue
        skipped_periods += 1
        path.append({"issue": issue, "bet": False, "cost": 0, "profit": 0,
                     "realized_net": realized_net,
                     "reason": "insufficient_budget"})

    return {
        "loss_limit": loss_limit,
        "periods_in_window": len(rows),
        "bet_periods": bet_periods,
        "skipped_periods": skipped_periods,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "roi": (total_profit / total_cost) if total_cost else None,
        "final_realized_net": realized_net,
        # --- 三个必须分开的指标 ---
        "budget_stop": {"stopped": stopped_at is not None, "stopped_at_issue": stopped_at},
        "touched_limit": bool(touched),
        "over_limit_violations": int(over_limit),
        "reduced_groups_used": bool(allow_reduce_groups),
        "path": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limits", default="500,1000")
    ap.add_argument("--policies", default="P_skip,P1_a,P1_b,P2,P3")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    d = json.load(open(args.rows, encoding="utf-8"))
    rows_by_pol = d["rows"]
    out = {"window": d.get("window"), "rules": RULES, "results": {}}

    for limit in [int(x) for x in args.limits.split(",")]:
        blk = {}
        for pol in args.policies.split(","):
            rows = rows_by_pol.get(pol)
            if rows is None:
                continue
            r = simulate_policy(rows, limit)
            blk[pol] = {k: v for k, v in r.items() if k != "path"}
            blk[pol]["n_bet_path_tail"] = r["path"][-1] if r["path"] else None
        out["results"]["limit_{}".format(limit)] = blk

    # 允许减仓的对照（明确标注违反预登记规则）
    for limit in [int(x) for x in args.limits.split(",")]:
        tbl = {}
        for pol in args.policies.split(","):
            rows = rows_by_pol.get(pol)
            if rows is None:
                continue
            r = simulate_policy(rows, limit, allow_reduce_groups=True)
            tbl[pol] = {k: v for k, v in r.items() if k != "path"}
        out["results"]["limit_{}_ALLOW_REDUCE_SENSITIVITY".format(limit)] = tbl

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    for lim, blk in out["results"].items():
        if "ALLOW_REDUCE" in lim:
            continue
        print("=== {} ===".format(lim))
        print("  {:<8s} {:>6s} {:>7s} {:>9s} {:>10s} {:>7s} {:>9s} {:>6s} {:>6s}".format(
            "政策", "参与期", "跳过期", "成本", "净利", "ROI", "停止期号", "触及", "超限"))
        for pol, r in blk.items():
            print("  {:<8s} {:>6d} {:>7d} {:>9d} {:>10d} {:>7s} {:>9s} {:>6s} {:>6d}".format(
                pol, r["bet_periods"], r["skipped_periods"], r["total_cost"],
                r["total_profit"],
                "n/a" if r["roi"] is None else "{:.4f}".format(r["roi"]),
                str(r["budget_stop"]["stopped_at_issue"]),
                "是" if r["touched_limit"] else "否", r["over_limit_violations"]))
    print("written:", args.out)


if __name__ == "__main__":
    main()

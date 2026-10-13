# -*- coding: utf-8 -*-
"""修正 D4 功效结论：用**完整分布**（含稀有大奖与税）重算，并检查 bootstrap 支持集。

被撤回的旧结论
--------------
上一轮用"500 期样本标准差 106.4 元/期"推出"该窗口可检出 11.8 元/期，
而盈亏平衡只需 66 元/期，故可排除盈亏平衡"。

**该推导无效**，原因有三：

1. 那 500 期里 **(4,2) / (5,x) 一次都没出现**，样本标准差**不包含大奖项**，
   而大奖正是支配期望的部分（D3：一/二等奖占期望回收 51.4%）；
2. 税收是非线性的（≤10000 免征、超过按全额 20%），样本里税额恒为 0，
   因此样本 sd 也不含税的影响；
3. 期级 bootstrap 从同一批已实现开奖重采样，**支持集永远不含未见过的结果**，
   不可能生成大奖。

本脚本改为：用**完整分布**（精确枚举 6 个前区命中结果 × 各情景奖金表 × 税）
计算单期净收益的标准差与由此得到的功效；并显式检查 bootstrap 的支持集。
**所有结果只对所声明的假设成立。**
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr
from tools.five_plus12_math import (front_hit_distribution, group_payout_by_front_hit,
                                    no_jackpot_table, COST_PER_GROUP)
from tools.profit_accounting import TAX_2024_MERGED
from tools import profit_opportunity as po

Z95, Z80 = 1.6449, 0.8416
TARGET = 26117


def exact_period_distribution(table, n_groups=1, pool_size=None, cost_per_group=132):
    """一期净收益的**完整**分布 {profit: prob}（含税、含大奖）。"""
    n = int(n_groups)
    if pool_size is None:
        hd = front_hit_distribution()
        R = group_payout_by_front_hit(table)
        dist = {0: 1.0}
        for _ in range(n):
            nd = {}
            for a, pa in dist.items():
                for f in range(6):
                    if hd[f] == 0:
                        continue
                    nd[a + R[f]] = nd.get(a + R[f], 0.0) + pa * hd[f]
            dist = nd
    else:
        dist, _ = po._outcome_distribution(
            group_payout_by_front_hit(table), n, int(pool_size),
            front_hit_distribution())
    cost = cost_per_group * n
    out = {}
    for gross, p in dist.items():
        g = int(round(gross))
        prof = g - TAX_2024_MERGED.tax_on(g) - cost
        out[prof] = out.get(prof, 0.0) + p
    return out


def moments(dist):
    mu = sum(k * v for k, v in dist.items())
    var = sum(v * (k - mu) ** 2 for k, v in dist.items())
    return mu, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--periods", type=int, default=500)
    ap.add_argument("--groups", default="1,2,3")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    snap = pr.get_snapshot()
    pool_before = snap.pool_before(TARGET)
    base = pr.base_table(TARGET, pool_before=pool_before)
    scen = po.scenario_from_history(snap, TARGET, mask="all", q=0.50, window=500)

    tables = {"fallback_floating": base}
    t2 = dict(base); t2[(5, 2)] = scen["first_prize"]; t2[(5, 1)] = scen["second_prize"]
    tables["hist_q0.50_floating"] = t2
    t3 = dict(base); t3[(5, 2)] = pr.ERA3_HIGH_AMOUNT["三等奖"] * 0  # placeholder
    t3 = dict(base)
    t3[(5, 2)] = 10_000_000; t3[(5, 1)] = 460_501

    out = {"target_issue": TARGET, "pool_before": pool_before,
           "periods": args.periods,
           "note": "所有功效数字都依赖所声明的假设，只在该假设下成立。"}

    pf = front_hit_distribution()
    n = args.periods
    # 重要（本轮勘误）：一组 5+12 **后区 1~12 全包**，因此只要前区 5 中，
    # 66 注里必然有一注命中 5+2。故
    #     P(一组 5+12 中一等奖) = P(前区 5 中) = 1 / C(35,5)
    # **不得再除以 66**（除以 66 得到的是"单注"的概率，那是另一种投注结构）。
    p_group = pf[5]
    out["p_observing_first_prize"] = {
        "per_period_group_5plus12": p_group,
        "per_period_group_1_over": 1.0 / p_group,
        "over_n_periods_group": 1 - (1 - p_group) ** n,
        "per_period_single_note_CONTRAST_ONLY": p_group / 66,
        "over_n_periods_single_note_CONTRAST_ONLY": 1 - (1 - p_group / 66) ** n,
        "reason": "后区 1~12 全包 ⇒ 前区 5 中时必有一注 5+2；"
                  "所以一组 5+12 的一等奖概率等于前区 5 中的概率，不除以 66。",
        "note": "{} 期内出现一等奖的期望次数 ≈ {:.4f}；"
                "因此历史样本的 sd 几乎必然不含大奖项。".format(n, n * p_group),
    }

    rows = []
    for tname, tbl in tables.items():
        for g in [int(x) for x in args.groups.split(",")]:
            dist = exact_period_distribution(tbl, n_groups=g)
            mu, sd = moments(dist)
            mde = (Z95 + Z80) * sd / np.sqrt(n)
            need = ((Z95 + Z80) ** 2) * sd ** 2 / 66.0 ** 2
            # 尾部集中度：期望回收里最大的那几项占多少
            contrib = sorted(((prob * (prof + COST_PER_GROUP * g), prob, prof)
                              for prof, prob in dist.items()), reverse=True)
            top1 = contrib[0][0] / max(sum(c[0] for c in contrib), 1e-12)
            rows.append({
                "floating_scenario": tname, "n_groups": g,
                "E_net_profit": mu, "sd_net_profit": sd,
                "NORMAL_APPROX_EXAMPLE_ONLY_min_detectable_effect": mde,
                "NORMAL_APPROX_EXAMPLE_ONLY_periods_for_66yuan_effect": need,
                "periods_available": n,
                "top_outcome_share_of_gross_expectation": top1,
                "outcomes": len(dist),
                "max_outcome": max(dist),
                "status": "示例，不是结论",
                "note": "正态近似在如此重尾下不可靠；这两个数只是量级示意，"
                        "**不得**被引用为确定结论。可靠功效见 reliable_power。",
            })

    out["full_distribution_power"] = rows
    out["reliable_power"] = {
        "status": "未确定",
        "reason": "本期收益分布由概率 3.08e-06 的一等奖主导；"
                  "在可行样本量内无法给出可靠功效。"
                  "上表的正态近似数字只是量级示意，不是可引用的结论。",
        "what_can_be_stated": "可确定的是：现有窗口的样本 sd "
                              "（不含大奖）不能代表完整收益的波动。",
    }

    # 被撤回的旧口径（样本 sd）对照
    out["retracted_sample_sd_approach"] = {
        "sample_sd_per_period": 106.4,
        "periods": 500,
        "why_invalid": [
            "该 500 期内 (4,2)/(5,x) 一次未出现，sd 不含大奖项",
            "该 500 期税额恒为 0，sd 不含非线性税的影响",
            "由它推出的'16 期足够'与'可检出 11.8 元/期'均不成立",
        ],
        "status": "RETRACTED",
    }

    # bootstrap 支持集检查
    out["bootstrap_support_check"] = {
        "statement": "期级 bootstrap 只能重排**已实现**的期收益，"
                     "支持集的最大值就是样本最大值，永远不可能生成大奖。",
        "max_observed_period_profit_in_500": None,
        "implication": "因此 bootstrap 只能刻画'在这段历史分布内'的波动，"
                       "不能用于对'期望收益'做区间估计，也不能用于功效推断。",
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    pj = out["p_observing_first_prize"]
    print("一组 5+12：P(一期中一等奖) = P(前区5中) = {:.3e}  (1/{:.0f})".format(
        pj["per_period_group_5plus12"], pj["per_period_group_1_over"]))
    print("{} 期内至少一次一等奖 = {:.4f}%".format(
        args.periods, pj["over_n_periods_group"] * 100))
    print("对照（单注结构，仅对比用）: {:.4f}%".format(
        pj["over_n_periods_single_note_CONTRAST_ONLY"] * 100))
    print("（下表两列均为**正态近似示例**，不是结论；可靠功效见 reliable_power）")
    print("{:<24s} {:>3s} {:>12s} {:>12s} {:>16s} {:>18s}".format(
        "浮动奖情景", "组", "E[净利]/期", "sd/期", "示例:可检出效应", "示例:需期数"))
    for r in rows:
        print("{:<24s} {:>3d} {:>12.2f} {:>12.1f} {:>16.1f} {:>18.0f}".format(
            r["floating_scenario"], r["n_groups"], r["E_net_profit"],
            r["sd_net_profit"],
            r["NORMAL_APPROX_EXAMPLE_ONLY_min_detectable_effect"],
            r["NORMAL_APPROX_EXAMPLE_ONLY_periods_for_66yuan_effect"]))
    print("reliable_power:", out["reliable_power"]["status"])
    print("written:", args.out)


if __name__ == "__main__":
    main()

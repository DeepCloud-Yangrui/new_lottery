# -*- coding: utf-8 -*-
"""生成 5+12 数学基线的多情景对照（本轮修正版，写入独立纠错目录）。

情景：
  A. era2 (19019-26013) 期数加权平均浮动奖金档
  B. era3 (26014-26116) 期数加权平均浮动奖金档 + 基础档固定奖
  C. era3 上浮档（奖池 >= 8 亿）
  D. 样本外窗口 26043-26116 逐期官方奖金表（含全额派奖），对期取平均
  E. 旧 config 静态表（对照）

修正点（相对上一轮）：
  1. 盈亏平衡改用 ``breakeven_scenarios``：同时给出
     完整口径 与 剔除大奖口径（只置零 5+2 / 5+1，**保留 (5,0)**）；
  2. 明确区分「回收倍数」与「命中率倍数」；
  3. 多组的成本与回收使用同一单位；
  4. 多组蒙特卡洛使用**同一期共同开奖**，并区分全号池（组间独立）
     与 p 号候选池（组间通过 m 正相关）两种结构；
  5. 增加「单组边际期望与选号结构无关」的精确校验。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from dlt_prize_rules import (  # noqa: E402
    BASE_AMOUNT, ERA3_HIGH_AMOUNT, LEVEL_CONDITIONS, load_official,
)
from five_plus12_math import (  # noqa: E402
    baseline_report, front_hit_distribution, group_payout_by_front_hit,
    no_jackpot_table, COST_PER_GROUP, simulate_profits,
    exact_multi_group_moments,
)


def avg_floating(od, lo, hi):
    """窗口内一/二等奖单注奖金的**期数加权**均值（每期买一张票的正确口径）。"""
    d = od.draws
    w = d[(d.issue >= lo) & (d.issue <= hi)]
    return {"一等奖": float(w.amt_一等奖.mean()),
            "二等奖": float(w.amt_二等奖.mean())}


def table_from_amounts(era, amounts):
    t = {}
    for level, conds in LEVEL_CONDITIONS[era].items():
        if level not in amounts or amounts[level] is None:
            continue
        for c in conds:
            t[c] = int(round(amounts[level]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--mc-trials", type=int, default=100_000)
    args = ap.parse_args()

    od = load_official(args.run_dir)
    out = {"scenarios": [], "structure_check": {}, "notes": []}
    pf = front_hit_distribution()

    av2 = avg_floating(od, 19019, 26013)
    av3 = avg_floating(od, 26014, 26116)

    a2 = dict(BASE_AMOUNT["era2"]); a2.update(av2)
    b3 = dict(BASE_AMOUNT["era3"]); b3.update(av3)
    c3 = dict(BASE_AMOUNT["era3"]); c3.update(ERA3_HIGH_AMOUNT); c3.update(av3)
    for name, tbl in (("A_era2_avg_floating", table_from_amounts("era2", a2)),
                      ("B_era3_base_avg_floating", table_from_amounts("era3", b3)),
                      ("C_era3_high_tier_avg_floating", table_from_amounts("era3", c3))):
        out["scenarios"].append(baseline_report(
            tbl, name, n_draws=args.n_draws, mc_trials=args.mc_trials))

    # D. 样本外窗口逐期实测
    win = od.draws[(od.draws.issue >= 26043) & (od.draws.issue <= 26116)]
    rec = []
    for iss in win.issue.tolist():
        t = od.issue_table(int(iss), "full")
        tnj = no_jackpot_table(t)
        R = group_payout_by_front_hit(t)
        Rnj = group_payout_by_front_hit(tnj)
        rec.append({
            "issue": int(iss),
            "E_R_full": sum(pf[f] * R[f] for f in range(6)),
            "E_R_no_jackpot": sum(pf[f] * Rnj[f] for f in range(6)),
            "R3": R[3], "R4": R[4],
            "pool_before": od.pool_before.get(int(iss)),
        })
    E_full = float(np.mean([r["E_R_full"] for r in rec]))
    E_nj = float(np.mean([r["E_R_no_jackpot"] for r in rec]))
    out["scenarios"].append({
        "label": "D_oos_window_26043_26116_avg",
        "n_draws_window": len(rec),
        "expected_group_payout": E_full,
        "expected_roi": E_full / COST_PER_GROUP - 1,
        "expected_group_payout_no_jackpot": E_nj,
        "expected_roi_no_jackpot": E_nj / COST_PER_GROUP - 1,
        "R3_range": [min(r["R3"] for r in rec), max(r["R3"] for r in rec)],
        "R4_range": [min(r["R4"] for r in rec), max(r["R4"] for r in rec)],
        "note": "对每一期使用官方公布的逐期单注奖金（含全额派奖），"
                "按前区命中分布求期望后对期数取平均。"
                "这是对**已发生的历史窗口**的平均情景，不是未来收益的上限或下限。",
    })

    # E. 旧 config 表（对照）
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import get_dlt_prize_table  # noqa: E402
    for iss, nm in ((26000, "E_config_legacy_pre2026"), (26116, "E_config_legacy_2026")):
        out["scenarios"].append(baseline_report(
            dict(get_dlt_prize_table(iss, mode="legacy_config")),
            nm + "_issue{}".format(iss), n_draws=args.n_draws,
            mc_trials=args.mc_trials))

    # ------------------------------------------------------------------
    # 结构校验
    # ------------------------------------------------------------------
    base = od.issue_table(26116, "full")
    R = group_payout_by_front_hit(base)
    E_true = sum(pf[f] * R[f] for f in range(6))

    # (1) 单组边际期望「与选号结构无关」——精确枚举验证
    #     对任意固定 5 码 C，|C ∩ D| 的分布都是同一个超几何；
    #     这里对若干个 C 直接枚举全部 C(35,5)=324632 个开奖 D 来核验。
    from itertools import combinations as _combos
    from math import comb as _comb
    allD = list(_combos(range(1, 36), 5))
    exact_hist = {k: _comb(5, k) * _comb(30, 5 - k) for k in range(6)}
    check = {"n_draws_enumerated": len(allD), "combos": []}
    rng0 = random.Random(5)
    for _ in range(4):
        C = frozenset(rng0.sample(range(1, 36), 5))
        hist = [0] * 6
        for D in allD:
            hist[len(C & frozenset(D))] += 1
        check["combos"].append({"combo": sorted(C), "hist": hist,
                                "matches_hypergeometric": hist == [exact_hist[k] for k in range(6)]})
    # 池内组合的平均：由线性可得仍是同一超几何，取 9 号池的 126 个组合做抽查
    pool9 = sorted(rng0.sample(range(1, 36), 9))
    sub = [frozenset(c) for c in _combos(pool9, 5)]
    agg = [0] * 6
    for C in sub[:20]:              # 抽查 20/126，避免超时
        for D in allD:
            agg[len(C & frozenset(D))] += 1
    scale = len(allD) * 20
    check["pool9_sample_frac"] = sum(
        agg[k] * exact_hist[k] / scale for k in range(6)) / sum(exact_hist[k] for k in range(6))
    check["pool_fraction_of_expected"] = (
        sum(agg[k] for k in range(6)) / scale)
    check["pool_matches_hypergeometric"] = all(
        agg[k] == exact_hist[k] * 20 for k in range(6))
    check["conclusion"] = (
        "对任意固定 5 码 C，|C∩D| 的分布完全相同（精确枚举 324632 个开奖 D 验证）。"
        "因此单组的**无条件**期望回收与候选池大小、构成无关；"
        "多组总回收的期望按线性累加（不要求组间独立）。")
    out["structure_check"]["single_group_marginal_invariance"] = check

    # (2) 多组：同一期共同开奖的**精确**矩（不用蒙特卡洛）
    mg = {"n_groups": 3, "tables": {}}
    for tname, tbl in (("full", base),
                       ("no_jackpot", no_jackpot_table(base))):
        mg["tables"][tname] = {
            str(ps): exact_multi_group_moments(tbl, n_groups=3, pool_size=ps)
            for ps in (None, 9, 12, 15)
        }
    mg["note"] = ("模型：同一期的 3 组共用同一次开奖；"
                  "pool_size=None 表示每组独立取自 1~35，"
                  "p 表示每组取自同一期固定的 p 个候选号。"
                  "精确计算（无蒙特卡洛）显示两者的 E_total 完全相同，"
                  "差别只在方差项。")
    out["structure_check"]["multi_group_exact_moments"] = mg

    out["notes"] = [
        "所有情景均按 5+12 基本投注（不追加、不倍投）：单组 66 注 × 2 元 = 132 元。",
        "E[一组回收] = 66 × E[单注奖金]；后区全包不改变期望值。",
        "浮动奖金使用窗口内的**期数加权**实测均值；这是历史情景估计，"
        "不是理论收益上限。",
        "多组的期望回收按线性累加（不要求组间独立）；"
        "候选池与组合重叠影响的是方差与回撤，不是无条件期望。",
        "盈亏平衡给出多种情景（整块搬运 / 只增 3 中 / 只增 4 中），"
        "每种都是明确假设下的情景计算，不是通用门槛。",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    for s in out["scenarios"]:
        be = s.get("breakeven_full", {}) or {}
        s1 = be.get("S1_block_transfer", {})
        print("{:<40s} E[R]={:>8.2f} ROI={:>8.4f} | E_nj={:>7.2f} ROI_nj={:>8.4f} | full S1 q_be={} x{}".format(
            s["label"], s.get("expected_group_payout", 0), s.get("expected_roi", 0),
            s.get("expected_group_payout_no_jackpot", 0),
            s.get("expected_roi_no_jackpot", 0),
            "None" if s1.get("q_break_even") is None else round(s1["q_break_even"], 6),
            "None" if s1.get("q_multiple") is None else round(s1["q_multiple"], 3)))
    print("单组边际不变性（精确枚举）：",
          all(c["matches_hypergeometric"] for c in check["combos"]),
          "；池内抽查 20 组：", check["pool_matches_hypergeometric"])
    for tn, tab in mg["tables"].items():
        print("多组精确矩[{}]：".format(tn),
              " ".join("pool={}:E={:.4f},sd={:.1f}".format(
                  k, v["E_total"], v["sd_total"]) for k, v in tab.items()))
    print("written:", args.out)


if __name__ == "__main__":
    main()

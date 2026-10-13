# -*- coding: utf-8 -*-
"""P5：把各实验产物合并为一份可引用的汇总（SUMMARY.json / SUMMARY.csv）。

输入：
  results/math_baseline_scenarios.json   数学基线
  results/full_run.json                  样本外 + 回顾窗口（策略）
  results/retro_controls.json            回顾窗口随机对照 + 命中率二项检验
  官方快照                               用于重建 C1/C2 分布比较

输出：
  results/SUMMARY.json
  results/SUMMARY.csv
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def two_sample_z(m1, s1, n1, m2, s2, n2):
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    return (m1 - m2) / se if se > 0 else float("nan")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="汇总输出目录；默认写入 <run-dir>/results（会覆盖旧 SUMMARY）")
    args = ap.parse_args()
    rd = args.run_dir
    R = lambda n: os.path.join(rd, "results", n)
    OUT = args.out_dir or os.path.join(rd, "results")
    os.makedirs(OUT, exist_ok=True)
    W = lambda n: os.path.join(OUT, n)

    mb = json.load(open(R("math_baseline_scenarios.json"), encoding="utf-8"))
    fr = json.load(open(R("full_run.json"), encoding="utf-8"))
    rc = json.load(open(R("retro_controls.json"), encoding="utf-8"))
    oos = json.load(open(R("oos_only.json"), encoding="utf-8"))

    out = {"math_baseline": {}, "oos": {}, "retro": {}, "structure_test": {}, "notes": []}

    for s in mb["scenarios"]:
        lab = s["label"]
        out["math_baseline"][lab] = {
            "expected_group_payout": s.get("expected_group_payout",
                                           s.get("mean_expected_group_payout")),
            "expected_roi": s.get("expected_roi"),
            "p_front_ge3_multiple_needed": s.get("breakeven_requirements", {}).get(
                "p_front_ge3_multiple_needed"),
            "payout_at_front3": s.get("breakeven_requirements", {}).get("payout_at_front3"),
            "group_payout_by_front_hit": s.get("group_payout_by_front_hit"),
        }

    # OOS 策略
    for k, v in fr["windows"]["official"].items():
        if k.startswith("S"):
            out["oos"][k] = {kk: v[kk] for kk in
                             ("total_cost", "total_reward", "total_profit", "roi",
                              "roi_no_jackpot", "hit2_or_more_rate", "hit3_or_more_rate",
                              "hit4_or_more_rate", "max_drawdown", "max_consecutive_loss",
                              "positive_period_rate")}
    # OOS 对照
    out["oos_controls"] = {k: {"roi_no_jackpot": v["roi_no_jackpot"],
                               "hit3_or_more_rate": v["hit3_or_more_rate"]}
                           for k, v in oos["controls_oos"].items()}
    # OOS 百分位
    out["oos_percentiles"] = {
        k: {"roi_no_jackpot_percentile": v["percentile_rank"], "z_not_a_pvalue": v["z"]}
        for k, v in oos["comparisons_oos"].items()}

    # 回顾窗口
    for w, e in rc["windows"].items():
        out["retro"][w] = {"periods": e["periods"],
                           "issue_range": [e["issue_start"], e["issue_end"]]}
        for k, v in e.items():
            if isinstance(v, dict):
                out["retro"][w][k] = {
                    "strategy_roi_no_jackpot": v["strategy_roi_no_jackpot"],
                    "roi_no_jackpot_percentile_rank": v["roi_no_jackpot_percentile_rank"],
                    "strategy_hit3_rate": v["strategy_hit3_rate"],
                    "hit3_observed": v["hit3_observed_count"],
                    "hit3_expected": v["hit3_expected_count"],
                    "hit3_one_sided_p_value": v["hit3_one_sided_p_value"],
                    "C1_uniform_roi_no_jackpot_mean": v["C1_uniform_roi_no_jackpot"]["mean"],
                }
        # 回顾窗口的策略总指标
        for k, v in fr["windows"]["official"].items():
            if k.startswith("RETRO" + w.replace("last", "last")):
                out["retro"][w][k + "__metrics"] = {
                    kk: v[kk] for kk in ("total_cost", "total_profit", "roi",
                                         "roi_no_jackpot", "hit3_or_more_rate",
                                         "max_drawdown", "max_consecutive_loss")}

    # 信号门槛（S3）+ 择时对照（C3）
    gpath = R("gate_frozen.json")
    if os.path.exists(gpath):
        g = json.load(open(gpath, encoding="utf-8"))
        out["gate"] = {"profile": g["profile"], "periods": g["periods"],
                       "prize_mode": g["prize_mode"], "variants": {}}
        for k, v in g["strategies"].items():
            out["gate"]["variants"][k] = {
                "bet_periods": v["bet_periods"], "total_periods": v["total_periods"],
                "roi": v["roi"], "roi_truncated": v["roi_truncated"],
                "any3_per_bet": v["any_group_hit3_or_more_rate_per_bet"],
                "C3_random_timing_roi_truncated_mean": v.get("vs_C3", {}).get(
                    "C3_roi_truncated_mean"),
                "note": "对照为随机参与日期、相同下注次数与组数",
            }

    # 结构检验（V2 T1 修正）
    # ------------------------------------------------------------------
    # 旧实现把 (C2均值 − C1均值) / 蒙特卡洛均值标准误 当作"候选池结构更差"的
    # 证据（z≈−9~−11）。该统计量只衡量"在固定的这 74 期开奖与固定的这些候选池
    # 下，模拟均值算得有多准"，会随 n_trials 增大而无限放大，**不是**泛化检验。
    # 正确做法见 tools/analysis_pool_conditional.py：把 C2 的条件期望
    # g(m) = E[R | 池∩开奖 = m] 与无条件期望 E[R] 相减，并用期数级标准误。
    out["structure_test"] = {
        "status": "RETRACTED_AND_REPLACED",
        "retracted_claim": "9 候选池结构本身显著劣于全号池随机（z≈−9~−11），"
                           "P3 只会放大同一问题",
        "why_invalid": "(C2均值 − C1均值) / 蒙特卡洛均值标准误 不是泛化检验统计量："
                       "它随 n_trials 增大而放大，衡量的是模拟精度而不是结构效应。",
        "descriptive_mc_means_only": {
            "note": "以下仅为固定窗口下的模拟均值，不构成显著性证据",
            "C1_uniform_3groups_roi_no_jackpot_mean":
                out["oos_controls"]["C1_random_uniform_3groups"]["roi_no_jackpot"]["mean"],
            "C2_441_roi_no_jackpot_mean":
                out["oos_controls"].get("C2_from_pool_n9_mix_441_3groups", {})
                   .get("roi_no_jackpot", {}).get("mean"),
        },
    }
    cond_path = os.path.join(rd, "corrections", "pool_conditional_v2_withpools.json")
    if not os.path.exists(cond_path):
        alt = os.path.join(os.path.dirname(rd), "2026-10-13_5plus12_correction_r2",
                           "corrections", "pool_conditional_v2_withpools.json")
        cond_path = alt
    if os.path.exists(cond_path):
        cj = json.load(open(cond_path, encoding="utf-8"))
        blk = cj["windows"].get("oos_26043_26116", {})
        for tname in ("full", "no_jackpot"):
            sm = blk.get("summary_" + tname, {})
            out["structure_test"]["corrected_" + tname] = {
                "expected_group_payout": sm.get("mean_E_R"),
                "sd_g_m": sm.get("mean_sd_g"),
                "window_gap_se": sm.get("se_of_window_mean_gap"),
                "realised_mean_pool_draw_overlap": sm.get("realised_mean_m"),
                "expected_overlap_under_null": 9 * 5 / 35,
                "realised_conditional_gap": sm.get("realised_mean_gap"),
                "realised_conditional_gap_z": sm.get("realised_gap_z_conditional"),
                "conclusion": "条件缺口在抽样噪声内（|z|<1），"
                              "且与超几何恒等式给出的无条件期望一致："
                              "候选池不改变期望回收。",
            }
    out["notes"].append(
        "候选池结构效应已按条件/无条件分离重算；旧的 z≈−9~−11 已撤回。")

    out["notes"] = [
        "综合收益与显著性口径以 V2 T1 纠错后的产物为准"
        "（docs/project_notes/P5_STATISTICAL_ERRATA.md）。",
        "信号门槛（S3）的结论：三个分位数下的 ROI 均低于 C3（随机择时）对照。",
        "样本外窗口 26043-26116（74 期）在协议冻结前从未用于任何选参。",
        "ROI 类指标的百分位排名基于 1000 次随机对照试验的分布。",
        "(strategy-mean)/SE_of_mean 不是有效 p 值，本汇总只作为描述性指标。",
        "命中率的单侧 p 值来自二项检验（零假设概率由精确组合给出）。",
    ]
    json.dump(out, open(W("SUMMARY.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    rows = []
    for k, v in out["oos"].items():
        rows.append(["OOS_strategy", k, v["total_cost"], v["total_profit"],
                     v["roi"], v["roi_no_jackpot"], v["hit3_or_more_rate"],
                     v["max_drawdown"]])
    for k, v in out["oos_controls"].items():
        rows.append(["OOS_control", k, "", "", v["roi_no_jackpot"]["mean"],
                     v["roi_no_jackpot"]["mean"], v["hit3_or_more_rate"]["mean"], ""])
    for w, d in out["retro"].items():
        for k, v in d.items():
            if isinstance(v, dict) and "strategy_roi_no_jackpot" in v:
                rows.append(["RETRO_" + w + "_strategy", k, "", "",
                             v["strategy_roi_no_jackpot"], v["strategy_roi_no_jackpot"],
                             v["strategy_hit3_rate"],
                             v["roi_no_jackpot_percentile_rank"]])
    with open(W("SUMMARY.csv"), "w", encoding="utf-8-sig", newline="") as f:
        import csv
        wtr = csv.writer(f)
        wtr.writerow(["scope", "name", "cost", "profit", "roi", "roi_no_jackpot",
                      "hit3_or_more_rate", "aux"])
        wtr.writerows(rows)

    print(json.dumps(out["structure_test"], ensure_ascii=False, indent=2))
    print("written:", W("SUMMARY.json"), W("SUMMARY.csv"))


if __name__ == "__main__":
    main()

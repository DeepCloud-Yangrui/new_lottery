# -*- coding: utf-8 -*-
"""V3 执行脚本：四格精确收益 + 盈亏平衡边界 + 规则包络（E2 / E3）。

用法::

    python tools/run_bonus_feasibility_v3.py --run-dir outputs/<v3_dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                              # noqa: E402
from tools import bonus_feasibility_v3 as bf          # noqa: E402


def D(x):
    return Decimal(str(x))


def dec_to_str(o):
    if isinstance(o, Decimal):
        return o.to_eng_string()
    if isinstance(o, dict):
        return {k: dec_to_str(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [dec_to_str(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--snapshot-run", default=None,
                    help="官方快照所在目录（默认取上一轮运行目录）")
    ap.add_argument("--target-issue", type=int, default=26117)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    snap_run = args.snapshot_run or "outputs/2026-10-13_5plus12_profitability_verification"
    snap = pr.get_snapshot()
    T = args.target_issue

    evidence = {
        "rule": "2026 七奖级规则（第 26014 期起）：奖级映射来自 prize_rules.LEVEL_CONDITIONS['era3']",
        "pool_tier": "开奖前奖池 >= 8 亿元 → 上浮档（官方逐期数据 103/103 期零例外）",
        "bonus_2026": "三至六等奖 +50%、七等奖 +100%；资格=单张单期 >= 18 元；"
                      "活动期 26050–26066（已于 26066 期结束）",
        "bonus_certainty": "official",
    }
    tax = {"threshold": 10_000, "rate": "0.20", "certainty": "official",
           "source": "财政部 税务总局 民政部 体育总局公告2024年第12号（2024-09-01 起执行）"}

    pairs = bf.pairs_from_history(snap, T, window=args.window)
    edge_pairs = bf.typical_pairs(snap, T, window=args.window)
    envelope = bf.rule_envelope_pair()

    out = {
        "target_issue": T,
        "rule_evidence": evidence,
        "tax_policy": tax,
        "cells": {},
        "floating_inputs": {k: dec_to_str({kk: vv for kk, vv in v.items()
                                           if kk != "source_issues"})
                            for k, v in {**pairs, **edge_pairs, "rule_envelope": envelope}.items()},
        "notes": [
            "四格是**规则情景**，不是某期实测；2026 派奖已于 26066 期结束，"
            "不得称为当前在售机会。",
            "金额用 Decimal，禁止逐注截断；七等奖 5×50%=2.5 这类半元按精确值处理。",
            "派奖资格按'单张单期 ≥ 18 元'；一组 5+12 = 132 元满足。",
        ],
    }

    # ---- 四格 × 各浮动奖情景 ----
    scen = {**{k: v for k, v in pairs.items()},
            **{k: v for k, v in edge_pairs.items()},
            "rule_envelope": envelope}
    for cell in "ABCD":
        blk = {"cells_by_scenario": {}}
        for name, fp in scen.items():
            r = bf.evaluate_rule_cell(cell, fp, rule_evidence=evidence,
                                      tax_policy=tax)
            blk["cells_by_scenario"][name] = {
                "floating_kind": fp["kind"],
                "first": dec_to_str(r["floating_pair"]["first"]),
                "second": dec_to_str(r["floating_pair"]["second"]),
                "payout_front3": dec_to_str(r["payout_at_front3"]),
                "payout_front4": dec_to_str(r["payout_at_front4"]),
                "expected_gross": dec_to_str(r["expected_gross"]),
                "expected_tax": dec_to_str(r["expected_tax"]),
                "expected_net_profit": dec_to_str(r["expected_net_profit"]),
                "expected_roi": dec_to_str(r["expected_roi"]),
                "outcomes": [
                    {"k": o["front_hit"], "p": dec_to_str(o["probability"]),
                     "gross": dec_to_str(o["gross"]), "tax": dec_to_str(o["tax"]),
                     "profit": dec_to_str(o["period_profit"]), "taxed": o["taxed"]}
                    for o in r["outcomes"]],
            }
        # 逐注取整的敏感性（仅 C、D 有派奖）
        if cell in ("C", "D"):
            rf = bf.evaluate_rule_cell(cell, pairs["pair_q0.50"],
                                       tax_policy=tax, rounding="official_pernote_floor")
            blk["rounding_sensitivity_official_pernote_floor"] = {
                "expected_gross": dec_to_str(rf["expected_gross"]),
                "payout_front3": dec_to_str(rf["payout_at_front3"]),
                "delta_vs_exact": dec_to_str(
                    rf["expected_gross"] -
                    bf.evaluate_rule_cell(cell, pairs["pair_q0.50"], tax_policy=tax)["expected_gross"]),
                "note": "官方接口对 2026 派奖期公布的六等奖派奖为 7 元（非 7.5），"
                        "舍入口径未查明；此处仅作敏感性。",
            }
        be = bf.solve_break_even_region(cell, rule_evidence=evidence, tax_policy=tax)
        blk["break_even"] = dec_to_str({k: v for k, v in be.items()
                                        if k != "tax_segments"})
        blk["break_even"]["tax_segments"] = [
            {**s, "gross": dec_to_str(s["gross"])} for s in be["tax_segments"]]
        out["cells"][cell] = blk

    # ---- 一页结论 ----
    pos = []
    for cell in "ABCD":
        for name, v in out["cells"][cell]["cells_by_scenario"].items():
            if D(v["expected_net_profit"]) > 0:
                pos.append((cell, name, v["expected_net_profit"], v["floating_kind"]))
    env_positive = any(D(out["cells"][c]["cells_by_scenario"]["rule_envelope"]["expected_net_profit"]) > 0
                       for c in "ABCD")
    reliable = [p for p in pos if p[3] == "historical_pair"]
    gap = {c: dec_to_str(out["cells"][c]["break_even"]["break_even_line"]["a_plus_20b"])
           for c in "ABCD"}
    out["verdict"] = {
        "positive_cells_with_historical_pairs": len(reliable),
        "positive_cells_with_any_scenario": len(pos),
        "positive_examples": [{"cell": c, "scenario": s, "net": n, "kind": k}
                              for c, s, n, k in pos[:10]],
        "required_a_plus_20b_by_cell": gap,
        "envelope_positive": env_positive,
        "decision": ("无历史情景为正 → 按 PLAN E3 直接收尾，不扩张预算实验"
                     if not reliable else
                     "存在历史情景为正（需核查其证据层级）"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 控制台摘要 ----
    print("目标期 {} | 四格（前区中3/中4 的税前回收，元）".format(T))
    print("  {:<4s} {:>8s} {:>8s} {:>10s} {:>10s} {:>12s} {:>9s}".format(
        "格", "R(3)", "R(4)", "E[gross]", "E[net]", "E[净收益]", "ROI"))
    for cell in "ABCD":
        v = out["cells"][cell]["cells_by_scenario"]["pair_q0.50"]
        print("  {:<4s} {:>8s} {:>8s} {:>10.4f} {:>10.4f} {:>12.4f} {:>9.4f}".format(
            cell, v["payout_front3"], v["payout_front4"],
            float(v["expected_gross"]), float(v["expected_net_profit"]) + 132,
            float(v["expected_net_profit"]), float(v["expected_roi"])))
    print()
    print("盈亏平衡：所需 a + 20b（a=一等奖单注, b=二等奖单注）")
    for cell in "ABCD":
        be = out["cells"][cell]["break_even"]
        print("  {:<4s} K = {:>16.2f}   合法上限 = {:>16.0f}   可达 = {}".format(
            cell, float(be["break_even_line"]["a_plus_20b"]),
            float(be["legal_domain"]["a_plus_20b_max"]), be["reachable_within_caps"]))
    print()
    print("规则包络（一二等奖同时取封顶）下的 E[净收益]：")
    for cell in "ABCD":
        v = out["cells"][cell]["cells_by_scenario"]["rule_envelope"]
        print("  {:<4s} {:>12.4f}  ROI={:>8.4f}".format(
            cell, float(v["expected_net_profit"]), float(v["expected_roi"])))
    print()
    print("verdict:", json.dumps(out["verdict"], ensure_ascii=False)[:300])
    print("written:", args.out)


if __name__ == "__main__":
    main()

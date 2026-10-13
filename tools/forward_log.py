# -*- coding: utf-8 -*-
"""前瞻记录（forward log）：在开奖**之前**记录当期方案，开奖后再回填结果。

为什么需要
----------
7001–26042 的全部数据已被历史实验反复使用，不可能再产生真正的样本外证据。
唯一能重新获得样本外证据的方式是：**先把方案写下来，再等开奖**。

纪律（必须遵守，否则前瞻记录失去样本外性质）
-------------------------------------------
1. 只在当期末开奖时运行 ``log``，把方案写入 ``outputs/forward_log/``；
2. 开奖后才运行 ``settle`` 回填结果；
3. **不得**根据前瞻记录的中间结果调整策略参数；
4. 累计到 ≥ 300 期后才允许做一次统计检验。

用法
----
    # 开奖前：记录方案（会拒绝已有开奖结果的目标期）
    python tools/forward_log.py log --strategy s1
    python tools/forward_log.py log --strategy s2 --profile n9_mix_441

    # 开奖后：回填结果与回收
    python tools/forward_log.py settle [--issue 26117]

    # 查看累计进度
    python tools/forward_log.py status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LOG_DIR = os.path.join(ROOT, "outputs", "forward_log")
LOG_PATH = os.path.join(LOG_DIR, "forward_log.jsonl")
DATA_CSV = os.path.join(ROOT, "data", "dlt", "data.csv")

FRONT_COLS = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
BACK_COLS = ["蓝球_1", "蓝球_2"]


def _load():
    if not os.path.exists(LOG_PATH):
        return []
    out = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append(rec):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _issue_drawn(issue):
    """检查目标期是否已有开奖结果（若数据文件已更新）。"""
    from feature_engineering import load_dlt_history
    if not os.path.exists(DATA_CSV):
        return None
    d = load_dlt_history(DATA_CSV)
    row = d[d["期数"] == int(issue)]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {"front": sorted(int(r[c]) for c in FRONT_COLS),
            "back": sorted(int(r[c]) for c in BACK_COLS)}


def cmd_log(args):
    import backtest_plus as bt
    from feature_engineering import load_dlt_history
    from inference_plus import maybe_load_rf_model
    from tools.eval_5plus12_frozen import StrategyRunner

    bt.set_prize_mode("official")
    data = load_dlt_history(DATA_CSV)
    latest = int(data["期数"].iloc[-1])
    target = latest + 1
    if _issue_drawn(target) is not None:
        raise SystemExit("目标期 {} 已有开奖结果，前瞻记录必须在开奖前进行".format(target))

    # 目标期尚未开奖，因此补一行占位（号码不参与选号，仅提供期号）
    dummy = {c: 1 for c in FRONT_COLS + BACK_COLS}
    dummy["期数"] = target
    data_ext = pd.concat([data, pd.DataFrame([dummy])], ignore_index=True)
    idx = len(data_ext) - 1

    rf, meta = maybe_load_rf_model()
    sr = StrategyRunner(rf, meta)
    if args.strategy == "s1":
        groups = sr.s1_groups("fwd", data_ext, idx, top_n=10, play=1,
                              score_mode="stat_miss_penalty")
        strategy_id = "s1_main_5plus12"
    else:
        groups = sr.s2_groups("fwd", data_ext, idx, args.profile, 3)
        strategy_id = "s2_{}".format(args.profile)

    if not groups:
        raise SystemExit("未能生成方案，未写入前瞻记录")

    rec = {
        "type": "bet",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "target_issue": target,
        "based_on_last_issue": latest,
        "strategy": strategy_id,
        "groups": [sorted(int(x) for x in g) for g in groups],
        "group_count": len(groups),
        "cost": 132 * len(groups),
        "prize_mode": "official（目标期未开奖，选号不使用奖金信息）",
        "note": "前瞻记录：本记录在开奖前写入，不得事后修改",
    }
    _append(rec)
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    print("已写入 " + LOG_PATH)


def cmd_settle(args):
    import prize_rules as pr
    recs = _load()
    settled = {r["target_issue"] for r in recs if r.get("type") == "settle"}
    todo = [r for r in recs if r.get("type") == "bet"
            and r["target_issue"] not in settled
            and (args.issue is None or r["target_issue"] == args.issue)]
    n = 0
    for r in todo:
        drawn = _issue_drawn(r["target_issue"])
        if drawn is None:
            continue
        issue = r["target_issue"]
        table = pr.get_prize_table(issue)
        tf, tb = set(drawn["front"]), set(drawn["back"])
        reward = 0
        max_hit = 0
        for g in r["groups"]:
            fh = len(set(g) & tf)
            max_hit = max(max_hit, fh)
            reward += pr.group_payout(table, fh)
        _append({
            "type": "settle",
            "settled_at": datetime.now().isoformat(timespec="seconds"),
            "target_issue": issue,
            "actual_front": drawn["front"],
            "actual_back": drawn["back"],
            "strategy": r["strategy"],
            "cost": r["cost"],
            "reward": int(reward),
            "profit": int(reward - r["cost"]),
            "max_front_hit": max_hit,
            "prize_table_source": "official(已开奖期号使用官方公布单注奖金)",
        })
        n += 1
    print("已回填 {} 期".format(n))


def cmd_status(args):
    recs = _load()
    bets = [r for r in recs if r.get("type") == "bet"]
    settles = [r for r in recs if r.get("type") == "settle"]
    total_cost = sum(r["cost"] for r in settles)
    total_reward = sum(r["reward"] for r in settles)
    print("前瞻记录：方案 {} 条，已回填 {} 条".format(len(bets), len(settles)))
    if settles:
        print("  累计成本 = {} 元".format(total_cost))
        print("  累计回收 = {} 元".format(total_reward))
        print("  累计 ROI  = {:.4f}".format(
            (total_reward - total_cost) / total_cost if total_cost else 0.0))
        hit3 = sum(1 for r in settles if r["max_front_hit"] >= 3)
        print("  至少一组中 3 个前区的期数 = {} / {}".format(hit3, len(settles)))
    if len(settles) < 300:
        print("  提示：样本量 < 300 期，尚不足以对'是否存在微弱优势'做检验。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("log")
    p1.add_argument("--strategy", default="s1", choices=["s1", "s2"])
    p1.add_argument("--profile", default="n9_mix_441")
    p1.set_defaults(func=cmd_log)
    p2 = sub.add_parser("settle")
    p2.add_argument("--issue", type=int, default=None)
    p2.set_defaults(func=cmd_settle)
    p3 = sub.add_parser("status")
    p3.set_defaults(func=cmd_status)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

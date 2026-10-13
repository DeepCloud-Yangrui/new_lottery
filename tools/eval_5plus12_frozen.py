# -*- coding: utf-8 -*-
"""P5 冻结协议执行脚本：5+12 策略盈利可行性验证。

协议见 docs/project_notes/P5_CURRENT_STATE_AND_PROTOCOL.md。
本脚本在**写定协议之后**运行，不增量搜索参数、不新增模型。

窗口
----
- 样本外（OOS）：期号 26043–26116（74 期），从未用于任何历史选参。
- 回顾（RETRO）：历史数据（7001–26042）最后 200 / 500 / 1000 期。

策略（仓库现有实现）
--------------------
- S1_main_5plus12   单组，Top-10 候选 + 规则过滤 + 融合评分
- S2_n9_<profile>   9 候选池 × 3 组贪心三元组覆盖缩水
- S3_gate_<...>     P2-4 信号门槛（复用 backtest_plus.run_n9_signal_gate_backtest）

对照
----
- C1 相同参与期次/组数的均匀随机选号
- C2 相同候选池、组内随机选号（结构对照）
- C3 随机参与期次、相同下注次数（择时对照，仅对门槛策略）
- C4 精确数学期望（解析基准）

计奖口径
--------
- official      官方逐期公布的单注奖金 + 全额派奖 —— 默认
- legacy_config 旧 config 静态表 —— 仅用于"新旧口径差异"对照
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                                    # noqa: E402
import backtest_plus as bt                                  # noqa: E402
from feature_engineering import load_dlt_history             # noqa: E402
from inference_plus import (                                 # noqa: E402
    maybe_load_rf_model, predict_rf_scores, generate_front_combos,
)

FRONT_COLS = bt.FRONT_COLS
BACK_COLS = bt.BACK_COLS
GROUP_COST = pr.COST_PER_GROUP


# ---------------------------------------------------------------------------
# 通用逐期评估
# ---------------------------------------------------------------------------

def score_period(issue, groups, tf_set, tb_set, prize_mode):
    """返回 (cost, reward, reward_no_jackpot, max_front_hit, hit_counts)。"""
    table = bt.compute_prize_table(issue, prize_mode)
    cost = GROUP_COST * len(groups)
    reward = 0
    reward_nj = 0
    max_hit = 0
    hits = {k: 0 for k in range(6)}
    for g in groups:
        fh = len(set(g) & tf_set)
        max_hit = max(max_hit, fh)
        hits[fh] += 1
        for bh, n in pr.BACK_HIT_TICKETS.items():
            v = table.get((fh, bh), 0) * n
            reward += v
            if (fh, bh) not in bt.JACKPOT_CONDITIONS:
                reward_nj += v
    return cost, reward, reward_nj, max_hit, hits


def summarize(rows, prize_mode):
    df = pd.DataFrame(rows)
    cost = int(df.cost.sum())
    rew = int(df.reward.sum())
    rew_nj = int(df.reward_no_jackpot.sum())
    prof = rew - cost
    prof_nj = rew_nj - cost
    roi = prof / cost if cost else 0.0
    roi_nj = prof_nj / cost if cost else 0.0
    roi_trunc = (float(np.minimum(df.reward.values, 10000).sum()) - cost) / cost if cost else 0.0
    cum = df.profit.values
    return {
        "prize_mode": prize_mode,
        "periods": len(df),
        "total_cost": cost,
        "total_reward": rew,
        "total_reward_no_jackpot": rew_nj,
        "total_profit": prof,
        "roi": roi,
        "roi_no_jackpot": roi_nj,
        "roi_truncated": roi_trunc,
        "hit2_or_more_rate": float((df.max_front_hit >= 2).mean()),
        "hit3_or_more_rate": float((df.max_front_hit >= 3).mean()),
        "hit4_or_more_rate": float((df.max_front_hit >= 4).mean()),
        "hit5_rate": float((df.max_front_hit >= 5).mean()),
        "avg_front_hit_max": float(df.max_front_hit.mean()),
        "positive_period_rate": float((df.profit > 0).mean()),
        "max_drawdown": bt.max_drawdown_from_zero(cum),
        "max_consecutive_loss": bt._max_loss_streak(df.profit.values),
        "best_period_profit": int(df.profit.max()),
        "worst_period_profit": int(df.profit.min()),
        "records": rows,
    }


class StrategyRunner:
    """按协议固定的策略定义生成每期的前区组。

    RF 推理是全流程的主要耗时（约 1.15 秒/期），且**与 profile、计奖口径无关**，
    因此对 (数据集标签, 期索引) 做一次性缓存。
    """

    def __init__(self, rf_model=None, rf_meta=None, score_windows=(10, 30, 100),
                 min_history=120):
        self.rf = rf_model
        self.rf_meta = rf_meta
        self.w = score_windows
        self.min_history = min_history
        self.rf_train_end = rf_meta.get("train_issue_end") if rf_meta else None
        self._cf = {}
        self._pool = {}

    def _rf_for(self, hist, issue):
        if (self.rf is not None and self.rf_train_end is not None
                and self.rf_train_end < int(issue) and len(hist) >= self.min_history):
            return self.rf
        return None

    def components(self, ds_tag, data_asc, idx):
        """一次性计算该期的 stat / rf 分量并缓存（与 profile、口径无关）。"""
        key = (ds_tag, idx)
        got = self._cf.get(key)
        if got is not None:
            return got
        hist = data_asc.iloc[:idx]
        issue = int(data_asc.iloc[idx]["期数"])
        stat_pen = bt._compute_stat_miss_penalty(hist, self.w)
        stat_bonus = bt._compute_stat_miss_bonus(hist, self.w)
        rf = self._rf_for(hist, issue)
        rfs = (predict_rf_scores(rf, hist, windows=self.w, min_history=self.min_history)
               if rf is not None else np.zeros(35, dtype=float))
        gap = bt._compute_gap_only_scores(hist)
        ranked = {
            "stat_miss_bonus": (np.argsort(stat_bonus)[::-1] + 1).tolist(),
            "gap_direct_top5": (np.argsort(gap)[::-1] + 1).tolist(),
            "rf_stat": ((np.argsort(0.65 * rfs + 0.35 * stat_pen)[::-1] + 1).tolist()
                        if rf is not None else None),
        }
        val = {"stat_miss_bonus": stat_bonus, "stat_miss_penalty": stat_pen,
               "rf": rfs, "ranked": ranked, "rf_used": rf is not None}
        self._cf[key] = val
        return val

    def s1_groups(self, ds_tag, data_asc, idx, top_n=10, play=1,
                  score_mode="stat_miss_penalty"):
        cfg = bt.plus_strategy["dlt"]
        comp = self.components(ds_tag, data_asc, idx)
        old = (cfg["top_n_front"], cfg["max_front_combos"], dict(cfg["ensemble_weights"]))
        try:
            cfg["top_n_front"] = top_n
            cfg["max_front_combos"] = 20
            w = bt._normalize_weights_no_lstm(cfg["ensemble_weights"])
            ens = w.get("rf", 0.0) * comp["rf"] + w.get("stat", 1.0) * comp[score_mode]
            last_front = set(int(data_asc.iloc[idx - 1][c]) for c in FRONT_COLS)
            _, combos = generate_front_combos(ens, last_front)
            if not combos:
                return []
            return [sorted(c[0]) for c in combos[:play]]
        finally:
            cfg["top_n_front"], cfg["max_front_combos"], cfg["ensemble_weights"] = old

    def s2_pool(self, ds_tag, data_asc, idx, profile):
        key = (ds_tag, idx, profile)
        got = self._pool.get(key)
        if got is not None:
            return got
        comp = self.components(ds_tag, data_asc, idx)
        pool = bt.generate_n9_candidate_pool(
            data_asc.iloc[:idx], profile,
            rf_model_static=(self.rf if comp["rf_used"] else None),
            rf_meta=self.rf_meta, score_windows=self.w,
            min_history=self.min_history, ranked_by_source=comp["ranked"])
        self._pool[key] = pool
        return pool

    def s2_groups(self, ds_tag, data_asc, idx, profile, front_group_count=3):
        pool = self.s2_pool(ds_tag, data_asc, idx, profile)
        if len(pool.get("candidates", [])) < 9:
            return []
        combo = bt.generate_3x5_from_n9_pool(
            pool["candidates"], ranked_by_source=pool.get("ranked_by_source"),
            profile_name=profile, candidate_sources=pool.get("candidate_sources"),
            front_group_count=front_group_count)
        return [list(g) for g in combo.get("groups", [])]


def run_strategy(data_asc, start_idx, end_idx, group_fn, prize_mode):
    """group_fn(data_asc, idx) -> list[list[int]]（每期要买的前区 5 码组）。"""
    rows = []
    for idx in range(start_idx, end_idx + 1):
        target = data_asc.iloc[idx]
        issue = int(target["期数"])
        tf = set(int(target[c]) for c in FRONT_COLS)
        tb = set(int(target[c]) for c in BACK_COLS)
        groups = group_fn(data_asc, idx)
        if not groups:
            rows.append({"issue": issue, "cost": 0, "reward": 0, "reward_no_jackpot": 0,
                         "profit": 0, "max_front_hit": 0, "groups": []})
            continue
        cost, rew, rew_nj, mh, _hits = score_period(issue, groups, tf, tb, prize_mode)
        rows.append({"issue": issue, "cost": cost, "reward": rew,
                     "reward_no_jackpot": rew_nj, "profit": rew - cost,
                     "max_front_hit": mh, "groups": groups})
    return summarize(rows, prize_mode)


def run_random_uniform(data_asc, start_idx, end_idx, n_groups, prize_mode,
                       n_trials, seed):
    out = []
    for t in range(n_trials):
        rng = random.Random(seed + t * 7919)
        rows = []
        for idx in range(start_idx, end_idx + 1):
            target = data_asc.iloc[idx]
            issue = int(target["期数"])
            tf = set(int(target[c]) for c in FRONT_COLS)
            tb = set(int(target[c]) for c in BACK_COLS)
            groups = [sorted(rng.sample(range(1, 36), 5)) for _ in range(n_groups)]
            cost, rew, rew_nj, mh, _ = score_period(issue, groups, tf, tb, prize_mode)
            rows.append({"issue": issue, "cost": cost, "reward": rew,
                         "reward_no_jackpot": rew_nj, "profit": rew - cost,
                         "max_front_hit": mh, "groups": groups})
        out.append(summarize(rows, prize_mode))
    return out


def run_random_from_pool(data_asc, start_idx, end_idx, pool_fn, n_groups,
                         prize_mode, n_trials, seed):
    """C2：候选池与策略相同，组内 5 码随机。"""
    out = []
    for t in range(n_trials):
        rng = random.Random(seed + t * 7919)
        rows = []
        for idx in range(start_idx, end_idx + 1):
            target = data_asc.iloc[idx]
            issue = int(target["期数"])
            tf = set(int(target[c]) for c in FRONT_COLS)
            tb = set(int(target[c]) for c in BACK_COLS)
            cand = pool_fn(data_asc, idx)
            if not cand or len(cand) < 5:
                rows.append({"issue": issue, "cost": 0, "reward": 0,
                             "reward_no_jackpot": 0, "profit": 0,
                             "max_front_hit": 0, "groups": []})
                continue
            groups = [sorted(rng.sample(cand, 5)) for _ in range(n_groups)]
            cost, rew, rew_nj, mh, _ = score_period(issue, groups, tf, tb, prize_mode)
            rows.append({"issue": issue, "cost": cost, "reward": rew,
                         "reward_no_jackpot": rew_nj, "profit": rew - cost,
                         "max_front_hit": mh, "groups": groups})
        out.append(summarize(rows, prize_mode))
    return out


def dist_summary(trials, key):
    vals = np.array([t[key] for t in trials], dtype=float)
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "p05": float(np.percentile(vals, 5)), "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)), "n": len(vals)}


def compare_to_random(strategy_val, trials, key):
    d = dist_summary(trials, key)
    se = d["std"] / np.sqrt(d["n"]) if d["n"] and d["std"] > 0 else float("nan")
    z = (strategy_val - d["mean"]) / se if se and se > 0 else float("nan")
    pct = float(np.mean([t[key] <= strategy_val for t in trials]))
    return {"strategy": strategy_val, "random_mean": d["mean"],
            "random_std": d["std"], "random_se_of_mean": se,
            "z": z, "percentile_rank": pct,
            "p05": d["p05"], "p95": d["p95"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-full", required=True, help="含样本外期次的完整 CSV")
    ap.add_argument("--data-hist", required=True, help="历史 CSV（截止 26042）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--oos-min", type=int, default=26043)
    ap.add_argument("--oos-max", type=int, default=26116)
    ap.add_argument("--retro-windows", default="200,500,1000")
    ap.add_argument("--only-oos", type=int, default=0,
                    help="1=只跑样本外窗口（内存/时间受限时使用）")
    args = ap.parse_args()

    bt.set_prize_mode("official")
    rf_model, rf_meta = maybe_load_rf_model()
    if rf_model is None:
        raise SystemExit("未找到 RF 模型，请先运行 run_train_rf_model.py")
    print("RF train_issue_end =", rf_meta.get("train_issue_end"))

    full = load_dlt_history(args.data_full)
    hist = load_dlt_history(args.data_hist)
    sr = StrategyRunner(rf_model, rf_meta)

    oos = full[(full["期数"] >= args.oos_min) & (full["期数"] <= args.oos_max)]
    oos_start = int(oos.index[0]); oos_end = int(oos.index[-1])
    print("OOS 窗口: index {}..{} 期号 {}..{}（{} 期）".format(
        oos_start, oos_end, int(oos.iloc[0]["期数"]), int(oos.iloc[-1]["期数"]), len(oos)))

    result = {
        "protocol": "docs/project_notes/P5_CURRENT_STATE_AND_PROTOCOL.md",
        "n_trials": args.n_trials, "seed": args.seed,
        "oos_issues": [args.oos_min, args.oos_max], "oos_periods": len(oos),
        "rf_train_issue_end": rf_meta.get("train_issue_end"),
        "windows": {},
    }

    PROFILES = ["n9_mix_441", "n9_mix_531", "n9_mix_621"]

    def make_s1(tag):
        return lambda d, i: sr.s1_groups(tag, d, i, top_n=10, play=1,
                                         score_mode="stat_miss_penalty")

    def make_s2(tag, prof):
        return lambda d, i: sr.s2_groups(tag, d, i, prof, 3)

    # ---------------- OOS ----------------
    w = {}
    for mode in ("official", "legacy_config"):
        blk = {}
        blk["S1_main_5plus12"] = run_strategy(
            full, oos_start, oos_end, make_s1("full"), mode)
        for prof in PROFILES:
            blk["S2_" + prof] = run_strategy(
                full, oos_start, oos_end, make_s2("full", prof), mode)

        # ---------------- 回顾窗口 ----------------
        for n in ([] if args.only_oos else [int(x) for x in args.retro_windows.split(",")]):
            s = max(121, len(hist) - n); e = len(hist) - 1
            tag = "last{}".format(n)
            blk["RETRO{}_S1_main".format(tag)] = run_strategy(
                hist, s, e, make_s1("hist"), mode)
            for prof in PROFILES:
                blk["RETRO{}_S2_{}".format(tag, prof)] = run_strategy(
                    hist, s, e, make_s2("hist", prof), mode)
        w[mode] = blk

    # ---------------- 随机对照（仅 official 口径）----------------
    ctrl = {}
    ctrl["C1_random_uniform_1group"] = run_random_uniform(
        full, oos_start, oos_end, 1, "official", args.n_trials, args.seed)
    ctrl["C1_random_uniform_3groups"] = run_random_uniform(
        full, oos_start, oos_end, 3, "official", args.n_trials, args.seed)

    def pool_of(profile):
        def _f(d, i):
            # run_random_from_pool 传入 (完整数据, 位置索引)
            p = sr.s2_pool("full", d, i, profile)
            return p.get("candidates", [])
        return _f

    for prof in PROFILES:
        ctrl["C2_from_pool_{}_3groups".format(prof)] = run_random_from_pool(
            full, oos_start, oos_end, pool_of(prof), 3, "official",
            max(1, args.n_trials // 5), args.seed)

    # ---------------- 汇总与比较 ----------------
    compar = {}
    oos_blk = w["official"]
    for name, key, ref in [
        ("S1_main_5plus12", "roi_no_jackpot", "C1_random_uniform_1group"),
        ("S1_main_5plus12", "hit3_or_more_rate", "C1_random_uniform_1group"),
        ("S2_n9_mix_441", "roi_no_jackpot", "C1_random_uniform_3groups"),
        ("S2_n9_mix_531", "roi_no_jackpot", "C1_random_uniform_3groups"),
        ("S2_n9_mix_621", "roi_no_jackpot", "C1_random_uniform_3groups"),
        ("S2_n9_mix_441", "roi_no_jackpot", "C2_from_pool_n9_mix_441_3groups"),
        ("S2_n9_mix_531", "roi_no_jackpot", "C2_from_pool_n9_mix_531_3groups"),
        ("S2_n9_mix_621", "roi_no_jackpot", "C2_from_pool_n9_mix_621_3groups"),
        ("S2_n9_mix_441", "hit3_or_more_rate", "C2_from_pool_n9_mix_441_3groups"),
        ("S2_n9_mix_531", "hit3_or_more_rate", "C2_from_pool_n9_mix_531_3groups"),
        ("S2_n9_mix_621", "hit3_or_more_rate", "C2_from_pool_n9_mix_621_3groups"),
    ]:
        if name in oos_blk and ref in ctrl:
            compar["{}_vs_{}_{}".format(name, ref, key)] = compare_to_random(
                oos_blk[name][key], ctrl[ref], key)

    # 剔除 records 以控制体积
    def strip(d):
        return {k: v for k, v in d.items() if k != "records"}

    result["windows"] = {m: {k: strip(v) for k, v in blk.items()}
                         for m, blk in w.items()}
    result["controls_oos"] = {k: strip(dist_summary(v, "roi_no_jackpot"))
                              | {"roi_no_jackpot": dist_summary(v, "roi_no_jackpot"),
                                 "hit3_or_more_rate": dist_summary(v, "hit3_or_more_rate"),
                                 "roi": dist_summary(v, "roi"),
                                 "total_profit": dist_summary(v, "total_profit"),
                                 "max_drawdown": dist_summary(v, "max_drawdown")}
                              for k, v in ctrl.items()}
    result["comparisons_oos"] = compar

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== OOS（{} 期）official 口径 ===".format(len(oos)))
    for k, v in oos_blk.items():
        if k.startswith("S"):
            print("  {:<22s} cost={:>7d} profit={:>9d} roi={:>8.4f} roi_nj={:>8.4f} "
                  "hit3+={:.4f}".format(k, v["total_cost"], v["total_profit"],
                                        v["roi"], v["roi_no_jackpot"],
                                        v["hit3_or_more_rate"]))
    print("\n=== 对照分布（C1/C2, OOS）===")
    for k, v in ctrl.items():
        d = dist_summary(v, "roi_no_jackpot")
        print("  {:<38s} roi_nj mean={:>8.4f} sd={:.4f} p05={:>8.4f} p95={:>8.4f}".format(
            k, d["mean"], d["std"], d["p05"], d["p95"]))
    print("\n=== 比较 ===")
    for k, v in compar.items():
        print("  {:<70s} z={:>7.2f} pct={:.3f}".format(k, v["z"], v["percentile_rank"]))
    print("\nwritten:", args.out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""候选池对照的**条件 / 无条件**分离分析（V2 T1 纠错核心）。

上一轮的错误
------------
上一轮把「同候选池随机组内选号（C2）的模拟均值」与「全号池均匀随机（C1）的
模拟均值」之差，除以**蒙特卡洛均值标准误**，得到 z ≈ −9 ~ −11，
并据此写成"9 候选池结构本身显著劣于全号池随机，P3 只会放大问题"。

这个推断有两个独立的错误：

1. **统计量用错**。除以 MC 均值标准误，衡量的是"在**固定的这 74 期开奖**与
   **固定的这些候选池**下，模拟均值算得有多准"，它会随 `n_trials` 增大而
   无限放大，**不是**"候选池降低未来期望"的检验统计量。
2. **被比较的对象不同**。C1 的随机化覆盖全部 35 个号码，因此它的每期条件期望
   恒等于无条件期望 E[R]（与开奖无关）；C2 的随机化只在**当期那个 9 号池内**，
   它的条件期望是 g(m_d)，其中 m_d = |池 ∩ 开奖|。两者之差是
   **该窗口内已实现的开奖与候选池重叠情况**的产物，不是结构性质。

本脚本给出正确的分解
--------------------
对任意固定 5 码组合 C 与均匀开奖 D：|C ∩ D| ~ Hypergeometric(35, 5, 5)，
**与 C 具体是什么无关**（见 ``tools/analysis_math_baseline.py`` 的精确枚举校验）。
因此：

    E[R | 全号池] ≡ E[R]                     （无条件期望，与开奖、与池无关）
    E[R | 9 号池, m] = g(m) = Σ_k P(k; 9, m, 5)·R(k)
    E_m[g(m)]        = E[R]                  （对 m ~ Hypergeometric(35,9,5)）

结论：**两种结构的无条件期望完全相同**。差别只体现在
（a）给定窗口的**已实现**重叠 m_d 造成的条件差异，
（b）总收益的**方差**（组间通过 m 正相关）。

输出
----
1. 解析：g(m)、m 的分布、sd(g(m))、以及 n 期窗口下"条件缺口"的抽样标准误；
2. 实测：把该窗口内策略实际使用的候选池重新生成，算出真实的 m_d 与条件缺口
   （仅当提供 --with-pools 时执行；需要 RF 推理）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prize_rules as pr                      # noqa: E402
from tools.five_plus12_math import (          # noqa: E402
    front_hit_distribution, group_payout_by_front_hit, no_jackpot_table,
    COST_PER_GROUP,
)

POOL_SIZE = 9


def hyper_m_prob(pool_size=POOL_SIZE):
    ms = list(range(0, min(pool_size, 5) + 1))
    pm = [comb(pool_size, x) * comb(35 - pool_size, 5 - x) / comb(35, 5) for x in ms]
    s = sum(pm)
    return ms, [x / s for x in pm]


def g_of_m(R, m, pool_size=POOL_SIZE):
    """g(m) = E[R | 池大小 pool_size, |池∩开奖| = m]。"""
    ks = list(range(0, min(m, 5) + 1))
    pk = [comb(m, k) * comb(pool_size - m, 5 - k) / comb(pool_size, 5) for k in ks]
    s = sum(pk)
    return sum(pk[i] * R[ks[i]] for i in range(len(ks))) / s


def analyse_window(issues, label, with_pools=None):
    pf = front_hit_distribution()
    ms, pm = hyper_m_prob()
    rec = []
    for iss in issues:
        t = pr.get_prize_table(int(iss), "official")
        for tname, tbl in (("full", t), ("no_jackpot", no_jackpot_table(t))):
            R = group_payout_by_front_hit(tbl)
            E = sum(pf[f] * R[f] for f in range(6))
            g = [g_of_m(R, m) for m in ms]
            Eg = sum(pm[i] * g[i] for i in range(len(ms)))
            Vg = sum(pm[i] * g[i] ** 2 for i in range(len(ms))) - Eg ** 2
            item = {"issue": int(iss), "table": tname, "E_R": E,
                    "E_g": Eg, "sd_g": float(np.sqrt(max(Vg, 0.0))),
                    "g_by_m": {str(ms[i]): g[i] for i in range(len(ms))}}
            if with_pools is not None and int(iss) in with_pools:
                m = with_pools[int(iss)]
                item["realised_m"] = m
                item["realised_conditional_E"] = g[m]
                item["realised_gap"] = g[m] - E
            rec.append(item)

    out = {"label": label, "n_periods": len(issues), "pool_size": POOL_SIZE,
           "m_distribution": {str(ms[i]): pm[i] for i in range(len(ms))},
           "periods": rec}
    for tname in ("full", "no_jackpot"):
        sub = [r for r in rec if r["table"] == tname]
        E = np.array([r["E_R"] for r in sub])
        sdg = np.array([r["sd_g"] for r in sub])
        out["summary_" + tname] = {
            "mean_E_R": float(E.mean()),
            "mean_sd_g": float(sdg.mean()),
            "se_of_window_mean_gap": float(sdg.mean() / np.sqrt(len(sub))),
            "note": ("窗口内 (1/n)Σ g(m_d) − E[R] 在零假设下的抽样标准误；"
                     "m_d 为当期候选池与开奖的重叠数。"),
        }
        have = [r for r in sub if "realised_gap" in r]
        if have:
            g = np.array([r["realised_gap"] for r in have], dtype=float)
            se = sdg.mean() / np.sqrt(len(g))
            out["summary_" + tname].update({
                "n_realised": len(have),
                "realised_mean_m": float(np.mean([r["realised_m"] for r in have])),
                "realised_mean_gap": float(g.mean()),
                "realised_gap_se": float(se),
                "realised_gap_z_conditional": float(g.mean() / se) if se else None,
                "realised_gap_as_roi": float(g.mean() / COST_PER_GROUP),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-pools", type=int, default=0,
                    help="1=重新生成策略实际候选池以计算已实现缺口（需 RF）")
    args = ap.parse_args()

    src = os.path.join(args.run_dir, "data", "dlt_full_7001_26116.csv")
    hist = os.path.join(args.run_dir, "data", "dlt_hist_7001_26042.csv")
    windows = []

    def issues_of(csv, lo, hi):
        import pandas as pd
        d = pd.read_csv(csv)
        col = "期数" if "期数" in d.columns else "issue"
        return [int(x) for x in d[col].tolist() if lo <= int(x) <= hi]

    windows.append(("oos_26043_26116", issues_of(src, 26043, 26116), src))
    for n in (200, 1000):
        iss = issues_of(hist, 0, 10 ** 9)[-n:]
        windows.append(("retro_last{}".format(n), iss, hist))

    pools = None
    if args.with_pools:
        pools = build_pools(src, windows[0][1])

    out = {"method": "条件/无条件分离（解析 + 定向实测）", "windows": {}}
    for label, iss, _ in windows:
        out["windows"][label] = analyse_window(iss, label, with_pools=pools)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    for label, blk in out["windows"].items():
        for tname in ("full", "no_jackpot"):
            s = blk["summary_" + tname]
            line = ("{} [{}] E[R]={:.2f}  sd(g(m))={:.2f}  窗口条件缺口SE={:.3f}元".format(
                label, tname, s["mean_E_R"], s["mean_sd_g"], s["se_of_window_mean_gap"]))
            if s.get("realised_mean_gap") is not None:
                line += "  | 已实现 mean_m={:.2f} 缺口={:.2f}元 ± {:.2f} (z={:.2f})".format(
                    s["realised_mean_m"], s["realised_mean_gap"],
                    s["realised_gap_se"], s["realised_gap_z_conditional"])
            print(line)
    print("written:", args.out)


def build_pools(csv, issues):
    """重新生成策略在同一窗口实际使用的 9 候选池，并返回 {issue: m}。"""
    import backtest_plus as bt
    from feature_engineering import load_dlt_history
    from inference_plus import maybe_load_rf_model
    from tools.eval_5plus12_frozen import StrategyRunner

    d = load_dlt_history(csv)
    rf, meta = maybe_load_rf_model()
    sr = StrategyRunner(rf, meta)
    idx_of = {int(r["期数"]): i for i, r in d.iterrows()}
    drawn = {int(r["期数"]): set(int(r[c]) for c in bt.FRONT_COLS)
             for _, r in d.iterrows()}
    out = {}
    for iss in issues:
        i = idx_of[int(iss)]
        pool = sr.s2_pool("full", d, i, "n9_mix_441")
        cand = set(pool.get("candidates", []))
        out[int(iss)] = len(cand & drawn[int(iss)])
    return out


if __name__ == "__main__":
    main()

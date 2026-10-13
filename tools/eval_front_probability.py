# -*- coding: utf-8 -*-
"""F1 前区边际概率 —— 时间顺序评估器（F1-2）。

用法::

    python -m tools.eval_front_probability \
        --config configs/front_probability_f1.json \
        --data data/dlt/data.csv \
        --run-dir outputs/<unique_run_id> \
        --device cuda:5

设计要点（对应 SERVER_F1_HANDOFF_PLAN.md §3、§F1-2）：

* 最近 800 个可建模期分 4 个互不重叠的 200 期块；扩展训练窗，首块训练 ≥ 500 期。
* 所有候选使用**完全相同的期次**；训练截止严格早于目标块。
* 主指标为**按期** Brier（每期先在 35 个号上求均值，再在期上聚合）。
* 主对比 L-U / R-U / N-U；同一期配对差；moving block bootstrap 长度 20 与 50、
  2000 次、seed=20261013；Bonferroni 家族 alpha=0.05，3 对比 × 2 侧，
  每个双侧区间 **98.333333%**（单侧尾概率 0.05/6）。
* 中断恢复：只有**完整写入且 code/data/config 三个 hash 全部一致**的折-候选单元
  才复用；不完整产物隔离到 ``_incomplete/`` 后重算该单元，不重复计算期次。
* 运行目录已存在且未显式允许恢复时**拒绝覆盖**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from feature_engineering import build_front_training_dataset, load_dlt_history  # noqa: E402
from tools.front_probability_models import (  # noqa: E402
    N_FRONT,
    brier_per_period,
    calibration_bins,
    fit_candidate,
    logloss_per_period,
    predict_candidate,
)

STATUS_FILE = "status.json"
PRED_FILE = "predictions.npz"
MODEL_FILE = "model.pkl"
RUN_LOG = "run.log"


class EvaluationError(RuntimeError):
    """输入或运行目录不满足冻结协议时抛出。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_config(config: Dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def code_hash() -> str:
    """当前源码指纹：git HEAD（若有）+ **源码文件内容**的 sha256。

    两个由对抗性复核发现的要点：

    * 只用 ``git status --porcelain`` 不够：对**已经处于修改状态**的文件再改一次，
      porcelain 输出不变，缓存单元会被误判为可复用。
    * **不在 git 仓库内时不能退化成常量**：原先返回 "no-git" 会让所有单元共享同一个
      代码指纹，改完代码仍会复用旧结果。这里改为对源码目录内容取哈希。
    """
    import subprocess

    names: List[str] = []
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        tracked = subprocess.run(["git", "ls-files", "tools", "tests", "configs", "*.py"],
                                 cwd=ROOT, capture_output=True, text=True,
                                 check=True).stdout.split()
        names = sorted(set(tracked))
    except Exception:
        head = "no-git"
        # 退化为直接扫描源码目录，绝不用常量
        for sub in ("tools", "tests", "configs"):
            base = os.path.join(ROOT, sub)
            for dirpath, _dirs, files in os.walk(base):
                for fn in sorted(files):
                    if fn.endswith((".py", ".json", ".txt")):
                        names.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
        for fn in sorted(os.listdir(ROOT)):
            if fn.endswith(".py"):
                names.append(fn)
        names = sorted(set(names))

    h = hashlib.sha256(head.encode("utf-8"))
    for rel in names:
        p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            h.update(("MISSING " + rel).encode("utf-8"))
            continue
        h.update(rel.encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    return "{}:{}".format(head[:12], h.hexdigest()[:16])


def alpha_tail(alpha: float, family_size: int) -> float:
    """Bonferroni 家族 alpha 分摊到每个单侧尾的概率。"""
    return float(alpha) / float(family_size)


def adjusted_two_sided_confidence(alpha: float, family_size: int) -> float:
    """对应的**双侧**区间置信度 = 1 - 2*(alpha/family_size)。

    家族 alpha=0.05、6 个尾 -> 0.98333333（协议写作 98.333333%）。
    """
    return 1.0 - 2.0 * alpha_tail(alpha, family_size)


def per_period_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """每期一个 Brier（先在 35 个号上取均值）。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.mean((p - y) ** 2, axis=1)


def per_period_logloss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p), axis=1)


# ---------------------------------------------------------------------------
# 数据与配置
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_data(path: str, config: Dict) -> Dict:
    """校验数据文件的大小/SHA256/行数/期号范围；不符即拒绝运行。"""
    if not os.path.exists(path):
        raise EvaluationError("数据文件不存在: {}".format(path))
    cfg = config.get("data", {})
    info = {"path": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}

    want = cfg.get("sha256")
    if want and info["sha256"] != want:
        raise EvaluationError(
            "输入 hash 不符：得到 {}，配置要求 {}".format(info["sha256"], want))
    if "expected_rows" in cfg:
        n_rows = len(pd.read_csv(path))
        info["rows"] = n_rows
        if int(cfg["expected_rows"]) != int(n_rows):
            raise EvaluationError("行数不符：得到 {}，配置要求 {}".format(
                n_rows, cfg["expected_rows"]))
    return info


def build_dataset(data_csv: str, config: Dict) -> Dict:
    """构建可建模的 (X, y, issues) 与升序历史表。"""
    feat = config.get("features", {})
    windows = tuple(int(w) for w in feat.get("windows", [10, 30, 100]))
    min_history = int(feat.get("min_history", 120))

    hist = load_dlt_history(data_csv)
    X, y, issues = build_front_training_dataset(hist, windows=windows,
                                               min_history=min_history)
    if X.shape[1] != int(feat.get("expected_dim", X.shape[1])):
        raise EvaluationError("特征维度不符：得到 {}，配置要求 {}".format(
            X.shape[1], feat.get("expected_dim")))
    if not np.all(y.sum(axis=1) == 5):
        raise EvaluationError("存在不是恰好 5 个正标签的训练期")
    return {"X": X.astype(float), "y": y.astype(int),
            "issues": issues.astype(int), "history": hist,
            "min_history": min_history}


def fold_ranges(config: Dict) -> List[Dict]:
    """返回冻结的四折（目标区间与训练区间，按可建模样本的下标）。"""
    blocks = config.get("evaluation", {}).get("frozen_blocks")
    if not blocks:
        raise EvaluationError("配置缺少 evaluation.frozen_blocks")
    folds = []
    prev_max = -1
    for b in blocks:
        f = dict(b)
        if f["train_index_stop"] > f["target_index_start"]:
            raise EvaluationError("fold {} 训练区间越过目标区间".format(f["fold"]))
        if f["target_index_start"] <= prev_max:
            raise EvaluationError("fold {} 与前一折的评估期重叠".format(f["fold"]))
        prev_max = f["target_index_stop"] - 1
        folds.append(f)
    min_first = int(config["evaluation"].get("min_first_train_periods", 0))
    if folds and folds[0]["n_train"] < min_first:
        raise EvaluationError("首块训练期 {} < 要求 {}".format(
            folds[0]["n_train"], min_first))
    return folds


def prepare_run_dir(run_dir: str, config: Dict, allow_resume: bool) -> None:
    """运行目录已存在时：允许恢复 -> 复用；否则拒绝覆盖。"""
    if os.path.exists(run_dir):
        if not allow_resume:
            raise EvaluationError(
                "运行目录已存在，拒绝覆盖：{}（如需续跑请显式 --resume）".format(run_dir))
    else:
        os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "units"), exist_ok=True)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def paired_block_bootstrap_ci(diff: np.ndarray, block_length: int,
                              n_resamples: int, seed: int,
                              confidence: float) -> Dict:
    """Moving block bootstrap 的均值置信区间（百分位法）。"""
    diff = np.asarray(diff, dtype=float)
    n = diff.shape[0]
    if n == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "block_length": block_length, "n_resamples": n_resamples,
                "seed": seed, "confidence": confidence, "n_periods": 0}
    L = int(min(int(block_length), n))
    n_blocks = int(np.ceil(n / L))
    starts_max = n - L
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, starts_max + 1, size=(int(n_resamples), n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(int(n_resamples), -1)[:, :n]
    means = diff[idx].mean(axis=1)

    tail = (1.0 - float(confidence)) / 2.0
    lo = float(np.quantile(means, tail))
    hi = float(np.quantile(means, 1.0 - tail))
    return {
        "mean": float(diff.mean()),
        "lower": lo,
        "upper": hi,
        "block_length": L,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "tail_probability": tail,
        "n_periods": int(n),
    }


def meets_interval_requirement(ci_a: Dict, ci_b: Dict) -> bool:
    """两种块长度下的调整区间下界是否都 > 0。"""
    return bool(ci_a["lower"] > 0.0 and ci_b["lower"] > 0.0)


def _block_direction(gain: np.ndarray, fold_ids: np.ndarray) -> Dict:
    out = {}
    for f in sorted(set(int(x) for x in fold_ids)):
        g = gain[fold_ids == f]
        out[str(f)] = {"mean_gain": float(g.mean()) if g.size else float("nan"),
                       "n": int(g.size), "positive": bool(g.size and g.mean() > 0)}
    return out


def summarize_candidate(recs: List[Dict], folds: List[Dict]) -> Dict:
    """按候选汇总期级指标、Top-5 命中与分块 Brier。"""
    issues = np.concatenate([r["issues"] for r in recs])
    p = np.vstack([r["p"] for r in recs])
    y = np.vstack([r["y"] for r in recs])
    fold_ids = np.concatenate([np.full(len(r["issues"]), r["fold"]) for r in recs])
    top5 = [t for r in recs for t in r["top5"]]

    briers = per_period_brier(p, y)
    loglosses = per_period_logloss(p, y)

    hits = []
    for k in range(len(top5)):
        idx = [t - 1 for t in top5[k]]
        hits.append(int(y[k, idx].sum()))
    hits = np.asarray(hits, dtype=int)

    per_block = {}
    for f in folds:
        m = fold_ids == f["fold"]
        per_block[str(f["fold"])] = {
            "n": int(m.sum()),
            "brier": float(briers[m].mean()) if m.any() else float("nan"),
            "issue_first": int(issues[m][0]) if m.any() else None,
            "issue_last": int(issues[m][-1]) if m.any() else None,
        }

    timing = float(sum(r["timing_seconds"] for r in recs))
    return {
        "n_periods": int(len(issues)),
        "overall_brier": float(briers.mean()),
        "overall_logloss": float(loglosses.mean()),
        "per_block_brier": per_block,
        "top5_mean_hits": float(hits.mean()),
        "k_distribution": {str(k): int((hits == k).sum()) for k in range(6)},
        "k_ge_3": int((hits >= 3).sum()),
        "k_ge_4": int((hits >= 4).sum()),
        "k_ge_5": int((hits == 5).sum()),
        "timing_seconds": timing,
        "mean_sum_p": float(np.mean(p.sum(axis=1))),
        "min_p": float(p.min()),
        "max_p": float(p.max()),
        "model_hashes": sorted({r["model_hash"] for r in recs}),
    }


def _align_pair(store: Dict[str, Dict], a: str, b: str):
    """按**同一期次**对齐两个候选的期级损失；不偷偷换样本。"""
    ia = {int(i): k for k, i in enumerate(store[a]["issues"])}
    ib = {int(i): k for k, i in enumerate(store[b]["issues"])}
    common = sorted(set(ia) & set(ib))
    ka = np.array([ia[i] for i in common], dtype=int)
    kb = np.array([ib[i] for i in common], dtype=int)
    return common, ka, kb


def compare_pair(store: Dict[str, Dict], baseline: str, model: str,
                 config: Dict) -> Dict:
    """gain = baseline_loss - model_loss（正 = 模型更好）。"""
    stats_cfg = config.get("statistics", {})
    boot = stats_cfg.get("bootstrap", {})
    conf = float(stats_cfg.get("per_comparison_two_sided_confidence",
                               adjusted_two_sided_confidence(
                                   stats_cfg.get("family_alpha", 0.05),
                                   stats_cfg.get("family_size", 6))))
    seed = int(boot.get("seed", 20261013))
    n_res = int(boot.get("n_resamples", 2000))
    lengths = list(boot.get("block_lengths", [20, 50]))

    common, ka, kb = _align_pair(store, baseline, model)
    if not common:
        return {"comparison": "{}-{}".format(model, baseline), "n_periods": 0,
                "error": "没有共同期次"}

    lb = per_period_brier(store[baseline]["p"][ka], store[baseline]["y"][ka])
    lm = per_period_brier(store[model]["p"][kb], store[model]["y"][kb])
    gain = lb - lm
    fold_ids = store[model]["fold_ids"][kb]

    cis = {}
    for L in lengths:
        cis[str(L)] = paired_block_bootstrap_ci(gain, L, n_res, seed, conf)

    direction = _block_direction(gain, fold_ids)
    n_pos_blocks = sum(1 for v in direction.values() if v["positive"])
    n_blocks = len(direction)

    ll_u = per_period_logloss(store[baseline]["p"][ka], store[baseline]["y"][ka])
    ll_m = per_period_logloss(store[model]["p"][kb], store[model]["y"][kb])

    return {
        "comparison": "{}-{}".format(model, baseline),
        "n_periods": int(len(common)),
        "excluded_periods_model_only": int(len(store[model]["issues"]) - len(common)),
        "excluded_periods_baseline_only": int(len(store[baseline]["issues"]) - len(common)),
        "overall_gain": float(gain.mean()),
        "overall_brier_model": float(lm.mean()),
        "overall_brier_baseline": float(lb.mean()),
        "overall_logloss_model": float(ll_m.mean()),
        "overall_logloss_baseline": float(ll_u.mean()),
        "block_direction": direction,
        "n_blocks_positive": int(n_pos_blocks),
        "n_blocks": int(n_blocks),
        "intervals": cis,
        "adjusted_two_sided_confidence": conf,
        "intervals_lower_bounds_positive": meets_interval_requirement(
            cis[str(lengths[0])], cis[str(lengths[-1])]) if len(lengths) >= 2
        else bool(cis[str(lengths[0])]["lower"] > 0),
    }


def run_verdict(summary: Dict, comparisons: Dict, config: Dict) -> Dict:
    """按预登记条件判定哪些候选通过；全部候选未完成则不作完整裁决。"""
    expected = list(config.get("statistics", {}).get("comparisons", []))
    models = sorted({c.split("-")[0] for c in expected if "-" in c})
    # 完整性门槛必须覆盖全部冻结候选（含 F）。只查 comparisons 里的模型名会漏掉 F，
    # 使预登记的「Brier 不高于 F」条件被静默跳过或跨不同期次集比较
    # （对抗性复核发现，已修）。
    needed = sorted(set(models) | {"U"} | set(config.get("candidates", {}).keys()))
    folds = config.get("evaluation", {}).get("frozen_blocks", [])
    expected_periods = int(sum(int(f["target_index_stop"]) - int(f["target_index_start"])
                               for f in folds))
    complete = bool(needed) and all(
        summary.get("candidates", {}).get(m, {}).get("n_periods", 0) == expected_periods
        and expected_periods > 0 for m in needed)
    missing = sorted(
        m for m in needed
        if summary.get("candidates", {}).get(m, {}).get("n_periods", 0) != expected_periods)

    promoted = []
    per_model = {}
    if complete:
        for m in models:
            comp = comparisons.get("{}-U".format(m))
            if not comp or comp.get("n_periods", 0) == 0:
                continue
            crit = {
                "overall_gain_positive": comp["overall_gain"] > 0,
                "at_least_3_of_4_blocks": comp["n_blocks_positive"] >= 3,
                "intervals_lower_bounds_positive": comp["intervals_lower_bounds_positive"],
                "logloss_not_worse_than_U":
                    comp["overall_logloss_model"] <= comp["overall_logloss_baseline"],
            }
            f_brier = summary["candidates"].get("F", {}).get("overall_brier")
            # F 缺失时该条件**不成立**，而不是被过滤掉 —— 预登记条件不得静默消失。
            crit["brier_not_worse_than_F"] = (
                bool(summary["candidates"][m]["overall_brier"] <= f_brier)
                if f_brier is not None else False)
            ok = all(bool(v) for v in crit.values())
            per_model[m] = {"criteria": crit, "promoted": bool(ok)}
            if ok:
                promoted.append(m)

    winner = None
    if promoted:
        eps = float(config.get("statistics", {}).get("tie_epsilon", 1e-12))
        order = config.get("statistics", {}).get("tie_preference_order", ["L", "R", "N"])
        best = min(summary["candidates"][m]["overall_brier"] for m in promoted)
        tied = [m for m in promoted
                if summary["candidates"][m]["overall_brier"] - best <= eps]
        for m in order:
            if m in tied:
                winner = m
                break

    return {
        "complete_comparison": bool(complete),
        "required_candidates": needed,
        "incomplete_candidates": missing,
        "per_model": per_model,
        "promoted": promoted,
        "selected": winner,
        "note": ("全部候选未完成，不作完整胜负裁决" if not complete
                 else "以上仅为历史时间顺序评估下的冻结候选，需后续未见数据验证"),
    }


def calibration_for(recs: List[Dict], n_bins: int = 10) -> Dict:
    p = np.vstack([r["p"] for r in recs]).ravel()
    y = np.vstack([r["y"] for r in recs]).ravel()
    return {"n_bins": n_bins, "bins": calibration_bins(p, y, n_bins)}


# ---------------------------------------------------------------------------
# 折-候选单元
# ---------------------------------------------------------------------------

def _unit_dir(run_dir: str, fold: int, candidate: str) -> str:
    return os.path.join(run_dir, "units", "f{}_cand{}".format(fold, candidate))


def _unit_usable(unit_dir: str, hashes: Dict) -> bool:
    sp = os.path.join(unit_dir, STATUS_FILE)
    if not (os.path.exists(sp) and os.path.exists(os.path.join(unit_dir, PRED_FILE))):
        return False
    try:
        with open(sp, encoding="utf-8") as fh:
            st = json.load(fh)
    except Exception:
        return False
    if not st.get("complete"):
        return False
    return all(st.get(k) == hashes.get(k) for k in ("code_hash", "data_hash", "config_hash"))


def _load_unit(unit_dir: str) -> Dict:
    with np.load(os.path.join(unit_dir, PRED_FILE), allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    rec["top5"] = np.asarray(rec["top5"]).astype(int).tolist()
    rec["candidate"] = str(rec["candidate"])
    rec["model_hash"] = str(rec["model_hash"])
    rec["fold"] = int(rec["fold"])
    rec["train_end"] = int(rec["train_end"])
    rec["feature_snapshot_hashes"] = [str(x) for x in rec["feature_snapshot_hashes"]]
    rec["issues"] = np.asarray(rec["issues"], dtype=np.int64)
    rec["fold_ids"] = np.asarray(rec["fold_ids"], dtype=np.int64)
    rec["timing_seconds"] = float(rec["timing_seconds"])
    rec["extras"] = json.loads(str(rec.get("extras", "{}")))
    rec["device_used"] = str(rec.get("device_used", ""))
    rec.pop("extras_raw", None)
    return rec


def _save_unit(unit_dir: str, rec: Dict, hashes: Dict, meta: Dict) -> None:
    os.makedirs(unit_dir, exist_ok=True)
    extras = rec.get("extras") or {}
    model_obj = rec.pop("_model_obj", None)

    np.savez_compressed(
        os.path.join(unit_dir, PRED_FILE),
        issues=np.asarray(rec["issues"], dtype=np.int64),
        p=np.asarray(rec["p"], dtype=np.float64),
        p_raw=np.asarray(rec["p_raw"], dtype=np.float64),
        y=np.asarray(rec["y"], dtype=np.int64),
        fold_ids=np.asarray(rec["fold_ids"], dtype=np.int64),
        top5=np.asarray(rec["top5"], dtype=np.int64),
        feature_snapshot_hashes=np.asarray(rec["feature_snapshot_hashes"], dtype=str),
        model_hash=np.array(rec["model_hash"]),
        candidate=np.array(rec["candidate"]),
        fold=np.array(int(rec["fold"])),
        train_end=np.array(int(rec["train_end"])),
        timing_seconds=np.array(float(rec["timing_seconds"])),
        device_used=np.array(str(rec.get("_device_used", ""))),
        extras=np.array(json.dumps(extras, ensure_ascii=False)),
    )

    # 协议要求模型写入独立运行目录；序列化失败不能拖垮整轮运行。
    model_note = "not_saved"
    if model_obj is not None:
        try:
            with open(os.path.join(unit_dir, MODEL_FILE), "wb") as fh:
                pickle.dump(model_obj, fh, protocol=4)
            model_note = "saved"
        except Exception as exc:                        # noqa: BLE001
            model_note = "pickle_failed: {}: {}".format(type(exc).__name__, exc)

    with open(os.path.join(unit_dir, STATUS_FILE), "w", encoding="utf-8") as fh:
        json.dump({"complete": True, "fold": int(rec["fold"]),
                   "candidate": rec["candidate"], **hashes, **meta,
                   "model_file": model_note,
                   "device_used": str(rec.get("_device_used", "")),
                   "extras": extras}, fh, ensure_ascii=False, indent=2)


def _fit_and_predict_unit(fold: Dict, candidate: str, dataset: Dict,
                          config: Dict, device: Optional[str]) -> Dict:
    """训练一个折-候选单元并预测该折的评估期。

    配置里的 ``frozen_blocks`` 索引以 **data.csv 升序后的行号** 为基准
    （``index_base = history_row``）；数据集 X 的第 0 行对应历史行 ``min_history``，
    故此处统一减去该偏移。
    """
    X, y, issues = dataset["X"], dataset["y"], dataset["issues"]
    off = int(dataset.get("min_history", 0))
    tr_lo, tr_hi = fold["train_index_start"] - off, fold["train_index_stop"] - off
    te_lo, te_hi = fold["target_index_start"] - off, fold["target_index_stop"] - off
    if min(tr_lo, te_lo) < 0 or max(tr_hi, te_hi) > len(X):
        raise EvaluationError(
            "折 {} 的索引越界：offset={} train=[{}, {}) target=[{}, {}) n_X={}".format(
                fold["fold"], off, tr_lo, tr_hi, te_lo, te_hi, len(X)))

    X_tr, y_tr, iss_tr = X[tr_lo:tr_hi], y[tr_lo:tr_hi], issues[tr_lo:tr_hi]
    X_te, iss_te = X[te_lo:te_hi], issues[te_lo:te_hi]
    y_te = y[te_lo:te_hi]

    # 显式把 --device 注入本次拟合：fit_candidate 没有 device 形参，候选 N 从
    # config.resources.device 解析设备。若不注入，命令行设备会被静默忽略，
    # 而日志/清单仍记录命令行的值（对抗性复核发现）。
    run_config = config
    if device is not None:
        run_config = dict(config)
        res = dict(config.get("resources", {}))
        res["device"] = str(device)
        run_config["resources"] = res

    t0 = time.time()
    fitted = fit_candidate(candidate, X_tr, y_tr, iss_tr, run_config)
    t_fit = time.time() - t0

    t0 = time.time()
    recs = predict_candidate(fitted, X_te, iss_te, dataset.get("history"))
    t_pred = time.time() - t0

    if len(recs) != (te_hi - te_lo):
        raise EvaluationError("预测期数与评估块不符：{} vs {}".format(
            len(recs), te_hi - te_lo))

    extras: Dict = {}
    if recs and "seed_epochs" in recs[0]:
        extras["seed_epochs"] = [int(e) for e in recs[0]["seed_epochs"]]
        extras["per_seed_epochs"] = [int(e) for e in recs[0]["seed_epochs"]]
    if isinstance(fitted, dict) and fitted.get("seeds"):
        extras["mlp_seeds"] = [
            {"seed": int(r.get("seed", -1)),
             "selected_epoch": int(r.get("selected_epoch", -1)),
             "epochs_run": int(r.get("epochs_run", -1)),
             "best_val_brier": float(r.get("best_val_brier", float("nan"))),
             "n_fit": int(r.get("n_fit", -1)), "n_valid": int(r.get("n_valid", -1))}
            for r in fitted["seeds"]]
        extras["mlp_status"] = fitted.get("status")
        extras["mlp_device"] = fitted.get("device")

    return {
        "fold": int(fold["fold"]),
        "candidate": candidate,
        "train_end": int(iss_tr[-1]),
        "train_start": int(iss_tr[0]),
        "n_train": int(len(iss_tr)),
        "issues": np.asarray([r["issue"] for r in recs], dtype=np.int64),
        "p": np.vstack([r["p"] for r in recs]),
        "p_raw": np.vstack([r["p_raw"] for r in recs]),
        "y": np.asarray(y_te),
        "fold_ids": np.full(len(recs), fold["fold"], dtype=np.int64),
        "top5": [r["top5"] for r in recs],
        "feature_snapshot_hashes": [r["feature_snapshot_hash"] for r in recs],
        "model_hash": recs[0]["model_hash"],
        "timing_seconds": float(t_fit + t_pred),
        "timing_fit_seconds": float(t_fit),
        "timing_predict_seconds": float(t_pred),
        "extras": extras,
        "_model_obj": fitted,          # 由 _save_unit 单独落盘；不进入 npz
        "_device_used": run_config.get("resources", {}).get("device"),
    }


def dataset_fingerprint(dataset: Dict) -> str:
    """数据集内容指纹：直接对 X/y/issues 的字节取哈希。

    不依赖配置里是否写了 sha256，因此不存在"<unset>"这种会把不同数据
    混成同一个缓存键的回退（对抗性复核发现）。
    """
    h = hashlib.sha256()
    for key in ("X", "y", "issues"):
        arr = np.ascontiguousarray(np.asarray(dataset[key]))
        h.update(key.encode("ascii"))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def run_folds(config: Dict, dataset: Dict, run_dir: str, device: Optional[str],
              candidates: Optional[Sequence[str]] = None,
              resume: bool = False) -> Dict:
    """按折 × 候选运行（或复用）所有单元。

    ``resume`` 默认 **False**：不显式要求就不复用既有产物。
    """
    folds = fold_ranges(config)
    cands = list(candidates) if candidates else list(config.get("candidates", {}).keys())
    os.makedirs(os.path.join(run_dir, "units"), exist_ok=True)

    hashes = {
        "code_hash": code_hash(),
        "data_hash": dataset_fingerprint(dataset),
        "config_hash": hash_config(config),
    }

    predictions: List[Dict] = []
    failures: List[Dict] = []
    unit_dirs: Dict[str, str] = {}
    computed = reused = 0

    for fold in folds:
        for cand in cands:
            ud = _unit_dir(run_dir, fold["fold"], cand)
            unit_dirs["{}_{}".format(fold["fold"], cand)] = ud

            if resume and _unit_usable(ud, hashes):
                predictions.append(_load_unit(ud))
                reused += 1
                continue

            if os.path.exists(ud):
                inc = os.path.join(run_dir, "_incomplete")
                os.makedirs(inc, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                dest = os.path.join(
                    inc, "f{}_cand{}_{}".format(fold["fold"], cand, stamp))
                n = 0
                while os.path.exists(dest):      # 同一秒内多次隔离时仍保持唯一
                    n += 1
                    dest = os.path.join(
                        inc, "f{}_cand{}_{}_{}".format(fold["fold"], cand, stamp, n))
                shutil.move(ud, dest)

            try:
                rec = _fit_and_predict_unit(fold, cand, dataset, config, device)
            except Exception as exc:                      # noqa: BLE001
                failures.append({
                    "fold": int(fold["fold"]), "candidate": cand,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "traceback": traceback.format_exc()[-1500:],
                })
                with open(os.path.join(run_dir, RUN_LOG), "a", encoding="utf-8") as fh:
                    fh.write("FAIL fold={} cand={} {}: {}\n".format(
                        fold["fold"], cand, type(exc).__name__, exc))
                continue

            _save_unit(ud, rec, hashes, {"train_end": rec["train_end"],
                                         "n_train": rec["n_train"]})
            predictions.append(rec)
            computed += 1
            with open(os.path.join(run_dir, RUN_LOG), "a", encoding="utf-8") as fh:
                fh.write("OK fold={} cand={} n_test={} train_end={} t={:.2f}s\n".format(
                    fold["fold"], cand, len(rec["issues"]), rec["train_end"],
                    rec["timing_seconds"]))

    return {"predictions": predictions, "failures": failures,
            "computed_units": computed, "reused_units": reused,
            "unit_dirs": unit_dirs, "hashes": hashes, "folds": folds,
            "candidates": cands}


# ---------------------------------------------------------------------------
# 顶层
# ---------------------------------------------------------------------------

def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def _environment_record() -> Dict:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn
        env["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda"] = getattr(torch.version, "cuda", None)
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled())
        env["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        env["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        env["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        env["gpu_names"] = [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
        # 在 run_folds 之后调用，故这里的重新播种不会影响已完成的拟合；
        # 记录 set_determinism 的完整返回，便于事后审计（复核发现其返回值曾被丢弃）。
        from tools.front_probability_mlp import set_determinism
        env["determinism_record"] = set_determinism(20261013, "cpu")
    except Exception as exc:                            # noqa: BLE001
        env["torch_error"] = "{}: {}".format(type(exc).__name__, exc)
    return env


def run_from_csv(config_path: str, data_csv: str, run_dir: str,
                 device: Optional[str] = None, resume: bool = False,
                 candidates: Optional[Sequence[str]] = None) -> Dict:
    """端到端：校验输入 -> 建特征 -> 逐折逐候选 -> 汇总 -> 落盘。"""
    config = load_config(config_path)
    data_info = verify_data(data_csv, config)
    prepare_run_dir(run_dir, config, allow_resume=resume)

    t_start = time.time()
    dataset = build_dataset(data_csv, config)
    folds = fold_ranges(config)

    with open(os.path.join(run_dir, RUN_LOG), "a", encoding="utf-8") as fh:
        fh.write("START config={} data={} sha256={} device={}\n".format(
            config_path, data_csv, data_info["sha256"], device))

    res = run_folds(config, dataset, run_dir, device=device,
                    candidates=candidates, resume=resume)

    # ---- 汇总 ----
    by_cand: Dict[str, List[Dict]] = {}
    for rec in res["predictions"]:
        by_cand.setdefault(rec["candidate"], []).append(rec)

    cand_summary = {c: summarize_candidate(recs, folds) for c, recs in by_cand.items()}

    # 比较用的存储（含 fold_ids）
    store = {}
    for c, recs in by_cand.items():
        store[c] = {
            "issues": np.concatenate([r["issues"] for r in recs]),
            "p": np.vstack([r["p"] for r in recs]),
            "y": np.vstack([r["y"] for r in recs]),
            "fold_ids": np.concatenate([r["fold_ids"] for r in recs]),
        }

    comparisons = {}
    for pair in config.get("statistics", {}).get("comparisons", []):
        if "-" not in pair:
            continue
        model, base = pair.split("-", 1)
        if model in store and base in store:
            comparisons[pair] = compare_pair(store, base, model, config)
        else:
            missing = [x for x in (model, base) if x not in store]
            comparisons[pair] = {"comparison": pair,
                                 "error": "候选未完成：{}".format(",".join(missing)),
                                 "n_periods": 0}

    summary = {
        "run_dir": run_dir,
        "config_path": config_path,
        "data": data_info,
        "hashes": res["hashes"],
        "environment": _environment_record(),
        "device": device,
        "n_blocks": len(folds),
        "blocks": folds,
        "candidates": cand_summary,
        "comparisons": comparisons,
        "failures": res["failures"],
        "computed_units": res["computed_units"],
        "reused_units": res["reused_units"],
        "wall_seconds": time.time() - t_start,
    }
    summary["verdict"] = run_verdict(summary, comparisons, config)

    # ---- 落盘 ----
    _write_json(os.path.join(run_dir, "manifest.json"), {
        "config_path": config_path,
        "config_hash": res["hashes"]["config_hash"],
        "code_hash": res["hashes"]["code_hash"],
        "data": data_info,
        "device": device,
        "candidates": res["candidates"],
        "folds": folds,
        "environment": summary["environment"],
        "seeds": {
            "tie_break": config.get("probability", {}).get("tie_break_seed"),
            "mlp": config.get("candidates", {}).get("N", {}).get("seeds"),
            "bootstrap": config.get("statistics", {}).get("bootstrap", {}).get("seed"),
        },
    })
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_json(os.path.join(run_dir, "block_scores.json"),
                {c: v["per_block_brier"] for c, v in cand_summary.items()})
    _write_json(os.path.join(run_dir, "intervals.json"), comparisons)
    _write_json(os.path.join(run_dir, "calibration.json"),
                {c: calibration_for(recs) for c, recs in by_cand.items()})

    with open(os.path.join(run_dir, "predictions.jsonl"), "w", encoding="utf-8") as fh:
        for rec in res["predictions"]:
            for k in range(len(rec["issues"])):
                fh.write(json.dumps({
                    "issue": int(rec["issues"][k]),
                    "model_id": rec["candidate"],      # 协议 §6 要求的字段名
                    "candidate": rec["candidate"],     # 同一值的别名，便于阅读
                    "fold": int(rec["fold"]),
                    "train_end": int(rec["train_end"]),
                    "model_hash": rec["model_hash"],
                    "feature_snapshot_hash": rec["feature_snapshot_hashes"][k],
                    "p_raw": [float(x) for x in rec["p_raw"][k]],
                    "p": [float(x) for x in rec["p"][k]],
                    "top5": rec["top5"][k],
                    "y": [int(x) for x in rec["y"][k]],
                }, ensure_ascii=False) + "\n")

    with open(os.path.join(run_dir, RUN_LOG), "a", encoding="utf-8") as fh:
        fh.write("DONE computed={} reused={} failures={} wall={:.1f}s\n".format(
            res["computed_units"], res["reused_units"], len(res["failures"]),
            summary["wall_seconds"]))

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="F1 前区概率模型时间顺序评估")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default=None,
                   help="例如 cuda:5；缺省取配置 resources.device")
    p.add_argument("--resume", action="store_true",
                   help="允许在已有运行目录中复用完整单元")
    p.add_argument("--candidates", default=None,
                   help="逗号分隔，仅用于冒烟；正式比较请留空")
    args = p.parse_args(argv)

    cands = args.candidates.split(",") if args.candidates else None
    try:
        summary = run_from_csv(args.config, args.data, args.run_dir,
                               device=args.device, resume=args.resume,
                               candidates=cands)
    except EvaluationError as e:
        print("拒绝运行：{}".format(e), file=sys.stderr)
        return 2

    for c, v in sorted(summary["candidates"].items()):
        print("{}: brier={:.6f} logloss={:.5f} top5_hits={:.4f} n={} t={:.1f}s".format(
            c, v["overall_brier"], v["overall_logloss"], v["top5_mean_hits"],
            v["n_periods"], v["timing_seconds"]))
    print("failures:", summary["failures"])
    print("verdict:", json.dumps(summary["verdict"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""F1 前区边际概率候选模型（U / F / L / R）与公共概率工具。

对应协议：``docs/project_notes/SERVER_F1_HANDOFF_PLAN.md`` §3 与
``docs/project_notes/FRONT_PROBABILITY_RESEARCH_PLAN.md`` §3、§6。

本模块只做**前区 35 个号码的边际概率**预测，固定取 Top-5 形成一组 5+12。
仅模拟，不投注，不读未来数据，不做参数搜索。

接口（与服务器交接计划 §F1-1 一致）::

    fit_candidate(model_id, X_train, y_train, train_issues, config) -> fitted
    predict_candidate(fitted, X_target, target_issues, history_before_target) -> records

约定：

* ``p_raw`` 是模型原始输出概率；``p`` 是固定 logit 偏移后 sum(p)=5 的评价概率。
* U / F 天然满足 sum(p)=5，不做额外变换。
* 平局顺序由 ``default_rng(tie_break_seed + issue)`` 的固定排列决定，
  与开奖结果无关。
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

FRONT_COLS = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
N_FRONT = 35
DEFAULT_TIE_BREAK_SEED = 20261013
DEFAULT_TARGET_SUM = 5.0
DEFAULT_CLIP_EPS = 1e-8
PROBE_SEED = 20261013
PROBE_ROWS = 8
PROBE_ROUND = 7          # 量化位数：吸收 BLAS 线程级浮点噪声，保留模型差异

MODEL_IDS = ("U", "F", "L", "R", "N")


# ---------------------------------------------------------------------------
# 概率工具
# ---------------------------------------------------------------------------

def uniform_probabilities(n_front: int = N_FRONT,
                          value: Optional[float] = None) -> np.ndarray:
    """U：每个号码 p = 5/35，总和恒为 5。"""
    if value is None:
        value = DEFAULT_TARGET_SUM / float(n_front)
    return np.full(n_front, float(value), dtype=float)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return np.log(p / (1.0 - p))


def _solve_offset(lg: np.ndarray, target_sum: float,
                  iters: int = 100, tol: float = 1e-14) -> np.ndarray:
    """求 a 使 sum(sigmoid(lg + a)) = target_sum，按最后一维向量化。

    g(a) = sum(sigmoid(lg+a)) - target 关于 a 严格单调递增，故用
    **带区间保护的牛顿法**：牛顿步落在 [lo, hi] 外时退化为二分。
    初值取一阶近似 a0 = (target - sum(p0)) / sum(p0(1-p0))，通常 3~6 步收敛。
    """
    shape = lg.shape[:-1]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    lo = np.full(shape, -60.0)
    hi = np.full(shape, 60.0)

    p0 = sig(lg)
    denom = np.maximum((p0 * (1.0 - p0)).sum(axis=-1), 1e-300)
    a = np.clip((target_sum - p0.sum(axis=-1)) / denom, lo, hi)

    for _ in range(iters):
        p = sig(lg + a[..., None])
        g = p.sum(axis=-1) - target_sum
        if np.all(np.abs(g) < 1e-13):
            break
        lo = np.where(g < 0.0, a, lo)
        hi = np.where(g < 0.0, hi, a)
        gp = np.maximum((p * (1.0 - p)).sum(axis=-1), 1e-300)
        newton = a - g / gp
        a_new = np.where((newton > lo) & (newton < hi), newton, 0.5 * (lo + hi))
        if np.all(np.abs(a_new - a) < tol):
            a = a_new
            break
        a = a_new
    return a


def apply_sum_constraint(p_raw: Sequence[float],
                         target_sum: float = DEFAULT_TARGET_SUM,
                         clip_eps: float = DEFAULT_CLIP_EPS) -> np.ndarray:
    """固定数量一致性修正：求常数 a 使 sum(sigmoid(logit(p_raw)+a)) = target_sum。

    这是**固定的确定性求根**，不涉及任何标签，也不称为经过验证的概率校准。
    支持 (35,) 与 (n, 35) 两种输入形状。
    """
    p = np.clip(np.asarray(p_raw, dtype=float), clip_eps, 1.0 - clip_eps)
    lg = _logit(p)
    a = _solve_offset(lg, float(target_sum))
    return 1.0 / (1.0 + np.exp(-(lg + a[..., None])))


def top5_from_probabilities(p: Sequence[float], issue: int,
                            tie_break_seed: int = DEFAULT_TIE_BREAK_SEED,
                            top_k: int = 5) -> List[int]:
    """按概率降序取 Top-5；完全相等时按固定的按期号排列打破平局。

    平局顺序来自 ``default_rng(tie_break_seed + issue)``，只依赖期号与预登记的
    固定种子，绝不使用目标期标签。
    """
    p = np.asarray(p, dtype=float)
    n = p.shape[0]
    perm = np.random.default_rng(int(tie_break_seed) + int(issue)).permutation(n)
    rank = np.empty(n, dtype=np.int64)
    rank[perm] = np.arange(n, dtype=np.int64)
    order = sorted(range(n), key=lambda j: (-float(p[j]), int(rank[j])))
    return [int(j) + 1 for j in order[:top_k]]


def brier_per_period(p: Sequence[float], y: Sequence[int]) -> float:
    """按期 Brier：先在 35 个号上取均值，合成一个期级损失。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def logloss_per_period(p: Sequence[float], y: Sequence[int],
                       eps: float = 1e-12) -> float:
    """按期边际 binary log loss（边际评分，不是整组联合似然）。"""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibration_bins(p: Sequence[float], y: Sequence[int],
                     n_bins: int = 10) -> List[Dict[str, float]]:
    """固定 10 个等宽概率箱的校准描述（只描述，不用于放宽晋级）。"""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        out.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": n,
            "mean_p": float(p[m].mean()) if n else float("nan"),
            "mean_y": float(y[m].mean()) if n else float("nan"),
        })
    return out


# ---------------------------------------------------------------------------
# 哈希
# ---------------------------------------------------------------------------

def feature_snapshot_hash(x: Sequence[float]) -> str:
    """目标期特征向量的 sha256（float64 原始字节）。"""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _probe_matrix(n_features: int, rows: int = PROBE_ROWS) -> np.ndarray:
    return np.random.default_rng(PROBE_SEED).random((rows, int(n_features)))


def model_hash(fitted: Dict) -> str:
    """模型指纹：参数 + 在固定探针矩阵上的（量化后）输出。

    L/R/N 用探针输出区分不同拟合；U/F 无参数依赖，退化为参数指纹。
    输出量化到 1e-7，以吸收 BLAS 线程级浮点噪声而不掩盖模型差异。
    """
    h = hashlib.sha256()
    h.update(str(fitted.get("model_id", "?")).encode("utf-8"))
    h.update(json.dumps(fitted.get("params", {}), sort_keys=True,
                        ensure_ascii=False).encode("utf-8"))
    probe = fitted.get("probe_output")
    if probe is not None:
        h.update(np.round(np.asarray(probe, dtype=np.float64),
                          PROBE_ROUND).tobytes())
    return h.hexdigest()


def _probe_output(make_predict, n_features: int) -> Optional[bytes]:
    try:
        out = make_predict(_probe_matrix(n_features))
    except Exception:
        return None
    return np.ascontiguousarray(np.asarray(out, dtype=np.float64))


# ---------------------------------------------------------------------------
# RF 正类概率
# ---------------------------------------------------------------------------

def proba_from_multioutput_classifier(model, X: np.ndarray) -> np.ndarray:
    """从 MultiOutputClassifier 取每个输出的**正类（标签 1）**概率。

    必须用各子估计器的 ``classes_`` 定位标签 1：
    - 只有类 0 时正类概率为 0；
    - 只有类 1 时正类概率为 1；
    - 禁止把单列概率直接当成正类概率。
    """
    X = np.asarray(X, dtype=float)
    estimators = list(getattr(model, "estimators_", []))
    P = np.zeros((X.shape[0], len(estimators)), dtype=float)
    for j, est in enumerate(estimators):
        classes = list(getattr(est, "classes_", []))
        if 1 in classes:
            P[:, j] = est.predict_proba(X)[:, classes.index(1)]
        else:
            P[:, j] = 0.0
    return P


# ---------------------------------------------------------------------------
# F：平滑频次基线
# ---------------------------------------------------------------------------

def smoothed_frequency_probabilities(history_df: pd.DataFrame, issue: int,
                                     window: int = 100,
                                     prior_strength: int = 100,
                                     prior_per_number: float = 5.0 / 35.0
                                     ) -> np.ndarray:
    """F：最近 ``window`` 期各号出现次数 + 先验，总和恒为 5。

    p_j = (count_j + prior_strength * prior_per_number) / (n_hist + prior_strength)

    只使用**严格早于 target issue** 的开奖；即使历史表里含目标期及之后的记录，
    也一律排除。窗口不足时按实际期数计算。
    """
    hist = history_df
    if hist is not None and len(hist) > 0 and "期数" in hist.columns:
        hist = hist[hist["期数"].astype(int) < int(issue)]
    else:
        hist = hist.iloc[0:0] if hist is not None else history_df
    hist = hist.tail(int(window))

    counts = np.zeros(N_FRONT, dtype=float)
    for row in hist[FRONT_COLS].to_numpy():
        for v in row:
            counts[int(v) - 1] += 1.0

    denom = float(len(hist)) + float(prior_strength)
    return (counts + float(prior_strength) * float(prior_per_number)) / denom


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------

def _candidate_cfg(config: Dict, model_id: str) -> Dict:
    cands = config.get("candidates", {})
    if model_id not in cands:
        raise KeyError("配置中缺少候选 {}".format(model_id))
    return cands[model_id]


def _prob_cfg(config: Dict) -> Dict:
    p = config.get("probability", {})
    return {
        "clip_eps": float(p.get("clip_eps", DEFAULT_CLIP_EPS)),
        "target_sum": float(p.get("target_sum", DEFAULT_TARGET_SUM)),
        "tie_break_seed": int(p.get("tie_break_seed", DEFAULT_TIE_BREAK_SEED)),
    }


def fit_candidate(model_id: str, X_train, y_train, train_issues, config: Dict) -> Dict:
    """在**当折训练部分**拟合一个冻结候选。

    Args:
        model_id: "U" / "F" / "L" / "R" / "N"
        X_train: (n, d) 特征，只能来自训练折
        y_train: (n, 35) 0/1 标签
        train_issues: (n,) 训练期号（供诊断与哈希）
        config: 冻结配置

    Returns:
        含 ``model_id`` / ``params`` / 模型对象 / ``model_hash`` 的字典。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler

    model_id = str(model_id).upper()
    if model_id == "N":
        # 延迟导入，避免与 front_probability_mlp 形成模块级循环依赖
        from tools.front_probability_mlp import fit_mlp_candidate
        return fit_mlp_candidate(X_train, y_train, train_issues, config)

    ccfg = _candidate_cfg(config, model_id)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    n_features = X_train.shape[1] if X_train.ndim == 2 else 0

    fitted: Dict = {
        "model_id": model_id,
        "params": dict(ccfg),
        "_prob_cfg": _prob_cfg(config),
        "n_features": int(n_features),
        "n_train": int(X_train.shape[0]) if X_train.ndim == 2 else 0,
        "train_issue_min": int(np.min(train_issues)) if len(train_issues) else None,
        "train_issue_max": int(np.max(train_issues)) if len(train_issues) else None,
    }

    if model_id == "U":
        fitted["probability"] = uniform_probabilities(
            N_FRONT, ccfg.get("probability"))

    elif model_id == "F":
        # F 是查历史的无参基线：训练期号仅用于记录
        pass

    elif model_id == "L":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        models = []
        for j in range(N_FRONT):
            # 冻结配置要求 L2。sklearn>=1.8 弃用了 penalty 关键字，且 lbfgs 的
            # 默认惩罚在各受支持版本里都是 L2，故此处不显式传 penalty，
            # 由 tests 中的正则化回归测试守住语义。
            clf = LogisticRegression(
                C=float(ccfg.get("C", 1.0)),
                solver=ccfg.get("solver", "lbfgs"),
                max_iter=int(ccfg.get("max_iter", 1000)),
                class_weight=ccfg.get("class_weight"),
            )
            clf.fit(Xs, y_train[:, j])
            models.append(clf)
        fitted["scaler"] = scaler
        fitted["models"] = models

    elif model_id == "R":
        base = RandomForestClassifier(
            n_estimators=int(ccfg["n_estimators"]),
            max_depth=int(ccfg["max_depth"]),
            min_samples_split=int(ccfg["min_samples_split"]),
            min_samples_leaf=int(ccfg["min_samples_leaf"]),
            random_state=int(ccfg["random_state"]),
            n_jobs=int(config.get("resources", {}).get("rf_n_jobs", 1)),
        )
        model = MultiOutputClassifier(base)
        model.fit(X_train, y_train)
        fitted["model"] = model

    else:
        raise ValueError("fit_candidate 不处理 {}（N 见 front_probability_mlp）".format(
            model_id))

    # 探针输出（用于模型指纹）
    if model_id == "L":
        fitted["probe_output"] = _probe_output(
            lambda P: np.column_stack([
                _binary_positive_proba(m, fitted["scaler"].transform(P))
                for m in fitted["models"]]), n_features)
    elif model_id == "R":
        fitted["probe_output"] = _probe_output(
            lambda P: proba_from_multioutput_classifier(fitted["model"], P), n_features)

    fitted["model_hash"] = model_hash(fitted)
    return fitted


def _binary_positive_proba(clf, X: np.ndarray) -> np.ndarray:
    """二分类器的正类概率（同样按 classes_ 定位标签 1）。"""
    classes = list(getattr(clf, "classes_", []))
    if 1 in classes:
        return clf.predict_proba(X)[:, classes.index(1)]
    return np.zeros(X.shape[0], dtype=float)


def _raw_probabilities(fitted: Dict, X_target: np.ndarray,
                       target_issues: Sequence[int],
                       history_before_target: Optional[pd.DataFrame]) -> np.ndarray:
    """返回 (n_target, 35) 原始概率。"""
    mid = fitted["model_id"]
    X_target = np.asarray(X_target, dtype=float)

    if mid == "U":
        return np.tile(fitted["probability"], (X_target.shape[0], 1))

    if mid == "F":
        if history_before_target is None:
            raise ValueError("F 需要显式传入 history_before_target")
        ccfg = fitted["params"]
        return np.vstack([
            smoothed_frequency_probabilities(
                history_before_target, int(iss),
                window=int(ccfg["window"]),
                prior_strength=int(ccfg["prior_strength"]),
                prior_per_number=float(ccfg["prior_per_number"]))
            for iss in target_issues])

    if mid == "L":
        Xs = fitted["scaler"].transform(X_target)
        return np.column_stack([
            _binary_positive_proba(m, Xs) for m in fitted["models"]])

    if mid == "R":
        return proba_from_multioutput_classifier(fitted["model"], X_target)

    raise ValueError("predict_candidate 不处理 {}（N 见 front_probability_mlp）".format(mid))


def predict_candidate(fitted: Dict, X_target, target_issues,
                      history_before_target: Optional[pd.DataFrame] = None) -> List[Dict]:
    """对目标期输出逐期记录。

    Returns:
        每期一个 dict，含 ``issue`` / ``model_id`` / ``p_raw`` / ``p`` / ``top5``
        / ``feature_snapshot_hash`` / ``model_hash``。
        ``fold`` 与 ``train_end`` 由评估器补齐。
    """
    pcfg = fitted.get("_prob_cfg", None)
    X_target = np.asarray(X_target, dtype=float)
    target_issues = [int(i) for i in target_issues]

    if fitted.get("model_id") == "N":
        from tools.front_probability_mlp import predict_mlp_candidate
        return predict_mlp_candidate(fitted, X_target, target_issues,
                                     history_before_target)

    raw = _raw_probabilities(fitted, X_target, target_issues, history_before_target)

    eps = DEFAULT_CLIP_EPS
    target_sum = DEFAULT_TARGET_SUM
    tie_seed = DEFAULT_TIE_BREAK_SEED
    if pcfg:
        eps, target_sum, tie_seed = pcfg["clip_eps"], pcfg["target_sum"], pcfg["tie_break_seed"]

    recs: List[Dict] = []
    for k, iss in enumerate(target_issues):
        pr = np.asarray(raw[k], dtype=float)
        if fitted["model_id"] in ("U", "F"):
            p = pr.copy()                      # 已满足 sum(p)=5，不做额外变换
        else:
            p = apply_sum_constraint(pr, target_sum=target_sum, clip_eps=eps)
        recs.append({
            "issue": iss,
            "model_id": fitted["model_id"],
            "p_raw": pr,
            "p": p,
            "top5": top5_from_probabilities(p, iss, tie_break_seed=tie_seed),
            "feature_snapshot_hash": feature_snapshot_hash(X_target[k]),
            "model_hash": fitted["model_hash"],
        })
    return recs


def attach_prob_cfg(fitted: Dict, config: Dict) -> Dict:
    """把概率修正参数挂到 fitted 上，供 predict_candidate 使用。"""
    fitted["_prob_cfg"] = _prob_cfg(config)
    return fitted

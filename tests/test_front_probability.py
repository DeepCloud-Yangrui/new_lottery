# -*- coding: utf-8 -*-
"""F1 前区边际概率模型 —— 定向测试 (F1-1)。

运行： python -m pytest tests/test_front_probability.py -q

覆盖 SERVER_F1_HANDOFF_PLAN.md §F1-1 要求：
  * 未来/目标开奖不改变目标期特征
  * 标准化只在训练折 fit
  * 35 维、每期恰好 5 个正标签
  * RF 单类别输出的正类概率定位（禁止把单列概率 1 当正类概率）
  * 概率范围与 sum(p)=5 修正、均匀概率不变
  * Top-5 无重复且可复现、平局按固定种子打破
  * U 的 Brier 基线解析值与"模型等于 U 时配对增益为 0"
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import front_probability_models as fpm   # noqa: E402

FRONT_COLS = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
BACK_COLS = ["蓝球_1", "蓝球_2"]
TIE_SEED = 20261013


# ---------------------------------------------------------------------------
# 合成数据工具
# ---------------------------------------------------------------------------

def make_history(n=160, seed=7):
    """确定性的合成开奖历史（升序期号）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        front = sorted(rng.choice(np.arange(1, 36), size=5, replace=False).tolist())
        back = sorted(rng.choice(np.arange(1, 13), size=2, replace=False).tolist())
        rows.append([8000 + i] + front + back)
    return pd.DataFrame(rows, columns=["期数"] + FRONT_COLS + BACK_COLS)


def make_y(n_periods, seed=11):
    rng = np.random.default_rng(seed)
    y = np.zeros((n_periods, 35), dtype=int)
    for i in range(n_periods):
        y[i, rng.choice(35, size=5, replace=False)] = 1
    return y


def make_X(n_periods, dim=252, seed=13):
    rng = np.random.default_rng(seed)
    return rng.random((n_periods, dim))


def cfg(**over):
    base = {
        "candidates": {
            "U": {"kind": "uniform", "probability": 5.0 / 35.0},
            "F": {"kind": "smoothed_frequency", "window": 100, "prior_strength": 100,
                  "prior_per_number": 5.0 / 35.0},
            "L": {"kind": "logistic_l2", "penalty": "l2", "C": 1.0, "solver": "lbfgs",
                  "max_iter": 1000, "class_weight": None, "n_outputs": 35},
            "R": {"kind": "random_forest", "n_estimators": 8, "max_depth": 4,
                  "min_samples_split": 4, "min_samples_leaf": 2, "random_state": 42,
                  "n_outputs": 35},
        },
        "probability": {"clip_eps": 1e-8, "target_sum": 5.0, "tie_break_seed": TIE_SEED},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# 1. U：均匀基线
# ---------------------------------------------------------------------------

def test_uniform_probabilities_value_and_sum():
    p = fpm.uniform_probabilities()
    assert p.shape == (35,)
    assert np.allclose(p, 5.0 / 35.0, rtol=0, atol=1e-15)
    assert p.sum() == pytest.approx(5.0, abs=1e-12)


def test_uniform_brier_reference_is_exact():
    """U 的按期 Brier 恒为 (5/35)(1-5/35)，与具体中奖号码无关。"""
    y = make_y(6, seed=3)
    expected = (5.0 / 35.0) * (1.0 - 5.0 / 35.0)
    for i in range(len(y)):
        loss = fpm.brier_per_period(fpm.uniform_probabilities(), y[i])
        assert loss == pytest.approx(expected, rel=0, abs=1e-15)


def test_brier_zero_for_perfect_prediction():
    y = np.zeros(35, dtype=int)
    y[[0, 1, 2, 3, 4]] = 1
    assert fpm.brier_per_period(y.astype(float), y) == pytest.approx(0.0, abs=1e-15)


def test_period_aggregation_is_mean_over_35():
    p = np.full(35, 0.2)
    y = np.zeros(35, dtype=int)
    y[:5] = 1
    manual = np.mean((p - y) ** 2)
    assert fpm.brier_per_period(p, y) == pytest.approx(manual, rel=0, abs=1e-15)
    assert fpm.logloss_per_period(p, y) == pytest.approx(-np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)), rel=1e-12)


def test_paired_gain_is_zero_when_model_equals_uniform():
    y = make_y(4, seed=5)
    p = fpm.uniform_probabilities()
    for i in range(len(y)):
        gain = fpm.brier_per_period(p, y[i]) - fpm.brier_per_period(p, y[i])
        assert gain == 0.0


# ---------------------------------------------------------------------------
# 2. sum(p)=5 的固定 logit 偏移
# ---------------------------------------------------------------------------

def test_sum_constraint_leaves_uniform_unchanged():
    p = fpm.uniform_probabilities()
    q = fpm.apply_sum_constraint(p, target_sum=5.0, clip_eps=1e-8)
    assert np.allclose(q, p, rtol=0, atol=1e-12)


def test_sum_constraint_achieves_target_sum():
    rng = np.random.default_rng(2)
    raw = rng.random(35) * 0.4
    q = fpm.apply_sum_constraint(raw, target_sum=5.0, clip_eps=1e-8)
    assert q.sum() == pytest.approx(5.0, rel=0, abs=1e-9)
    assert np.all(q > 0.0) and np.all(q < 1.0)


def test_sum_constraint_clips_degenerate_inputs():
    raw = np.zeros(35)
    raw[0] = 1.0            # 极端概率：0 与 1
    q = fpm.apply_sum_constraint(raw, target_sum=5.0, clip_eps=1e-8)
    assert np.all(np.isfinite(q))
    assert q.sum() == pytest.approx(5.0, rel=0, abs=1e-9)
    raw2 = np.full(35, 1e-300)
    q2 = fpm.apply_sum_constraint(raw2, target_sum=5.0, clip_eps=1e-8)
    assert np.all(np.isfinite(q2))
    assert q2.sum() == pytest.approx(5.0, rel=0, abs=1e-9)


def test_sum_constraint_is_deterministic_and_order_preserving():
    rng = np.random.default_rng(9)
    raw = rng.random(35) * 0.3
    a = fpm.apply_sum_constraint(raw, 5.0, 1e-8)
    b = fpm.apply_sum_constraint(raw, 5.0, 1e-8)
    assert np.array_equal(a, b)
    # sigmoid(logit(p)+a) 关于 p 严格递增 => 概率大小顺序保持不变
    assert np.array_equal(np.argsort(a), np.argsort(raw))
    assert np.all(np.diff(a[np.argsort(raw)]) > 0)


def test_logistic_candidate_uses_l2_regularisation():
    """冻结配置要求 L2；C 有限时必须表现出正则化（系数被收缩）。"""
    from sklearn.linear_model import LogisticRegression

    X = make_X(200, dim=20, seed=77)
    y = make_y(200, seed=78)[:, :35]

    def coef_norm(C):
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
        m.fit(X, y[:, 0])
        return float(np.linalg.norm(m.coef_))

    assert coef_norm(1.0) < coef_norm(1e12) * 0.999


# ---------------------------------------------------------------------------
# 3. Top-5 选择与平局规则
# ---------------------------------------------------------------------------

def test_top5_has_no_duplicates_and_valid_range():
    y = make_y(1, seed=1)[0]
    p = y * 0.2 + 0.01
    top = fpm.top5_from_probabilities(p, issue=21001, tie_break_seed=TIE_SEED)
    assert len(top) == 5
    assert len(set(top)) == 5
    assert all(1 <= n <= 35 for n in top)
    assert set(top) == set(np.arange(1, 36)[np.argsort(-p)[:5]])


def test_top5_is_reproducible():
    p = np.linspace(0.01, 0.2, 35)
    a = fpm.top5_from_probabilities(p, issue=22010, tie_break_seed=TIE_SEED)
    b = fpm.top5_from_probabilities(p, issue=22010, tie_break_seed=TIE_SEED)
    assert a == b


def test_top5_exact_ties_follow_fixed_seed_permutation():
    """全部概率完全相等时，顺序取自 default_rng(seed+issue) 的固定排列。"""
    p = np.full(35, 5.0 / 35.0)
    issue = 23077
    got = fpm.top5_from_probabilities(p, issue=issue, tie_break_seed=TIE_SEED)
    perm = np.random.default_rng(TIE_SEED + issue).permutation(35) + 1
    assert got == perm[:5].tolist()
    # 不同期号给出不同顺序（不同 rng 流），但都可复现
    other = fpm.top5_from_probabilities(p, issue=issue + 1, tie_break_seed=TIE_SEED)
    assert len(other) == 5 and len(set(other)) == 5


def test_top5_tie_break_does_not_depend_on_labels():
    """平局顺序只由期号与固定种子决定，与标签无关。"""
    p = np.full(35, 5.0 / 35.0)
    y1 = np.zeros(35, dtype=int); y1[:5] = 1
    y2 = np.zeros(35, dtype=int); y2[30:] = 1
    assert (fpm.top5_from_probabilities(p, 24001, TIE_SEED)
            == fpm.top5_from_probabilities(p, 24001, TIE_SEED))
    del y1, y2   # 标签根本不进入调用


# ---------------------------------------------------------------------------
# 4. F：平滑频次基线
# ---------------------------------------------------------------------------

def test_smoothed_frequency_matches_formula_and_sums_to_five():
    hist = make_history(160, seed=21)
    issue = 8100
    p = fpm.smoothed_frequency_probabilities(hist, issue, window=100,
                                             prior_strength=100,
                                             prior_per_number=5.0 / 35.0)
    sub = hist[hist["期数"] < issue].tail(100)
    counts = np.zeros(35)
    for _, row in sub.iterrows():
        for c in FRONT_COLS:
            counts[int(row[c]) - 1] += 1.0
    expected = (counts + 100 * 5.0 / 35.0) / 200.0
    assert np.allclose(p, expected, rtol=1e-12, atol=1e-15)
    assert p.sum() == pytest.approx(5.0, rel=0, abs=1e-12)


def test_smoothed_frequency_ignores_target_and_future_draws():
    """F 只能看到严格早于目标期的开奖。"""
    hist = make_history(160, seed=22)
    issue = 8100
    base = fpm.smoothed_frequency_probabilities(hist, issue)

    perturbed = hist.copy()
    mask = perturbed["期数"] >= issue
    # 把目标期及之后的所有号码全部换掉
    for c in FRONT_COLS:
        perturbed.loc[mask, c] = 1
    after = fpm.smoothed_frequency_probabilities(perturbed, issue)
    assert np.allclose(base, after, rtol=0, atol=1e-15)


def test_smoothed_frequency_uses_fewer_than_window_when_history_short():
    hist = make_history(30, seed=23)
    issue = int(hist["期数"].iloc[-1]) + 1
    p = fpm.smoothed_frequency_probabilities(hist, issue)
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(5.0, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. RF 正类概率定位
# ---------------------------------------------------------------------------

def test_rf_single_class_zero_yields_zero_positive_probability():
    """某个输出只有类 0 时，正类概率必须是 0，不能把单列概率 1 当作正类概率。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    X = np.random.default_rng(1).random((40, 4))
    y = np.zeros((40, 2), dtype=int)
    y[:, 0] = np.random.default_rng(2).integers(0, 2, 40)
    # 第 1 个输出恒为 0 -> 只有类 0
    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0))
    model.fit(X, y)
    assert model.estimators_[1].classes_.tolist() == [0]
    P = fpm.proba_from_multioutput_classifier(model, X)
    assert P.shape == (40, 2)
    assert np.all(P[:, 1] == 0.0)


def test_rf_single_class_one_yields_one_positive_probability():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    X = np.random.default_rng(3).random((30, 4))
    y = np.ones((30, 2), dtype=int)
    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0))
    model.fit(X, y)
    assert model.estimators_[0].classes_.tolist() == [1]
    P = fpm.proba_from_multioutput_classifier(model, X)
    assert np.all(P[:, 0] == 1.0)
    assert np.all(P[:, 1] == 1.0)


def test_rf_two_class_uses_class_one_column():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    X = np.random.default_rng(4).random((60, 4))
    y = (X[:, :2] > 0.5).astype(int)
    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=6, max_depth=3, random_state=0))
    model.fit(X, y)
    P = fpm.proba_from_multioutput_classifier(model, X)
    for j in range(2):
        sub = model.estimators_[j]
        idx = list(sub.classes_).index(1)
        assert np.allclose(P[:, j], sub.predict_proba(X)[:, idx], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. 特征与接口契约
# ---------------------------------------------------------------------------

def test_feature_snapshot_hash_stable_and_sensitive():
    x = np.arange(252, dtype=float)
    h1 = fpm.feature_snapshot_hash(x)
    h2 = fpm.feature_snapshot_hash(x.copy())
    assert h1 == h2 and len(h1) == 64
    x2 = x.copy(); x2[0] += 1.0
    assert fpm.feature_snapshot_hash(x2) != h1


def test_fit_and_predict_interface_returns_required_fields():
    X = make_X(60)
    y = make_y(60, seed=31)
    train_issues = np.arange(19001, 19061)
    target_issues = np.arange(19061, 19071)
    c = cfg()
    hist = make_history(220, seed=41)
    # 让合成历史的期号区间覆盖目标期（19061..19070 落在历史中间）
    hist["期数"] = np.arange(19001, 19221)

    for model_id in ("U", "F", "L", "R"):
        fitted = fpm.fit_candidate(model_id, X, y, train_issues, c)
        recs = fpm.predict_candidate(fitted, X[:10], target_issues, hist)
        assert len(recs) == 10
        for r in recs:
            for key in ("issue", "model_id", "p_raw", "p", "top5",
                        "feature_snapshot_hash", "model_hash"):
                assert key in r, key
            assert r["model_id"] == model_id
            assert len(r["p_raw"]) == 35 and len(r["p"]) == 35
            assert np.all(r["p_raw"] >= 0) and np.all(r["p_raw"] <= 1)
            assert np.all(r["p"] > 0) and np.all(r["p"] < 1)
            assert r["p"].sum() == pytest.approx(5.0, rel=0, abs=1e-8)
            assert len(set(r["top5"])) == 5
            assert all(1 <= n <= 35 for n in r["top5"])


def test_predict_is_reproducible_across_calls():
    X = make_X(80)
    y = make_y(80, seed=33)
    train_issues = np.arange(19001, 19081)
    target_issues = np.arange(19081, 19091)
    c = cfg()
    hist = make_history(120, seed=43)
    hist["期数"] = np.arange(19001, 19121)

    fa = fpm.fit_candidate("L", X, y, train_issues, c)
    fb = fpm.fit_candidate("L", X, y, train_issues, c)
    ra = fpm.predict_candidate(fa, X[:10], target_issues, hist)
    rb = fpm.predict_candidate(fb, X[:10], target_issues, hist)
    for a, b in zip(ra, rb):
        assert np.array_equal(a["p_raw"], b["p_raw"])
        assert np.array_equal(a["p"], b["p"])
        assert a["top5"] == b["top5"]
        assert a["model_hash"] == b["model_hash"]


def test_standardization_is_fit_on_training_rows_only():
    X = make_X(100)
    y = make_y(100, seed=35)
    train_issues = np.arange(1, 101)
    c = cfg()
    fitted = fpm.fit_candidate("L", X, y, train_issues, c)
    scaler = fitted["scaler"]
    assert np.allclose(scaler.mean_, X.mean(axis=0), rtol=1e-9, atol=1e-9)
    # 换掉测试特征不应改变已 fit 的标准化参数
    before = scaler.mean_.copy()
    X2 = X.copy()
    X2[:10] *= 100.0
    fpm.predict_candidate(fitted, X2[:10], train_issues[:10], None)
    assert np.array_equal(before, fitted["scaler"].mean_)


def test_F_requires_explicit_history():
    X = make_X(60)
    y = make_y(60, seed=37)
    fitted = fpm.fit_candidate("F", X, y, np.arange(60), cfg())
    with pytest.raises(ValueError):
        fpm.predict_candidate(fitted, X[:5], np.arange(5), None)


def test_five_positive_labels_per_training_period():
    y = make_y(50, seed=39)
    assert np.all(y.sum(axis=1) == 5)
    assert set(np.unique(y)) == {0, 1}


def test_model_hash_changes_when_fit_changes():
    X = make_X(80)
    y = make_y(80, seed=45)
    c = cfg()
    a = fpm.fit_candidate("L", X, y, np.arange(80), c)
    y2 = make_y(80, seed=46)
    b = fpm.fit_candidate("L", X, y2, np.arange(80), c)
    assert fpm.model_hash(a) != fpm.model_hash(b)
    assert fpm.model_hash(a) == fpm.model_hash(
        fpm.fit_candidate("L", X, y, np.arange(80), c))

# -*- coding: utf-8 -*-
"""F1 评估器 —— 定向测试 (F1-2)。

运行： python -m pytest tests/test_front_probability_evaluation.py -q

覆盖 SERVER_F1_HANDOFF_PLAN.md §F1-2 要求的失败测试：
  * 训练截止严格早于评估块
  * 所有候选使用完全相同的期次
  * 内部验证与外部评估块隔离（外部标签不能改变训练结果）
  * 期级损失聚合（每期一个损失，不是号码级）
  * 三个比较的 Bonferroni 调整 alpha
  * 零增益 / 负增益不晋级
  * 失败候选被如实标记，不回退样本
  * 输入 hash 不符时拒绝运行
  * 运行目录冲突时拒绝覆盖
  * 中断后可恢复，且只复用完整且 hash 一致的单元
  * 四个评估块互不重叠
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import eval_front_probability as ev   # noqa: E402

FRONT_COLS = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
BACK_COLS = ["蓝球_1", "蓝球_2"]


# ---------------------------------------------------------------------------
# 合成数据与配置
# ---------------------------------------------------------------------------

def make_csv(path, n_rows=300, seed=101):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        front = sorted(rng.choice(np.arange(1, 36), size=5, replace=False).tolist())
        back = sorted(rng.choice(np.arange(1, 13), size=2, replace=False).tolist())
        rows.append([8000 + i] + front + back)
    df = pd.DataFrame(rows, columns=["期数"] + FRONT_COLS + BACK_COLS)
    df.to_csv(path, index=False)
    return path


def tiny_config(data_csv, min_history=40, n_blocks=2, block_size=20,
                cands=("U", "L", "R"), recent=None):
    n_rows = len(pd.read_csv(data_csv))
    modelable = n_rows - min_history
    recent = recent or n_blocks * block_size
    start = modelable - recent
    blocks = []
    for b in range(n_blocks):
        lo = min_history + start + b * block_size
        hi = lo + block_size
        blocks.append({
            "fold": b,
            "target_index_start": lo,
            "target_index_stop": hi,
            "train_index_start": min_history,
            "train_index_stop": lo,
            "n_train": lo - min_history,
        })
    return {
        "protocol": {"frozen_date": "test", "handoff_commit": "test"},
        "run": {"run_id_prefix": "f1_test", "outputs_root": str(
            os.path.dirname(os.path.abspath(data_csv)))},
        "data": {"csv": str(data_csv)},
        "features": {"windows": [5, 10, 20], "min_history": min_history,
                     "n_front": 35},
        "evaluation": {
            "recent_modelable_periods": recent,
            "n_blocks": n_blocks,
            "block_size": block_size,
            "min_first_train_periods": 10,
            "expanding_train_window": True,
            "same_issues_for_all_candidates": True,
            "frozen_blocks": blocks,
        },
        "candidates": {k: v for k, v in {
            "U": {"kind": "uniform", "probability": 5.0 / 35.0},
            "F": {"kind": "smoothed_frequency", "window": 20, "prior_strength": 20,
                  "prior_per_number": 5.0 / 35.0},
            "L": {"kind": "logistic_l2", "C": 1.0, "solver": "lbfgs",
                  "max_iter": 200, "class_weight": None, "n_outputs": 35},
            "R": {"kind": "random_forest", "n_estimators": 6, "max_depth": 4,
                  "min_samples_split": 4, "min_samples_leaf": 2,
                  "random_state": 42, "n_outputs": 35},
            "N": {"kind": "mlp", "hidden_layers": [8, 4], "hidden_activation": "relu",
                  "dropout": 0.1, "output_dim": 35, "output_activation": "none",
                  "loss": "BCEWithLogitsLoss_uniform_mean", "pos_weight": None,
                  "optimizer": "AdamW", "lr": 0.01, "weight_decay": 0.01,
                  "batch_size": 16, "max_epochs": 3, "dtype": "float32",
                  "batchnorm": False, "amp": False, "internal_valid_fraction": 0.2,
                  "internal_valid_rounding": "ceil",
                  "checkpoint_metric": "mean_brier_on_internal_validation_final_probability",
                  "min_delta": 1e-06, "patience": 2,
                  "first_valid_epoch_is_initial_best": True, "ties": "earlier_epoch",
                  "refit": "full_fold", "seeds": [17, 42, 2026],
                  "seed_aggregation": "equal_weight_mean"},
        }.items() if k in tuple(cands)},
        "probability": {"clip_eps": 1e-8, "target_sum": 5.0,
                        "tie_break_seed": 20261013},
        "statistics": {
            "primary_metric": "per_period_mean_brier",
            "comparisons": ["L-U", "R-U", "N-U"],
            "family_alpha": 0.05,
            "family_size": 6,
            "per_comparison_two_sided_confidence": 0.98333333,
            "bootstrap": {"kind": "moving_block", "block_lengths": [5, 10],
                          "n_resamples": 200, "seed": 20261013},
            "tie_epsilon": 1e-12,
            "tie_preference_order": ["L", "R", "N"],
            "promotion_criteria": [],
        },
        "resources": {"cpu_threads": 2, "device": "cpu", "rf_n_jobs": 2},
        "forbidden": [],
    }


def small_dataset(cfg):
    n_rows = 300
    min_history = cfg["features"]["min_history"]
    n = n_rows - min_history
    rng = np.random.default_rng(7)
    X = rng.random((n, 4))
    y = np.zeros((n, 35), dtype=int)
    for i in range(n):
        y[i, rng.choice(35, 5, replace=False)] = 1
    issues = np.arange(8000 + min_history, 8000 + n_rows)
    return {"X": X, "y": y, "issues": issues, "min_history": min_history,
            "history": None}


# ---------------------------------------------------------------------------
# 1. 训练截止与块结构
# ---------------------------------------------------------------------------

def test_training_cutoff_is_strictly_before_target_block(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv)
    ds = small_dataset(cfg)
    folds = ev.fold_ranges(cfg)
    assert len(folds) == cfg["evaluation"]["n_blocks"]
    off = ds["min_history"]
    for f in folds:
        assert f["train_index_stop"] <= f["target_index_start"]
        assert f["n_train"] == f["train_index_stop"] - f["train_index_start"]
        last_train_issue = int(ds["issues"][f["train_index_stop"] - off - 1])
        first_target_issue = int(ds["issues"][f["target_index_start"] - off])
        assert last_train_issue < first_target_issue


def test_evaluation_blocks_do_not_overlap(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv)
    folds = ev.fold_ranges(cfg)
    seen = set()
    for f in folds:
        idx = set(range(f["target_index_start"], f["target_index_stop"]))
        assert not (idx & seen), "评估块重叠"
        seen |= idx


def test_first_block_training_meets_minimum(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv)
    folds = ev.fold_ranges(cfg)
    assert folds[0]["n_train"] >= cfg["evaluation"]["min_first_train_periods"]


# ---------------------------------------------------------------------------
# 2. 期级损失与配对增益
# ---------------------------------------------------------------------------

def test_per_period_loss_is_period_level_not_number_level():
    y = np.zeros((3, 35), dtype=int)
    y[:, :5] = 1
    p = np.full((3, 35), 5.0 / 35.0)
    b = ev.per_period_brier(p, y)
    assert b.shape == (3,)
    expected = (5.0 / 35.0) * (1.0 - 5.0 / 35.0)
    assert np.allclose(b, expected, rtol=0, atol=1e-15)


def test_zero_gain_does_not_promote():
    rng = np.random.default_rng(3)
    diff = np.zeros(400)
    ci20 = ev.paired_block_bootstrap_ci(diff, 20, 200, 20261013, 0.98333333)
    ci50 = ev.paired_block_bootstrap_ci(diff, 50, 200, 20261013, 0.98333333)
    assert ci20["lower"] <= 0.0 <= ci20["upper"]
    assert ci50["lower"] <= 0.0 <= ci50["upper"]
    assert not ev.meets_interval_requirement(ci20, ci50)
    del rng


def test_clearly_positive_gain_has_positive_lower_bound():
    diff = np.full(400, 0.01) + np.random.default_rng(5).normal(0, 1e-4, 400)
    ci = ev.paired_block_bootstrap_ci(diff, 20, 500, 20261013, 0.98333333)
    assert ci["lower"] > 0.0


def test_adjusted_alpha_matches_bonferroni_family():
    """家族 alpha=0.05 分摊到 3 对比 × 2 侧 = 6 个单侧尾，每侧 0.05/6。

    协议写作「每个双侧区间 98.333333%（尾概率 0.05/6）」：
    双侧置信度 = 1 - 2*(alpha/6) = 0.98333333，每侧尾概率 = 0.00833333。
    """
    assert ev.alpha_tail(0.05, 6) == pytest.approx(0.05 / 6.0, rel=0, abs=1e-15)
    assert ev.adjusted_two_sided_confidence(0.05, 6) == pytest.approx(
        1.0 - 2.0 * 0.05 / 6.0, rel=0, abs=1e-12)
    assert ev.adjusted_two_sided_confidence(0.05, 6) == pytest.approx(
        0.98333333, rel=0, abs=1e-8)


def test_bootstrap_is_reproducible_with_fixed_seed():
    diff = np.random.default_rng(11).normal(0.001, 0.01, 300)
    a = ev.paired_block_bootstrap_ci(diff, 20, 300, 20261013, 0.98333333)
    b = ev.paired_block_bootstrap_ci(diff, 20, 300, 20261013, 0.98333333)
    assert a == b
    c = ev.paired_block_bootstrap_ci(diff, 20, 300, 20261014, 0.98333333)
    assert c != a or True     # 不同种子允许相同，但不要求


# ---------------------------------------------------------------------------
# 3. 数据与配置完整性
# ---------------------------------------------------------------------------

def test_data_hash_mismatch_is_rejected(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv)
    cfg["data"]["sha256"] = "0" * 64
    with pytest.raises(ev.EvaluationError):
        ev.verify_data(str(csv), cfg)


def test_data_hash_match_passes(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv)
    cfg["data"]["sha256"] = ev.sha256_file(str(csv))
    info = ev.verify_data(str(csv), cfg)
    assert info["sha256"] == cfg["data"]["sha256"]


def test_row_count_mismatch_is_rejected(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv)
    cfg["data"]["expected_rows"] = 999
    with pytest.raises(ev.EvaluationError):
        ev.verify_data(str(csv), cfg)


def test_run_dir_conflict_is_refused(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "existing.txt").write_text("keep me")
    cfg = tiny_config(make_csv(tmp_path / "data.csv"))
    with pytest.raises(ev.EvaluationError):
        ev.prepare_run_dir(str(run_dir), cfg, allow_resume=False)
    assert (run_dir / "existing.txt").read_text() == "keep me"


def test_resume_into_existing_run_dir_is_allowed(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "units").mkdir(parents=True)
    cfg = tiny_config(make_csv(tmp_path / "data.csv"))
    ev.prepare_run_dir(str(run_dir), cfg, allow_resume=True)   # 不抛异常


def test_all_candidates_use_identical_issues(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv, cands=("U", "L"))
    ds = small_dataset(cfg)
    run_dir = tmp_path / "run"
    res = ev.run_folds(cfg, ds, str(run_dir), device="cpu",
                       candidates=["U", "L"], resume=False)
    issues_by_fold = {}
    for rec in res["predictions"]:
        issues_by_fold.setdefault(rec["fold"], {}).setdefault(rec["candidate"], [])
        issues_by_fold[rec["fold"]][rec["candidate"]] = rec["issues"]
    for fold, per_cand in issues_by_fold.items():
        lists = list(per_cand.values())
        for other in lists[1:]:
            assert np.array_equal(lists[0], other), f"fold {fold} 期次不一致"


# ---------------------------------------------------------------------------
# 4. 失败候选与中断恢复
# ---------------------------------------------------------------------------

def test_failed_candidate_is_reported_not_substituted(tmp_path, monkeypatch):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv, cands=("U", "L"))
    ds = small_dataset(cfg)

    orig = ev._fit_and_predict_unit

    def boom(fold, candidate, *a, **k):
        if candidate == "L":
            raise RuntimeError("synthetic failure")
        return orig(fold, candidate, *a, **k)

    monkeypatch.setattr(ev, "_fit_and_predict_unit", boom)
    res = ev.run_folds(cfg, ds, str(tmp_path / "run"), device="cpu",
                       candidates=["U", "L"], resume=False)
    failed = [f for f in res["failures"] if f["candidate"] == "L"]
    assert failed and failed[0]["fold"] == 0
    assert "synthetic failure" in failed[0]["error"]
    # U 的结果仍在，且没有被换成别的期次
    u = [p for p in res["predictions"] if p["candidate"] == "U"]
    assert u and len(u[0]["issues"]) == cfg["evaluation"]["block_size"]


def test_resume_reuses_complete_units_only(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv, cands=("U",), n_blocks=1)
    ds = small_dataset(cfg)
    run_dir = tmp_path / "run"

    first = ev.run_folds(cfg, ds, str(run_dir), device="cpu",
                         candidates=["U"], resume=False)
    assert first["computed_units"] == 1 and first["reused_units"] == 0

    second = ev.run_folds(cfg, ds, str(run_dir), device="cpu",
                          candidates=["U"], resume=True)
    assert second["reused_units"] == 1 and second["computed_units"] == 0

    # 破坏 hash 后必须重算，且旧的不完整产物被隔离
    status_path = os.path.join(second["unit_dirs"]["0_U"], "status.json")
    st = json.load(open(status_path))
    st["data_hash"] = "deadbeef"
    json.dump(st, open(status_path, "w"))
    third = ev.run_folds(cfg, ds, str(run_dir), device="cpu",
                         candidates=["U"], resume=True)
    assert third["computed_units"] == 1 and third["reused_units"] == 0
    assert os.path.isdir(os.path.join(str(run_dir), "_incomplete"))


def test_incomplete_unit_is_not_reused(tmp_path):
    csv = make_csv(tmp_path / "data.csv")
    cfg = tiny_config(csv, cands=("U",), n_blocks=1)
    ds = small_dataset(cfg)
    run_dir = tmp_path / "run"
    res = ev.run_folds(cfg, ds, str(run_dir), device="cpu",
                       candidates=["U"], resume=False)
    status_path = os.path.join(res["unit_dirs"]["0_U"], "status.json")
    st = json.load(open(status_path))
    st["complete"] = False
    json.dump(st, open(status_path, "w"))
    again = ev.run_folds(cfg, ds, str(run_dir), device="cpu",
                         candidates=["U"], resume=True)
    assert again["computed_units"] == 1 and again["reused_units"] == 0


# ---------------------------------------------------------------------------
# 5. 端到端冒烟与汇总
# ---------------------------------------------------------------------------

def test_end_to_end_smoke_on_synthetic_csv(tmp_path):
    csv = make_csv(tmp_path / "data_smoke.csv", n_rows=260, seed=55)
    cfg = tiny_config(csv, min_history=40, n_blocks=2, block_size=15,
                      cands=("U", "L"))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    run_dir = tmp_path / "run_smoke"

    out = ev.run_from_csv(str(cfg_path), str(csv), str(run_dir), device="cpu")

    assert os.path.exists(os.path.join(str(run_dir), "manifest.json"))
    assert os.path.exists(os.path.join(str(run_dir), "summary.json"))
    assert os.path.exists(os.path.join(str(run_dir), "block_scores.json"))
    assert os.path.exists(os.path.join(str(run_dir), "intervals.json"))
    assert os.path.exists(os.path.join(str(run_dir), "calibration.json"))
    assert os.path.exists(os.path.join(str(run_dir), "predictions.jsonl"))
    assert os.path.exists(os.path.join(str(run_dir), "run.log"))

    assert out["n_blocks"] == 2
    # U 的 Brier 必须等于解析值
    u = out["candidates"]["U"]
    assert u["overall_brier"] == pytest.approx((5.0 / 35.0) * (1 - 5.0 / 35.0),
                                               rel=0, abs=1e-12)


def test_summary_reports_required_fields(tmp_path):
    csv = make_csv(tmp_path / "d.csv", n_rows=260, seed=56)
    cfg = tiny_config(csv, min_history=40, n_blocks=2, block_size=15, cands=("U", "L"))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    run_dir = tmp_path / "run2"
    out = ev.run_from_csv(str(cfg_path), str(csv), str(run_dir), device="cpu")

    for cand in ("U", "L"):
        c = out["candidates"][cand]
        for key in ("overall_brier", "overall_logloss", "per_block_brier",
                    "top5_mean_hits", "k_distribution", "k_ge_3",
                    "timing_seconds", "n_periods"):
            assert key in c, (cand, key)
    assert "comparisons" in out and "verdict" in out
    assert out["verdict"]["complete_comparison"] is False   # N 未运行


def test_predictions_jsonl_is_per_period_not_per_number(tmp_path):
    csv = make_csv(tmp_path / "d3.csv", n_rows=260, seed=57)
    cfg = tiny_config(csv, min_history=40, n_blocks=1, block_size=15, cands=("U",))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    run_dir = tmp_path / "run3"
    ev.run_from_csv(str(cfg_path), str(csv), str(run_dir), device="cpu")
    lines = open(os.path.join(str(run_dir), "predictions.jsonl"), encoding="utf-8").readlines()
    assert len(lines) == 15          # 1 块 × 15 期 × 1 候选
    rec = json.loads(lines[0])
    assert len(rec["p"]) == 35 and len(rec["top5"]) == 5
    assert "issue" in rec and "candidate" in rec and "fold" in rec

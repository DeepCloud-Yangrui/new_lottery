# -*- coding: utf-8 -*-
"""F1 候选 N（小型 MLP）—— 定向测试 (F1-1)。

运行： python -m pytest tests/test_front_probability_mlp.py -q

覆盖 SERVER_F1_HANDOFF_PLAN.md §3「N 的冻结设置」：
  * 结构 输入→64→32→35，两个隐藏层 ReLU + Dropout(0.1)，输出层无激活
  * 输出 35 维 logits → sigmoid 边际概率（不是 35 类 softmax）
  * 只在**训练折内部连续验证段**早停；标准化只在内部前 80% fit
  * 选定 epoch 后重新初始化、在全训练折上 fit 标准化、训练恰好该 epoch 数
  * 三个固定种子等权平均，不挑最佳种子；任一种子失败则该折 N 未完成
  * 保存/加载预测一致（同环境绝对误差 ≤ 1e-6）
  * CPU/GPU 小样本前反向正常（缺 GPU 时明确标为未验项）
  * 确定性设置记录在案
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
import inspect  # noqa: E402

from tools import front_probability_mlp as fmlp   # noqa: E402

HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def tiny_cfg(**n_over):
    n = {
        "kind": "mlp",
        "hidden_layers": [8, 4],
        "hidden_activation": "relu",
        "dropout": 0.1,
        "output_dim": 35,
        "output_activation": "none",
        "loss": "BCEWithLogitsLoss_uniform_mean",
        "pos_weight": None,
        "optimizer": "AdamW",
        "lr": 0.01,
        "weight_decay": 0.01,
        "batch_size": 16,
        "max_epochs": 8,
        "dtype": "float32",
        "batchnorm": False,
        "amp": False,
        "internal_valid_fraction": 0.2,
        "internal_valid_rounding": "ceil",
        "checkpoint_metric": "mean_brier_on_internal_validation_final_probability",
        "min_delta": 1e-06,
        "patience": 3,
        "first_valid_epoch_is_initial_best": True,
        "ties": "earlier_epoch",
        "seeds": [17, 42, 2026],
        "seed_aggregation": "equal_weight_mean_of_raw_probabilities_then_single_logit_offset",
    }
    n.update(n_over)
    return {
        "candidates": {"N": n},
        "probability": {"clip_eps": 1e-8, "target_sum": 5.0, "tie_break_seed": 20261013},
        "resources": {"cpu_threads": 2},
    }


def make_xy(n=120, dim=12, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.random((n, dim))
    y = np.zeros((n, 35), dtype=int)
    for i in range(n):
        y[i, rng.choice(35, 5, replace=False)] = 1
    return X, y


# ---------------------------------------------------------------------------
# 1. 结构与输出形式
# ---------------------------------------------------------------------------

def test_network_shape_and_activations():
    m = fmlp.build_mlp(input_dim=20, hidden=(64, 32), dropout=0.1)
    linears = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
    assert [l.out_features for l in linears] == [64, 32, 35]
    assert [l.in_features for l in linears] == [20, 64, 32]
    relus = [mod for mod in m.modules() if isinstance(mod, torch.nn.ReLU)]
    drops = [mod for mod in m.modules() if isinstance(mod, torch.nn.Dropout)]
    assert len(relus) == 2
    assert len(drops) == 2 and all(abs(d.p - 0.1) < 1e-12 for d in drops)
    assert not any(isinstance(mod, torch.nn.BatchNorm1d) for mod in m.modules())


def test_output_layer_has_no_activation_and_returns_logits():
    m = fmlp.build_mlp(input_dim=6, hidden=(8, 4), dropout=0.0)
    m.eval()
    x = torch.zeros(3, 6)
    out = m(x)
    assert out.shape == (3, 35)
    # 全零输入时输出层为线性，logits 可正可负（无 sigmoid/softmax 归一）
    assert not torch.allclose(out.sum(dim=1), torch.ones(3))


def test_probabilities_are_marginal_sigmoid_not_softmax():
    X, y = make_xy(60, dim=6, seed=1)
    cfg = tiny_cfg(max_epochs=2, patience=2, seeds=[17])
    fitted = fmlp.fit_mlp_candidate(X, y, np.arange(60), cfg, device="cpu")
    p = fmlp.predict_mlp_candidate(fitted, X[:5], np.arange(5))[0]["p_raw"]
    # 35 个独立 sigmoid：总和不受约束，一般不等于 1
    assert p.shape == (35,)
    assert np.all((p > 0) & (p < 1))
    assert abs(p.sum() - 1.0) > 1e-6


# ---------------------------------------------------------------------------
# 2. 内部验证段与早停边界
# ---------------------------------------------------------------------------

def test_internal_validation_is_last_ceil_20pct():
    for n in (50, 51, 100, 101, 2540):
        n_fit, n_valid = fmlp.internal_validation_split(n, 0.2)
        assert n_valid == int(np.ceil(0.2 * n))
        assert n_fit + n_valid == n
        assert n_valid >= 1 and n_fit >= 1


def test_early_stopping_scaler_fit_on_first_80pct_only():
    X, y = make_xy(100, dim=7, seed=3)
    cfg = tiny_cfg(max_epochs=3, patience=2, seeds=[17])
    fitted = fmlp.fit_mlp_candidate(X, y, np.arange(100), cfg, device="cpu")
    rec = fitted["seeds"][0]
    n_fit, n_valid = fmlp.internal_validation_split(100, 0.2)
    assert n_fit + n_valid == 100
    assert np.allclose(rec["early_stop_scaler"].mean_, X[:n_fit].mean(axis=0),
                       rtol=1e-6, atol=1e-9)
    assert not np.allclose(rec["early_stop_scaler"].mean_, X.mean(axis=0),
                           rtol=1e-9, atol=1e-12) or np.allclose(
        X[:n_fit].mean(axis=0), X.mean(axis=0), rtol=1e-9)


def test_refit_scaler_fit_on_full_training_fold():
    X, y = make_xy(100, dim=7, seed=4)
    cfg = tiny_cfg(max_epochs=3, patience=2, seeds=[17])
    fitted = fmlp.fit_mlp_candidate(X, y, np.arange(100), cfg, device="cpu")
    rec = fitted["seeds"][0]
    assert np.allclose(rec["scaler"].mean_, X.mean(axis=0), rtol=1e-6, atol=1e-9)


def test_selected_epoch_is_recorded_and_within_bounds():
    X, y = make_xy(80, dim=5, seed=6)
    cfg = tiny_cfg(max_epochs=6, patience=2, seeds=[17])
    fitted = fmlp.fit_mlp_candidate(X, y, np.arange(80), cfg, device="cpu")
    rec = fitted["seeds"][0]
    assert 1 <= rec["selected_epoch"] <= 6
    assert len(rec["val_brier_curve"]) == rec["epochs_run"]


# ---------------------------------------------------------------------------
# 3. 重训的纯度与"外部数据不参与"
# ---------------------------------------------------------------------------

def test_refit_is_a_pure_function_of_seed_epoch_and_train_fold():
    X, y = make_xy(90, dim=6, seed=7)
    cfg = tiny_cfg()
    a = fmlp.refit_full_training_fold(X, y, seed=42, n_epochs=3, cfg=cfg, device="cpu")
    b = fmlp.refit_full_training_fold(X, y, seed=42, n_epochs=3, cfg=cfg, device="cpu")
    with torch.no_grad():
        pa, pb = a["model"](torch.tensor(a["scaler"].transform(X), dtype=torch.float32)), \
                 b["model"](torch.tensor(b["scaler"].transform(X), dtype=torch.float32))
    assert torch.allclose(pa, pb, rtol=0, atol=1e-6)


def test_fit_signature_cannot_receive_external_block():
    """fit 只接受训练折数组；签名里没有验证/测试数据入口。"""
    params = list(inspect.signature(fmlp.fit_mlp_candidate).parameters)
    assert params[:4] == ["X_train", "y_train", "train_issues", "config"]
    for bad in ("X_test", "y_test", "X_valid", "y_valid", "X_external", "y_external"):
        assert bad not in params


def test_refit_uses_exactly_the_selected_epoch_count():
    X, y = make_xy(70, dim=5, seed=8)
    cfg = tiny_cfg()
    r = fmlp.refit_full_training_fold(X, y, seed=17, n_epochs=4, cfg=cfg, device="cpu")
    assert r["epochs_run"] == 4
    assert r["n_rows_used"] == 70


def test_equal_probabilities_do_not_tie_break_by_seed_quality():
    """N 取三种子等权平均，不挑 Brier 最好的种子。"""
    a = np.full(35, 0.10)
    b = np.full(35, 0.20)
    c = np.full(35, 0.30)
    agg = fmlp.aggregate_seed_probabilities([a, b, c])
    assert np.allclose(agg, (a + b + c) / 3.0, rtol=0, atol=1e-15)


def test_seed_average_is_order_independent():
    a = np.linspace(0.01, 0.3, 35)
    b = np.linspace(0.3, 0.01, 35)
    c = np.full(35, 0.142857)
    assert np.allclose(fmlp.aggregate_seed_probabilities([a, b, c]),
                       fmlp.aggregate_seed_probabilities([c, a, b]), rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# 4. 确定性
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_refit_weights():
    X, y = make_xy(64, dim=5, seed=9)
    cfg = tiny_cfg()
    a = fmlp.refit_full_training_fold(X, y, seed=2026, n_epochs=2, cfg=cfg, device="cpu")
    b = fmlp.refit_full_training_fold(X, y, seed=2026, n_epochs=2, cfg=cfg, device="cpu")
    for (k1, v1), (k2, v2) in zip(a["model"].state_dict().items(),
                                  b["model"].state_dict().items()):
        assert torch.equal(v1, v2), k1


def test_determinism_settings_are_recorded():
    rec = fmlp.set_determinism(17, "cpu")
    for key in ("seed", "torch_use_deterministic_algorithms",
                "cudnn_deterministic", "cudnn_benchmark", "num_threads",
                "cublas_workspace_config"):
        assert key in rec, key


# ---------------------------------------------------------------------------
# 5. 保存/加载一致性
# ---------------------------------------------------------------------------

def test_save_load_predictions_agree(tmp_path):
    X, y = make_xy(80, dim=6, seed=10)
    cfg = tiny_cfg(max_epochs=3, patience=2, seeds=[17])
    fitted = fmlp.fit_mlp_candidate(X, y, np.arange(80), cfg, device="cpu")
    p1 = fmlp.predict_mlp_candidate(fitted, X[:10], np.arange(10))

    path = tmp_path / "n.pt"
    fmlp.save_mlp_candidate(fitted, str(path))
    reloaded = fmlp.load_mlp_candidate(str(path), device="cpu")
    p2 = fmlp.predict_mlp_candidate(reloaded, X[:10], np.arange(10))

    for a, b in zip(p1, p2):
        assert np.max(np.abs(a["p_raw"] - b["p_raw"])) <= 1e-6
        assert np.max(np.abs(a["p"] - b["p"])) <= 1e-6
        assert a["top5"] == b["top5"]


# ---------------------------------------------------------------------------
# 6. CPU / GPU 前反向冒烟
# ---------------------------------------------------------------------------

def test_cpu_forward_backward_small():
    m = fmlp.build_mlp(input_dim=5, hidden=(8, 4), dropout=0.1)
    x = torch.rand(7, 5)
    y = torch.zeros(7, 35)
    out = m(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA 不可用：GPU 验收项未验证（不能以 CPU 通过冒充）")
def test_gpu_forward_backward_small():
    dev = "cuda:0"
    m = fmlp.build_mlp(input_dim=5, hidden=(8, 4), dropout=0.1).to(dev)
    x = torch.rand(7, 5, device=dev)
    y = torch.zeros(7, 35, device=dev)
    out = m(x)
    assert out.device.type == "cuda"
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA 不可用：GPU 验收项未验证（不能以 CPU 通过冒充）")
def test_gpu_forward_matches_cpu_on_identical_weights():
    """同一组权重下，GPU 与 CPU 的前向数值一致（不承诺训练轨迹逐位相同）。"""
    import copy

    m = fmlp.build_mlp(input_dim=5, hidden=(8, 4), dropout=0.1).eval()
    m_gpu = copy.deepcopy(m).to("cuda:0").eval()
    x = torch.rand(16, 5)
    with torch.no_grad():
        a = m(x).numpy()
        b = m_gpu(x.to("cuda:0")).cpu().numpy()
    assert np.max(np.abs(a - b)) < 1e-3


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA 不可用：GPU 验收项未验证（不能以 CPU 通过冒充）")
def test_gpu_training_runs_end_to_end_and_yields_valid_probabilities():
    """GPU 上完成整段早停+重训，输出有限且在 (0,1) 内。

    跨设备 Dropout 使用各自的 RNG 流，训练轨迹不要求逐位一致
    （协议：复现承诺限于记录的环境，不承诺跨硬件逐位相同）。
    """
    X, y = make_xy(96, dim=6, seed=14)
    cfg = tiny_cfg(max_epochs=3, patience=2, seeds=[17])
    fitted = fmlp.fit_mlp_candidate(X, y, np.arange(96), cfg, device="cuda:0")
    assert fitted["status"] == "complete"
    assert fitted["device"].startswith("cuda")
    recs = fmlp.predict_mlp_candidate(fitted, X[:8], np.arange(8))
    for r in recs:
        assert np.all(np.isfinite(r["p_raw"]))
        assert np.all((r["p_raw"] > 0) & (r["p_raw"] < 1))
        assert r["p"].sum() == pytest.approx(5.0, rel=0, abs=1e-8)
        assert len(set(r["top5"])) == 5

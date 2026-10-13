# -*- coding: utf-8 -*-
"""F1 候选 N：小型 MLP（PyTorch）。

冻结设置见 ``docs/project_notes/SERVER_F1_HANDOFF_PLAN.md`` §3「N 的冻结设置」，
配置见 ``configs/front_probability_f1.json`` 的 ``candidates.N``。

要点
----
* 结构：输入 → 64 → 32 → 35；两个隐藏层 ReLU + Dropout(0.1)；输出层无激活。
  输出 35 个 logits，各自 sigmoid 得到**边际**概率（不是 35 类 softmax）。
* 优化：AdamW(lr=1e-3, weight_decay=1e-2)、batch 64、最多 200 epoch、
  BCEWithLogitsLoss 均匀平均、不加正类权重。
* 早停：训练折最后 ceil(20%) 作为**内部连续验证段**，标准化只在内部前 80% fit；
  每 epoch 按验证段**最终概率**（已作 sum(p)=5 修正）的平均 Brier 选 checkpoint，
  最小改善 1e-6、patience 20、首次有效 epoch 为初始最好点、平局取更早 epoch。
* 重训：选定 epoch 后**重新初始化**，在**全训练折**上 fit 标准化，训练恰好该 epoch 数。
  外部 200 期块绝不参与早停或权重选择。
* 种子：17 / 42 / 2026 三次都跑；N = 三个原始概率向量的等权平均，
  再作一次 sum(p)=5 的统一 logit 偏移。**不挑最佳种子**。
  任一种子失败 → 该折 N 标记未完成。
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Sequence

# cuBLAS 确定性要求：必须在 cuBLAS 初始化之前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from tools.front_probability_models import (  # noqa: E402
    DEFAULT_CLIP_EPS,
    DEFAULT_TARGET_SUM,
    DEFAULT_TIE_BREAK_SEED,
    N_FRONT,
    apply_sum_constraint,
    feature_snapshot_hash,
    model_hash,
    top5_from_probabilities,
)

CUBLAS_WORKSPACE_CONFIG = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
DEFAULT_SEEDS = (17, 42, 2026)


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------

def set_determinism(seed: int, device: str) -> Dict:
    """固定 Python / NumPy / PyTorch 随机状态与确定性设置，并记录实际取值。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    torch.set_num_threads(int(os.environ.get("F1_CPU_THREADS", "8")))

    return {
        "seed": seed,
        "device": str(device),
        "torch_use_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "num_threads": int(torch.get_num_threads()),
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
    }


# ---------------------------------------------------------------------------
# 结构
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, hidden: Sequence[int] = (64, 32),
              dropout: float = 0.1) -> nn.Module:
    """输入 → hidden... → 35 logits。隐藏层 ReLU + Dropout，输出层无激活。"""
    layers: List[nn.Module] = []
    prev = int(input_dim)
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), nn.ReLU(), nn.Dropout(float(dropout))]
        prev = int(h)
    layers.append(nn.Linear(prev, N_FRONT))
    return nn.Sequential(*layers)


def internal_validation_split(n_train: int, frac: float = 0.2):
    """训练折最后 ceil(frac*n) 期作为内部连续验证段。"""
    n_train = int(n_train)
    n_valid = int(math.ceil(float(frac) * n_train))
    n_valid = max(1, min(n_valid, n_train - 1))
    return n_train - n_valid, n_valid


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def _to_tensor(a, device):
    return torch.tensor(np.ascontiguousarray(a, dtype=np.float32), dtype=torch.float32,
                        device=device)


def _make_loader(Xs: np.ndarray, y: np.ndarray, batch_size: int,
                 gen_seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(gen_seed))
    ds = TensorDataset(_to_tensor(Xs, "cpu"), _to_tensor(y, "cpu"))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=True,
                      generator=g, num_workers=0, drop_last=False)


def _train_epochs(model: nn.Module, Xs: np.ndarray, y: np.ndarray,
                  cfg: Dict, n_epochs: int, gen_seed: int,
                  device: str) -> None:
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    loader = _make_loader(Xs, y, int(cfg["batch_size"]), gen_seed)
    model.train()
    for _ in range(int(n_epochs)):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def _final_probabilities(model: nn.Module, Xs: np.ndarray, device: str,
                         target_sum: float, clip_eps: float) -> np.ndarray:
    """模型输出的**最终概率**：sigmoid 后按 sum(p)=target_sum 作统一 logit 偏移。"""
    model.eval()
    logits = model(_to_tensor(Xs, device)).detach().cpu().numpy().astype(np.float64)
    raw = 1.0 / (1.0 + np.exp(-logits))
    return apply_sum_constraint(raw, target_sum=target_sum, clip_eps=clip_eps)


def _n_cfg(cfg: Dict) -> Dict:
    """接受完整配置或已展开的 N 子配置。"""
    if isinstance(cfg, dict) and isinstance(cfg.get("candidates"), dict):
        return cfg["candidates"]["N"]
    return cfg


def refit_full_training_fold(X_train, y_train, seed: int, n_epochs: int,
                             cfg: Dict, device: str) -> Dict:
    """重新初始化，在**全训练折**上 fit 标准化并训练恰好 n_epochs 个 epoch。

    纯函数：仅由 (X_train, y_train, seed, n_epochs, cfg, device) 决定。
    """
    cfg = _n_cfg(cfg)
    set_determinism(seed, device)
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    torch.manual_seed(int(seed))
    model = build_mlp(X.shape[1], tuple(cfg["hidden_layers"]),
                      float(cfg["dropout"])).to(device)
    _train_epochs(model, Xs, y, cfg, int(n_epochs), int(seed) + 1000003, device)
    model.eval()                     # 返回的模型供推理使用，关闭 Dropout
    return {"model": model, "scaler": scaler,
            "epochs_run": int(n_epochs), "n_rows_used": int(X.shape[0])}


def train_one_seed(X_train, y_train, seed: int, cfg: Dict, device: str) -> Dict:
    """单个种子：内部验证早停选 epoch，再在全训练折上重训同样 epoch 数。"""
    cfg = _n_cfg(cfg)
    seed = int(seed)
    set_determinism(seed, device)
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    n_fit, n_valid = internal_validation_split(X.shape[0],
                                               float(cfg["internal_valid_fraction"]))

    Xf, yf = X[:n_fit], y[:n_fit]
    Xv, yv = X[n_fit:], y[n_fit:]
    scaler_es = StandardScaler().fit(Xf)          # 只在前 80% fit
    Xfs = scaler_es.transform(Xf)
    Xvs = scaler_es.transform(Xv)

    torch.manual_seed(seed)
    model = build_mlp(X.shape[1], tuple(cfg["hidden_layers"]),
                      float(cfg["dropout"])).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    loader = _make_loader(Xfs, yf, int(cfg["batch_size"]), seed + 1000003)

    yv_np = yv.astype(np.float64)
    target_sum = float(cfg.get("_target_sum", DEFAULT_TARGET_SUM))
    clip_eps = float(cfg.get("_clip_eps", DEFAULT_CLIP_EPS))
    min_delta = float(cfg["min_delta"])
    patience = int(cfg["patience"])
    max_epochs = int(cfg["max_epochs"])

    best_epoch = 1
    best_brier = None
    wait = 0
    curve: List[float] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()

        p_val = _final_probabilities(model, Xvs, device, target_sum, clip_eps)
        brier = float(np.mean((p_val - yv_np) ** 2))
        curve.append(brier)

        if best_brier is None or brier < best_brier - min_delta:
            # 首次有效 epoch 即初始最好点；平局（不满足 min_delta）保留更早 epoch
            best_brier = brier
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    refit = refit_full_training_fold(X, y, seed, best_epoch, cfg, device)
    return {
        "seed": seed,
        "selected_epoch": int(best_epoch),
        "epochs_run": len(curve),
        "val_brier_curve": curve,
        "best_val_brier": float(best_brier),
        "n_fit": int(n_fit),
        "n_valid": int(n_valid),
        "early_stop_scaler": scaler_es,
        "scaler": refit["scaler"],
        "model": refit["model"],
        "refit_epochs": refit["epochs_run"],
    }


def aggregate_seed_probabilities(per_seed: Sequence[np.ndarray]) -> np.ndarray:
    """N 的预登记聚合：三个原始概率向量的**等权平均**（不挑最佳种子）。"""
    arrs = [np.asarray(a, dtype=float) for a in per_seed]
    if not arrs:
        raise ValueError("没有可聚合的种子概率")
    return np.mean(np.stack(arrs, axis=0), axis=0)


# ---------------------------------------------------------------------------
# 统一接口
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[str], config: Dict) -> str:
    dev = device if device is not None else config.get("resources", {}).get("device")
    if not dev:
        dev = "cpu"
    dev = str(dev)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求 {} 但 CUDA 不可用；不得静默回退到 CPU".format(dev))
    return dev


def fit_mlp_candidate(X_train, y_train, train_issues, config: Dict,
                      device: Optional[str] = None) -> Dict:
    """拟合候选 N：三个固定种子各自早停+重训，等权平均。"""
    from tools.front_probability_models import _prob_cfg

    ccfg = dict(config["candidates"]["N"])
    pcfg = _prob_cfg(config)
    ccfg["_target_sum"] = pcfg["target_sum"]
    ccfg["_clip_eps"] = pcfg["clip_eps"]
    dev = _resolve_device(device, config)

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    seeds = list(ccfg.get("seeds", DEFAULT_SEEDS))

    per_seed: List[Dict] = []
    failures: List[Dict] = []
    for s in seeds:
        try:
            per_seed.append(train_one_seed(X, y, s, ccfg, dev))
        except Exception as exc:                       # noqa: BLE001
            failures.append({"seed": int(s),
                             "error": "{}: {}".format(type(exc).__name__, exc)})

    status = "complete" if (not failures and len(per_seed) == len(seeds)) else "incomplete"

    fitted: Dict = {
        "model_id": "N",
        "params": {k: v for k, v in ccfg.items() if not k.startswith("_")},
        "_prob_cfg": pcfg,
        "device": dev,
        "n_features": int(X.shape[1]),
        "n_train": int(X.shape[0]),
        "train_issue_min": int(np.min(train_issues)) if len(train_issues) else None,
        "train_issue_max": int(np.max(train_issues)) if len(train_issues) else None,
        "seeds": per_seed,
        "failures": failures,
        "status": status,
        "hidden_layers": list(ccfg["hidden_layers"]),
        "dropout": float(ccfg["dropout"]),
    }

    if status == "complete":
        probe = np.random.default_rng(20261013).random((8, int(X.shape[1])))
        per_seed_probe = []
        for rec in per_seed:
            Xps = rec["scaler"].transform(probe)
            rec["model"].eval()
            with torch.no_grad():
                lg = rec["model"](_to_tensor(Xps, dev)).detach().cpu().numpy()
            per_seed_probe.append(1.0 / (1.0 + np.exp(-lg)))
        fitted["probe_output"] = aggregate_seed_probabilities(per_seed_probe)

    fitted["model_hash"] = model_hash(fitted)
    return fitted


def predict_mlp_candidate(fitted: Dict, X_target, target_issues,
                          history_before_target=None) -> List[Dict]:
    """逐期输出 N 的原始概率与修正后概率（历史参数对 N 不适用，保留以统一签名）。"""
    if fitted.get("status") != "complete":
        raise ValueError("该折 N 未完成（种子失败：{}），不得用于比较".format(
            fitted.get("failures")))

    X = np.asarray(X_target, dtype=float)
    target_issues = [int(i) for i in target_issues]
    pcfg = fitted.get("_prob_cfg", {})
    target_sum = float(pcfg.get("target_sum", DEFAULT_TARGET_SUM))
    clip_eps = float(pcfg.get("clip_eps", DEFAULT_CLIP_EPS))
    tie_seed = int(pcfg.get("tie_break_seed", DEFAULT_TIE_BREAK_SEED))
    dev = fitted["device"]

    per_seed = []
    for rec in fitted["seeds"]:
        Xs = rec["scaler"].transform(X)
        rec["model"].eval()
        with torch.no_grad():
            lg = rec["model"](_to_tensor(Xs, dev)).detach().cpu().numpy()
        per_seed.append(1.0 / (1.0 + np.exp(-lg)))

    raw = aggregate_seed_probabilities(per_seed)          # 先种子平均
    p = apply_sum_constraint(raw, target_sum=target_sum, clip_eps=clip_eps)

    recs: List[Dict] = []
    for k, iss in enumerate(target_issues):
        recs.append({
            "issue": iss,
            "model_id": "N",
            "p_raw": np.asarray(raw[k], dtype=float),
            "p": np.asarray(p[k], dtype=float),
            "top5": top5_from_probabilities(p[k], iss, tie_break_seed=tie_seed),
            "feature_snapshot_hash": feature_snapshot_hash(X[k]),
            "model_hash": fitted["model_hash"],
            "seed_epochs": [int(r["selected_epoch"]) for r in fitted["seeds"]],
        })
    return recs


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------

def _scaler_arrays(sc: StandardScaler) -> Dict:
    return {
        "mean": np.asarray(sc.mean_, dtype=float),
        "scale": np.asarray(sc.scale_, dtype=float),
        "var": np.asarray(sc.var_, dtype=float),
        "n_samples_seen": float(getattr(sc, "n_samples_seen_", 0.0)),
    }


def _scaler_from_arrays(d: Dict) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = np.asarray(d["mean"], dtype=float)
    sc.scale_ = np.asarray(d["scale"], dtype=float)
    sc.var_ = np.asarray(d["var"], dtype=float)
    sc.n_samples_seen_ = d["n_samples_seen"]
    sc.n_features_in_ = sc.mean_.shape[0]
    return sc


def save_mlp_candidate(fitted: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model_id": "N",
        "status": fitted["status"],
        "params": fitted["params"],
        "_prob_cfg": fitted.get("_prob_cfg"),
        "model_hash": fitted["model_hash"],
        "hidden_layers": fitted["hidden_layers"],
        "dropout": fitted["dropout"],
        "n_features": fitted["n_features"],
        "failures": fitted["failures"],
        "seeds": [],
        "scaler_es": [],
        "scaler_refit": [],
        "selected_epochs": [],
    }
    for rec in fitted["seeds"]:
        payload["seeds"].append({k: v.detach().cpu() for k, v in
                                 rec["model"].state_dict().items()})
        payload["scaler_es"].append(_scaler_arrays(rec["early_stop_scaler"]))
        payload["scaler_refit"].append(_scaler_arrays(rec["scaler"]))
        payload["selected_epochs"].append(int(rec["selected_epoch"]))
    torch.save(payload, path)


def load_mlp_candidate(path: str, device: str = "cpu") -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    fitted = {
        "model_id": "N",
        "status": payload["status"],
        "params": payload["params"],
        "_prob_cfg": payload.get("_prob_cfg"),
        "model_hash": payload["model_hash"],
        "hidden_layers": payload["hidden_layers"],
        "dropout": payload["dropout"],
        "n_features": payload["n_features"],
        "failures": payload["failures"],
        "device": device,
        "seeds": [],
    }
    for i, sd in enumerate(payload["seeds"]):
        model = build_mlp(payload["n_features"], tuple(payload["hidden_layers"]),
                          payload["dropout"]).to(device)
        model.load_state_dict(sd)
        model.eval()
        fitted["seeds"].append({
            "seed": int(payload["params"].get("seeds", DEFAULT_SEEDS)[i]),
            "selected_epoch": int(payload["selected_epochs"][i]),
            "model": model,
            "early_stop_scaler": _scaler_from_arrays(payload["scaler_es"][i]),
            "scaler": _scaler_from_arrays(payload["scaler_refit"][i]),
        })
    return fitted

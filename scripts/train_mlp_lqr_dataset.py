#!/usr/bin/env python3
"""
Train an MLP on LQR-generated MuJoCo self-balancing robot datasets.

Typical use:
    python scripts/train_mlp_lqr_dataset.py \
        --csv datasets/test_lqr_cad_step.csv \
        --target-mode residual \
        --history 5 \
        --epochs 80 \
        --batch-size 4096 \
        --out models/mlp_lqr_residual.pt

Design intent:
    - Split by episode, not by individual row, to avoid train/test leakage.
    - Remove early-fallen episodes by default.
    - Train either a residual/error compensator or direct state estimator.
    - Save model, normalization statistics, feature names, target names, and metrics.

Dependencies:
    pip install pandas numpy scikit-learn torch matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURE_CANDIDATES = [
    # command / controller information
    "speed_setpoint_m_s",
    "yaw_setpoint_rad_s",
    "ctrl_l_rad_s",
    "ctrl_r_rad_s",
    "lqr_motor_vel_cmd_rad_s",
    "lqr_motor_vel_raw_rad_s",
    "lqr_saturated",
    # noisy pseudo-sensors
    "gyro_x_meas_rad_s",
    "gyro_y_meas_rad_s",
    "gyro_z_meas_rad_s",
    "acc_x_meas_m_s2",
    "acc_y_meas_m_s2",
    "acc_z_meas_m_s2",
    "enc_l_pos_meas_rad",
    "enc_r_pos_meas_rad",
    "enc_l_vel_meas_rad_s",
    "enc_r_vel_meas_rad_s",
    "enc_forward_vel_meas_m_s",
    "vision_x_meas_m",
    "vision_y_meas_m",
    "vision_z_meas_m",
    "vision_yaw_meas_rad",
]

RESIDUAL_TARGET_CANDIDATES = [
    "target_encoder_forward_vel_error_m_s",
    "target_gyro_y_error_rad_s",
    "target_gyro_x_error_rad_s",  # fallback for older collectors
    "target_vision_x_error_m",
    "target_vision_y_error_m",
    "target_vision_z_error_m",
    "target_vision_yaw_error_rad",
]

STATE_TARGET_CANDIDATES = [
    "pitch_rad",
    "wy_rad_s",
    "encoder_forward_vel_true_m_s",
    "x_m",
    "y_m",
    "z_m",
    "yaw_rad",
]


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("--hidden must contain at least one integer, e.g. 128,128")
    return vals


def available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def drop_bad_episodes(df: pd.DataFrame, min_episode_rows_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
    if "episode" not in df.columns:
        return df, []
    counts = df.groupby("episode").size()
    if counts.empty:
        return df, []
    max_rows = int(counts.max())
    threshold = int(math.floor(max_rows * min_episode_rows_ratio))
    keep_eps = counts[counts >= threshold].index
    dropped_eps = sorted(set(counts.index.tolist()) - set(keep_eps.tolist()))
    filtered = df[df["episode"].isin(keep_eps)].copy()
    return filtered, [int(e) for e in dropped_eps]


def make_history_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    history: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stack current and previous samples per episode.

    history=1 means current sample only.
    history=5 means [t, t-1, t-2, t-3, t-4].
    """
    if history < 1:
        raise ValueError("history must be >= 1")
    if "episode" not in df.columns:
        df = df.copy()
        df["episode"] = 0

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    ep_parts: List[np.ndarray] = []

    sorted_cols = ["episode"]
    if "control_step" in df.columns:
        sorted_cols.append("control_step")
    elif "time_s" in df.columns:
        sorted_cols.append("time_s")
    df = df.sort_values(sorted_cols).reset_index(drop=True)

    for ep, g in df.groupby("episode", sort=False):
        f = g[list(feature_cols)].to_numpy(dtype=np.float32)
        y = g[list(target_cols)].to_numpy(dtype=np.float32)
        n = len(g)
        if n < history:
            continue
        rows = []
        ys = []
        eps = []
        for i in range(history - 1, n):
            # Current first, then past samples.
            window = [f[i - lag] for lag in range(history)]
            rows.append(np.concatenate(window, axis=0))
            ys.append(y[i])
            eps.append(ep)
        X_parts.append(np.vstack(rows))
        y_parts.append(np.vstack(ys))
        ep_parts.append(np.asarray(eps))

    if not X_parts:
        raise ValueError("No samples left after history stacking. Lower --history or check dataset.")

    X = np.vstack(X_parts).astype(np.float32)
    y = np.vstack(y_parts).astype(np.float32)
    eps = np.concatenate(ep_parts)

    feature_names = []
    for lag in range(history):
        suffix = "t" if lag == 0 else f"t_minus_{lag}"
        for c in feature_cols:
            feature_names.append(f"{c}__{suffix}")

    return X, y, eps, feature_names


def split_by_episode(
    episodes: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_eps = np.unique(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)

    n = len(unique_eps)
    n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
    if n_test + n_val >= n:
        n_test = max(0, min(n_test, n - 2))
        n_val = max(0, min(n_val, n - n_test - 1))

    test_eps = set(unique_eps[:n_test])
    val_eps = set(unique_eps[n_test:n_test + n_val])
    train_eps = set(unique_eps[n_test + n_val:])

    train_idx = np.array([ep in train_eps for ep in episodes])
    val_idx = np.array([ep in val_eps for ep in episodes])
    test_idx = np.array([ep in test_eps for ep in episodes])
    return train_idx, val_idx, test_idx


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    err = y_pred - y_true
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    return rmse, mae


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        bs = xb.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        bs = xb.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def predict_numpy(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
        preds.append(model(xb).cpu().numpy())
    return np.vstack(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP on LQR MuJoCo dataset CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="Dataset CSV path.")
    parser.add_argument("--out", type=Path, default=Path("models/mlp_lqr.pt"), help="Output .pt model path.")
    parser.add_argument("--target-mode", choices=["residual", "state"], default="residual")
    parser.add_argument("--features", type=str, default="", help="Comma-separated feature columns. Default: auto.")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated target columns. Default: auto.")
    parser.add_argument("--history", type=int, default=5, help="Number of past control samples to stack.")
    parser.add_argument("--min-episode-rows-ratio", type=float, default=0.8, help="Drop episodes shorter than ratio*max_rows.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--hidden", type=parse_hidden, default=parse_hidden("128,128,64"))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    print(f"Loaded: {args.csv}")
    print(f"Raw rows: {len(df):,}, columns: {len(df.columns)}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f"Rows after dropna/dropinf: {len(df):,}")

    df, dropped_eps = drop_bad_episodes(df, args.min_episode_rows_ratio)
    if dropped_eps:
        print(f"Dropped early/short episodes: {dropped_eps}")
    print(f"Rows after episode filter: {len(df):,}")

    if args.features:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        feature_cols = available_columns(df, DEFAULT_FEATURE_CANDIDATES)

    if args.targets:
        target_cols = [c.strip() for c in args.targets.split(",") if c.strip()]
    else:
        candidates = RESIDUAL_TARGET_CANDIDATES if args.target_mode == "residual" else STATE_TARGET_CANDIDATES
        target_cols = available_columns(df, candidates)

    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    if not feature_cols:
        raise ValueError("No feature columns selected.")
    if not target_cols:
        raise ValueError("No target columns selected. Use --targets explicitly.")

    print("Feature columns:")
    for c in feature_cols:
        print(f"  - {c}")
    print("Target columns:")
    for c in target_cols:
        print(f"  - {c}")

    X, y, eps, feature_names = make_history_features(df, feature_cols, target_cols, args.history)
    train_idx, val_idx, test_idx = split_by_episode(eps, args.val_ratio, args.test_ratio, args.seed)

    print(f"Stacked X: {X.shape}, y: {y.shape}, history={args.history}")
    print(f"Train/val/test rows: {train_idx.sum():,}/{val_idx.sum():,}/{test_idx.sum():,}")
    print(f"Train episodes: {len(np.unique(eps[train_idx]))}, val: {len(np.unique(eps[val_idx]))}, test: {len(np.unique(eps[test_idx]))}")

    x_scaler = StandardScaler().fit(X[train_idx])
    y_scaler = StandardScaler().fit(y[train_idx])

    Xs = x_scaler.transform(X).astype(np.float32)
    ys = y_scaler.transform(y).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(Xs[train_idx]), torch.from_numpy(ys[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(Xs[val_idx]), torch.from_numpy(ys[val_idx]))
    test_X = Xs[test_idx]
    test_y_scaled = ys[test_idx]
    test_y_true = y[test_idx]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = MLPRegressor(Xs.shape[1], ys.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    patience = 15
    bad_epochs = 0
    history_log = []

    print(f"Training on {device}...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_loss = evaluate_loss(model, val_loader, device) if len(val_ds) > 0 else train_loss
        history_log.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss})

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"epoch {epoch:04d} | train_mse_scaled={train_loss:.6g} | val_mse_scaled={val_loss:.6g}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best val={best_val:.6g}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if len(test_X) > 0:
        pred_scaled = predict_numpy(model, test_X, device)
        pred = y_scaler.inverse_transform(pred_scaled)
        rmse, mae = rmse_mae(test_y_true, pred)
    else:
        rmse = np.full(len(target_cols), np.nan)
        mae = np.full(len(target_cols), np.nan)

    print("\nTest metrics in original units:")
    metrics = {}
    for i, name in enumerate(target_cols):
        metrics[name] = {"rmse": float(rmse[i]), "mae": float(mae[i])}
        print(f"  {name:40s} RMSE={rmse[i]:.6g} | MAE={mae[i]:.6g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_dim": int(Xs.shape[1]),
            "out_dim": int(ys.shape[1]),
            "hidden": list(args.hidden),
            "dropout": float(args.dropout),
        },
        "feature_cols_base": list(feature_cols),
        "feature_names": list(feature_names),
        "target_cols": list(target_cols),
        "history": int(args.history),
        "target_mode": args.target_mode,
        "x_scaler_mean": x_scaler.mean_.astype(np.float32),
        "x_scaler_scale": x_scaler.scale_.astype(np.float32),
        "y_scaler_mean": y_scaler.mean_.astype(np.float32),
        "y_scaler_scale": y_scaler.scale_.astype(np.float32),
        "metrics": metrics,
        "training_log": history_log,
        "dropped_episodes": dropped_eps,
    }
    torch.save(ckpt, args.out)
    print(f"\nSaved model: {args.out}")

    metrics_path = args.out.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "dropped_episodes": dropped_eps, "target_cols": target_cols}, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()

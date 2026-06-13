from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .neural_ladder import (
    BlockSplits,
    LadderSeries,
    lag_diagnostic,
    load_ladder_series,
    make_block_splits,
    peak_timing_diagnostic,
)


class ForecastGRULeadDelta(nn.Module):
    def __init__(self, n_past_feat: int, n_future_feat: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_past_feat, hidden, batch_first=True)
        self.future_proj = nn.Sequential(
            nn.Linear(n_future_feat * horizon, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x_past)
        h_last = h[:, -1, :]
        fut = self.future_proj(x_future.reshape(x_future.size(0), -1))
        return self.head(torch.cat([h_last, fut], dim=-1)).squeeze(-1)


def _zscore_train(arr: np.ndarray, split: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = arr[split]
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1.0e-9
    return (arr - mu) / sd, mu, sd


def _compose_delta_past_features(
    climate_window: np.ndarray,
    head_window: np.ndarray,
    include_dhead: bool,
) -> np.ndarray:
    if include_dhead:
        dhead = np.diff(head_window, prepend=head_window[0]).astype(np.float32)
        return np.concatenate([climate_window, head_window[:, None], dhead[:, None]], axis=1)
    return np.concatenate([climate_window, head_window[:, None]], axis=1)


def build_direct_delta_split(
    series: LadderSeries,
    splits: BlockSplits,
    window: int,
    horizon: int,
    include_dhead: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    head = series.head_interp.astype(np.float32)
    features = _compose_delta_past_features(series.climate, head, include_dhead=include_dhead).astype(np.float32)
    feat_norm, feat_mu, feat_sd = _zscore_train(features, splits.train)

    n_clim = series.climate.shape[1]
    climate_mu = feat_mu[:n_clim].astype(np.float32)
    climate_sd = feat_sd[:n_clim].astype(np.float32)

    def train_delta_stats(st: slice) -> tuple[float, float]:
        starts = np.arange(st.start, st.stop - window - horizon + 1)
        h_anchor = np.array([head[i + window - 1] for i in starts], dtype=np.float32)
        y_head = np.array([head[i + window + horizon - 1] for i in starts], dtype=np.float32)
        delta = y_head - h_anchor
        return float(delta.mean()), float(delta.std() + 1.0e-9)

    delta_mu, delta_sd = train_delta_stats(splits.train)

    def build(split: slice) -> dict[str, np.ndarray]:
        starts = np.arange(split.start, split.stop - window - horizon + 1)
        if len(starts) == 0:
            raise ValueError(f"Split too short for window={window}, horizon={horizon}: {split}")

        x_past = np.stack([feat_norm[i : i + window] for i in starts]).astype(np.float32)
        x_future = np.stack(
            [
                ((series.climate[i + window : i + window + horizon] - climate_mu) / climate_sd).astype(np.float32)
                for i in starts
            ]
        ).astype(np.float32)
        h_anchor = np.array([head[i + window - 1] for i in starts], dtype=np.float32)
        y_head = np.array([head[i + window + horizon - 1] for i in starts], dtype=np.float32)
        y_delta = y_head - h_anchor
        y_delta_norm = ((y_delta - delta_mu) / delta_sd).astype(np.float32)
        rain_future = np.stack([series.rain_mm[i + window : i + window + horizon] for i in starts]).astype(np.float32)
        rain_sum = rain_future.sum(axis=1).astype(np.float32)
        target_dates = np.array([series.dates[i + window + horizon - 1] for i in starts]).astype("datetime64[D]")
        event_weight = 1.0 + np.abs(y_delta) / (np.std(y_delta) + 1.0e-9)
        return {
            "x_past": x_past,
            "x_future": x_future,
            "h_anchor": h_anchor,
            "y_head": y_head,
            "y_delta": y_delta.astype(np.float32),
            "y_delta_norm": y_delta_norm,
            "rain_sum": rain_sum,
            "target_dates": target_dates,
            "event_weight": event_weight.astype(np.float32),
        }

    return {
        "train": build(splits.train),
        "val": build(splits.val),
        "test": build(splits.test),
        "norm": {
            "feat_mu": feat_mu.astype(np.float32),
            "feat_sd": feat_sd.astype(np.float32),
            "climate_mu": climate_mu,
            "climate_sd": climate_sd,
            "delta_mu": delta_mu,
            "delta_sd": delta_sd,
        },
    }


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (weight * (pred - target) ** 2).mean()


def _direct_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    resid = pred.astype(float) - obs.astype(float)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "nse": float(1.0 - ss_res / (ss_tot + 1.0e-12)),
        "corr": float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 1.0e-9 and obs.std() > 1.0e-9 else float("nan"),
        "n_eval": int(pred.size),
    }


def _ws1_penalty_seq(y_head: torch.Tensor) -> torch.Tensor:
    return ((y_head[1:] - y_head[:-1]) ** 2).mean()


def _ws2_penalty_seq(y_head: torch.Tensor) -> torch.Tensor:
    if y_head.numel() < 3:
        return torch.zeros((), dtype=y_head.dtype)
    second = y_head[2:] - 2.0 * y_head[1:-1] + y_head[:-2]
    return (second**2).mean()


def _ode_penalty_seq(
    y_head: torch.Tensor,
    rain_sum: torch.Tensor,
    gamma_r: torch.Tensor,
    gamma_d: torch.Tensor,
    h_ref: torch.Tensor,
) -> torch.Tensor:
    if y_head.numel() < 2:
        return torch.zeros((), dtype=y_head.dtype)
    dh = y_head[1:] - y_head[:-1]
    rhs = gamma_r * rain_sum[1:] - gamma_d * (y_head[:-1] - h_ref)
    return ((dh - rhs) ** 2).mean()


def train_direct_delta_variant(
    split_data: dict[str, dict[str, np.ndarray]],
    variant: str,
    seed: int = 42,
    epochs: int = 150,
    patience: int = 25,
    hidden: int = 64,
    lr: float = 1.0e-3,
    lambda_penalty: float = 1.0,
    event_weight_scale: float = 1.0,
) -> tuple[nn.Module, dict[str, dict], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]
    norm = split_data["norm"]

    model = ForecastGRULeadDelta(
        n_past_feat=train["x_past"].shape[-1],
        n_future_feat=train["x_future"].shape[-1],
        horizon=train["x_future"].shape[1],
        hidden=hidden,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1.0e-5)

    raw_gamma_r = raw_gamma_d = raw_h_ref = None
    opt_phys = None
    if variant == "ode":
        raw_gamma_r = torch.tensor(-4.0, requires_grad=True)
        raw_gamma_d = torch.tensor(-1.0, requires_grad=True)
        raw_h_ref = torch.tensor(float(train["h_anchor"].mean()), requires_grad=True)
        opt_phys = torch.optim.Adam([raw_gamma_r, raw_gamma_d, raw_h_ref], lr=5.0e-2)

    tt = lambda x: torch.from_numpy(x)
    xpt = tt(train["x_past"])
    xft = tt(train["x_future"])
    ydt = tt(train["y_delta_norm"])
    yhp = tt(train["y_head"])
    hat = tt(train["h_anchor"])
    rsum_t = tt(train["rain_sum"])
    wgt = 1.0 + event_weight_scale * (tt(train["event_weight"]) - 1.0)

    xpv = tt(val["x_past"])
    xfv = tt(val["x_future"])
    ydv = tt(val["y_delta_norm"])
    yhv = tt(val["y_head"])
    hav = tt(val["h_anchor"])

    best_val = float("inf")
    best_state = None
    best_phys = None
    no_imp = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        if opt_phys is not None:
            opt_phys.zero_grad()

        pred_delta_norm = model(xpt, xft)
        pred_delta_phys = pred_delta_norm * norm["delta_sd"] + norm["delta_mu"]
        pred_head = hat + pred_delta_phys

        loss = _weighted_mse(pred_delta_norm, ydt, wgt)
        loss = loss + 0.25 * torch.mean((pred_head - yhp) ** 2)
        if variant == "ws1":
            loss = loss + lambda_penalty * _ws1_penalty_seq(pred_head)
        elif variant == "ws2":
            loss = loss + lambda_penalty * _ws2_penalty_seq(pred_head)
        elif variant == "ode":
            gamma_r = 1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)
            gamma_d = 1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)
            loss = loss + lambda_penalty * _ode_penalty_seq(pred_head, rsum_t, gamma_r, gamma_d, raw_h_ref)

        loss.backward()
        opt.step()
        if opt_phys is not None:
            opt_phys.step()

        model.eval()
        with torch.no_grad():
            pred_val_norm = model(xpv, xfv)
            pred_val_phys = pred_val_norm * norm["delta_sd"] + norm["delta_mu"]
            pred_val_head = hav + pred_val_phys
            val_loss = torch.mean((pred_val_phys - (yhv - hav)) ** 2).item()
        sched.step(val_loss)

        if val_loss < best_val - 1.0e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if variant == "ode":
                best_phys = {
                    "gamma_r": float((1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)).item()),
                    "gamma_d": float((1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)).item()),
                    "h_ref": float(raw_h_ref.item()),
                }
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state is None:
        raise RuntimeError("Direct-delta lead training failed to produce a best state.")
    model.load_state_dict(best_state)
    model.eval()

    outputs: dict[str, dict] = {}
    for name, block in [("train", train), ("val", val), ("test", test)]:
        with torch.no_grad():
            pred_delta_norm = model(tt(block["x_past"]), tt(block["x_future"]))
        pred_delta_phys = (pred_delta_norm * norm["delta_sd"] + norm["delta_mu"]).cpu().numpy()
        pred_head = block["h_anchor"] + pred_delta_phys
        metrics = _direct_metrics(pred_head, block["y_head"])
        metrics.update(lag_diagnostic(pred_head, block["y_head"], max_lag=max(14, block["x_future"].shape[1])))
        metrics.update(peak_timing_diagnostic(pred_head, block["y_head"]))
        outputs[name] = {
            "pred_head": pred_head.astype(np.float32),
            "obs_head": block["y_head"].astype(np.float32),
            "pred_delta": pred_delta_phys.astype(np.float32),
            "obs_delta": block["y_delta"].astype(np.float32),
            "target_dates": block["target_dates"].astype("datetime64[D]").astype(str),
            "metrics": metrics,
        }

    meta = {
        "variant": f"direct_delta_{variant}",
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "hidden": hidden,
        "lr": lr,
        "lambda_penalty": lambda_penalty,
        "event_weight_scale": event_weight_scale,
        "include_dhead": bool(train["x_past"].shape[-1] > train["x_future"].shape[-1] + 1),
        "best_val_loss": best_val,
        "target_mode": "one_step_delta" if int(train["x_future"].shape[1]) == 1 else "direct_scalar_delta",
    }
    if best_phys is not None:
        meta["physics_params"] = best_phys
    return model, outputs, meta


def save_direct_delta_run(output_dir: Path, series: LadderSeries, outputs: dict, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stem": series.stem,
        **meta,
        "train": outputs["train"]["metrics"],
        "val": outputs["val"]["metrics"],
        "test": outputs["test"]["metrics"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for split_name, block in outputs.items():
        np.savez(
            output_dir / f"{split_name}_predictions.npz",
            pred_head=np.asarray(block["pred_head"]),
            obs_head=np.asarray(block["obs_head"]),
            pred_delta=np.asarray(block["pred_delta"]),
            obs_delta=np.asarray(block["obs_delta"]),
            target_dates=np.asarray(block["target_dates"]),
        )

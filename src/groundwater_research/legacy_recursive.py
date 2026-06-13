from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .neural_ladder import BlockSplits, LadderSeries, lag_diagnostic, peak_timing_diagnostic


class ForecastLegacyGRU(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x_seq)
        return self.head(h[:, -1, :])


def build_legacy_recursive_split(
    series: LadderSeries,
    splits: BlockSplits,
    window: int,
) -> dict[str, dict[str, np.ndarray] | dict[str, np.ndarray | float]]:
    head = series.head_interp.astype(np.float32)
    climate = series.climate.astype(np.float32)

    train_head = head[splits.train]
    head_center = float(train_head.mean())
    climate_mu = climate[splits.train].mean(axis=0).astype(np.float32)
    climate_sd = (climate[splits.train].std(axis=0) + 1.0e-9).astype(np.float32)

    head_centered = head - head_center
    climate_norm = ((climate - climate_mu) / climate_sd).astype(np.float32)

    def build(split: slice) -> dict[str, np.ndarray]:
        start0 = max(0, split.start - window)
        starts = np.arange(start0, split.stop - window, dtype=int)
        target_days = starts + window
        keep = (target_days >= split.start) & (target_days < split.stop)
        starts = starts[keep]
        target_days = target_days[keep]
        if starts.size == 0:
            raise ValueError(f"Split too short for window={window}: {split}")

        x_seq = np.stack(
            [
                np.concatenate(
                    [
                        climate_norm[i + 1 : i + window + 1],
                        head_centered[i : i + window, None],
                    ],
                    axis=1,
                ).astype(np.float32)
                for i in starts
            ]
        ).astype(np.float32)
        y_head = head[target_days].astype(np.float32)[:, None]
        y_centered = head_centered[target_days].astype(np.float32)[:, None]
        rain_target = series.rain_mm[target_days].astype(np.float32)
        target_dates = series.dates[target_days].astype("datetime64[D]")
        return {
            "x_seq": x_seq,
            "y_head": y_head,
            "y_centered": y_centered,
            "rain_target": rain_target,
            "target_dates": target_dates,
            "target_day_index": target_days.astype(np.int32),
        }

    return {
        "train": build(splits.train),
        "val": build(splits.val),
        "test": build(splits.test),
        "norm": {
            "head_center": head_center,
            "climate_mu": climate_mu,
            "climate_sd": climate_sd,
        },
    }


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def _ws1_penalty(pred_centered: torch.Tensor) -> torch.Tensor:
    if pred_centered.numel() < 2:
        return torch.zeros((), dtype=pred_centered.dtype)
    return torch.mean((pred_centered[1:] - pred_centered[:-1]) ** 2)


def _ws2_penalty(pred_centered: torch.Tensor) -> torch.Tensor:
    if pred_centered.numel() < 3:
        return torch.zeros((), dtype=pred_centered.dtype)
    second = pred_centered[2:] - 2.0 * pred_centered[1:-1] + pred_centered[:-2]
    return torch.mean(second**2)


def _ode_penalty(
    pred_head: torch.Tensor,
    rain_target: torch.Tensor,
    gamma_r: torch.Tensor,
    gamma_d: torch.Tensor,
    h_ref: torch.Tensor,
) -> torch.Tensor:
    if pred_head.numel() < 2:
        return torch.zeros((), dtype=pred_head.dtype)
    dh = pred_head[1:] - pred_head[:-1]
    rhs = gamma_r * rain_target[1:] * 1.0e-3 - gamma_d * (pred_head[:-1] - h_ref)
    return torch.mean((dh - rhs) ** 2)


def train_legacy_recursive_variant(
    split_data: dict[str, dict[str, np.ndarray]],
    variant: str,
    seed: int = 42,
    epochs: int = 150,
    patience: int = 25,
    hidden: int = 64,
    lr: float = 1.0e-3,
    lambda_penalty: float = 1.0,
) -> tuple[nn.Module, dict[str, dict], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]
    norm = split_data["norm"]

    model = ForecastLegacyGRU(n_feat=train["x_seq"].shape[-1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1.0e-5)

    raw_gamma_r = raw_gamma_d = raw_h_ref = None
    opt_phys = None
    if variant == "ode":
        raw_gamma_r = torch.tensor(-4.0, requires_grad=True)
        raw_gamma_d = torch.tensor(-1.0, requires_grad=True)
        raw_h_ref = torch.tensor(float(train["y_head"].mean()), requires_grad=True)
        opt_phys = torch.optim.Adam([raw_gamma_r, raw_gamma_d, raw_h_ref], lr=5.0e-2)

    tt = lambda x: torch.from_numpy(x)
    xtr = tt(train["x_seq"])
    ytr = tt(train["y_centered"])
    ytr_abs = tt(train["y_head"])
    rtr = tt(train["rain_target"])
    xva = tt(val["x_seq"])
    yva = tt(val["y_centered"])

    best_val = float("inf")
    best_state = None
    best_phys = None
    no_imp = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        if opt_phys is not None:
            opt_phys.zero_grad()

        pred_centered = model(xtr)
        loss = _mse(pred_centered, ytr)
        if variant == "ws1":
            loss = loss + lambda_penalty * _ws1_penalty(pred_centered[:, 0])
        elif variant == "ws2":
            loss = loss + lambda_penalty * _ws2_penalty(pred_centered[:, 0])
        elif variant == "ode":
            gamma_r = 1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)
            gamma_d = 1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)
            pred_head = pred_centered[:, 0] + float(norm["head_center"])
            loss = loss + lambda_penalty * _ode_penalty(pred_head, rtr, gamma_r, gamma_d, raw_h_ref)

        loss.backward()
        opt.step()
        if opt_phys is not None:
            opt_phys.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(xva)
            val_loss = _mse(pred_val, yva).item()
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
        raise RuntimeError("Legacy recursive training failed to produce a best state.")
    model.load_state_dict(best_state)
    model.eval()

    outputs: dict[str, dict] = {}
    for name, block in [("train", train), ("val", val), ("test", test)]:
        with torch.no_grad():
            pred_centered = model(tt(block["x_seq"])).cpu().numpy()[:, 0]
        pred_head = pred_centered + float(norm["head_center"])
        obs_head = block["y_head"][:, 0].astype(np.float32)
        outputs[name] = {
            "pred_head": pred_head.astype(np.float32),
            "obs_head": obs_head,
            "target_dates": block["target_dates"].astype("datetime64[D]").astype(str),
        }

    meta = {
        "variant": f"legacy_{variant}",
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "hidden": hidden,
        "lr": lr,
        "lambda_penalty": lambda_penalty,
        "best_val_loss": best_val,
        "target_mode": "one_step_recursive_legacy",
    }
    if best_phys is not None:
        meta["physics_params"] = best_phys
    return model, outputs, meta


def recursive_rollout_legacy_head(
    model,
    series: LadderSeries,
    split: slice,
    norm: dict[str, np.ndarray | float],
    window: int,
    forecast_horizon: int,
) -> dict[str, np.ndarray | dict]:
    if split.start < window:
        raise ValueError(f"Recursive legacy rollout requires split.start >= window, got {split.start} < {window}")

    climate = series.climate.astype(np.float32)
    head = series.head_interp.astype(np.float32)
    climate_mu = np.asarray(norm["climate_mu"], dtype=np.float32)
    climate_sd = np.asarray(norm["climate_sd"], dtype=np.float32)
    head_center = float(norm["head_center"])

    preds: list[float] = []
    obs: list[float] = []
    dates: list[np.datetime64] = []

    model.eval()
    for start in range(split.start, split.stop - forecast_horizon + 1):
        hist = head[start - window : start].copy()
        for step in range(forecast_horizon):
            current = start + step
            climate_seq = climate[current - window + 1 : current + 1]
            x_seq = np.concatenate(
                [
                    ((climate_seq - climate_mu) / climate_sd).astype(np.float32),
                    (hist - head_center)[:, None].astype(np.float32),
                ],
                axis=1,
            )[None, :, :]
            with torch.no_grad():
                pred_centered = model(torch.from_numpy(x_seq)).cpu().numpy()[0, 0]
            yhat = pred_centered + head_center
            hist = np.concatenate([hist[1:], np.array([yhat], dtype=np.float32)])
        preds.append(float(hist[-1]))
        obs.append(float(head[start + forecast_horizon - 1]))
        dates.append(series.dates[start + forecast_horizon - 1].astype("datetime64[D]"))

    pred = np.asarray(preds, dtype=np.float32)
    obs_arr = np.asarray(obs, dtype=np.float32)
    resid = pred - obs_arr
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs_arr - obs_arr.mean()) ** 2))
    metrics = {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "nse": float(1.0 - ss_res / (ss_tot + 1.0e-12)),
        "corr": float(np.corrcoef(pred, obs_arr)[0, 1]) if pred.std() > 1.0e-9 and obs_arr.std() > 1.0e-9 else float("nan"),
        "n_eval": int(pred.size),
    }
    metrics.update(lag_diagnostic(pred, obs_arr, max_lag=max(14, forecast_horizon)))
    metrics.update(peak_timing_diagnostic(pred, obs_arr))
    return {
        "pred": pred,
        "obs": obs_arr,
        "dates": np.asarray(dates).astype("datetime64[D]").astype(str),
        "metrics": metrics,
    }

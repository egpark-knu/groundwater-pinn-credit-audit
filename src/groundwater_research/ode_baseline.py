from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .neural_ladder import lag_diagnostic, metrics_from_rollout, metrics_from_sequence, peak_timing_diagnostic


@dataclass
class ODEParams:
    gamma_r: float
    gamma_d: float
    h_ref: float
    tau_days: float


def _h_ref_bounds_from_train(h_start: np.ndarray, y_seq_phys: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(h_start, dtype=float).ravel(), np.asarray(y_seq_phys, dtype=float).ravel()])
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else 0.0
        lo = center - 1.0
        hi = center + 1.0
    return lo, hi


def _init_raw_h_ref(mean_h: float, h_ref_bounds: tuple[float, float] | None) -> torch.Tensor:
    if h_ref_bounds is None:
        return torch.tensor(float(mean_h), requires_grad=True)
    lo, hi = h_ref_bounds
    scaled = (float(mean_h) - lo) / max(hi - lo, 1.0e-6)
    scaled = float(np.clip(scaled, 1.0e-4, 1.0 - 1.0e-4))
    return torch.tensor(float(np.log(scaled / (1.0 - scaled))), requires_grad=True)


def _bounded_h_ref(raw_h_ref: torch.Tensor, h_ref_bounds: tuple[float, float] | None) -> torch.Tensor:
    if h_ref_bounds is None:
        return raw_h_ref
    lo, hi = h_ref_bounds
    return float(lo) + float(hi - lo) * torch.sigmoid(raw_h_ref)


def simulate_effective_ode(
    h_start: np.ndarray,
    rain_future: np.ndarray,
    gamma_r: float,
    gamma_d: float,
    h_ref: float,
    tau_days: float,
) -> np.ndarray:
    h_start = np.asarray(h_start, dtype=np.float32)
    rain_future = np.asarray(rain_future, dtype=np.float32)
    alpha = float(np.exp(-1.0 / max(float(tau_days), 1.0)))

    prev_p = rain_future[:, 0]
    p_eff = [prev_p]
    for idx in range(1, rain_future.shape[1]):
        prev_p = alpha * prev_p + (1.0 - alpha) * rain_future[:, idx]
        p_eff.append(prev_p)
    p_eff = np.stack(p_eff, axis=1) * 1.0e-3

    preds = []
    prev_h = h_start.astype(np.float32)
    for idx in range(rain_future.shape[1]):
        dh = gamma_r * p_eff[:, idx] - gamma_d * (prev_h - h_ref)
        next_h = prev_h + dh
        preds.append(next_h)
        prev_h = next_h
    return np.stack(preds, axis=1)


def _simulate_effective_ode_torch(
    h_start: torch.Tensor,
    rain_future: torch.Tensor,
    gamma_r: torch.Tensor,
    gamma_d: torch.Tensor,
    h_ref: torch.Tensor,
    tau_days: float,
) -> torch.Tensor:
    alpha = float(np.exp(-1.0 / max(float(tau_days), 1.0)))
    prev_p = rain_future[:, 0]
    p_eff = [prev_p]
    for idx in range(1, rain_future.size(1)):
        prev_p = alpha * prev_p + (1.0 - alpha) * rain_future[:, idx]
        p_eff.append(prev_p)
    p_eff = torch.stack(p_eff, dim=1) * 1.0e-3

    preds = []
    prev_h = h_start
    for idx in range(rain_future.size(1)):
        dh = gamma_r * p_eff[:, idx] - gamma_d * (prev_h - h_ref)
        next_h = prev_h + dh
        preds.append(next_h)
        prev_h = next_h
    return torch.stack(preds, dim=1)


def fit_standalone_ode(
    split_data: dict[str, dict[str, np.ndarray]],
    seed: int = 42,
    epochs: int = 400,
    patience: int = 40,
    lr: float = 5.0e-2,
    tau_candidates: tuple[float, ...] = (3.0, 7.0, 14.0, 30.0),
    bound_h_ref: bool = False,
) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]

    h_train = torch.from_numpy(train["h_start"])
    r_train = torch.from_numpy(train["rain_future"])
    y_train = torch.from_numpy(train["y_seq_phys"])
    h_val = torch.from_numpy(val["h_start"])
    r_val = torch.from_numpy(val["rain_future"])
    y_val = torch.from_numpy(val["y_seq_phys"])
    h_ref_bounds = _h_ref_bounds_from_train(train["h_start"], train["y_seq_phys"]) if bound_h_ref else None

    best = {
        "val_loss": float("inf"),
        "params": None,
    }

    for tau_days in tau_candidates:
        raw_gamma_r = torch.tensor(-4.0, requires_grad=True)
        raw_gamma_d = torch.tensor(-1.0, requires_grad=True)
        raw_h_ref = _init_raw_h_ref(float(train["h_start"].mean()), h_ref_bounds)
        opt = torch.optim.Adam([raw_gamma_r, raw_gamma_d, raw_h_ref], lr=lr)

        best_local_val = float("inf")
        best_local = None
        no_imp = 0
        for _ in range(epochs):
            opt.zero_grad()
            gamma_r = 1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)
            gamma_d = 1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)
            h_ref = _bounded_h_ref(raw_h_ref, h_ref_bounds)
            pred = _simulate_effective_ode_torch(
                h_start=h_train,
                rain_future=r_train,
                gamma_r=gamma_r,
                gamma_d=gamma_d,
                h_ref=h_ref,
                tau_days=tau_days,
            )
            loss = torch.mean((pred - y_train) ** 2)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred_val = _simulate_effective_ode_torch(
                    h_start=h_val,
                    rain_future=r_val,
                    gamma_r=1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r),
                    gamma_d=1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d),
                    h_ref=_bounded_h_ref(raw_h_ref, h_ref_bounds),
                    tau_days=tau_days,
                )
                val_loss = torch.mean((pred_val - y_val) ** 2).item()

            if val_loss < best_local_val - 1.0e-8:
                best_local_val = val_loss
                best_local = {
                    "gamma_r": float((1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)).item()),
                    "gamma_d": float((1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)).item()),
                    "h_ref": float(_bounded_h_ref(raw_h_ref, h_ref_bounds).item()),
                    "tau_days": float(tau_days),
                }
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                break

        if best_local is not None and best_local_val < best["val_loss"]:
            best = {
                "val_loss": best_local_val,
                "params": best_local,
            }

    if best["params"] is None:
        raise RuntimeError("Standalone ODE fitting failed to find valid parameters.")

    params = ODEParams(**best["params"])
    outputs = {}
    for name, block in [("train", train), ("val", val), ("test", test)]:
        pred = simulate_effective_ode(
            h_start=block["h_start"],
            rain_future=block["rain_future"],
            gamma_r=params.gamma_r,
            gamma_d=params.gamma_d,
            h_ref=params.h_ref,
            tau_days=params.tau_days,
        )
        outputs[name] = {
            "pred_seq_phys": pred,
            "obs_seq_phys": block["y_seq_phys"],
            "target_dates": block["target_dates"].astype("datetime64[D]").astype(str),
            "forecast_dates": block["forecast_dates"].astype("datetime64[D]").astype(str),
            "metrics": metrics_from_sequence(pred, block["y_seq_phys"]),
        }

    meta = {
        "variant": "ode_only_bounded_h_ref" if bound_h_ref else "ode_only",
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "lr": lr,
        "best_val_loss": best["val_loss"],
        "h_ref_bound_mode": "train_observed_range" if bound_h_ref else "unbounded",
        "h_ref_bounds": list(h_ref_bounds) if h_ref_bounds is not None else None,
        "physics_params": {
            "gamma_r": params.gamma_r,
            "gamma_d": params.gamma_d,
            "h_ref": params.h_ref,
            "tau_days": params.tau_days,
        },
    }
    return outputs, meta


def rollout_standalone_ode(
    series,
    split: slice,
    horizon: int,
    params: ODEParams,
) -> dict[str, np.ndarray | dict]:
    if split.start < 1:
        raise ValueError("Standalone ODE rollout requires at least one prior day.")

    pred_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = []
    date_all: list[np.ndarray] = []
    t = split.start
    while t < split.stop:
        block_len = min(horizon, split.stop - t)
        prev_h = float(series.head_interp[t - 1])
        rain_block = series.rain_mm[t : t + block_len].astype(np.float32)[None, :]
        pred_block = simulate_effective_ode(
            h_start=np.array([prev_h], dtype=np.float32),
            rain_future=rain_block,
            gamma_r=params.gamma_r,
            gamma_d=params.gamma_d,
            h_ref=params.h_ref,
            tau_days=params.tau_days,
        )[0, :block_len]
        pred_all.append(pred_block.astype(np.float32))
        obs_all.append(series.head_interp[t : t + block_len].astype(np.float32))
        date_all.append(series.dates[t : t + block_len].astype("datetime64[D]"))
        t += horizon

    pred = np.concatenate(pred_all)
    obs = np.concatenate(obs_all)
    dates = np.concatenate(date_all)
    metrics = metrics_from_rollout(pred, obs)
    metrics.update(lag_diagnostic(pred, obs, max_lag=max(horizon, 14)))
    metrics.update(peak_timing_diagnostic(pred, obs))
    return {
        "pred": pred,
        "obs": obs,
        "dates": dates.astype("datetime64[D]").astype(str),
        "metrics": metrics,
    }

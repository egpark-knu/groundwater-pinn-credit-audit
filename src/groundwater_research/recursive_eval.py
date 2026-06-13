from __future__ import annotations

import numpy as np
import torch

from .neural_ladder import lag_diagnostic, peak_timing_diagnostic


def _compose_delta_past_features(
    climate_window: np.ndarray,
    head_window: np.ndarray,
    include_dhead: bool,
) -> np.ndarray:
    if include_dhead:
        dhead = np.diff(head_window, prepend=head_window[0]).astype(np.float32)
        return np.concatenate([climate_window, head_window[:, None], dhead[:, None]], axis=1)
    return np.concatenate([climate_window, head_window[:, None]], axis=1)


def recursive_rollout_one_step_head(
    model,
    series,
    split: slice,
    norm: dict[str, np.ndarray | float],
    window: int,
    forecast_horizon: int,
) -> dict[str, np.ndarray | dict]:
    step_mu = np.asarray(norm["step_mu"], dtype=np.float32)
    step_sd = np.asarray(norm["step_sd"], dtype=np.float32)
    head_mu = float(norm["head_mu"])
    head_sd = float(norm["head_sd"])

    head = series.head_interp.astype(np.float32)
    climate = series.climate.astype(np.float32)

    preds: list[float] = []
    obs: list[float] = []
    dates: list[np.datetime64] = []

    model.eval()
    for start in range(split.start, split.stop - forecast_horizon + 1):
        if start < window:
            raise ValueError(f"Split start {start} is shorter than window={window}")
        hist = head[start - window : start].copy()
        for step in range(forecast_horizon):
            current = start + step
            step_window = np.concatenate(
                [climate[current - window + 1 : current + 1], hist[:, None]],
                axis=1,
            ).astype(np.float32)
            x_step = ((step_window - step_mu) / step_sd)[None, :, :].astype(np.float32)
            with torch.no_grad():
                pred_norm = model(torch.from_numpy(x_step)).cpu().numpy()[0]
            yhat = pred_norm * head_sd + head_mu
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


def recursive_block_rollout_one_step_head(
    model,
    series,
    split: slice,
    norm: dict[str, np.ndarray | float],
    window: int,
    forecast_horizon: int,
) -> dict[str, np.ndarray | dict]:
    step_mu = np.asarray(norm["step_mu"], dtype=np.float32)
    step_sd = np.asarray(norm["step_sd"], dtype=np.float32)
    head_mu = float(norm["head_mu"])
    head_sd = float(norm["head_sd"])

    head = series.head_interp.astype(np.float32)
    climate = series.climate.astype(np.float32)

    pred_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = []
    date_all: list[np.ndarray] = []

    model.eval()
    t = split.start
    if t < window:
        raise ValueError(f"Split start {t} is shorter than window={window}")
    while t < split.stop:
        block_len = min(forecast_horizon, split.stop - t)
        hist = head[t - window : t].copy()
        pred_block: list[float] = []
        for step in range(block_len):
            current = t + step
            step_window = np.concatenate(
                [climate[current - window + 1 : current + 1], hist[:, None]],
                axis=1,
            ).astype(np.float32)
            x_step = ((step_window - step_mu) / step_sd)[None, :, :].astype(np.float32)
            with torch.no_grad():
                pred_norm = model(torch.from_numpy(x_step)).cpu().numpy()[0]
            yhat = pred_norm * head_sd + head_mu
            pred_block.append(float(yhat))
            hist = np.concatenate([hist[1:], np.array([yhat], dtype=np.float32)])
        pred_all.append(np.asarray(pred_block, dtype=np.float32))
        obs_all.append(head[t : t + block_len].astype(np.float32))
        date_all.append(series.dates[t : t + block_len].astype("datetime64[D]"))
        t += forecast_horizon

    pred = np.concatenate(pred_all)
    obs = np.concatenate(obs_all)
    dates = np.concatenate(date_all)
    resid = pred - obs
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    metrics = {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "nse": float(1.0 - ss_res / (ss_tot + 1.0e-12)),
        "corr": float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 1.0e-9 and obs.std() > 1.0e-9 else float("nan"),
        "n_pred_days": int(pred.size),
    }
    metrics.update(lag_diagnostic(pred, obs, max_lag=max(14, forecast_horizon)))
    metrics.update(peak_timing_diagnostic(pred, obs))
    return {
        "pred": pred,
        "obs": obs,
        "dates": dates.astype("datetime64[D]").astype(str),
        "metrics": metrics,
    }


def recursive_rollout_one_step_delta(
    model,
    series,
    split: slice,
    norm: dict[str, np.ndarray | float],
    window: int,
    forecast_horizon: int,
    include_dhead: bool = True,
) -> dict[str, np.ndarray | dict]:
    feat_mu = np.asarray(norm["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(norm["feat_sd"], dtype=np.float32)
    climate_mu = np.asarray(norm["climate_mu"], dtype=np.float32)
    climate_sd = np.asarray(norm["climate_sd"], dtype=np.float32)
    delta_mu = float(norm["delta_mu"])
    delta_sd = float(norm["delta_sd"])

    head = series.head_interp.astype(np.float32)
    climate = series.climate.astype(np.float32)

    preds: list[float] = []
    obs: list[float] = []
    dates: list[np.datetime64] = []

    model.eval()
    for start in range(split.start, split.stop - forecast_horizon + 1):
        hist = head[start - window : start].copy()
        for step in range(forecast_horizon):
            current = start + step
            past_feat = _compose_delta_past_features(
                climate[current - window : current],
                hist,
                include_dhead=include_dhead,
            ).astype(np.float32)
            x_past = ((past_feat - feat_mu) / feat_sd)[None, :, :].astype(np.float32)
            x_future = (((climate[current][None, :] - climate_mu) / climate_sd)[None, :, :]).astype(np.float32)
            with torch.no_grad():
                pred_delta_norm = model(torch.from_numpy(x_past), torch.from_numpy(x_future)).cpu().numpy()[0]
            yhat = hist[-1] + pred_delta_norm * delta_sd + delta_mu
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


def recursive_block_rollout_one_step_delta(
    model,
    series,
    split: slice,
    norm: dict[str, np.ndarray | float],
    window: int,
    forecast_horizon: int,
    include_dhead: bool = True,
) -> dict[str, np.ndarray | dict]:
    feat_mu = np.asarray(norm["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(norm["feat_sd"], dtype=np.float32)
    climate_mu = np.asarray(norm["climate_mu"], dtype=np.float32)
    climate_sd = np.asarray(norm["climate_sd"], dtype=np.float32)
    delta_mu = float(norm["delta_mu"])
    delta_sd = float(norm["delta_sd"])

    head = series.head_interp.astype(np.float32)
    climate = series.climate.astype(np.float32)

    pred_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = []
    date_all: list[np.ndarray] = []

    model.eval()
    t = split.start
    while t < split.stop:
        block_len = min(forecast_horizon, split.stop - t)
        hist = head[t - window : t].copy()
        pred_block: list[float] = []
        for step in range(block_len):
            current = t + step
            past_feat = _compose_delta_past_features(
                climate[current - window : current],
                hist,
                include_dhead=include_dhead,
            ).astype(np.float32)
            x_past = ((past_feat - feat_mu) / feat_sd)[None, :, :].astype(np.float32)
            x_future = (((climate[current][None, :] - climate_mu) / climate_sd)[None, :, :]).astype(np.float32)
            with torch.no_grad():
                pred_delta_norm = model(torch.from_numpy(x_past), torch.from_numpy(x_future)).cpu().numpy()[0]
            yhat = hist[-1] + pred_delta_norm * delta_sd + delta_mu
            pred_block.append(float(yhat))
            hist = np.concatenate([hist[1:], np.array([yhat], dtype=np.float32)])
        pred_all.append(np.asarray(pred_block, dtype=np.float32))
        obs_all.append(head[t : t + block_len].astype(np.float32))
        date_all.append(series.dates[t : t + block_len].astype("datetime64[D]"))
        t += forecast_horizon

    pred = np.concatenate(pred_all)
    obs = np.concatenate(obs_all)
    dates = np.concatenate(date_all)
    resid = pred - obs
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    metrics = {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "nse": float(1.0 - ss_res / (ss_tot + 1.0e-12)),
        "corr": float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 1.0e-9 and obs.std() > 1.0e-9 else float("nan"),
        "n_pred_days": int(pred.size),
    }
    metrics.update(lag_diagnostic(pred, obs, max_lag=max(14, forecast_horizon)))
    metrics.update(peak_timing_diagnostic(pred, obs))
    return {
        "pred": pred,
        "obs": obs,
        "dates": dates.astype("datetime64[D]").astype(str),
        "metrics": metrics,
    }

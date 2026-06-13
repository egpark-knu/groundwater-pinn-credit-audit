from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .neural_ladder import LadderSeries, lag_diagnostic, peak_timing_diagnostic


def build_nf_frame(series: LadderSeries) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "unique_id": series.stem,
            "ds": pd.to_datetime(series.dates.astype("datetime64[ns]")),
            "y": series.head_interp.astype(float),
        }
    )
    for idx, col in enumerate(series.climate_cols):
        df[col] = series.climate[:, idx].astype(float)
    return df


def make_recursive_history_and_future(
    full_df: pd.DataFrame,
    start_idx: int,
    step_idx: int,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = start_idx + step_idx
    hist_start = max(0, current - window)
    hist_df = full_df.iloc[hist_start:current].copy()
    if len(hist_df) < window:
        raise ValueError(f"NHITS recursive step requires at least {window} history rows, got {len(hist_df)}.")
    futr_df = full_df.iloc[current : current + 1].drop(columns=["y"]).copy()
    return hist_df, futr_df


def require_neuralforecast():
    try:
        from neuralforecast import NeuralForecast  # type: ignore
        from neuralforecast.models import NHITS  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when dependency missing
        raise ImportError(
            "NHITS baseline requires `neuralforecast` and its Lightning dependencies."
        ) from exc
    return NeuralForecast, NHITS


def fit_nhits_one_step(
    train_df: pd.DataFrame,
    exog_cols: list[str],
    input_size: int = 30,
    max_steps: int = 200,
    random_seed: int = 42,
):
    NeuralForecast, NHITS = require_neuralforecast()
    model = NHITS(
        h=1,
        input_size=input_size,
        futr_exog_list=exog_cols,
        start_padding_enabled=True,
        scaler_type="standard",
        max_steps=max_steps,
        random_seed=random_seed,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df)
    return nf


def recursive_rollout_nhits(
    nf,
    series: LadderSeries,
    split: slice,
    window: int,
    forecast_horizon: int,
) -> dict[str, np.ndarray | dict]:
    full_df = build_nf_frame(series)
    exog_cols = list(series.climate_cols)
    preds: list[float] = []
    obs: list[float] = []
    dates: list[str] = []

    model_name = getattr(nf.models[0], "alias", None) or nf.models[0].__class__.__name__

    for start in range(split.start, split.stop - forecast_horizon + 1):
        hist_df = full_df.iloc[max(0, start - window) : start].copy()
        if len(hist_df) < window:
            raise ValueError(f"NHITS recursive rollout requires at least {window} history rows.")
        for step in range(forecast_horizon):
            futr_df = full_df.iloc[start + step : start + step + 1].drop(columns=["y"]).copy()
            pred_df = nf.predict(df=hist_df, futr_df=futr_df)
            yhat = float(pred_df.iloc[0][model_name])
            new_row = full_df.iloc[start + step : start + step + 1].copy()
            new_row.loc[:, "y"] = yhat
            hist_df = pd.concat([hist_df, new_row], ignore_index=True).iloc[-window:].copy()
        preds.append(float(hist_df.iloc[-1]["y"]))
        obs.append(float(series.head_interp[start + forecast_horizon - 1]))
        dates.append(str(series.dates[start + forecast_horizon - 1].astype("datetime64[D]")))

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
        "dates": np.asarray(dates),
        "metrics": metrics,
    }


def recursive_block_rollout_nhits(
    nf,
    series: LadderSeries,
    split: slice,
    window: int,
    forecast_horizon: int,
) -> dict[str, np.ndarray | dict]:
    full_df = build_nf_frame(series)
    pred_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = []
    date_all: list[np.ndarray] = []

    model_name = getattr(nf.models[0], "alias", None) or nf.models[0].__class__.__name__

    t = split.start
    while t < split.stop:
        block_len = min(forecast_horizon, split.stop - t)
        hist_df = full_df.iloc[max(0, t - window) : t].copy()
        if len(hist_df) < window:
            raise ValueError(f"NHITS recursive rollout requires at least {window} history rows.")
        pred_block: list[float] = []
        for step in range(block_len):
            futr_df = full_df.iloc[t + step : t + step + 1].drop(columns=["y"]).copy()
            pred_df = nf.predict(df=hist_df, futr_df=futr_df)
            yhat = float(pred_df.iloc[0][model_name])
            pred_block.append(yhat)
            new_row = full_df.iloc[t + step : t + step + 1].copy()
            new_row.loc[:, "y"] = yhat
            hist_df = pd.concat([hist_df, new_row], ignore_index=True).iloc[-window:].copy()
        pred_all.append(np.asarray(pred_block, dtype=np.float32))
        obs_all.append(series.head_interp[t : t + block_len].astype(np.float32))
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

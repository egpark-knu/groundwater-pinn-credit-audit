from __future__ import annotations

import numpy as np

from .neural_ladder import lag_diagnostic, metrics_from_rollout, peak_timing_diagnostic


def rollout_persistence(
    series,
    split: slice,
    horizon: int,
) -> dict[str, np.ndarray | dict]:
    if split.start < 1:
        raise ValueError("Persistence rollout requires at least one prior day.")

    pred_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = []
    date_all: list[np.ndarray] = []

    t = split.start
    while t < split.stop:
        block_len = min(horizon, split.stop - t)
        h0 = float(series.head_interp[t - 1])
        pred_block = np.full(block_len, h0, dtype=np.float32)
        pred_all.append(pred_block)
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

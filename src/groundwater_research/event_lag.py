from __future__ import annotations

import numpy as np


def select_event_centers(obs: np.ndarray, k: int = 3, min_gap: int = 21) -> list[int]:
    obs = np.asarray(obs, dtype=float)
    shock = np.abs(np.diff(obs))
    order = np.argsort(shock)[::-1]
    centers: list[int] = []
    for idx in order:
        center = int(idx + 1)
        if all(abs(center - prev) >= min_gap for prev in centers):
            centers.append(center)
        if len(centers) >= k:
            break
    return sorted(centers)


def local_event_lags(
    pred: np.ndarray,
    obs: np.ndarray,
    dates: np.ndarray,
    centers: list[int],
    half_window: int = 14,
) -> list[dict[str, int | str | float]]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    dates = np.asarray(dates).astype("datetime64[D]")
    rows: list[dict[str, int | str | float]] = []
    for center in centers:
        lo = max(0, center - half_window)
        hi = min(len(obs), center + half_window + 1)
        pred_w = pred[lo:hi]
        obs_w = obs[lo:hi]
        dates_w = dates[lo:hi]
        peak_lag = int(np.argmax(pred_w) - np.argmax(obs_w))
        trough_lag = int(np.argmin(pred_w) - np.argmin(obs_w))
        if len(pred_w) >= 2 and len(obs_w) >= 2:
            drop_lag = int(np.argmin(np.diff(pred_w)) - np.argmin(np.diff(obs_w)))
            rise_lag = int(np.argmax(np.diff(pred_w)) - np.argmax(np.diff(obs_w)))
        else:
            drop_lag = 0
            rise_lag = 0
        rows.append(
            {
                "center_date": str(dates[center]),
                "window_start": str(dates_w[0]),
                "window_end": str(dates_w[-1]),
                "peak_lag_days": peak_lag,
                "trough_lag_days": trough_lag,
                "drop_lag_days": drop_lag,
                "rise_lag_days": rise_lag,
            }
        )
    return rows


def summarize_local_event_lags(rows: list[dict[str, int | str | float]]) -> dict[str, float | int]:
    if not rows:
        return {
            "n_events": 0,
            "median_abs_peak_lag_days": 0.0,
            "max_abs_peak_lag_days": 0,
            "median_abs_trough_lag_days": 0.0,
            "max_abs_trough_lag_days": 0,
            "median_abs_drop_lag_days": 0.0,
            "max_abs_drop_lag_days": 0,
            "median_abs_rise_lag_days": 0.0,
            "max_abs_rise_lag_days": 0,
        }

    peak = np.array([abs(int(r["peak_lag_days"])) for r in rows], dtype=float)
    trough = np.array([abs(int(r["trough_lag_days"])) for r in rows], dtype=float)
    drop = np.array([abs(int(r["drop_lag_days"])) for r in rows], dtype=float)
    rise = np.array([abs(int(r["rise_lag_days"])) for r in rows], dtype=float)
    return {
        "n_events": int(len(rows)),
        "median_abs_peak_lag_days": float(np.median(peak)),
        "max_abs_peak_lag_days": int(np.max(peak)),
        "median_abs_trough_lag_days": float(np.median(trough)),
        "max_abs_trough_lag_days": int(np.max(trough)),
        "median_abs_drop_lag_days": float(np.median(drop)),
        "max_abs_drop_lag_days": int(np.max(drop)),
        "median_abs_rise_lag_days": float(np.median(rise)),
        "max_abs_rise_lag_days": int(np.max(rise)),
    }

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .neural_ladder import LadderSeries


def _robust_std(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0
    return float(np.std(clean))


def _flatline_flags(head: np.ndarray, min_days: int) -> np.ndarray:
    values = np.asarray(head, dtype=float)
    flags = np.zeros(values.shape, dtype=bool)
    start = 0
    n = len(values)
    while start < n:
        if not np.isfinite(values[start]):
            start += 1
            continue
        stop = start + 1
        while stop < n and np.isfinite(values[stop]) and values[stop] == values[start]:
            stop += 1
        if stop - start >= min_days:
            flags[start:stop] = True
        start = stop
    return flags


def detect_head_outliers(
    head: np.ndarray,
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    flatline_min_days: int = 10,
    jump_sigma: float = 3.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    values = np.asarray(head, dtype=float)
    finite = np.isfinite(values)
    flags_iqr = np.zeros(values.shape, dtype=bool)
    flags_z = np.zeros(values.shape, dtype=bool)
    flags_jump = np.zeros(values.shape, dtype=bool)

    clean = values[finite]
    if clean.size:
        q1, q3 = np.percentile(clean, [25, 75])
        iqr = float(q3 - q1)
        lo = float(q1 - iqr_multiplier * iqr)
        hi = float(q3 + iqr_multiplier * iqr)
        flags_iqr = finite & ((values < lo) | (values > hi))
        mu = float(np.mean(clean))
        sd = float(np.std(clean) + 1.0e-12)
        flags_z = finite & (np.abs((values - mu) / sd) > z_threshold)
    else:
        lo = hi = mu = sd = float("nan")

    flags_flat = _flatline_flags(values, min_days=flatline_min_days)
    diff = np.diff(values, prepend=np.nan)
    diff_sd = _robust_std(diff)
    if diff_sd > 0:
        flags_jump = np.isfinite(diff) & (np.abs(diff) > jump_sigma * diff_sd)

    flags = flags_iqr | flags_z | flags_flat | flags_jump
    report = {
        "n_total": int(values.size),
        "n_finite": int(finite.sum()),
        "n_iqr": int(flags_iqr.sum()),
        "n_zscore": int(flags_z.sum()),
        "n_flatline": int(flags_flat.sum()),
        "n_jump": int(flags_jump.sum()),
        "n_flagged": int(flags.sum()),
        "flagged_ratio_total": float(flags.sum() / max(values.size, 1)),
        "flagged_ratio_finite": float(flags.sum() / max(finite.sum(), 1)),
        "iqr_lower": lo,
        "iqr_upper": hi,
        "head_mean": mu,
        "head_std": sd,
        "diff_std": diff_sd,
    }
    return flags, report


def clean_ladder_series(
    series: LadderSeries,
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    flatline_min_days: int = 10,
    jump_sigma: float = 3.0,
) -> tuple[LadderSeries, dict[str, float | int]]:
    flags, report = detect_head_outliers(
        series.head_raw,
        iqr_multiplier=iqr_multiplier,
        z_threshold=z_threshold,
        flatline_min_days=flatline_min_days,
        jump_sigma=jump_sigma,
    )
    cleaned_raw = np.asarray(series.head_raw, dtype=float).copy()
    cleaned_raw[flags] = np.nan
    cleaned_interp = (
        pd.Series(cleaned_raw)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )
    cleaned = replace(series, head_raw=cleaned_raw, head_interp=cleaned_interp)
    return cleaned, report


def per_date_outlier_frame(series: LadderSeries, flags: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": series.dates.astype("datetime64[D]").astype(str),
            "head_raw": series.head_raw,
            "is_outlier": np.asarray(flags, dtype=bool),
        }
    )

from __future__ import annotations

import numpy as np

from groundwater_research.neural_ladder import LadderSeries
from scripts.run_physics_input_dnn import (
    align_feature_to_series,
    subset_series_to_dates,
    with_external_head_feature,
)


def _toy_series() -> LadderSeries:
    return LadderSeries(
        stem="toy",
        dates=np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"),
        head_raw=np.array([1.0, 2.0, 3.0]),
        head_interp=np.array([1.0, 2.0, 3.0]),
        rain_mm=np.array([0.0, 5.0, 0.0]),
        climate=np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 0.0]], dtype=np.float32),
        climate_cols=["TEMP", "RAIN"],
    )


def test_subset_series_to_dates_preserves_matching_window() -> None:
    series = _toy_series()
    subset = subset_series_to_dates(series, "2024-01-02", "2024-01-03")

    assert subset.dates.shape == (2,)
    assert np.allclose(subset.head_interp, [2.0, 3.0])
    assert np.allclose(subset.climate[:, 1], [5.0, 0.0])


def test_align_feature_to_series_matches_dates() -> None:
    series = _toy_series()
    feature_dates = np.array(["2024-01-01", "2024-01-03"], dtype="datetime64[D]")
    feature_values = np.array([10.0, 30.0])

    aligned = align_feature_to_series(series, feature_dates, feature_values)

    assert np.allclose(aligned, [10.0, 20.0, 30.0])


def test_with_external_head_feature_appends_climate_column() -> None:
    series = _toy_series()
    augmented = with_external_head_feature(series, np.array([10.0, 20.0, 30.0]))

    assert augmented.climate.shape == (3, 3)
    assert augmented.climate_cols[-1] == "ESMDA_HEAD"
    assert np.allclose(augmented.climate[:, -1], [10.0, 20.0, 30.0])

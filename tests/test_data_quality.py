from __future__ import annotations

import numpy as np

from groundwater_research.data_quality import detect_head_outliers


def test_detect_head_outliers_combines_iqr_zscore_flatline_and_jump_flags() -> None:
    head = np.array(
        [
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            50.0,
            1.5,
            *([2.0] * 10),
            2.1,
            2.2,
            8.0,
            2.3,
        ],
        dtype=float,
    )

    flags, report = detect_head_outliers(head, flatline_min_days=10)

    assert flags[5]
    assert flags[7:17].all()
    assert flags[19]
    assert report["n_iqr"] >= 1
    assert report["n_flatline"] == 10
    assert report["n_jump"] >= 1
    assert report["n_flagged"] == int(flags.sum())

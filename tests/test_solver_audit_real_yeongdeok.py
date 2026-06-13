from __future__ import annotations

import numpy as np

from scripts.run_solver_audit_real_yeongdeok import (
    augment_observation_for_rmse_nse,
    derive_real_aquifer_config,
    dual_objective,
    nse_score,
    observation_operator,
    select_site_window,
)
from groundwater_research.virtual_aquifer import SpatialAquiferConfig


def test_select_site_window_keeps_matching_lengths() -> None:
    dates = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        dtype="datetime64[D]",
    )
    obs = np.array([1.0, 2.0, 3.0])
    rain = np.array([0.0, 10.0, 0.0])
    valid = np.array([True, True, False])

    window = select_site_window(dates, obs, rain, valid, "2024-01-02", "2024-01-03")

    assert window["dates"].shape == (2,)
    assert np.allclose(window["obs"], [2.0, 3.0])
    assert np.allclose(window["rain_mm"], [10.0, 0.0])
    assert np.array_equal(window["valid_mask"], [True, False])


def test_derive_real_aquifer_config_sets_west_chd_to_low_head_quantile() -> None:
    obs = np.array([7.8, 8.0, 8.2, 8.4, 8.6])
    cfg, base_head, west_chd = derive_real_aquifer_config(obs, epsilon_m=0.05)

    assert isinstance(cfg, SpatialAquiferConfig)
    assert np.isclose(west_chd, np.quantile(obs, 0.10))
    assert np.isclose(base_head, west_chd + 0.05)
    assert np.isclose(cfg.west_chd_head_m, west_chd)
    assert cfg.observation_cells == ((5, 5),)


def test_dual_objective_combines_mse_and_nse_penalty() -> None:
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 4.0])

    expected_mse = 1.0 / 3.0
    expected_nse = 1.0 - 1.0 / 2.0

    assert np.isclose(nse_score(predicted, observed), expected_nse)
    assert np.isclose(
        dual_objective(predicted, observed, w_mse=0.7, w_nse=0.3),
        0.7 * expected_mse + 0.3 * (1.0 - expected_nse),
    )


def test_augmented_observation_adds_standardized_trajectory_component() -> None:
    observed = np.array([10.0, 12.0, 14.0])
    predicted = np.array([[10.0, 12.0, 13.0], [11.0, 13.0, 15.0]])

    pred_aug, obs_aug = augment_observation_for_rmse_nse(
        predicted,
        observed,
        w_mse=1.0,
        w_nse=4.0,
    )

    assert pred_aug.shape == (2, 6)
    assert obs_aug.shape == (6,)
    assert np.allclose(pred_aug[:, :3], predicted)
    assert np.allclose(obs_aug[:3], observed)
    assert np.isclose(obs_aug[3:].mean(), 0.0)
    assert np.isclose(obs_aug[3:].std(), 2.0)


def test_observation_operator_delta_uses_first_difference() -> None:
    predicted = np.array([[10.0, 11.0, 10.5], [9.0, 9.5, 10.0]])
    observed = np.array([10.0, 10.25, 10.75])

    pred_op, obs_op = observation_operator(predicted, observed, mode="delta")

    assert pred_op.shape == (2, 2)
    assert obs_op.shape == (2,)
    assert np.allclose(pred_op[0], [1.0, -0.5])
    assert np.allclose(pred_op[1], [0.5, 0.5])
    assert np.allclose(obs_op, [0.25, 0.5])


def test_observation_operator_baseline_removes_initial_level() -> None:
    predicted = np.array([[10.0, 11.0, 10.5]])
    observed = np.array([9.5, 10.0, 10.25])

    pred_op, obs_op = observation_operator(predicted, observed, mode="baseline")

    assert np.allclose(pred_op, [[0.0, 1.0, 0.5]])
    assert np.allclose(obs_op, [0.0, 0.5, 0.75])

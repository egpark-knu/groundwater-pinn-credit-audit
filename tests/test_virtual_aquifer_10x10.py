from __future__ import annotations

import numpy as np

from groundwater_research.virtual_aquifer import (
    SpatialAquiferConfig,
    build_10x10_model,
    build_spatial_logk_field,
    center_observation_cell,
    run_10x10_center_hydrograph,
    spatial_exponential_covariance,
    spatial_theta_to_fields,
    west_chd_period_data,
)


def test_spatial_aquifer_config_defines_10x10_contract() -> None:
    cfg = SpatialAquiferConfig()

    assert cfg.shape == (10, 10)
    assert cfg.n_cells == 100
    assert cfg.cell_size_m == 100.0
    assert len(cfg.observation_cells) == 1
    assert cfg.observation_cells == (center_observation_cell(cfg),)
    assert all(0 <= row < cfg.nrow and 0 <= col < cfg.ncol for row, col in cfg.observation_cells)

    chd = west_chd_period_data(cfg)
    assert len(chd) == cfg.nrow
    assert [cell for cell, _head in chd] == [(0, row, 0) for row in range(cfg.nrow)]
    assert {head for _cell, head in chd} == {cfg.west_chd_head_m}


def test_spatial_logk_field_is_reproducible_and_scaled() -> None:
    cfg = SpatialAquiferConfig()
    field_a = build_spatial_logk_field(cfg, np.random.default_rng(42))
    field_b = build_spatial_logk_field(cfg, np.random.default_rng(42))

    assert field_a.shape == cfg.shape
    assert np.allclose(field_a, field_b)
    assert abs(float(field_a.mean()) - cfg.mean_ln_k_m_per_s) < 1.0e-10
    assert abs(float(field_a.std(ddof=1)) - cfg.std_ln_k) < 1.0e-10
    assert np.isfinite(field_a).all()


def test_spatial_correlation_length_is_grid_scale_invariant() -> None:
    cfg_100 = SpatialAquiferConfig(cell_size_m=100.0)
    cfg_500 = SpatialAquiferConfig(cell_size_m=500.0)

    assert cfg_100.corr_len_cells == 5.0
    assert cfg_500.corr_len_cells == 5.0
    assert cfg_100.corr_len_m == 500.0
    assert cfg_500.corr_len_m == 2500.0

    cov_100 = spatial_exponential_covariance(cfg_100)
    cov_500 = spatial_exponential_covariance(cfg_500)
    center_idx = 5 * cfg_100.ncol + 5
    five_cells_west_idx = 5 * cfg_100.ncol + 0

    ratio_100 = cov_100[center_idx, five_cells_west_idx] / cov_100[center_idx, center_idx]
    ratio_500 = cov_500[center_idx, five_cells_west_idx] / cov_500[center_idx, center_idx]

    assert np.isclose(ratio_100, np.exp(-1.0))
    assert np.isclose(ratio_500, np.exp(-1.0))
    assert np.isclose(ratio_100, ratio_500)


def test_build_10x10_model_writes_modflow_input(tmp_path) -> None:
    cfg = SpatialAquiferConfig()
    k_m_per_day = np.full(cfg.shape, cfg.mean_k_m_per_day)
    sim = build_10x10_model(
        model_ws=tmp_path,
        k_m_per_day=k_m_per_day,
        recharge_m_per_day=cfg.default_recharge_m_per_day,
        config=cfg,
    )

    sim.write_simulation(silent=True)

    assert (tmp_path / "mfsim.nam").exists()
    assert (tmp_path / "spatial_aquifer.nam").exists()
    assert (tmp_path / "spatial_aquifer.dis").exists()
    assert (tmp_path / "spatial_aquifer.chd").exists()
    assert (tmp_path / "spatial_aquifer.rch").exists()


def test_spatial_theta_to_fields_uses_k_in_m_per_s_and_recharge_multiplier() -> None:
    cfg = SpatialAquiferConfig()
    theta = np.full(cfg.n_cells + 1, cfg.mean_ln_k_m_per_s)
    theta[-1] = np.log(1.25)

    k_m_per_day, recharge_m_per_day = spatial_theta_to_fields(theta, cfg)

    assert k_m_per_day.shape == cfg.shape
    assert np.allclose(k_m_per_day, cfg.mean_k_m_per_day)
    assert recharge_m_per_day == cfg.default_recharge_m_per_day * 1.25


def test_center_observation_cell_is_single_well_at_grid_center() -> None:
    cfg = SpatialAquiferConfig()

    assert center_observation_cell(cfg) == (cfg.nrow // 2, cfg.ncol // 2)


def test_run_center_hydrograph_returns_one_value_per_recharge_day(tmp_path) -> None:
    cfg = SpatialAquiferConfig()
    recharge = np.array([0.0, cfg.default_recharge_m_per_day, cfg.default_recharge_m_per_day * 2.0])
    k_m_per_day = np.full(cfg.shape, cfg.mean_k_m_per_day)
    sy = np.full(cfg.shape, cfg.specific_yield)

    hydrograph = run_10x10_center_hydrograph(
        model_ws=tmp_path,
        k_m_per_day=k_m_per_day,
        recharge_m_per_day=recharge,
        config=cfg,
        sy=sy,
    )

    assert hydrograph.shape == (len(recharge),)
    assert np.isfinite(hydrograph).all()

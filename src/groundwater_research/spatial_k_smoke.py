from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import flopy
import numpy as np

from groundwater_research.virtual_aquifer import find_mf6_executable


@dataclass
class SpatialKSmokeConfig:
    n_cells: int = 40
    cell_size_m: float = 100.0
    top_m: float = 15.0
    botm_m: float = 0.0
    initial_head_m: float = 7.0
    outlet_head_m: float = 5.0
    sy: float = 0.08
    ss_m_inv: float = 1.0e-5
    hclose: float = 1.0e-4
    rclose: float = 1.0e-3
    obs_col: int = 10


def sample_correlated_logk_fields(
    n_members: int,
    n_cells: int,
    cell_size_m: float,
    corr_len_m: float,
    mean_logk: float = 0.0,
    std_logk: float = 0.8,
    seed: int = 260409,
    min_k: float = 0.03,
    max_k: float = 30.0,
) -> np.ndarray:
    """Sample 1D Gaussian log-K fields with a fixed marginal variance."""
    if n_members <= 0:
        raise ValueError("n_members must be positive")
    if n_cells <= 1:
        raise ValueError("n_cells must be greater than one")
    if corr_len_m <= 0.0:
        raise ValueError("corr_len_m must be positive")

    x = np.arange(n_cells, dtype=float) * cell_size_m
    dist = np.abs(x[:, None] - x[None, :])
    cov = (std_logk**2) * np.exp(-(dist**2) / (2.0 * corr_len_m**2))
    cov += 1.0e-8 * np.eye(n_cells)
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_members, n_cells))
    fields = mean_logk + z @ chol.T
    return np.clip(fields, np.log(min_k), np.log(max_k))


def build_pulse_recharge(
    n_days: int,
    pulse_days: tuple[int, ...] = (12, 36, 72),
    pulse_mm: float = 20.0,
    recharge_fraction: float = 0.20,
    background_mm: float = 0.0,
) -> np.ndarray:
    """Create a simple recharge sequence from synthetic precipitation pulses."""
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    precip_mm = np.full(n_days, float(background_mm), dtype=float)
    for day in pulse_days:
        if 0 <= day < n_days:
            precip_mm[day] += pulse_mm
    return np.maximum(precip_mm, 0.0) * 1.0e-3 * recharge_fraction


def summarize_member_heads(heads: np.ndarray) -> dict[str, float]:
    """Summarize member-wise head response at a single observation location."""
    arr = np.asarray(heads, dtype=float)
    if arr.ndim != 2:
        raise ValueError("heads must have shape (n_members, n_times)")
    if arr.shape[0] < 2:
        raise ValueError("Need at least two members to summarize spread")
    member_amplitude = np.ptp(arr, axis=1)
    temporal_std = np.std(arr, axis=0, ddof=1)
    member_range = np.ptp(arr, axis=0)
    return {
        "mean_temporal_std": float(np.mean(temporal_std)),
        "max_temporal_std": float(np.max(temporal_std)),
        "mean_member_amplitude": float(np.mean(member_amplitude)),
        "max_member_range": float(np.max(member_range)),
    }


def harmonic_mean_k(logk: np.ndarray) -> float:
    k = np.exp(np.asarray(logk, dtype=float))
    return float(len(k) / np.sum(1.0 / np.maximum(k, 1.0e-12)))


def build_spatial_k_transect_sim(
    model_ws: Path,
    logk: np.ndarray,
    recharge_m_per_day: np.ndarray,
    config: SpatialKSmokeConfig,
) -> flopy.mf6.MFSimulation:
    """Build a small 1D MODFLOW transect with steady-state spin-up."""
    model_ws.mkdir(parents=True, exist_ok=True)
    logk = np.asarray(logk, dtype=float)
    recharge_m_per_day = np.asarray(recharge_m_per_day, dtype=float)
    n_cells = len(logk)
    nper = len(recharge_m_per_day) + 1

    sim = flopy.mf6.MFSimulation(
        sim_name="spatial_k_smoke",
        version="mf6",
        exe_name=find_mf6_executable(),
        sim_ws=str(model_ws),
    )
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=nper,
        perioddata=[(1.0, 1, 1.0)] * nper,
    )
    flopy.mf6.ModflowIms(
        sim,
        complexity="MODERATE",
        outer_maximum=100,
        inner_maximum=100,
        outer_dvclose=config.hclose,
        inner_dvclose=config.hclose,
        rcloserecord=config.rclose,
        linear_acceleration="BICGSTAB",
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname="spatial_k_smoke",
        save_flows=True,
        newtonoptions="UNDER_RELAXATION",
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=1,
        ncol=n_cells,
        delr=config.cell_size_m,
        delc=config.cell_size_m,
        top=config.top_m,
        botm=config.botm_m,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=config.initial_head_m)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        icelltype=1,
        k=np.exp(logk).reshape(1, 1, n_cells),
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        save_flows=True,
        iconvert=1,
        ss=config.ss_m_inv,
        sy=config.sy,
        steady_state={0: True},
        transient={idx: True for idx in range(1, nper)},
    )

    chd_spd = {idx: [((0, 0, n_cells - 1), config.outlet_head_m)] for idx in range(nper)}
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd, save_flows=True)

    steady_recharge = float(np.mean(recharge_m_per_day))
    rch_spd = {
        0: [((0, 0, col), steady_recharge) for col in range(n_cells)],
    }
    rch_spd.update(
        {
            idx + 1: [((0, 0, col), float(rate)) for col in range(n_cells)]
            for idx, rate in enumerate(recharge_m_per_day)
        }
    )
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data=rch_spd, save_flows=True)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="spatial_k_smoke.hds",
        budget_filerecord="spatial_k_smoke.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def run_spatial_k_member(
    model_ws: Path,
    logk: np.ndarray,
    recharge_m_per_day: np.ndarray,
    config: SpatialKSmokeConfig,
) -> dict[str, np.ndarray | float]:
    sim = build_spatial_k_transect_sim(model_ws, logk, recharge_m_per_day, config)
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    if not success:
        raise RuntimeError(f"MF6 failed in {model_ws}")

    hds = flopy.utils.HeadFile(str(model_ws / "spatial_k_smoke.hds"))
    heads = np.array(
        [
            hds.get_data(kstpkper=(0, idx + 1))[0, 0, :]
            for idx in range(len(recharge_m_per_day))
        ],
        dtype=float,
    )
    obs_col = int(np.clip(config.obs_col, 0, len(logk) - 1))
    obs_head = heads[:, obs_col]
    return {
        "obs_head": obs_head,
        "final_head": heads[-1],
        "harmonic_k": harmonic_mean_k(logk),
        "local_k": float(np.exp(logk[obs_col])),
        "logk_std": float(np.std(logk, ddof=1)),
    }

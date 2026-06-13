from __future__ import annotations

import csv
import json
import os
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import flopy
import numpy as np
import pandas as pd


DEFAULT_GROUNDWATER_ROOT = Path(
    os.environ.get("NGMS_GROUNDWATER_ROOT", Path(__file__).resolve().parents[2] / "data" / "groundwater")
)
DEFAULT_CASE_CATALOG = Path(
    os.environ.get(
        "PINN_CASE_CATALOG",
        Path(__file__).resolve().parents[2] / "results" / "data_screening" / "groundwater_case_catalog.csv",
    )
)
MF6_CANDIDATES = [
    Path(path)
    for path in os.environ.get("MF6_EXE_CANDIDATES", "").split(os.pathsep)
    if path
]


@dataclass
class SiteSeries:
    stem: str
    dates: np.ndarray
    obs_raw: np.ndarray
    obs_interp: np.ndarray
    obs_valid_mask: np.ndarray
    rain_mm: np.ndarray
    temp_c: np.ndarray
    material_class: str
    archetype: str


@dataclass
class VirtualAquiferConfig:
    cell_size_m: float = 500.0
    layer1_thickness_m: float = 8.0
    layer2_thickness_m: float = 45.0
    top_buffer_m: float = 5.0
    vertical_k_m_per_day: float = 0.05
    specific_storage_m_inv: float = 1.0e-5
    recharge_fraction: float = 0.20
    steady_recharge_m_per_day: Optional[float] = None
    period_days: float = 1.0
    outer_maximum: int = 300
    inner_maximum: int = 200
    hclose: float = 1.0e-4
    rclose: float = 1.0e-3


@dataclass
class SpatialAquiferConfig:
    """Conceptual 10x10 virtual aquifer for spatial-K physical-credit audits."""

    nrow: int = 10
    ncol: int = 10
    cell_size_m: float = 100.0
    top_m: float = 10.0
    bottom_m: float = 0.0
    base_head_m: float = 9.5
    west_chd_epsilon_m: float = 0.05
    specific_storage_m_inv: float = 1.0e-5
    specific_yield: float = 0.08
    recharge_fraction: float = 0.20
    default_precip_m_per_day: float = 0.005
    period_days: float = 1.0
    mean_ln_k_m_per_s: float = float(np.log(1.0e-4))
    std_ln_k: float = 1.0
    corr_len_cells: float = 5.0
    obs_error_m: float = 0.05
    outer_maximum: int = 200
    inner_maximum: int = 100
    hclose: float = 1.0e-5
    rclose: float = 1.0e-4
    observation_cells: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: ((5, 5),)
    )
    mf6_exe: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrow, self.ncol)

    @property
    def n_cells(self) -> int:
        return self.nrow * self.ncol

    @property
    def west_chd_head_m(self) -> float:
        return self.base_head_m - self.west_chd_epsilon_m

    @property
    def default_recharge_m_per_day(self) -> float:
        return self.recharge_fraction * self.default_precip_m_per_day

    @property
    def mean_k_m_per_day(self) -> float:
        # The audit prior is specified in m/s; MODFLOW6 expects m/day.
        return float(np.exp(self.mean_ln_k_m_per_s) * 86400.0)

    @property
    def corr_len_m(self) -> float:
        # Keep spatial structure invariant in grid-cell units when cell size changes.
        return float(self.corr_len_cells * self.cell_size_m)


@dataclass
class VirtualAquiferParams:
    log_k1: float
    log_k2: float
    log_sy1: float
    log_ghb_mult: float
    h_ref: float
    tau_rch_days: float = 1.0

    @property
    def k1(self) -> float:
        return float(np.exp(self.log_k1))

    @property
    def k2(self) -> float:
        return float(np.exp(self.log_k2))

    @property
    def sy1(self) -> float:
        return float(np.exp(self.log_sy1))

    @property
    def sy2(self) -> float:
        return float(np.exp(self.log_sy1) * 0.1)

    @property
    def ghb_mult(self) -> float:
        return float(np.exp(self.log_ghb_mult))

    def to_dict(self) -> dict:
        values = asdict(self)
        values.update(
            {
                "k1": self.k1,
                "k2": self.k2,
                "sy1": self.sy1,
                "sy2": self.sy2,
                "ghb_mult": self.ghb_mult,
            }
        )
        return values


def parse_material_from_stem(stem: str) -> str:
    stem = unicodedata.normalize("NFC", stem)
    if stem.endswith("_암반"):
        return "암반"
    if stem.endswith("_충적"):
        return "충적"
    if "_암반" in stem:
        return "암반"
    return "충적"


def suggest_archetype_from_catalog(
    stem: str,
    catalog_path: Path = DEFAULT_CASE_CATALOG,
) -> Optional[str]:
    stem = unicodedata.normalize("NFC", stem)
    if not catalog_path.exists():
        return None
    with catalog_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if unicodedata.normalize("NFC", row.get("stem", "")) != stem:
                continue
            archetype = row.get("archetype_suggested", "").strip()
            if archetype in {"coastal", "inland"}:
                return archetype
            return None
    return None


def find_mf6_executable() -> str:
    for candidate in MF6_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    maybe_path = shutil.which("mf6")
    if maybe_path:
        return maybe_path
    raise FileNotFoundError("No MODFLOW 6 executable found in expected locations.")


def interpolate_series(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(values)
    if mask.all():
        return values.astype(float), mask
    idx = np.arange(len(values))
    interp = values.astype(float).copy()
    interp[~mask] = np.interp(idx[~mask], idx[mask], interp[mask])
    return interp, mask


def load_site_series(
    stem: str,
    archetype: str,
    groundwater_root: Path = DEFAULT_GROUNDWATER_ROOT,
) -> SiteSeries:
    stem = unicodedata.normalize("NFC", stem)
    wt_path = groundwater_root / "waterlevel" / f"{stem}_WT.txt"
    cl_path = groundwater_root / "climate" / f"{stem}_CL.txt"
    if not wt_path.exists():
        wt_path = groundwater_root / "waterlevel" / unicodedata.normalize("NFD", f"{stem}_WT.txt")
    if not cl_path.exists():
        cl_path = groundwater_root / "climate" / unicodedata.normalize("NFD", f"{stem}_CL.txt")
    if not wt_path.exists() or not cl_path.exists():
        raise FileNotFoundError(f"Missing WT/CL pair for {stem}")

    wt_df = pd.read_csv(wt_path, sep="\t")
    cl_df = pd.read_csv(cl_path, sep="\t")
    df = wt_df.merge(cl_df, on="Date", how="inner")
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")

    obs_raw = pd.to_numeric(df["Value"], errors="coerce").to_numpy(dtype=float)
    obs_interp, obs_valid = interpolate_series(obs_raw)
    rain_mm = pd.to_numeric(df["RAIN"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    temp_c = (
        pd.to_numeric(df["TEMP"], errors="coerce")
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )
    return SiteSeries(
        stem=stem,
        dates=df["date"].to_numpy(dtype="datetime64[ns]"),
        obs_raw=obs_raw,
        obs_interp=obs_interp,
        obs_valid_mask=obs_valid,
        rain_mm=rain_mm,
        temp_c=temp_c,
        material_class=parse_material_from_stem(stem),
        archetype=archetype,
    )


def build_recharge_series(
    rain_mm: np.ndarray,
    recharge_fraction: float = 0.20,
    tau_days: float = 1.0,
) -> np.ndarray:
    """Convert precipitation to effective recharge with a fixed RPR.

    The recharge-to-precipitation ratio is deliberately not an ES-MDA parameter.
    This keeps recharge uncertainty from being hidden as a calibration degree of
    freedom in the low-dimensional virtual aquifer audit.
    """
    rain_m = np.maximum(rain_mm, 0.0) * 1.0e-3
    if tau_days <= 1.0:
        return recharge_fraction * rain_m
    alpha = float(np.exp(-1.0 / tau_days))
    smooth = np.zeros_like(rain_m, dtype=float)
    smooth[0] = rain_m[0]
    for idx in range(1, len(rain_m)):
        smooth[idx] = alpha * smooth[idx - 1] + (1.0 - alpha) * rain_m[idx]
    return recharge_fraction * smooth


def _make_period_data(nper: int, period_days: float) -> list[tuple[float, int, float]]:
    return [(period_days, 1, 1.0)] * nper


def build_virtual_aquifer_sim(
    model_ws: Path,
    site: SiteSeries,
    params: VirtualAquiferParams,
    config: VirtualAquiferConfig,
) -> flopy.mf6.MFSimulation:
    model_ws.mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(
        sim_name="virtual_aquifer",
        version="mf6",
        exe_name=find_mf6_executable(),
        sim_ws=str(model_ws),
    )
    ntrans = len(site.dates)
    nper = ntrans + 1
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=nper,
        perioddata=_make_period_data(nper, config.period_days),
    )
    flopy.mf6.ModflowIms(
        sim,
        complexity="COMPLEX",
        outer_maximum=config.outer_maximum,
        inner_maximum=config.inner_maximum,
        outer_dvclose=config.hclose,
        inner_dvclose=config.hclose,
        rcloserecord=config.rclose,
        linear_acceleration="BICGSTAB",
        under_relaxation="DBD",
    )

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname="virtual_aquifer",
        save_flows=True,
        newtonoptions="UNDER_RELAXATION",
    )

    obs_max = float(np.nanmax(site.obs_interp))
    top = obs_max + config.top_buffer_m
    botm = np.zeros((2, 1, 1), dtype=float)
    botm[0, 0, 0] = top - config.layer1_thickness_m
    botm[1, 0, 0] = botm[0, 0, 0] - config.layer2_thickness_m

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=2,
        nrow=1,
        ncol=1,
        delr=config.cell_size_m,
        delc=config.cell_size_m,
        top=np.array([[top]], dtype=float),
        botm=botm,
        idomain=np.ones((2, 1, 1), dtype=int),
    )

    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        icelltype=[1, 0],
        k=np.array([[[params.k1]], [[params.k2]]], dtype=float),
        k33=np.array(
            [[[config.vertical_k_m_per_day]], [[config.vertical_k_m_per_day]]],
            dtype=float,
        ),
        k33overk=False,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        save_flows=True,
        iconvert=[1, 0],
        ss=config.specific_storage_m_inv,
        sy=[params.sy1, params.sy2],
        steady_state={0: True},
        transient={idx: True for idx in range(1, nper)},
    )
    flopy.mf6.ModflowGwfic(gwf, strt=float(params.h_ref))

    recharge = build_recharge_series(
        site.rain_mm,
        recharge_fraction=config.recharge_fraction,
        tau_days=params.tau_rch_days,
    )
    steady_recharge = (
        float(config.steady_recharge_m_per_day)
        if config.steady_recharge_m_per_day is not None
        else float(np.mean(recharge))
    )
    rch_spd = {
        0: [((0, 0, 0), steady_recharge)],
    }
    rch_spd.update(
        {
            idx + 1: [((0, 0, 0), float(rate))]
            for idx, rate in enumerate(recharge)
        }
    )
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data=rch_spd)

    face2 = config.cell_size_m * config.layer2_thickness_m
    cond2 = max(params.k2 * face2 / config.cell_size_m * params.ghb_mult, 1.0e-8)
    ghb_spd = {
        idx: [((1, 0, 0), float(params.h_ref), float(cond2))]
        for idx in range(nper)
    }
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_spd)

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="virtual_aquifer.hds",
        budget_filerecord="virtual_aquifer.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def _grid_cell_centers(config: SpatialAquiferConfig) -> np.ndarray:
    rows, cols = np.indices(config.shape)
    return np.column_stack(
        [
            (cols.ravel() + 0.5) * config.cell_size_m,
            (rows.ravel() + 0.5) * config.cell_size_m,
        ]
    )


def spatial_exponential_covariance(config: SpatialAquiferConfig) -> np.ndarray:
    centers = _grid_cell_centers(config)
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.sqrt(np.sum(delta**2, axis=2))
    return (config.std_ln_k**2) * np.exp(-distance / config.corr_len_m)


def build_spatial_logk_field(
    config: SpatialAquiferConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a reproducible log-K field in ln(m/s) with exact sample scaling."""
    covariance = spatial_exponential_covariance(config)
    sample = rng.multivariate_normal(
        mean=np.zeros(config.n_cells),
        cov=covariance + 1.0e-10 * np.eye(config.n_cells),
        method="svd",
    )
    sample = sample - sample.mean()
    sample_std = float(sample.std(ddof=1))
    if sample_std <= 0.0:
        raise ValueError("Degenerate spatial log-K sample.")
    sample = sample / sample_std * config.std_ln_k
    return (config.mean_ln_k_m_per_s + sample).reshape(config.shape)


def west_chd_period_data(
    config: SpatialAquiferConfig,
) -> list[tuple[tuple[int, int, int], float]]:
    return [((0, row, 0), float(config.west_chd_head_m)) for row in range(config.nrow)]


def uniform_recharge_period_data(
    config: SpatialAquiferConfig,
    recharge_m_per_day: float,
) -> list[tuple[tuple[int, int, int], float]]:
    return [
        ((0, row, col), float(recharge_m_per_day))
        for row in range(config.nrow)
        for col in range(config.ncol)
    ]


def center_observation_cell(config: SpatialAquiferConfig) -> tuple[int, int]:
    return (config.nrow // 2, config.ncol // 2)


def spatial_theta_to_fields(
    theta: np.ndarray,
    config: SpatialAquiferConfig,
) -> tuple[np.ndarray, float]:
    """Convert [ln K(cell, m/s), ln recharge multiplier] to MF6-ready fields."""
    theta = np.asarray(theta, dtype=float)
    expected = config.n_cells + 1
    if theta.shape != (expected,):
        raise ValueError(f"Expected theta shape {(expected,)}, got {theta.shape}.")
    k_m_per_day = np.exp(theta[: config.n_cells]).reshape(config.shape) * 86400.0
    recharge_m_per_day = config.default_recharge_m_per_day * float(np.exp(theta[-1]))
    return k_m_per_day, recharge_m_per_day


def build_10x10_model(
    model_ws: Path,
    k_m_per_day: np.ndarray,
    recharge_m_per_day: float | np.ndarray,
    config: SpatialAquiferConfig,
    sy: float | np.ndarray | None = None,
) -> flopy.mf6.MFSimulation:
    """Build a steady 10x10 MODFLOW6 model for spatial-K audit runs."""
    k_m_per_day = np.asarray(k_m_per_day, dtype=float)
    if k_m_per_day.shape != config.shape:
        raise ValueError(f"Expected K field shape {config.shape}, got {k_m_per_day.shape}.")
    if np.any(k_m_per_day <= 0.0):
        raise ValueError("K field must be strictly positive.")
    sy_values = config.specific_yield if sy is None else sy
    sy_array = np.asarray(sy_values, dtype=float)
    if sy_array.shape == ():
        sy_for_mf6: float | np.ndarray = float(sy_array)
    elif sy_array.shape == config.shape:
        sy_for_mf6 = sy_array.reshape((1, config.nrow, config.ncol))
    else:
        raise ValueError(f"Expected sy scalar or shape {config.shape}, got {sy_array.shape}.")
    if np.any(np.asarray(sy_for_mf6) <= 0.0):
        raise ValueError("Specific yield must be strictly positive.")
    recharge_series = np.atleast_1d(np.asarray(recharge_m_per_day, dtype=float))
    if recharge_series.ndim != 1:
        raise ValueError("recharge_m_per_day must be a scalar or one-dimensional series.")
    if np.any(recharge_series < 0.0):
        raise ValueError("Recharge must be non-negative.")
    nper = len(recharge_series)
    is_transient = nper > 1

    model_ws = Path(model_ws)
    exe_name = str(config.mf6_exe or find_mf6_executable())
    sim = flopy.mf6.MFSimulation(
        sim_name="spatial_aquifer",
        version="mf6",
        exe_name=exe_name,
        sim_ws=str(model_ws),
    )
    flopy.mf6.ModflowTdis(
        sim,
        nper=nper,
        perioddata=[(float(config.period_days), 1, 1.0)] * nper,
        time_units="DAYS",
    )
    flopy.mf6.ModflowIms(
        sim,
        complexity="SIMPLE",
        outer_maximum=config.outer_maximum,
        inner_maximum=config.inner_maximum,
        outer_dvclose=config.hclose,
        inner_dvclose=config.hclose,
        rcloserecord=[config.rclose, "STRICT"],
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname="spatial_aquifer",
        save_flows=True,
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=config.nrow,
        ncol=config.ncol,
        delr=config.cell_size_m,
        delc=config.cell_size_m,
        top=config.top_m,
        botm=[config.bottom_m],
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        icelltype=1 if is_transient else 0,
        k=k_m_per_day.reshape((1, config.nrow, config.ncol)),
    )
    if is_transient:
        flopy.mf6.ModflowGwfsto(
            gwf,
            save_flows=True,
            iconvert=1,
            ss=config.specific_storage_m_inv,
            sy=sy_for_mf6,
            steady_state={0: False},
            transient={idx: True for idx in range(nper)},
        )
    flopy.mf6.ModflowGwfic(gwf, strt=float(config.base_head_m))
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data={idx: west_chd_period_data(config) for idx in range(nper)},
    )
    flopy.mf6.ModflowGwfrch(
        gwf,
        stress_period_data={
            idx: uniform_recharge_period_data(config, float(rate))
            for idx, rate in enumerate(recharge_series)
        },
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="spatial_aquifer.hds",
        budget_filerecord="spatial_aquifer.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def run_10x10_forward(
    model_ws: Path,
    k_m_per_day: np.ndarray,
    recharge_m_per_day: float | np.ndarray,
    config: SpatialAquiferConfig,
) -> np.ndarray:
    sim = build_10x10_model(model_ws, k_m_per_day, recharge_m_per_day, config)
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    if not success:
        raise RuntimeError(f"MF6 failed for 10x10 run in {model_ws}")
    hds = flopy.utils.HeadFile(str(Path(model_ws) / "spatial_aquifer.hds"))
    head = hds.get_data(kstpkper=(0, 0))[0]
    return np.array([head[row, col] for row, col in config.observation_cells], dtype=float)


def run_10x10_center_hydrograph(
    model_ws: Path,
    k_m_per_day: np.ndarray,
    recharge_m_per_day: np.ndarray,
    config: SpatialAquiferConfig,
    sy: float | np.ndarray | None = None,
) -> np.ndarray:
    recharge_series = np.asarray(recharge_m_per_day, dtype=float)
    if recharge_series.ndim != 1:
        raise ValueError("recharge_m_per_day must be a one-dimensional daily series.")
    sim = build_10x10_model(model_ws, k_m_per_day, recharge_series, config, sy=sy)
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    if not success:
        raise RuntimeError(f"MF6 failed for 10x10 transient run in {model_ws}")
    hds = flopy.utils.HeadFile(str(Path(model_ws) / "spatial_aquifer.hds"))
    row, col = center_observation_cell(config)
    return np.array(
        [hds.get_data(kstpkper=(0, idx))[0, row, col] for idx in range(len(recharge_series))],
        dtype=float,
    )


def run_virtual_aquifer(
    model_ws: Path,
    site: SiteSeries,
    params: VirtualAquiferParams,
    config: VirtualAquiferConfig,
) -> tuple[bool, dict]:
    sim = build_virtual_aquifer_sim(model_ws, site, params, config)
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    if not success:
        return False, {}

    hds = flopy.utils.HeadFile(str(model_ws / "virtual_aquifer.hds"))
    layer_index = 1 if site.material_class == "암반" else 0
    steady_head = float(hds.get_data(kstpkper=(0, 0))[layer_index, 0, 0])
    pred = np.array(
        [
            hds.get_data(kstpkper=(0, idx + 1))[layer_index, 0, 0]
            for idx in range(len(site.dates))
        ]
    )
    valid = site.obs_valid_mask
    rmse = float(np.sqrt(np.mean((pred[valid] - site.obs_raw[valid]) ** 2)))
    bias = float(np.mean(pred[valid] - site.obs_raw[valid]))
    corr = float(np.corrcoef(pred[valid], site.obs_raw[valid])[0, 1])
    ss_res = float(np.sum((pred[valid] - site.obs_raw[valid]) ** 2))
    ss_tot = float(np.sum((site.obs_raw[valid] - site.obs_raw[valid].mean()) ** 2))
    nse = float(1.0 - ss_res / (ss_tot + 1.0e-12))
    payload = {
        "pred_head": pred,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "nse": nse,
        "params": params.to_dict(),
        "steady_head": steady_head,
        "dates": site.dates.astype("datetime64[D]").astype(str),
        "obs": site.obs_raw,
        "valid_mask": valid.astype(int),
    }
    return True, payload


def default_prior_for_site(site: SiteSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if site.archetype == "coastal":
        h_ref_mean = 0.0
        h_ref_std = 2.0
    else:
        h_ref_mean = float(np.nanquantile(site.obs_interp, 0.10))
        h_ref_std = 3.0

    mean = np.array(
        [
            np.log(10.0),
            np.log(1.0),
            np.log(0.08),
            np.log(1.0),
            h_ref_mean,
        ],
        dtype=float,
    )
    std = np.array([0.9, 0.9, 0.5, 0.5, h_ref_std], dtype=float)
    lower = np.array([np.log(0.1), np.log(0.01), np.log(0.01), np.log(0.05), h_ref_mean - 10.0])
    upper = np.array([np.log(300.0), np.log(50.0), np.log(0.35), np.log(20.0), h_ref_mean + 10.0])
    return mean, std, lower, upper


def default_prior_reduced_for_site(
    site: SiteSeries,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if site.archetype == "coastal":
        h_ref_mean = 0.0
        h_ref_std = 2.0
    else:
        h_ref_mean = float(np.nanquantile(site.obs_interp, 0.10))
        h_ref_std = 3.0

    mean = np.array(
        [
            np.log(5.0),
            np.log(0.08),
            h_ref_mean,
        ],
        dtype=float,
    )
    std = np.array([0.8, 0.5, h_ref_std], dtype=float)
    lower = np.array([np.log(0.1), np.log(0.01), h_ref_mean - 10.0], dtype=float)
    upper = np.array([np.log(100.0), np.log(0.35), h_ref_mean + 10.0], dtype=float)
    return mean, std, lower, upper


def save_forward_payload(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array_payload = {}
    for key, value in payload.items():
        if key == "params":
            continue
        array_payload[key] = np.asarray(value)
    np.savez(output_path, **array_payload)
    meta = output_path.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "rmse": payload["rmse"],
                "bias": payload["bias"],
                "corr": payload["corr"],
                "nse": payload["nse"],
                "params": payload["params"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

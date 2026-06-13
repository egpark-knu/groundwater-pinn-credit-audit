from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.esmda import esmda_update  # noqa: E402
from groundwater_research.virtual_aquifer import (  # noqa: E402
    SpatialAquiferConfig,
    build_recharge_series,
    build_spatial_logk_field,
    center_observation_cell,
    load_site_series,
    run_10x10_center_hydrograph,
    suggest_archetype_from_catalog,
)


def parse_alpha_seq(text: str, n_assim: int) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("alpha sequence must contain at least one value")
    if len(values) < n_assim:
        values.extend([values[-1]] * (n_assim - len(values)))
    return values[:n_assim]


def rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


def nse_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    residual = predicted - observed
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1.0e-12))


def dual_objective(
    predicted: np.ndarray,
    observed: np.ndarray,
    w_mse: float = 1.0,
    w_nse: float = 1.0,
) -> float:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    mse = float(np.mean((predicted - observed) ** 2))
    return float(w_mse * mse + w_nse * (1.0 - nse_score(predicted, observed)))


def ensemble_fit_summary(
    predictions: np.ndarray,
    observed: np.ndarray,
    w_mse: float = 1.0,
    w_nse: float = 1.0,
) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=float)
    rmse_values = np.array([rmse(row, observed) for row in predictions], dtype=float)
    nse_values = np.array([nse_score(row, observed) for row in predictions], dtype=float)
    dual_values = np.array([dual_objective(row, observed, w_mse=w_mse, w_nse=w_nse) for row in predictions], dtype=float)
    return {
        "mean_member_hydrograph_rmse": float(rmse_values.mean()),
        "best_member_hydrograph_rmse": float(rmse_values.min()),
        "mean_member_hydrograph_nse": float(nse_values.mean()),
        "best_member_hydrograph_nse": float(nse_values.max()),
        "mean_member_dual_objective": float(dual_values.mean()),
        "best_member_dual_objective": float(dual_values.min()),
    }


def augment_observation_for_rmse_nse(
    predicted: np.ndarray,
    observed: np.ndarray,
    w_mse: float = 1.0,
    w_nse: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment heads with a standardized trajectory component for ES-MDA."""

    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predicted.ndim != 2:
        raise ValueError(f"Expected predicted shape (n_ensemble, n_obs), got {predicted.shape}.")
    if observed.ndim != 1:
        raise ValueError(f"Expected observed shape (n_obs,), got {observed.shape}.")
    if predicted.shape[1] != observed.shape[0]:
        raise ValueError(f"Prediction/observation length mismatch: {predicted.shape[1]} vs {observed.shape[0]}.")
    obs_mu = float(observed.mean())
    obs_sd = float(observed.std() + 1.0e-12)
    amp_scale = float(np.sqrt(max(w_mse, 0.0)))
    shape_scale = float(np.sqrt(max(w_nse, 0.0)))
    pred_aug = np.concatenate(
        [
            amp_scale * predicted,
            shape_scale * ((predicted - obs_mu) / obs_sd),
        ],
        axis=1,
    )
    obs_aug = np.concatenate(
        [
            amp_scale * observed,
            shape_scale * ((observed - obs_mu) / obs_sd),
        ]
    )
    return pred_aug, obs_aug


def observation_operator(
    predicted: np.ndarray,
    observed: np.ndarray,
    mode: str = "absolute",
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predicted.ndim != 2:
        raise ValueError(f"Expected predicted shape (n_ensemble, n_obs), got {predicted.shape}.")
    if observed.ndim != 1:
        raise ValueError(f"Expected observed shape (n_obs,), got {observed.shape}.")
    if predicted.shape[1] != observed.shape[0]:
        raise ValueError(f"Prediction/observation length mismatch: {predicted.shape[1]} vs {observed.shape[0]}.")
    if mode == "absolute":
        return predicted, observed
    if mode == "delta":
        return np.diff(predicted, axis=1), np.diff(observed)
    if mode == "baseline":
        return predicted - predicted[:, :1], observed - observed[0]
    raise ValueError(f"Unsupported observation mode: {mode}")


def transformed_fit_summary(
    predictions: np.ndarray,
    observed: np.ndarray,
    mode: str,
    w_mse: float = 1.0,
    w_nse: float = 1.0,
) -> dict[str, float]:
    pred_op, obs_op = observation_operator(predictions, observed, mode=mode)
    summary = ensemble_fit_summary(pred_op, obs_op, w_mse=w_mse, w_nse=w_nse)
    return {f"{mode}_{key}": value for key, value in summary.items()}


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x, ddof=1) <= 0.0 or np.std(y, ddof=1) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def english_site_label(stem: str) -> str:
    labels = {
        "영덕영해_암반": "Yeongdeok Yeonghae, bedrock",
        "영덕달산_암반": "Yeongdeok Dalsan, bedrock",
    }
    return labels.get(stem, stem)


def select_site_window(
    dates: np.ndarray,
    obs: np.ndarray,
    rain_mm: np.ndarray,
    valid_mask: np.ndarray,
    start: str,
    end: str,
) -> dict[str, np.ndarray]:
    date_days = np.asarray(dates).astype("datetime64[D]")
    mask = (date_days >= np.datetime64(start)) & (date_days <= np.datetime64(end))
    if int(mask.sum()) == 0:
        raise ValueError(f"Empty real-data inverse window: {start} to {end}")
    return {
        "dates": date_days[mask],
        "obs": np.asarray(obs, dtype=float)[mask],
        "rain_mm": np.asarray(rain_mm, dtype=float)[mask],
        "valid_mask": np.asarray(valid_mask, dtype=bool)[mask],
    }


def derive_real_aquifer_config(
    obs: np.ndarray,
    epsilon_m: float = 0.05,
    cell_size_m: float = 100.0,
    corr_len_cells: float = 5.0,
) -> tuple[SpatialAquiferConfig, float, float]:
    finite_obs = np.asarray(obs, dtype=float)
    finite_obs = finite_obs[np.isfinite(finite_obs)]
    if finite_obs.size < 2:
        raise ValueError("Need at least two finite observations for CHD derivation.")
    west_chd = float(np.quantile(finite_obs, 0.10))
    base_head = west_chd + float(epsilon_m)
    config = SpatialAquiferConfig(
        base_head_m=base_head,
        west_chd_epsilon_m=float(epsilon_m),
        cell_size_m=float(cell_size_m),
        corr_len_cells=float(corr_len_cells),
        observation_cells=(center_observation_cell(SpatialAquiferConfig()),),
    )
    return config, base_head, west_chd


def standardized_spatial_field(config: SpatialAquiferConfig, rng: np.random.Generator) -> np.ndarray:
    raw = build_spatial_logk_field(config, rng)
    return (raw - config.mean_ln_k_m_per_s) / config.std_ln_k


def build_spatial_lnsy_field(
    config: SpatialAquiferConfig,
    rng: np.random.Generator,
    mean_ln_sy: float,
    std_ln_sy: float,
) -> np.ndarray:
    field = mean_ln_sy + std_ln_sy * standardized_spatial_field(config, rng)
    return np.clip(field, np.log(0.01), np.log(0.35))


def theta_to_k_sy_fields(theta_row: np.ndarray, config: SpatialAquiferConfig) -> tuple[np.ndarray, np.ndarray]:
    theta_row = np.asarray(theta_row, dtype=float)
    expected = config.n_cells * 2
    if theta_row.shape != (expected,):
        raise ValueError(f"Expected theta shape {(expected,)}, got {theta_row.shape}.")
    k_m_per_day = np.exp(theta_row[: config.n_cells]).reshape(config.shape) * 86400.0
    sy = np.exp(theta_row[config.n_cells :]).reshape(config.shape)
    return k_m_per_day, sy


def sample_spatial_theta_ensemble(
    config: SpatialAquiferConfig,
    n_ensemble: int,
    rng: np.random.Generator,
    mean_ln_sy: float,
    std_ln_sy: float,
) -> np.ndarray:
    theta = np.empty((n_ensemble, config.n_cells * 2), dtype=float)
    for member in range(n_ensemble):
        theta[member, : config.n_cells] = build_spatial_logk_field(config, rng).ravel()
        theta[member, config.n_cells :] = build_spatial_lnsy_field(
            config,
            rng,
            mean_ln_sy=mean_ln_sy,
            std_ln_sy=std_ln_sy,
        ).ravel()
    return theta


def make_bounds(config: SpatialAquiferConfig) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(config.n_cells * 2, dtype=float)
    upper = np.empty(config.n_cells * 2, dtype=float)
    lower[: config.n_cells] = np.log(1.0e-6)
    upper[: config.n_cells] = np.log(1.0e-2)
    lower[config.n_cells :] = np.log(0.01)
    upper[config.n_cells :] = np.log(0.35)
    return lower, upper


def _forward_worker(
    args: tuple[int, np.ndarray, np.ndarray, SpatialAquiferConfig, str, int],
) -> tuple[int, np.ndarray]:
    member, theta_row, fixed_recharge, config, worker_root_text, assim = args
    model_ws = Path(worker_root_text) / f"assim_{assim:02d}" / f"member_{member:03d}"
    if model_ws.exists():
        shutil.rmtree(model_ws)
    k_m_per_day, sy = theta_to_k_sy_fields(theta_row, config)
    hydrograph = run_10x10_center_hydrograph(
        model_ws,
        k_m_per_day,
        fixed_recharge,
        config,
        sy=sy,
    )
    shutil.rmtree(model_ws, ignore_errors=True)
    return member, hydrograph


def run_ensemble_forward(
    theta: np.ndarray,
    fixed_recharge: np.ndarray,
    config: SpatialAquiferConfig,
    worker_root: Path,
    assim: int,
    parallel: int,
) -> np.ndarray:
    args = [
        (member, theta[member].copy(), fixed_recharge.copy(), config, str(worker_root), assim)
        for member in range(theta.shape[0])
    ]
    predictions = np.empty((theta.shape[0], len(fixed_recharge)), dtype=float)
    if parallel <= 1:
        for item in args:
            member, hydrograph = _forward_worker(item)
            predictions[member] = hydrograph
        return predictions
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        for member, hydrograph in executor.map(_forward_worker, args):
            predictions[member] = hydrograph
    return predictions


def posterior_metrics(config: SpatialAquiferConfig, theta_post: np.ndarray) -> dict[str, float]:
    ln_k = theta_post[:, : config.n_cells]
    ln_sy = theta_post[:, config.n_cells :]
    post_k_mean = ln_k.mean(axis=0).reshape(config.shape)
    post_sy_mean = ln_sy.mean(axis=0).reshape(config.shape)
    post_k_std = ln_k.std(axis=0, ddof=1).reshape(config.shape)
    post_sy_std = ln_sy.std(axis=0, ddof=1).reshape(config.shape)
    same_cell_corr = np.array([safe_corr(ln_k[:, idx], ln_sy[:, idx]) for idx in range(config.n_cells)])
    row, col = center_observation_cell(config)
    return {
        "posterior_center_ln_k_mean": float(post_k_mean[row, col]),
        "posterior_center_ln_sy_mean": float(post_sy_mean[row, col]),
        "posterior_mean_ln_k_std": float(np.mean(post_k_std)),
        "posterior_mean_ln_sy_std": float(np.mean(post_sy_std)),
        "posterior_geometric_mean_k_m_per_s": float(np.exp(post_k_mean.mean())),
        "posterior_arithmetic_mean_k_m_per_s": float(np.mean(np.exp(post_k_mean))),
        "posterior_geometric_mean_sy": float(np.exp(post_sy_mean.mean())),
        "posterior_corr_mean_ln_k_vs_mean_ln_sy": safe_corr(ln_k.mean(axis=1), ln_sy.mean(axis=1)),
        "posterior_cellwise_k_sy_corr_mean": float(np.nanmean(same_cell_corr)),
        "posterior_cellwise_k_sy_corr_min": float(np.nanmin(same_cell_corr)),
        "posterior_cellwise_k_sy_corr_max": float(np.nanmax(same_cell_corr)),
        "posterior_cellwise_abs_k_sy_corr_mean": float(np.nanmean(np.abs(same_cell_corr))),
        "ode_scalar_k_ambiguity_note": (
            "The ODE-loss scalar k has no unique mapping to center-cell K, arithmetic/geometric/"
            "harmonic field summaries, or any identifiable spatial K-Sy compensation pattern."
        ),
    }


def plot_real_hydrograph(
    out_root: Path,
    site_label: str,
    dates: np.ndarray,
    observed: np.ndarray,
    prior_pred: np.ndarray,
    post_pred: np.ndarray,
    fixed_recharge: np.ndarray,
) -> None:
    x = pd.to_datetime(dates.astype("datetime64[D]").astype(str))
    fig, ax = plt.subplots(figsize=(12.8, 5.2), constrained_layout=True)
    ax2 = ax.twinx()
    ax2.bar(x, fixed_recharge * 1000.0, color="lightgray", width=1.0, alpha=0.45, label="fixed recharge")
    p10, p90 = np.percentile(prior_pred, [10, 90], axis=0)
    q10, q90 = np.percentile(post_pred, [10, 90], axis=0)
    ax.fill_between(x, p10, p90, color="0.55", alpha=0.18, label="prior ensemble P10-P90")
    ax.fill_between(x, q10, q90, color="#1f77b4", alpha=0.18, label="posterior ensemble P10-P90")
    ax.scatter(x, observed, s=10, color="black", alpha=0.42, label="observed head")
    ax.plot(x, post_pred.mean(axis=0), color="#1f77b4", lw=1.7, label="posterior mean")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{site_label} head (m)")
    ax2.set_ylabel("Fixed recharge (mm/day)")
    ax.set_title(f"Real single-well hydrograph fitting: {site_label}")
    ax.grid(alpha=0.18)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.savefig(out_root / "fig_real_hydrograph_fit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_real_posterior_fields(
    out_root: Path,
    site_label: str,
    config: SpatialAquiferConfig,
    theta_post: np.ndarray,
) -> None:
    ln_k = theta_post[:, : config.n_cells]
    ln_sy = theta_post[:, config.n_cells :]
    post_k_mean = ln_k.mean(axis=0).reshape(config.shape)
    post_k_std = ln_k.std(axis=0, ddof=1).reshape(config.shape)
    post_sy_mean = ln_sy.mean(axis=0).reshape(config.shape)
    post_sy_std = ln_sy.std(axis=0, ddof=1).reshape(config.shape)
    k_sy_corr = np.array([safe_corr(ln_k[:, idx], ln_sy[:, idx]) for idx in range(config.n_cells)]).reshape(config.shape)
    row, col = center_observation_cell(config)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True)
    panels = [
        (post_k_mean, "Posterior mean ln K", "viridis"),
        (post_k_std, "Posterior ln K uncertainty", "magma"),
        (np.exp(post_k_mean), "Posterior K (m/s)", "viridis"),
        (post_sy_mean, "Posterior mean ln Sy", "cividis"),
        (post_sy_std, "Posterior ln Sy uncertainty", "magma"),
        (k_sy_corr, "Posterior corr lnK-lnSy", "coolwarm"),
    ]
    for ax, (field, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(field, origin="lower", cmap=cmap)
        ax.scatter(col, row, marker="x", s=70, c="white", linewidths=2.0)
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"Real inverse posterior fields from one {site_label} hydrograph", fontsize=14)
    fig.savefig(out_root / "fig_real_posterior_k_sy_fields.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    out_root: Path,
    site_stem: str,
    site_label: str,
    dates: np.ndarray,
    observed: np.ndarray,
    valid_mask: np.ndarray,
    rain_mm: np.ndarray,
    fixed_recharge: np.ndarray,
    config: SpatialAquiferConfig,
    base_head: float,
    west_chd: float,
    theta_prior: np.ndarray,
    theta_post: np.ndarray,
    prior_pred: np.ndarray,
    post_pred: np.ndarray,
    history: list[dict],
    elapsed_s: float,
    objective_w_mse: float,
    objective_w_nse: float,
    observation_mode: str,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    ln_k = theta_post[:, : config.n_cells]
    ln_sy = theta_post[:, config.n_cells :]
    np.save(out_root / "dates.npy", dates.astype("datetime64[D]"))
    np.save(out_root / "observed_hydrograph.npy", observed)
    np.save(out_root / "valid_mask.npy", valid_mask.astype(bool))
    np.save(out_root / "rain_mm.npy", rain_mm)
    np.save(out_root / "fixed_recharge_m_per_day.npy", fixed_recharge)
    np.save(out_root / "theta_prior.npy", theta_prior)
    np.save(out_root / "theta_posterior.npy", theta_post)
    np.save(out_root / "posterior_ln_k_mean.npy", ln_k.mean(axis=0).reshape(config.shape))
    np.save(out_root / "posterior_ln_k_std.npy", ln_k.std(axis=0, ddof=1).reshape(config.shape))
    np.save(out_root / "posterior_ln_sy_mean.npy", ln_sy.mean(axis=0).reshape(config.shape))
    np.save(out_root / "posterior_ln_sy_std.npy", ln_sy.std(axis=0, ddof=1).reshape(config.shape))
    np.save(out_root / "prior_hydrograph_ensemble.npy", prior_pred)
    np.save(out_root / "posterior_hydrograph_ensemble.npy", post_pred)
    pd.DataFrame(history).to_csv(out_root / "esmda_history.csv", index=False)
    pd.DataFrame(
        {
            "parameter": [f"ln_k_cell_{idx:03d}" for idx in range(config.n_cells)]
            + [f"ln_sy_cell_{idx:03d}" for idx in range(config.n_cells)],
            "prior_mean": theta_prior.mean(axis=0),
            "prior_std": theta_prior.std(axis=0, ddof=1),
            "posterior_mean": theta_post.mean(axis=0),
            "posterior_std": theta_post.std(axis=0, ddof=1),
        }
    ).to_csv(out_root / "parameter_summary.csv", index=False)

    metrics = {
        "prior_mean_hydrograph_rmse": rmse(prior_pred.mean(axis=0), observed),
        "prior_mean_hydrograph_nse": nse_score(prior_pred.mean(axis=0), observed),
        "prior_mean_dual_objective": dual_objective(
            prior_pred.mean(axis=0),
            observed,
            w_mse=objective_w_mse,
            w_nse=objective_w_nse,
        ),
        "posterior_mean_hydrograph_rmse": rmse(post_pred.mean(axis=0), observed),
        "posterior_mean_hydrograph_nse": nse_score(post_pred.mean(axis=0), observed),
        "posterior_mean_dual_objective": dual_objective(
            post_pred.mean(axis=0),
            observed,
            w_mse=objective_w_mse,
            w_nse=objective_w_nse,
        ),
        "best_posterior_member_hydrograph_rmse": float(np.min([rmse(row, observed) for row in post_pred])),
        "best_posterior_member_hydrograph_nse": float(np.max([nse_score(row, observed) for row in post_pred])),
        "best_posterior_member_dual_objective": float(
            np.min([dual_objective(row, observed, w_mse=objective_w_mse, w_nse=objective_w_nse) for row in post_pred])
        ),
        **{
            f"prior_mean_{observation_mode}_rmse": rmse(
                observation_operator(prior_pred, observed, mode=observation_mode)[0].mean(axis=0),
                observation_operator(prior_pred, observed, mode=observation_mode)[1],
            ),
            f"prior_mean_{observation_mode}_nse": nse_score(
                observation_operator(prior_pred, observed, mode=observation_mode)[0].mean(axis=0),
                observation_operator(prior_pred, observed, mode=observation_mode)[1],
            ),
            f"prior_mean_{observation_mode}_dual_objective": dual_objective(
                observation_operator(prior_pred, observed, mode=observation_mode)[0].mean(axis=0),
                observation_operator(prior_pred, observed, mode=observation_mode)[1],
                w_mse=objective_w_mse,
                w_nse=objective_w_nse,
            ),
            f"posterior_mean_{observation_mode}_rmse": rmse(
                observation_operator(post_pred, observed, mode=observation_mode)[0].mean(axis=0),
                observation_operator(post_pred, observed, mode=observation_mode)[1],
            ),
            f"posterior_mean_{observation_mode}_nse": nse_score(
                observation_operator(post_pred, observed, mode=observation_mode)[0].mean(axis=0),
                observation_operator(post_pred, observed, mode=observation_mode)[1],
            ),
            f"posterior_mean_{observation_mode}_dual_objective": dual_objective(
                observation_operator(post_pred, observed, mode=observation_mode)[0].mean(axis=0),
                observation_operator(post_pred, observed, mode=observation_mode)[1],
                w_mse=objective_w_mse,
                w_nse=objective_w_nse,
            ),
        },
        "observed_head_std_m": float(np.std(observed, ddof=1)),
        "observed_head_range_m": float(np.ptp(observed)),
        "rain_total_mm": float(np.sum(rain_mm)),
        "fixed_recharge_total_m": float(np.sum(fixed_recharge)),
        "elapsed_s": float(elapsed_s),
        **posterior_metrics(config, theta_post),
    }
    summary = {
        "purpose": (
            f"Real single-well inverse audit: fit the {site_label} observed hydrograph "
            "with the same task form as DL models, while returning posterior K(x,y) and Sy(x,y)."
        ),
        "framing_guardrail": "No synthetic truth and no forecast horse race; parameter-quality audit only.",
        "site": {
            "stem": site_stem,
            "english_label": site_label,
            "window_start": str(dates[0]),
            "window_end": str(dates[-1]),
            "n_days": int(len(dates)),
            "missing_or_interpolated_days": int((~valid_mask).sum()),
        },
        "domain": {
            "nrow": config.nrow,
            "ncol": config.ncol,
            "n_cells": config.n_cells,
            "cell_size_m": config.cell_size_m,
            "domain_size_m": float(config.ncol * config.cell_size_m),
            "corr_len_cells": float(config.corr_len_cells),
            "corr_len_m": float(config.corr_len_m),
            "well_cell": list(center_observation_cell(config)),
            "base_head_m": float(base_head),
            "west_chd_head_m": float(west_chd),
            "east_boundary": "no-flow",
            "north_south_boundary": "no-flow",
            "recharge_policy": "fixed_RPR_0.20_times_observed_daily_precipitation",
            "recharge_estimated": False,
        },
        "esmda": {
            "n_ensemble": int(theta_prior.shape[0]),
            "n_assimilation": int(len([row for row in history if row["posterior_state"] == "pre_update"])),
            "state_dimension": int(theta_prior.shape[1]),
            "state_components": ["lnK_100_cells", "lnSy_100_cells"],
            "observation_dimension": int(len(observed)),
            "observation_mode": observation_mode,
            "observation_dimension_transformed": int(len(observation_operator(post_pred, observed, mode=observation_mode)[1])),
            "observation_objective": f"dual_{observation_mode}_and_standardized_{observation_mode}_augmented_es_mda",
            "dual_objective_formula": "w_mse*MSE + w_nse*(1-NSE)",
            "objective_w_mse": float(objective_w_mse),
            "objective_w_nse": float(objective_w_nse),
        },
        "history": history,
        "metrics": metrics,
    }
    summary = sanitize_for_json(summary)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    readme = f"""# Real Yeongdeok Single-Well Solver Audit

This is the real-data replacement for the synthetic 10x10 toy problem.

## Contract

- Well: {site_label} (`{site_stem}`).
- Task: fit one observed head hydrograph at the center cell.
- Recharge: fixed RPR = 0.20 x observed daily precipitation; recharge is not estimated.
- State: 100 ln(K) cells + 100 ln(Sy) cells.
- Cell size: {config.cell_size_m:.1f} m; domain width: {config.ncol * config.cell_size_m:.1f} m.
- Correlation length: {config.corr_len_cells:.1f} cells = {config.corr_len_m:.1f} m.
- CHD west: 10th percentile observed head for the selected window.
- No-flow: east, north, south.
- ES-MDA observation mode: `{observation_mode}`.
- ES-MDA observation: augmented transformed vector `[y, standardized y]`, where y is defined by the observation mode.
- Audit objective: {objective_w_mse:.3g} * MSE + {objective_w_nse:.3g} * (1 - NSE).

## Current Result

- Prior mean hydrograph RMSE: {metrics['prior_mean_hydrograph_rmse']:.4f} m.
- Posterior mean hydrograph RMSE: {metrics['posterior_mean_hydrograph_rmse']:.4f} m.
- Posterior mean hydrograph NSE: {metrics['posterior_mean_hydrograph_nse']:.4f}.
- Posterior mean dual objective: {metrics['posterior_mean_dual_objective']:.6f}.
- Posterior mean `{observation_mode}` RMSE: {metrics[f'posterior_mean_{observation_mode}_rmse']:.6f}.
- Posterior mean `{observation_mode}` NSE: {metrics[f'posterior_mean_{observation_mode}_nse']:.6f}.
- Best posterior member RMSE: {metrics['best_posterior_member_hydrograph_rmse']:.4f} m.
- Observed head std: {metrics['observed_head_std_m']:.4f} m.
- Mean posterior ln(K) uncertainty: {metrics['posterior_mean_ln_k_std']:.4f}.
- Mean posterior ln(Sy) uncertainty: {metrics['posterior_mean_ln_sy_std']:.4f}.
- Mean absolute cell-wise K-Sy compensation corr: {metrics['posterior_cellwise_abs_k_sy_corr_mean']:.4f}.

## Interpretation

No true K field exists for this real inverse. The valid comparison is therefore not whether K is "correct", but whether the same hydrograph-fitting task produces auditable parameter objects. MODFLOW6+ES-MDA produces posterior K(x,y), Sy(x,y), uncertainty, and K-Sy compensation diagnostics; ODE-loss gives scalar parameters with no unique field-scale interpretation.
"""
    (out_root / "README.md").write_text(readme)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="영덕영해_암반")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--nens", type=int, default=100)
    ap.add_argument("--n-assim", type=int, default=4)
    ap.add_argument("--alpha-seq", default="9.33,7,4,2")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--obs-error", type=float, default=0.05)
    ap.add_argument("--objective-w-mse", type=float, default=1.0)
    ap.add_argument("--objective-w-nse", type=float, default=1.0)
    ap.add_argument("--observation-mode", choices=["absolute", "delta", "baseline"], default="absolute")
    ap.add_argument("--cell-size-m", type=float, default=100.0)
    ap.add_argument("--corr-len-cells", type=float, default=5.0)
    ap.add_argument("--mean-sy", type=float, default=0.08)
    ap.add_argument("--std-ln-sy", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=260410)
    ap.add_argument("--output-dir", default=str(ROOT / "results/solver_audit_real_yeongdeok"))
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    worker_root = out_root / "_mf6_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    archetype = suggest_archetype_from_catalog(args.stem) or "coastal"
    site_label = english_site_label(args.stem)
    site = load_site_series(args.stem, archetype=archetype)
    window = select_site_window(
        site.dates,
        site.obs_interp,
        site.rain_mm,
        site.obs_valid_mask,
        args.start,
        args.end,
    )
    config, base_head, west_chd = derive_real_aquifer_config(
        window["obs"],
        cell_size_m=args.cell_size_m,
        corr_len_cells=args.corr_len_cells,
    )
    fixed_recharge = build_recharge_series(window["rain_mm"], recharge_fraction=0.20, tau_days=1.0)
    alpha_seq = parse_alpha_seq(args.alpha_seq, args.n_assim)
    rng = np.random.default_rng(args.seed)
    mean_ln_sy = float(np.log(args.mean_sy))

    theta = sample_spatial_theta_ensemble(config, args.nens, rng, mean_ln_sy, args.std_ln_sy)
    theta_prior = theta.copy()
    lower, upper = make_bounds(config)
    history: list[dict] = []
    start_time = time.time()

    prior_pred = run_ensemble_forward(theta, fixed_recharge, config, worker_root, 0, args.parallel)
    history.append(
        {
            "assimilation": 0,
            "alpha": None,
            **ensemble_fit_summary(
                prior_pred,
                window["obs"],
                w_mse=args.objective_w_mse,
                w_nse=args.objective_w_nse,
            ),
            **transformed_fit_summary(
                prior_pred,
                window["obs"],
                mode=args.observation_mode,
                w_mse=args.objective_w_mse,
                w_nse=args.objective_w_nse,
            ),
            "posterior_state": "prior",
        }
    )
    for assim, alpha in enumerate(alpha_seq, start=1):
        preds = run_ensemble_forward(theta, fixed_recharge, config, worker_root, assim, args.parallel)
        history.append(
            {
                "assimilation": assim,
                "alpha": float(alpha),
                **ensemble_fit_summary(
                    preds,
                    window["obs"],
                    w_mse=args.objective_w_mse,
                    w_nse=args.objective_w_nse,
                ),
                **transformed_fit_summary(
                    preds,
                    window["obs"],
                    mode=args.observation_mode,
                    w_mse=args.objective_w_mse,
                    w_nse=args.objective_w_nse,
                ),
                "posterior_state": "pre_update",
            }
        )
        pred_op, obs_op = observation_operator(preds, window["obs"], mode=args.observation_mode)
        pred_aug, obs_aug = augment_observation_for_rmse_nse(
            pred_op,
            obs_op,
            w_mse=args.objective_w_mse,
            w_nse=args.objective_w_nse,
        )
        theta = esmda_update(
            theta=theta,
            predicted=pred_aug,
            observed=obs_aug,
            alpha=float(alpha),
            obs_error_std=args.obs_error,
            rng=rng,
            lower=lower,
            upper=upper,
        )

    post_pred = run_ensemble_forward(theta, fixed_recharge, config, worker_root, args.n_assim + 1, args.parallel)
    history.append(
        {
            "assimilation": args.n_assim + 1,
            "alpha": None,
            **ensemble_fit_summary(
                post_pred,
                window["obs"],
                w_mse=args.objective_w_mse,
                w_nse=args.objective_w_nse,
            ),
            **transformed_fit_summary(
                post_pred,
                window["obs"],
                mode=args.observation_mode,
                w_mse=args.objective_w_mse,
                w_nse=args.objective_w_nse,
            ),
            "posterior_state": "final",
        }
    )
    elapsed_s = time.time() - start_time

    summary = write_outputs(
        out_root=out_root,
        site_stem=args.stem,
        site_label=site_label,
        dates=window["dates"],
        observed=window["obs"],
        valid_mask=window["valid_mask"],
        rain_mm=window["rain_mm"],
        fixed_recharge=fixed_recharge,
        config=config,
        base_head=base_head,
        west_chd=west_chd,
        theta_prior=theta_prior,
        theta_post=theta,
        prior_pred=prior_pred,
        post_pred=post_pred,
        history=history,
        elapsed_s=elapsed_s,
        objective_w_mse=args.objective_w_mse,
        objective_w_nse=args.objective_w_nse,
        observation_mode=args.observation_mode,
    )
    plot_real_hydrograph(out_root, site_label, window["dates"], window["obs"], prior_pred, post_pred, fixed_recharge)
    plot_real_posterior_fields(out_root, site_label, config, theta)
    shutil.rmtree(worker_root, ignore_errors=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

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
    build_spatial_logk_field,
    center_observation_cell,
    run_10x10_center_hydrograph,
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


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x, ddof=1) <= 0.0 or np.std(y, ddof=1) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_synthetic_precip_m_per_day(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic daily precipitation; recharge is fixed externally at RPR=0.20."""
    days = np.arange(n_days)
    seasonal_prob = 0.18 + 0.12 * np.sin(2.0 * np.pi * (days - 90) / 365.0)
    storm = rng.random(n_days) < np.clip(seasonal_prob, 0.05, 0.35)
    precip_mm = np.zeros(n_days, dtype=float)
    precip_mm[storm] = rng.gamma(shape=1.8, scale=11.0, size=int(storm.sum()))
    for day in [32, 76, 129, 188, 241, 318]:
        if day < n_days:
            width = min(3, n_days - day)
            precip_mm[day : day + width] += np.linspace(28.0, 10.0, width)
    return precip_mm * 1.0e-3


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


def field_metrics(
    config: SpatialAquiferConfig,
    true_ln_k: np.ndarray,
    true_ln_sy: np.ndarray,
    theta_post: np.ndarray,
) -> dict[str, float]:
    post_ln_k = theta_post[:, : config.n_cells]
    post_ln_sy = theta_post[:, config.n_cells :]
    post_k_mean = post_ln_k.mean(axis=0).reshape(config.shape)
    post_sy_mean = post_ln_sy.mean(axis=0).reshape(config.shape)
    k_residual = post_k_mean.ravel() - true_ln_k.ravel()
    sy_residual = post_sy_mean.ravel() - true_ln_sy.ravel()
    center_row, center_col = center_observation_cell(config)
    true_center = float(true_ln_k[center_row, center_col])
    post_center = float(post_k_mean[center_row, center_col])
    true_geom = float(true_ln_k.mean())
    true_arith = float(np.log(np.mean(np.exp(true_ln_k))))
    true_harm = float(-np.log(np.mean(1.0 / np.exp(true_ln_k))))
    return {
        "ln_k_spatial_corr": safe_corr(true_ln_k.ravel(), post_k_mean.ravel()),
        "ln_k_rmse": float(np.sqrt(np.mean(k_residual**2))),
        "ln_k_mae": float(np.mean(np.abs(k_residual))),
        "ln_sy_spatial_corr": safe_corr(true_ln_sy.ravel(), post_sy_mean.ravel()),
        "ln_sy_rmse": float(np.sqrt(np.mean(sy_residual**2))),
        "true_center_ln_k": true_center,
        "posterior_center_ln_k_mean": post_center,
        "center_ln_k_abs_error": abs(post_center - true_center),
        "true_geometric_mean_ln_k": true_geom,
        "true_arithmetic_mean_ln_k": true_arith,
        "true_harmonic_mean_ln_k": true_harm,
        "posterior_geometric_mean_k_m_per_s": float(np.exp(post_k_mean.mean())),
        "posterior_arithmetic_mean_k_m_per_s": float(np.mean(np.exp(post_k_mean))),
        "ode_scalar_k_ambiguity_note": (
            "The scalar ODE-loss k cannot identify whether it corresponds to center-cell K, "
            "geometric mean K, arithmetic mean K, harmonic mean K, or another effective summary."
        ),
    }


def compensation_metrics(config: SpatialAquiferConfig, theta_post: np.ndarray) -> dict[str, float]:
    ln_k = theta_post[:, : config.n_cells]
    ln_sy = theta_post[:, config.n_cells :]
    same_cell_corr = np.array(
        [safe_corr(ln_k[:, idx], ln_sy[:, idx]) for idx in range(config.n_cells)],
        dtype=float,
    )
    return {
        "posterior_corr_mean_ln_k_vs_mean_ln_sy": safe_corr(ln_k.mean(axis=1), ln_sy.mean(axis=1)),
        "posterior_cellwise_k_sy_corr_mean": float(np.nanmean(same_cell_corr)),
        "posterior_cellwise_k_sy_corr_min": float(np.nanmin(same_cell_corr)),
        "posterior_cellwise_k_sy_corr_max": float(np.nanmax(same_cell_corr)),
        "posterior_cellwise_abs_k_sy_corr_mean": float(np.nanmean(np.abs(same_cell_corr))),
    }


def plot_hydrograph(
    out_root: Path,
    observed: np.ndarray,
    truth: np.ndarray,
    prior_pred: np.ndarray,
    post_pred: np.ndarray,
    fixed_recharge: np.ndarray,
) -> None:
    days = np.arange(len(observed))
    fig, ax = plt.subplots(figsize=(12.8, 5.2), constrained_layout=True)
    ax2 = ax.twinx()
    ax2.bar(days, fixed_recharge * 1000.0, color="lightgray", width=1.0, alpha=0.45, label="fixed recharge")
    p10, p90 = np.percentile(prior_pred, [10, 90], axis=0)
    q10, q90 = np.percentile(post_pred, [10, 90], axis=0)
    ax.fill_between(days, p10, p90, color="0.55", alpha=0.18, label="prior ensemble P10-P90")
    ax.fill_between(days, q10, q90, color="#1f77b4", alpha=0.18, label="posterior ensemble P10-P90")
    ax.plot(days, truth, color="black", lw=1.4, label="truth")
    ax.scatter(days, observed, s=8, color="black", alpha=0.35, label="synthetic observations")
    ax.plot(days, post_pred.mean(axis=0), color="#1f77b4", lw=1.6, label="posterior mean")
    ax.set_xlabel("Day")
    ax.set_ylabel("Center-well head (m)")
    ax2.set_ylabel("Recharge (mm/day)")
    ax.set_title("Single center-well transient hydrograph fitting by MODFLOW6 + ES-MDA")
    ax.grid(alpha=0.18)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.savefig(out_root / "fig_center_well_hydrograph_fit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_k_sy_fields(
    out_root: Path,
    config: SpatialAquiferConfig,
    true_ln_k: np.ndarray,
    true_ln_sy: np.ndarray,
    theta_post: np.ndarray,
) -> None:
    post_k_mean = theta_post[:, : config.n_cells].mean(axis=0).reshape(config.shape)
    post_k_std = theta_post[:, : config.n_cells].std(axis=0, ddof=1).reshape(config.shape)
    post_sy_mean = theta_post[:, config.n_cells :].mean(axis=0).reshape(config.shape)
    k_sy_corr = np.array(
        [safe_corr(theta_post[:, idx], theta_post[:, config.n_cells + idx]) for idx in range(config.n_cells)],
        dtype=float,
    ).reshape(config.shape)
    row, col = center_observation_cell(config)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True)
    panels = [
        (true_ln_k, "True ln K field", "viridis"),
        (post_k_mean, "Posterior mean ln K", "viridis"),
        (post_k_std, "Posterior ln K uncertainty", "magma"),
        (true_ln_sy, "True ln Sy field", "cividis"),
        (post_sy_mean, "Posterior mean ln Sy", "cividis"),
        (k_sy_corr, "Posterior corr lnK-lnSy", "coolwarm"),
    ]
    for ax, (field, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(field, origin="lower", cmap=cmap)
        ax.scatter(col, row, marker="x", s=70, c="white", linewidths=2.0)
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle("100-cell posterior K and Sy fields from one center-well hydrograph", fontsize=14)
    fig.savefig(out_root / "fig_spatial_k_sy_posterior_single_well.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ax2.scatter(true_ln_k.ravel(), post_k_mean.ravel(), s=25, alpha=0.75)
    lo = min(float(true_ln_k.min()), float(post_k_mean.min()))
    hi = max(float(true_ln_k.max()), float(post_k_mean.max()))
    ax2.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--")
    ax2.set_xlabel("True ln K")
    ax2.set_ylabel("Posterior mean ln K")
    ax2.set_title("Spatial K recovery from one hydrograph")
    ax2.grid(alpha=0.18)
    fig2.savefig(out_root / "fig_true_vs_posterior_ln_k_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig2)


def write_outputs(
    out_root: Path,
    config: SpatialAquiferConfig,
    fixed_recharge: np.ndarray,
    precip: np.ndarray,
    true_ln_k: np.ndarray,
    true_ln_sy: np.ndarray,
    observed: np.ndarray,
    truth: np.ndarray,
    theta_prior: np.ndarray,
    theta_post: np.ndarray,
    prior_pred: np.ndarray,
    post_pred: np.ndarray,
    history: list[dict],
    elapsed_s: float,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    post_k_mean = theta_post[:, : config.n_cells].mean(axis=0).reshape(config.shape)
    post_k_std = theta_post[:, : config.n_cells].std(axis=0, ddof=1).reshape(config.shape)
    post_sy_mean = theta_post[:, config.n_cells :].mean(axis=0).reshape(config.shape)
    post_sy_std = theta_post[:, config.n_cells :].std(axis=0, ddof=1).reshape(config.shape)

    np.save(out_root / "true_ln_k.npy", true_ln_k)
    np.save(out_root / "true_ln_sy.npy", true_ln_sy)
    np.save(out_root / "posterior_ln_k_mean.npy", post_k_mean)
    np.save(out_root / "posterior_ln_k_std.npy", post_k_std)
    np.save(out_root / "posterior_ln_sy_mean.npy", post_sy_mean)
    np.save(out_root / "posterior_ln_sy_std.npy", post_sy_std)
    np.save(out_root / "theta_prior.npy", theta_prior)
    np.save(out_root / "theta_posterior.npy", theta_post)
    np.save(out_root / "precip_m_per_day.npy", precip)
    np.save(out_root / "fixed_recharge_m_per_day.npy", fixed_recharge)
    np.save(out_root / "truth_hydrograph.npy", truth)
    np.save(out_root / "observed_hydrograph.npy", observed)
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

    hydrograph_metrics = {
        "prior_mean_hydrograph_rmse": rmse(prior_pred.mean(axis=0), observed),
        "posterior_mean_hydrograph_rmse": rmse(post_pred.mean(axis=0), observed),
        "best_posterior_member_hydrograph_rmse": float(np.min([rmse(row, observed) for row in post_pred])),
        "truth_vs_observed_rmse_noise_floor": rmse(truth, observed),
    }
    metrics = {
        **hydrograph_metrics,
        **field_metrics(config, true_ln_k, true_ln_sy, theta_post),
        **compensation_metrics(config, theta_post),
        "elapsed_s": float(elapsed_s),
    }
    summary = {
        "purpose": (
            "Single-center-well transient hydrograph fitting audit: DL and solver share the same "
            "hydrograph-matching task, but solver posterior remains 100-cell K and 100-cell Sy fields."
        ),
        "framing_guardrail": "This is a parameter-quality audit, not a solver-vs-DL forecast horse race.",
        "domain": {
            "nrow": config.nrow,
            "ncol": config.ncol,
            "n_cells": config.n_cells,
            "cell_size_m": config.cell_size_m,
            "well_cell": list(center_observation_cell(config)),
            "west_chd_head_m": config.west_chd_head_m,
            "east_boundary": "no-flow",
            "north_south_boundary": "no-flow",
            "n_days": int(len(observed)),
            "recharge_policy": "fixed_RPR_0.20_times_synthetic_daily_precipitation",
            "recharge_estimated": False,
        },
        "esmda": {
            "n_ensemble": int(theta_prior.shape[0]),
            "n_assimilation": int(len([row for row in history if row["posterior_state"] == "pre_update"])),
            "state_dimension": int(theta_prior.shape[1]),
            "state_components": ["lnK_100_cells", "lnSy_100_cells"],
            "observation_dimension": int(len(observed)),
        },
        "history": history,
        "metrics": metrics,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def write_readme(out_root: Path, summary: dict) -> None:
    metrics = summary["metrics"]
    text = f"""# 10x10 Single-Well Transient Solver Audit

This run supersedes the archived five-well campaign. The task is intentionally aligned with the DL forecasting task: fit one center-well transient head hydrograph.

## Contract

- Grid: 10 x 10 MODFLOW6 cells, 100 m cell size, one layer.
- Observation: one center cell, row 5 / col 5.
- Forcing: synthetic 365-day precipitation, recharge fixed as RPR 0.20 x precipitation.
- State: 100 cell-wise ln(K) parameters plus 100 cell-wise ln(Sy) parameters.
- Recharge is not estimated.
- ES-MDA: Ne = {summary['esmda']['n_ensemble']}, Na = {summary['esmda']['n_assimilation']}.

## Result

- Prior mean hydrograph RMSE: {metrics['prior_mean_hydrograph_rmse']:.4f} m.
- Posterior mean hydrograph RMSE: {metrics['posterior_mean_hydrograph_rmse']:.4f} m.
- Posterior ln(K) spatial correlation with truth: {metrics['ln_k_spatial_corr']:.4f}.
- Posterior ln(K) RMSE: {metrics['ln_k_rmse']:.4f}.
- Posterior ln(Sy) spatial correlation with truth: {metrics['ln_sy_spatial_corr']:.4f}.
- Mean absolute cell-wise K-Sy compensation corr: {metrics['posterior_cellwise_abs_k_sy_corr_mean']:.4f}.

## Interpretation

The solver and DL models are both asked to match a single hydrograph. The difference is the physical-credit object: MODFLOW6+ES-MDA returns posterior K(x,y) and Sy(x,y) fields with uncertainty and compensation diagnostics; an ODE-loss forecaster returns scalar parameters whose relationship to cell-scale K/Sy remains ambiguous.
"""
    (out_root / "README.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nens", type=int, default=100)
    ap.add_argument("--n-assim", type=int, default=4)
    ap.add_argument("--alpha-seq", default="9.33,7,4,2")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--n-days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=260410)
    ap.add_argument("--obs-error", type=float, default=0.03)
    ap.add_argument("--mean-sy", type=float, default=0.08)
    ap.add_argument("--std-ln-sy", type=float, default=0.25)
    ap.add_argument("--output-dir", default=str(ROOT / "results/solver_audit_10x10_single_well"))
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    worker_root = out_root / "_mf6_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    config = SpatialAquiferConfig(obs_error_m=args.obs_error)
    rng = np.random.default_rng(args.seed)
    alpha_seq = parse_alpha_seq(args.alpha_seq, args.n_assim)
    precip = build_synthetic_precip_m_per_day(args.n_days, rng)
    fixed_recharge = config.recharge_fraction * precip
    mean_ln_sy = float(np.log(args.mean_sy))

    true_ln_k = build_spatial_logk_field(config, rng)
    true_ln_sy = build_spatial_lnsy_field(config, rng, mean_ln_sy, args.std_ln_sy)
    true_k, true_sy = theta_to_k_sy_fields(np.concatenate([true_ln_k.ravel(), true_ln_sy.ravel()]), config)
    truth = run_10x10_center_hydrograph(out_root / "truth_model", true_k, fixed_recharge, config, sy=true_sy)
    observed = truth + args.obs_error * rng.standard_normal(args.n_days)

    theta = sample_spatial_theta_ensemble(config, args.nens, rng, mean_ln_sy, args.std_ln_sy)
    theta_prior = theta.copy()
    lower, upper = make_bounds(config)
    history: list[dict] = []
    start = time.time()

    prior_pred = run_ensemble_forward(theta, fixed_recharge, config, worker_root, 0, args.parallel)
    history.append(
        {
            "assimilation": 0,
            "alpha": None,
            "mean_member_hydrograph_rmse": float(np.mean([rmse(row, observed) for row in prior_pred])),
            "best_member_hydrograph_rmse": float(np.min([rmse(row, observed) for row in prior_pred])),
            "posterior_state": "prior",
        }
    )

    for assim, alpha in enumerate(alpha_seq, start=1):
        preds = run_ensemble_forward(theta, fixed_recharge, config, worker_root, assim, args.parallel)
        member_rmse = np.array([rmse(preds[member], observed) for member in range(args.nens)])
        history.append(
            {
                "assimilation": assim,
                "alpha": float(alpha),
                "mean_member_hydrograph_rmse": float(member_rmse.mean()),
                "best_member_hydrograph_rmse": float(member_rmse.min()),
                "posterior_state": "pre_update",
            }
        )
        theta = esmda_update(
            theta=theta,
            predicted=preds,
            observed=observed,
            alpha=float(alpha),
            obs_error_std=args.obs_error,
            rng=rng,
            lower=lower,
            upper=upper,
        )

    post_pred = run_ensemble_forward(theta, fixed_recharge, config, worker_root, args.n_assim + 1, args.parallel)
    final_rmse = np.array([rmse(post_pred[member], observed) for member in range(args.nens)])
    history.append(
        {
            "assimilation": args.n_assim + 1,
            "alpha": None,
            "mean_member_hydrograph_rmse": float(final_rmse.mean()),
            "best_member_hydrograph_rmse": float(final_rmse.min()),
            "posterior_state": "final",
        }
    )
    elapsed = time.time() - start

    summary = write_outputs(
        out_root=out_root,
        config=config,
        fixed_recharge=fixed_recharge,
        precip=precip,
        true_ln_k=true_ln_k,
        true_ln_sy=true_ln_sy,
        observed=observed,
        truth=truth,
        theta_prior=theta_prior,
        theta_post=theta,
        prior_pred=prior_pred,
        post_pred=post_pred,
        history=history,
        elapsed_s=elapsed,
    )
    plot_hydrograph(out_root, observed, truth, prior_pred, post_pred, fixed_recharge)
    plot_k_sy_fields(out_root, config, true_ln_k, true_ln_sy, theta)
    write_readme(out_root, summary)
    shutil.rmtree(worker_root, ignore_errors=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

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
    run_10x10_forward,
    spatial_theta_to_fields,
)


PARAMETER_NAMES = [f"ln_k_cell_{idx:03d}" for idx in range(100)] + ["ln_recharge_multiplier"]


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


def sample_spatial_theta_ensemble(
    config: SpatialAquiferConfig,
    n_ensemble: int,
    rng: np.random.Generator,
    recharge_log_std: float,
) -> np.ndarray:
    theta = np.empty((n_ensemble, config.n_cells + 1), dtype=float)
    for member in range(n_ensemble):
        theta[member, : config.n_cells] = build_spatial_logk_field(config, rng).ravel()
        theta[member, -1] = recharge_log_std * rng.standard_normal()
    return theta


def make_bounds(config: SpatialAquiferConfig) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(config.n_cells + 1, np.log(1.0e-6), dtype=float)
    upper = np.full(config.n_cells + 1, np.log(1.0e-2), dtype=float)
    lower[-1] = np.log(0.2)
    upper[-1] = np.log(2.0)
    return lower, upper


def _forward_worker(args: tuple[int, np.ndarray, SpatialAquiferConfig, str, int]) -> tuple[int, np.ndarray]:
    member, theta_row, config, worker_root_text, assim = args
    worker_root = Path(worker_root_text)
    model_ws = worker_root / f"assim_{assim:02d}" / f"member_{member:03d}"
    if model_ws.exists():
        shutil.rmtree(model_ws)
    k_m_per_day, recharge_m_per_day = spatial_theta_to_fields(theta_row, config)
    obs = run_10x10_forward(model_ws, k_m_per_day, recharge_m_per_day, config)
    shutil.rmtree(model_ws, ignore_errors=True)
    return member, obs


def run_ensemble_forward(
    theta: np.ndarray,
    config: SpatialAquiferConfig,
    worker_root: Path,
    assim: int,
    parallel: int,
) -> np.ndarray:
    args = [
        (member, theta[member].copy(), config, str(worker_root), assim)
        for member in range(theta.shape[0])
    ]
    predictions = np.empty((theta.shape[0], len(config.observation_cells)), dtype=float)
    if parallel <= 1:
        for item in args:
            member, obs = _forward_worker(item)
            predictions[member] = obs
        return predictions

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        for member, obs in executor.map(_forward_worker, args):
            predictions[member] = obs
    return predictions


def field_metrics(true_ln_k: np.ndarray, estimated_ln_k: np.ndarray) -> dict[str, float]:
    truth = true_ln_k.ravel()
    estimate = estimated_ln_k.ravel()
    residual = estimate - truth
    corr = float(np.corrcoef(truth, estimate)[0, 1])
    return {
        "ln_k_rmse": float(np.sqrt(np.mean(residual**2))),
        "ln_k_mae": float(np.mean(np.abs(residual))),
        "ln_k_spatial_corr": corr,
        "ln_k_bias": float(np.mean(residual)),
    }


def recharge_compensation_metrics(theta_prior: np.ndarray, theta_post: np.ndarray) -> dict[str, float]:
    def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x, ddof=1) <= 0.0 or np.std(y, ddof=1) <= 0.0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    prior_ln_arithmetic_k = np.log(np.mean(np.exp(theta_prior[:, :-1]), axis=1))
    post_ln_arithmetic_k = np.log(np.mean(np.exp(theta_post[:, :-1]), axis=1))
    prior_corr_cells = np.array([safe_corr(theta_prior[:, idx], theta_prior[:, -1]) for idx in range(theta_prior.shape[1] - 1)])
    post_corr_cells = np.array([safe_corr(theta_post[:, idx], theta_post[:, -1]) for idx in range(theta_post.shape[1] - 1)])
    return {
        "prior_corr_ln_arithmetic_k_vs_ln_recharge": safe_corr(prior_ln_arithmetic_k, theta_prior[:, -1]),
        "posterior_corr_ln_arithmetic_k_vs_ln_recharge": safe_corr(post_ln_arithmetic_k, theta_post[:, -1]),
        "posterior_cellwise_corr_mean": float(np.nanmean(post_corr_cells)),
        "posterior_cellwise_corr_min": float(np.nanmin(post_corr_cells)),
        "posterior_cellwise_corr_max": float(np.nanmax(post_corr_cells)),
        "posterior_cellwise_abs_corr_mean": float(np.nanmean(np.abs(post_corr_cells))),
    }


def write_outputs(
    out_root: Path,
    config: SpatialAquiferConfig,
    true_ln_k: np.ndarray,
    theta_prior: np.ndarray,
    theta_post: np.ndarray,
    observed_heads: np.ndarray,
    truth_heads: np.ndarray,
    history: list[dict],
    elapsed_s: float,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    post_ln_k_mean = theta_post[:, : config.n_cells].mean(axis=0).reshape(config.shape)
    post_ln_k_std = theta_post[:, : config.n_cells].std(axis=0, ddof=1).reshape(config.shape)
    prior_ln_k_mean = theta_prior[:, : config.n_cells].mean(axis=0).reshape(config.shape)
    recharge_corr_map = np.array(
        [
            np.corrcoef(theta_post[:, idx], theta_post[:, -1])[0, 1]
            for idx in range(config.n_cells)
        ],
        dtype=float,
    ).reshape(config.shape)

    metrics = field_metrics(true_ln_k, post_ln_k_mean)
    metrics.update(recharge_compensation_metrics(theta_prior, theta_post))
    metrics.update(
        {
            "n_cells": config.n_cells,
            "n_obs_wells": len(config.observation_cells),
            "elapsed_s": float(elapsed_s),
            "ode_scalar_k_interpretation": (
                "A single ODE-loss k is comparable only to a scalar summary of the posterior field, "
                "not to the recovered 100-cell spatial K structure."
            ),
            "posterior_ln_k_mean_scalar_summary": float(post_ln_k_mean.mean()),
            "posterior_k_geometric_mean_m_per_s": float(np.exp(post_ln_k_mean.mean())),
            "posterior_k_arithmetic_mean_m_per_s": float(np.mean(np.exp(post_ln_k_mean))),
            "posterior_recharge_multiplier_mean": float(np.mean(np.exp(theta_post[:, -1]))),
            "posterior_recharge_multiplier_std": float(np.std(np.exp(theta_post[:, -1]), ddof=1)),
        }
    )

    np.save(out_root / "true_ln_k.npy", true_ln_k)
    np.save(out_root / "prior_ln_k_mean.npy", prior_ln_k_mean)
    np.save(out_root / "posterior_ln_k_mean.npy", post_ln_k_mean)
    np.save(out_root / "posterior_ln_k_std.npy", post_ln_k_std)
    np.save(out_root / "posterior_logk_recharge_corr.npy", recharge_corr_map)
    np.save(out_root / "theta_prior.npy", theta_prior)
    np.save(out_root / "theta_posterior.npy", theta_post)
    np.save(out_root / "observed_heads.npy", observed_heads)
    np.save(out_root / "truth_heads.npy", truth_heads)

    pd.DataFrame(history).to_csv(out_root / "esmda_history.csv", index=False)
    pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "prior_mean": theta_prior.mean(axis=0),
            "prior_std": theta_prior.std(axis=0, ddof=1),
            "posterior_mean": theta_post.mean(axis=0),
            "posterior_std": theta_post.std(axis=0, ddof=1),
        }
    ).to_csv(out_root / "parameter_summary.csv", index=False)

    summary = {
        "purpose": (
            "10x10 synthetic MODFLOW6 ES-MDA audit for parameter-quality contrast: "
            "spatial solver posterior K(x,y) versus scalar ODE-loss k."
        ),
        "framing_guardrail": "This is not a forecast-accuracy horse race.",
        "domain": {
            "nrow": config.nrow,
            "ncol": config.ncol,
            "cell_size_m": config.cell_size_m,
            "west_chd_head_m": config.west_chd_head_m,
            "east_boundary": "no-flow",
            "north_south_boundary": "no-flow",
            "recharge_policy": "RPR_0.20_times_synthetic_precipitation",
        },
        "observation_cells": [list(item) for item in config.observation_cells],
        "history": history,
        "metrics": metrics,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def plot_recovery(out_root: Path, true_ln_k: np.ndarray, theta_prior: np.ndarray, theta_post: np.ndarray, config: SpatialAquiferConfig) -> None:
    prior_mean = theta_prior[:, : config.n_cells].mean(axis=0).reshape(config.shape)
    post_mean = theta_post[:, : config.n_cells].mean(axis=0).reshape(config.shape)
    post_std = theta_post[:, : config.n_cells].std(axis=0, ddof=1).reshape(config.shape)
    error = post_mean - true_ln_k
    corr_map = np.array(
        [np.corrcoef(theta_post[:, idx], theta_post[:, -1])[0, 1] for idx in range(config.n_cells)],
        dtype=float,
    ).reshape(config.shape)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), constrained_layout=True)
    panels = [
        (true_ln_k, "True ln K field\n(m/s)", "viridis"),
        (prior_mean, "Prior ensemble mean\nln K", "viridis"),
        (post_mean, "Posterior ensemble mean\nln K", "viridis"),
        (post_std, "Posterior ln K std\nuncertainty", "magma"),
        (error, "Posterior - truth\nln K error", "coolwarm"),
        (corr_map, "Posterior corr\nln K(cell), ln R mult", "coolwarm"),
    ]
    for ax, (field, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(field, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        for row, col in config.observation_cells:
            ax.scatter(col, row, marker="x", s=44, c="white", linewidths=1.5)
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle("10x10 virtual aquifer ES-MDA: spatial K recovery and compensation", fontsize=14)
    fig.savefig(out_root / "fig_spatial_k_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_compensation(out_root: Path, theta_prior: np.ndarray, theta_post: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    ax_scatter, ax_hist = axes
    prior_ln_arithmetic_k = np.log(np.mean(np.exp(theta_prior[:, :-1]), axis=1))
    post_ln_arithmetic_k = np.log(np.mean(np.exp(theta_post[:, :-1]), axis=1))
    ax_scatter.scatter(prior_ln_arithmetic_k, theta_prior[:, -1], s=28, facecolors="none", edgecolors="0.45", label="prior")
    ax_scatter.scatter(post_ln_arithmetic_k, theta_post[:, -1], s=34, color="#1f77b4", alpha=0.82, label="posterior")
    ax_scatter.set_xlabel("ln arithmetic mean K across 100 cells")
    ax_scatter.set_ylabel("ln recharge multiplier")
    ax_scatter.set_title("K-recharge compensation in ensemble space")
    ax_scatter.grid(alpha=0.18)
    ax_scatter.legend(frameon=False)

    cell_corr = np.array(
        [np.corrcoef(theta_post[:, idx], theta_post[:, -1])[0, 1] for idx in range(theta_post.shape[1] - 1)],
        dtype=float,
    )
    ax_hist.hist(cell_corr[np.isfinite(cell_corr)], bins=16, color="#2ca02c", alpha=0.78)
    ax_hist.axvline(0.0, color="black", lw=0.9)
    ax_hist.set_xlabel("corr(ln K cell, ln recharge multiplier)")
    ax_hist.set_ylabel("Cell count")
    ax_hist.set_title("Cell-wise posterior compensation")
    ax_hist.grid(alpha=0.18)
    fig.savefig(out_root / "fig_parameter_compensation_10x10.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raise SystemExit(
        "Deprecated design: the 5-well/recharge-multiplier 10x10 audit was archived. "
        "Use scripts/run_solver_audit_10x10_single_well.py for the approved fixed-RPR "
        "single-center-well K+Sy audit."
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--nens", type=int, default=50)
    ap.add_argument("--n-assim", type=int, default=4)
    ap.add_argument("--alpha-seq", default="9.33,7,4,2")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=260410)
    ap.add_argument("--obs-error", type=float, default=0.05)
    ap.add_argument("--recharge-log-std", type=float, default=0.35)
    ap.add_argument("--output-dir", default=str(ROOT / "results/solver_audit_10x10"))
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

    true_ln_k = build_spatial_logk_field(config, rng)
    true_theta = np.concatenate([true_ln_k.ravel(), np.array([0.0])])
    true_k_m_per_day, true_recharge = spatial_theta_to_fields(true_theta, config)
    truth_heads = run_10x10_forward(out_root / "truth_model", true_k_m_per_day, true_recharge, config)
    observed_heads = truth_heads + args.obs_error * rng.standard_normal(len(truth_heads))

    theta = sample_spatial_theta_ensemble(config, args.nens, rng, args.recharge_log_std)
    theta_prior = theta.copy()
    lower, upper = make_bounds(config)
    history: list[dict] = []
    start = time.time()

    for assim, alpha in enumerate(alpha_seq, start=1):
        preds = run_ensemble_forward(theta, config, worker_root, assim, args.parallel)
        member_rmse = np.array([rmse(preds[member], observed_heads) for member in range(args.nens)])
        history.append(
            {
                "assimilation": assim,
                "alpha": float(alpha),
                "mean_member_obs_rmse": float(member_rmse.mean()),
                "best_member_obs_rmse": float(member_rmse.min()),
                "posterior_state": "pre_update",
            }
        )
        theta = esmda_update(
            theta=theta,
            predicted=preds,
            observed=observed_heads,
            alpha=float(alpha),
            obs_error_std=args.obs_error,
            rng=rng,
            lower=lower,
            upper=upper,
        )

    final_preds = run_ensemble_forward(theta, config, worker_root, args.n_assim + 1, args.parallel)
    final_rmse = np.array([rmse(final_preds[member], observed_heads) for member in range(args.nens)])
    history.append(
        {
            "assimilation": args.n_assim + 1,
            "alpha": None,
            "mean_member_obs_rmse": float(final_rmse.mean()),
            "best_member_obs_rmse": float(final_rmse.min()),
            "posterior_state": "final",
        }
    )
    elapsed = time.time() - start

    summary = write_outputs(
        out_root=out_root,
        config=config,
        true_ln_k=true_ln_k,
        theta_prior=theta_prior,
        theta_post=theta,
        observed_heads=observed_heads,
        truth_heads=truth_heads,
        history=history,
        elapsed_s=elapsed,
    )
    plot_recovery(out_root, true_ln_k, theta_prior, theta, config)
    plot_compensation(out_root, theta_prior, theta)
    shutil.rmtree(worker_root, ignore_errors=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

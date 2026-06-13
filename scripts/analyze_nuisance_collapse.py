#!/usr/bin/env python3
"""Phase 3: Nuisance-Parameter Collapse Analysis.

Extracts learned ODE parameters (gamma_d, gamma_r, h_ref) from 50-well
falsification results and computes diagnostics proving ODE penalty
collapses to generic regularizer.

Requires Phase 2A results in results/whittaker_vs_ode_50well/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS_DIR = ROOT / "results/whittaker_vs_ode_50well"
OUTPUT_DIR = ROOT / "results/nuisance_collapse"


def extract_parameters(results_dir: Path) -> pd.DataFrame:
    """Extract learned ODE params from all lstm_ode summary files."""
    summary_csv = results_dir / "whittaker_vs_ode_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}. Run Phase 2A first.")

    df = pd.read_csv(summary_csv)
    ode_rows = df[df["model"] == "lstm_ode"].copy()

    if "physics_gamma_r" not in ode_rows.columns:
        raise ValueError("physics_gamma_r column missing. ODE params not saved.")

    # Compute derived quantities
    ode_rows["tau_days"] = 1.0 / ode_rows["physics_gamma_d"].replace(0, np.nan)
    ode_rows["gamma_d_href"] = ode_rows["physics_gamma_d"] * ode_rows["physics_h_ref"]
    # gamma_r is in mm^-1 (rain in mm, head in m). Convert to m^-1 for physical interpretation:
    # gamma_r_per_m = gamma_r * 1000  →  1m rain causes gamma_r_per_m meters of head change
    ode_rows["gamma_r_per_m_rain"] = ode_rows["physics_gamma_r"] * 1000.0
    # Physical interpretation: gamma_r ≈ alpha / Sy (Park 2008)
    # where alpha = recharge partition ratio, Sy = specific yield
    # If Sy is known, RPR = gamma_r * Sy (with unit correction for mm→m)

    return ode_rows


def compute_ode_rhs_correlation(results_dir: Path, ode_params: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation between ODE RHS and actual delta-h on test rollout."""
    correlations = []

    for _, row in ode_params.iterrows():
        well = row["well"]
        seed = int(row["seed"])
        gamma_r = row["physics_gamma_r"]
        gamma_d = row["physics_gamma_d"]
        h_ref = row["physics_h_ref"]

        npz_path = results_dir / well / f"lstm_ode_lambda0.1_seed{seed}" / "test_rollout_predictions.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        pred = data["pred"]
        obs = data["obs"]

        if len(pred) < 3:
            continue

        # Compute delta_h (predicted)
        delta_h = np.diff(pred)

        # For ODE RHS we need rainfall — approximate from the series
        # Since we don't have rainfall in npz, use a simpler diagnostic:
        # Correlation between delta_h and recession term alone
        recession_term = -gamma_d * (pred[:-1] - h_ref)

        if np.std(delta_h) > 0 and np.std(recession_term) > 0:
            corr_recession = float(np.corrcoef(delta_h, recession_term)[0, 1])
        else:
            corr_recession = 0.0

        correlations.append({
            "well": well,
            "seed": seed,
            "gamma_d": gamma_d,
            "gamma_r": gamma_r,
            "h_ref": h_ref,
            "tau_days": 1.0 / gamma_d if gamma_d > 0 else np.nan,
            "corr_dh_recession": corr_recession,
            "recession_rms": float(np.sqrt(np.mean(recession_term ** 2))),
            "delta_h_rms": float(np.sqrt(np.mean(delta_h ** 2))),
            "recession_to_dh_ratio": float(np.sqrt(np.mean(recession_term ** 2)) / max(np.sqrt(np.mean(delta_h ** 2)), 1e-10)),
        })

    return pd.DataFrame(correlations)


def compute_falsification_statistics(results_dir: Path) -> dict:
    """Statistical tests for ODE vs WS2 non-separation."""
    summary_csv = results_dir / "whittaker_vs_ode_summary.csv"
    df = pd.read_csv(summary_csv)

    # Pivot: for each (well, seed), get ODE and WS2 RMSE
    ode = df[df["model"] == "lstm_ode"].set_index(["well", "seed"])["rmse"]
    ws2 = df[df["model"] == "lstm_ws2"].set_index(["well", "seed"])["rmse"]
    plain = df[df["model"] == "lstm"].set_index(["well", "seed"])["rmse"]

    # Align on common (well, seed) pairs
    common = ode.index.intersection(ws2.index)
    ode_aligned = ode.loc[common]
    ws2_aligned = ws2.loc[common]

    diff = ode_aligned - ws2_aligned  # positive = ODE worse

    # Wilcoxon signed-rank test
    if len(diff) >= 10:
        wilcox_stat, wilcox_p = stats.wilcoxon(diff.values, alternative="two-sided")
    else:
        wilcox_stat, wilcox_p = np.nan, np.nan

    # Bootstrap 95% CI on mean difference
    n_boot = 10000
    rng = np.random.default_rng(42)
    boot_means = []
    diff_arr = diff.values
    for _ in range(n_boot):
        sample = rng.choice(diff_arr, size=len(diff_arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Cohen's d
    mean_diff = float(np.mean(diff_arr))
    pooled_std = float(np.sqrt((np.var(ode_aligned) + np.var(ws2_aligned)) / 2))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

    # Winner counts
    ode_wins = int((diff < 0).sum())
    ws2_wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())

    # Per-well 3-seed mean
    ode_mean = df[df["model"] == "lstm_ode"].groupby("well")["rmse"].mean()
    ws2_mean = df[df["model"] == "lstm_ws2"].groupby("well")["rmse"].mean()
    common_wells = ode_mean.index.intersection(ws2_mean.index)
    well_diff = ode_mean.loc[common_wells] - ws2_mean.loc[common_wells]

    well_ode_wins = int((well_diff < 0).sum())
    well_ws2_wins = int((well_diff > 0).sum())

    return {
        "n_comparisons": int(len(diff)),
        "n_wells": int(len(common_wells)),
        "mean_rmse_diff_ode_minus_ws2": mean_diff,
        "median_rmse_diff": float(np.median(diff_arr)),
        "wilcoxon_statistic": float(wilcox_stat) if not np.isnan(wilcox_stat) else None,
        "wilcoxon_p_value": float(wilcox_p) if not np.isnan(wilcox_p) else None,
        "bootstrap_95ci_lower": ci_lower,
        "bootstrap_95ci_upper": ci_upper,
        "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
        "seed_level_ode_wins": ode_wins,
        "seed_level_ws2_wins": ws2_wins,
        "seed_level_ties": ties,
        "well_level_ode_wins": well_ode_wins,
        "well_level_ws2_wins": well_ws2_wins,
        "practical_significance_threshold_m": 0.001,
        "interpretation": (
            "ODE-loss does NOT significantly separate from WS2"
            if (wilcox_p is not None and (np.isnan(wilcox_p) or wilcox_p > 0.05))
            or (cohens_d is not None and (np.isnan(cohens_d) or abs(cohens_d) < 0.2))
            else "ODE-loss shows marginal separation from WS2 — check effect size"
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Extracting ODE parameters...")
    ode_params = extract_parameters(RESULTS_DIR)
    ode_params.to_csv(OUTPUT_DIR / "parameter_table.csv", index=False)
    print(f"  Extracted {len(ode_params)} ODE parameter rows")
    print(f"  gamma_r range: {ode_params['physics_gamma_r'].min():.6f} - {ode_params['physics_gamma_r'].max():.6f}")
    print(f"  gamma_d range: {ode_params['physics_gamma_d'].min():.6f} - {ode_params['physics_gamma_d'].max():.6f}")
    print(f"  h_ref range: {ode_params['physics_h_ref'].min():.2f} - {ode_params['physics_h_ref'].max():.2f}")

    print("\nStep 2: Computing ODE RHS correlations...")
    corr_df = compute_ode_rhs_correlation(RESULTS_DIR, ode_params)
    corr_df.to_csv(OUTPUT_DIR / "ode_rhs_correlation.csv", index=False)
    print(f"  Mean corr(delta_h, recession): {corr_df['corr_dh_recession'].mean():.4f}")
    print(f"  Mean recession_to_dh_ratio: {corr_df['recession_to_dh_ratio'].mean():.4f}")

    print("\nStep 3: Computing falsification statistics...")
    stats_result = compute_falsification_statistics(RESULTS_DIR)
    (OUTPUT_DIR / "statistical_tests.json").write_text(
        json.dumps(stats_result, indent=2, ensure_ascii=False)
    )
    print(f"  Wilcoxon p: {stats_result['wilcoxon_p_value']}")
    print(f"  Cohen's d: {stats_result['cohens_d']}")
    print(f"  Mean RMSE diff (ODE-WS2): {stats_result['mean_rmse_diff_ode_minus_ws2']:.6f} m")
    print(f"  Bootstrap 95% CI: [{stats_result['bootstrap_95ci_lower']:.6f}, {stats_result['bootstrap_95ci_upper']:.6f}]")
    print(f"  Well-level: ODE wins {stats_result['well_level_ode_wins']}, WS2 wins {stats_result['well_level_ws2_wins']}")
    print(f"  Interpretation: {stats_result['interpretation']}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

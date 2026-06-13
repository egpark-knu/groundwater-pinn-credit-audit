from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


VARIANT_ORDER = ["persistence", "ode_only", "gru", "ws2", "ode"]
PAIRED_COMPARISONS = [
    ("ode_vs_gru", "ode", "gru"),
    ("ws2_vs_gru", "ws2", "gru"),
    ("ode_vs_ws2", "ode", "ws2"),
    ("ode_only_vs_persistence", "ode_only", "persistence"),
    ("ode_vs_persistence", "ode", "persistence"),
    ("gru_vs_persistence", "gru", "persistence"),
]
LOWER_BETTER = {"nrmse_std", "nrmse_range", "rmse"}
HIGHER_BETTER = {"skill_vs_persistence_rmse", "nse", "corr"}
BOOTSTRAP_SEED = 20260611
BOOTSTRAP_ITERATIONS = 10000
KOREAN_RE = re.compile(r"[\uac00-\ud7a3]")


def normalize_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value))


def sort_variants(df: pd.DataFrame, column: str = "variant") -> pd.DataFrame:
    order = {variant: idx for idx, variant in enumerate(VARIANT_ORDER)}
    return (
        df.assign(_variant_order=df[column].map(order).fillna(len(order)))
        .sort_values(["_variant_order", column])
        .drop(columns=["_variant_order"])
        .reset_index(drop=True)
    )


def assert_row_alignment(raw: pd.DataFrame, anon: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(raw) != len(anon):
        return [f"row count mismatch: raw={len(raw)} anon={len(anon)}"]
    text_cols = ["horizon", "variant", "requested_variant", "seed"]
    number_cols = [
        "test_rollout_rmse",
        "test_rollout_nse",
        "test_rollout_corr",
        "test_final_rmse",
        "test_final_nse",
        "best_val_loss",
    ]
    for column in text_cols:
        if not raw[column].astype(str).equals(anon[column].astype(str)):
            errors.append(f"row alignment mismatch in {column}")
    for column in number_cols:
        left = pd.to_numeric(raw[column], errors="coerce")
        right = pd.to_numeric(anon[column], errors="coerce")
        if not np.allclose(left, right, rtol=1.0e-12, atol=1.0e-12, equal_nan=True):
            errors.append(f"row alignment mismatch in {column}")
    return errors


def no_identifier_leak(path: Path) -> tuple[bool, list[str]]:
    text = path.read_text(encoding="utf-8")
    leaks: list[str] = []
    if KOREAN_RE.search(text):
        leaks.append("korean_codepoint")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if "stem" in first_line:
        leaks.append("stem_column")
    return not leaks, leaks


def metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("nrmse_std", "size"),
            n_wells=("well_label", "nunique"),
            rmse_mean=("rmse", "mean"),
            nrmse_std_mean=("nrmse_std", "mean"),
            nrmse_std_median=("nrmse_std", "median"),
            nrmse_range_mean=("nrmse_range", "mean"),
            nrmse_range_median=("nrmse_range", "median"),
            skill_vs_persistence_mean=("skill_vs_persistence_rmse", "mean"),
            skill_vs_persistence_median=("skill_vs_persistence_rmse", "median"),
            nse_mean=("nse", "mean"),
            corr_mean=("corr", "mean"),
        )
        .reset_index()
    )
    if group_cols == ["variant"]:
        return out.sort_values("nrmse_std_mean").reset_index(drop=True)
    return out.sort_values(group_cols[:-1] + ["nrmse_std_mean"]).reset_index(drop=True)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = [
        (idx, float(p))
        for idx, p in enumerate(p_values)
        if not pd.isna(p)
    ]
    adjusted = [math.nan] * len(p_values)
    if not indexed:
        return adjusted
    m = len(indexed)
    running = 0.0
    for rank, (idx, p_value) in enumerate(sorted(indexed, key=lambda item: item[1]), start=1):
        candidate = min(1.0, (m - rank + 1) * p_value)
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def paired_stats(well_mean: pd.DataFrame, metric: str) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", well_mean)]
    for material, group in well_mean.groupby("material_group", dropna=False):
        scopes.append(("material_group", str(material), group))

    rows = []
    for scope_type, scope_value, scope_df in scopes:
        wide = (
            scope_df.pivot_table(
                index=["well_label", "material_group"],
                columns="variant",
                values=metric,
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        for comparison, treatment, comparator in PAIRED_COMPARISONS:
            paired = wide[["well_label", treatment, comparator]].dropna()
            diff = (paired[treatment] - paired[comparator]).to_numpy(dtype=float)
            n = int(len(diff))
            if n == 0:
                rows.append(
                    {
                        "metric": metric,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "comparison": comparison,
                        "treatment": treatment,
                        "comparator": comparator,
                        "n_pairs": 0,
                        "status": "no_pairs",
                    }
                )
                continue

            resampled = diff[rng.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))]
            mean_boot = resampled.mean(axis=1)
            nonzero = diff[np.abs(diff) > 1.0e-12]
            if len(nonzero) == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            if metric in LOWER_BETTER:
                treatment_better = diff < -1.0e-12
                treatment_worse = diff > 1.0e-12
                if diff.mean() < 0:
                    direction = "treatment_better"
                elif diff.mean() > 0:
                    direction = "comparator_better"
                else:
                    direction = "tie"
            elif metric in HIGHER_BETTER:
                treatment_better = diff > 1.0e-12
                treatment_worse = diff < -1.0e-12
                if diff.mean() > 0:
                    direction = "treatment_better"
                elif diff.mean() < 0:
                    direction = "comparator_better"
                else:
                    direction = "tie"
            else:
                raise ValueError(f"unknown metric direction: {metric}")

            n_treatment_better = int(treatment_better.sum())
            n_treatment_worse = int(treatment_worse.sum())
            rows.append(
                {
                    "metric": metric,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "comparison": comparison,
                    "treatment": treatment,
                    "comparator": comparator,
                    "n_pairs": n,
                    "treatment_mean": float(paired[treatment].mean()),
                    "comparator_mean": float(paired[comparator].mean()),
                    "diff_mean": float(diff.mean()),
                    "diff_median": float(np.median(diff)),
                    "diff_mean_bootstrap_ci95_low": float(np.quantile(mean_boot, 0.025)),
                    "diff_mean_bootstrap_ci95_high": float(np.quantile(mean_boot, 0.975)),
                    "n_treatment_better": n_treatment_better,
                    "n_treatment_worse": n_treatment_worse,
                    "n_ties": int(n - n_treatment_better - n_treatment_worse),
                    "fraction_treatment_better": float(n_treatment_better / n),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_two_sided": p_value,
                    "direction_by_mean": direction,
                    "status": "ok",
                }
            )

    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    adjusted_parts = []
    for (metric_name, scope_type, scope_value), group in stats.groupby(
        ["metric", "scope_type", "scope_value"], dropna=False
    ):
        group = group.copy()
        group["wilcoxon_p_holm_scope"] = holm_adjust(group["wilcoxon_p_two_sided"].tolist())
        adjusted_parts.append(group)
    return pd.concat(adjusted_parts, ignore_index=True)


def write_markdown(
    path: Path,
    overall: pd.DataFrame,
    by_material: pd.DataFrame,
    paired: pd.DataFrame,
    verification: dict,
) -> None:
    def md_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
        frame = df[columns].copy()
        if n is not None:
            frame = frame.head(n)
        for column in frame.columns:
            if pd.api.types.is_float_dtype(frame[column]):
                frame[column] = frame[column].map(lambda value: f"{value:.4f}")
        return frame.to_markdown(index=False)

    key_pairs = paired[
        (paired["metric"].isin(["nrmse_std", "skill_vs_persistence_rmse", "nse"]))
        & (paired["scope_type"].eq("overall"))
        & (paired["comparison"].isin(["ode_vs_gru", "ode_vs_ws2", "ws2_vs_gru"]))
    ].copy()

    lines = [
        "# Scale-Independent Metric Audit",
        "",
        "This audit extends the clean canonical80 central matched-arm matrix with scale-independent metrics for reviewer concerns about RMSE scale dependence.",
        "",
        "## Verification",
        "",
        f"- Status: `{verification['status']}`",
        f"- Raw rows: {verification['raw_rows']}",
        f"- Seed rows written: {verification['seed_rows']}",
        f"- Well-mean rows written: {verification['well_mean_rows']}",
        f"- Expected wells: {verification['expected_wells']}",
        f"- Identifier leak check: `{verification['identifier_leak_check']}`",
        "",
        "## Overall Variant Summary",
        "",
        md_table(
            overall,
            [
                "variant",
                "n",
                "n_wells",
                "rmse_mean",
                "nrmse_std_mean",
                "nrmse_range_mean",
                "skill_vs_persistence_mean",
                "nse_mean",
                "corr_mean",
            ],
        ),
        "",
        "## Material-Group Summary",
        "",
        md_table(
            by_material,
            [
                "material_group",
                "variant",
                "n_wells",
                "nrmse_std_mean",
                "skill_vs_persistence_mean",
                "nse_mean",
                "corr_mean",
            ],
        ),
        "",
        "## Key Paired Contrasts At Well-Mean Grain",
        "",
        md_table(
            key_pairs,
            [
                "metric",
                "comparison",
                "n_pairs",
                "treatment_mean",
                "comparator_mean",
                "diff_mean",
                "diff_mean_bootstrap_ci95_low",
                "diff_mean_bootstrap_ci95_high",
                "wilcoxon_p_holm_scope",
                "direction_by_mean",
            ],
        ),
        "",
        "## Interpretation Boundary",
        "",
        "Normalized RMSE and persistence-relative skill are scale-aware audit metrics, not new physical-identifiability evidence. They test whether the central matched-arm interpretation depends on raw-meter RMSE alone.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("resubmit/results/central_matched_50well_3seed_canonical80_20260610"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/well_selection/selected_50_wells.csv"),
    )
    args = parser.parse_args()

    root = args.root
    raw_path = root / "horizon_sensitivity_summary.csv"
    anon_path = root / "horizon_sensitivity_summary.anonymized.csv"
    raw = pd.read_csv(raw_path)
    anon = pd.read_csv(anon_path)
    selection = pd.read_csv(args.selection)

    alignment_errors = assert_row_alignment(raw, anon)
    if alignment_errors:
        raise SystemExit("; ".join(alignment_errors))

    raw = raw.copy()
    raw["stem_nfc"] = raw["stem"].map(normalize_text)
    selection = selection.copy()
    selection["stem_nfc"] = selection["stem"].map(normalize_text)
    needed_selection_cols = [
        "stem_nfc",
        "waterlevel_std",
        "waterlevel_range",
        "variability_tier",
    ]
    merged = pd.concat(
        [
            raw,
            anon[["well_label", "material_group"]],
        ],
        axis=1,
    ).merge(selection[needed_selection_cols], on="stem_nfc", how="left", validate="many_to_one")

    expected_wells = int(selection["stem_nfc"].nunique())
    errors: list[str] = []
    if len(merged) != 750:
        errors.append(f"expected 750 rows, got {len(merged)}")
    if merged["stem_nfc"].nunique() != expected_wells:
        errors.append(
            f"expected {expected_wells} source wells, got {merged['stem_nfc'].nunique()}"
        )
    if merged[["waterlevel_std", "waterlevel_range"]].isna().any().any():
        errors.append("missing selected-well variability metadata")
    if (merged["waterlevel_std"] <= 0).any() or (merged["waterlevel_range"] <= 0).any():
        errors.append("nonpositive variability metadata")
    if errors:
        raise SystemExit("; ".join(errors))

    merged["rmse"] = merged["test_rollout_rmse"]
    merged["nse"] = merged["test_rollout_nse"]
    merged["corr"] = merged["test_rollout_corr"]
    merged["nrmse_std"] = merged["rmse"] / merged["waterlevel_std"]
    merged["nrmse_range"] = merged["rmse"] / merged["waterlevel_range"]

    persistence = (
        merged[merged["variant"].eq("persistence")][["stem_nfc", "seed", "rmse"]]
        .rename(columns={"rmse": "persistence_rmse"})
        .copy()
    )
    merged = merged.merge(persistence, on=["stem_nfc", "seed"], how="left", validate="many_to_one")
    merged["skill_vs_persistence_rmse"] = 1.0 - (merged["rmse"] / merged["persistence_rmse"])

    metric_columns = [
        "rmse",
        "nrmse_std",
        "nrmse_range",
        "skill_vs_persistence_rmse",
        "nse",
        "corr",
    ]
    if not np.isfinite(merged[metric_columns].to_numpy(dtype=float)).all():
        raise SystemExit("nonfinite scale-independent metric values")

    seed_rows = merged[
        [
            "well_label",
            "material_group",
            "variability_tier",
            "horizon",
            "variant",
            "requested_variant",
            "seed",
            "rmse",
            "nrmse_std",
            "nrmse_range",
            "skill_vs_persistence_rmse",
            "nse",
            "corr",
            "persistence_rmse",
        ]
    ].copy()
    seed_rows = seed_rows.sort_values(["well_label", "seed", "variant"]).reset_index(drop=True)

    well_mean = (
        seed_rows.groupby(["well_label", "material_group", "variability_tier", "variant"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            rmse=("rmse", "mean"),
            nrmse_std=("nrmse_std", "mean"),
            nrmse_range=("nrmse_range", "mean"),
            skill_vs_persistence_rmse=("skill_vs_persistence_rmse", "mean"),
            nse=("nse", "mean"),
            corr=("corr", "mean"),
        )
        .reset_index()
    )

    overall = sort_variants(metric_summary(seed_rows, ["variant"]))
    by_material = sort_variants(metric_summary(seed_rows, ["material_group", "variant"]))
    by_variability = sort_variants(metric_summary(seed_rows, ["variability_tier", "variant"]))

    paired = pd.concat(
        [paired_stats(well_mean, metric) for metric in ["nrmse_std", "nrmse_range", "skill_vs_persistence_rmse", "nse"]],
        ignore_index=True,
    )

    output_paths = {
        "seed_rows": root / "canonical80_scale_independent_seed_rows.anonymized.csv",
        "well_mean": root / "canonical80_scale_independent_well_mean.anonymized.csv",
        "overall": root / "canonical80_scale_independent_summary_by_variant.csv",
        "by_material": root / "canonical80_scale_independent_summary_by_material.csv",
        "by_variability": root / "canonical80_scale_independent_summary_by_variability.csv",
        "paired": root / "canonical80_scale_independent_paired_stats.csv",
        "markdown": root / "canonical80_scale_independent_metric_audit_20260611.md",
        "verification": root / "canonical80_scale_independent_verification.json",
    }

    seed_rows.to_csv(output_paths["seed_rows"], index=False)
    well_mean.to_csv(output_paths["well_mean"], index=False)
    overall.to_csv(output_paths["overall"], index=False)
    by_material.to_csv(output_paths["by_material"], index=False)
    by_variability.to_csv(output_paths["by_variability"], index=False)
    paired.to_csv(output_paths["paired"], index=False)

    leak_results = {
        name: no_identifier_leak(path)
        for name, path in output_paths.items()
        if name
        in {
            "seed_rows",
            "well_mean",
            "overall",
            "by_material",
            "by_variability",
            "paired",
        }
    }
    leaks = {name: leak for name, (ok, leak) in leak_results.items() if not ok}
    identifier_status = "pass" if not leaks else "fail"

    verification = {
        "status": "pass" if identifier_status == "pass" else "fail",
        "root": str(root),
        "raw_rows": int(len(raw)),
        "seed_rows": int(len(seed_rows)),
        "well_mean_rows": int(len(well_mean)),
        "expected_wells": expected_wells,
        "observed_wells": int(seed_rows["well_label"].nunique()),
        "variants": sorted(seed_rows["variant"].unique().tolist()),
        "row_alignment_check": "pass",
        "variability_join_check": "pass",
        "identifier_leak_check": identifier_status,
        "identifier_leaks": leaks,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "outputs": {name: str(path) for name, path in output_paths.items() if name != "verification"},
    }

    write_markdown(output_paths["markdown"], overall, by_material, paired, verification)
    output_paths["verification"].write_text(
        json.dumps(verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

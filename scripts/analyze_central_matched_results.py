from __future__ import annotations

import argparse
import json
import math
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
BOOTSTRAP_SEED = 20260611
BOOTSTRAP_ITERATIONS = 10000


def material_class(stem: str) -> str:
    if "암반" in stem:
        return "암반"
    if "충적" in stem:
        return "충적"
    return "unknown"


def material_group(stem: str) -> str:
    if "암반" in stem:
        return "bedrock"
    if "충적" in stem:
        return "alluvial"
    return "unknown"


def sort_variants(df: pd.DataFrame, column: str = "variant") -> pd.DataFrame:
    order = {variant: idx for idx, variant in enumerate(VARIANT_ORDER)}
    return (
        df.assign(_variant_order=df[column].map(order).fillna(len(order)))
        .sort_values(["_variant_order", column])
        .drop(columns=["_variant_order"])
        .reset_index(drop=True)
    )


def metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("test_rollout_rmse", "size"),
            n_wells=("stem", "nunique"),
            rmse_mean=("test_rollout_rmse", "mean"),
            rmse_median=("test_rollout_rmse", "median"),
            rmse_std=("test_rollout_rmse", "std"),
            nse_mean=("test_rollout_nse", "mean"),
            corr_mean=("test_rollout_corr", "mean"),
        )
        .reset_index()
    )
    if group_cols == ["variant"]:
        out = out.drop(columns=["n_wells"])
        return out.sort_values("rmse_mean").reset_index(drop=True)
    return out.sort_values(group_cols[:-1] + ["rmse_mean"]).reset_index(drop=True)


def seed_comparison(df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        df.pivot_table(
            index=["stem", "material_class", "material_group", "seed"],
            columns="variant",
            values="test_rollout_rmse",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for variant in VARIANT_ORDER:
        if variant not in wide:
            wide[variant] = math.nan
    wide["ode_minus_gru_rmse"] = wide["ode"] - wide["gru"]
    wide["ws2_minus_gru_rmse"] = wide["ws2"] - wide["gru"]
    wide["ode_minus_ws2_rmse"] = wide["ode"] - wide["ws2"]
    wide["ode_only_minus_persistence_rmse"] = wide["ode_only"] - wide["persistence"]
    wide["ode_minus_persistence_rmse"] = wide["ode"] - wide["persistence"]
    wide["gru_minus_persistence_rmse"] = wide["gru"] - wide["persistence"]
    wide["winner"] = wide[VARIANT_ORDER].idxmin(axis=1)
    columns = [
        "stem",
        "material_class",
        "material_group",
        "seed",
        "gru",
        "ode",
        "ode_only",
        "persistence",
        "ws2",
        "ode_minus_gru_rmse",
        "ws2_minus_gru_rmse",
        "ode_minus_ws2_rmse",
        "ode_only_minus_persistence_rmse",
        "ode_minus_persistence_rmse",
        "gru_minus_persistence_rmse",
        "winner",
    ]
    return wide[columns].sort_values(["stem", "seed"]).reset_index(drop=True)


def regime_comparison(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for material, group in seed_df.groupby("material_class", dropna=False):
        rows.append(
            {
                "material_class": material,
                "material_group": group["material_group"].iloc[0],
                "n": int(len(group)),
                "n_wells": int(group["stem"].nunique()),
                "ode_minus_gru_mean": float(group["ode_minus_gru_rmse"].mean()),
                "ws2_minus_gru_mean": float(group["ws2_minus_gru_rmse"].mean()),
                "ode_minus_ws2_mean": float(group["ode_minus_ws2_rmse"].mean()),
                "ode_only_minus_persistence_mean": float(
                    group["ode_only_minus_persistence_rmse"].mean()
                ),
                "ode_win_rate": float((group["winner"] == "ode").mean()),
                "ode_only_win_rate": float((group["winner"] == "ode_only").mean()),
                "persistence_win_rate": float((group["winner"] == "persistence").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("material_class").reset_index(drop=True)


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


def well_level_frame(seed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_df.groupby(["stem", "material_class", "material_group"], dropna=False)[
            VARIANT_ORDER
        ]
        .mean()
        .reset_index()
    )


def paired_stat_rows(paired_frame: pd.DataFrame, grain: str) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", paired_frame)]
    for group_name, group in paired_frame.groupby("material_group", dropna=False):
        scopes.append(("material_group", str(group_name), group))

    rows = []
    for scope_type, scope_value, group in scopes:
        for comparison, treatment, comparator in PAIRED_COMPARISONS:
            id_cols = ["stem"] + (["seed"] if "seed" in group.columns else [])
            paired = group[id_cols + [treatment, comparator]].dropna()
            diff = (paired[treatment] - paired[comparator]).to_numpy(dtype=float)
            n = int(len(diff))
            if n == 0:
                rows.append(
                    {
                        "grain": grain,
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
            median_boot = np.median(resampled, axis=1)
            nonzero = diff[np.abs(diff) > 1.0e-12]
            if len(nonzero) == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            n_treatment_better = int((diff < -1.0e-12).sum())
            n_treatment_worse = int((diff > 1.0e-12).sum())
            n_ties = int(n - n_treatment_better - n_treatment_worse)
            diff_mean = float(diff.mean())
            if diff_mean < 0:
                direction = "treatment_lower_rmse"
            elif diff_mean > 0:
                direction = "treatment_higher_rmse"
            else:
                direction = "tie"
            rows.append(
                {
                    "scope_type": scope_type,
                    "grain": grain,
                    "scope_value": scope_value,
                    "comparison": comparison,
                    "treatment": treatment,
                    "comparator": comparator,
                    "n_pairs": n,
                    "treatment_rmse_mean": float(paired[treatment].mean()),
                    "comparator_rmse_mean": float(paired[comparator].mean()),
                    "diff_mean": diff_mean,
                    "diff_median": float(np.median(diff)),
                    "diff_std": float(np.std(diff, ddof=1)) if n > 1 else 0.0,
                    "diff_mean_bootstrap_ci95_low": float(np.quantile(mean_boot, 0.025)),
                    "diff_mean_bootstrap_ci95_high": float(np.quantile(mean_boot, 0.975)),
                    "diff_median_bootstrap_ci95_low": float(np.quantile(median_boot, 0.025)),
                    "diff_median_bootstrap_ci95_high": float(np.quantile(median_boot, 0.975)),
                    "n_treatment_better": n_treatment_better,
                    "n_treatment_worse": n_treatment_worse,
                    "n_ties": n_ties,
                    "fraction_treatment_better": float(n_treatment_better / n),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_two_sided": p_value,
                    "direction_by_mean": direction,
                    "status": "ok",
                }
            )

    out = pd.DataFrame(rows)
    if "wilcoxon_p_two_sided" in out:
        out["wilcoxon_p_holm_all"] = holm_adjust(out["wilcoxon_p_two_sided"].tolist())
    return out.sort_values(["grain", "scope_type", "scope_value", "comparison"]).reset_index(drop=True)


def paired_verification(
    paired_stats: pd.DataFrame,
    paired_frame: pd.DataFrame,
    grain: str,
) -> dict:
    expected_rows = (1 + int(paired_frame["material_group"].nunique())) * len(PAIRED_COMPARISONS)
    required_columns = {
        "grain",
        "scope_type",
        "scope_value",
        "comparison",
        "n_pairs",
        "diff_mean",
        "diff_mean_bootstrap_ci95_low",
        "diff_mean_bootstrap_ci95_high",
        "wilcoxon_p_two_sided",
        "wilcoxon_p_holm_all",
    }
    return {
        "status": "pass"
        if len(paired_stats) == expected_rows
        and required_columns.issubset(set(paired_stats.columns))
        and int((paired_stats["status"] == "ok").sum()) == expected_rows
        and set(paired_stats["grain"].unique().tolist()) == {grain}
        and not any("stem" in col.lower() for col in paired_stats.columns)
        else "fail",
        "grain": grain,
        "expected_rows": expected_rows,
        "observed_rows": int(len(paired_stats)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "scopes": sorted(paired_stats["scope_value"].unique().tolist()),
        "comparisons": [item[0] for item in PAIRED_COMPARISONS],
        "identifier_columns_present": [
            col for col in paired_stats.columns if "stem" in col.lower()
        ],
        "unit_count_overall": int(len(paired_frame)),
        "unit_counts_by_material_group": {
            str(k): int(v)
            for k, v in paired_frame["material_group"].value_counts().sort_index().items()
        },
    }


def load_summary_jsons(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("summary.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        physics = obj.get("physics_params") or {}
        if obj.get("variant") not in {"ode", "ode_only"}:
            continue
        rows.append(
            {
                "path": str(path),
                "stem": obj.get("stem"),
                "variant": obj.get("variant"),
                "seed": obj.get("seed"),
                "gamma_r": physics.get("gamma_r"),
                "gamma_d": physics.get("gamma_d"),
                "h_ref": physics.get("h_ref"),
                "tau_days": physics.get("tau_days", obj.get("tau_days")),
                "epochs": obj.get("epochs"),
                "patience": obj.get("patience"),
                "lr": obj.get("lr"),
            }
        )
    return rows


def physics_bounds(rows: pd.DataFrame, manifest: dict) -> dict:
    observed = {}
    for variant, group in rows.groupby("variant", dropna=False):
        entry = {"n": int(len(group))}
        for field in ["gamma_r", "gamma_d", "h_ref", "tau_days"]:
            series = pd.to_numeric(group[field], errors="coerce").dropna()
            if series.empty:
                continue
            entry[f"{field}_min"] = float(series.min())
            entry[f"{field}_max"] = float(series.max())
            entry[f"{field}_median"] = float(series.median())
        observed[str(variant)] = entry
    contract = manifest.get("contract", {})
    return {
        "observed": observed,
        "code_bounds": {
            "gamma_r": "[1.0e-4, 0.5001] from 1.0e-4 + 0.5 * sigmoid(raw_gamma_r)",
            "gamma_d": "[1.0e-4, 0.2001] from 1.0e-4 + 0.2 * sigmoid(raw_gamma_d)",
            "tau_days_ode_only": contract.get("ode_tau_candidates"),
            "tau_days_neural_ode": contract.get("tau_days"),
            "h_ref": "unconstrained trainable scalar in current code",
        },
        "bound_check": {
            "gamma_r_all_within_code_bound": bool(
                ((pd.to_numeric(rows["gamma_r"], errors="coerce").dropna() >= 1.0e-4)
                 & (pd.to_numeric(rows["gamma_r"], errors="coerce").dropna() <= 0.5001)).all()
            ),
            "gamma_d_all_within_code_bound": bool(
                ((pd.to_numeric(rows["gamma_d"], errors="coerce").dropna() >= 1.0e-4)
                 & (pd.to_numeric(rows["gamma_d"], errors="coerce").dropna() <= 0.2001)).all()
            ),
        },
    }


def contamination_check(clean_df: pd.DataFrame, screening_csv: Path | None) -> dict:
    if not screening_csv or not screening_csv.exists():
        return {"status": "not_run", "reason": "screening CSV not supplied or missing"}
    screen = pd.read_csv(screening_csv)
    merged = clean_df.merge(
        screen,
        on=["stem", "horizon", "variant", "seed"],
        suffixes=("_clean", "_screening"),
        how="inner",
    )
    identical = merged[
        (merged["test_rollout_rmse_clean"] - merged["test_rollout_rmse_screening"]).abs()
        <= 1.0e-12
    ]
    non_persistence = identical[identical["variant"] != "persistence"]
    return {
        "status": "pass" if len(non_persistence) == 0 else "fail",
        "matched_cells": int(len(merged)),
        "identical_rmse_cells": int(len(identical)),
        "identical_rmse_persistence_cells": int((identical["variant"] == "persistence").sum()),
        "identical_rmse_non_persistence_cells": int(len(non_persistence)),
    }


def write_markdown(
    path: Path,
    title: str,
    verification: dict,
    group_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    regime_diff: pd.DataFrame,
    winner_counts: pd.DataFrame,
    physics: dict,
    contamination: dict,
    paired_stats: pd.DataFrame,
    supplementary_paired_stats: pd.DataFrame,
) -> None:
    display_cols = [
        "grain",
        "scope_value",
        "comparison",
        "n_pairs",
        "diff_mean",
        "diff_mean_bootstrap_ci95_low",
        "diff_mean_bootstrap_ci95_high",
        "wilcoxon_p_holm_all",
        "direction_by_mean",
    ]
    lines = [
        f"# {title}",
        "",
        "## Coverage Verification",
        f"- Rows: {verification['n_rows']} / {verification['expected_rows']}",
        f"- Wells: {verification['n_stems']}; variants: {verification['variants']}; seeds: {verification['seeds']}",
        f"- Missing cells: {verification['missing_count']}; duplicate cells: {verification['duplicate_count']}",
        "- Interpretation guard: this is the clean canonical80 central matched-arm run. "
        "It can replace the earlier fast screening matrix for central matched-arm performance summaries, "
        "but it still does not complete manuscript update, debate, WRR logic, or final HP/Sentinel gates.",
        "",
        "## Overall Variant Summary",
        "```csv",
        group_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Regime Summary",
        "```csv",
        regime_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Regime Arm Differences",
        "```csv",
        regime_diff.to_csv(index=False).strip(),
        "```",
        "",
        "## Winner Counts",
        "```csv",
        winner_counts.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired Statistical Evidence",
        "- Primary differences are treatment RMSE minus comparator RMSE at the well grain after averaging the 3 seeds for each well; negative values favor the treatment arm.",
        f"- Bootstrap confidence intervals use {BOOTSTRAP_ITERATIONS} deterministic resamples with seed {BOOTSTRAP_SEED}.",
        "```csv",
        paired_stats[display_cols].to_csv(index=False).strip(),
        "```",
        "- Supplementary seed-well consistency statistics are written separately and should not be used as the manuscript's primary hypothesis-test grain.",
        "```csv",
        supplementary_paired_stats[display_cols].to_csv(index=False).strip(),
        "```",
        "",
        "## Physics Parameter Bounds",
        f"- gamma_r within code bound: {physics['bound_check']['gamma_r_all_within_code_bound']}.",
        f"- gamma_d within code bound: {physics['bound_check']['gamma_d_all_within_code_bound']}.",
        f"- ODE-only tau candidates from manifest: {physics['code_bounds']['tau_days_ode_only']}.",
        "- h_ref remains an unconstrained trainable scalar in the current code; bounded-h_ref diagnostics remain a separate evidence family.",
        "",
        "## Screening Contamination Cross-Check",
        f"- Status: {contamination.get('status')}.",
        f"- Matched cells: {contamination.get('matched_cells')}.",
        f"- Identical RMSE cells: {contamination.get('identical_rmse_cells')}; "
        f"persistence: {contamination.get('identical_rmse_persistence_cells')}; "
        f"non-persistence: {contamination.get('identical_rmse_non_persistence_cells')}.",
        "",
        "## Remaining Gate",
        "- Use these clean-root numbers to update manuscript-facing claim tables and draft text.",
        "- Do not use the old mixed-budget root as canonical evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Central matched result root")
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--screening-csv", default="")
    parser.add_argument("--prefix", default="canonical80")
    parser.add_argument("--title", default="Clean Canonical80 Central Matched Analysis")
    args = parser.parse_args()

    root = Path(args.root)
    summary_csv = root / "horizon_sensitivity_summary.csv"
    manifest_json = root / "horizon_sensitivity_manifest.json"
    df = pd.read_csv(summary_csv)
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    expected = {
        (stem, horizon, variant, seed)
        for stem in contract.get("stems", [])
        for horizon in contract.get("horizons", [])
        for variant in contract.get("variants", [])
        for seed in contract.get("seeds", [])
    }
    observed = {
        (row.stem, int(row.horizon), row.requested_variant if "requested_variant" in df.columns else row.variant, int(row.seed))
        for row in df.itertuples(index=False)
    }
    duplicates = int(
        df.duplicated(
            subset=["stem", "horizon", "requested_variant" if "requested_variant" in df.columns else "variant", "seed"]
        ).sum()
    )
    df["material_class"] = df["stem"].map(material_class)
    df["material_group"] = df["stem"].map(material_group)

    group_summary = metric_summary(df, ["variant"])
    regime_summary = metric_summary(df, ["material_class", "material_group", "variant"])
    seed_df = seed_comparison(df)
    regime_diff = regime_comparison(seed_df)
    well_df = well_level_frame(seed_df)
    paired_stats = paired_stat_rows(well_df, "well_mean_3seed")
    seed_paired_stats = paired_stat_rows(seed_df, "seed_well_supplementary")
    paired_stats_verification = paired_verification(
        paired_stats, well_df, "well_mean_3seed"
    )
    seed_paired_stats_verification = paired_verification(
        seed_paired_stats, seed_df, "seed_well_supplementary"
    )
    winner_counts = (
        seed_df["winner"]
        .value_counts()
        .rename_axis("winner")
        .reset_index(name="n_seed_well_wins")
    )

    physics_rows = pd.DataFrame(load_summary_jsons(root))
    physics = physics_bounds(physics_rows, manifest)
    contamination = contamination_check(df, Path(args.screening_csv) if args.screening_csv else None)
    verification = {
        "summary_csv": str(summary_csv),
        "manifest_json": str(manifest_json),
        "n_rows": int(len(df)),
        "expected_rows": int(len(expected)),
        "n_stems": int(df["stem"].nunique()),
        "n_variants": int(df["variant"].nunique()),
        "variants": sorted(df["variant"].unique().tolist()),
        "seeds": sorted(int(x) for x in df["seed"].unique().tolist()),
        "missing_count": int(len(expected - observed)),
        "duplicate_count": duplicates,
        "manifest_n_success": int(manifest.get("n_success", -1)),
        "manifest_n_error": int(manifest.get("n_error", -1)),
        "contamination_check": contamination,
        "paired_stats_verification": paired_stats_verification,
        "seed_paired_stats_verification": seed_paired_stats_verification,
        "status": "pass"
        if len(df) == len(expected)
        and not (expected - observed)
        and duplicates == 0
        and int(manifest.get("n_success", -1)) == len(expected)
        and int(manifest.get("n_error", -1)) == 0
        and contamination.get("status") in {"pass", "not_run"}
        and paired_stats_verification["status"] == "pass"
        and seed_paired_stats_verification["status"] == "pass"
        else "fail",
    }

    outputs = {
        f"{args.prefix}_group_summary_by_variant.csv": group_summary,
        f"{args.prefix}_regime_summary.csv": regime_summary,
        f"{args.prefix}_regime_comparison.csv": regime_diff,
        f"{args.prefix}_seed_comparison.csv": seed_df,
        f"{args.prefix}_paired_stats.csv": paired_stats,
        f"{args.prefix}_paired_stats_seed_level.csv": seed_paired_stats,
        f"{args.prefix}_winner_counts.csv": winner_counts,
        "physics_parameter_rows.csv": physics_rows,
    }
    for name, data in outputs.items():
        data.to_csv(root / name, index=False)
    (root / "physics_parameter_bounds_summary.json").write_text(
        json.dumps(physics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / f"{args.prefix}_verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (root / f"{args.prefix}_paired_stats_verification.json").write_text(
        json.dumps(paired_stats_verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / f"{args.prefix}_paired_stats_seed_level_verification.json").write_text(
        json.dumps(seed_paired_stats_verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(
        root / f"{args.prefix}_analysis_20260611.md",
        args.title,
        verification,
        group_summary,
        regime_summary,
        regime_diff,
        winner_counts,
        physics,
        contamination,
        paired_stats,
        seed_paired_stats,
    )
    print(json.dumps(verification, ensure_ascii=False))
    if verification["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

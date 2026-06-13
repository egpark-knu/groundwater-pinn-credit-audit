from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd


KOREAN_RE = re.compile(r"[\uac00-\ud7a3]")
EXPECTED_MODELS = ["gru", "lstm", "patchtst"]
EXPECTED_SEEDS = [7, 42, 99]


def normalize_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value))


def has_korean(path: Path) -> bool:
    return bool(KOREAN_RE.search(path.read_text(encoding="utf-8")))


def build_label_map(central_raw: Path, central_anon: Path) -> pd.DataFrame:
    raw = pd.read_csv(central_raw)
    anon = pd.read_csv(central_anon)
    if len(raw) != len(anon):
        raise ValueError(f"central raw/anonymized row mismatch: {len(raw)} vs {len(anon)}")
    mapping = pd.concat(
        [
            raw[["stem"]].rename(columns={"stem": "source_stem"}),
            anon[["well_label", "material_group"]],
        ],
        axis=1,
    ).drop_duplicates()
    mapping["stem_nfc"] = mapping["source_stem"].map(normalize_text)
    if mapping["stem_nfc"].nunique() != mapping["well_label"].nunique():
        raise ValueError("central source-to-label map is not one-to-one")
    return mapping[["stem_nfc", "well_label", "material_group"]].reset_index(drop=True)


def collect_rows(root: Path, label_map: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing_best_val_loss = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        row = data["row"]
        detail = data["detail"]
        meta = data["meta"]
        well = normalize_text(row["well"])
        best_val_loss = meta.get("best_val_loss")
        if best_val_loss is None or (isinstance(best_val_loss, float) and math.isnan(best_val_loss)):
            missing_best_val_loss.append(str(summary_path))
        rows.append(
            {
                "stem_nfc": well,
                "model": str(row["model"]),
                "seed": int(row["seed"]),
                "validation_best_delta_mse": float(best_val_loss),
                "test_recursive_rmse": float(row["rmse"]),
                "test_recursive_nse": float(row["nse"]),
                "test_recursive_mae": float(row["mae"]),
                "test_recursive_corr": float(row["corr"]),
                "test_recursive_peak_lag_days": int(row["peak_lag_days"]),
                "test_recursive_trough_lag_days": int(row["trough_lag_days"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "window": int(detail.get("window", 30)),
                "forecast_horizon": int(detail.get("forecast_horizon", 7)),
                "target_mode": str(meta.get("target_mode", "")),
            }
        )
    if missing_best_val_loss:
        raise ValueError(f"missing best_val_loss in {len(missing_best_val_loss)} summaries")
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no summary rows found under {root}")

    selection = selection.copy()
    selection["stem_nfc"] = selection["stem_nfc"].map(normalize_text)
    supplement = (
        df.merge(label_map, on="stem_nfc", how="left", validate="many_to_one")
        .merge(selection[["stem_nfc", "variability_tier"]], on="stem_nfc", how="left", validate="many_to_one")
    )
    if supplement[["well_label", "material_group", "variability_tier"]].isna().any().any():
        raise ValueError("failed to map all architecture rows to anonymized labels/material metadata")
    return supplement


def summarize(seed_rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        seed_rows.groupby(group_cols, dropna=False)
        .agg(
            n=("validation_best_delta_mse", "size"),
            n_wells=("well_label", "nunique"),
            validation_best_delta_mse_mean=("validation_best_delta_mse", "mean"),
            validation_best_delta_mse_median=("validation_best_delta_mse", "median"),
            validation_best_delta_mse_std=("validation_best_delta_mse", "std"),
            test_recursive_rmse_mean=("test_recursive_rmse", "mean"),
            test_recursive_rmse_median=("test_recursive_rmse", "median"),
            test_recursive_nse_mean=("test_recursive_nse", "mean"),
            test_recursive_mae_mean=("test_recursive_mae", "mean"),
            test_recursive_corr_mean=("test_recursive_corr", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def write_markdown(path: Path, overall: pd.DataFrame, verification: dict) -> None:
    display = overall.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    lines = [
        "# Architecture Train/Validation Supplement Availability Audit",
        "",
        "This audit addresses WRR R1-S19 for the 50-well architecture-diversity experiment used in Figure 3.",
        "",
        "## Status",
        "",
        f"- Verification status: `{verification['status']}`",
        f"- Summary rows: {verification['summary_rows']}",
        f"- Expected rows: {verification['expected_rows']}",
        f"- Anonymized wells: {verification['observed_wells']}",
        f"- Validation-loss metadata available: {verification['validation_loss_available']}",
        f"- Train predictions available: {verification['train_predictions_available']}",
        f"- Validation predictions available: {verification['val_predictions_available']}",
        f"- Identifier leak check: `{verification['identifier_leak_check']}`",
        "",
        "## Model-Level Validation Metadata",
        "",
        display.to_markdown(index=False),
        "",
        "## Interpretation Boundary",
        "",
        "`best_val_loss` is the one-step validation delta-MSE used for early stopping. It is useful as validation-performance metadata but is not directly comparable to seven-day recursive test RMSE. The original 50-well architecture run did not save train or validation prediction arrays, and no model checkpoints were retained in the result tree; therefore train-set performance cannot be reconstructed from current artifacts without rerunning the architecture experiment or changing the runner to persist train/validation predictions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/architecture_diversity_50well"))
    parser.add_argument("--selection", type=Path, default=Path("results/well_selection/selected_50_wells.csv"))
    parser.add_argument(
        "--central-raw",
        type=Path,
        default=Path("resubmit/results/central_matched_50well_3seed_canonical80_20260610/horizon_sensitivity_summary.csv"),
    )
    parser.add_argument(
        "--central-anon",
        type=Path,
        default=Path("resubmit/results/central_matched_50well_3seed_canonical80_20260610/horizon_sensitivity_summary.anonymized.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resubmit/results/architecture_train_validation_supplement_20260611"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(args.selection)
    label_map = build_label_map(args.central_raw, args.central_anon)
    rows = collect_rows(args.root, label_map, selection)

    expected_rows = int(selection["stem_nfc"].nunique() * len(EXPECTED_MODELS) * len(EXPECTED_SEEDS))
    train_predictions = sorted(args.root.glob("*/*/train_predictions.npz"))
    val_predictions = sorted(args.root.glob("*/*/val_predictions.npz"))
    checkpoints = sorted(
        list(args.root.glob("*/*/*.pt"))
        + list(args.root.glob("*/*/*.pth"))
        + list(args.root.glob("*/*/*checkpoint*"))
        + list(args.root.glob("*/*/*model*"))
    )

    seed_rows = rows[
        [
            "well_label",
            "material_group",
            "variability_tier",
            "model",
            "seed",
            "validation_best_delta_mse",
            "test_recursive_rmse",
            "test_recursive_nse",
            "test_recursive_mae",
            "test_recursive_corr",
            "test_recursive_peak_lag_days",
            "test_recursive_trough_lag_days",
            "elapsed_seconds",
            "window",
            "forecast_horizon",
            "target_mode",
        ]
    ].sort_values(["well_label", "model", "seed"])
    overall = summarize(seed_rows, ["model"])
    by_material = summarize(seed_rows, ["material_group", "model"])
    by_variability = summarize(seed_rows, ["variability_tier", "model"])

    output_paths = {
        "seed_rows": output_dir / "architecture_validation_seed_rows.anonymized.csv",
        "overall": output_dir / "architecture_validation_summary_by_model.csv",
        "by_material": output_dir / "architecture_validation_summary_by_material.csv",
        "by_variability": output_dir / "architecture_validation_summary_by_variability.csv",
        "markdown": output_dir / "architecture_train_validation_availability_audit_20260611.md",
        "verification": output_dir / "architecture_train_validation_verification.json",
    }
    seed_rows.to_csv(output_paths["seed_rows"], index=False)
    overall.to_csv(output_paths["overall"], index=False)
    by_material.to_csv(output_paths["by_material"], index=False)
    by_variability.to_csv(output_paths["by_variability"], index=False)

    leak_paths = [
        output_paths["seed_rows"],
        output_paths["overall"],
        output_paths["by_material"],
        output_paths["by_variability"],
    ]
    leak_failures = [str(path) for path in leak_paths if has_korean(path)]
    verification = {
        "status": "pass" if len(rows) == expected_rows and not leak_failures else "fail",
        "root": str(args.root),
        "summary_rows": int(len(rows)),
        "expected_rows": expected_rows,
        "observed_wells": int(seed_rows["well_label"].nunique()),
        "observed_models": sorted(seed_rows["model"].unique().tolist()),
        "observed_seeds": sorted(int(seed) for seed in seed_rows["seed"].unique().tolist()),
        "validation_loss_available": int(seed_rows["validation_best_delta_mse"].notna().sum()),
        "train_predictions_available": len(train_predictions),
        "val_predictions_available": len(val_predictions),
        "checkpoints_available": len(checkpoints),
        "identifier_leak_check": "pass" if not leak_failures else "fail",
        "identifier_leak_failures": leak_failures,
        "train_metric_blocker": "The 50-well architecture-diversity result tree retains summary.json, best_val_loss, and test_rollout_predictions.npz only. It does not retain train_predictions.npz, val_predictions.npz, model checkpoints, or train metrics needed to reconstruct train-set performance without rerun.",
        "outputs": {name: str(path) for name, path in output_paths.items() if name != "verification"},
    }
    write_markdown(output_paths["markdown"], overall, verification)
    output_paths["verification"].write_text(
        json.dumps(verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

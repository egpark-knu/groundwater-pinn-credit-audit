from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.architecture_diversity import train_plain_architecture_delta_variant  # noqa: E402
from groundwater_research.data_quality import clean_ladder_series  # noqa: E402
from groundwater_research.direct_delta_lead import build_direct_delta_split, train_direct_delta_variant  # noqa: E402
from groundwater_research.neural_ladder import LadderSeries, load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta  # noqa: E402

DEFAULT_AUDIT_DIRS = {
    "영덕영해_암반": ROOT / "results/solver_audit_real_yeongdeok",
    "영덕달산_암반": ROOT / "results/solver_audit_real_dalsan",
}


@dataclass(frozen=True)
class PhysicsInputRun:
    architecture: str
    condition: str
    seed: int


def subset_series_to_dates(series: LadderSeries, start: str, end: str) -> LadderSeries:
    date_days = series.dates.astype("datetime64[D]")
    mask = (date_days >= np.datetime64(start)) & (date_days <= np.datetime64(end))
    if int(mask.sum()) == 0:
        raise ValueError(f"No series samples within {start} to {end}.")
    return LadderSeries(
        stem=series.stem,
        dates=series.dates[mask],
        head_raw=series.head_raw[mask],
        head_interp=series.head_interp[mask],
        rain_mm=series.rain_mm[mask],
        climate=series.climate[mask].astype(np.float32),
        climate_cols=list(series.climate_cols),
    )


def align_feature_to_series(
    series: LadderSeries,
    feature_dates: np.ndarray,
    feature_values: np.ndarray,
) -> np.ndarray:
    target_index = pd.to_datetime(series.dates.astype("datetime64[D]").astype(str))
    feature_index = pd.to_datetime(np.asarray(feature_dates).astype("datetime64[D]").astype(str))
    feature = pd.Series(np.asarray(feature_values, dtype=float), index=feature_index).sort_index()
    aligned = feature.reindex(target_index.union(feature.index)).interpolate(method="time").reindex(target_index)
    aligned = aligned.bfill().ffill()
    if aligned.isna().any():
        raise ValueError("Failed to align ES-MDA head feature to the neural series dates.")
    return aligned.to_numpy(dtype=np.float32)


def with_external_head_feature(series: LadderSeries, esmda_head: np.ndarray) -> LadderSeries:
    esmda_head = np.asarray(esmda_head, dtype=np.float32)
    if esmda_head.shape != (len(series.dates),):
        raise ValueError(f"Expected ES-MDA head feature shape {(len(series.dates),)}, got {esmda_head.shape}.")
    climate = np.column_stack([series.climate.astype(np.float32), esmda_head]).astype(np.float32)
    return LadderSeries(
        stem=series.stem,
        dates=series.dates.copy(),
        head_raw=series.head_raw.copy(),
        head_interp=series.head_interp.copy(),
        rain_mm=series.rain_mm.copy(),
        climate=climate,
        climate_cols=list(series.climate_cols) + ["ESMDA_HEAD"],
    )


def load_esmda_posterior_mean_head(audit_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    dates = np.load(audit_dir / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    posterior = np.load(audit_dir / "posterior_hydrograph_ensemble.npy", allow_pickle=False)
    if posterior.ndim != 2:
        raise ValueError(f"Expected posterior ensemble shape (n_ensemble, n_days), got {posterior.shape}.")
    return dates, posterior.mean(axis=0).astype(np.float32)


def _train_model(
    architecture: str,
    condition: str,
    split_data: dict,
    seed: int,
    epochs: int,
    patience: int,
    lambda_penalty: float,
):
    if architecture == "lstm":
        variant = "lstm_ode" if condition == "ode_loss" else "lstm"
        return train_plain_architecture_delta_variant(
            split_data,
            variant=variant,
            seed=seed,
            epochs=epochs,
            patience=patience,
            hidden=64,
            lr=1.0e-3,
            lambda_penalty=lambda_penalty if condition == "ode_loss" else 0.0,
            event_weight_scale=0.0,
        )
    if architecture == "gru":
        variant = "ode" if condition == "ode_loss" else "gru"
        return train_direct_delta_variant(
            split_data,
            variant=variant,
            seed=seed,
            epochs=epochs,
            patience=patience,
            hidden=64,
            lr=1.0e-3,
            lambda_penalty=lambda_penalty if condition == "ode_loss" else 0.0,
            event_weight_scale=0.0,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def run_one(
    run: PhysicsInputRun,
    base_series: LadderSeries,
    augmented_series: LadderSeries,
    output_dir: Path,
    window: int,
    forecast_horizon: int,
    epochs: int,
    patience: int,
    lambda_penalty: float,
) -> dict:
    start = perf_counter()
    series = augmented_series if run.condition == "esmda_head_input" else base_series
    splits = make_block_splits(len(series.head_interp))
    split_data = build_direct_delta_split(series, splits, window=window, horizon=1, include_dhead=True)
    model, _, meta = _train_model(
        architecture=run.architecture,
        condition=run.condition,
        split_data=split_data,
        seed=run.seed,
        epochs=epochs,
        patience=patience,
        lambda_penalty=lambda_penalty,
    )
    rollout = recursive_block_rollout_one_step_delta(
        model=model,
        series=series,
        split=splits.test,
        norm=split_data["norm"],
        window=window,
        forecast_horizon=forecast_horizon,
        include_dhead=True,
    )
    run_dir = output_dir / series.stem / f"{run.architecture}_{run.condition}_seed{run.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "test_rollout_predictions.npz",
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )
    metrics = rollout["metrics"]
    row = {
        "well": series.stem,
        "architecture": run.architecture,
        "condition": run.condition,
        "seed": int(run.seed),
        "window": int(window),
        "forecast_horizon": int(forecast_horizon),
        "lambda": float(lambda_penalty if run.condition == "ode_loss" else 0.0),
        "rmse": float(metrics["rmse"]),
        "nse": float(metrics["nse"]),
        "mae": float(metrics["mae"]),
        "bias": float(metrics["bias"]),
        "corr": float(metrics["corr"]),
        "best_lag_days": int(metrics["best_lag_days"]),
        "peak_lag_days": int(metrics["peak_lag_days"]),
        "trough_lag_days": int(metrics["trough_lag_days"]),
        "elapsed_seconds": float(perf_counter() - start),
        "feature_contract": "ES-MDA posterior mean head is a retrospective solver-informed input, not an operational forecast feature."
        if run.condition == "esmda_head_input"
        else "No ES-MDA head input.",
        "variant_meta": str(meta["variant"]),
        "regularizer": str(meta.get("regularizer", "none")),
        "target_mode": str(meta["target_mode"]),
    }
    if "physics_params" in meta:
        row.update(
            {
                "physics_gamma_r": float(meta["physics_params"]["gamma_r"]),
                "physics_gamma_d": float(meta["physics_params"]["gamma_d"]),
                "physics_h_ref": float(meta["physics_params"]["h_ref"]),
            }
        )
    (run_dir / "summary.json").write_text(
        json.dumps({"row": row, "rollout_metrics": metrics, "meta": meta}, indent=2, ensure_ascii=False)
    )
    return row


def write_outputs(rows: list[dict], output_dir: Path, manifest_extra: dict) -> dict:
    df = pd.DataFrame(rows).sort_values(["architecture", "rmse"]).reset_index(drop=True)
    summary_csv = output_dir / "physics_input_dnn_summary.csv"
    comparison_csv = output_dir / "physics_input_dnn_comparison.csv"
    df.to_csv(summary_csv, index=False)
    comparison = df.pivot_table(index=["well", "architecture"], columns="condition", values="rmse", aggfunc="first").reset_index()
    for col in ["plain", "ode_loss", "esmda_head_input"]:
        if col not in comparison.columns:
            comparison[col] = np.nan
    comparison["ode_minus_plain_rmse"] = comparison["ode_loss"] - comparison["plain"]
    comparison["esmda_head_minus_plain_rmse"] = comparison["esmda_head_input"] - comparison["plain"]
    comparison["esmda_head_minus_ode_rmse"] = comparison["esmda_head_input"] - comparison["ode_loss"]
    comparison.to_csv(comparison_csv, index=False)
    seed_comparison_csv = output_dir / "physics_input_dnn_seed_comparison.csv"
    group_comparison_csv = output_dir / "physics_input_dnn_group_comparison.csv"
    seed_comp = df.pivot_table(index=["well", "architecture", "seed"], columns="condition", values="rmse", aggfunc="first").reset_index()
    for col in ["plain", "ode_loss", "esmda_head_input"]:
        if col not in seed_comp.columns:
            seed_comp[col] = np.nan
    seed_comp["ode_minus_plain_rmse"] = seed_comp["ode_loss"] - seed_comp["plain"]
    seed_comp["esmda_head_minus_plain_rmse"] = seed_comp["esmda_head_input"] - seed_comp["plain"]
    seed_comp["esmda_head_minus_ode_rmse"] = seed_comp["esmda_head_input"] - seed_comp["ode_loss"]
    seed_comp["winner"] = seed_comp[["plain", "ode_loss", "esmda_head_input"]].idxmin(axis=1)
    seed_comp.to_csv(seed_comparison_csv, index=False)
    group = (
        df.groupby(["well", "architecture", "condition"], dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            nse_mean=("nse", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    group_wide = group.pivot_table(index=["well", "architecture"], columns="condition", values="rmse_mean", aggfunc="first").reset_index()
    for col in ["plain", "ode_loss", "esmda_head_input"]:
        if col not in group_wide.columns:
            group_wide[col] = np.nan
    group_wide["ode_minus_plain_rmse"] = group_wide["ode_loss"] - group_wide["plain"]
    group_wide["esmda_head_minus_plain_rmse"] = group_wide["esmda_head_input"] - group_wide["plain"]
    group_wide["esmda_head_minus_ode_rmse"] = group_wide["esmda_head_input"] - group_wide["ode_loss"]
    group_wide["winner"] = group_wide[["plain", "ode_loss", "esmda_head_input"]].idxmin(axis=1)
    group_wide.to_csv(group_comparison_csv, index=False)
    manifest = {
        "n_success": int(len(df)),
        "summary_csv": str(summary_csv),
        "comparison_csv": str(comparison_csv),
        "seed_comparison_csv": str(seed_comparison_csv),
        "group_comparison_csv": str(group_comparison_csv),
        "contract": {
            "task": "recursive_7day_delta_forecast",
            "comparison": "plain vs ode_loss vs esmda_head_input",
            "physics_input": "posterior_mean_hydrograph_from_real_es_mda",
            "phase_b_excluded": "K/Sy 100x2 field autoencoder latent is explicitly deferred.",
        },
        **manifest_extra,
    }
    (output_dir / "physics_input_dnn_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="영덕영해_암반")
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--audit-dir", default=str(ROOT / "results/solver_audit_real_yeongdeok"))
    ap.add_argument("--audit-dirs", nargs="*", default=None, help="Optional WELL=PATH entries.")
    ap.add_argument("--architectures", nargs="+", choices=["lstm", "gru"], default=["lstm", "gru"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lambda-ode", type=float, default=0.1)
    ap.add_argument("--clean-head-outliers", action="store_true")
    ap.add_argument("--output-dir", default=str(ROOT / "results/physics_input_dnn"))
    return ap.parse_args()


def _parse_audit_dir_map(entries: list[str] | None, default_audit_dir: str) -> dict[str, Path]:
    mapping = dict(DEFAULT_AUDIT_DIRS)
    if entries:
        for entry in entries:
            if "=" not in entry:
                raise ValueError(f"Expected WELL=PATH audit-dir entry, got {entry!r}.")
            well, path = entry.split("=", 1)
            mapping[well] = Path(path)
    mapping.setdefault("영덕영해_암반", Path(default_audit_dir))
    return mapping


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = args.wells if args.wells is not None else [args.stem]
    seeds = args.seeds if args.seeds is not None else [args.seed]
    audit_dirs = _parse_audit_dir_map(args.audit_dirs, args.audit_dir)
    rows: list[dict] = []
    conditions = ["plain", "ode_loss", "esmda_head_input"]
    n_total = len(wells) * len(args.architectures) * len(conditions) * len(seeds)
    counter = 0
    feature_records: list[dict] = []
    for well in wells:
        audit_dir = Path(audit_dirs.get(well, Path(args.audit_dir)))
        feature_dates, esmda_head = load_esmda_posterior_mean_head(audit_dir)
        series = load_ladder_series(well)
        series = subset_series_to_dates(series, str(feature_dates[0]), str(feature_dates[-1]))
        if args.clean_head_outliers:
            series, outlier_report = clean_ladder_series(series)
        else:
            outlier_report = None
        aligned = align_feature_to_series(series, feature_dates, esmda_head)
        augmented = with_external_head_feature(series, aligned)
        feature_path = output_dir / f"esmda_head_feature_aligned_{well}.npz"
        np.savez(
            feature_path,
            dates=series.dates.astype("datetime64[D]").astype(str),
            esmda_head=aligned,
            observed_head=series.head_interp,
        )
        feature_records.append(
            {
                "well": well,
                "audit_dir": str(audit_dir),
                "feature_path": str(feature_path),
                "date_start": str(series.dates.astype("datetime64[D]")[0]),
                "date_end": str(series.dates.astype("datetime64[D]")[-1]),
                "head_outlier_cleaned": bool(args.clean_head_outliers),
                "outlier_report": outlier_report,
            }
        )
        for arch in args.architectures:
            for seed in seeds:
                for condition in conditions:
                    counter += 1
                    run = PhysicsInputRun(architecture=arch, condition=condition, seed=int(seed))
                    print(
                        f"[{counter}/{n_total}] well={well} architecture={arch} condition={condition} seed={seed}",
                        flush=True,
                    )
                    row = run_one(
                        run=run,
                        base_series=series,
                        augmented_series=augmented,
                        output_dir=output_dir,
                        window=args.window,
                        forecast_horizon=args.forecast_horizon,
                        epochs=args.epochs,
                        patience=args.patience,
                        lambda_penalty=args.lambda_ode,
                    )
                    rows.append(row)
                    print(f"    rmse={row['rmse']:.6f} nse={row['nse']:.6f}", flush=True)

    manifest = write_outputs(
        rows,
        output_dir,
        {
            "wells": wells,
            "seeds": [int(seed) for seed in seeds],
            "feature_records": feature_records,
        },
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

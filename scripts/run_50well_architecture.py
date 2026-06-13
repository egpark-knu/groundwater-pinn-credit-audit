from __future__ import annotations

import argparse
import json
import sys
import unicodedata
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
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.patchtst_ladder import train_patchtst_delta_variant  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta  # noqa: E402


DEFAULT_WELLS_CSV = ROOT / "results/well_selection/selected_50_wells.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results/architecture_diversity_50well"
DEFAULT_MODELS = ["lstm", "gru", "patchtst"]
DEFAULT_SEEDS = [7, 42, 99]
SUMMARY_COLUMNS = [
    "well",
    "model",
    "seed",
    "rmse",
    "nse",
    "mae",
    "bias",
    "corr",
    "peak_lag_days",
    "trough_lag_days",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class Architecture50Run:
    well: str
    model: str
    seed: int
    lambda_value: float = 0.0


def read_selected_wells(csv_path: Path = DEFAULT_WELLS_CSV, limit: int | None = None) -> list[str]:
    df = pd.read_csv(csv_path)
    if "stem_nfc" in df.columns:
        values = df["stem_nfc"]
    elif "stem" in df.columns:
        values = df["stem"]
    else:
        raise ValueError(f"{csv_path} must contain either 'stem_nfc' or 'stem'.")
    wells = [
        unicodedata.normalize("NFC", str(value))
        for value in values
        if pd.notna(value) and str(value).strip()
    ]
    if not wells:
        raise ValueError(f"No wells found in {csv_path}.")
    return wells[:limit] if limit is not None else wells


def plan_architecture_runs(
    wells: list[str],
    models: list[str] = DEFAULT_MODELS,
    seeds: list[int] = DEFAULT_SEEDS,
) -> list[Architecture50Run]:
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unsupported 50-well architecture model(s): {unknown}")
    return [
        Architecture50Run(
            well=unicodedata.normalize("NFC", well),
            model=model,
            seed=int(seed),
        )
        for well in wells
        for model in models
        for seed in seeds
    ]


def build_run_dir(output_dir: Path, run: Architecture50Run) -> Path:
    return output_dir / run.well / f"{run.model}_seed{run.seed}"


def _write_rollout(out_dir: Path, rollout: dict) -> None:
    np.savez(
        out_dir / "test_rollout_predictions.npz",
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )


def _train_model(
    model_name: str,
    split_data: dict,
    seed: int,
    epochs: int,
    patience: int,
    hidden: int,
    lr: float,
) -> tuple[object, dict]:
    if model_name == "lstm":
        model, _, meta = train_plain_architecture_delta_variant(
            split_data,
            variant="lstm",
            seed=seed,
            epochs=epochs,
            patience=patience,
            hidden=hidden,
            lr=lr,
            lambda_penalty=0.0,
            event_weight_scale=0.0,
        )
        return model, meta
    if model_name == "gru":
        # The repository's GRU implementation is the direct-delta variant.
        model, _, meta = train_direct_delta_variant(
            split_data,
            variant="gru",
            seed=seed,
            epochs=epochs,
            patience=patience,
            hidden=hidden,
            lr=lr,
            lambda_penalty=0.0,
            event_weight_scale=0.0,
        )
        return model, meta
    if model_name == "patchtst":
        model, _, meta = train_patchtst_delta_variant(
            split_data,
            variant="patchtst",
            seed=seed,
            epochs=epochs,
            patience=patience,
            lr=lr,
            lambda_penalty=0.0,
            event_weight_scale=0.0,
            patch_len=7,
            stride=7,
            d_model=32,
            n_heads=4,
            n_layers=2,
        )
        return model, meta
    raise ValueError(f"Unsupported model: {model_name}")


def run_one(
    run: Architecture50Run,
    output_dir: Path,
    window: int,
    forecast_horizon: int,
    epochs: int,
    patience: int,
    hidden: int,
    lr: float,
) -> dict:
    start = perf_counter()
    series = load_ladder_series(run.well)
    series, outlier_report = clean_ladder_series(series)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_direct_delta_split(series, splits, window=window, horizon=1, include_dhead=True)
    model, meta = _train_model(
        model_name=run.model,
        split_data=split_data,
        seed=run.seed,
        epochs=epochs,
        patience=patience,
        hidden=hidden,
        lr=lr,
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

    elapsed = perf_counter() - start
    metrics = rollout["metrics"]
    row = {
        "well": series.stem,
        "model": run.model,
        "seed": int(run.seed),
        "rmse": float(metrics["rmse"]),
        "nse": float(metrics["nse"]),
        "mae": float(metrics["mae"]),
        "bias": float(metrics["bias"]),
        "corr": float(metrics["corr"]),
        "peak_lag_days": int(metrics["peak_lag_days"]),
        "trough_lag_days": int(metrics["trough_lag_days"]),
        "elapsed_seconds": float(elapsed),
    }
    detail = {
        **row,
        "window": int(window),
        "forecast_horizon": int(forecast_horizon),
        "lambda": float(run.lambda_value),
        "best_lag_days": int(metrics["best_lag_days"]),
        "n_pred_days": int(metrics["n_pred_days"]),
        "variant_meta": str(meta["variant"]),
        "target_mode": str(meta["target_mode"]),
        "regularizer": "none",
        "head_outlier_cleaned": True,
        "outlier_n_flagged": int(outlier_report["n_flagged"]),
        "outlier_ratio_total": float(outlier_report["flagged_ratio_total"]),
    }
    run_dir = build_run_dir(output_dir, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "row": row,
                "detail": detail,
                "rollout_metrics": metrics,
                "meta": meta,
                "outlier_report": outlier_report,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    _write_rollout(run_dir, rollout)
    return row


def write_outputs(
    rows: list[dict],
    errors: list[dict],
    output_dir: Path,
    wells: list[str],
    models: list[str],
    seeds: list[int],
    window: int,
    forecast_horizon: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "architecture_diversity_50well_summary.csv"
    summary_alias_csv = output_dir / "summary.csv"
    errors_json = output_dir / "architecture_diversity_50well_errors.json"
    group_csv = output_dir / "architecture_diversity_50well_group_summary.csv"
    winners_csv = output_dir / "architecture_diversity_50well_mean_winners.csv"

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[SUMMARY_COLUMNS].sort_values(["well", "model", "seed"]).reset_index(drop=True)
        df.to_csv(summary_csv, index=False)
        df.to_csv(summary_alias_csv, index=False)
        group = (
            df.groupby(["well", "model"], dropna=False)
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                nse_mean=("nse", "mean"),
                nse_std=("nse", "std"),
                mae_mean=("mae", "mean"),
                bias_mean=("bias", "mean"),
                corr_mean=("corr", "mean"),
                peak_lag_median=("peak_lag_days", "median"),
                trough_lag_median=("trough_lag_days", "median"),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
            .sort_values(["well", "rmse_mean"])
        )
        group.to_csv(group_csv, index=False)
        winners = group.loc[group.groupby("well")["rmse_mean"].idxmin()].sort_values("well")
        winners.to_csv(winners_csv, index=False)
    else:
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_csv, index=False)
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_alias_csv, index=False)

    if errors:
        errors_json.write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    else:
        errors_json.unlink(missing_ok=True)

    manifest = {
        "n_success": int(len(rows)),
        "n_error": int(len(errors)),
        "summary_csv": str(summary_csv),
        "summary_alias_csv": str(summary_alias_csv),
        "group_summary_csv": str(group_csv),
        "mean_winners_csv": str(winners_csv),
        "errors_json": str(errors_json) if errors else None,
        "contract": {
            "n_selected_wells": len(wells),
            "models": models,
            "seeds": [int(seed) for seed in seeds],
            "expected_runs": len(wells) * len(models) * len(seeds),
            "regularization": "none",
            "lambda": 0.0,
            "window": int(window),
            "training_horizon": 1,
            "forecast_horizon": int(forecast_horizon),
            "head_outlier_cleaned": True,
            "rollout": "one_step_daily_delta_recursive_7day",
        },
    }
    (output_dir / "architecture_diversity_50well_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run 50-well plain architecture diversity experiment.")
    ap.add_argument("--wells-csv", default=str(DEFAULT_WELLS_CSV))
    ap.add_argument("--wells", nargs="+", default=None, help="Optional explicit well stems; overrides --wells-csv.")
    ap.add_argument("--limit", type=int, default=None, help="Optional first-N limit for smoke runs.")
    ap.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wells = (
        [unicodedata.normalize("NFC", well) for well in args.wells]
        if args.wells is not None
        else read_selected_wells(Path(args.wells_csv), limit=args.limit)
    )
    if args.wells is not None and args.limit is not None:
        wells = wells[: args.limit]
    runs = plan_architecture_runs(wells=wells, models=args.models, seeds=args.seeds)
    rows: list[dict] = []
    errors: list[dict] = []

    for idx, run in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] well={run.well} model={run.model} seed={run.seed}", flush=True)
        try:
            row = run_one(
                run=run,
                output_dir=output_dir,
                window=args.window,
                forecast_horizon=args.forecast_horizon,
                epochs=args.epochs,
                patience=args.patience,
                hidden=args.hidden,
                lr=args.lr,
            )
            rows.append(row)
            print(
                f"    rmse={row['rmse']:.6f} nse={row['nse']:.6f} "
                f"peak_lag={row['peak_lag_days']} trough_lag={row['trough_lag_days']}",
                flush=True,
            )
        except Exception as exc:
            err = {"well": run.well, "model": run.model, "seed": int(run.seed), "error": str(exc)}
            errors.append(err)
            print(f"    ERROR: {exc}", flush=True)

    manifest = write_outputs(
        rows=rows,
        errors=errors,
        output_dir=output_dir,
        wells=wells,
        models=args.models,
        seeds=args.seeds,
        window=args.window,
        forecast_horizon=args.forecast_horizon,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(f"{len(errors)} architecture diversity run(s) failed.")


if __name__ == "__main__":
    main()

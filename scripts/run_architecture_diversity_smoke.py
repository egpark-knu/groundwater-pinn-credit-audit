from __future__ import annotations

import argparse
import json
import shutil
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
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.patchtst_ladder import train_patchtst_delta_variant  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta  # noqa: E402


DEFAULT_WELLS = [
    "거제신현_암반",
    "영덕도천_천부_충적",
    "창원북면_충적",
    "안동태화_충적",
    "영덕달산_암반",
    "울진울진_암반",
]
DEFAULT_MODELS = ["narx", "lstm", "gru", "patchtst"]


@dataclass(frozen=True)
class ArchitectureRun:
    well: str
    model: str
    seed: int
    lambda_value: float = 0.0


def plan_architecture_runs(
    wells: list[str],
    models: list[str],
    seed: int | None = None,
    seeds: list[int] | None = None,
) -> list[ArchitectureRun]:
    supported = set(DEFAULT_MODELS)
    unknown = sorted(set(models) - supported)
    if unknown:
        raise ValueError(f"Unsupported architecture model(s): {unknown}")
    if seeds is None:
        if seed is None:
            raise ValueError("Either seed or seeds must be provided.")
        seeds = [int(seed)]
    return [ArchitectureRun(well=well, model=model, seed=int(run_seed)) for well in wells for model in models for run_seed in seeds]


def _write_rollout(out_dir: Path, rollout: dict) -> None:
    np.savez(
        out_dir / "test_rollout_predictions.npz",
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )


def _train_delta_model(
    model_name: str,
    split_data: dict,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[object, dict]:
    if model_name == "gru":
        model, _, meta = train_direct_delta_variant(
            split_data,
            variant="gru",
            seed=seed,
            epochs=epochs,
            patience=patience,
            hidden=64,
            lr=1.0e-3,
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
            lr=1.0e-3,
            lambda_penalty=0.0,
            event_weight_scale=0.0,
            patch_len=7,
            stride=7,
            d_model=32,
            n_heads=4,
            n_layers=2,
        )
        return model, meta
    if model_name in {"narx", "lstm"}:
        model, _, meta = train_plain_architecture_delta_variant(
            split_data,
            variant=model_name,
            seed=seed,
            epochs=epochs,
            patience=patience,
            hidden=64,
            lr=1.0e-3,
            event_weight_scale=0.0,
        )
        return model, meta
    raise ValueError(f"Unsupported model: {model_name}")


def run_one(
    run: ArchitectureRun,
    window: int,
    forecast_horizon: int,
    epochs: int,
    patience: int,
    output_dir: Path,
    clean_head_outliers: bool = False,
) -> dict:
    start = perf_counter()
    series = load_ladder_series(run.well)
    outlier_report = None
    if clean_head_outliers:
        series, outlier_report = clean_ladder_series(series)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_direct_delta_split(series, splits, window=window, horizon=1, include_dhead=True)

    model, meta = _train_delta_model(run.model, split_data, seed=run.seed, epochs=epochs, patience=patience)
    rollout = recursive_block_rollout_one_step_delta(
        model=model,
        series=series,
        split=splits.test,
        norm=split_data["norm"],
        window=window,
        forecast_horizon=forecast_horizon,
        include_dhead=True,
    )

    run_dir = output_dir / series.stem / f"{run.model}_seed{run.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    elapsed = perf_counter() - start
    metrics = rollout["metrics"]
    row = {
        "well": series.stem,
        "model": run.model,
        "seed": int(run.seed),
        "window": int(window),
        "forecast_horizon": int(forecast_horizon),
        "lambda": float(run.lambda_value),
        "rmse": float(metrics["rmse"]),
        "nse": float(metrics["nse"]),
        "best_lag_days": int(metrics["best_lag_days"]),
        "peak_lag_days": int(metrics["peak_lag_days"]),
        "trough_lag_days": int(metrics["trough_lag_days"]),
        "mae": float(metrics["mae"]),
        "bias": float(metrics["bias"]),
        "corr": float(metrics["corr"]),
        "elapsed_seconds": float(elapsed),
        "variant_meta": str(meta["variant"]),
        "target_mode": str(meta["target_mode"]),
        "regularizer": "none",
        "head_outlier_cleaned": bool(clean_head_outliers),
    }
    if outlier_report is not None:
        row.update(
            {
                "outlier_n_flagged": int(outlier_report["n_flagged"]),
                "outlier_ratio_total": float(outlier_report["flagged_ratio_total"]),
            }
        )
    (run_dir / "summary.json").write_text(
        json.dumps({"row": row, "rollout_metrics": metrics, "meta": meta}, indent=2, ensure_ascii=False)
    )
    _write_rollout(run_dir, rollout)
    return row


def _write_outputs(df: pd.DataFrame, errors: list[dict], output_dir: Path) -> dict:
    summary_csv = output_dir / "architecture_diversity_smoke_summary.csv"
    winners_csv = output_dir / "architecture_diversity_smoke_winners.csv"
    group_summary_csv = output_dir / "architecture_diversity_group_summary.csv"
    mean_winners_csv = output_dir / "architecture_diversity_mean_winners.csv"
    if not df.empty:
        df = df.sort_values(["well", "rmse"]).reset_index(drop=True)
        df.to_csv(summary_csv, index=False)
        winners = df.loc[df.groupby("well")["rmse"].idxmin()].sort_values("well")
        winners.to_csv(winners_csv, index=False)
        group = (
            df.groupby(["well", "model"], dropna=False)
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                nse_mean=("nse", "mean"),
                best_lag_median=("best_lag_days", "median"),
                n_seeds=("seed", "nunique"),
                outlier_ratio_total=("outlier_ratio_total", "first"),
            )
            .reset_index()
            .sort_values(["well", "rmse_mean"])
        )
        group.to_csv(group_summary_csv, index=False)
        mean_winners = group.loc[group.groupby("well")["rmse_mean"].idxmin()].sort_values("well")
        mean_winners.to_csv(mean_winners_csv, index=False)
        research_dir = ROOT / "results/research_summaries"
        research_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_csv, research_dir / "six_well_architecture_diversity_summary.csv")
        shutil.copy2(winners_csv, research_dir / "six_well_architecture_diversity_winners.csv")
        shutil.copy2(group_summary_csv, research_dir / "six_well_architecture_diversity_group_summary.csv")
        shutil.copy2(mean_winners_csv, research_dir / "six_well_architecture_diversity_mean_winners.csv")
    if errors:
        (output_dir / "architecture_diversity_smoke_errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False)
        )
    else:
        (output_dir / "architecture_diversity_smoke_errors.json").unlink(missing_ok=True)
    manifest = {
        "n_success": int(len(df)),
        "n_error": int(len(errors)),
        "summary_csv": str(summary_csv),
        "winners_csv": str(winners_csv),
        "group_summary_csv": str(group_summary_csv),
        "mean_winners_csv": str(mean_winners_csv),
        "contract": {
            "evaluation": "one_step_delta_recursive_7day",
            "model_space": "plain architecture only",
            "regularizers": "none",
            "lambda": 0.0,
            "head_outlier_cleaned": bool(df["head_outlier_cleaned"].all()) if "head_outlier_cleaned" in df else False,
        },
    }
    (output_dir / "architecture_diversity_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=DEFAULT_WELLS)
    ap.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=42, help="Single-seed fallback when --seeds is omitted.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--clean-head-outliers", action="store_true")
    ap.add_argument("--output-dir", default=str(ROOT / "results/architecture_diversity_smoke"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[dict] = []

    runs = plan_architecture_runs(args.wells, args.models, seed=args.seed, seeds=args.seeds)
    for idx, run in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] well={run.well} model={run.model} seed={run.seed}", flush=True)
        try:
            row = run_one(
                run=run,
                window=args.window,
                forecast_horizon=args.forecast_horizon,
                epochs=args.epochs,
                patience=args.patience,
                output_dir=output_dir,
                clean_head_outliers=args.clean_head_outliers,
            )
            rows.append(row)
            print(f"    rmse={row['rmse']:.6f} nse={row['nse']:.6f} best_lag={row['best_lag_days']}", flush=True)
        except Exception as exc:
            err = {"well": run.well, "model": run.model, "seed": run.seed, "error": str(exc)}
            errors.append(err)
            print(f"    ERROR: {exc}", flush=True)

    df = pd.DataFrame(rows)
    manifest = _write_outputs(df, errors, output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(f"{len(errors)} architecture smoke run(s) failed.")


if __name__ == "__main__":
    main()

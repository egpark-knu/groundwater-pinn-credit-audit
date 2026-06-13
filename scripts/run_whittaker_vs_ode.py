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
from groundwater_research.direct_delta_lead import build_direct_delta_split  # noqa: E402
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta  # noqa: E402


DEFAULT_WELLS = [
    "거제신현_암반",
    "영덕도천_천부_충적",
    "창원북면_충적",
    "안동태화_충적",
    "영덕달산_암반",
    "울진울진_암반",
]
DEFAULT_MODELS = ["lstm", "lstm_ode", "lstm_ws2"]


@dataclass(frozen=True)
class MatchedLambdaRun:
    well: str
    model: str
    seed: int
    lambda_value: float


def plan_matched_lambda_runs(wells: list[str], seed: int, lambda_value: float) -> list[MatchedLambdaRun]:
    runs: list[MatchedLambdaRun] = []
    for well in wells:
        runs.append(MatchedLambdaRun(well=well, model="lstm", seed=int(seed), lambda_value=0.0))
        runs.append(MatchedLambdaRun(well=well, model="lstm_ode", seed=int(seed), lambda_value=float(lambda_value)))
        runs.append(MatchedLambdaRun(well=well, model="lstm_ws2", seed=int(seed), lambda_value=float(lambda_value)))
    return runs


def plan_full_lambda_runs(
    wells: list[str],
    seeds: list[int],
    lambda_values: list[float],
    models: list[str] | None = None,
) -> list[MatchedLambdaRun]:
    models = DEFAULT_MODELS if models is None else models
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unsupported Whittaker-vs-ODE model(s): {unknown}")
    runs: list[MatchedLambdaRun] = []
    for well in wells:
        for seed in seeds:
            for model in models:
                for lam in lambda_values:
                    runs.append(MatchedLambdaRun(well=well, model=model, seed=int(seed), lambda_value=float(lam)))
    return runs


def _write_rollout(out_dir: Path, rollout: dict) -> None:
    np.savez(
        out_dir / "test_rollout_predictions.npz",
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )


def run_one(
    run: MatchedLambdaRun,
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
    model, _, meta = train_plain_architecture_delta_variant(
        split_data,
        variant=run.model,
        seed=run.seed,
        epochs=epochs,
        patience=patience,
        hidden=64,
        lr=1.0e-3,
        lambda_penalty=run.lambda_value,
        event_weight_scale=0.0,
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

    run_dir = output_dir / series.stem / f"{run.model}_lambda{run.lambda_value:g}_seed{run.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
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
        "elapsed_seconds": float(perf_counter() - start),
        "variant_meta": str(meta["variant"]),
        "regularizer": str(meta["regularizer"]),
        "target_mode": str(meta["target_mode"]),
        "head_outlier_cleaned": bool(clean_head_outliers),
        "run_reused": False,
    }
    if outlier_report is not None:
        row.update(
            {
                "outlier_n_flagged": int(outlier_report["n_flagged"]),
                "outlier_ratio_total": float(outlier_report["flagged_ratio_total"]),
            }
        )
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
    _write_rollout(run_dir, rollout)
    return row


def _run_dir(output_dir: Path, stem: str, model: str, lambda_value: float, seed: int) -> Path:
    return output_dir / stem / f"{model}_lambda{lambda_value:g}_seed{seed}"


def clone_plain_lstm_lambda_row(
    source_row: dict,
    run: MatchedLambdaRun,
    output_dir: Path,
) -> dict:
    row = dict(source_row)
    row["lambda"] = float(run.lambda_value)
    row["run_reused"] = True
    row["lambda_alias_from"] = 0.0
    row["elapsed_seconds"] = 0.0
    source_dir = _run_dir(output_dir, str(source_row["well"]), "lstm", 0.0, int(run.seed))
    target_dir = _run_dir(output_dir, str(source_row["well"]), "lstm", float(run.lambda_value), int(run.seed))
    target_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        src_npz = source_dir / "test_rollout_predictions.npz"
        if src_npz.exists():
            import shutil

            shutil.copy2(src_npz, target_dir / "test_rollout_predictions.npz")
    (target_dir / "summary.json").write_text(
        json.dumps({"row": row, "note": "Plain LSTM has no lambda-dependent penalty; row is aliased from lambda=0."}, indent=2, ensure_ascii=False)
    )
    return row


def write_comparison(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    wide = summary.pivot(index="well", columns="model", values="rmse").reset_index()
    wide["ode_minus_plain_rmse"] = wide["lstm_ode"] - wide["lstm"]
    wide["ws2_minus_plain_rmse"] = wide["lstm_ws2"] - wide["lstm"]
    wide["ode_minus_ws2_rmse"] = wide["lstm_ode"] - wide["lstm_ws2"]
    wide["winner"] = summary.loc[summary.groupby("well")["rmse"].idxmin()].set_index("well").loc[wide["well"], "model"].values
    wide.to_csv(output_dir / "whittaker_vs_ode_comparison.csv", index=False)
    research_dir = ROOT / "results/research_summaries"
    research_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(research_dir / "whittaker_vs_ode_lstm_matched_lambda.csv", index=False)
    return wide


def write_full_lambda_outputs(summary: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    group_csv = output_dir / "whittaker_vs_ode_group_summary.csv"
    seed_comp_csv = output_dir / "whittaker_vs_ode_seed_lambda_comparison.csv"
    group_comp_csv = output_dir / "whittaker_vs_ode_group_lambda_comparison.csv"
    ode_win_csv = output_dir / "ode_win_conditions.csv"

    group = (
        summary.groupby(["well", "model", "lambda"], dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            nse_mean=("nse", "mean"),
            best_lag_median=("best_lag_days", "median"),
            n_seeds=("seed", "nunique"),
            head_outlier_cleaned=("head_outlier_cleaned", "all"),
            outlier_ratio_total=("outlier_ratio_total", "first"),
            n_reused=("run_reused", "sum"),
        )
        .reset_index()
        .sort_values(["well", "lambda", "rmse_mean"])
    )
    group.to_csv(group_csv, index=False)

    seed_wide = summary.pivot_table(index=["well", "seed", "lambda"], columns="model", values="rmse", aggfunc="first").reset_index()
    seed_wide["ode_minus_plain_rmse"] = seed_wide["lstm_ode"] - seed_wide["lstm"]
    seed_wide["ws2_minus_plain_rmse"] = seed_wide["lstm_ws2"] - seed_wide["lstm"]
    seed_wide["ode_minus_ws2_rmse"] = seed_wide["lstm_ode"] - seed_wide["lstm_ws2"]
    seed_wide["winner"] = seed_wide[["lstm", "lstm_ode", "lstm_ws2"]].idxmin(axis=1)
    seed_wide.to_csv(seed_comp_csv, index=False)

    group_wide = group.pivot_table(index=["well", "lambda"], columns="model", values="rmse_mean", aggfunc="first").reset_index()
    group_wide["ode_minus_plain_rmse"] = group_wide["lstm_ode"] - group_wide["lstm"]
    group_wide["ws2_minus_plain_rmse"] = group_wide["lstm_ws2"] - group_wide["lstm"]
    group_wide["ode_minus_ws2_rmse"] = group_wide["lstm_ode"] - group_wide["lstm_ws2"]
    group_wide["winner"] = group_wide[["lstm", "lstm_ode", "lstm_ws2"]].idxmin(axis=1)
    group_wide.to_csv(group_comp_csv, index=False)

    ode_wins = seed_wide[seed_wide["winner"].eq("lstm_ode") | (seed_wide["ode_minus_ws2_rmse"] < 0)].copy()
    ode_wins.to_csv(ode_win_csv, index=False)

    research_dir = ROOT / "results/research_summaries"
    research_dir.mkdir(parents=True, exist_ok=True)
    group.to_csv(research_dir / "whittaker_vs_ode_3seed_group_summary.csv", index=False)
    seed_wide.to_csv(research_dir / "whittaker_vs_ode_3seed_seed_lambda_comparison.csv", index=False)
    group_wide.to_csv(research_dir / "whittaker_vs_ode_3seed_group_lambda_comparison.csv", index=False)

    return {
        "group_summary_csv": str(group_csv),
        "seed_lambda_comparison_csv": str(seed_comp_csv),
        "group_lambda_comparison_csv": str(group_comp_csv),
        "ode_win_conditions_csv": str(ode_win_csv),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=DEFAULT_WELLS)
    ap.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--lambda-value", type=float, default=0.1)
    ap.add_argument("--lambda-values", nargs="+", type=float, default=None)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--clean-head-outliers", action="store_true")
    ap.add_argument("--output-dir", default=str(ROOT / "results/whittaker_vs_ode"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[dict] = []
    if args.seeds is not None or args.lambda_values is not None:
        seeds = args.seeds if args.seeds is not None else [args.seed]
        lambda_values = args.lambda_values if args.lambda_values is not None else [args.lambda_value]
        runs = plan_full_lambda_runs(args.wells, seeds=seeds, lambda_values=lambda_values, models=args.models)
    else:
        runs = plan_matched_lambda_runs(args.wells, seed=args.seed, lambda_value=args.lambda_value)
    plain_cache: dict[tuple[str, int], dict] = {}

    for idx, run in enumerate(runs, start=1):
        print(
            f"[{idx}/{len(runs)}] well={run.well} model={run.model} "
            f"lambda={run.lambda_value:g} seed={run.seed}",
            flush=True,
        )
        try:
            if run.model == "lstm" and float(run.lambda_value) != 0.0:
                cache_key = (run.well, int(run.seed))
                if cache_key not in plain_cache:
                    base_run = MatchedLambdaRun(well=run.well, model="lstm", seed=run.seed, lambda_value=0.0)
                    plain_cache[cache_key] = run_one(
                        run=base_run,
                        window=args.window,
                        forecast_horizon=args.forecast_horizon,
                        epochs=args.epochs,
                        patience=args.patience,
                        output_dir=output_dir,
                        clean_head_outliers=args.clean_head_outliers,
                    )
                row = clone_plain_lstm_lambda_row(plain_cache[cache_key], run=run, output_dir=output_dir)
            else:
                row = run_one(
                    run=run,
                    window=args.window,
                    forecast_horizon=args.forecast_horizon,
                    epochs=args.epochs,
                    patience=args.patience,
                    output_dir=output_dir,
                    clean_head_outliers=args.clean_head_outliers,
                )
                if run.model == "lstm" and float(run.lambda_value) == 0.0:
                    plain_cache[(run.well, int(run.seed))] = row
            rows.append(row)
            print(f"    rmse={row['rmse']:.6f} nse={row['nse']:.6f}", flush=True)
        except Exception as exc:
            err = {"well": run.well, "model": run.model, "seed": run.seed, "lambda": run.lambda_value, "error": str(exc)}
            errors.append(err)
            print(f"    ERROR: {exc}", flush=True)

    df = pd.DataFrame(rows)
    summary_csv = output_dir / "whittaker_vs_ode_summary.csv"
    if not df.empty:
        df = df.sort_values(["well", "seed", "lambda", "model"]).reset_index(drop=True)
        df.to_csv(summary_csv, index=False)
        if {"seed", "lambda"}.issubset(df.columns) and (df[["well", "seed", "lambda"]].drop_duplicates().shape[0] > df["well"].nunique()):
            extra_outputs = write_full_lambda_outputs(df, output_dir)
        else:
            write_comparison(df, output_dir)
            extra_outputs = {}
    if errors:
        (output_dir / "whittaker_vs_ode_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    else:
        (output_dir / "whittaker_vs_ode_errors.json").unlink(missing_ok=True)
    manifest = {
        "n_success": int(len(rows)),
        "n_error": int(len(errors)),
        "summary_csv": str(summary_csv),
        "comparison_csv": str(output_dir / "whittaker_vs_ode_comparison.csv"),
        "contract": {
            "architecture": "LSTM",
            "variants": ["plain LSTM", "LSTM+ODE", "LSTM+WS2"],
            "evaluation": "one_step_delta_recursive_7day",
            "regularizer_comparator": "Whittaker WS2",
            "lambda_values": args.lambda_values if args.lambda_values is not None else [args.lambda_value],
            "seeds": args.seeds if args.seeds is not None else [args.seed],
            "head_outlier_cleaned": bool(args.clean_head_outliers),
            "plain_lstm_lambda_rows": "aliased from lambda=0 because plain LSTM has no lambda-dependent penalty",
        },
    }
    manifest.update(extra_outputs if "extra_outputs" in locals() else {})
    (output_dir / "whittaker_vs_ode_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(f"{len(errors)} Whittaker-vs-ODE run(s) failed.")


if __name__ == "__main__":
    main()

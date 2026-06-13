from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.direct_delta_lead import build_direct_delta_split, train_direct_delta_variant  # noqa: E402
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.nhits_baseline import build_nf_frame, fit_nhits_one_step, recursive_block_rollout_nhits  # noqa: E402
from groundwater_research.patchtst_ladder import train_patchtst_delta_variant  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta  # noqa: E402


DEFAULT_WELLS = ["울진울진_암반", "안동태화_충적", "영덕달산_암반"]
DEFAULT_MODELS = ["gru", "patchtst", "nhits"]


def _write_rollout(out_dir: Path, rollout: dict) -> None:
    np.savez(
        out_dir / "test_rollout_predictions.npz",
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )


def run_one(
    well: str,
    model_name: str,
    seed: int,
    window: int,
    forecast_horizon: int,
    epochs: int,
    patience: int,
    nhits_steps: int,
    output_dir: Path,
) -> dict:
    start = perf_counter()
    series = load_ladder_series(well)
    splits = make_block_splits(len(series.head_interp))
    run_dir = output_dir / series.stem / f"{model_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "gru":
        split_data = build_direct_delta_split(series, splits, window=window, horizon=1, include_dhead=True)
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
        rollout = recursive_block_rollout_one_step_delta(
            model=model,
            series=series,
            split=splits.test,
            norm=split_data["norm"],
            window=window,
            forecast_horizon=forecast_horizon,
            include_dhead=True,
        )
        variant_meta = meta["variant"]
    elif model_name == "patchtst":
        split_data = build_direct_delta_split(series, splits, window=window, horizon=1, include_dhead=True)
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
        rollout = recursive_block_rollout_one_step_delta(
            model=model,
            series=series,
            split=splits.test,
            norm=split_data["norm"],
            window=window,
            forecast_horizon=forecast_horizon,
            include_dhead=True,
        )
        variant_meta = meta["variant"]
    elif model_name == "nhits":
        full_df = build_nf_frame(series)
        train_df = full_df.iloc[splits.train].copy()
        nf = fit_nhits_one_step(
            train_df=train_df,
            exog_cols=list(series.climate_cols),
            input_size=window,
            max_steps=nhits_steps,
            random_seed=seed,
        )
        rollout = recursive_block_rollout_nhits(
            nf=nf,
            series=series,
            split=splits.test,
            window=window,
            forecast_horizon=forecast_horizon,
        )
        variant_meta = "nhits_one_step_recursive"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    elapsed = perf_counter() - start
    metrics = rollout["metrics"]
    row = {
        "well": series.stem,
        "model": model_name,
        "seed": int(seed),
        "window": int(window),
        "forecast_horizon": int(forecast_horizon),
        "rmse": float(metrics["rmse"]),
        "nse": float(metrics["nse"]),
        "best_lag_days": int(metrics["best_lag_days"]),
        "peak_lag_days": int(metrics["peak_lag_days"]),
        "trough_lag_days": int(metrics["trough_lag_days"]),
        "mae": float(metrics["mae"]),
        "bias": float(metrics["bias"]),
        "corr": float(metrics["corr"]),
        "elapsed_seconds": float(elapsed),
        "variant_meta": variant_meta,
    }
    (run_dir / "summary.json").write_text(json.dumps({"row": row, "rollout": metrics}, indent=2, ensure_ascii=False))
    _write_rollout(run_dir, rollout)
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=DEFAULT_WELLS)
    ap.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--nhits-steps", type=int, default=200)
    ap.add_argument("--output-dir", default=str(ROOT / "results/expanded_wells_smoke"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[dict] = []

    planned = [(well, model) for well in args.wells for model in args.models]
    for idx, (well, model_name) in enumerate(planned, start=1):
        print(f"[{idx}/{len(planned)}] well={well} model={model_name} seed={args.seed}", flush=True)
        try:
            row = run_one(
                well=well,
                model_name=model_name,
                seed=args.seed,
                window=args.window,
                forecast_horizon=args.forecast_horizon,
                epochs=args.epochs,
                patience=args.patience,
                nhits_steps=args.nhits_steps,
                output_dir=output_dir,
            )
            rows.append(row)
            print(f"    rmse={row['rmse']:.6f} nse={row['nse']:.6f} best_lag={row['best_lag_days']}", flush=True)
        except Exception as exc:
            err = {"well": well, "model": model_name, "seed": args.seed, "error": str(exc)}
            errors.append(err)
            print(f"    ERROR: {exc}", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["well", "rmse"]).reset_index(drop=True)
        df.to_csv(output_dir / "expanded_wells_smoke_summary.csv", index=False)
        winners = df.loc[df.groupby("well")["rmse"].idxmin()].sort_values("well")
        winners.to_csv(output_dir / "expanded_wells_smoke_winners.csv", index=False)
    if errors:
        (output_dir / "expanded_wells_smoke_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    manifest = {
        "n_success": len(rows),
        "n_error": len(errors),
        "summary_csv": str(output_dir / "expanded_wells_smoke_summary.csv"),
        "winners_csv": str(output_dir / "expanded_wells_smoke_winners.csv"),
        "contract": {
            "evaluation": "one_step_daily_recursive_7day",
            "model_space": "delta for GRU/PatchTST; one-step recursive head for N-HiTS",
            "window": args.window,
            "forecast_horizon": args.forecast_horizon,
            "seed": args.seed,
        },
    }
    (output_dir / "expanded_wells_smoke_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

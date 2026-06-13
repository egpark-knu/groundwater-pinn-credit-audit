from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.direct_delta_lead import build_direct_delta_split, train_direct_delta_variant  # noqa: E402
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta  # noqa: E402


@dataclass(frozen=True)
class SweepRun:
    well: str
    model: str
    lambda_value: float
    seed: int


def configure_korean_font() -> None:
    for name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def plan_runs(
    wells: list[str],
    models: list[str],
    lambda_values: list[float],
    seeds: list[int],
) -> list[SweepRun]:
    runs: list[SweepRun] = []
    for well in wells:
        well_nfc = unicodedata.normalize("NFC", well)
        for seed in seeds:
            for model in models:
                if model == "gru":
                    runs.append(SweepRun(well=well_nfc, model="gru", lambda_value=0.0, seed=int(seed)))
                    continue
                if model == "ode":
                    for lam in lambda_values:
                        runs.append(SweepRun(well=well_nfc, model="ode", lambda_value=float(lam), seed=int(seed)))
                    continue
                raise ValueError(f"Unsupported model: {model}")
    return runs


def lambda_plot_value(value: float, min_positive: float) -> float:
    if value <= 0.0:
        return min_positive / 10.0
    return value


def _safe_lambda_label(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:g}".replace(".", "p")


def _run_one(
    run: SweepRun,
    output_dir: Path,
    window: int,
    forecast_horizon: int,
    epochs: int,
    patience: int,
    hidden: int,
    lr: float,
    event_weight_scale: float,
) -> dict:
    start = perf_counter()
    series = load_ladder_series(run.well)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_direct_delta_split(
        series,
        splits,
        window=window,
        horizon=1,
        include_dhead=True,
    )
    model, _, meta = train_direct_delta_variant(
        split_data,
        variant=run.model,
        seed=run.seed,
        epochs=epochs,
        patience=patience,
        hidden=hidden,
        lr=lr,
        lambda_penalty=run.lambda_value,
        event_weight_scale=event_weight_scale,
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
        "well": run.well,
        "lambda": run.lambda_value,
        "seed": run.seed,
        "model": run.model,
        "rmse": float(metrics["rmse"]),
        "nse": float(metrics["nse"]),
        "peak_lag": int(metrics["peak_lag_days"]),
        "trough_lag": int(metrics["trough_lag_days"]),
        "best_lag_days": int(metrics["best_lag_days"]),
        "mae": float(metrics["mae"]),
        "bias": float(metrics["bias"]),
        "corr": float(metrics["corr"]),
        "n_pred_days": int(metrics["n_pred_days"]),
        "elapsed_seconds": float(elapsed),
        "variant_meta": meta["variant"],
        "event_weight_scale": float(event_weight_scale),
        "window": int(window),
        "forecast_horizon": int(forecast_horizon),
        "target_mode": meta["target_mode"],
        "physics_gamma_r": np.nan,
        "physics_gamma_d": np.nan,
        "physics_h_ref": np.nan,
    }
    if "physics_params" in meta:
        row.update(
            {
                "physics_gamma_r": float(meta["physics_params"]["gamma_r"]),
                "physics_gamma_d": float(meta["physics_params"]["gamma_d"]),
                "physics_h_ref": float(meta["physics_params"]["h_ref"]),
            }
        )

    run_dir = output_dir / run.well / f"{run.model}_lambda{_safe_lambda_label(run.lambda_value)}_seed{run.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            **meta,
            "model": run.model,
            "lambda_value": run.lambda_value,
            "model_space": "delta",
            "window": window,
            "forecast_horizon": forecast_horizon,
            "rollout_contract": "one_step_daily_recursive_7day",
        },
        "rollout": metrics,
        "row": row,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    np.savez(
        run_dir / "test_rollout_predictions.npz",
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )
    return row


def write_summary(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.sort_values(["well", "seed", "model", "lambda"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "lambda_sweep_summary.csv", index=False)

    agg = (
        df.groupby(["well", "model", "lambda"], dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            nse_mean=("nse", "mean"),
            nse_std=("nse", "std"),
            best_lag_median=("best_lag_days", "median"),
            peak_lag_median=("peak_lag", "median"),
            trough_lag_median=("trough_lag", "median"),
            n_runs=("rmse", "size"),
        )
        .reset_index()
    )
    agg.to_csv(output_dir / "lambda_sweep_group_summary.csv", index=False)
    return df


def plot_lambda_sensitivity(df: pd.DataFrame, output_dir: Path) -> Path:
    configure_korean_font()
    ode_df = df[df["model"] == "ode"].copy()
    if ode_df.empty:
        raise ValueError("No ODE rows to plot.")
    positives = [float(v) for v in ode_df["lambda"].unique() if float(v) > 0]
    min_positive = min(positives) if positives else 1.0
    ode_df["lambda_plot"] = ode_df["lambda"].map(lambda x: lambda_plot_value(float(x), min_positive))
    seeds = sorted(ode_df["seed"].unique())
    wells = sorted(ode_df["well"].unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.0 * len(seeds), 4.5), sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    color_map = {well: plt.cm.tab10(idx % 10) for idx, well in enumerate(wells)}
    for ax, seed in zip(axes, seeds):
        sub_seed = ode_df[ode_df["seed"] == seed]
        for well in wells:
            sub = sub_seed[sub_seed["well"] == well].sort_values("lambda_plot")
            ax.plot(
                sub["lambda_plot"],
                sub["rmse"],
                marker="o",
                lw=1.6,
                color=color_map[well],
                label=well,
            )
        gru = df[(df["model"] == "gru") & (df["seed"] == seed)]
        for _, row in gru.iterrows():
            ax.scatter(
                lambda_plot_value(0.0, min_positive),
                row["rmse"],
                marker="x",
                s=45,
                color=color_map[row["well"]],
                alpha=0.75,
            )
        ax.set_xscale("log")
        ticks = [lambda_plot_value(0.0, min_positive)] + sorted(positives)
        labels = ["0"] + [f"{v:g}" for v in sorted(positives)]
        ax.set_xticks(ticks, labels)
        ax.set_title(f"seed={seed}")
        ax.set_xlabel("ODE-loss weight λ")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Recursive 7-day RMSE (m)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("ODE-loss λ sensitivity under locked recursive 7-day rollout")
    fig.tight_layout()
    out = output_dir / "fig_lambda_sensitivity.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", choices=["gru", "ode"], default=["gru", "ode"])
    ap.add_argument("--lambda-values", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--event-weight-scale", type=float, default=0.0)
    ap.add_argument("--output-dir", default=str(ROOT / "results/lambda_sweep_w30"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = plan_runs(args.wells, args.models, args.lambda_values, args.seeds)
    rows: list[dict] = []
    errors: list[dict] = []
    for idx, run in enumerate(runs, start=1):
        print(
            f"[{idx}/{len(runs)}] well={run.well} model={run.model} "
            f"lambda={run.lambda_value:g} seed={run.seed}",
            flush=True,
        )
        try:
            row = _run_one(
                run,
                output_dir=output_dir,
                window=args.window,
                forecast_horizon=args.forecast_horizon,
                epochs=args.epochs,
                patience=args.patience,
                hidden=args.hidden,
                lr=args.lr,
                event_weight_scale=args.event_weight_scale,
            )
            rows.append(row)
            print(
                f"    rmse={row['rmse']:.6f} nse={row['nse']:.6f} "
                f"best_lag={row['best_lag_days']}",
                flush=True,
            )
        except Exception as exc:
            err = {
                "well": run.well,
                "model": run.model,
                "lambda": run.lambda_value,
                "seed": run.seed,
                "error": str(exc),
            }
            errors.append(err)
            print(f"    ERROR: {exc}", flush=True)

    if errors:
        (output_dir / "lambda_sweep_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    if not rows:
        raise SystemExit("No successful lambda sweep runs.")

    df = write_summary(rows, output_dir)
    fig_path = plot_lambda_sensitivity(df, output_dir)
    manifest = {
        "n_success": len(rows),
        "n_error": len(errors),
        "summary_csv": str(output_dir / "lambda_sweep_summary.csv"),
        "group_summary_csv": str(output_dir / "lambda_sweep_group_summary.csv"),
        "figure": str(fig_path),
        "contract": {
            "model_space": "delta",
            "training_horizon": 1,
            "evaluation": "one_step_daily_recursive_7day",
            "window": args.window,
            "forecast_horizon": args.forecast_horizon,
            "event_weight_scale": args.event_weight_scale,
        },
    }
    (output_dir / "lambda_sweep_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

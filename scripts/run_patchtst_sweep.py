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
from groundwater_research.patchtst_ladder import train_patchtst_delta_variant  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta  # noqa: E402


@dataclass(frozen=True)
class PatchTSTSweepRun:
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


def plan_patchtst_runs(
    wells: list[str],
    models: list[str],
    lambda_values: list[float],
    seeds: list[int],
) -> list[PatchTSTSweepRun]:
    runs: list[PatchTSTSweepRun] = []
    sweep_models = {"patchtst_ws2", "patchtst_ode"}
    single_lambda_models = {"patchtst", "legacy_gru"}
    supported = sweep_models | single_lambda_models
    for model in models:
        if model not in supported:
            raise ValueError(f"Unsupported model: {model}")
    for well in wells:
        well_nfc = unicodedata.normalize("NFC", well)
        for seed in seeds:
            for model in models:
                if model in single_lambda_models:
                    runs.append(PatchTSTSweepRun(well=well_nfc, model=model, lambda_value=0.0, seed=int(seed)))
                else:
                    for lam in lambda_values:
                        runs.append(PatchTSTSweepRun(well=well_nfc, model=model, lambda_value=float(lam), seed=int(seed)))
    return runs


def lambda_plot_value(value: float, min_positive: float) -> float:
    if value <= 0.0:
        return min_positive / 10.0
    return value


def _safe_lambda_label(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:g}".replace(".", "p")


def _train_for_run(
    run: PatchTSTSweepRun,
    split_data: dict,
    epochs: int,
    patience: int,
    lr: float,
    event_weight_scale: float,
    patch_len: int,
    stride: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
):
    if run.model == "legacy_gru":
        return train_direct_delta_variant(
            split_data,
            variant="gru",
            seed=run.seed,
            epochs=epochs,
            patience=patience,
            hidden=64,
            lr=lr,
            lambda_penalty=0.0,
            event_weight_scale=event_weight_scale,
        )
    return train_patchtst_delta_variant(
        split_data,
        variant=run.model,
        seed=run.seed,
        epochs=epochs,
        patience=patience,
        lr=lr,
        lambda_penalty=run.lambda_value,
        event_weight_scale=event_weight_scale,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )


def _run_one(
    run: PatchTSTSweepRun,
    output_dir: Path,
    window: int,
    forecast_horizon: int,
    epochs: int,
    patience: int,
    lr: float,
    event_weight_scale: float,
    patch_len: int,
    stride: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
) -> dict:
    start = perf_counter()
    series = load_ladder_series(run.well)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_direct_delta_split(series, splits, window=window, horizon=1, include_dhead=True)
    model, _, meta = _train_for_run(
        run,
        split_data=split_data,
        epochs=epochs,
        patience=patience,
        lr=lr,
        event_weight_scale=event_weight_scale,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
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
        "lambda": float(run.lambda_value),
        "seed": int(run.seed),
        "model": run.model,
        "architecture": "GRU" if run.model == "legacy_gru" else "PatchTST",
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
        "patch_len": int(patch_len) if run.model != "legacy_gru" else np.nan,
        "stride": int(stride) if run.model != "legacy_gru" else np.nan,
        "d_model": int(d_model) if run.model != "legacy_gru" else np.nan,
        "n_heads": int(n_heads) if run.model != "legacy_gru" else np.nan,
        "n_layers": int(n_layers) if run.model != "legacy_gru" else np.nan,
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
            "primary_architecture_lock": "PatchTST primary; GRU legacy baseline only",
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
    df.to_csv(output_dir / "patchtst_sweep_summary.csv", index=False)
    # Compatibility alias for downstream figure scripts that expect lambda sweep naming.
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
    agg.to_csv(output_dir / "patchtst_sweep_group_summary.csv", index=False)
    agg.to_csv(output_dir / "lambda_sweep_group_summary.csv", index=False)
    return df


def plot_patchtst_lambda_sensitivity(df: pd.DataFrame, output_dir: Path) -> Path:
    configure_korean_font()
    plot_df = df[df["model"].isin(["patchtst_ws2", "patchtst_ode"])].copy()
    if plot_df.empty:
        raise ValueError("No PatchTST WS2/ODE rows to plot.")
    positives = [float(v) for v in plot_df["lambda"].unique() if float(v) > 0]
    min_positive = min(positives) if positives else 1.0
    plot_df["lambda_plot"] = plot_df["lambda"].map(lambda x: lambda_plot_value(float(x), min_positive))
    seeds = sorted(plot_df["seed"].unique())
    wells = sorted(plot_df["well"].unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.4 * len(seeds), 4.5), sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    colors = {well: plt.cm.tab10(idx % 10) for idx, well in enumerate(wells)}
    markers = {"patchtst_ws2": "s", "patchtst_ode": "o"}
    labels = {"patchtst_ws2": "PatchTST+WS2", "patchtst_ode": "PatchTST+ODE"}
    for ax, seed in zip(axes, seeds):
        sub_seed = plot_df[plot_df["seed"] == seed]
        for model in ["patchtst_ws2", "patchtst_ode"]:
            for well in wells:
                sub = sub_seed[(sub_seed["well"] == well) & (sub_seed["model"] == model)].sort_values("lambda_plot")
                if sub.empty:
                    continue
                ax.plot(
                    sub["lambda_plot"],
                    sub["rmse"],
                    marker=markers[model],
                    lw=1.4,
                    alpha=0.82,
                    color=colors[well],
                    linestyle="-" if model == "patchtst_ode" else "--",
                    label=f"{well} {labels[model]}",
                )
        plain = df[(df["model"] == "patchtst") & (df["seed"] == seed)]
        for _, row in plain.iterrows():
            ax.scatter(
                lambda_plot_value(0.0, min_positive),
                row["rmse"],
                marker="x",
                s=50,
                color=colors[row["well"]],
                alpha=0.9,
            )
        ax.set_xscale("log")
        ticks = [lambda_plot_value(0.0, min_positive)] + sorted(positives)
        labels_tick = ["0"] + [f"{v:g}" for v in sorted(positives)]
        ax.set_xticks(ticks, labels_tick)
        ax.set_title(f"seed={seed}")
        ax.set_xlabel("Regularization / ODE-loss weight λ")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Recursive 7-day RMSE (m)")
    handles, labels_all = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles[: min(len(handles), 8)], labels_all[: min(len(labels_all), 8)], fontsize=7)
    fig.suptitle("PatchTST λ sensitivity under locked recursive 7-day rollout")
    fig.tight_layout()
    out = output_dir / "fig_patchtst_lambda_sensitivity.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", required=True)
    ap.add_argument(
        "--models",
        nargs="+",
        choices=["patchtst", "patchtst_ws2", "patchtst_ode", "legacy_gru"],
        default=["patchtst", "patchtst_ws2", "patchtst_ode", "legacy_gru"],
    )
    ap.add_argument("--lambda-values", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--event-weight-scale", type=float, default=0.0)
    ap.add_argument("--patch-len", type=int, default=7)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--output-dir", default=str(ROOT / "results/patchtst_sweep_w30"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = plan_patchtst_runs(args.wells, args.models, args.lambda_values, args.seeds)
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
                lr=args.lr,
                event_weight_scale=args.event_weight_scale,
                patch_len=args.patch_len,
                stride=args.stride,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
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
        (output_dir / "patchtst_sweep_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    if not rows:
        raise SystemExit("No successful PatchTST sweep runs.")

    df = write_summary(rows, output_dir)
    fig_path = plot_patchtst_lambda_sensitivity(df, output_dir)
    manifest = {
        "n_success": len(rows),
        "n_error": len(errors),
        "summary_csv": str(output_dir / "patchtst_sweep_summary.csv"),
        "group_summary_csv": str(output_dir / "patchtst_sweep_group_summary.csv"),
        "figure": str(fig_path),
        "contract": {
            "primary_architecture": "PatchTST",
            "legacy_baseline": "GRU lambda=0 only",
            "model_space": "delta",
            "training_horizon": 1,
            "evaluation": "one_step_daily_recursive_7day",
            "window": args.window,
            "forecast_horizon": args.forecast_horizon,
            "event_weight_scale": args.event_weight_scale,
            "patch_len": args.patch_len,
            "stride": args.stride,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
        },
    }
    (output_dir / "patchtst_sweep_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

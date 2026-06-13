from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np


def configure_korean_font() -> None:
    for name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def load_split(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def extract_metrics(summary: dict) -> tuple[float, float]:
    if "test_rollout" in summary:
        return float(summary["test_rollout"]["rmse"]), float(summary["test_rollout"]["nse"])
    if "rollout" in summary:
        return float(summary["rollout"]["rmse"]), float(summary["rollout"]["nse"])
    if "test" in summary:
        return float(summary["test"]["rmse_final"]), float(summary["test"]["nse_final"])
    raise KeyError("Could not find rollout/test metrics in summary.")


def main() -> None:
    configure_korean_font()
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--variants", default="gru,ws1,ws2,ode")
    ap.add_argument(
        "--allow-direct-fallback",
        action="store_true",
        help="Allow fallback to direct final-horizon predictions when rollout predictions are missing.",
    )
    args = ap.parse_args()

    stem_root = Path(args.root) / args.stem
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    colors = {"persistence": "#566573", "gru": "#154c79", "ws1": "#b9770e", "ws2": "#7d3c98", "ode": "#b03a2e", "ode_only": "#117864"}
    labels = {"persistence": "Persistence", "gru": "GRU", "ws1": "GRU+WS1", "ws2": "GRU+WS2", "ode": "GRU+ODE", "ode_only": "Standalone ODE"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    ax_full, ax_zoom = axes
    obs_dates = None
    obs_final = None
    scores = []

    for variant in variants:
        run_dir = next(iter(sorted(stem_root.glob(f"{variant}_seed*/"))), None)
        if run_dir is None:
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        rollout_path = run_dir / "test_rollout_predictions.npz"
        if rollout_path.exists():
            data = load_split(rollout_path)
            dates = data["dates"].astype("datetime64[D]")
            obs = data["obs"].astype(float)
            pred = data["pred"].astype(float)
            rmse, nse = extract_metrics(summary)
        else:
            if not args.allow_direct_fallback:
                raise FileNotFoundError(
                    f"{run_dir} does not contain test_rollout_predictions.npz. "
                    "This plot is rollout-only by default because direct final-horizon predictions are not valid "
                    "for the manuscript's recursive 7-day task. Re-run with --allow-direct-fallback only for "
                    "exploratory audit plots."
                )
            data = load_split(run_dir / "test_predictions.npz")
            dates = data["target_dates"].astype("datetime64[D]")
            obs = data["obs_seq_phys"][:, -1].astype(float)
            pred = data["pred_seq_phys"][:, -1].astype(float)
            rmse, nse = extract_metrics(summary)
        if obs_dates is None:
            obs_dates = dates
            obs_final = obs
            ax_full.plot(dates, obs, color="black", lw=1.0, label="Observed")
        ax_full.plot(dates, pred, color=colors.get(variant, None), lw=1.1, label=labels.get(variant, variant))
        scores.append((variant, rmse, nse))

    if obs_dates is None:
        raise FileNotFoundError(f"No runs found for {args.stem} under {stem_root}")

    swing = np.abs(np.diff(obs_final))
    pivot = int(np.argmax(swing)) + 1 if len(swing) else len(obs_dates) // 2
    lo = max(0, pivot - 25)
    hi = min(len(obs_dates), pivot + 35)
    ax_zoom.plot(obs_dates[lo:hi], obs_final[lo:hi], color="black", lw=1.0, label="Observed")

    for variant in variants:
        run_dir = next(iter(sorted(stem_root.glob(f"{variant}_seed*/"))), None)
        if run_dir is None:
            continue
        rollout_path = run_dir / "test_rollout_predictions.npz"
        if rollout_path.exists():
            data = load_split(rollout_path)
            dates = data["dates"].astype("datetime64[D]")
            pred = data["pred"].astype(float)
        else:
            if not args.allow_direct_fallback:
                raise FileNotFoundError(
                    f"{run_dir} does not contain test_rollout_predictions.npz. "
                    "This plot is rollout-only by default because direct final-horizon predictions are not valid "
                    "for the manuscript's recursive 7-day task. Re-run with --allow-direct-fallback only for "
                    "exploratory audit plots."
                )
            data = load_split(run_dir / "test_predictions.npz")
            dates = data["target_dates"].astype("datetime64[D]")
            pred = data["pred_seq_phys"][:, -1].astype(float)
        ax_zoom.plot(dates[lo:hi], pred[lo:hi], color=colors.get(variant, None), lw=1.1, label=labels.get(variant, variant))

    score_text = "\n".join(
        f"{labels.get(v, v)}: rollout RMSE={rmse:.3f}, NSE={nse:.3f}"
        for v, rmse, nse in scores
    )
    ax_full.text(
        0.01,
        0.98,
        score_text,
        transform=ax_full.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )

    ax_full.set_title(f"{args.stem} | Test Recursive Rollout Comparison")
    ax_zoom.set_title("Zoom Around Largest Observed Swing")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = stem_root / "comparison_test_fit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()

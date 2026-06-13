from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.spatial_k_smoke import (  # noqa: E402
    SpatialKSmokeConfig,
    build_pulse_recharge,
    run_spatial_k_member,
    sample_correlated_logk_fields,
    summarize_member_heads,
)


def configure_font() -> None:
    for name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def parse_scales(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one correlation scale is required")
    return values


def main() -> None:
    configure_font()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr-scales-m", default="100,300,900,1800")
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--n-days", type=int, default=96)
    ap.add_argument("--n-cells", type=int, default=40)
    ap.add_argument("--cell-size-m", type=float, default=100.0)
    ap.add_argument("--obs-col", type=int, default=10)
    ap.add_argument("--seed", type=int, default=260409)
    ap.add_argument("--output-dir", default=str(ROOT / "results/solver_audit/spatial_k_smoke"))
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    config = SpatialKSmokeConfig(
        n_cells=args.n_cells,
        cell_size_m=args.cell_size_m,
        obs_col=args.obs_col,
    )
    recharge = build_pulse_recharge(n_days=args.n_days)
    scales = parse_scales(args.corr_scales_m)

    member_rows: list[dict] = []
    scale_rows: list[dict] = []
    heads_by_scale: dict[float, np.ndarray] = {}
    logk_by_scale: dict[float, np.ndarray] = {}

    for scale_idx, corr_len_m in enumerate(scales):
        logk_fields = sample_correlated_logk_fields(
            n_members=args.n_members,
            n_cells=args.n_cells,
            cell_size_m=args.cell_size_m,
            corr_len_m=corr_len_m,
            seed=args.seed + scale_idx,
        )
        member_heads = []
        for member, logk in enumerate(logk_fields):
            model_ws = out_root / f"corr_{int(corr_len_m):04d}m" / f"member_{member:03d}"
            result = run_spatial_k_member(model_ws, logk, recharge, config)
            obs_head = np.asarray(result["obs_head"], dtype=float)
            member_heads.append(obs_head)
            member_rows.append(
                {
                    "corr_len_m": corr_len_m,
                    "member": member,
                    "harmonic_k": float(result["harmonic_k"]),
                    "local_k": float(result["local_k"]),
                    "logk_std": float(result["logk_std"]),
                    "head_mean": float(np.mean(obs_head)),
                    "head_std": float(np.std(obs_head, ddof=1)),
                    "head_amplitude": float(np.ptp(obs_head)),
                    "head_peak_day": int(np.argmax(obs_head)),
                    "head_min_day": int(np.argmin(obs_head)),
                }
            )

        heads = np.asarray(member_heads, dtype=float)
        heads_by_scale[corr_len_m] = heads
        logk_by_scale[corr_len_m] = logk_fields
        summary = summarize_member_heads(heads)
        harmonic_values = [float(row["harmonic_k"]) for row in member_rows if row["corr_len_m"] == corr_len_m]
        summary.update(
            {
                "corr_len_m": corr_len_m,
                "n_members": args.n_members,
                "mean_harmonic_k": float(np.mean(harmonic_values)),
                "mean_arithmetic_k": float(np.mean(np.exp(logk_fields))),
                "mean_logk_std": float(np.mean(np.std(logk_fields, axis=1, ddof=1))),
            }
        )
        scale_rows.append(summary)

    member_df = pd.DataFrame(member_rows)
    scale_df = pd.DataFrame(scale_rows)
    member_csv = out_root / "spatial_k_member_metrics.csv"
    scale_csv = out_root / "spatial_k_scale_summary.csv"
    member_df.to_csv(member_csv, index=False)
    scale_df.to_csv(scale_csv, index=False)

    np.savez(
        out_root / "spatial_k_smoke_arrays.npz",
        recharge=recharge,
        scales=np.asarray(scales, dtype=float),
        **{f"heads_{int(scale)}m": value for scale, value in heads_by_scale.items()},
        **{f"logk_{int(scale)}m": value for scale, value in logk_by_scale.items()},
    )

    days = np.arange(args.n_days)
    x = np.arange(args.n_cells) * args.cell_size_m
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    ax_k, ax_h, ax_spread, ax_amp = axes.ravel()

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(scales)))
    for color, corr_len_m in zip(colors, scales):
        logk_fields = logk_by_scale[corr_len_m]
        for idx in range(min(4, args.n_members)):
            ax_k.plot(x, np.exp(logk_fields[idx]), color=color, alpha=0.35, lw=1.0)
        ax_k.plot([], [], color=color, label=f"L={corr_len_m:g} m")
    ax_k.axvline(args.obs_col * args.cell_size_m, color="black", ls="--", lw=0.8, alpha=0.7)
    ax_k.set_title("Sample K-field transects by correlation scale")
    ax_k.set_xlabel("Distance from no-flow divide (m)")
    ax_k.set_ylabel("K (m/day)")
    ax_k.set_yscale("log")
    ax_k.grid(alpha=0.16)
    ax_k.legend(fontsize=8)

    ax_rain = ax_h.twinx()
    ax_rain.bar(days, recharge * 1000.0, width=1.0, color="lightgray", alpha=0.45, label="recharge")
    for color, corr_len_m in zip(colors, scales):
        heads = heads_by_scale[corr_len_m]
        mean_head = heads.mean(axis=0)
        lo = np.percentile(heads, 10, axis=0)
        hi = np.percentile(heads, 90, axis=0)
        ax_h.plot(days, mean_head, color=color, lw=1.6, label=f"L={corr_len_m:g} m")
        ax_h.fill_between(days, lo, hi, color=color, alpha=0.12)
    ax_h.set_title("Observation-well head response under identical recharge")
    ax_h.set_xlabel("Day")
    ax_h.set_ylabel("Head (m)")
    ax_rain.set_ylabel("Recharge (mm/day)")
    ax_h.grid(alpha=0.16)
    ax_h.legend(fontsize=8)

    ax_spread.plot(scale_df["corr_len_m"], scale_df["mean_temporal_std"], marker="o", label="mean temporal ensemble std")
    ax_spread.plot(scale_df["corr_len_m"], scale_df["max_temporal_std"], marker="o", label="max temporal ensemble std")
    ax_spread.set_xscale("log")
    ax_spread.set_xlabel("K-field correlation length (m)")
    ax_spread.set_ylabel("Head spread (m)")
    ax_spread.set_title("Correlation scale changes ensemble head spread")
    ax_spread.grid(alpha=0.16)
    ax_spread.legend(fontsize=8)

    ax_amp.plot(scale_df["corr_len_m"], scale_df["mean_member_amplitude"], marker="o", label="mean member amplitude")
    ax_amp.plot(scale_df["corr_len_m"], scale_df["max_member_range"], marker="o", label="max inter-member range")
    ax_amp.set_xscale("log")
    ax_amp.set_xlabel("K-field correlation length (m)")
    ax_amp.set_ylabel("Head response metric (m)")
    ax_amp.set_title("Same forcing, different spatial K prior")
    ax_amp.grid(alpha=0.16)
    ax_amp.legend(fontsize=8)

    fig.suptitle(
        "Synthetic spatial-K smoke test: K correlation scale affects transient head response",
        fontsize=14,
    )
    fig.tight_layout()
    fig_path = out_root / "spatial_k_smoke.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")

    summary = {
        "purpose": (
            "Synthetic smoke test showing that spatial K-field covariance assumptions can change "
            "transient head responses under identical recharge; not a real-domain inversion."
        ),
        "corr_scales_m": scales,
        "n_members": args.n_members,
        "n_cells": args.n_cells,
        "cell_size_m": args.cell_size_m,
        "obs_col": args.obs_col,
        "fixed_recharge_fraction": 0.20,
        "member_csv": str(member_csv),
        "scale_csv": str(scale_csv),
        "figure": str(fig_path),
    }
    (out_root / "spatial_k_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

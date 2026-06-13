from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


VARIANT_LABELS = {
    "persistence": "Persistence",
    "ode_only": "Standalone ODE",
    "gru": "GRU",
    "ws2": "GRU+WS2",
    "ode": "GRU+ODE-loss",
}

VARIANT_ORDER = ["persistence", "ode_only", "ws2", "gru", "ode"]
REGIME_ORDER = ["bedrock", "alluvial"]
REGIME_LABELS = {"bedrock": "Bedrock", "alluvial": "Alluvial"}
COLORS = {
    "persistence": "#6B7280",
    "ode_only": "#2F6F6D",
    "gru": "#4C78A8",
    "ws2": "#F58518",
    "ode": "#B279A2",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_sources(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "group": pd.read_csv(root / "canonical80_clean_group_summary_by_variant.csv"),
        "regime": pd.read_csv(root / "canonical80_clean_regime_summary.csv"),
        "regime_comparison": pd.read_csv(root / "canonical80_clean_regime_comparison.csv"),
        "paired": pd.read_csv(root / "canonical80_clean_paired_stats.csv"),
    }


def figure_central_summary(data: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    group = data["group"].copy()
    group["_order"] = group["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)})
    group = group.sort_values("_order").reset_index(drop=True)

    x = np.arange(len(group))
    values = group["rmse_mean"].to_numpy(dtype=float)
    medians = group["rmse_median"].to_numpy(dtype=float)
    colors = [COLORS[v] for v in group["variant"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bars = ax.bar(x, values, color=colors, edgecolor="#1f2937", linewidth=0.8)
    ax.scatter(
        x,
        medians,
        marker="D",
        s=44,
        color="white",
        edgecolor="#111827",
        zorder=3,
        label="Median RMSE",
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.014,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in group["variant"]], rotation=18, ha="right")
    ax.set_ylabel("Mean recursive RMSE (m)", fontweight="bold", labelpad=8)
    ax.set_xlabel("Matched central arm", fontweight="bold", labelpad=8)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.91), frameon=False)

    paths = []
    for ext in ["png", "pdf"]:
        path = out_dir / f"fig07_clean_canonical80_central_summary.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def figure_regime_paired(data: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    regime = data["regime"].copy()
    regime["_variant_order"] = regime["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)})
    regime["_regime_order"] = regime["material_group"].map({v: i for i, v in enumerate(REGIME_ORDER)})
    regime = regime.sort_values(["_regime_order", "_variant_order"])

    paired = data["paired"].copy()
    keep = [
        ("bedrock", "ode_vs_gru"),
        ("bedrock", "ws2_vs_gru"),
        ("alluvial", "ode_vs_gru"),
        ("alluvial", "ode_vs_ws2"),
        ("alluvial", "ode_only_vs_persistence"),
    ]
    paired = paired[
        (paired["scope_type"] == "material_group")
        & (paired[["scope_value", "comparison"]].apply(tuple, axis=1).isin(keep))
    ].copy()
    paired["_order"] = paired.apply(lambda row: keep.index((row["scope_value"], row["comparison"])), axis=1)
    paired = paired.sort_values("_order").reset_index(drop=True)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.4, 4.8),
        gridspec_kw={"width_ratios": [1.1, 1.35]},
        constrained_layout=True,
    )

    width = 0.16
    x_base = np.arange(len(REGIME_ORDER))
    offsets = np.linspace(-2, 2, len(VARIANT_ORDER)) * width
    for idx, variant in enumerate(VARIANT_ORDER):
        subset = regime[regime["variant"] == variant].set_index("material_group").loc[REGIME_ORDER]
        ax0.bar(
            x_base + offsets[idx],
            subset["rmse_mean"].to_numpy(dtype=float),
            width=width,
            color=COLORS[variant],
            edgecolor="#1f2937",
            linewidth=0.5,
            label=VARIANT_LABELS[variant],
        )
    ax0.set_xticks(x_base)
    ax0.set_xticklabels([REGIME_LABELS[g] for g in REGIME_ORDER])
    ax0.set_ylabel("Mean recursive RMSE (m)", fontweight="bold", labelpad=8)
    ax0.set_xlabel("Hydrogeologic group", fontweight="bold", labelpad=8)
    ax0.set_title("A", loc="left", pad=10)
    ax0.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
    ax0.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, frameon=False)

    y = np.arange(len(paired))
    diffs = paired["diff_mean"].to_numpy(dtype=float)
    low = paired["diff_mean_bootstrap_ci95_low"].to_numpy(dtype=float)
    high = paired["diff_mean_bootstrap_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([diffs - low, high - diffs])
    colors = ["#B279A2", "#F58518", "#B279A2", "#B279A2", "#2F6F6D"]
    ax1.axvline(0, color="#111827", linewidth=1.0)
    ax1.errorbar(
        diffs,
        y,
        xerr=xerr,
        fmt="o",
        markersize=6,
        capsize=4,
        color="#111827",
        ecolor="#4b5563",
        linewidth=1.2,
        zorder=2,
    )
    ax1.scatter(diffs, y, s=52, color=colors, edgecolor="#111827", zorder=3)
    labels = []
    for _, row in paired.iterrows():
        label = {
            "ode_vs_gru": "ODE-loss - GRU",
            "ws2_vs_gru": "WS2 - GRU",
            "ode_vs_ws2": "ODE-loss - WS2",
            "ode_only_vs_persistence": "Standalone ODE - persistence",
        }[row["comparison"]]
        labels.append(f"{REGIME_LABELS[row['scope_value']]}: {label}")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Paired mean RMSE difference (m)", fontweight="bold", labelpad=8)
    ax1.set_title("B", loc="left", pad=10)
    ax1.grid(axis="x", color="#d1d5db", linewidth=0.6, alpha=0.8)
    for yi, row in enumerate(paired.itertuples(index=False)):
        text = f"Holm p={row.wilcoxon_p_holm_all:.3g}"
        ax1.text(
            high[yi] + 0.01,
            yi,
            text,
            va="center",
            ha="left",
            fontsize=8.5,
            color="#374151",
        )
    ax1.set_xlim(min(low) - 0.03, max(high) + 0.18)

    paths = []
    for ext in ["png", "pdf"]:
        path = out_dir / f"fig08_clean_canonical80_regime_paired.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("resubmit/results/central_matched_50well_3seed_canonical80_20260610"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("resubmit/results/clean_canonical80_figures_20260611"),
    )
    args = parser.parse_args()

    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = read_sources(args.source_root)
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=mpl.rcParams["font.serif"]),
        fallback_to_default=True,
    )
    paths = []
    paths.extend(figure_central_summary(data, args.out_dir))
    paths.extend(figure_regime_paired(data, args.out_dir))

    manifest = {
        "status": "pass",
        "placement_assumption": "full-width manuscript quantitative result figures",
        "source_root": str(args.source_root),
        "outputs": [str(path) for path in paths],
        "source_files": [
            "canonical80_clean_group_summary_by_variant.csv",
            "canonical80_clean_regime_summary.csv",
            "canonical80_clean_regime_comparison.csv",
            "canonical80_clean_paired_stats.csv",
        ],
        "font_family_requested": mpl.rcParams["font.serif"],
        "font_path_resolved": font_path,
        "claim_boundary": (
            "Figures show clean canonical80 performance and paired/regime caution; "
            "they do not promote neural ODE-loss to physical credit."
        ),
    }
    (args.out_dir / "clean_canonical80_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for path in paths:
        print(path)
    print(args.out_dir / "clean_canonical80_figure_manifest.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build Figure 5 with the R1-S23 component-alignment diagnostic.

The figure uses the existing nuisance-collapse parameter table for coefficient
panels and the source-audited R1-S23 component rows for the center panel. It
does not rerun model training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARAMETER_TABLE = ROOT / "results/nuisance_collapse/parameter_table.csv"
COMPONENT_DIR = ROOT / "resubmit/results/r1_s23_component_evidence_20260611"
COMPONENT_ROWS = COMPONENT_DIR / "r1_s23_component_rows.anonymized.csv"
COMPONENT_VERIFICATION = COMPONENT_DIR / "r1_s23_component_verification.json"
OUT_LOCAL_PNG = COMPONENT_DIR / "fig05_nuisance_parameter_collapse.png"
OUT_LOCAL_PDF = COMPONENT_DIR / "fig05_nuisance_parameter_collapse.pdf"
OUT_FIGURES = ROOT / "results/figures_v6/fig05_nuisance_parameter_collapse.png"
OUT_SUBMISSION = ROOT / "submission/fig05_nuisance_parameter_collapse.png"
OUT_SUBMISSION_V9 = ROOT / "submission_v9_package/figures/fig05_nuisance_parameter_collapse.png"
MANIFEST = COMPONENT_DIR / "fig05_component_alignment_manifest.json"


COLORS = {
    "Bedrock": "#4C78A8",
    "Alluvial": "#C44E52",
}


def classify_aquifer(well_name: str) -> str:
    text = str(well_name)
    if "암반" in text:
        return "Bedrock"
    if "충적" in text:
        return "Alluvial"
    return "Other"


def mean_value(df: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(df[column], errors="coerce").dropna().mean())


def build_figure() -> dict[str, object]:
    params = pd.read_csv(PARAMETER_TABLE)
    rows = pd.read_csv(COMPONENT_ROWS)
    verification = json.loads(COMPONENT_VERIFICATION.read_text())

    if verification.get("status") != "pass":
        raise RuntimeError(f"Component verification is not pass: {verification}")
    if len(rows) != 150 or int((rows["status"] == "ok").sum()) != 150:
        raise RuntimeError("Expected 150 ok component rows for Figure 5.")

    params = params.copy()
    params["aquifer_type"] = params["well"].map(classify_aquifer)
    params["gamma_d_href"] = params["physics_gamma_d"] * params["physics_h_ref"]
    rows = rows[rows["status"] == "ok"].copy()

    means = {
        "rainfall": mean_value(rows, "corr_delta_rainfall_term"),
        "recession": mean_value(rows, "corr_delta_recession_term"),
        "full_rhs": mean_value(rows, "corr_delta_rhs"),
        "rhs_to_delta_median": float(pd.to_numeric(rows["rhs_to_delta_ratio"], errors="coerce").median()),
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), constrained_layout=True)

    ax = axes[0]
    for aquifer, sub in params.groupby("aquifer_type"):
        ax.scatter(
            sub["physics_gamma_d"],
            sub["physics_gamma_r"],
            s=34,
            alpha=0.68,
            color=COLORS.get(aquifer, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            label=aquifer,
        )
    ax.set_title("(a) Learned coefficients")
    ax.set_xlabel(r"Recession coefficient, $\gamma_d$ (d$^{-1}$)")
    ax.set_ylabel(r"Rainfall-response coefficient, $\gamma_r$ (m mm$^{-1}$)")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(alpha=0.18, linewidth=0.7)

    ax = axes[1]
    component_columns = [
        ("Rainfall", "corr_delta_rainfall_term", COLORS["Bedrock"]),
        ("Recession", "corr_delta_recession_term", "#7F7F7F"),
        ("Full RHS", "corr_delta_rhs", COLORS["Alluvial"]),
    ]
    data = [pd.to_numeric(rows[col], errors="coerce").dropna().to_numpy() for _, col, _ in component_columns]
    box = ax.boxplot(
        data,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, (_, _, color) in zip(box["boxes"], component_columns):
        patch.set_facecolor(color)
        patch.set_alpha(0.38)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(20260611)
    for idx, ((label, col, color), values) in enumerate(zip(component_columns, data), start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full_like(values, idx, dtype=float) + jitter,
            values,
            s=13,
            alpha=0.34,
            color=color,
            edgecolor="none",
        )
        mean = float(np.mean(values))
        ax.scatter([idx], [mean], s=64, marker="D", color=color, edgecolor="black", linewidth=0.45, zorder=4)
        ax.text(idx, mean + 0.035, f"mean={mean:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#666666", linewidth=0.9)
    ax.set_xticks([1, 2, 3], [item[0] for item in component_columns])
    ax.set_ylim(-0.35, 0.8)
    ax.set_title(r"(b) Component alignment")
    ax.set_ylabel(r"Correlation with predicted $\Delta h_t$")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)

    ax = axes[2]
    for aquifer, sub in params.groupby("aquifer_type"):
        ax.scatter(
            sub["physics_h_ref"],
            sub["gamma_d_href"],
            s=34,
            alpha=0.68,
            color=COLORS.get(aquifer, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            label=aquifer,
        )
    ax.axhline(0, color="#888888", linewidth=0.9)
    ax.set_title(r"(c) Reference-head offset")
    ax.set_xlabel(r"Reference head, $h_{\mathrm{ref}}$ (m)")
    ax.set_ylabel(r"Offset contribution, $\gamma_d h_{\mathrm{ref}}$ (m d$^{-1}$)")
    ax.grid(alpha=0.18, linewidth=0.7)

    fig.savefig(OUT_LOCAL_PNG, dpi=300)
    fig.savefig(OUT_LOCAL_PDF)
    plt.close(fig)

    OUT_FIGURES.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUBMISSION_V9.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_LOCAL_PNG, OUT_FIGURES)
    shutil.copyfile(OUT_LOCAL_PNG, OUT_SUBMISSION)
    shutil.copyfile(OUT_LOCAL_PNG, OUT_SUBMISSION_V9)

    manifest = {
        "status": "pass",
        "source_parameter_table": str(PARAMETER_TABLE.relative_to(ROOT)),
        "source_component_rows": str(COMPONENT_ROWS.relative_to(ROOT)),
        "source_component_verification": str(COMPONENT_VERIFICATION.relative_to(ROOT)),
        "outputs": [
            str(OUT_LOCAL_PNG.relative_to(ROOT)),
            str(OUT_LOCAL_PDF.relative_to(ROOT)),
            str(OUT_FIGURES.relative_to(ROOT)),
            str(OUT_SUBMISSION.relative_to(ROOT)),
            str(OUT_SUBMISSION_V9.relative_to(ROOT)),
        ],
        "component_rows": int(len(rows)),
        "parameter_rows": int(len(params)),
        "component_means": means,
        "claim_boundary": (
            "Figure 5 visualizes component alignment for the stored LSTM+ODE "
            "rollouts. It supports R1-S23 component-evidence wording but does "
            "not prove physical parameter identifiability."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    manifest = build_figure()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

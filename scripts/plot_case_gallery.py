from __future__ import annotations

import math
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.virtual_aquifer import DEFAULT_GROUNDWATER_ROOT


def configure_korean_font() -> None:
    for name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def _load_series(stem: str) -> pd.DataFrame:
    wt_path = DEFAULT_GROUNDWATER_ROOT / "waterlevel" / f"{stem}_WT.txt"
    cl_path = DEFAULT_GROUNDWATER_ROOT / "climate" / f"{stem}_CL.txt"
    if not wt_path.exists():
        wt_path = DEFAULT_GROUNDWATER_ROOT / "waterlevel" / unicodedata.normalize("NFD", f"{stem}_WT.txt")
    if not cl_path.exists():
        cl_path = DEFAULT_GROUNDWATER_ROOT / "climate" / unicodedata.normalize("NFD", f"{stem}_CL.txt")
    wt_df = pd.read_csv(wt_path, sep="\t")
    cl_df = pd.read_csv(cl_path, sep="\t")
    df = wt_df.merge(cl_df, on="Date", how="inner")
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    df["head"] = pd.to_numeric(df["Value"], errors="coerce")
    df["RAIN"] = pd.to_numeric(df["RAIN"], errors="coerce").fillna(0.0)
    return df


def main() -> None:
    configure_korean_font()
    shortlist = pd.read_csv(ROOT / "results/data_screening/main_case_shortlist.csv")
    n = len(shortlist)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), sharex=False)
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, shortlist.iterrows()):
        df = _load_series(row["stem"])
        ax.plot(df["date"], df["head"], color="#154c79", lw=1.2, label="Groundwater level")
        ax.set_title(
            f"{row['stem']} | {row['archetype_suggested']} | std={row['waterlevel_std']:.2f}",
            fontsize=10,
        )
        ax.set_ylabel("Head")
        rain_ax = ax.twinx()
        rain_ax.fill_between(df["date"], 0.0, df["RAIN"], color="#7fb3d5", alpha=0.18)
        rain_ax.set_ylim(0, max(float(df["RAIN"].quantile(0.98)) * 3.0, 1.0))
        rain_ax.set_yticks([])
        ax.grid(alpha=0.18)
        text = (
            f"{row['record_length_days']} d\n"
            f"{row['material_class']} / {row['metadata_match_mode']}\n"
            f"{row['selection_rationale']}"
        )
        ax.text(
            0.01,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Recommended Main-Text Base Sites", fontsize=14, y=0.995)
    fig.tight_layout()
    out = ROOT / "results/data_screening/main_case_gallery.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()

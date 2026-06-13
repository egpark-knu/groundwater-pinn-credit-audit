from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


FIGURE_FILENAMES = {
    "fig01": "fig01_study_area_wells.png",
    "fig02": "fig02_hydrograph_fits.png",
    "fig03": "fig03_lambda_sensitivity.png",
    "fig04": "fig04_architecture_winner_heatmap.png",
    "fig05": "fig05_whittaker_vs_ode.png",
    "fig06": "fig06_solver_parameter_compensation.png",
    "fig07": "fig07_physical_credit_ladder.png",
}

LOCKED_WELLS = [
    "거제신현_암반",
    "영덕도천_천부_충적",
    "창원북면_충적",
    "안동태화_충적",
    "영덕달산_암반",
    "울진울진_암반",
]
WELL_LABELS = {
    "거제신현_암반": "W006, bedrock",
    "영덕도천_천부_충적": "W005, alluvial",
    "창원북면_충적": "W002, alluvial",
    "안동태화_충적": "W001, alluvial",
    "영덕달산_암반": "W003, bedrock",
    "울진울진_암반": "W004, bedrock",
}
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "patchtst": "PatchTST",
    "lstm_ode": "LSTM+ODE",
    "lstm_ws2": "LSTM+WS2",
}
MODEL_COLORS = {
    "lstm": "#2f6f63",
    "gru": "#6f6f6f",
    "patchtst": "#1f5a85",
    "lstm_ode": "#b33a3a",
    "lstm_ws2": "#7f4c99",
}

GEODATA_ROOT = Path(os.environ.get("KOREA_GEODATA_ROOT", ROOT / "data" / "geodata"))
ARCH_ROOT = ROOT / "results/architecture_diversity_3seed"
WVO_ROOT = ROOT / "results/whittaker_vs_ode"


def configure_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 300
    rcParams["axes.labelweight"] = "bold"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def figure_label(stem: str, *, multiline: bool = False) -> str:
    label = WELL_LABELS[stem]
    return label.replace(", ", "\n") if multiline else label


def load_model_contrast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["stem"] = df["stem"].map(nfc)
    return df[df["model_family"] == "delta_recursive_w30"].copy()


def select_best_ode_seed_by_well(seed_table: pd.DataFrame) -> dict[str, int]:
    df = seed_table.copy()
    if "stem" not in df.columns and "well" in df.columns:
        df = df.rename(columns={"well": "stem"})
    df["stem"] = df["stem"].map(nfc)
    ode = df[df["variant"] == "ode"].copy()
    idx = ode.groupby("stem")["rmse"].idxmin()
    return {str(row["stem"]): int(row["seed"]) for _, row in ode.loc[idx].iterrows()}


def load_six_well_architecture_summary(root: Path = ARCH_ROOT) -> pd.DataFrame:
    path = root / "architecture_diversity_smoke_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["well"] = df["well"].map(nfc)
    return df[df["well"].isin(LOCKED_WELLS)].copy()


def load_six_well_architecture_group(root: Path = ARCH_ROOT) -> pd.DataFrame:
    path = root / "architecture_diversity_group_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["well"] = df["well"].map(nfc)
    return df[df["well"].isin(LOCKED_WELLS)].copy()


def load_six_well_mean_winners(root: Path = ARCH_ROOT) -> pd.DataFrame:
    path = root / "architecture_diversity_mean_winners.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["well"] = df["well"].map(nfc)
    return df[df["well"].isin(LOCKED_WELLS)].copy()


def load_whittaker_vs_ode_summary(root: Path = WVO_ROOT) -> pd.DataFrame:
    path = root / "whittaker_vs_ode_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["well"] = df["well"].map(nfc)
    return df[df["well"].isin(LOCKED_WELLS)].copy()


def load_whittaker_vs_ode_comparison(root: Path = WVO_ROOT) -> pd.DataFrame:
    path = root / "whittaker_vs_ode_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["well"] = df["well"].map(nfc)
    return df[df["well"].isin(LOCKED_WELLS)].copy()


def _stem_dir(root: Path, stem: str) -> Path:
    stem = nfc(stem)
    direct = root / stem
    if direct.exists():
        return direct
    candidates = {nfc(p.name): p for p in root.iterdir() if p.is_dir()}
    if stem not in candidates:
        raise FileNotFoundError(f"No directory for stem {stem} under {root}")
    return candidates[stem]


def load_architecture_rollout(stem: str, model: str, seed: int, root: Path = ARCH_ROOT) -> dict[str, np.ndarray]:
    run_dir = _stem_dir(root, stem) / f"{model}_seed{seed}"
    with np.load(run_dir / "test_rollout_predictions.npz", allow_pickle=False) as data:
        return {
            "dates": data["dates"].astype("datetime64[D]"),
            "obs": data["obs"].astype(float),
            "pred": data["pred"].astype(float),
        }


def load_well_points() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(ROOT / "results/data_screening/groundwater_case_catalog.gpkg").to_crs("EPSG:5186")
    gdf["stem"] = gdf["stem"].map(nfc)
    wells = gdf[gdf["stem"].isin(LOCKED_WELLS)].copy()
    missing = sorted(set(LOCKED_WELLS) - set(wells["stem"]))
    if missing:
        raise ValueError(f"Missing well geometry rows: {missing}")
    return wells


def generate_site_figure(output: Path) -> dict:
    wells = load_well_points()
    sido = gpd.read_file(GEODATA_ROOT / "sido_all.gpkg").to_crs("EPSG:5186")
    sigungu = gpd.read_file(GEODATA_ROOT / "sigungu_all.gpkg").to_crs("EPSG:5186")
    districts = sigungu[sigungu.intersects(wells.union_all().convex_hull.buffer(90000))]

    fig = plt.figure(figsize=(11.2, 7.4))
    ax = fig.add_axes([0.07, 0.08, 0.63, 0.84])
    inset = fig.add_axes([0.735, 0.59, 0.22, 0.31])
    info = fig.add_axes([0.725, 0.08, 0.25, 0.43])

    districts.plot(ax=ax, color="#f2efe8", edgecolor="#b6b0a7", linewidth=0.55)
    wells.plot(ax=ax, color="#bb2f2a", edgecolor="white", linewidth=0.8, markersize=78, zorder=5)

    ordered = wells.set_index("stem").loc[LOCKED_WELLS].reset_index()
    for idx, row in ordered.iterrows():
        x, y = row.geometry.x, row.geometry.y
        ax.text(x + 4200, y + 3200, f"{idx + 1}", fontsize=9, weight="bold", color="#222222")
    minx, miny, maxx, maxy = wells.total_bounds
    ax.set_xlim(minx - 90000, maxx + 90000)
    ax.set_ylim(miny - 70000, maxy + 90000)
    ax.set_xlabel("EPSG:5186 Easting (m)")
    ax.set_ylabel("EPSG:5186 Northing (m)")
    ax.grid(alpha=0.18)
    ax.annotate("N", xy=(0.95, 0.90), xytext=(0.95, 0.78), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.5}, ha="center")
    ax.plot([0.08, 0.22], [0.07, 0.07], transform=ax.transAxes, color="black", lw=2)
    ax.text(0.15, 0.087, "~50 km", transform=ax.transAxes, ha="center", fontsize=8)

    sido.plot(ax=inset, color="#f8f8f8", edgecolor="#999999", linewidth=0.35)
    wells.plot(ax=inset, color="#bb2f2a", markersize=18)
    inset.set_axis_off()

    info.set_axis_off()
    info.set_xlim(0, 1)
    info.set_ylim(0, 1)
    info.text(0.0, 1.0, "Locked wells", fontsize=11, weight="bold", va="top")
    for i, stem in enumerate(LOCKED_WELLS):
        y = 0.88 - i * 0.14
        name, aquifer = WELL_LABELS[stem].split(", ", 1)
        info.scatter([0.04], [y], s=52, color="#bb2f2a")
        info.text(0.12, y + 0.028, f"{i + 1}. {name}", fontsize=8.4, weight="bold", va="center")
        info.text(0.12, y - 0.032, aquifer, fontsize=7.8, va="center", color="#444444")

    fig.savefig(output)
    plt.close(fig)
    return {"wells": int(len(wells)), "source": "groundwater_case_catalog.gpkg + Korean administrative boundaries"}


def _representative_seed(summary: pd.DataFrame, stem: str, model: str, mean_rmse: float) -> int:
    sub = summary[(summary["well"] == stem) & (summary["model"] == model)].copy()
    sub["distance_to_mean"] = (sub["rmse"] - mean_rmse).abs()
    return int(sub.sort_values(["distance_to_mean", "rmse", "seed"]).iloc[0]["seed"])


def generate_hydrograph_fits(output: Path) -> dict:
    summary = load_six_well_architecture_summary()
    winners = load_six_well_mean_winners().set_index("well").loc[LOCKED_WELLS].reset_index()

    fig, axes = plt.subplots(3, 2, figsize=(13.4, 10.2), sharex=False)
    axes = axes.ravel()
    manifest: dict[str, dict] = {}
    for ax, row in zip(axes, winners.itertuples(index=False)):
        stem = str(row.well)
        model = str(row.model)
        seed = _representative_seed(summary, stem, model, float(row.rmse_mean))
        rollout = load_architecture_rollout(stem, model, seed)
        ax.scatter(rollout["dates"], rollout["obs"], s=5.5, color="black", alpha=0.42)
        ax.plot(rollout["dates"], rollout["pred"], color=MODEL_COLORS[model], lw=1.15)
        ax.set_ylabel("Head (m)", fontsize=12)
        ax.set_title(WELL_LABELS[stem], loc="left", fontsize=14, weight="bold")
        ax.tick_params(axis="both", labelsize=10)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
        ax.grid(alpha=0.18)
        ax.text(
            0.01,
            0.91,
            f"3-seed winner: {MODEL_LABELS[model]}\nmean RMSE={row.rmse_mean:.3f} m; shown seed={seed}",
            transform=ax.transAxes,
            fontsize=8.2,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#cccccc"},
        )
        manifest[WELL_LABELS[stem]] = {"winner": model, "representative_seed": seed, "rmse_mean": float(row.rmse_mean)}
    axes[-2].set_xlabel("Test date", fontsize=12)
    axes[-1].set_xlabel("Test date", fontsize=12)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5.5, markerfacecolor="black", markeredgecolor="black", alpha=0.55, label="Observed"),
        Line2D([0], [0], color=MODEL_COLORS["lstm"], lw=2.0, label=MODEL_LABELS["lstm"]),
        Line2D([0], [0], color=MODEL_COLORS["gru"], lw=2.0, label=MODEL_LABELS["gru"]),
        Line2D([0], [0], color=MODEL_COLORS["patchtst"], lw=2.0, label=MODEL_LABELS["patchtst"]),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False, fontsize=11)
    fig.suptitle("Representative recursive 7-day hydrograph fits using each well's 3-seed mean winner", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return manifest


def generate_lambda_figure(output: Path) -> dict:
    path = ROOT / "results/patchtst_sweep_w30/patchtst_sweep_group_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["well"] = df["well"].map(nfc)
    wells = [w for w in LOCKED_WELLS if w in set(df["well"])]
    wells = [w for w in wells if w in {"거제신현_암반", "영덕도천_천부_충적", "창원북면_충적"}]
    x_map = {0.0: 1e-4, 0.001: 0.001, 0.01: 0.01, 0.1: 0.1, 1.0: 1.0}

    fig, axes = plt.subplots(1, len(wells), figsize=(17.8, 5.9), sharey=False)
    if len(wells) == 1:
        axes = [axes]
    for ax, stem in zip(axes, wells):
        sub = df[df["well"] == stem].copy()
        for model, color, label, marker in [
            ("patchtst_ode", MODEL_COLORS["lstm_ode"], "PatchTST+ODE", "o"),
            ("patchtst_ws2", MODEL_COLORS["lstm_ws2"], "PatchTST+WS2", "s"),
        ]:
            rows = sub[sub["model"] == model].sort_values("lambda")
            x = rows["lambda"].map(x_map).to_numpy(dtype=float)
            y = rows["rmse_mean"].to_numpy(dtype=float)
            yerr = rows["rmse_std"].fillna(0).to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=yerr, marker=marker, ms=8.5, lw=2.4, capsize=4.5, color=color, label=label)

        plain = sub[sub["model"] == "patchtst"]
        if not plain.empty:
            y_plain = float(plain["rmse_mean"].iloc[0])
            ax.axhline(y_plain, color=MODEL_COLORS["patchtst"], lw=2.0, ls="--", label="Plain PatchTST")

        ax.set_xscale("log")
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ["0", "0.001", "0.01", "0.1", "1.0"], fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(figure_label(stem, multiline=True), fontsize=15, weight="bold")
        ax.set_xlabel("Loss weight λ", fontsize=15)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Recursive 7-day RMSE (m)", fontsize=15)
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend_.remove() if axes[-1].legend_ else None
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"source": str(path), "redrawn": True, "font_scale": "large", "wells": [WELL_LABELS[w] for w in wells], "bytes": output.stat().st_size}


def _rank_range_annotation(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["seed_rank"] = ranked.groupby(["well", "seed"])["rmse"].rank(method="min", ascending=True).astype(int)
    rank_ranges = (
        ranked.groupby(["well", "model"])
        .agg(rank_min=("seed_rank", "min"), rank_max=("seed_rank", "max"))
        .reset_index()
    )
    return rank_ranges


def generate_architecture_winner_heatmap(output: Path) -> dict:
    group = load_six_well_architecture_group()
    summary = load_six_well_architecture_summary()
    rank_ranges = _rank_range_annotation(summary)
    models = ["lstm", "gru", "patchtst"]

    pivot = group.pivot(index="well", columns="model", values="rmse_mean").loc[LOCKED_WELLS, models]
    ranks = rank_ranges.set_index(["well", "model"])
    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(models)), [MODEL_LABELS[m] for m in models])
    ax.set_yticks(np.arange(len(LOCKED_WELLS)), [WELL_LABELS[w] for w in LOCKED_WELLS])
    ax.set_xlabel("Plain architecture")
    ax.set_ylabel("Monitoring well")

    winner_counts: dict[str, int] = {}
    for i, stem in enumerate(LOCKED_WELLS):
        winner = pivot.loc[stem].idxmin()
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        for j, model in enumerate(models):
            rr = ranks.loc[(stem, model)]
            value = pivot.loc[stem, model]
            ax.text(j, i, f"{value:.3f}\nrank {int(rr.rank_min)}-{int(rr.rank_max)}", ha="center", va="center", fontsize=8, color="white" if value > pivot.values.mean() else "black")
            if model == winner:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="#f2c94c", linewidth=2.2))

    cbar = plt.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Mean recursive 7-day RMSE (m), lower is better")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"winner_counts": winner_counts, "models": models}


def generate_whittaker_vs_ode(output: Path) -> dict:
    comp = load_whittaker_vs_ode_comparison().set_index("well").loc[LOCKED_WELLS].reset_index()
    models = ["lstm", "lstm_ode", "lstm_ws2"]
    fig, ax = plt.subplots(figsize=(11.6, 5.7))
    x = np.arange(len(LOCKED_WELLS))
    width = 0.24
    offsets = [-width, 0.0, width]
    for model, offset in zip(models, offsets):
        ax.bar(x + offset, comp[model], width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.92)

    for i, row in comp.iterrows():
        winner = str(row["winner"])
        winner_idx = models.index(winner)
        ax.scatter(x[i] + offsets[winner_idx], row[winner], marker="*", s=90, color="#f2c94c", edgecolor="#333333", zorder=5)

    ws2_wins = int((comp["winner"] == "lstm_ws2").sum())
    ode_wins = int((comp["winner"] == "lstm_ode").sum())
    plain_wins = int((comp["winner"] == "lstm").sum())
    mean_abs_gap = float(comp["ode_minus_ws2_rmse"].abs().mean())
    ax.set_xticks(x, [figure_label(w, multiline=True) for w in LOCKED_WELLS], fontsize=8.8)
    ax.set_ylabel("Recursive 7-day RMSE (m)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=3)
    ax.text(
        0.01,
        0.96,
        f"λ=0.1 for both penalties; mean |ODE-WS2| = {mean_abs_gap:.4f} m",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"ws2_wins": ws2_wins, "ode_wins": ode_wins, "plain_wins": plain_wins, "mean_abs_ode_ws2_gap_m": mean_abs_gap}


def generate_solver_compensation(output: Path) -> dict:
    root = ROOT / "results/solver_audit/steady_rpr20_continuous_single_case"
    run_dirs = sorted([p for p in root.glob("*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(root)
    run_dir = run_dirs[0]
    prior = np.load(run_dir / "theta_prior.npy")
    posterior = np.load(run_dir / "theta_posterior.npy")
    corr = float(np.corrcoef(posterior[:, 0], posterior[:, 2])[0, 1])

    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    ax.scatter(
        prior[:, 0],
        prior[:, 2],
        facecolors="none",
        edgecolors="#7d8794",
        linewidth=1.5,
        s=88,
        label="Prior ensemble (open circles)",
        alpha=0.95,
    )
    sc = ax.scatter(
        posterior[:, 0],
        posterior[:, 2],
        c=np.arange(len(posterior)),
        cmap="magma",
        marker="^",
        edgecolor="black",
        linewidth=0.7,
        s=96,
        label="Posterior ensemble (filled triangles)",
        zorder=5,
    )
    ax.set_xlabel(r"$\log K_{\mathrm{eff}}$")
    ax.set_ylabel(r"$h_{\mathrm{ref}}$ (m)")
    ax.text(
        0.04,
        0.95,
        rf"posterior corr($\log K_{{\mathrm{{eff}}}}$, $h_{{\mathrm{{ref}}}}$) = {corr:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=True, loc="lower right", fontsize=9.5, facecolor="white", edgecolor="#cccccc", framealpha=0.92)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Posterior member index")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"posterior_corr_logk_href": corr, "n_members": int(len(posterior))}


def generate_physical_credit_ladder(output: Path) -> dict:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_axis_off()
    rungs = [
        ("1", "Predictive surrogate\nLSTM / PatchTST / GRU", "Can claim:\nrecursive skill\nCannot claim:\nphysical mechanism", "#d9e8f2"),
        ("2", "Structured regularizer\nWhittaker WS2", "Can claim:\ntrajectory smoothing\nCannot claim:\naquifer physics", "#e8e1f0"),
        ("3", "Reduced ODE-loss\nlumped dh/dt residual", "Can claim:\nfalsifiable soft physics\nOnly if:\nseparates from WS2", "#f2dedb"),
        ("4", "Physics-accountable model\nMODFLOW + ES-MDA", "Can claim:\nauditable assumptions\nExposes:\nparameter compensation", "#dfebdf"),
    ]
    xs = np.linspace(0.09, 0.84, len(rungs))
    for idx, ((num, title, text, color), x) in enumerate(zip(rungs, xs)):
        box = FancyBboxPatch((x, 0.33), 0.18, 0.42, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=color, edgecolor="#333333", linewidth=1.1)
        ax.add_patch(box)
        ax.text(x + 0.09, 0.69, f"Rung {num}", ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + 0.09, 0.58, title, ha="center", va="center", fontsize=9.6, weight="bold")
        ax.text(x + 0.09, 0.42, text, ha="center", va="center", fontsize=8.5)
        if idx < len(rungs) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.015, 0.54), xytext=(x + 0.195, 0.54), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#333333"})
    ax.text(0.50, 0.87, "Physical credit must be earned, not inferred from a physics-looking loss", ha="center", fontsize=14, weight="bold")
    ax.text(0.50, 0.18, "The ODE residual remains a hypothesis until it separates from matched regularization and exposes auditable physical behavior.", ha="center", fontsize=9.8, color="#333333")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"rungs": len(rungs), "rendering": "matplotlib_reproducible_schematic"}


def generate_all(output_dir: Path = ROOT / "results/figures") -> dict:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    generators = {
        "fig01": generate_site_figure,
        "fig02": generate_hydrograph_fits,
        "fig03": generate_lambda_figure,
        "fig04": generate_architecture_winner_heatmap,
        "fig05": generate_whittaker_vs_ode,
        "fig06": generate_solver_compensation,
        "fig07": generate_physical_credit_ladder,
    }
    manifest: dict[str, dict] = {}
    for key, generator in generators.items():
        path = output_dir / FIGURE_FILENAMES[key]
        meta = generator(path)
        manifest[key] = {"path": str(path), "bytes": path.stat().st_size, **meta}
        print(path)
    manifest_path = output_dir / "publication_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    manifest = generate_all()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

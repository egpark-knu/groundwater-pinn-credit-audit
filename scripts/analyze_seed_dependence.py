from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_INPUT = ROOT / "results/architecture_diversity_3seed/architecture_diversity_smoke_summary.csv"
DEFAULT_OUTPUT = ROOT / "results/seed_dependence_analysis"
DEFAULT_SEEDS = [7, 42, 99]


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _rank_correlation(ranks_a: pd.Series, ranks_b: pd.Series, method: str) -> float:
    aligned = pd.concat([ranks_a, ranks_b], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method))


def compute_seed_dependence(
    df: pd.DataFrame,
    seed_order: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_order = DEFAULT_SEEDS if seed_order is None else [int(seed) for seed in seed_order]
    data = df.copy()
    data["well"] = data["well"].map(nfc)
    data = data[data["seed"].isin(seed_order)].copy()
    data["rmse_rank"] = data.groupby(["well", "seed"])["rmse"].rank(method="min", ascending=True).astype(int)

    winner_rows = data.loc[data.groupby(["well", "seed"])["rmse"].idxmin()].copy()
    winner_rows = winner_rows.sort_values(["well", "seed"]).reset_index(drop=True)
    winner_rows = winner_rows.rename(columns={"model": "winner", "rmse": "winner_rmse"})
    winner_columns = ["well", "seed", "winner", "winner_rmse"]
    for optional in ["nse", "best_lag_days"]:
        if optional in winner_rows.columns:
            winner_columns.append(optional)
    winner_table = winner_rows[winner_columns].copy()

    summaries: list[dict] = []
    rank_corr_rows: list[dict] = []
    for well, well_df in data.groupby("well", sort=True):
        seed_winners = winner_table[winner_table["well"] == well].set_index("seed").reindex(seed_order)
        winner_sequence = [str(value) for value in seed_winners["winner"].dropna().tolist()]
        transition_count = sum(a != b for a, b in zip(winner_sequence[:-1], winner_sequence[1:]))
        distinct_winners = sorted(set(winner_sequence))

        rank_by_seed = {
            int(seed): rows.set_index("model")["rmse_rank"].sort_index()
            for seed, rows in well_df.groupby("seed")
        }
        spearmans: list[float] = []
        kendalls: list[float] = []
        for seed_a, seed_b in combinations(seed_order, 2):
            if seed_a not in rank_by_seed or seed_b not in rank_by_seed:
                continue
            spearman = _rank_correlation(rank_by_seed[seed_a], rank_by_seed[seed_b], method="spearman")
            kendall = _rank_correlation(rank_by_seed[seed_a], rank_by_seed[seed_b], method="kendall")
            spearmans.append(spearman)
            kendalls.append(kendall)
            rank_corr_rows.append(
                {
                    "well": well,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "spearman_rank_corr": spearman,
                    "kendall_tau": kendall,
                }
            )

        summaries.append(
            {
                "well": well,
                "seed_winner_sequence": " -> ".join(winner_sequence),
                "distinct_seed_winners": int(len(distinct_winners)),
                "winner_transition_count": int(transition_count),
                "mean_spearman_rank_corr": float(pd.Series(spearmans).mean()) if spearmans else float("nan"),
                "min_spearman_rank_corr": float(pd.Series(spearmans).min()) if spearmans else float("nan"),
                "mean_kendall_tau": float(pd.Series(kendalls).mean()) if kendalls else float("nan"),
                "min_kendall_tau": float(pd.Series(kendalls).min()) if kendalls else float("nan"),
            }
        )

    summary = pd.DataFrame(summaries).sort_values("well").reset_index(drop=True)
    rank_corr = pd.DataFrame(rank_corr_rows).sort_values(["well", "seed_a", "seed_b"]).reset_index(drop=True)
    return summary, winner_table, rank_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    summary, winner_table, rank_corr = compute_seed_dependence(df, seed_order=args.seeds)

    summary_csv = output_dir / "seed_dependence_summary.csv"
    winner_csv = output_dir / "seed_winner_table.csv"
    rank_csv = output_dir / "seed_rank_correlation.csv"
    summary.to_csv(summary_csv, index=False)
    winner_table.to_csv(winner_csv, index=False)
    rank_corr.to_csv(rank_csv, index=False)

    overall = {
        "input": str(args.input),
        "seed_order": args.seeds,
        "summary_csv": str(summary_csv),
        "seed_winner_table_csv": str(winner_csv),
        "seed_rank_correlation_csv": str(rank_csv),
        "n_wells": int(summary["well"].nunique()),
        "wells_with_seed_winner_switch": int((summary["distinct_seed_winners"] > 1).sum()),
        "mean_winner_transition_count": float(summary["winner_transition_count"].mean()),
        "mean_spearman_rank_corr": float(summary["mean_spearman_rank_corr"].mean()),
        "mean_kendall_tau": float(summary["mean_kendall_tau"].mean()),
    }
    (output_dir / "seed_dependence_manifest.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    research_dir = ROOT / "results/research_summaries"
    research_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(research_dir / "seed_dependence_summary.csv", index=False)
    winner_table.to_csv(research_dir / "seed_winner_table.csv", index=False)
    rank_corr.to_csv(research_dir / "seed_rank_correlation.csv", index=False)
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

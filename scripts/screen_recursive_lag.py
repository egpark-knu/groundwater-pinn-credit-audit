from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.direct_delta_lead import build_direct_delta_split, train_direct_delta_variant  # noqa: E402
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta, recursive_block_rollout_one_step_head  # noqa: E402
from groundwater_research.recursive_head import build_recursive_head_split, train_recursive_head_variant  # noqa: E402


def load_stems(args: argparse.Namespace) -> list[str]:
    if args.stems:
        return [s.strip() for s in args.stems.split(",") if s.strip()]
    if args.stems_csv:
        with open(args.stems_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [row["stem"] for row in reader]
    raise ValueError("Provide either --stems or --stems-csv.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-space", choices=["head", "delta"], required=True)
    ap.add_argument("--variant", choices=["gru", "ws1", "ws2", "ode"], required=True)
    ap.add_argument("--stems")
    ap.add_argument("--stems-csv")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--event-weight-scale", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    for stem in load_stems(args):
        try:
            series = load_ladder_series(stem)
            splits = make_block_splits(len(series.head_interp))
            if args.model_space == "head":
                split_data = build_recursive_head_split(series, splits, window=args.window)
                model, _, meta = train_recursive_head_variant(
                    split_data,
                    variant=args.variant,
                    seed=args.seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    hidden=args.hidden,
                    lr=args.lr,
                    lambda_penalty=args.lambda_penalty,
                )
                rollout = recursive_block_rollout_one_step_head(
                    model=model,
                    series=series,
                    split=splits.test,
                    norm=split_data["norm"],
                    window=args.window,
                    forecast_horizon=args.forecast_horizon,
                )
            else:
                split_data = build_direct_delta_split(series, splits, window=args.window, horizon=1)
                model, _, meta = train_direct_delta_variant(
                    split_data,
                    variant=args.variant,
                    seed=args.seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    hidden=args.hidden,
                    lr=args.lr,
                    lambda_penalty=args.lambda_penalty,
                    event_weight_scale=args.event_weight_scale,
                )
                rollout = recursive_block_rollout_one_step_delta(
                    model=model,
                    series=series,
                    split=splits.test,
                    norm=split_data["norm"],
                    window=args.window,
                    forecast_horizon=args.forecast_horizon,
                )
            row = {"stem": stem, **rollout["metrics"], **meta}
        except Exception as exc:
            row = {"stem": stem, "error": str(exc)}
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(output_path)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

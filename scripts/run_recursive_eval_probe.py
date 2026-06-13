from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.direct_delta_lead import build_direct_delta_split, train_direct_delta_variant  # noqa: E402
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.recursive_head import build_recursive_head_split, train_recursive_head_variant  # noqa: E402
from groundwater_research.recursive_eval import recursive_block_rollout_one_step_delta, recursive_block_rollout_one_step_head  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--model-space", choices=["head", "delta"], required=True)
    ap.add_argument("--variant", choices=["gru", "ws1", "ws2", "ode"], required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--event-weight-scale", type=float, default=0.0)
    ap.add_argument("--without-dhead", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    series = load_ladder_series(args.stem)
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
        split_data = build_direct_delta_split(
            series,
            splits,
            window=args.window,
            horizon=1,
            include_dhead=not args.without_dhead,
        )
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
            include_dhead=not args.without_dhead,
        )

    payload = {
        "meta": {
            **meta,
            "model_space": args.model_space,
            "window": args.window,
            "forecast_horizon": args.forecast_horizon,
        },
        "rollout": rollout["metrics"],
    }
    if args.output_dir:
        out_root = Path(args.output_dir) / series.stem / f"{args.variant}_seed{args.seed}"
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        np.savez(
            out_root / "test_rollout_predictions.npz",
            pred=np.asarray(rollout["pred"]),
            obs=np.asarray(rollout["obs"]),
            dates=np.asarray(rollout["dates"]),
        )
        print(out_root)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

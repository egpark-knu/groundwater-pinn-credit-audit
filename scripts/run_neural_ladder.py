from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.neural_ladder import (
    build_sequence_split,
    load_ladder_series,
    make_block_splits,
    rollout_sequence_model,
    save_ladder_run,
    train_ladder_variant,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--variant", choices=["gru", "ws1", "ws2", "ode"], required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--tau-days", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow-direct-lead-exploratory",
        action="store_true",
        help="Acknowledge that this script runs an exploratory direct h-step setup that is not allowed for main-paper evidence.",
    )
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "results/predictive_ladder"),
    )
    args = ap.parse_args()

    if not args.allow_direct_lead_exploratory:
        raise SystemExit(
            "run_neural_ladder.py is disabled for main research use. "
            "It trains a direct multi-horizon/final-horizon exploratory setup. "
            "Use scripts/run_recursive_eval_probe.py or scripts/screen_recursive_head.py for manuscript-grade "
            "one-step recursive 7-day evaluation. Re-run only with --allow-direct-lead-exploratory if you explicitly "
            "want an exploratory audit artifact."
        )

    series = load_ladder_series(args.stem)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_sequence_split(series, splits, window=args.window, horizon=args.horizon)
    _, outputs, meta = train_ladder_variant(
        split_data,
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        hidden=args.hidden,
        lr=args.lr,
        lambda_penalty=args.lambda_penalty,
        tau_days=args.tau_days,
    )
    outputs["test_rollout"] = rollout_sequence_model(
        model=_,
        series=series,
        split=splits.test,
        norm=split_data["norm"],
        window=args.window,
        horizon=args.horizon,
    )
    out_dir = Path(args.output_root) / args.stem / f"{args.variant}_seed{args.seed}"
    save_ladder_run(out_dir, series, outputs, meta)
    print((out_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()

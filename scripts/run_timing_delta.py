from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.timing_delta import (  # noqa: E402
    build_delta_split,
    load_ladder_series,
    make_block_splits,
    rollout_delta_model,
    save_delta_run,
    train_delta_variant,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--variant", choices=["gru", "ws2", "ode"], required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--delta-loss-weight", type=float, default=1.0)
    ap.add_argument("--tau-days", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow-direct-lead-exploratory",
        action="store_true",
        help="Acknowledge that this script runs an exploratory direct multi-step decoder that is not allowed for main-paper evidence.",
    )
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "results/timing_delta"),
    )
    args = ap.parse_args()

    if not args.allow_direct_lead_exploratory:
        raise SystemExit(
            "run_timing_delta.py is disabled for main research use. "
            "It trains a direct multi-step decoder, not the manuscript-grade one-step recursive task. "
            "Use scripts/run_recursive_eval_probe.py for strict 1-day x 7 recursive evaluation. "
            "Re-run only with --allow-direct-lead-exploratory if you explicitly want an exploratory audit artifact."
        )

    series = load_ladder_series(args.stem)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_delta_split(series, splits, window=args.window, horizon=args.horizon)
    model, outputs, meta = train_delta_variant(
        split_data,
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        hidden=args.hidden,
        lr=args.lr,
        lambda_penalty=args.lambda_penalty,
        delta_loss_weight=args.delta_loss_weight,
        tau_days=args.tau_days,
    )
    outputs["test_rollout"] = rollout_delta_model(
        model=model,
        series=series,
        split=splits.test,
        norm=split_data["norm"],
        window=args.window,
        horizon=args.horizon,
    )
    out_dir = Path(args.output_root) / args.stem / f"delta_{args.variant}_seed{args.seed}"
    save_delta_run(out_dir, series, outputs, meta)
    print((out_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()

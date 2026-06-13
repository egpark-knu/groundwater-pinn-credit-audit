from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.direct_delta_lead import (  # noqa: E402
    build_direct_delta_split,
    load_ladder_series,
    make_block_splits,
    save_direct_delta_run,
    train_direct_delta_variant,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--variant", choices=["gru", "ws1", "ws2", "ode"], required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--event-weight-scale", type=float, default=1.0)
    ap.add_argument("--without-dhead", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", default=str(ROOT / "results/direct_delta_lead"))
    args = ap.parse_args()

    series = load_ladder_series(args.stem)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_direct_delta_split(
        series,
        splits,
        window=args.window,
        horizon=args.horizon,
        include_dhead=not args.without_dhead,
    )
    _, outputs, meta = train_direct_delta_variant(
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
    out_dir = Path(args.output_root) / args.stem / f"{meta['variant']}_seed{args.seed}"
    save_direct_delta_run(out_dir, series, outputs, meta)
    print((out_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()

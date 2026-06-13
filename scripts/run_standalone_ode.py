from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.neural_ladder import (  # noqa: E402
    build_sequence_split,
    load_ladder_series,
    make_block_splits,
    save_ladder_run,
)
from groundwater_research.ode_baseline import ODEParams, fit_standalone_ode, rollout_standalone_ode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5.0e-2)
    ap.add_argument("--tau-grid", default="3,7,14,30")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "results/predictive_ladder"),
    )
    args = ap.parse_args()

    tau_candidates = tuple(float(x) for x in args.tau_grid.split(",") if x.strip())
    series = load_ladder_series(args.stem)
    splits = make_block_splits(len(series.head_interp))
    split_data = build_sequence_split(series, splits, window=args.window, horizon=args.horizon)
    outputs, meta = fit_standalone_ode(
        split_data,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        tau_candidates=tau_candidates,
    )
    params = ODEParams(**meta["physics_params"])
    outputs["test_rollout"] = rollout_standalone_ode(
        series=series,
        split=splits.test,
        horizon=args.horizon,
        params=params,
    )
    out_dir = Path(args.output_root) / args.stem / f"ode_only_seed{args.seed}"
    save_ladder_run(out_dir, series, outputs, meta)
    print((out_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()

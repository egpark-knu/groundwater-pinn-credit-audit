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

from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402
from groundwater_research.nhits_baseline import build_nf_frame, fit_nhits_one_step, recursive_block_rollout_nhits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--forecast-horizon", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    series = load_ladder_series(args.stem)
    splits = make_block_splits(len(series.head_interp))
    full_df = build_nf_frame(series)
    train_df = full_df.iloc[splits.train].copy()
    nf = fit_nhits_one_step(
        train_df=train_df,
        exog_cols=list(series.climate_cols),
        input_size=args.window,
        max_steps=args.max_steps,
        random_seed=args.seed,
    )
    rollout = recursive_block_rollout_nhits(
        nf=nf,
        series=series,
        split=splits.test,
        window=args.window,
        forecast_horizon=args.forecast_horizon,
    )

    out = {
        "stem": series.stem,
        "variant": "NHITS",
        "seed": args.seed,
        "window": args.window,
        "forecast_horizon": args.forecast_horizon,
        "max_steps": args.max_steps,
        **rollout["metrics"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    np.savez(
        output_path.with_name(f"{output_path.stem.replace('_summary', '')}_rollout_predictions.npz"),
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )
    print(output_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

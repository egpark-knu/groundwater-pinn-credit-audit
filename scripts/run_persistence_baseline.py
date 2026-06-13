from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.baselines import rollout_persistence  # noqa: E402
from groundwater_research.neural_ladder import load_ladder_series, make_block_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "results/predictive_ladder"),
    )
    args = ap.parse_args()

    series = load_ladder_series(args.stem)
    splits = make_block_splits(len(series.head_interp))
    rollout = rollout_persistence(series=series, split=splits.test, horizon=args.horizon)

    out_dir = Path(args.output_root) / args.stem / "persistence_seed0"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stem": series.stem,
        "variant": "persistence",
        "seed": 0,
        "horizon": args.horizon,
        "test_rollout": rollout["metrics"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    npz_path = out_dir / "test_rollout_predictions.npz"
    import numpy as np

    np.savez(
        npz_path,
        pred=np.asarray(rollout["pred"]),
        obs=np.asarray(rollout["obs"]),
        dates=np.asarray(rollout["dates"]),
    )
    print((out_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()

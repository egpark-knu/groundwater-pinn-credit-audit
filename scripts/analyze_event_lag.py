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

from groundwater_research.event_lag import local_event_lags, select_event_centers, summarize_local_event_lags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--variants", default="persistence,gru,ws2,ode")
    ap.add_argument("--k-events", type=int, default=3)
    ap.add_argument("--min-gap", type=int, default=21)
    ap.add_argument("--half-window", type=int, default=14)
    args = ap.parse_args()

    stem_root = Path(args.root) / args.stem
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    first_run = next(iter(sorted(stem_root.glob(f"{variants[0]}_seed*/"))), None)
    if first_run is None:
        raise FileNotFoundError(f"No run found for {args.stem} and {variants[0]}")
    with np.load(first_run / "test_rollout_predictions.npz") as d:
        obs = d["obs"].astype(float)
        dates = d["dates"].astype("datetime64[D]")
    centers = select_event_centers(obs, k=args.k_events, min_gap=args.min_gap)

    payload: dict[str, object] = {
        "stem": args.stem,
        "event_centers": [str(dates[c]) for c in centers],
        "variants": {},
        "variant_summary": {},
    }
    for variant in variants:
        run_dir = next(iter(sorted(stem_root.glob(f"{variant}_seed*/"))), None)
        if run_dir is None:
            continue
        with np.load(run_dir / "test_rollout_predictions.npz") as d:
            pred = d["pred"].astype(float)
            obs = d["obs"].astype(float)
            dates = d["dates"].astype("datetime64[D]")
        rows = local_event_lags(
            pred=pred,
            obs=obs,
            dates=dates,
            centers=centers,
            half_window=args.half_window,
        )
        payload["variants"][variant] = rows
        payload["variant_summary"][variant] = summarize_local_event_lags(rows)

    out = stem_root / "event_lag_analysis.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()

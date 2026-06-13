from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stem", required=True)
    ap.add_argument(
        "--allow-direct-lead-exploratory",
        action="store_true",
        help="Acknowledge that this summary is for legacy direct final-horizon outputs only.",
    )
    args = ap.parse_args()

    if not args.allow_direct_lead_exploratory:
        raise SystemExit(
            "summarize_ladder_runs.py summarizes legacy direct final-horizon outputs only. "
            "Do not use it for the manuscript's recursive 7-day task. "
            "Use rollout-based summaries from recursive evidence directories instead. "
            "Re-run only with --allow-direct-lead-exploratory if you explicitly want an audit summary."
        )

    root = Path(args.root) / args.stem
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "stem": summary["stem"],
                "variant": summary["variant"],
                "seed": summary["seed"],
                "test_rmse_final": summary["test"]["rmse_final"],
                "test_nse_final": summary["test"]["nse_final"],
                "test_corr_final": summary["test"]["corr_final"],
                "test_rmse_seq": summary["test"]["rmse_seq"],
                "best_val_loss": summary["best_val_loss"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["variant", "seed"])
    out = root / "variant_summary.csv"
    df.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

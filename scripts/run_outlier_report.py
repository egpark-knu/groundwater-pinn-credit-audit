from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.data_quality import clean_ladder_series, detect_head_outliers, per_date_outlier_frame  # noqa: E402
from groundwater_research.neural_ladder import load_ladder_series  # noqa: E402


DEFAULT_WELLS = [
    "거제신현_암반",
    "영덕도천_천부_충적",
    "창원북면_충적",
    "안동태화_충적",
    "영덕달산_암반",
    "울진울진_암반",
]


def run_report(wells: list[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_date_dir = output_dir / "per_well_flags"
    per_date_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for well in wells:
        series = load_ladder_series(well)
        flags, report = detect_head_outliers(series.head_raw)
        cleaned, _ = clean_ladder_series(series)
        row = {"well": series.stem, **report}
        row["cleaned_head_min"] = float(pd.Series(cleaned.head_interp).min())
        row["cleaned_head_max"] = float(pd.Series(cleaned.head_interp).max())
        rows.append(row)
        per_date_outlier_frame(series, flags).to_csv(per_date_dir / f"{series.stem}_outlier_flags.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "outlier_report.csv", index=False)
    summary = {
        "n_wells": int(len(df)),
        "total_flagged": int(df["n_flagged"].sum()),
        "mean_flagged_ratio_total": float(df["flagged_ratio_total"].mean()),
        "max_flagged_ratio_total": float(df["flagged_ratio_total"].max()),
        "report_csv": str(output_dir / "outlier_report.csv"),
    }
    (output_dir / "outlier_report_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=DEFAULT_WELLS)
    ap.add_argument("--output-dir", default=str(ROOT / "results/data_quality"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = run_report(args.wells, Path(args.output_dir))
    print(df[["well", "n_total", "n_flagged", "flagged_ratio_total", "n_iqr", "n_zscore", "n_flatline", "n_jump"]].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.shortlist import build_shortlist, load_catalog


def main() -> None:
    out_dir = ROOT / "results/data_screening"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_catalog()
    picked, appendix = build_shortlist(df)

    picked_path = out_dir / "main_case_shortlist.csv"
    appendix_path = out_dir / "appendix_pair_candidates.csv"
    rationale_path = out_dir / "main_case_shortlist_summary.json"

    picked.to_csv(picked_path, index=False)
    appendix.to_csv(appendix_path, index=False)

    summary = {
        "n_main_text_base": int(len(picked)),
        "n_appendix_pairs": int(len(appendix)),
        "main_text_bases": picked[
            [
                "stem",
                "station_name",
                "archetype_suggested",
                "material_class",
                "record_length_days",
                "waterlevel_std",
                "evidence_score",
                "selection_rationale",
            ]
        ].to_dict(orient="records"),
        "appendix_pairs": appendix[
            ["stem", "station_name", "material_class", "selection_rationale"]
        ].to_dict(orient="records"),
    }
    rationale_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

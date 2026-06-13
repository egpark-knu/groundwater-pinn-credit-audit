from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
COMPAT_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f]")


def _material_group(stem: str) -> str:
    if "충적" in stem:
        return "alluvial"
    if "암반" in stem:
        return "bedrock"
    return "unknown"


def _load_aliases(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    ok_records = [record for record in records if record.get("status") == "ok"]
    return {str(record["stem"]): f"well_{idx:03d}" for idx, record in enumerate(ok_records, start=1)}


def _contains_korean(value: str) -> bool:
    return bool(HANGUL_RE.search(value) or COMPAT_HANGUL_RE.search(value))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a reporting-safe anonymized copy of a horizon sensitivity CSV."
    )
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--pair-manifest", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--verification-json", required=True)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    pair_manifest = Path(args.pair_manifest)
    output_csv = Path(args.output_csv)
    verification_json = Path(args.verification_json)

    aliases = _load_aliases(pair_manifest)
    rows = list(csv.DictReader(input_csv.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows found in {input_csv}")
    if "stem" not in rows[0]:
        raise SystemExit("input CSV must contain a stem column")

    output_rows: list[dict[str, str]] = []
    missing_aliases: list[str] = []
    for row in rows:
        stem = row["stem"]
        alias = aliases.get(stem)
        if alias is None:
            missing_aliases.append(stem)
            continue
        out_row = {
            "well_label": alias,
            "material_group": _material_group(stem),
        }
        for key, value in row.items():
            if key == "stem":
                continue
            out_row[key] = value
        output_rows.append(out_row)

    if missing_aliases:
        unique = sorted(set(missing_aliases))
        raise SystemExit(f"missing aliases for {len(unique)} source stem(s): {unique[:5]}")

    fieldnames = list(output_rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    output_text = output_csv.read_text(encoding="utf-8")
    korean_matches = sum(1 for _ in HANGUL_RE.finditer(output_text)) + sum(
        1 for _ in COMPAT_HANGUL_RE.finditer(output_text)
    )
    source_stem_leaks = sorted(
        {stem for stem in aliases if stem and stem in output_text}
    )
    verification = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "alias_source": "pair_manifest_ok_record_order",
        "alias_count": len(aliases),
        "row_count": len(output_rows),
        "unique_well_labels": len({row["well_label"] for row in output_rows}),
        "source_column_removed": "stem" not in fieldnames,
        "korean_codepoint_matches": korean_matches,
        "source_stem_leaks": source_stem_leaks,
        "status": "pass" if korean_matches == 0 and not source_stem_leaks else "fail",
    }
    verification_json.parent.mkdir(parents=True, exist_ok=True)
    verification_json.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    if verification["status"] != "pass":
        raise SystemExit(json.dumps(verification, ensure_ascii=False))
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    main()

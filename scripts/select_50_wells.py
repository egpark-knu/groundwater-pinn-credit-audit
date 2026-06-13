#!/usr/bin/env python3
"""Select 50 wells from maintext candidates for large-scale PINN audit.

Strategy:
- Include all 6 legacy wells (backward compatibility)
- Enforce record_length >= 1461 days (4yr for 60/20/20 split)
- Stratified sampling: bedrock >= 20, alluvial >= 15
- Climate column count == 5 (exclude 2-col Gangneung wells)
- Prefer longer records + higher variability
- Geographic diversity via province spread
"""

import pandas as pd
import unicodedata
import json
import os
from pathlib import Path

PROJECT = Path(os.environ.get("PINN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CATALOG = PROJECT / "results/data_screening/groundwater_maintext_seed_candidates.csv"
OUTPUT_DIR = PROJECT / "results/well_selection"
OUTPUT_CSV = OUTPUT_DIR / "selected_50_wells.csv"
OUTPUT_JSON = OUTPUT_DIR / "selection_summary.json"

# 6 legacy wells (must include)
LEGACY_WELLS = [
    "거제신현_암반",
    "영덕도천_천부_충적",
    "창원북면_충적",
    "안동태화_충적",
    "영덕달산_암반",
    "울진울진_암반",
]

TARGET_N = 50
MIN_RECORD_DAYS = 1461  # 4 years
MIN_CLIMATE_COLS = 5


def nfc(s):
    return unicodedata.normalize("NFC", str(s)) if pd.notna(s) else ""


def main():
    df = pd.read_csv(CATALOG)
    df["stem_nfc"] = df["stem"].apply(nfc)
    df["mat_nfc"] = df["material_class"].apply(nfc)

    # Normalize legacy well names
    legacy_nfc = [unicodedata.normalize("NFC", w) for w in LEGACY_WELLS]

    # Step 1: Quality filter
    mask = (
        (df["record_length_days"] >= MIN_RECORD_DAYS)
        & (df["climate_column_count"] >= MIN_CLIMATE_COLS)
        & (df["waterlevel_missing_ratio"] <= 0.02)
        & (df["waterlevel_max_gap"] <= 30)
    )
    pool = df[mask].copy()
    print(f"After quality filter: {len(pool)} wells")

    # Step 2: Ensure legacy wells are in pool (relax filter if needed)
    legacy_in_pool = pool[pool["stem_nfc"].isin(legacy_nfc)]
    legacy_missing = set(legacy_nfc) - set(legacy_in_pool["stem_nfc"])
    if legacy_missing:
        # Force-add missing legacy wells from full catalog
        for stem in legacy_missing:
            row = df[df["stem_nfc"] == stem]
            if len(row) > 0:
                pool = pd.concat([pool, row], ignore_index=True)
                print(f"  Force-added legacy well: {stem}")

    # Step 3: Mark legacy wells
    pool["is_legacy"] = pool["stem_nfc"].isin(legacy_nfc)

    # Step 4: Score for selection priority
    # Higher score = more desirable
    pool["score"] = (
        pool["record_length_days"] / 7305  # normalize to 0-1
        + pool["waterlevel_std"].rank(pct=True)  # prefer higher variability
        + (pool["mat_nfc"] == "충적").astype(float) * 0.5  # boost alluvial (scarce)
    )

    # Step 5: Select legacy wells first
    selected = pool[pool["is_legacy"]].copy()
    remaining = pool[~pool["is_legacy"]].copy()
    n_needed = TARGET_N - len(selected)
    print(f"Legacy wells selected: {len(selected)}")
    print(f"Need {n_needed} more from {len(remaining)} candidates")

    # Step 6: Stratified selection
    # Target: bedrock >= 20, alluvial >= 15
    legacy_bedrock = (selected["mat_nfc"] == "암반").sum()
    legacy_alluvial = (selected["mat_nfc"] == "충적").sum()

    need_bedrock = max(0, 20 - legacy_bedrock)
    need_alluvial = max(0, 15 - legacy_alluvial)

    # Select alluvial first (scarce resource)
    alluvial_pool = remaining[remaining["mat_nfc"] == "충적"].sort_values(
        "score", ascending=False
    )
    alluvial_select = alluvial_pool.head(need_alluvial)
    remaining = remaining[~remaining.index.isin(alluvial_select.index)]

    # Select bedrock
    bedrock_pool = remaining[remaining["mat_nfc"] == "암반"].sort_values(
        "score", ascending=False
    )
    bedrock_select = bedrock_pool.head(need_bedrock)
    remaining = remaining[~remaining.index.isin(bedrock_select.index)]

    # Fill remainder with highest-scored from any type
    n_fill = n_needed - len(alluvial_select) - len(bedrock_select)
    fill_select = remaining.sort_values("score", ascending=False).head(n_fill)

    # Combine
    selected = pd.concat(
        [selected, alluvial_select, bedrock_select, fill_select], ignore_index=True
    )

    # Step 7: Deduplicate by climate_hash (keep higher score)
    selected = selected.sort_values("score", ascending=False)
    before_dedup = len(selected)
    # Don't dedup legacy wells
    non_legacy = selected[~selected["is_legacy"]]
    legacy = selected[selected["is_legacy"]]
    used_hashes = set(legacy["climate_hash"])

    kept = []
    for _, row in non_legacy.iterrows():
        h = row["climate_hash"]
        if h not in used_hashes:
            kept.append(row)
            used_hashes.add(h)
        else:
            print(f"  Dedup removed: {row['stem_nfc']} (same climate as existing)")
    non_legacy_kept = pd.DataFrame(kept)
    selected = pd.concat([legacy, non_legacy_kept], ignore_index=True)

    if len(selected) < TARGET_N:
        # Backfill from remaining pool
        backfill_pool = pool[~pool.index.isin(selected.index)]
        backfill_pool = backfill_pool[
            ~backfill_pool["climate_hash"].isin(used_hashes)
        ]
        backfill = backfill_pool.sort_values("score", ascending=False).head(
            TARGET_N - len(selected)
        )
        selected = pd.concat([selected, backfill], ignore_index=True)

    print(f"\nFinal selection: {len(selected)} wells")

    # Step 8: Summary statistics
    summary = {
        "total_selected": len(selected),
        "bedrock": int((selected["mat_nfc"] == "암반").sum()),
        "alluvial": int((selected["mat_nfc"] == "충적").sum()),
        "unclassified": int(
            ((selected["mat_nfc"] != "암반") & (selected["mat_nfc"] != "충적")).sum()
        ),
        "coastal": int((selected["archetype_suggested"] == "coastal").sum()),
        "inland": int((selected["archetype_suggested"] == "inland").sum()),
        "mean_record_days": float(selected["record_length_days"].mean()),
        "min_record_days": int(selected["record_length_days"].min()),
        "max_record_days": int(selected["record_length_days"].max()),
        "variability_tiers": selected["variability_tier"].value_counts().to_dict(),
        "provinces": selected["province"].value_counts().to_dict(),
        "legacy_included": [
            s for s in legacy_nfc if s in selected["stem_nfc"].values
        ],
        "dedup_removed": before_dedup - len(selected) if before_dedup > len(selected) else 0,
    }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

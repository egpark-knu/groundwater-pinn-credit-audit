from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("PINN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CATALOG_CSV = ROOT / "results/data_screening/groundwater_case_catalog.csv"


@dataclass(frozen=True)
class ShortlistTargets:
    coastal_total: int = 2
    inland_total: int = 4


def _metadata_score(mode: str) -> float:
    mapping = {
        "exact": 1.2,
        "exact_multi": 1.0,
        "station_depth": 0.9,
        "station_material": 0.9,
        "station_only": 0.8,
        "station_ambiguous": 0.5,
        "contains": 0.3,
    }
    return mapping.get(str(mode), 0.0)


def _record_score(days: float) -> float:
    if days >= 3650:
        return 1.6
    if days >= 1825:
        return 1.1
    if days >= 1095:
        return 0.7
    return 0.2


def _variability_score(std_value: float) -> float:
    if std_value < 0.15:
        return -0.6
    if std_value < 0.30:
        return 0.1
    if std_value < 0.70:
        return 0.8
    if std_value < 2.50:
        return 1.2
    if std_value < 6.0:
        return 0.8
    return 0.3


def _range_penalty(range_value: float) -> float:
    if range_value <= 15.0:
        return 0.0
    if range_value <= 40.0:
        return -0.25
    if range_value <= 80.0:
        return -0.6
    return -1.0


def _geometry_score(row: pd.Series) -> float:
    score = 0.0
    if np.isfinite(row["shoreline_dist_m"]):
        score += 0.3
    if np.isfinite(row["river_dist_m"]) and row["river_dist_m"] <= 250.0:
        score += 0.3
    return score


def _material_bucket(value: str) -> str:
    return "rock" if str(value) == "암반" else "alluvial"


def load_catalog(catalog_csv: Path = CATALOG_CSV) -> pd.DataFrame:
    df = pd.read_csv(catalog_csv)
    pair_counts = df.groupby("station_name")["stem"].count().rename("station_pair_count")
    df = df.merge(pair_counts, on="station_name", how="left")
    df["material_bucket"] = df["material_class"].map(_material_bucket)
    df["evidence_score"] = (
        df["record_length_days"].map(_record_score)
        + df["waterlevel_std"].map(_variability_score)
        + df["waterlevel_range"].map(_range_penalty)
        + df["metadata_match_mode"].map(_metadata_score)
        + df.apply(_geometry_score, axis=1)
        + np.where(df["station_pair_count"] >= 2, 0.3, 0.0)
    )
    df["solver_audit_friendly"] = (
        (df["waterlevel_std"] >= 0.2)
        & (df["waterlevel_std"] <= 2.5)
        & (df["waterlevel_range"] <= 15.0)
        & np.isfinite(df["shoreline_dist_m"])
        & np.isfinite(df["river_dist_m"])
    )
    return df


def _pick_one(
    pool: pd.DataFrame,
    chosen_stations: set[str],
    chosen_climates: set[str],
) -> pd.Series | None:
    for _, row in pool.iterrows():
        if row["station_name"] in chosen_stations:
            continue
        if row["climate_hash"] in chosen_climates:
            continue
        return row
    return None


def build_shortlist(
    df: pd.DataFrame,
    targets: ShortlistTargets = ShortlistTargets(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = df[df["screen_main_text_eligible"]].copy()
    eligible = eligible[
        (eligible["metadata_match_mode"] == "exact")
        & eligible["shoreline_dist_m"].notna()
        & eligible["river_dist_m"].notna()
        & eligible["material_class"].notna()
    ].copy()
    eligible = eligible.sort_values(
        ["evidence_score", "solver_audit_friendly", "record_length_days", "waterlevel_std"],
        ascending=[False, False, False, False],
    )

    chosen_rows: list[pd.Series] = []
    chosen_stations: set[str] = set()
    chosen_climates: set[str] = set()

    quota_plan = [
        ("coastal", "rock", 1),
        ("coastal", "alluvial", 1),
        ("inland", "rock", 2),
        ("inland", "alluvial", 2),
    ]

    for archetype, material_bucket, quota in quota_plan:
        pool = eligible[
            (eligible["archetype_suggested"] == archetype)
            & (eligible["material_bucket"] == material_bucket)
        ]
        for _ in range(quota):
            row = _pick_one(pool, chosen_stations, chosen_climates)
            if row is None:
                break
            chosen_rows.append(row)
            chosen_stations.add(row["station_name"])
            chosen_climates.add(row["climate_hash"])

    picked = pd.DataFrame(chosen_rows).drop_duplicates(subset=["stem"]).copy()
    counts = picked["archetype_suggested"].value_counts().to_dict()
    fill_plan = [
        ("coastal", targets.coastal_total - int(counts.get("coastal", 0))),
        ("inland", targets.inland_total - int(counts.get("inland", 0))),
    ]
    for archetype, remaining in fill_plan:
        if remaining <= 0:
            continue
        pool = eligible[eligible["archetype_suggested"] == archetype]
        for _ in range(remaining):
            row = _pick_one(pool, chosen_stations, chosen_climates)
            if row is None:
                break
            chosen_rows.append(row)
            chosen_stations.add(row["station_name"])
            chosen_climates.add(row["climate_hash"])

    picked = pd.DataFrame(chosen_rows).drop_duplicates(subset=["stem"]).copy()
    picked = picked.sort_values(
        ["archetype_suggested", "material_bucket", "evidence_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    picked["selection_role"] = "main_text_base"
    picked["selection_rank"] = np.arange(1, len(picked) + 1)
    picked["solver_audit_priority"] = picked["solver_audit_friendly"]
    picked["selection_rationale"] = picked.apply(build_rationale, axis=1)

    appendix_rows = []
    stations = set(picked["station_name"])
    for station in stations:
        base_stems = set(picked.loc[picked["station_name"] == station, "stem"])
        station_rows = df[df["station_name"] == station].copy()
        station_rows = station_rows[~station_rows["stem"].isin(base_stems)]
        station_rows = station_rows.sort_values(
            ["screen_main_text_eligible", "record_length_days", "waterlevel_std"],
            ascending=[False, False, False],
        )
        if station_rows.empty:
            continue
        companion = station_rows.iloc[0].copy()
        companion["selection_role"] = "appendix_pair"
        companion["selection_rank"] = np.nan
        companion["selection_rationale"] = (
            f"paired with {next(iter(base_stems))} to support within-site comparison"
        )
        appendix_rows.append(companion)

    appendix = pd.DataFrame(appendix_rows)
    return picked, appendix


def build_rationale(row: pd.Series) -> str:
    phrases = []
    if row["record_length_days"] >= 3650:
        phrases.append("long record")
    elif row["record_length_days"] >= 1825:
        phrases.append("multi-year record")
    if row["metadata_match_mode"] == "exact":
        phrases.append("exact metadata match")
    if row["station_pair_count"] >= 2:
        phrases.append("within-site pair available")
    if 0.3 <= row["waterlevel_std"] <= 2.5:
        phrases.append("moderate dynamic range")
    elif row["waterlevel_std"] > 2.5:
        phrases.append("high dynamic range")
    if row["archetype_suggested"] == "coastal":
        phrases.append("clear coastal geometry support")
    else:
        phrases.append("interior setting with outlet proxy framing")
    return "; ".join(phrases)

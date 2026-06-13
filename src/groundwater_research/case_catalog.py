from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point


ROOT = Path(os.environ.get("PINN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
GROUNDWATER_ROOT = Path(
    os.environ.get("NGMS_GROUNDWATER_ROOT", ROOT / "data" / "groundwater")
)
GEODATA_ROOT = Path(
    os.environ.get("KOREA_GEODATA_ROOT", ROOT / "data" / "geodata")
)
WELL_METADATA = GEODATA_ROOT / "national_monitoring_wells.gpkg"
SHORELINE = GEODATA_ROOT / "shoreline.gpkg"
RIVER = GEODATA_ROOT / "river.gpkg"
DEM90 = GEODATA_ROOT / "dem_90.tif"


STEM_PATTERNS = [
    re.compile(
        r"^(?P<station>.+)_(?P<depth>심부1|심부2|천부)_(?P<material>암반|충적)$"
    ),
    re.compile(r"^(?P<station>.+)_(?P<material>암반|충적)$"),
    re.compile(r"^(?P<station>.+)_(?P<depth>심부1|심부2|천부)$"),
]


@dataclass
class ParsedStem:
    stem: str
    station_name: str
    depth_class: Optional[str]
    material_class: Optional[str]


def parse_stem(stem: str) -> ParsedStem:
    stem = unicodedata.normalize("NFC", stem)
    for pattern in STEM_PATTERNS:
        match = pattern.match(stem)
        if match:
            return ParsedStem(
                stem=stem,
                station_name=match.group("station"),
                depth_class=match.groupdict().get("depth"),
                material_class=match.groupdict().get("material"),
            )
    return ParsedStem(stem=stem, station_name=stem, depth_class=None, material_class=None)


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def longest_flat_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    prev = None
    for value in values:
        if pd.isna(value):
            current = 0
            prev = None
            continue
        if prev is not None and value == prev:
            current += 1
        else:
            current = 1
        best = max(best, current)
        prev = value
    return int(best)


def variability_tier(std_value: float) -> str:
    if std_value < 0.5:
        return "low"
    if std_value < 1.5:
        return "moderate"
    if std_value < 3.0:
        return "high"
    return "very_high"


def read_unique_wells() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(WELL_METADATA)
    gdf = gdf[["시도", "시군구", "측정소명", "관정", "표준코드", "주소", "geometry"]].copy()
    gdf = gdf.to_crs("EPSG:5186")
    for col in ["시도", "시군구", "측정소명", "관정", "표준코드", "주소"]:
        gdf[col] = gdf[col].astype(str).map(lambda x: unicodedata.normalize("NFC", x))
    gdf["stem_exact"] = gdf["측정소명"].astype(str) + "_" + gdf["관정"].astype(str)
    gdf = gdf.sort_values(["측정소명", "관정", "표준코드"])
    return gdf.drop_duplicates(subset=["stem_exact", "geometry"]).reset_index(drop=True)


def match_well(parsed: ParsedStem, wells: gpd.GeoDataFrame) -> tuple[Optional[pd.Series], str]:
    exact = wells[wells["stem_exact"] == parsed.stem]
    if len(exact) == 1:
        return exact.iloc[0], "exact"
    if len(exact) > 1:
        return exact.iloc[0], "exact_multi"

    station = wells[wells["측정소명"] == parsed.station_name]
    if len(station) == 1:
        return station.iloc[0], "station_only"

    if len(station) > 1:
        if parsed.depth_class:
            depth_match = station[station["관정"] == parsed.depth_class]
            if len(depth_match) >= 1:
                return depth_match.iloc[0], "station_depth"
        if parsed.material_class:
            material_match = station[station["관정"] == parsed.material_class]
            if len(material_match) >= 1:
                return material_match.iloc[0], "station_material"
        return station.iloc[0], "station_ambiguous"

    contains = wells[wells["측정소명"].astype(str).str.contains(parsed.station_name, regex=False)]
    if len(contains) == 1:
        return contains.iloc[0], "contains"
    return None, "none"


def screen_main_text(df: pd.DataFrame) -> pd.DataFrame:
    screened = df[
        (df["waterlevel_missing_ratio"] <= 0.01)
        & (df["climate_missing_ratio"] <= 0.01)
        & (df["waterlevel_max_gap"] <= 14)
        & (df["waterlevel_flat_run"] < 30)
    ].copy()
    screened = screened.sort_values(
        [
            "climate_hash",
            "station_name",
            "waterlevel_missing_ratio",
            "climate_missing_ratio",
            "waterlevel_max_gap",
            "waterlevel_flat_run",
            "waterlevel_std",
        ],
        ascending=[True, True, True, True, True, True, False],
    )
    screened["screen_rank_within_climate"] = screened.groupby("climate_hash").cumcount() + 1
    screened["screen_rank_within_station"] = screened.groupby("station_name").cumcount() + 1
    screened["screen_unique_climate"] = screened["screen_rank_within_climate"] == 1
    screened["screen_unique_station"] = screened["screen_rank_within_station"] == 1
    screened["screen_main_text_eligible"] = (
        screened["screen_unique_climate"] & screened["screen_unique_station"]
    )
    return screened


def build_case_catalog(
    groundwater_root: Path = GROUNDWATER_ROOT,
    shoreline_path: Path = SHORELINE,
    river_path: Path = RIVER,
    dem_path: Path = DEM90,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    wt_dir = groundwater_root / "waterlevel"
    cl_dir = groundwater_root / "climate"
    wells = read_unique_wells()
    shoreline_geom = gpd.read_file(shoreline_path).to_crs("EPSG:5186").union_all()
    river_geom = gpd.read_file(river_path).to_crs("EPSG:5186").union_all()

    rows: list[dict] = []
    geometries: list[Optional[Point]] = []
    with rasterio.open(dem_path) as dem:
        for wt_path in sorted(wt_dir.glob("*_WT.txt")):
            stem = unicodedata.normalize("NFC", wt_path.name[:-7])
            cl_path = cl_dir / f"{stem}_CL.txt"
            if not cl_path.exists():
                alt = cl_dir / unicodedata.normalize("NFD", f"{stem}_CL.txt")
                if alt.exists():
                    cl_path = alt
                else:
                    continue

            parsed = parse_stem(stem)
            wt_df = pd.read_csv(wt_path, sep="\t")
            cl_df = pd.read_csv(cl_path, sep="\t")

            wt_values = pd.to_numeric(wt_df["Value"], errors="coerce")
            wt_missing = wt_values.isna().to_numpy()
            climate_missing_rows = (
                cl_df.drop(columns=["Date"]).isna().any(axis=1).sum()
            )
            climate_hash = hashlib.md5(cl_path.read_bytes()).hexdigest()[:12]

            well_match, match_mode = match_well(parsed, wells)
            geometry = well_match.geometry if well_match is not None else None
            shoreline_dist = np.nan
            river_dist = np.nan
            dem90_elev = np.nan
            address = None
            province = None
            district = None
            standard_code = None
            if geometry is not None:
                shoreline_dist = float(geometry.distance(shoreline_geom))
                river_dist = float(geometry.distance(river_geom))
                dem90_elev = float(next(dem.sample([(geometry.x, geometry.y)]))[0])
                address = str(well_match["주소"])
                province = str(well_match["시도"])
                district = str(well_match["시군구"])
                standard_code = str(well_match["표준코드"])

            rows.append(
                {
                    "stem": stem,
                    "station_name": parsed.station_name,
                    "depth_class": parsed.depth_class,
                    "material_class": parsed.material_class,
                    "record_length_days": len(wt_df),
                    "date_start": str(wt_df["Date"].iloc[0]),
                    "date_end": str(wt_df["Date"].iloc[-1]),
                    "waterlevel_missing_rows": int(wt_values.isna().sum()),
                    "waterlevel_missing_ratio": float(wt_values.isna().mean()),
                    "waterlevel_max_gap": longest_true_run(wt_missing),
                    "waterlevel_flat_run": longest_flat_run(wt_values.to_numpy()),
                    "waterlevel_std": float(wt_values.std(skipna=True)),
                    "waterlevel_range": float(
                        wt_values.max(skipna=True) - wt_values.min(skipna=True)
                    ),
                    "variability_tier": variability_tier(
                        float(wt_values.std(skipna=True))
                    ),
                    "climate_missing_rows": int(climate_missing_rows),
                    "climate_missing_ratio": float(climate_missing_rows / len(cl_df)),
                    "climate_column_count": int(len(cl_df.columns) - 1),
                    "climate_hash": climate_hash,
                    "metadata_match_mode": match_mode,
                    "province": province,
                    "district": district,
                    "address": address,
                    "standard_code": standard_code,
                    "shoreline_dist_m": shoreline_dist,
                    "river_dist_m": river_dist,
                    "dem90_elev_m": dem90_elev,
                    "archetype_suggested": (
                        "coastal" if np.isfinite(shoreline_dist) and shoreline_dist <= 10000 else "inland"
                    ),
                }
            )
            geometries.append(geometry)

    df = pd.DataFrame(rows)
    screened = screen_main_text(df)
    screen_cols = [
        "stem",
        "screen_rank_within_climate",
        "screen_rank_within_station",
        "screen_unique_climate",
        "screen_unique_station",
        "screen_main_text_eligible",
    ]
    df = df.merge(screened[screen_cols], on="stem", how="left")
    for col in screen_cols[1:]:
        if col not in df.columns:
            continue
        if col.startswith("screen_"):
            df[col] = df[col].fillna(False)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:5186")
    return df, gdf


def save_case_catalog(output_dir: Path = ROOT / "results/data_screening") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df, gdf = build_case_catalog()
    screened = df[df["screen_main_text_eligible"]].copy()
    screened = screened.sort_values(
        ["archetype_suggested", "variability_tier", "waterlevel_missing_rows", "climate_missing_rows"]
    )

    csv_path = output_dir / "groundwater_case_catalog.csv"
    gpkg_path = output_dir / "groundwater_case_catalog.gpkg"
    screened_path = output_dir / "groundwater_maintext_seed_candidates.csv"
    summary_path = output_dir / "groundwater_case_catalog_summary.json"

    df.to_csv(csv_path, index=False)
    gdf.to_file(gpkg_path, driver="GPKG")
    screened.to_csv(screened_path, index=False)

    summary = {
        "n_total_pairs": int(len(df)),
        "n_maintext_seed_candidates": int(len(screened)),
        "n_unique_climate_groups": int(df["climate_hash"].nunique()),
        "n_exact_metadata_matches": int((df["metadata_match_mode"] == "exact").sum()),
        "n_with_geometry": int(gdf.geometry.notna().sum()),
        "coastal_seed_candidates": int((screened["archetype_suggested"] == "coastal").sum()),
        "inland_seed_candidates": int((screened["archetype_suggested"] == "inland").sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return {
        "catalog_csv": str(csv_path),
        "catalog_gpkg": str(gpkg_path),
        "screened_csv": str(screened_path),
        "summary_json": str(summary_path),
    }

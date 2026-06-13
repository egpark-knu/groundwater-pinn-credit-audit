#!/usr/bin/env python3
"""Audit R1-S23 component evidence for ODE coefficient absorption claims.

This script uses already-generated 50-well LSTM-ODE rollout artifacts. It does
not rerun training. The purpose is to separate what the current artifacts can
support from what still requires a stronger physical-identifiability study.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


RESULTS_DIR = ROOT / "results/whittaker_vs_ode_50well"
PARAMETER_TABLE = ROOT / "results/nuisance_collapse/parameter_table.csv"
OUTPUT_DIR = ROOT / "resubmit/results/r1_s23_component_evidence_20260611"
GROUNDWATER_ROOT = Path(
    os.environ.get("NGMS_GROUNDWATER_ROOT", ROOT / "data" / "groundwater")
)
SELECTION_CSV = ROOT / "results/well_selection/selected_50_wells.csv"
CLIMATE_COLS = ["TEMP", "RAIN", "HUMID", "HPA", "WIND"]


@dataclass
class RainSeries:
    dates: np.ndarray
    rain_mm: np.ndarray


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", str(value))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_rain_series(stem: str) -> RainSeries:
    stem = _nfc(stem)
    wt_path = GROUNDWATER_ROOT / "waterlevel" / f"{stem}_WT.txt"
    cl_path = GROUNDWATER_ROOT / "climate" / f"{stem}_CL.txt"
    if not wt_path.exists():
        wt_path = GROUNDWATER_ROOT / "waterlevel" / unicodedata.normalize("NFD", f"{stem}_WT.txt")
    if not cl_path.exists():
        cl_path = GROUNDWATER_ROOT / "climate" / unicodedata.normalize("NFD", f"{stem}_CL.txt")
    if not wt_path.exists() or not cl_path.exists():
        selection = pd.read_csv(SELECTION_CSV)
        selection["stem_nfc_local"] = selection["stem"].map(_nfc)
        selected = selection[selection["stem_nfc_local"] == stem]
        if selected.empty:
            raise FileNotFoundError(f"Missing selection row for {stem}")
        station = str(selected.iloc[0]["station_name"])
        candidates = [_nfc(station), stem]
        if "_" in stem:
            candidates.append(stem.split("_")[0])
        climate_path = GROUNDWATER_ROOT / "groundwater_climate.csv"
        if not climate_path.exists():
            raise FileNotFoundError(f"Missing WT/CL pair and combined climate CSV for {stem}")
        usecols = ["station_id", "date", "rain"]
        climate = pd.read_csv(climate_path, usecols=usecols)
        climate = climate[climate["station_id"].map(_nfc).isin(candidates)].copy()
        if climate.empty:
            raise FileNotFoundError(f"Missing combined climate rows for {stem} / {station}")
        climate["date"] = pd.to_datetime(climate["date"])
        climate = climate.sort_values("date")
        rain = pd.to_numeric(climate["rain"], errors="coerce").interpolate(limit_direction="both").bfill().ffill()
        return RainSeries(
            dates=climate["date"].to_numpy(dtype="datetime64[D]"),
            rain_mm=rain.to_numpy(dtype=float),
        )
    wt_df = pd.read_csv(wt_path, sep="\t")
    cl_df = pd.read_csv(cl_path, sep="\t")
    df = wt_df.merge(cl_df, on="Date", how="inner")
    rain = pd.to_numeric(df["RAIN"], errors="coerce").interpolate(limit_direction="both").bfill().ffill()
    return RainSeries(
        dates=pd.to_datetime(df["Date"].astype(str), format="%Y%m%d").to_numpy(dtype="datetime64[D]"),
        rain_mm=rain.to_numpy(dtype=float),
    )


def compute_component_rows(params: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stems = sorted({_nfc(v) for v in params["well"].astype(str).unique()})
    anon = {stem: f"W{idx:03d}" for idx, stem in enumerate(stems, start=1)}

    series_cache = {stem: load_rain_series(stem) for stem in stems}

    for _, row in params.iterrows():
        stem = _nfc(row["well"])
        seed = int(row["seed"])
        gamma_r = float(row["physics_gamma_r"])
        gamma_d = float(row["physics_gamma_d"])
        h_ref = float(row["physics_h_ref"])
        anon_rollout = f"results/whittaker_vs_ode_50well/{anon[stem]}/lstm_ode_lambda0.1_seed{seed}/test_rollout_predictions.npz"
        run_dir = RESULTS_DIR / stem / f"lstm_ode_lambda0.1_seed{seed}"
        npz_path = run_dir / "test_rollout_predictions.npz"
        if not npz_path.exists():
            rows.append(
                {
                    "well_id": anon[stem],
                    "seed": seed,
                    "status": "missing_rollout",
                    "source_rollout": anon_rollout,
                }
            )
            continue

        data = np.load(npz_path)
        pred = np.asarray(data["pred"], dtype=float)
        dates = np.asarray(data["dates"]).astype("datetime64[D]")
        if pred.size < 3 or dates.size != pred.size:
            rows.append(
                {
                    "well_id": anon[stem],
                    "seed": seed,
                    "status": "invalid_rollout",
                    "source_rollout": anon_rollout,
                }
            )
            continue

        series = series_cache[stem]
        series_dates = series.dates.astype("datetime64[D]")
        rain_by_date = {d: float(r) for d, r in zip(series_dates, series.rain_mm)}
        rain = np.asarray([rain_by_date.get(d, np.nan) for d in dates], dtype=float)

        dh = np.diff(pred)
        rain_step = rain[1:]
        prev_head = pred[:-1]
        valid = np.isfinite(dh) & np.isfinite(rain_step) & np.isfinite(prev_head)
        if valid.sum() < 3:
            rows.append(
                {
                    "well_id": anon[stem],
                    "seed": seed,
                    "status": "insufficient_aligned_steps",
                    "source_rollout": anon_rollout,
                }
            )
            continue

        dh = dh[valid]
        rain_step = rain_step[valid]
        prev_head = prev_head[valid]
        rainfall_term = gamma_r * rain_step
        recession_term = -gamma_d * (prev_head - h_ref)
        rhs = rainfall_term + recession_term
        residual = dh - rhs

        rows.append(
            {
                "well_id": anon[stem],
                "seed": seed,
                "status": "ok",
                "n_steps": int(dh.size),
                "gamma_r": gamma_r,
                "gamma_d": gamma_d,
                "h_ref": h_ref,
                "rainfall_term_rms": float(np.sqrt(np.mean(rainfall_term**2))),
                "recession_term_rms": float(np.sqrt(np.mean(recession_term**2))),
                "rhs_rms": float(np.sqrt(np.mean(rhs**2))),
                "delta_h_rms": float(np.sqrt(np.mean(dh**2))),
                "residual_rms": float(np.sqrt(np.mean(residual**2))),
                "rainfall_to_delta_ratio": float(
                    np.sqrt(np.mean(rainfall_term**2)) / max(np.sqrt(np.mean(dh**2)), 1.0e-12)
                ),
                "recession_to_delta_ratio": float(
                    np.sqrt(np.mean(recession_term**2)) / max(np.sqrt(np.mean(dh**2)), 1.0e-12)
                ),
                "rhs_to_delta_ratio": float(np.sqrt(np.mean(rhs**2)) / max(np.sqrt(np.mean(dh**2)), 1.0e-12)),
                "corr_delta_rainfall_term": _corr(dh, rainfall_term),
                "corr_delta_recession_term": _corr(dh, recession_term),
                "corr_delta_rhs": _corr(dh, rhs),
                "corr_rainfall_recession_terms": _corr(rainfall_term, recession_term),
                "rain_positive_fraction": float(np.mean(rain_step > 0.0)),
                "source_rollout": anon_rollout,
            }
        )

    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    ok = rows[rows["status"] == "ok"].copy()
    metrics = [
        "rainfall_term_rms",
        "recession_term_rms",
        "rhs_rms",
        "delta_h_rms",
        "residual_rms",
        "rainfall_to_delta_ratio",
        "recession_to_delta_ratio",
        "rhs_to_delta_ratio",
        "corr_delta_rainfall_term",
        "corr_delta_recession_term",
        "corr_delta_rhs",
        "corr_rainfall_recession_terms",
        "rain_positive_fraction",
    ]
    summary_rows = []
    for metric in metrics:
        values = pd.to_numeric(ok[metric], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()) if values.size else None,
                "median": float(values.median()) if values.size else None,
                "p05": float(values.quantile(0.05)) if values.size else None,
                "p95": float(values.quantile(0.95)) if values.size else None,
                "min": float(values.min()) if values.size else None,
                "max": float(values.max()) if values.size else None,
            }
        )
    summary = pd.DataFrame(summary_rows)
    row_text = rows.to_csv(index=False)
    contains_source_stems = bool(any("\uac00" <= ch <= "\ud7a3" for ch in row_text))
    verification = {
        "status": "pass" if len(ok) == len(rows) and len(ok) == 150 else "partial",
        "input_parameter_table": str(PARAMETER_TABLE.relative_to(ROOT)),
        "input_rollout_root": str(RESULTS_DIR.relative_to(ROOT)),
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
        "parameter_rows": int(len(rows)),
        "ok_rows": int(len(ok)),
        "missing_or_invalid_rows": int(len(rows) - len(ok)),
        "unique_anonymous_wells": int(ok["well_id"].nunique()) if not ok.empty else 0,
        "seeds": sorted(int(v) for v in ok["seed"].dropna().unique().tolist()),
        "contains_source_stems": contains_source_stems,
        "claim_boundary": (
            "Component decomposition supports a claim that the tested ODE penalty's "
            "learned coefficients behave as coupled fitting terms in the stored "
            "rollouts. It does not prove physical identifiability or measured "
            "aquifer-property recovery."
        ),
    }
    return summary, verification


def write_markdown(summary: pd.DataFrame, verification: dict[str, object]) -> str:
    def metric_value(metric: str, field: str) -> float | None:
        sub = summary[summary["metric"] == metric]
        if sub.empty:
            return None
        value = sub.iloc[0][field]
        return None if pd.isna(value) else float(value)

    lines = [
        "# R1-S23 Component Evidence Audit",
        "",
        "Date: 2026-06-11",
        "",
        "Purpose: test the reviewer concern that coefficient-absorption language",
        "needs direct support rather than relying only on coefficient ranges.",
        "",
        "## Sources",
        "",
        f"- Parameter rows: `{verification['input_parameter_table']}`",
        f"- Rollout root: `{verification['input_rollout_root']}`",
        "- Component equation audited on stored recursive predictions:",
        "  `RHS = gamma_r * rainfall_mm - gamma_d * (predicted_head - h_ref)`.",
        "",
        "## Verification",
        "",
        f"- Status: `{verification['status']}`",
        f"- ODE parameter rows: {verification['parameter_rows']}",
        f"- Rows with aligned component evidence: {verification['ok_rows']}",
        f"- Anonymous wells: {verification['unique_anonymous_wells']}",
        f"- Seeds: {verification['seeds']}",
        "",
        "## Aggregate Findings",
        "",
        "| metric | mean | median | p05 | p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in [
        "corr_delta_rainfall_term",
        "corr_delta_recession_term",
        "corr_delta_rhs",
        "rainfall_to_delta_ratio",
        "recession_to_delta_ratio",
        "rhs_to_delta_ratio",
        "residual_rms",
        "rain_positive_fraction",
    ]:
        row = summary[summary["metric"] == metric].iloc[0]
        lines.append(
            f"| {metric} | {row['mean']:.6f} | {row['median']:.6f} | {row['p05']:.6f} | {row['p95']:.6f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        f"- Mean correlation between predicted head change and the full RHS is {metric_value('corr_delta_rhs', 'mean'):.3f}.",
        f"- Mean correlation with the rainfall component alone is {metric_value('corr_delta_rainfall_term', 'mean'):.3f}.",
        f"- Mean correlation with the recession component alone is {metric_value('corr_delta_recession_term', 'mean'):.3f}.",
        f"- The median RHS-to-delta RMS ratio is {metric_value('rhs_to_delta_ratio', 'median'):.3f}.",
        "",
        "These results support a bounded statement that, in the stored tested",
        "formulation, the ODE residual components do not behave like an independently",
        "identified physical driver of the neural rollout. The component evidence",
        "is consistent with datum, damping, and forcing-scale compensation in the",
        "reduced residual.",
        "",
        "## Claim Boundary",
        "",
        "Supported wording:",
        "",
        "> A component audit of the stored LSTM-ODE rollouts found weak alignment",
        "> between the predicted head increments and the reconstructed ODE RHS,",
        "> including the rainfall and recession components. Together with the",
        "> parameter-bound and bounded-reference-head diagnostics, this supports",
        "> interpreting the coefficients as coupled fitting terms in the tested",
        "> residual rather than as independently identified aquifer properties.",
        "",
        "Blocked wording:",
        "",
        "> The component audit proves that all groundwater ODE/PINN formulations",
        "> cannot learn physical parameters.",
        "",
        "> The learned coefficients have been validated against measured aquifer",
        "> properties.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params = pd.read_csv(PARAMETER_TABLE)
    rows = compute_component_rows(params)
    rows_path = OUTPUT_DIR / "r1_s23_component_rows.anonymized.csv"
    rows.to_csv(rows_path, index=False)

    summary, verification = summarize(rows)
    summary_path = OUTPUT_DIR / "r1_s23_component_summary.csv"
    verification_path = OUTPUT_DIR / "r1_s23_component_verification.json"
    markdown_path = OUTPUT_DIR / "r1_s23_component_evidence_audit_20260611.md"
    summary.to_csv(summary_path, index=False)
    verification_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False))
    markdown_path.write_text(write_markdown(summary, verification))

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {verification_path}")
    print(f"Wrote {markdown_path}")
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

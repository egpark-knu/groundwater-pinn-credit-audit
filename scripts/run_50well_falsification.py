#!/usr/bin/env python3
"""Run Whittaker-vs-ODE falsification on 50 selected wells, 3 seeds.

Wraps run_whittaker_vs_ode.py with wells from selected_50_wells.csv.
Estimated: 450 runs × ~10s ≈ 75 min on CPU.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
WELLS_CSV = PROJECT / "results/well_selection/selected_50_wells.csv"
SCRIPT = PROJECT / "scripts/run_whittaker_vs_ode.py"
OUTPUT_DIR = PROJECT / "results/whittaker_vs_ode_50well"


def main():
    df = pd.read_csv(WELLS_CSV)
    wells = df["stem"].tolist()
    print(f"Running falsification on {len(wells)} wells, 3 seeds...")

    cmd = [
        PYTHON, str(SCRIPT),
        "--wells", *wells,
        "--seeds", "7", "42", "99",
        "--lambda-value", "0.1",
        "--window", "30",
        "--forecast-horizon", "7",
        "--epochs", "80",
        "--patience", "15",
        "--clean-head-outliers",
        "--output-dir", str(OUTPUT_DIR),
    ]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()

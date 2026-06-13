from __future__ import annotations

import pandas as pd

from scripts.run_50well_architecture import (
    DEFAULT_MODELS,
    DEFAULT_SEEDS,
    build_run_dir,
    plan_architecture_runs,
    read_selected_wells,
)


def test_read_selected_wells_prefers_stem_nfc_column(tmp_path) -> None:
    csv_path = tmp_path / "selected_50_wells.csv"
    pd.DataFrame(
        {
            "stem": ["A_old", "B_old"],
            "stem_nfc": ["A", "B"],
            "score": [1.0, 0.5],
        }
    ).to_csv(csv_path, index=False)

    assert read_selected_wells(csv_path) == ["A", "B"]


def test_plan_architecture_runs_covers_50_wells_three_models_three_seeds() -> None:
    wells = [f"W{i:02d}" for i in range(50)]

    runs = plan_architecture_runs(wells=wells, models=DEFAULT_MODELS, seeds=DEFAULT_SEEDS)

    assert len(runs) == 50 * 3 * 3
    assert {run.model for run in runs} == {"lstm", "gru", "patchtst"}
    assert {run.seed for run in runs} == {7, 42, 99}
    assert all(run.lambda_value == 0.0 for run in runs)


def test_build_run_dir_uses_model_seed_without_lambda_label(tmp_path) -> None:
    run = plan_architecture_runs(wells=["WellA"], models=["patchtst"], seeds=[42])[0]

    run_dir = build_run_dir(tmp_path, run)

    assert run_dir == tmp_path / "WellA" / "patchtst_seed42"

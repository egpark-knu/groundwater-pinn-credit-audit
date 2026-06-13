from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_publication_figures import (
    FIGURE_FILENAMES,
    LOCKED_WELLS,
    WELL_LABELS,
    load_model_contrast,
    load_six_well_architecture_group,
    select_best_ode_seed_by_well,
)


def test_required_figure_filename_contract() -> None:
    assert FIGURE_FILENAMES == {
        "fig01": "fig01_study_area_wells.png",
        "fig02": "fig02_hydrograph_fits.png",
        "fig03": "fig03_lambda_sensitivity.png",
        "fig04": "fig04_architecture_winner_heatmap.png",
        "fig05": "fig05_whittaker_vs_ode.png",
        "fig06": "fig06_solver_parameter_compensation.png",
        "fig07": "fig07_physical_credit_ladder.png",
    }


def test_select_best_ode_seed_by_well_uses_minimum_recursive_rmse() -> None:
    df = pd.DataFrame(
        [
            {"stem": "A", "variant": "ode", "seed": 7, "rmse": 0.2},
            {"stem": "A", "variant": "ode", "seed": 42, "rmse": 0.1},
            {"stem": "A", "variant": "gru", "seed": 7, "rmse": 0.05},
            {"stem": "B", "variant": "ode", "seed": 99, "rmse": 0.3},
        ]
    )
    assert select_best_ode_seed_by_well(df) == {"A": 42, "B": 99}


def test_load_model_contrast_excludes_modern_baseline_for_core_panel() -> None:
    df = load_model_contrast(Path("results/research_summaries/key_case_model_contrast_strict_w30.csv"))
    assert set(df["model_family"]) == {"delta_recursive_w30"}
    assert {"gru", "ws2", "ode"} <= set(df["variant"])


def test_six_well_architecture_group_has_cleaned_three_seed_contract() -> None:
    df = load_six_well_architecture_group()

    assert set(df["well"]) == set(LOCKED_WELLS)
    assert set(df["model"]) == {"gru", "lstm", "patchtst"}
    assert set(df["n_seeds"]) == {3}


def test_publication_well_labels_are_english_only() -> None:
    expected = {
        "거제신현_암반": "Geoje Sinhyeon, bedrock",
        "영덕도천_천부_충적": "Yeongdeok Docheon, shallow alluvial",
        "창원북면_충적": "Changwon Bukmyeon, alluvial",
        "안동태화_충적": "Andong Taehwa, alluvial",
        "영덕달산_암반": "Yeongdeok Dalsan, bedrock",
        "울진울진_암반": "Uljin, bedrock",
    }

    assert WELL_LABELS == expected
    assert all(not any("\uac00" <= ch <= "\ud7a3" for ch in label) for label in WELL_LABELS.values())

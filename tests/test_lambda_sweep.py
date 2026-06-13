from __future__ import annotations

from scripts.run_lambda_sweep import lambda_plot_value, plan_runs


def test_plan_runs_uses_single_gru_baseline_and_full_ode_sweep() -> None:
    runs = plan_runs(
        wells=["A", "B"],
        models=["gru", "ode"],
        lambda_values=[0.0, 0.001, 0.01],
        seeds=[7, 42],
    )

    gru_runs = [run for run in runs if run.model == "gru"]
    ode_runs = [run for run in runs if run.model == "ode"]

    assert len(gru_runs) == 2 * 2
    assert {run.lambda_value for run in gru_runs} == {0.0}
    assert len(ode_runs) == 2 * 2 * 3
    assert {run.lambda_value for run in ode_runs} == {0.0, 0.001, 0.01}


def test_lambda_plot_value_maps_zero_below_positive_grid() -> None:
    assert lambda_plot_value(0.0, min_positive=0.001) == 0.0001
    assert lambda_plot_value(0.01, min_positive=0.001) == 0.01

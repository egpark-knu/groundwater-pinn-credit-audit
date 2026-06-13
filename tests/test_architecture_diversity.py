from __future__ import annotations

import torch

from groundwater_research.architecture_diversity import ForecastLSTMLeadDelta, ForecastNARXLeadDelta, variant_regularizer_kind
from scripts.run_architecture_diversity_smoke import plan_architecture_runs
from scripts.run_whittaker_vs_ode import plan_full_lambda_runs, plan_matched_lambda_runs


def test_narx_forward_returns_one_delta_per_sample() -> None:
    model = ForecastNARXLeadDelta(n_past_feat=7, n_future_feat=5, horizon=1, window=30, hidden=32)

    out = model(torch.randn(4, 30, 7), torch.randn(4, 1, 5))

    assert out.shape == (4,)


def test_lstm_forward_returns_one_delta_per_sample() -> None:
    model = ForecastLSTMLeadDelta(n_past_feat=7, n_future_feat=5, horizon=1, hidden=32)

    out = model(torch.randn(4, 30, 7), torch.randn(4, 1, 5))

    assert out.shape == (4,)


def test_architecture_smoke_plan_is_plain_architecture_only() -> None:
    runs = plan_architecture_runs(
        wells=["W1", "W2", "W3", "W4", "W5", "W6"],
        models=["narx", "lstm", "gru", "patchtst"],
        seed=42,
    )

    assert len(runs) == 6 * 4
    assert {run.model for run in runs} == {"narx", "lstm", "gru", "patchtst"}
    assert {run.seed for run in runs} == {42}
    assert all(run.lambda_value == 0.0 for run in runs)


def test_architecture_smoke_plan_accepts_multiple_seeds() -> None:
    runs = plan_architecture_runs(
        wells=["W1", "W2"],
        models=["lstm", "gru", "patchtst"],
        seeds=[7, 42, 99],
    )

    assert len(runs) == 2 * 3 * 3
    assert {run.seed for run in runs} == {7, 42, 99}
    assert {run.model for run in runs} == {"lstm", "gru", "patchtst"}


def test_lstm_regularizer_variants_are_explicitly_classified() -> None:
    assert variant_regularizer_kind("lstm") == "plain"
    assert variant_regularizer_kind("lstm_ode") == "ode"
    assert variant_regularizer_kind("lstm_ws2") == "ws2"


def test_whittaker_vs_ode_plan_uses_matched_lambda_for_lstm_only() -> None:
    runs = plan_matched_lambda_runs(wells=["W1", "W2"], seed=42, lambda_value=0.1)

    assert len(runs) == 2 * 3
    assert {run.model for run in runs} == {"lstm", "lstm_ode", "lstm_ws2"}
    assert {run.seed for run in runs} == {42}
    assert {(run.model, run.lambda_value) for run in runs} == {
        ("lstm", 0.0),
        ("lstm_ode", 0.1),
        ("lstm_ws2", 0.1),
    }


def test_whittaker_full_lambda_plan_covers_three_models_lambda_grid_and_seeds() -> None:
    runs = plan_full_lambda_runs(wells=["W1", "W2"], seeds=[7, 42], lambda_values=[0.0, 0.1, 1.0])

    assert len(runs) == 2 * 2 * 3 * 3
    assert {run.model for run in runs} == {"lstm", "lstm_ode", "lstm_ws2"}
    assert {run.seed for run in runs} == {7, 42}
    assert {run.lambda_value for run in runs} == {0.0, 0.1, 1.0}

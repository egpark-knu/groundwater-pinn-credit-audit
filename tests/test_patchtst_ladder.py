from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_patchtst_sweep import plan_patchtst_runs
from groundwater_research.patchtst_ladder import ForecastPatchTSTLeadDelta


def test_patchtst_forward_returns_one_delta_per_sample() -> None:
    model = ForecastPatchTSTLeadDelta(
        n_past_feat=7,
        n_future_feat=5,
        horizon=1,
        window=30,
        patch_len=7,
        stride=7,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )

    out = model(torch.randn(4, 30, 7), torch.randn(4, 1, 5))

    assert out.shape == (4,)


def test_patchtst_patches_include_most_recent_week_when_window_not_divisible_by_stride() -> None:
    model = ForecastPatchTSTLeadDelta(
        n_past_feat=7,
        n_future_feat=5,
        horizon=1,
        window=30,
        patch_len=7,
        stride=7,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )

    assert model.patch_starts.tolist() == [0, 7, 14, 21, 23]


def test_patchtst_head_uses_all_encoded_patches_not_mean_pool_only() -> None:
    model = ForecastPatchTSTLeadDelta(
        n_past_feat=7,
        n_future_feat=5,
        horizon=1,
        window=30,
        patch_len=7,
        stride=7,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )

    assert model.head[1].in_features == (model.n_patches + 1) * 32


def test_patchtst_sweep_plan_keeps_plain_and_legacy_gru_to_lambda_zero() -> None:
    runs = plan_patchtst_runs(
        wells=["A"],
        models=["patchtst", "patchtst_ws2", "patchtst_ode", "legacy_gru"],
        lambda_values=[0.0, 0.001, 0.01],
        seeds=[7, 42],
    )

    plain = [r for r in runs if r.model == "patchtst"]
    legacy = [r for r in runs if r.model == "legacy_gru"]
    ws2 = [r for r in runs if r.model == "patchtst_ws2"]
    ode = [r for r in runs if r.model == "patchtst_ode"]

    assert len(plain) == 2
    assert {r.lambda_value for r in plain} == {0.0}
    assert len(legacy) == 2
    assert {r.lambda_value for r in legacy} == {0.0}
    assert len(ws2) == 2 * 3
    assert {r.lambda_value for r in ws2} == {0.0, 0.001, 0.01}
    assert len(ode) == 2 * 3
    assert {r.lambda_value for r in ode} == {0.0, 0.001, 0.01}

from __future__ import annotations

import pandas as pd

from scripts.analyze_seed_dependence import compute_seed_dependence


def test_seed_dependence_counts_winner_transitions_and_rank_stability() -> None:
    df = pd.DataFrame(
        [
            {"well": "W1", "seed": 7, "model": "a", "rmse": 1.0},
            {"well": "W1", "seed": 7, "model": "b", "rmse": 2.0},
            {"well": "W1", "seed": 7, "model": "c", "rmse": 3.0},
            {"well": "W1", "seed": 42, "model": "a", "rmse": 1.5},
            {"well": "W1", "seed": 42, "model": "b", "rmse": 1.0},
            {"well": "W1", "seed": 42, "model": "c", "rmse": 3.0},
            {"well": "W1", "seed": 99, "model": "a", "rmse": 1.1},
            {"well": "W1", "seed": 99, "model": "b", "rmse": 1.2},
            {"well": "W1", "seed": 99, "model": "c", "rmse": 3.0},
        ]
    )

    summary, winner_table, rank_corr = compute_seed_dependence(df, seed_order=[7, 42, 99])

    row = summary.iloc[0]
    assert row["well"] == "W1"
    assert row["distinct_seed_winners"] == 2
    assert row["winner_transition_count"] == 2
    assert row["seed_winner_sequence"] == "a -> b -> a"
    assert set(winner_table["winner"]) == {"a", "b"}
    assert len(rank_corr) == 3

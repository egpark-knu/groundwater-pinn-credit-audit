from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np


def load_payload(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    return payload


def configure_korean_font() -> None:
    for name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def main() -> None:
    configure_korean_font()
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_dir")
    args = ap.parse_args()

    pilot_dir = Path(args.pilot_dir)
    summary = json.loads((pilot_dir / "pilot_summary.json").read_text())
    train = load_payload(pilot_dir / "best_member_train.npz")
    test = load_payload(pilot_dir / "best_member_test.npz")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_train, ax_test, ax_resid, ax_scatter = axes.ravel()

    for ax, payload, label in [
        (ax_train, train, "Train"),
        (ax_test, test, "Test"),
    ]:
        dates = payload["dates"].astype("datetime64[D]")
        obs = payload["obs"].astype(float)
        pred = payload["pred_head"].astype(float)
        mask = payload["valid_mask"].astype(bool)
        ax.plot(dates, pred, color="#b03a2e", lw=1.2, label="Virtual aquifer")
        ax.plot(dates[mask], obs[mask], color="#154c79", lw=0.9, label="Observed")
        ax.set_title(
            f"{label} | RMSE={float(payload['rmse']):.3f} | NSE={float(payload['nse']):.3f}"
        )
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=8)

    test_obs = test["obs"].astype(float)
    test_pred = test["pred_head"].astype(float)
    test_mask = test["valid_mask"].astype(bool)
    test_dates = test["dates"].astype("datetime64[D]")
    residual = test_pred[test_mask] - test_obs[test_mask]

    ax_resid.axhline(0.0, color="black", lw=0.8, alpha=0.7)
    ax_resid.plot(test_dates[test_mask], residual, color="#7d3c98", lw=1.0)
    ax_resid.set_title("Test Residual")
    ax_resid.grid(alpha=0.18)

    ax_scatter.scatter(test_obs[test_mask], test_pred[test_mask], s=14, alpha=0.7, color="#117a65")
    lo = min(np.nanmin(test_obs[test_mask]), np.nanmin(test_pred[test_mask]))
    hi = max(np.nanmax(test_obs[test_mask]), np.nanmax(test_pred[test_mask]))
    ax_scatter.plot([lo, hi], [lo, hi], color="black", lw=0.8, linestyle="--")
    ax_scatter.set_xlabel("Observed")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title(f"Test Scatter | corr={float(test['corr']):.3f}")
    ax_scatter.grid(alpha=0.18)

    fig.suptitle(
        f"{summary['stem']} | {summary['archetype']} | best member {summary['best_member']['member']}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    out = pilot_dir / "pilot_fit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()

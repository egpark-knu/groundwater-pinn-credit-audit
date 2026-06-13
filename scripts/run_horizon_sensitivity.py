from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.baselines import rollout_persistence  # noqa: E402
from groundwater_research.neural_ladder import (  # noqa: E402
    build_sequence_split,
    load_ladder_series,
    make_block_splits,
    rollout_sequence_model,
    save_ladder_run,
    train_ladder_variant,
)
from groundwater_research.ode_baseline import ODEParams, fit_standalone_ode, rollout_standalone_ode  # noqa: E402


def _row_key(row: dict) -> tuple[str, int, str, int]:
    requested_variant = row.get("requested_variant")
    if requested_variant is None or pd.isna(requested_variant):
        requested_variant = row.get("variant")
    return (str(row["stem"]), int(row["horizon"]), str(requested_variant), int(row["seed"]))


def _summary_row(summary_path: Path, stem: str, horizon: int, variant: str, seed: int) -> dict:
    summary = json.loads(summary_path.read_text())
    return {
        "stem": stem,
        "horizon": horizon,
        "variant": summary.get("variant", variant),
        "requested_variant": variant,
        "seed": seed,
        "test_rollout_rmse": summary["test_rollout"]["rmse"],
        "test_rollout_nse": summary["test_rollout"]["nse"],
        "test_rollout_corr": summary["test_rollout"]["corr"],
        "test_final_rmse": summary.get("test", {}).get("rmse_final", float("nan")),
        "test_final_nse": summary.get("test", {}).get("nse_final", float("nan")),
        "best_val_loss": summary["best_val_loss"],
    }


def _float_equal(left: object, right: float, *, tol: float = 1.0e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def _resume_contract_mismatches(summary: dict, variant: str, args: argparse.Namespace) -> list[str]:
    if variant == "persistence":
        return []
    if variant in {"ode_only", "ode_only_bounded"}:
        expected = {
            "epochs": int(args.ode_epochs),
            "patience": int(args.ode_patience),
            "lr": float(args.ode_lr),
        }
    else:
        expected = {
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "hidden": int(args.hidden),
            "lr": float(args.lr),
            "lambda_penalty": float(args.lambda_penalty),
            "tau_days": float(args.tau_days),
        }
    mismatches = []
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            matches = _float_equal(actual, expected_value)
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(f"{key}: existing={actual!r} current={expected_value!r}")
    return mismatches


def _write_partial(rows: list[dict], partial_csv: Path) -> None:
    partial_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["stem", "horizon", "requested_variant", "seed"]).to_csv(partial_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True, help="Comma-separated site stems")
    ap.add_argument("--variants", default="gru,ws2,ode,ode_only")
    ap.add_argument("--horizons", default="1,3,7,14")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--tau-days", type=float, default=14.0)
    ap.add_argument("--ode-epochs", type=int, default=300)
    ap.add_argument("--ode-patience", type=int, default=30)
    ap.add_argument("--ode-lr", type=float, default=5.0e-2)
    ap.add_argument("--ode-tau-grid", default="3,7,14,30")
    ap.add_argument(
        "--groundwater-root",
        default=None,
        help="Optional root containing waterlevel/*_WT.txt and climate/*_CL.txt pairs.",
    )
    ap.add_argument(
        "--output-root",
        default=str(ROOT / "results/predictive_ladder_horizon"),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing partial CSV and summary.json files, skipping completed cells.",
    )
    ap.add_argument(
        "--allow-mixed-resume",
        action="store_true",
        help="Allow --resume to reuse existing summary.json files whose training budget differs from this command.",
    )
    ap.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
        help="Stop after this many newly executed cells; 0 means no limit.",
    )
    ap.add_argument(
        "--anonymize-progress",
        action="store_true",
        help="Print anonymous well labels in progress output instead of source stems.",
    )
    args = ap.parse_args()

    stems = [x.strip() for x in args.stems.split(",") if x.strip()]
    stem_labels = {stem: f"well_{idx:03d}" for idx, stem in enumerate(stems, start=1)}
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ode_tau_candidates = tuple(float(x) for x in args.ode_tau_grid.split(",") if x.strip())
    out_root = Path(args.output_root)
    groundwater_root = Path(args.groundwater_root) if args.groundwater_root else None
    rows = []
    partial_csv = out_root / "horizon_sensitivity_summary.partial.csv"
    total_runs = len(stems) * len(horizons) * len(seeds) * len(variants)
    run_idx = 0
    new_runs = 0
    completed: set[tuple[str, int, str, int]] = set()
    expected_keys = {
        (stem, horizon, variant, seed)
        for stem in stems
        for horizon in horizons
        for variant in variants
        for seed in seeds
    }
    partial_has_out_of_scope_rows = False
    stop_after_limit = False

    if args.resume and partial_csv.exists():
        partial_df = pd.read_csv(partial_csv)
        if "requested_variant" not in partial_df.columns:
            partial_df["requested_variant"] = partial_df["variant"]
        rows = partial_df.to_dict(orient="records")
        completed = {_row_key(row) for row in rows}
        partial_has_out_of_scope_rows = bool(completed - expected_keys)
        print(f"Resuming from {partial_csv}: {len(rows)} existing rows", flush=True)

    for stem in stems:
        series = load_ladder_series(stem, groundwater_root=groundwater_root) if groundwater_root else load_ladder_series(stem)
        splits = make_block_splits(len(series.head_interp))
        for horizon in horizons:
            split_data = build_sequence_split(series, splits, window=args.window, horizon=horizon)
            for seed in seeds:
                for variant in variants:
                    run_idx += 1
                    progress_stem = stem_labels[stem] if args.anonymize_progress else stem
                    key = (stem, horizon, variant, seed)
                    run_dir = out_root / f"h{horizon:02d}" / stem / f"{variant}_seed{seed}"
                    summary_path = run_dir / "summary.json"
                    if args.resume and summary_path.exists() and not args.allow_mixed_resume:
                        summary_for_contract = json.loads(summary_path.read_text())
                        mismatches = _resume_contract_mismatches(summary_for_contract, variant, args)
                        if mismatches:
                            raise SystemExit(
                                "Resume contract mismatch for "
                                f"stem={progress_stem} horizon={horizon} variant={variant} seed={seed}: "
                                + "; ".join(mismatches)
                                + ". Use a clean output root, rerun the cell under the intended budget, "
                                + "or pass --allow-mixed-resume only for explicitly accepted screening evidence."
                            )
                    if args.resume and key not in completed and summary_path.exists():
                        rows.append(_summary_row(summary_path, stem, horizon, variant, seed))
                        completed.add(key)
                        _write_partial(rows, partial_csv)
                    if args.resume and key in completed:
                        print(
                            f"[{run_idx}/{total_runs}] skip existing stem={progress_stem} horizon={horizon} variant={variant} seed={seed}",
                            flush=True,
                        )
                        continue
                    if args.max_new_runs and new_runs >= args.max_new_runs:
                        stop_after_limit = True
                        break
                    print(
                        f"[{run_idx}/{total_runs}] stem={progress_stem} horizon={horizon} variant={variant} seed={seed}",
                        flush=True,
                    )
                    if variant == "persistence":
                        outputs = {"test_rollout": rollout_persistence(series=series, split=splits.test, horizon=horizon)}
                        meta = {
                            "variant": "persistence",
                            "seed": seed,
                            "best_val_loss": float("nan"),
                        }
                    elif variant in {"ode_only", "ode_only_bounded"}:
                        outputs, meta = fit_standalone_ode(
                            split_data,
                            seed=seed,
                            epochs=args.ode_epochs,
                            patience=args.ode_patience,
                            lr=args.ode_lr,
                            tau_candidates=ode_tau_candidates,
                            bound_h_ref=variant == "ode_only_bounded",
                        )
                        outputs["test_rollout"] = rollout_standalone_ode(
                            series=series,
                            split=splits.test,
                            horizon=horizon,
                            params=ODEParams(**meta["physics_params"]),
                        )
                    else:
                        train_variant = "ode" if variant == "ode_bounded" else variant
                        model, outputs, meta = train_ladder_variant(
                            split_data,
                            variant=train_variant,
                            seed=seed,
                            epochs=args.epochs,
                            patience=args.patience,
                            hidden=args.hidden,
                            lr=args.lr,
                            lambda_penalty=args.lambda_penalty,
                            tau_days=args.tau_days,
                            bound_h_ref=variant == "ode_bounded",
                        )
                        outputs["test_rollout"] = rollout_sequence_model(
                            model=model,
                            series=series,
                            split=splits.test,
                            norm=split_data["norm"],
                            window=args.window,
                            horizon=horizon,
                        )
                    if variant == "persistence":
                        run_dir.mkdir(parents=True, exist_ok=True)
                        summary = {
                            "stem": series.stem,
                            **meta,
                            "test_rollout": outputs["test_rollout"]["metrics"],
                        }
                        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
                        import numpy as np

                        np.savez(
                            run_dir / "test_rollout_predictions.npz",
                            pred=np.asarray(outputs["test_rollout"]["pred"]),
                            obs=np.asarray(outputs["test_rollout"]["obs"]),
                            dates=np.asarray(outputs["test_rollout"]["dates"]),
                        )
                    else:
                        save_ladder_run(run_dir, series, outputs, meta)
                        summary = json.loads((run_dir / "summary.json").read_text())
                    rows.append(
                        {
                            "stem": stem,
                            "horizon": horizon,
                            "variant": summary.get("variant", variant),
                            "requested_variant": variant,
                            "seed": seed,
                            "test_rollout_rmse": summary["test_rollout"]["rmse"],
                            "test_rollout_nse": summary["test_rollout"]["nse"],
                            "test_rollout_corr": summary["test_rollout"]["corr"],
                            "test_final_rmse": summary.get("test", {}).get("rmse_final", float("nan")),
                            "test_final_nse": summary.get("test", {}).get("nse_final", float("nan")),
                            "best_val_loss": summary["best_val_loss"],
                        }
                    )
                    completed.add(key)
                    new_runs += 1
                    _write_partial(rows, partial_csv)
                if stop_after_limit:
                    break
            if stop_after_limit:
                break
        if stop_after_limit:
            break
    if stop_after_limit:
        print(f"Stopped after {new_runs} new run(s); partial CSV retained at {partial_csv}")
        return
    if args.resume and partial_has_out_of_scope_rows:
        _write_partial(rows, partial_csv)
        print(
            "Resume did not finalize because the partial CSV contains rows outside "
            "the current command contract; rerun with the full intended contract."
        )
        return
    df = pd.DataFrame(rows).sort_values(["stem", "horizon", "variant", "seed"])
    out_csv = out_root / "horizon_sensitivity_summary.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    partial_csv.unlink(missing_ok=True)
    manifest = {
        "n_success": int(len(rows)),
        "n_error": 0,
        "summary_csv": str(out_csv),
        "contract": {
            "stems": stems,
            "variants": variants,
            "horizons": horizons,
            "seeds": seeds,
            "window": int(args.window),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "hidden": int(args.hidden),
            "lr": float(args.lr),
            "lambda_penalty": float(args.lambda_penalty),
            "tau_days": float(args.tau_days),
            "ode_epochs": int(args.ode_epochs),
            "ode_patience": int(args.ode_patience),
            "ode_lr": float(args.ode_lr),
            "ode_tau_candidates": ode_tau_candidates,
            "groundwater_root": str(groundwater_root) if groundwater_root else None,
            "rollout": "recursive_block_rollout_by_requested_horizon",
        },
    }
    (out_root / "horizon_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out_csv)


if __name__ == "__main__":
    main()

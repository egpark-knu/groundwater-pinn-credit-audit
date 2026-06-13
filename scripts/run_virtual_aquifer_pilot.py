from __future__ import annotations

import argparse
from dataclasses import replace
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from groundwater_research.esmda import esmda_update, initialize_gaussian_ensemble
from groundwater_research.virtual_aquifer import (
    SiteSeries,
    VirtualAquiferConfig,
    VirtualAquiferParams,
    build_recharge_series,
    default_prior_for_site,
    default_prior_reduced_for_site,
    load_site_series,
    run_virtual_aquifer,
    save_forward_payload,
    suggest_archetype_from_catalog,
)


def select_window(site: SiteSeries, start: str, end: str) -> SiteSeries:
    mask = (site.dates >= np.datetime64(start)) & (site.dates <= np.datetime64(end))
    if int(mask.sum()) == 0:
        raise ValueError(
            f"Empty window for {site.stem}: {start} to {end}. "
            f"Available range is {site.dates[0]} to {site.dates[-1]}."
        )
    return SiteSeries(
        stem=site.stem,
        dates=site.dates[mask],
        obs_raw=site.obs_raw[mask],
        obs_interp=site.obs_interp[mask],
        obs_valid_mask=site.obs_valid_mask[mask],
        rain_mm=site.rain_mm[mask],
        temp_c=site.temp_c[mask],
        material_class=site.material_class,
        archetype=site.archetype,
    )


def slice_forward_payload_by_date(payload: dict, start: str, end: str) -> dict:
    dates = np.asarray(payload["dates"]).astype("datetime64[D]")
    mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    if int(mask.sum()) == 0:
        raise ValueError(f"Empty payload slice: {start} to {end}")

    obs = np.asarray(payload["obs"], dtype=float)[mask]
    pred = np.asarray(payload["pred_head"], dtype=float)[mask]
    valid = np.asarray(payload["valid_mask"]).astype(bool)[mask]
    if int(valid.sum()) == 0:
        raise ValueError(f"No valid observations in payload slice: {start} to {end}")

    residual = pred[valid] - obs[valid]
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    if len(residual) > 1 and np.nanstd(obs[valid]) > 0.0 and np.nanstd(pred[valid]) > 0.0:
        corr = float(np.corrcoef(pred[valid], obs[valid])[0, 1])
    else:
        corr = float("nan")
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((obs[valid] - obs[valid].mean()) ** 2))
    nse = float(1.0 - ss_res / (ss_tot + 1.0e-12))

    sliced = {
        "pred_head": pred,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "nse": nse,
        "params": payload["params"],
        "steady_head": payload.get("steady_head"),
        "dates": dates[mask].astype(str),
        "obs": obs,
        "valid_mask": valid.astype(int),
    }
    return sliced


def params_from_theta(theta: np.ndarray, tau_days: float) -> VirtualAquiferParams:
    return VirtualAquiferParams(
        log_k1=float(theta[0]),
        log_k2=float(theta[1]),
        log_sy1=float(theta[2]),
        log_ghb_mult=float(theta[3]),
        h_ref=float(theta[4]),
        tau_rch_days=float(tau_days),
    )


def params_from_theta_reduced(theta: np.ndarray, tau_days: float) -> VirtualAquiferParams:
    return VirtualAquiferParams(
        log_k1=float(theta[0]),
        log_k2=float(theta[0]),
        log_sy1=float(theta[1]),
        log_ghb_mult=0.0,
        h_ref=float(theta[2]),
        tau_rch_days=float(tau_days),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="상주신상_충적")
    ap.add_argument("--archetype", choices=["auto", "coastal", "inland"], default="auto")
    ap.add_argument("--train-start", default="2005-01-01")
    ap.add_argument("--train-end", default="2008-12-31")
    ap.add_argument("--test-start", default="2009-01-01")
    ap.add_argument("--test-end", default="2009-12-31")
    ap.add_argument("--nens", type=int, default=10)
    ap.add_argument("--n-assim", type=int, default=2)
    ap.add_argument("--obs-error", type=float, default=0.25)
    ap.add_argument("--tau-days", type=float, default=1.0)
    ap.add_argument("--weekly-stride", type=int, default=7)
    ap.add_argument("--param-mode", choices=["reduced", "full"], default="reduced")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "results/solver_audit/pilot"),
    )
    ap.add_argument("--seed", type=int, default=260409)
    args = ap.parse_args()

    resolved_archetype = args.archetype
    if resolved_archetype == "auto":
        resolved_archetype = suggest_archetype_from_catalog(args.stem) or "inland"

    site_full = load_site_series(args.stem, archetype=resolved_archetype)
    train_site = select_window(site_full, args.train_start, args.train_end)
    test_site = select_window(site_full, args.test_start, args.test_end)
    eval_site = select_window(site_full, args.train_start, args.test_end)
    config = VirtualAquiferConfig()
    train_steady_recharge = float(
        np.mean(
            build_recharge_series(
                train_site.rain_mm,
                recharge_fraction=config.recharge_fraction,
                tau_days=args.tau_days,
            )
        )
    )
    eval_config = replace(config, steady_recharge_m_per_day=train_steady_recharge)
    if args.param_mode == "reduced":
        mean, std, lower, upper = default_prior_reduced_for_site(train_site)
        parameter_names = ["log_k_eff", "log_sy1", "h_ref"]
        make_params = params_from_theta_reduced
    else:
        mean, std, lower, upper = default_prior_for_site(train_site)
        parameter_names = [
            "log_k1",
            "log_k2",
            "log_sy1",
            "log_ghb_mult",
            "h_ref",
        ]
        make_params = params_from_theta
    rng = np.random.default_rng(args.seed)
    theta = initialize_gaussian_ensemble(mean, std, args.nens, rng, lower, upper)
    theta_prior = theta.copy()

    out_root = Path(args.output_dir) / args.stem
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    obs_idx = np.arange(0, len(train_site.dates), args.weekly_stride)
    obs_assim = train_site.obs_interp[obs_idx]
    history: list[dict] = []

    for assim in range(args.n_assim):
        preds = []
        rmse_train = []
        for member in range(args.nens):
            member_ws = out_root / f"assim_{assim+1:02d}" / f"member_{member:03d}"
            params = make_params(theta[member], args.tau_days)
            ok, payload = run_virtual_aquifer(member_ws, train_site, params, config)
            if not ok:
                raise RuntimeError(f"MF6 failed for member {member} in assimilation {assim+1}")
            preds.append(payload["pred_head"][obs_idx])
            rmse_train.append(payload["rmse"])
        preds_arr = np.asarray(preds, dtype=float)
        theta = esmda_update(
            theta=theta,
            predicted=preds_arr,
            observed=obs_assim,
            alpha=float(args.n_assim),
            obs_error_std=args.obs_error,
            rng=rng,
            lower=lower,
            upper=upper,
        )
        history.append(
            {
                "assimilation": assim + 1,
                "mean_member_train_rmse": float(np.mean(rmse_train)),
                "best_member_train_rmse": float(np.min(rmse_train)),
            }
        )

    final_results = []
    for member in range(args.nens):
        member_ws = out_root / "final_eval" / f"member_{member:03d}"
        params = make_params(theta[member], args.tau_days)
        ok_eval, payload_eval = run_virtual_aquifer(member_ws / "continuous", eval_site, params, eval_config)
        if not ok_eval:
            raise RuntimeError(f"Final evaluation failed for member {member}")
        payload_train = slice_forward_payload_by_date(payload_eval, args.train_start, args.train_end)
        payload_test = slice_forward_payload_by_date(payload_eval, args.test_start, args.test_end)
        final_results.append(
            {
                "member": member,
                "train_rmse": payload_train["rmse"],
                "train_nse": payload_train["nse"],
                "test_rmse": payload_test["rmse"],
                "test_nse": payload_test["nse"],
                "params": payload_test["params"],
                "payload_train": payload_train,
                "payload_test": payload_test,
            }
        )

    best = min(final_results, key=lambda item: item["train_rmse"])
    best_train_npz = out_root / "best_member_train.npz"
    best_test_npz = out_root / "best_member_test.npz"
    best_legacy_npz = out_root / "best_member_forward.npz"
    save_forward_payload(best_train_npz, best["payload_train"])
    save_forward_payload(best_test_npz, best["payload_test"])
    save_forward_payload(best_legacy_npz, best["payload_test"])
    np.save(out_root / "theta_prior.npy", theta_prior)
    np.save(out_root / "theta_posterior.npy", theta)

    summary = {
        "stem": args.stem,
        "archetype": resolved_archetype,
        "archetype_requested": args.archetype,
        "param_mode": args.param_mode,
        "parameter_names": parameter_names,
        "train_window": [args.train_start, args.train_end],
        "test_window": [args.test_start, args.test_end],
        "nens": args.nens,
        "n_assim": args.n_assim,
        "obs_error": args.obs_error,
        "tau_days": args.tau_days,
        "recharge_policy": "fixed_RPR_0.20_of_daily_precipitation",
        "steady_state_initialization": "one steady-state stress period before transient daily simulation",
        "final_evaluation": "single continuous train-start to test-end transient simulation sliced into train/test metrics",
        "steady_recharge_m_per_day": train_steady_recharge,
        "recharge_fraction": config.recharge_fraction,
        "weekly_stride": args.weekly_stride,
        "assimilation_history": history,
        "best_member": {
            "member": int(best["member"]),
            "train_rmse": float(best["train_rmse"]),
            "train_nse": float(best["train_nse"]),
            "test_rmse": float(best["test_rmse"]),
            "test_nse": float(best["test_nse"]),
            "params": best["params"],
        },
    }
    (out_root / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEFAULT_GROUNDWATER_ROOT = Path(
    os.environ.get("NGMS_GROUNDWATER_ROOT", Path(__file__).resolve().parents[2] / "data" / "groundwater")
)
CLIMATE_COLS = ["TEMP", "RAIN", "HUMID", "HPA", "WIND"]


@dataclass
class LadderSeries:
    stem: str
    dates: np.ndarray
    head_raw: np.ndarray
    head_interp: np.ndarray
    rain_mm: np.ndarray
    climate: np.ndarray
    climate_cols: list[str]


@dataclass
class BlockSplits:
    train: slice
    val: slice
    test: slice


class ForecastGRUSeq(nn.Module):
    def __init__(self, n_past_feat: int, n_future_feat: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(n_past_feat, hidden, batch_first=True)
        self.future_proj = nn.Sequential(
            nn.Linear(n_future_feat * horizon, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x_past)
        h_last = h[:, -1, :]
        fut = self.future_proj(x_future.reshape(x_future.size(0), -1))
        return self.head(torch.cat([h_last, fut], dim=-1))


def _read_pair(stem: str, groundwater_root: Path = DEFAULT_GROUNDWATER_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = unicodedata.normalize("NFC", stem)
    wt_path = groundwater_root / "waterlevel" / f"{stem}_WT.txt"
    cl_path = groundwater_root / "climate" / f"{stem}_CL.txt"
    if not wt_path.exists():
        wt_path = groundwater_root / "waterlevel" / unicodedata.normalize("NFD", f"{stem}_WT.txt")
    if not cl_path.exists():
        cl_path = groundwater_root / "climate" / unicodedata.normalize("NFD", f"{stem}_CL.txt")
    if not wt_path.exists() or not cl_path.exists():
        raise FileNotFoundError(f"Missing WT/CL pair for {stem}")
    return pd.read_csv(wt_path, sep="\t"), pd.read_csv(cl_path, sep="\t")


def load_ladder_series(stem: str, groundwater_root: Path = DEFAULT_GROUNDWATER_ROOT) -> LadderSeries:
    wt_df, cl_df = _read_pair(stem, groundwater_root)
    df = wt_df.merge(cl_df, on="Date", how="inner")
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")

    head_raw = pd.to_numeric(df["Value"], errors="coerce").to_numpy(dtype=float)
    head_interp = pd.Series(head_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)

    climate_cols = []
    climate_arrs = []
    for col in CLIMATE_COLS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both").bfill().ffill()
        else:
            values = pd.Series(np.zeros(len(df), dtype=float))
        climate_cols.append(col)
        climate_arrs.append(values.to_numpy(dtype=float))
    climate = np.column_stack(climate_arrs).astype(np.float32)
    rain_idx = climate_cols.index("RAIN")
    rain_mm = climate[:, rain_idx].astype(float)

    return LadderSeries(
        stem=unicodedata.normalize("NFC", stem),
        dates=df["date"].to_numpy(dtype="datetime64[ns]"),
        head_raw=head_raw,
        head_interp=head_interp,
        rain_mm=rain_mm,
        climate=climate,
        climate_cols=climate_cols,
    )


def make_block_splits(n: int, train_frac: float = 0.6, val_frac: float = 0.2) -> BlockSplits:
    i_tr = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return BlockSplits(train=slice(0, i_tr), val=slice(i_tr, i_val), test=slice(i_val, n))


def _zscore_train(arr: np.ndarray, split: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = arr[split]
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1.0e-9
    return (arr - mu) / sd, mu, sd


def build_sequence_split(
    series: LadderSeries,
    splits: BlockSplits,
    window: int,
    horizon: int,
) -> dict[str, dict[str, np.ndarray]]:
    features = np.concatenate([series.climate, series.head_interp[:, None]], axis=1)
    feat_norm, feat_mu, feat_sd = _zscore_train(features, splits.train)
    head_norm = feat_norm[:, -1]
    climate_norm = feat_norm[:, :-1]
    head_mu = float(feat_mu[-1])
    head_sd = float(feat_sd[-1])

    def build(split: slice) -> dict[str, np.ndarray]:
        n_sub = split.stop - split.start - window - horizon + 1
        if n_sub <= 0:
            raise ValueError(f"Split too short for window={window}, horizon={horizon}: {split}")
        starts = np.arange(split.start, split.stop - window - horizon + 1)
        x_past = np.stack([feat_norm[i : i + window] for i in starts]).astype(np.float32)
        x_future = np.stack([climate_norm[i + window : i + window + horizon] for i in starts]).astype(np.float32)
        y_seq = np.stack([head_norm[i + window : i + window + horizon] for i in starts]).astype(np.float32)
        y_seq_phys = np.stack([series.head_interp[i + window : i + window + horizon] for i in starts]).astype(np.float32)
        rain_future = np.stack([series.rain_mm[i + window : i + window + horizon] for i in starts]).astype(np.float32)
        h_start = np.array([series.head_interp[i + window - 1] for i in starts], dtype=np.float32)
        forecast_dates = np.stack([series.dates[i + window : i + window + horizon] for i in starts]).astype("datetime64[D]")
        return {
            "x_past": x_past,
            "x_future": x_future,
            "y_seq": y_seq,
            "y_seq_phys": y_seq_phys,
            "rain_future": rain_future,
            "h_start": h_start,
            "forecast_dates": forecast_dates,
            "target_dates": forecast_dates[:, -1],
        }

    return {
        "train": build(splits.train),
        "val": build(splits.val),
        "test": build(splits.test),
        "norm": {
            "feat_mu": feat_mu.astype(np.float32),
            "feat_sd": feat_sd.astype(np.float32),
            "head_mu": head_mu,
            "head_sd": head_sd,
            "climate_mu": feat_mu[:-1].astype(np.float32),
            "climate_sd": feat_sd[:-1].astype(np.float32),
        },
    }


def denorm_head(y_norm: torch.Tensor, head_mu: float, head_sd: float) -> torch.Tensor:
    return y_norm * head_sd + head_mu


def ws1_penalty(y_phys: torch.Tensor, h_start: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([h_start[:, None], y_phys[:, :-1]], dim=1)
    return ((y_phys - prev) ** 2).mean()


def ws2_penalty(y_phys: torch.Tensor, h_start: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([h_start[:, None], y_phys[:, :-1]], dim=1)
    prev2 = torch.cat([h_start[:, None], prev[:, :-1]], dim=1)
    second = y_phys - 2.0 * prev + prev2
    return (second[:, 1:] ** 2).mean()


def ode_penalty(
    y_phys: torch.Tensor,
    h_start: torch.Tensor,
    rain_future: torch.Tensor,
    gamma_r: torch.Tensor,
    gamma_d: torch.Tensor,
    h_ref: torch.Tensor,
    tau_days: float,
) -> torch.Tensor:
    alpha = float(np.exp(-1.0 / max(tau_days, 1.0)))
    p_eff = []
    prev_p = rain_future[:, 0]
    p_eff.append(prev_p)
    for idx in range(1, rain_future.size(1)):
        prev_p = alpha * prev_p + (1.0 - alpha) * rain_future[:, idx]
        p_eff.append(prev_p)
    p_eff_t = torch.stack(p_eff, dim=1) * 1.0e-3
    prev_h = torch.cat([h_start[:, None], y_phys[:, :-1]], dim=1)
    dh = y_phys - prev_h
    rhs = gamma_r * p_eff_t - gamma_d * (prev_h - h_ref)
    return ((dh - rhs) ** 2).mean()


def h_ref_bounds_from_train(h_start: np.ndarray, y_seq_phys: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(h_start, dtype=float).ravel(), np.asarray(y_seq_phys, dtype=float).ravel()])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return lo - 1.0, hi + 1.0
    return lo, hi


def init_raw_h_ref(mean_h: float, h_ref_bounds: tuple[float, float] | None) -> torch.Tensor:
    if h_ref_bounds is None:
        return torch.tensor(float(mean_h), requires_grad=True)
    lo, hi = h_ref_bounds
    scaled = (float(mean_h) - lo) / max(hi - lo, 1.0e-6)
    scaled = float(np.clip(scaled, 1.0e-4, 1.0 - 1.0e-4))
    return torch.tensor(float(np.log(scaled / (1.0 - scaled))), requires_grad=True)


def bounded_h_ref(raw_h_ref: torch.Tensor, h_ref_bounds: tuple[float, float] | None) -> torch.Tensor:
    if h_ref_bounds is None:
        return raw_h_ref
    lo, hi = h_ref_bounds
    return float(lo) + float(hi - lo) * torch.sigmoid(raw_h_ref)


def metrics_from_sequence(pred_seq_phys: np.ndarray, obs_seq_phys: np.ndarray) -> dict:
    pred_final = pred_seq_phys[:, -1].astype(float)
    obs_final = obs_seq_phys[:, -1].astype(float)
    resid = pred_final - obs_final
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs_final - obs_final.mean()) ** 2))
    nse = float(1.0 - ss_res / (ss_tot + 1.0e-12))
    corr = float(np.corrcoef(pred_final, obs_final)[0, 1]) if pred_final.std() > 1e-9 and obs_final.std() > 1e-9 else float("nan")
    seq_rmse = float(np.sqrt(np.mean((pred_seq_phys.astype(float) - obs_seq_phys.astype(float)) ** 2)))
    return {
        "rmse_final": rmse,
        "mae_final": mae,
        "bias_final": bias,
        "nse_final": nse,
        "corr_final": corr,
        "rmse_seq": seq_rmse,
    }


def metrics_from_rollout(pred: np.ndarray, obs: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    resid = pred - obs
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    nse = float(1.0 - ss_res / (ss_tot + 1.0e-12))
    corr = float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 1.0e-9 and obs.std() > 1.0e-9 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "nse": nse,
        "corr": corr,
        "n_pred_days": int(pred.size),
    }


def lag_diagnostic(pred: np.ndarray, obs: np.ndarray, max_lag: int) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    best_lag = 0
    best_corr = float("-inf")
    lag0_corr = float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 1.0e-9 and obs.std() > 1.0e-9 else float("nan")
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = pred[-lag:]
            y = obs[: len(x)]
        elif lag > 0:
            x = pred[:-lag]
            y = obs[lag:]
        else:
            x = pred
            y = obs
        if len(x) < 10 or x.std() < 1.0e-9 or y.std() < 1.0e-9:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if corr > best_corr:
            best_lag = lag
            best_corr = corr
    return {
        "lag0_corr": lag0_corr,
        "best_lag_days": int(best_lag),
        "best_lag_corr": float(best_corr),
    }


def peak_timing_diagnostic(pred: np.ndarray, obs: np.ndarray) -> dict[str, int | float]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    peak_lag = int(np.argmax(pred) - np.argmax(obs))
    trough_lag = int(np.argmin(pred) - np.argmin(obs))
    if len(pred) >= 2 and len(obs) >= 2:
        pred_drop = int(np.argmin(np.diff(pred)))
        obs_drop = int(np.argmin(np.diff(obs)))
        pred_rise = int(np.argmax(np.diff(pred)))
        obs_rise = int(np.argmax(np.diff(obs)))
        drop_lag = pred_drop - obs_drop
        rise_lag = pred_rise - obs_rise
    else:
        drop_lag = 0
        rise_lag = 0
    return {
        "peak_lag_days": int(peak_lag),
        "trough_lag_days": int(trough_lag),
        "drop_lag_days": int(drop_lag),
        "rise_lag_days": int(rise_lag),
    }


def rollout_sequence_model(
    model: nn.Module,
    series: LadderSeries,
    split: slice,
    norm: dict[str, np.ndarray | float],
    window: int,
    horizon: int,
) -> dict[str, np.ndarray | dict]:
    if split.start < window:
        raise ValueError(f"Rollout split start {split.start} is shorter than window={window}")

    feat_mu = np.asarray(norm["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(norm["feat_sd"], dtype=np.float32)
    climate_mu = np.asarray(norm["climate_mu"], dtype=np.float32)
    climate_sd = np.asarray(norm["climate_sd"], dtype=np.float32)
    head_mu = float(norm["head_mu"])
    head_sd = float(norm["head_sd"])

    pred_all: list[np.ndarray] = []
    obs_all: list[np.ndarray] = []
    date_all: list[np.ndarray] = []

    model.eval()
    t = split.start
    while t < split.stop:
        block_len = min(horizon, split.stop - t)
        past_feat = np.concatenate(
            [
                series.climate[t - window : t],
                series.head_interp[t - window : t, None],
            ],
            axis=1,
        ).astype(np.float32)
        x_past = ((past_feat - feat_mu) / feat_sd)[None, :, :]

        future_clim = series.climate[t : t + block_len].astype(np.float32)
        if block_len < horizon:
            pad = np.repeat(future_clim[-1:, :], horizon - block_len, axis=0)
            future_clim = np.concatenate([future_clim, pad], axis=0)
        x_future = ((future_clim - climate_mu) / climate_sd)[None, :, :]

        with torch.no_grad():
            pred_norm = model(torch.from_numpy(x_past), torch.from_numpy(x_future))
            pred_block = denorm_head(pred_norm, head_mu, head_sd).cpu().numpy()[0, :block_len]

        pred_all.append(pred_block.astype(np.float32))
        obs_all.append(series.head_interp[t : t + block_len].astype(np.float32))
        date_all.append(series.dates[t : t + block_len].astype("datetime64[D]"))
        t += horizon

    pred = np.concatenate(pred_all)
    obs = np.concatenate(obs_all)
    dates = np.concatenate(date_all)
    metrics = metrics_from_rollout(pred, obs)
    metrics.update(lag_diagnostic(pred, obs, max_lag=max(horizon, 14)))
    metrics.update(peak_timing_diagnostic(pred, obs))
    return {
        "pred": pred,
        "obs": obs,
        "dates": dates.astype("datetime64[D]").astype(str),
        "metrics": metrics,
    }


def train_ladder_variant(
    split_data: dict[str, dict[str, np.ndarray]],
    variant: str,
    seed: int = 42,
    epochs: int = 150,
    patience: int = 25,
    hidden: int = 64,
    lr: float = 1.0e-3,
    lambda_penalty: float = 1.0,
    tau_days: float = 14.0,
    bound_h_ref: bool = False,
) -> tuple[nn.Module, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]
    norm = split_data["norm"]

    model = ForecastGRUSeq(
        n_past_feat=train["x_past"].shape[-1],
        n_future_feat=train["x_future"].shape[-1],
        horizon=train["y_seq"].shape[-1],
        hidden=hidden,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1.0e-5)

    raw_gamma_r = raw_gamma_d = raw_h_ref = None
    opt_phys = None
    if variant == "ode":
        h_ref_bounds = h_ref_bounds_from_train(train["h_start"], train["y_seq_phys"]) if bound_h_ref else None
        raw_gamma_r = torch.tensor(-4.0, requires_grad=True)
        raw_gamma_d = torch.tensor(-1.0, requires_grad=True)
        raw_h_ref = init_raw_h_ref(float(train["h_start"].mean()), h_ref_bounds)
        opt_phys = torch.optim.Adam([raw_gamma_r, raw_gamma_d, raw_h_ref], lr=5.0e-2)
    else:
        h_ref_bounds = None

    tt = lambda x: torch.from_numpy(x)
    xpt = tt(train["x_past"])
    xft = tt(train["x_future"])
    ytr = tt(train["y_seq"])
    ytr_phys = tt(train["y_seq_phys"])
    hst = tt(train["h_start"])
    rft = tt(train["rain_future"])
    xpv = tt(val["x_past"])
    xfv = tt(val["x_future"])
    yvv = tt(val["y_seq"])
    yvv_phys = tt(val["y_seq_phys"])

    lossfn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    best_phys = None
    no_imp = 0
    batch_size = 64

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xpt))
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            if opt_phys is not None:
                opt_phys.zero_grad()
            pred = model(xpt[idx], xft[idx])
            loss = lossfn(pred, ytr[idx])
            pred_phys = denorm_head(pred, norm["head_mu"], norm["head_sd"])
            if variant == "ws1":
                loss = loss + lambda_penalty * ws1_penalty(pred_phys, hst[idx])
            elif variant == "ws2":
                loss = loss + lambda_penalty * ws2_penalty(pred_phys, hst[idx])
            elif variant == "ode":
                gamma_r = 1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)
                gamma_d = 1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)
                h_ref = bounded_h_ref(raw_h_ref, h_ref_bounds)
                loss = loss + lambda_penalty * ode_penalty(
                    pred_phys,
                    hst[idx],
                    rft[idx],
                    gamma_r=gamma_r,
                    gamma_d=gamma_d,
                    h_ref=h_ref,
                    tau_days=tau_days,
                )
            loss.backward()
            opt.step()
            if opt_phys is not None:
                opt_phys.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(xpv, xfv)
            val_loss = lossfn(pred_val, yvv).item()
            pred_val_phys = denorm_head(pred_val, norm["head_mu"], norm["head_sd"]).numpy()
            val_metrics = metrics_from_sequence(pred_val_phys, yvv_phys.numpy())
        sched.step(val_loss)

        if val_loss < best_val - 1.0e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if variant == "ode":
                best_phys = {
                    "gamma_r": float((1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)).item()),
                    "gamma_d": float((1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)).item()),
                    "h_ref": float(bounded_h_ref(raw_h_ref, h_ref_bounds).item()),
                }
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    outputs = {}
    for name, block in [("train", train), ("val", val), ("test", test)]:
        with torch.no_grad():
            pred_norm = model(tt(block["x_past"]), tt(block["x_future"]))
        pred_phys = denorm_head(pred_norm, norm["head_mu"], norm["head_sd"]).cpu().numpy()
        metrics = metrics_from_sequence(pred_phys, block["y_seq_phys"])
        outputs[name] = {
            "pred_seq_phys": pred_phys,
            "obs_seq_phys": block["y_seq_phys"],
            "target_dates": block["target_dates"].astype("datetime64[D]").astype(str),
            "forecast_dates": block["forecast_dates"].astype("datetime64[D]").astype(str),
            "metrics": metrics,
        }

    meta = {
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "hidden": hidden,
        "lr": lr,
        "lambda_penalty": lambda_penalty,
        "tau_days": tau_days,
        "best_val_loss": best_val,
    }
    if best_phys is not None:
        meta["variant"] = "ode_bounded_h_ref" if bound_h_ref else variant
        meta["h_ref_bound_mode"] = "train_observed_range" if bound_h_ref else "unbounded"
        meta["h_ref_bounds"] = list(h_ref_bounds) if h_ref_bounds is not None else None
        meta["physics_params"] = best_phys
    return model, outputs, meta


def save_ladder_run(output_dir: Path, series: LadderSeries, outputs: dict, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stem": series.stem,
        **meta,
        "train": outputs["train"]["metrics"],
        "val": outputs["val"]["metrics"],
        "test": outputs["test"]["metrics"],
    }
    if "test_rollout" in outputs:
        summary["test_rollout"] = outputs["test_rollout"]["metrics"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for split_name, block in outputs.items():
        if split_name == "test_rollout":
            np.savez(
                output_dir / "test_rollout_predictions.npz",
                pred=np.asarray(block["pred"]),
                obs=np.asarray(block["obs"]),
                dates=np.asarray(block["dates"]),
            )
            continue
        np.savez(
            output_dir / f"{split_name}_predictions.npz",
            pred_seq_phys=block["pred_seq_phys"],
            obs_seq_phys=block["obs_seq_phys"],
            target_dates=np.asarray(block["target_dates"]),
            forecast_dates=np.asarray(block["forecast_dates"]),
        )

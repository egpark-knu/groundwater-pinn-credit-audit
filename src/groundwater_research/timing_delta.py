from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .neural_ladder import (
    BlockSplits,
    LadderSeries,
    denorm_head,
    lag_diagnostic,
    load_ladder_series,
    make_block_splits,
    metrics_from_rollout,
    ode_penalty,
    peak_timing_diagnostic,
    ws2_penalty,
)


class ForecastGRUAutoHead(nn.Module):
    def __init__(self, n_past_feat: int, n_future_feat: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.horizon = horizon
        self.enc = nn.GRU(n_past_feat, hidden, batch_first=True)
        self.dec = nn.GRUCell(n_future_feat + 1, hidden)
        self.head_out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor, h0_norm: torch.Tensor) -> torch.Tensor:
        h, _ = self.enc(x_past)
        state = h[:, -1, :]
        prev_head = h0_norm
        outs = []
        for t in range(self.horizon):
            dec_in = torch.cat([x_future[:, t, :], prev_head[:, None]], dim=-1)
            state = self.dec(dec_in, state)
            next_head = self.head_out(state).squeeze(-1)
            outs.append(next_head)
            prev_head = next_head
        return torch.stack(outs, dim=1)


def _zscore_train(arr: np.ndarray, split: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = arr[split]
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1.0e-9
    return (arr - mu) / sd, mu, sd


def build_delta_split(
    series: LadderSeries,
    splits: BlockSplits,
    window: int,
    horizon: int,
) -> dict[str, dict[str, np.ndarray]]:
    head = series.head_interp.astype(np.float32)
    dhead = np.diff(head, prepend=head[0]).astype(np.float32)
    features = np.concatenate([series.climate, head[:, None], dhead[:, None]], axis=1)
    feat_norm, feat_mu, feat_sd = _zscore_train(features, splits.train)

    n_clim = series.climate.shape[1]
    climate_mu = feat_mu[:n_clim].astype(np.float32)
    climate_sd = feat_sd[:n_clim].astype(np.float32)
    head_mu = float(head[splits.train].mean())
    head_sd = float(head[splits.train].std() + 1.0e-9)

    def build(split: slice) -> dict[str, np.ndarray]:
        starts = np.arange(split.start, split.stop - window - horizon + 1)
        if len(starts) == 0:
            raise ValueError(f"Split too short for window={window}, horizon={horizon}: {split}")

        x_past = np.stack([feat_norm[i : i + window] for i in starts]).astype(np.float32)
        x_future = np.stack(
            [
                ((series.climate[i + window : i + window + horizon] - climate_mu) / climate_sd).astype(np.float32)
                for i in starts
            ]
        ).astype(np.float32)
        y_head_phys = np.stack([head[i + window : i + window + horizon] for i in starts]).astype(np.float32)
        h_start = np.array([head[i + window - 1] for i in starts], dtype=np.float32)
        y_head_norm = ((y_head_phys - head_mu) / head_sd).astype(np.float32)
        y_delta_phys = np.empty_like(y_head_phys)
        y_delta_phys[:, 0] = y_head_phys[:, 0] - h_start
        y_delta_phys[:, 1:] = y_head_phys[:, 1:] - y_head_phys[:, :-1]
        rain_future = np.stack([series.rain_mm[i + window : i + window + horizon] for i in starts]).astype(np.float32)
        forecast_dates = np.stack([series.dates[i + window : i + window + horizon] for i in starts]).astype("datetime64[D]")
        return {
            "x_past": x_past,
            "x_future": x_future,
            "y_head_phys": y_head_phys,
            "y_head_norm": y_head_norm,
            "y_delta_phys": y_delta_phys,
            "h_start": h_start,
            "rain_future": rain_future,
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
            "climate_mu": climate_mu,
            "climate_sd": climate_sd,
            "head_mu": head_mu,
            "head_sd": head_sd,
        },
    }


def head_to_delta(head_phys: torch.Tensor, h_start: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([h_start[:, None], head_phys[:, :-1]], dim=1)
    return head_phys - prev


def _metrics_from_head_seq(pred_head_phys: np.ndarray, obs_head_phys: np.ndarray) -> dict[str, float]:
    resid = pred_head_phys[:, -1] - obs_head_phys[:, -1]
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs_head_phys[:, -1] - obs_head_phys[:, -1].mean()) ** 2))
    return {
        "rmse_final": float(np.sqrt(np.mean(resid**2))),
        "mae_final": float(np.mean(np.abs(resid))),
        "bias_final": float(np.mean(resid)),
        "nse_final": float(1.0 - ss_res / (ss_tot + 1.0e-12)),
        "corr_final": float(np.corrcoef(pred_head_phys[:, -1], obs_head_phys[:, -1])[0, 1])
        if pred_head_phys[:, -1].std() > 1.0e-9 and obs_head_phys[:, -1].std() > 1.0e-9
        else float("nan"),
        "rmse_seq": float(np.sqrt(np.mean((pred_head_phys - obs_head_phys) ** 2))),
    }


def train_delta_variant(
    split_data: dict[str, dict[str, np.ndarray]],
    variant: str,
    seed: int = 42,
    epochs: int = 150,
    patience: int = 25,
    hidden: int = 64,
    lr: float = 1.0e-3,
    lambda_penalty: float = 1.0,
    delta_loss_weight: float = 1.0,
    tau_days: float = 14.0,
) -> tuple[nn.Module, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]
    norm = split_data["norm"]

    model = ForecastGRUAutoHead(
        n_past_feat=train["x_past"].shape[-1],
        n_future_feat=train["x_future"].shape[-1],
        horizon=train["y_head_norm"].shape[-1],
        hidden=hidden,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1.0e-5)

    raw_gamma_r = raw_gamma_d = raw_h_ref = None
    opt_phys = None
    if variant == "ode":
        raw_gamma_r = torch.tensor(-4.0, requires_grad=True)
        raw_gamma_d = torch.tensor(-1.0, requires_grad=True)
        raw_h_ref = torch.tensor(float(train["h_start"].mean()), requires_grad=True)
        opt_phys = torch.optim.Adam([raw_gamma_r, raw_gamma_d, raw_h_ref], lr=5.0e-2)

    tt = lambda x: torch.from_numpy(x)
    xpt = tt(train["x_past"])
    xft = tt(train["x_future"])
    yht = tt(train["y_head_norm"])
    yht_phys = tt(train["y_head_phys"])
    ydt_phys = tt(train["y_delta_phys"])
    hst = tt(train["h_start"])
    hst_norm = (hst - norm["head_mu"]) / norm["head_sd"]
    rft = tt(train["rain_future"])

    xpv = tt(val["x_past"])
    xfv = tt(val["x_future"])
    yhv = tt(val["y_head_norm"])
    yhv_phys = tt(val["y_head_phys"])
    ydv_phys = tt(val["y_delta_phys"])
    hsv = tt(val["h_start"])
    hsv_norm = (hsv - norm["head_mu"]) / norm["head_sd"]
    rfv = tt(val["rain_future"])

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
            pred_head_norm = model(xpt[idx], xft[idx], hst_norm[idx])
            pred_head_phys = denorm_head(pred_head_norm, norm["head_mu"], norm["head_sd"])
            pred_delta_phys = head_to_delta(pred_head_phys, hst[idx])
            loss = lossfn(pred_head_norm, yht[idx]) + delta_loss_weight * lossfn(pred_delta_phys, ydt_phys[idx])
            if variant == "ws2":
                loss = loss + lambda_penalty * ws2_penalty(pred_head_phys, hst[idx])
            elif variant == "ode":
                gamma_r = 1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)
                gamma_d = 1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)
                h_ref = raw_h_ref
                loss = loss + lambda_penalty * ode_penalty(
                    pred_head_phys,
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
            pred_head_norm = model(xpv, xfv, hsv_norm)
            pred_head_phys = denorm_head(pred_head_norm, norm["head_mu"], norm["head_sd"])
            pred_delta_phys = head_to_delta(pred_head_phys, hsv)
            val_loss = (lossfn(pred_head_norm, yhv) + delta_loss_weight * lossfn(pred_delta_phys, ydv_phys)).item()
        sched.step(val_loss)

        if val_loss < best_val - 1.0e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if variant == "ode":
                best_phys = {
                    "gamma_r": float((1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)).item()),
                    "gamma_d": float((1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)).item()),
                    "h_ref": float(raw_h_ref.item()),
                }
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state is None:
        raise RuntimeError("Timing-delta variant failed to produce a best state.")
    model.load_state_dict(best_state)
    model.eval()

    outputs: dict[str, dict] = {}
    for name, block in [("train", train), ("val", val), ("test", test)]:
        with torch.no_grad():
            h0_norm = (tt(block["h_start"]) - norm["head_mu"]) / norm["head_sd"]
            pred_head_norm = model(tt(block["x_past"]), tt(block["x_future"]), h0_norm)
        pred_head_phys = denorm_head(pred_head_norm, norm["head_mu"], norm["head_sd"]).cpu().numpy()
        outputs[name] = {
            "pred_seq_phys": pred_head_phys,
            "obs_seq_phys": block["y_head_phys"],
            "target_dates": block["target_dates"].astype("datetime64[D]").astype(str),
            "forecast_dates": block["forecast_dates"].astype("datetime64[D]").astype(str),
            "metrics": _metrics_from_head_seq(pred_head_phys, block["y_head_phys"]),
        }

    meta = {
        "variant": f"delta_{variant}",
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "hidden": hidden,
        "lr": lr,
        "lambda_penalty": lambda_penalty,
        "delta_loss_weight": delta_loss_weight,
        "tau_days": tau_days,
        "best_val_loss": best_val,
        "decoder_mode": "autoregressive_head",
    }
    if best_phys is not None:
        meta["physics_params"] = best_phys
    return model, outputs, meta


def rollout_delta_model(
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
    head = series.head_interp.astype(np.float32)
    dhead = np.diff(head, prepend=head[0]).astype(np.float32)

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
                head[t - window : t, None],
                dhead[t - window : t, None],
            ],
            axis=1,
        ).astype(np.float32)
        x_past = ((past_feat - feat_mu) / feat_sd)[None, :, :]
        future_clim = series.climate[t : t + block_len].astype(np.float32)
        if block_len < horizon:
            pad = np.repeat(future_clim[-1:, :], horizon - block_len, axis=0)
            future_clim = np.concatenate([future_clim, pad], axis=0)
        x_future = ((future_clim - climate_mu) / climate_sd)[None, :, :]
        h0 = float(head[t - 1])
        h0_norm = torch.tensor([(h0 - norm["head_mu"]) / norm["head_sd"]], dtype=torch.float32)

        with torch.no_grad():
            pred_head_norm = model(torch.from_numpy(x_past), torch.from_numpy(x_future), h0_norm)
            pred_block = denorm_head(pred_head_norm, norm["head_mu"], norm["head_sd"]).cpu().numpy()[0, :block_len]

        pred_all.append(pred_block.astype(np.float32))
        obs_all.append(head[t : t + block_len].astype(np.float32))
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


def save_delta_run(output_dir: Path, series: LadderSeries, outputs: dict, meta: dict) -> None:
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

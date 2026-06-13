from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .direct_delta_lead import _direct_metrics, _ode_penalty_seq, _weighted_mse, _ws2_penalty_seq
from .neural_ladder import lag_diagnostic, peak_timing_diagnostic


class ForecastNARXLeadDelta(nn.Module):
    """Plain NARX-style MLP for one-step normalized delta prediction.

    The model uses the same `(x_past, x_future) -> delta_norm` contract as the
    GRU and PatchTST delta models, so recursive 7-day evaluation stays identical.
    """

    def __init__(
        self,
        n_past_feat: int,
        n_future_feat: int,
        horizon: int,
        window: int,
        hidden: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.window = int(window)
        self.horizon = int(horizon)
        n_in = int(n_past_feat) * self.window + int(n_future_feat) * self.horizon
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_past.reshape(x_past.size(0), -1), x_future.reshape(x_future.size(0), -1)], dim=-1)
        return self.net(z).squeeze(-1)


class ForecastLSTMLeadDelta(nn.Module):
    """Plain LSTM one-step normalized-delta forecaster."""

    def __init__(
        self,
        n_past_feat: int,
        n_future_feat: int,
        horizon: int,
        hidden: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.lstm = nn.LSTM(n_past_feat, hidden, batch_first=True)
        self.future_proj = nn.Sequential(
            nn.Linear(n_future_feat * self.horizon, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x_past)
        h_last = h[:, -1, :]
        fut = self.future_proj(x_future.reshape(x_future.size(0), -1))
        return self.head(torch.cat([h_last, fut], dim=-1)).squeeze(-1)


def variant_regularizer_kind(variant: str) -> str:
    aliases = {
        "narx": "plain",
        "lstm": "plain",
        "lstm_ode": "ode",
        "lstm_ws2": "ws2",
    }
    if variant not in aliases:
        raise ValueError(f"Unsupported plain architecture variant: {variant}")
    return aliases[variant]


def _build_plain_architecture(
    variant: str,
    train: dict[str, np.ndarray],
    hidden: int,
) -> nn.Module:
    n_past_feat = train["x_past"].shape[-1]
    n_future_feat = train["x_future"].shape[-1]
    horizon = train["x_future"].shape[1]
    window = train["x_past"].shape[1]
    if variant == "narx":
        return ForecastNARXLeadDelta(n_past_feat, n_future_feat, horizon, window, hidden=hidden)
    if variant in {"lstm", "lstm_ode", "lstm_ws2"}:
        return ForecastLSTMLeadDelta(n_past_feat, n_future_feat, horizon, hidden=hidden)
    raise ValueError(f"Unsupported plain architecture: {variant}")


def train_plain_architecture_delta_variant(
    split_data: dict[str, dict[str, np.ndarray]],
    variant: str,
    seed: int = 42,
    epochs: int = 80,
    patience: int = 15,
    hidden: int = 64,
    lr: float = 1.0e-3,
    lambda_penalty: float = 0.0,
    event_weight_scale: float = 0.0,
) -> tuple[nn.Module, dict[str, dict], dict]:
    """Train a plain NARX-DNN or LSTM under the locked delta protocol."""

    if variant not in {"narx", "lstm", "lstm_ode", "lstm_ws2"}:
        raise ValueError(f"Unsupported plain architecture: {variant}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    kind = variant_regularizer_kind(variant)

    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]
    norm = split_data["norm"]

    model = _build_plain_architecture(variant, train=train, hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4 if variant == "narx" else 0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1.0e-5)
    raw_gamma_r = raw_gamma_d = raw_h_ref = None
    opt_phys = None
    if kind == "ode":
        raw_gamma_r = torch.tensor(-4.0, requires_grad=True)
        raw_gamma_d = torch.tensor(-1.0, requires_grad=True)
        raw_h_ref = torch.tensor(float(train["h_anchor"].mean()), requires_grad=True)
        opt_phys = torch.optim.Adam([raw_gamma_r, raw_gamma_d, raw_h_ref], lr=5.0e-2)

    tt = lambda x: torch.from_numpy(x)
    xpt = tt(train["x_past"])
    xft = tt(train["x_future"])
    ydt = tt(train["y_delta_norm"])
    yhp = tt(train["y_head"])
    hat = tt(train["h_anchor"])
    rsum_t = tt(train["rain_sum"])
    wgt = 1.0 + event_weight_scale * (tt(train["event_weight"]) - 1.0)

    xpv = tt(val["x_past"])
    xfv = tt(val["x_future"])
    yhv = tt(val["y_head"])
    hav = tt(val["h_anchor"])

    best_val = float("inf")
    best_state = None
    best_phys = None
    no_imp = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        if opt_phys is not None:
            opt_phys.zero_grad()
        pred_delta_norm = model(xpt, xft)
        pred_delta_phys = pred_delta_norm * norm["delta_sd"] + norm["delta_mu"]
        pred_head = hat + pred_delta_phys
        loss = _weighted_mse(pred_delta_norm, ydt, wgt)
        loss = loss + 0.25 * torch.mean((pred_head - yhp) ** 2)
        if kind == "ws2":
            loss = loss + lambda_penalty * _ws2_penalty_seq(pred_head)
        elif kind == "ode":
            gamma_r = 1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)
            gamma_d = 1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)
            loss = loss + lambda_penalty * _ode_penalty_seq(pred_head, rsum_t, gamma_r, gamma_d, raw_h_ref)
        loss.backward()
        opt.step()
        if opt_phys is not None:
            opt_phys.step()

        model.eval()
        with torch.no_grad():
            pred_val_norm = model(xpv, xfv)
            pred_val_phys = pred_val_norm * norm["delta_sd"] + norm["delta_mu"]
            val_loss = torch.mean((pred_val_phys - (yhv - hav)) ** 2).item()
        sched.step(val_loss)

        if val_loss < best_val - 1.0e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if kind == "ode":
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
        raise RuntimeError(f"{variant} delta training failed to produce a best state.")
    model.load_state_dict(best_state)
    model.eval()

    outputs: dict[str, dict] = {}
    for name, block in [("train", train), ("val", val), ("test", test)]:
        with torch.no_grad():
            pred_delta_norm = model(tt(block["x_past"]), tt(block["x_future"]))
        pred_delta_phys = (pred_delta_norm * norm["delta_sd"] + norm["delta_mu"]).cpu().numpy()
        pred_head = block["h_anchor"] + pred_delta_phys
        metrics = _direct_metrics(pred_head, block["y_head"])
        metrics.update(lag_diagnostic(pred_head, block["y_head"], max_lag=max(14, block["x_future"].shape[1])))
        metrics.update(peak_timing_diagnostic(pred_head, block["y_head"]))
        outputs[name] = {
            "pred_head": pred_head.astype(np.float32),
            "obs_head": block["y_head"].astype(np.float32),
            "pred_delta": pred_delta_phys.astype(np.float32),
            "obs_delta": block["y_delta"].astype(np.float32),
            "target_dates": block["target_dates"].astype("datetime64[D]").astype(str),
            "metrics": metrics,
        }

    meta = {
        "variant": f"{variant}_delta_{kind}",
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "hidden": hidden,
        "lr": lr,
        "lambda_penalty": lambda_penalty,
        "regularizer": kind,
        "event_weight_scale": event_weight_scale,
        "best_val_loss": best_val,
        "target_mode": "one_step_delta",
    }
    if best_phys is not None:
        meta["physics_params"] = best_phys
    return model, outputs, meta

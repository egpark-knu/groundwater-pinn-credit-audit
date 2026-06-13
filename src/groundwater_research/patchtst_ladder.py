from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .direct_delta_lead import (
    _direct_metrics,
    _ode_penalty_seq,
    _weighted_mse,
    _ws2_penalty_seq,
)
from .neural_ladder import lag_diagnostic, peak_timing_diagnostic


class ForecastPatchTSTLeadDelta(nn.Module):
    """Lightweight PatchTST-style one-step delta forecaster.

    The model keeps the same `(x_past, x_future) -> normalized delta` interface as
    the locked recursive evaluator. Future weather is projected separately so the
    rollout contract stays identical to the existing delta protocol.
    """

    def __init__(
        self,
        n_past_feat: int,
        n_future_feat: int,
        horizon: int,
        window: int,
        patch_len: int = 7,
        stride: int = 7,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        if patch_len > window:
            raise ValueError(f"patch_len={patch_len} exceeds window={window}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.window = int(window)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.horizon = int(horizon)
        starts = list(range(0, self.window - self.patch_len + 1, self.stride))
        final_start = self.window - self.patch_len
        if starts[-1] != final_start:
            starts.append(final_start)
        self.register_buffer("patch_starts", torch.tensor(starts, dtype=torch.long), persistent=False)
        self.n_patches = len(starts)

        self.patch_proj = nn.Linear(self.patch_len * n_past_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.future_proj = nn.Sequential(
            nn.Linear(n_future_feat * horizon, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_in = d_model * self.n_patches + d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.patch_len, device=x_past.device, dtype=torch.long)
        patch_index = self.patch_starts.to(x_past.device).unsqueeze(1) + offsets.unsqueeze(0)
        patches = x_past[:, patch_index, :].reshape(x_past.size(0), self.n_patches, -1)
        tokens = self.patch_proj(patches) + self.pos
        encoded = self.encoder(tokens).reshape(x_past.size(0), -1)
        future = self.future_proj(x_future.reshape(x_future.size(0), -1))
        return self.head(torch.cat([encoded, future], dim=-1)).squeeze(-1)


def _variant_kind(variant: str) -> str:
    aliases = {
        "patchtst": "plain",
        "plain": "plain",
        "patchtst_ws2": "ws2",
        "ws2": "ws2",
        "patchtst_ode": "ode",
        "ode": "ode",
    }
    if variant not in aliases:
        raise ValueError(f"Unsupported PatchTST variant: {variant}")
    return aliases[variant]


def train_patchtst_delta_variant(
    split_data: dict[str, dict[str, np.ndarray]],
    variant: str,
    seed: int = 42,
    epochs: int = 80,
    patience: int = 15,
    lr: float = 1.0e-3,
    lambda_penalty: float = 1.0,
    event_weight_scale: float = 0.0,
    patch_len: int = 7,
    stride: int = 7,
    d_model: int = 32,
    n_heads: int = 4,
    n_layers: int = 2,
) -> tuple[nn.Module, dict[str, dict], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    kind = _variant_kind(variant)
    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]
    norm = split_data["norm"]

    model = ForecastPatchTSTLeadDelta(
        n_past_feat=train["x_past"].shape[-1],
        n_future_feat=train["x_future"].shape[-1],
        horizon=train["x_future"].shape[1],
        window=train["x_past"].shape[1],
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
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
        raise RuntimeError("PatchTST delta training failed to produce a best state.")
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
        "variant": f"patchtst_delta_{kind}",
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "lr": lr,
        "lambda_penalty": lambda_penalty,
        "event_weight_scale": event_weight_scale,
        "patch_len": patch_len,
        "stride": stride,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "best_val_loss": best_val,
        "target_mode": "one_step_delta",
    }
    if best_phys is not None:
        meta["physics_params"] = best_phys
    return model, outputs, meta

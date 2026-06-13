from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .neural_ladder import BlockSplits, LadderSeries


class ForecastGRUStep(nn.Module):
    def __init__(self, n_step_feat: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_step_feat, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_step: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x_step)
        return self.head(h[:, -1, :]).squeeze(-1)


def compose_recursive_step_window(
    climate: np.ndarray,
    head: np.ndarray,
    target_idx: int,
    window: int,
) -> np.ndarray:
    if target_idx < window:
        raise ValueError(f"target_idx={target_idx} is shorter than window={window}")
    climate_step = climate[target_idx - window + 1 : target_idx + 1]
    prev_head = head[target_idx - window : target_idx]
    if climate_step.shape[0] != window or prev_head.shape[0] != window:
        raise ValueError("Failed to build a full recursive step window.")
    return np.concatenate([climate_step, prev_head[:, None]], axis=1).astype(np.float32)


def build_recursive_head_split(
    series: LadderSeries,
    splits: BlockSplits,
    window: int,
) -> dict[str, dict[str, np.ndarray]]:
    head = series.head_interp.astype(np.float32)
    climate = series.climate.astype(np.float32)

    def target_range(split: slice) -> np.ndarray:
        lo = max(split.start, window)
        hi = split.stop
        targets = np.arange(lo, hi, dtype=int)
        if len(targets) == 0:
            raise ValueError(f"Split too short for window={window}: {split}")
        return targets

    train_targets = target_range(splits.train)
    train_x_raw = np.stack([compose_recursive_step_window(climate, head, t, window) for t in train_targets]).astype(np.float32)
    step_mu = train_x_raw.reshape(-1, train_x_raw.shape[-1]).mean(axis=0)
    step_sd = train_x_raw.reshape(-1, train_x_raw.shape[-1]).std(axis=0) + 1.0e-9
    head_mu = float(head[train_targets].mean())
    head_sd = float(head[train_targets].std() + 1.0e-9)

    def build(split: slice) -> dict[str, np.ndarray]:
        targets = target_range(split)
        x_raw = np.stack([compose_recursive_step_window(climate, head, t, window) for t in targets]).astype(np.float32)
        return {
            "x_step": ((x_raw - step_mu) / step_sd).astype(np.float32),
            "y_head": head[targets].astype(np.float32),
            "y_head_norm": ((head[targets] - head_mu) / head_sd).astype(np.float32),
            "rain_target": series.rain_mm[targets].astype(np.float32),
            "h_prev": head[targets - 1].astype(np.float32),
            "target_dates": series.dates[targets].astype("datetime64[D]"),
        }

    return {
        "train": build(splits.train),
        "val": build(splits.val),
        "test": build(splits.test),
        "norm": {
            "step_mu": step_mu.astype(np.float32),
            "step_sd": step_sd.astype(np.float32),
            "head_mu": head_mu,
            "head_sd": head_sd,
        },
    }


def denorm_head(y_norm: torch.Tensor, head_mu: float, head_sd: float) -> torch.Tensor:
    return y_norm * head_sd + head_mu


def ordered_ws1_penalty(pred_head: torch.Tensor) -> torch.Tensor:
    if pred_head.numel() < 2:
        return torch.zeros((), dtype=pred_head.dtype)
    return torch.mean((pred_head[1:] - pred_head[:-1]) ** 2)


def ordered_ws2_penalty(pred_head: torch.Tensor) -> torch.Tensor:
    if pred_head.numel() < 3:
        return torch.zeros((), dtype=pred_head.dtype)
    second = pred_head[2:] - 2.0 * pred_head[1:-1] + pred_head[:-2]
    return torch.mean(second**2)


def ordered_ode_penalty(
    pred_head: torch.Tensor,
    rain_target: torch.Tensor,
    gamma_r: torch.Tensor,
    gamma_d: torch.Tensor,
    h_ref: torch.Tensor,
) -> torch.Tensor:
    if pred_head.numel() < 2:
        return torch.zeros((), dtype=pred_head.dtype)
    dh = pred_head[1:] - pred_head[:-1]
    rhs = gamma_r * rain_target[1:] * 1.0e-3 - gamma_d * (pred_head[:-1] - h_ref)
    return torch.mean((dh - rhs) ** 2)


def train_recursive_head_variant(
    split_data: dict[str, dict[str, np.ndarray]],
    variant: str,
    seed: int = 42,
    epochs: int = 150,
    patience: int = 25,
    hidden: int = 64,
    lr: float = 1.0e-3,
    lambda_penalty: float = 1.0,
) -> tuple[nn.Module, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]
    norm = split_data["norm"]

    model = ForecastGRUStep(n_step_feat=train["x_step"].shape[-1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1.0e-5)

    raw_gamma_r = raw_gamma_d = raw_h_ref = None
    opt_phys = None
    if variant == "ode":
        raw_gamma_r = torch.tensor(-4.0, requires_grad=True)
        raw_gamma_d = torch.tensor(-1.0, requires_grad=True)
        raw_h_ref = torch.tensor(float(train["y_head"].mean()), requires_grad=True)
        opt_phys = torch.optim.Adam([raw_gamma_r, raw_gamma_d, raw_h_ref], lr=5.0e-2)

    tt = lambda x: torch.from_numpy(x)
    xtr = tt(train["x_step"])
    ytr = tt(train["y_head_norm"])
    ytr_phys = tt(train["y_head"])
    htr = tt(train["h_prev"])
    rtr = tt(train["rain_target"])

    xva = tt(val["x_step"])
    yva = tt(val["y_head_norm"])

    best_val = float("inf")
    best_state = None
    best_phys = None
    no_imp = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        if opt_phys is not None:
            opt_phys.zero_grad()

        pred_norm_full = model(xtr)
        pred_phys_full = denorm_head(pred_norm_full, norm["head_mu"], norm["head_sd"])
        loss = torch.mean((pred_norm_full - ytr) ** 2)
        if variant == "ws1":
            loss = loss + lambda_penalty * ordered_ws1_penalty(pred_phys_full)
        elif variant == "ws2":
            loss = loss + lambda_penalty * ordered_ws2_penalty(pred_phys_full)
        elif variant == "ode":
            gamma_r = 1.0e-4 + 0.5 * torch.sigmoid(raw_gamma_r)
            gamma_d = 1.0e-4 + 0.2 * torch.sigmoid(raw_gamma_d)
            loss = loss + lambda_penalty * ordered_ode_penalty(pred_phys_full, rtr, gamma_r, gamma_d, raw_h_ref)
        loss.backward()
        opt.step()
        if opt_phys is not None:
            opt_phys.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(xva)
            val_loss = torch.mean((pred_val - yva) ** 2).item()
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
        raise RuntimeError("Recursive head training failed to produce a best state.")
    model.load_state_dict(best_state)
    model.eval()

    outputs = {}
    for name, block in [("train", train), ("val", val), ("test", test)]:
        with torch.no_grad():
            pred_norm = model(tt(block["x_step"]))
        pred_phys = denorm_head(pred_norm, norm["head_mu"], norm["head_sd"]).cpu().numpy()
        outputs[name] = {
            "pred_head": pred_phys.astype(np.float32),
            "obs_head": block["y_head"].astype(np.float32),
            "target_dates": block["target_dates"].astype("datetime64[D]").astype(str),
        }

    meta = {
        "variant": f"recursive_head_{variant}",
        "alignment": "legacy_recursive_step",
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "hidden": hidden,
        "lr": lr,
        "lambda_penalty": lambda_penalty,
        "best_val_loss": best_val,
    }
    if best_phys is not None:
        meta["physics_params"] = best_phys
    return model, outputs, meta

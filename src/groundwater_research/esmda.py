from __future__ import annotations

import numpy as np


def initialize_gaussian_ensemble(
    mean: np.ndarray,
    std: np.ndarray,
    n_ensemble: int,
    rng: np.random.Generator,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a low-dimensional Gaussian ensemble."""
    theta = mean[None, :] + std[None, :] * rng.standard_normal((n_ensemble, len(mean)))
    if lower is not None or upper is not None:
        theta = np.clip(theta, lower, upper)
    return theta


def esmda_update(
    theta: np.ndarray,
    predicted: np.ndarray,
    observed: np.ndarray,
    alpha: float,
    obs_error_std: float,
    rng: np.random.Generator,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    """
    Low-dimensional ES-MDA update.

    Shapes:
    - theta: (n_ensemble, n_params)
    - predicted: (n_ensemble, n_obs)
    - observed: (n_obs,)
    """
    n_ensemble = theta.shape[0]
    if n_ensemble < 2:
        raise ValueError("Need at least two ensemble members for ES-MDA.")

    theta_mean = theta.mean(axis=0, keepdims=True)
    pred_mean = predicted.mean(axis=0, keepdims=True)
    a_theta = theta - theta_mean
    a_pred = predicted - pred_mean

    c_md = (a_theta.T @ a_pred) / (n_ensemble - 1)
    c_dd = (a_pred.T @ a_pred) / (n_ensemble - 1)
    r = (alpha * obs_error_std**2) * np.eye(predicted.shape[1])
    kalman = c_md @ np.linalg.inv(c_dd + r)

    d_pert = observed[None, :] + np.sqrt(alpha) * obs_error_std * rng.standard_normal(
        size=predicted.shape
    )
    updated = theta + (d_pert - predicted) @ kalman.T

    if lower is not None or upper is not None:
        updated = np.clip(updated, lower, upper)
    return updated


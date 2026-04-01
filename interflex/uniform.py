"""Uniform confidence interval computation.

Exact translation of ``uniform.R``.
"""

from __future__ import annotations

import warnings

import numpy as np


def calculate_uniform_quantiles(
    theta_matrix: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float, float]:
    """Compute uniform confidence bands via bisection on quantile level.

    Parameters
    ----------
    theta_matrix : ndarray (k, N)
        k evaluation points, N bootstrap/simulation draws.
    alpha : float
        Significance level (default 0.05 for 95% bands).

    Returns
    -------
    Q_j : ndarray (k, 2)
        Lower and upper uniform quantile bounds per evaluation point.
    zeta_hat : float
        Calibrated zeta level.
    coverage : float
        Actual joint coverage fraction.
    """
    # Drop columns with any NaN
    cols_with_na = np.any(np.isnan(theta_matrix), axis=0)
    theta_matrix = theta_matrix[:, ~cols_with_na]

    k = theta_matrix.shape[0]
    N = theta_matrix.shape[1]

    if N == 0:
        warnings.warn("No valid bootstrap/simulation draws for uniform CI.")
        return np.zeros((k, 2)), alpha / (2 * k), 0.0

    def check_condition(zeta: float) -> float:
        """Compute fraction of draws where ALL rows fall within quantile bands."""
        Q_j = np.column_stack([
            np.quantile(theta_matrix, zeta, axis=1),
            np.quantile(theta_matrix, 1 - zeta, axis=1),
        ])  # (k, 2)
        # Check if each draw is within bounds for ALL rows
        within_lower = theta_matrix >= Q_j[:, 0:1]  # (k, N) broadcasting
        within_upper = theta_matrix <= Q_j[:, 1:2]
        all_within = np.all(within_lower & within_upper, axis=0)  # (N,)
        return np.mean(all_within)

    zeta_lower = alpha / (2 * k)
    zeta_upper = alpha / 2

    while zeta_upper - zeta_lower > 1e-6:
        zeta_mid = (zeta_lower + zeta_upper) / 2
        if check_condition(zeta_mid) < 1 - alpha:
            zeta_upper = zeta_mid
        else:
            zeta_lower = zeta_mid

    zeta_hat = zeta_lower
    coverage = check_condition(zeta_hat)

    if np.isclose(zeta_hat, alpha / (2 * k)):
        warnings.warn(
            "Insufficient bootstrap samples for Bootstrapped Uniform "
            "Confidence Interval. Using Bonferroni CI by default."
        )

    Q_j = np.column_stack([
        np.quantile(theta_matrix, zeta_hat, axis=1),
        np.quantile(theta_matrix, 1 - zeta_hat, axis=1),
    ])

    return Q_j, zeta_hat, coverage


def calculate_delta_uniform_ci(
    Sigma_hat: np.ndarray,
    alpha: float = 0.05,
    N: int = 2000,
) -> float:
    """Compute critical value *q* for delta-method uniform confidence interval.

    Parameters
    ----------
    Sigma_hat : ndarray (k, k)
        TE/ME covariance matrix across evaluation points.
    alpha : float
        Significance level.
    N : int
        Number of MVN draws.

    Returns
    -------
    q : float
        Critical value such that P(max|Z_i / sqrt(Sigma_ii)| <= q) = 1 - alpha.
    """
    k = Sigma_hat.shape[0]
    rng = np.random.default_rng()

    # Ensure Sigma_hat is positive semi-definite
    # Small regularization if needed
    try:
        V = rng.multivariate_normal(np.zeros(k), Sigma_hat, size=N)
    except np.linalg.LinAlgError:
        # Add small ridge
        Sigma_reg = Sigma_hat + np.eye(k) * 1e-10
        V = rng.multivariate_normal(np.zeros(k), Sigma_reg, size=N)

    diag_vals = np.diag(Sigma_hat)
    # Guard against zero variance
    safe_diag = np.where(diag_vals > 0, diag_vals, 1e-20)
    Sigma_inv_sqrt = 1.0 / np.sqrt(safe_diag)

    max_vals = np.max(np.abs(V * Sigma_inv_sqrt[None, :]), axis=1)  # (N,)
    q = float(np.percentile(max_vals, 100 * (1 - alpha)))
    return q

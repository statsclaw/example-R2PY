"""Frisch-Waugh-Lovell demeaning and IV-FWL for fixed effects models.

Exact translation of ``fastplm.cpp`` and ``iv_fastplm.cpp``.
"""

from __future__ import annotations

import numpy as np


def fwl_demean(
    data: np.ndarray,
    FE: np.ndarray,
    weight: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Frisch-Waugh-Lovell iterative demeaning.

    Parameters
    ----------
    data : ndarray (n, k)
        Column 0 is Y; columns 1..k-1 are covariates.
    FE : ndarray (n, m)
        Fixed-effect group indicators (integer-coded).
    weight : ndarray (n,)
        Observation weights.
    tol : float
        Convergence tolerance (sum of absolute differences).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    coefficients : ndarray (p,) or empty
        OLS coefficients on demeaned data.
    residuals : ndarray (n,)
        Residuals on unweighted scale.
    niter : int
        Number of iterations performed.
    mu : float
        Grand mean.
    """
    n, k = data.shape
    p = k - 1  # number of covariates
    m = FE.shape[1] if FE.ndim == 2 else 1
    if FE.ndim == 1:
        FE = FE[:, None]

    data = data.copy().astype(np.float64)
    data_bak = data.copy()
    data_old = np.zeros_like(data)
    diff = 100.0
    niter = 0

    # Pre-compute group indices for each FE column
    fe_groups: list[list[tuple[np.ndarray, ...]]] = []
    for col in range(m):
        unique_vals = np.unique(FE[:, col])
        groups = []
        for g in unique_vals:
            mask = FE[:, col] == g
            groups.append((mask,))
        fe_groups.append(groups)

    while diff > tol and niter < max_iter:
        for col in range(m):
            # Weighted data
            data_wei = data * weight[:, None]
            for mask_tuple in fe_groups[col]:
                mask = mask_tuple[0]
                sum_w = weight[mask].sum()
                if sum_w > 0:
                    group_mean = data_wei[mask].sum(axis=0) / sum_w
                else:
                    group_mean = np.zeros(k)
                data[mask] -= group_mean

        diff = np.abs(data - data_old).sum()
        data_old = data.copy()
        niter += 1

    # Apply sqrt(weight) for WLS
    sqrt_w = np.sqrt(weight)
    y = data[:, 0] * sqrt_w
    coefficients = np.array([], dtype=np.float64)
    coeff_full = np.zeros(p)
    nan_mask = np.zeros(p, dtype=bool)

    if p > 0:
        X = data[:, 1:] * sqrt_w[:, None]

        # Drop zero-variance columns (set coef to NaN) — matches C++ behavior
        keep_cols = []
        for i in range(p):
            if np.unique(X[:, i]).shape[0] <= 1:
                coeff_full[i] = np.nan
                nan_mask[i] = True
            else:
                keep_cols.append(i)

        if len(keep_cols) > 0:
            X_reduced = X[:, keep_cols]
            coef_reduced, _, _, _ = np.linalg.lstsq(X_reduced, y, rcond=None)
            # Fill back into full vector
            idx = 0
            for i in range(p):
                if not nan_mask[i]:
                    coeff_full[i] = coef_reduced[idx]
                    idx += 1
            residuals = (y - X[:, keep_cols] @ coef_reduced) / sqrt_w
        else:
            residuals = y / sqrt_w

        coefficients = coeff_full
    else:
        residuals = y / sqrt_w

    # Grand mean
    y_bak = data_bak[:, 0]
    mu = np.sum(y_bak * weight) / np.sum(weight)
    if p > 0:
        coef_for_mu = coefficients.copy()
        coef_for_mu[np.isnan(coef_for_mu)] = 0
        X_bak = data_bak[:, 1:]
        X_wmean = np.sum(X_bak * weight[:, None], axis=0) / np.sum(weight)
        mu = mu - X_wmean @ coef_for_mu

    return coefficients, residuals, niter, mu


def iv_fwl(
    Y: np.ndarray,
    X_endog: np.ndarray,
    Z_exog: np.ndarray,
    IV: np.ndarray,
    FE: np.ndarray,
    weight: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Instrumental variables with FWL demeaning (2SLS after demeaning).

    Exact translation of ``iv_fastplm.cpp``.

    Parameters
    ----------
    Y : ndarray (n,) or (n, 1)
        Outcome variable.
    X_endog : ndarray (n, k_X)
        Endogenous regressors.
    Z_exog : ndarray (n, k_Z) or None
        Exogenous regressors (non-endogenous covariates).
    IV : ndarray (n, k_IV)
        Instruments.
    FE : ndarray (n, m)
        Fixed-effect group indicators (integer-coded).
    weight : ndarray (n,)
        Observation weights.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    coefficients : ndarray (k_X + k_Z,)
        2SLS coefficients.
    residuals : ndarray (n,)
        Residuals on unweighted scale.
    niter : int
        Number of FWL iterations.
    mu : float
        Grand mean.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    X_endog = np.atleast_2d(X_endog)
    if X_endog.shape[0] == 1 and X_endog.shape[1] == len(Y):
        X_endog = X_endog.T
    IV = np.atleast_2d(IV)
    if IV.shape[0] == 1 and IV.shape[1] == len(Y):
        IV = IV.T
    if Z_exog is not None:
        Z_exog = np.atleast_2d(Z_exog)
        if Z_exog.shape[0] == 1 and Z_exog.shape[1] == len(Y):
            Z_exog = Z_exog.T

    n = len(Y)

    # Concatenate [Y, X_endog, Z_exog, IV] into data matrix
    parts = [Y[:, None], X_endog]
    k_endog = X_endog.shape[1]
    k_exog = 0
    if Z_exog is not None and Z_exog.shape[1] > 0:
        parts.append(Z_exog)
        k_exog = Z_exog.shape[1]
    k_iv = IV.shape[1]
    parts.append(IV)
    data_full = np.hstack(parts)

    # Apply FWL demeaning (same as fwl_demean but we want all columns demeaned)
    data_bak = data_full.copy()
    _coef_dm, _resid_dm, niter, _mu_dm = fwl_demean(
        data_full, FE, weight, tol=tol, max_iter=max_iter
    )
    # After demeaning, re-read from data_full (modified in-place by fwl_demean??)
    # Actually fwl_demean copies data, so we need to re-demean manually.
    # Let's do the demeaning inline to match the C++ exactly.

    data_dm = data_full.copy()
    FE_arr = np.atleast_2d(FE) if FE.ndim == 1 else FE
    m_fe = FE_arr.shape[1]
    data_old = np.zeros_like(data_dm)
    diff_val = 100.0
    niter = 0
    while diff_val > tol and niter < max_iter:
        for col in range(m_fe):
            data_wei = data_dm * weight[:, None]
            unique_vals = np.unique(FE_arr[:, col])
            for g in unique_vals:
                mask = FE_arr[:, col] == g
                sum_w = weight[mask].sum()
                if sum_w > 0:
                    group_mean = data_wei[mask].sum(axis=0) / sum_w
                else:
                    group_mean = np.zeros(data_dm.shape[1])
                data_dm[mask] -= group_mean
        diff_val = np.abs(data_dm - data_old).sum()
        data_old = data_dm.copy()
        niter += 1

    # Apply sqrt(weight)
    sqrt_w = np.sqrt(weight)
    data_w = data_dm * sqrt_w[:, None]

    # Recover components
    col = 0
    Y_dm = data_w[:, col]
    col += 1
    X_endog_dm = data_w[:, col : col + k_endog]
    col += k_endog
    if k_exog > 0:
        Z_exog_dm = data_w[:, col : col + k_exog]
        col += k_exog
    else:
        Z_exog_dm = np.empty((n, 0))
    IV_dm = data_w[:, col : col + k_iv]

    # 2SLS
    Z_full = np.hstack([Z_exog_dm, IV_dm]) if k_exog > 0 else IV_dm
    X_full = np.hstack([X_endog_dm, Z_exog_dm]) if k_exog > 0 else X_endog_dm

    inv_ZZ = np.linalg.inv(Z_full.T @ Z_full)
    PzX = Z_full @ inv_ZZ @ (Z_full.T @ X_full)
    PzY = Z_full @ inv_ZZ @ (Z_full.T @ Y_dm)
    coefficients, _, _, _ = np.linalg.lstsq(PzX, PzY, rcond=None)
    coefficients = coefficients.ravel()

    residuals_w = Y_dm - X_full @ coefficients
    residuals = residuals_w / sqrt_w

    # Grand mean
    Y_bak = data_bak[:, 0]
    mu = np.sum(Y_bak * weight) / np.sum(weight)
    p_total = k_endog + k_exog
    if p_total > 0:
        X_bak = data_bak[:, 1 : 1 + p_total]
        coef_for_mu = coefficients.copy()
        coef_for_mu[np.isnan(coef_for_mu)] = 0
        X_wmean = np.sum(X_bak * weight[:, None], axis=0) / np.sum(weight)
        mu = mu - X_wmean @ coef_for_mu

    return coefficients, residuals, niter, mu

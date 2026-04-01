"""Variance-covariance estimators: cluster-robust, HC1, and PCSE."""

from __future__ import annotations

import warnings

import numpy as np


def vcov_cluster(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Cluster-robust variance-covariance estimator.

    Direct translation of ``vcluster.R``.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix (with intercept if applicable).
    residuals : ndarray (n,)
        Model residuals.
    cluster : ndarray (n,)
        Cluster labels.
    rank : int
        Model rank *K* (number of estimated coefficients).

    Returns
    -------
    ndarray (p, p)
        Cluster-robust variance-covariance matrix.
    """
    n = X.shape[0]
    p = X.shape[1]
    unique_clusters = np.unique(cluster)
    M = len(unique_clusters)
    K = rank

    # Degrees-of-freedom correction  (matches R sandwich)
    dfc = (M / (M - 1)) * ((n - 1) / (n - K))

    # Score matrix: S_i = X_i * e_i  (element-wise, each row)
    score = X * residuals[:, None]  # (n, p)

    # Aggregate scores by cluster
    # Build (M, p) matrix of cluster-summed scores
    u = np.zeros((M, p))
    # Use a mapping for speed
    cluster_map = {c: idx for idx, c in enumerate(unique_clusters)}
    cluster_idx = np.array([cluster_map[c] for c in cluster])
    np.add.at(u, cluster_idx, score)

    # Meat: u' u / n
    meat = u.T @ u / n  # (p, p)

    # Bread: inv(X'X / n)
    XtX_n = X.T @ X / n  # (p, p)
    bread = np.linalg.inv(XtX_n)

    vcov = dfc * bread @ meat @ bread
    return vcov


def robust_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    rank: int,
) -> np.ndarray:
    """HC1 heteroskedasticity-consistent variance-covariance estimator.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix.
    residuals : ndarray (n,)
        Model residuals.
    rank : int
        Model rank *K*.

    Returns
    -------
    ndarray (p, p)
    """
    n = X.shape[0]
    K = rank

    XtX_inv = np.linalg.inv(X.T @ X)

    # HC1 correction factor
    hc1_factor = n / (n - K)

    # Sandwich: (X'X)^-1  X' diag(e^2) X  (X'X)^-1
    Xe2 = X * (residuals ** 2)[:, None]  # (n, p)
    meat = X.T @ Xe2  # (p, p)  — equivalent to X' diag(e^2) X

    vcov = hc1_factor * XtX_inv @ meat @ XtX_inv
    return vcov


def pcse_vcov(
    model_residuals: np.ndarray,
    X: np.ndarray,
    group_n: np.ndarray,
    group_t: np.ndarray,
    pairwise: bool = True,
) -> np.ndarray:
    """Panel-corrected standard errors (Beck & Katz 1995).

    Parameters
    ----------
    model_residuals : ndarray (n,)
        Model residuals.
    X : ndarray (n, p)
        Design matrix.
    group_n : ndarray (n,)
        Panel unit identifiers.
    group_t : ndarray (n,)
        Time period identifiers.
    pairwise : bool
        If True, use pairwise-complete observations for Omega estimation.

    Returns
    -------
    ndarray (p, p)
    """
    units = np.unique(group_n)
    times = np.unique(group_t)
    N_units = len(units)
    T_total = len(times)

    # Map units / times to indices
    unit_map = {u: i for i, u in enumerate(units)}
    time_map = {t: i for i, t in enumerate(times)}

    # Build residual panel matrix (T x N), NaN where missing
    resid_panel = np.full((T_total, N_units), np.nan)
    unit_idx = np.array([unit_map[u] for u in group_n])
    time_idx = np.array([time_map[t] for t in group_t])
    resid_panel[time_idx, unit_idx] = model_residuals

    # Estimate cross-unit error covariance Omega (N x N)
    Omega = np.zeros((N_units, N_units))
    for i in range(N_units):
        for j in range(i, N_units):
            e_i = resid_panel[:, i]
            e_j = resid_panel[:, j]
            if pairwise:
                valid = ~np.isnan(e_i) & ~np.isnan(e_j)
                T_ij = np.sum(valid)
                if T_ij > 0:
                    Omega[i, j] = np.sum(e_i[valid] * e_j[valid]) / T_ij
                else:
                    Omega[i, j] = 0.0
            else:
                # Complete case: use all time periods
                valid = ~np.isnan(e_i) & ~np.isnan(e_j)
                Omega[i, j] = np.sum(e_i[valid] * e_j[valid]) / T_total
            Omega[j, i] = Omega[i, j]

    # Construct Omega_full (n x n) via Kronecker-like structure
    # For each pair of observations, if they share the same time period,
    # their error covariance is Omega[unit_i, unit_j]
    # Efficient implementation: X' (Omega_kron) X
    # = sum over all time periods t: X_t' Omega X_t
    # where X_t is the design matrix for observations at time t

    XtX_inv = np.linalg.inv(X.T @ X)
    p = X.shape[1]
    meat = np.zeros((p, p))

    for t_val in range(T_total):
        # Find observations at this time period
        mask_t = time_idx == t_val
        if not np.any(mask_t):
            continue
        X_t = X[mask_t]  # (n_t, p)
        units_t = unit_idx[mask_t]  # unit indices for these obs

        n_t = X_t.shape[0]
        # Build local Omega for this time slice
        Omega_t = np.zeros((n_t, n_t))
        for ii in range(n_t):
            for jj in range(n_t):
                Omega_t[ii, jj] = Omega[units_t[ii], units_t[jj]]

        meat += X_t.T @ Omega_t @ X_t

    vcov = XtX_inv @ meat @ XtX_inv
    return vcov

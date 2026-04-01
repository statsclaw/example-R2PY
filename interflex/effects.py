"""Treatment effect and marginal effect computation.

Contains ``gen_general_te()`` and ``gen_ate()`` — direct translation of the
corresponding inner functions in ``linear.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from ._typing import Method, TreatType


# ---------------------------------------------------------------------------
# Link-function helpers
# ---------------------------------------------------------------------------

def _apply_link(method: Method, link: np.ndarray) -> np.ndarray:
    """Apply the inverse-link function to *link* values."""
    if method == "linear":
        return link
    if method == "logit":
        return expit(link)
    if method == "probit":
        return norm.cdf(link)
    if method in ("poisson", "nbinom"):
        return np.exp(link)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# gen_general_te
# ---------------------------------------------------------------------------

def gen_general_te(
    model_coef: dict[str, float],
    X_eval: np.ndarray,
    X: str,
    D: str,
    Z: list[str] | None,
    Z_ref: np.ndarray | dict[str, float] | None,
    Z_X: list[str] | None,
    full_moderate: bool,
    method: Method,
    treat_type: TreatType,
    use_fe: bool,
    base: str | None,
    D_sample: dict | None = None,
    char: str | None = None,
    D_ref: float | None = None,
    diff_values: np.ndarray | None = None,
    difference_name: list[str] | None = None,
) -> dict[str, Any]:
    """Compute TE/ME (and predictions, link values, differences) at evaluation points.

    Returns
    -------
    dict  with keys depending on treat_type:
        discrete : TE, E_pred, E_base, link_1, link_0, diff_estimate
        continuous : ME, E_pred, link, diff_estimate
    """
    neval = len(X_eval)
    _coef = model_coef  # shorthand

    def _get(name: str) -> float:
        return _coef.get(name, 0.0)

    # ------------------------------------------------------------------
    # Inner: gen_te (no FE)
    # ------------------------------------------------------------------
    def gen_te(coef: dict[str, float], xvals: np.ndarray) -> dict:
        c = coef
        def _c(name: str) -> float:
            return c.get(name, 0.0)

        if treat_type == "discrete":
            assert char is not None
            link_1 = (
                _c("(Intercept)")
                + xvals * _c(X)
                + _c(f"D.{char}")
                + xvals * _c(f"DX.{char}")
            )
            link_0 = _c("(Intercept)") + xvals * _c(X)

            if Z is not None:
                for a in Z:
                    zval = _zref(a)
                    link_1 = link_1 + zval * _c(a)
                    link_0 = link_0 + zval * _c(a)
                    if full_moderate:
                        link_1 = link_1 + zval * _c(f"{a}.X") * xvals
                        link_0 = link_0 + zval * _c(f"{a}.X") * xvals

            if method == "linear":
                TE = link_1 - link_0
                E_pred = link_1
                E_base = link_0
            elif method == "logit":
                E_pred = expit(link_1)
                E_base = expit(link_0)
                TE = E_pred - E_base
            elif method == "probit":
                E_pred = norm.cdf(link_1)
                E_base = norm.cdf(link_0)
                TE = E_pred - E_base
            else:  # poisson / nbinom
                E_pred = np.exp(link_1)
                E_base = np.exp(link_0)
                TE = E_pred - E_base

            return {"TE": TE, "E_pred": E_pred, "E_base": E_base,
                    "link_1": link_1, "link_0": link_0}

        else:  # continuous
            assert D_ref is not None
            link = (
                _c("(Intercept)")
                + xvals * _c(X)
                + _c(D) * D_ref
                + _c("DX") * xvals * D_ref
            )
            if Z is not None:
                for a in Z:
                    zval = _zref(a)
                    link = link + zval * _c(a)
                    if full_moderate:
                        link = link + zval * _c(f"{a}.X") * xvals

            linear_me = _c(D) + _c("DX") * xvals
            if method == "linear":
                ME = linear_me
                E_pred = link
            elif method == "logit":
                ME = np.exp(link) / (1 + np.exp(link)) ** 2 * linear_me
                E_pred = expit(link)
            elif method == "probit":
                ME = linear_me * norm.pdf(link)
                E_pred = norm.cdf(link)
            else:  # poisson / nbinom
                ME = np.exp(link) * linear_me
                E_pred = np.exp(link)

            return {"ME": ME, "E_pred": E_pred, "link": link}

    # ------------------------------------------------------------------
    # Inner: gen_te_fe (with FE)
    # ------------------------------------------------------------------
    def gen_te_fe(coef: dict[str, float], xvals: np.ndarray) -> dict:
        c = coef
        def _c(name: str) -> float:
            return c.get(name, 0.0)

        if treat_type == "discrete":
            assert char is not None
            TE = _c(f"D.{char}") + xvals * _c(f"DX.{char}")
            zeros = np.zeros_like(xvals)
            return {"TE": TE, "E_pred": zeros, "E_base": zeros,
                    "link_1": zeros, "link_0": zeros}
        else:
            ME = _c(D) + _c("DX") * xvals
            zeros = np.zeros_like(xvals)
            return {"ME": ME, "E_pred": zeros, "link": zeros}

    # ------------------------------------------------------------------
    # Helper for Z_ref access
    # ------------------------------------------------------------------
    def _zref(name: str) -> float:
        if Z_ref is None:
            return 0.0
        if isinstance(Z_ref, dict):
            return Z_ref.get(name, 0.0)
        # Assume it's array-like aligned with Z list
        if Z is not None:
            try:
                idx = Z.index(name)
                return float(Z_ref[idx]) if idx < len(Z_ref) else 0.0
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    if use_fe:
        output = gen_te_fe(_coef, X_eval)
    else:
        output = gen_te(_coef, X_eval)

    # Differences at diff_values
    diff_estimate = None
    if diff_values is not None and len(diff_values) >= 2:
        if use_fe:
            diff_out = gen_te_fe(_coef, np.asarray(diff_values))
        else:
            diff_out = gen_te(_coef, np.asarray(diff_values))

        target = diff_out.get("TE", diff_out.get("ME"))

        if len(diff_values) == 2:
            diff_estimate = np.array([target[1] - target[0]])
        elif len(diff_values) == 3:
            diff_estimate = np.array([
                target[1] - target[0],
                target[2] - target[1],
                target[2] - target[0],
            ])

    output["diff_estimate"] = diff_estimate
    return output


# ---------------------------------------------------------------------------
# gen_ate / gen_ame
# ---------------------------------------------------------------------------

def gen_ate(
    data: Any,  # pd.DataFrame
    model_coef: dict[str, float],
    model_vcov: np.ndarray | None,
    X: str,
    D: str,
    Z: list[str] | None,
    Z_X: list[str] | None,
    method: Method,
    treat_type: TreatType,
    use_fe: bool,
    full_moderate: bool,
    char: str | None = None,
    delta: bool = False,
    weight_col: str = "WEIGHTS",
    coef_names: list[str] | None = None,
) -> float | dict:
    """Compute ATE (discrete) or AME (continuous), optionally with delta-method SE.

    Parameters
    ----------
    data : DataFrame
        The analysis data (with WEIGHTS column).
    model_coef : dict
        Named coefficient dict.
    model_vcov : ndarray or None
        Full variance-covariance matrix (needed if delta=True).
    char : str or None
        Treatment arm label (discrete only).
    delta : bool
        If True, also return SE via delta method.

    Returns
    -------
    float (if delta=False) or dict with keys ATE/AME and sd (if delta=True).
    """
    import pandas as pd

    _coef = model_coef

    def _c(name: str) -> float:
        return _coef.get(name, 0.0)

    # ------------------------------------------------------------------
    # FE path
    # ------------------------------------------------------------------
    if use_fe:
        return _gen_ate_fe(
            data, model_coef, model_vcov, X, D, Z, Z_X,
            method, treat_type, full_moderate, char, delta, weight_col, coef_names,
        )

    # ------------------------------------------------------------------
    # No-FE path
    # ------------------------------------------------------------------
    if treat_type == "discrete":
        assert char is not None
        sub = data[data[D] == char]
        w = sub[weight_col].values

        x_vals = sub[X].values
        link_1 = (
            _c("(Intercept)") + x_vals * _c(X)
            + _c(f"D.{char}") + x_vals * _c(f"DX.{char}")
        )
        link_0 = _c("(Intercept)") + x_vals * _c(X)

        if Z is not None:
            for a in Z:
                z_vals = sub[a].values
                link_1 = link_1 + z_vals * _c(a)
                link_0 = link_0 + z_vals * _c(a)
                if full_moderate:
                    link_1 = link_1 + z_vals * _c(f"{a}.X") * x_vals
                    link_0 = link_0 + z_vals * _c(f"{a}.X") * x_vals

        if method == "linear":
            TE = link_1 - link_0
        elif method == "logit":
            TE = expit(link_1) - expit(link_0)
        elif method == "probit":
            TE = norm.cdf(link_1) - norm.cdf(link_0)
        else:
            TE = np.exp(link_1) - np.exp(link_0)

        ATE = np.average(TE, weights=w)
        if not delta:
            return float(ATE)

        # Delta method SE
        return _ate_delta_discrete(
            sub, model_coef, model_vcov, X, D, Z, Z_X,
            method, full_moderate, char, w, ATE, coef_names,
        )

    else:  # continuous
        w = data[weight_col].values
        x_vals = data[X].values
        d_vals = data[D].values

        link = (
            _c("(Intercept)") + x_vals * _c(X)
            + _c(D) * d_vals + _c("DX") * x_vals * d_vals
        )
        if Z is not None:
            for a in Z:
                z_vals = data[a].values
                link = link + z_vals * _c(a)
                if full_moderate:
                    link = link + z_vals * _c(f"{a}.X") * x_vals

        linear_me = _c(D) + _c("DX") * x_vals
        if method == "linear":
            ME = linear_me
        elif method == "logit":
            ME = np.exp(link) / (1 + np.exp(link)) ** 2 * linear_me
        elif method == "probit":
            ME = linear_me * norm.pdf(link)
        else:
            ME = np.exp(link) * linear_me

        AME = float(np.average(ME, weights=w))
        if not delta:
            return AME

        return _ame_delta_continuous(
            data, model_coef, model_vcov, X, D, Z, Z_X,
            method, full_moderate, w, AME, coef_names,
        )


# ---------------------------------------------------------------------------
# FE ATE/AME
# ---------------------------------------------------------------------------

def _gen_ate_fe(
    data, model_coef, model_vcov, X, D, Z, Z_X,
    method, treat_type, full_moderate, char, delta, weight_col, coef_names,
):
    _coef = model_coef
    def _c(name: str) -> float:
        return _coef.get(name, 0.0)

    if treat_type == "discrete":
        assert char is not None
        sub = data[data[D] == char]
        w = sub[weight_col].values
        x_vals = sub[X].values
        TE = _c(f"D.{char}") + x_vals * _c(f"DX.{char}")
        ATE = float(np.average(TE, weights=w))
        if not delta:
            return ATE

        # Delta SE for FE discrete
        target_slice = [f"D.{char}", f"DX.{char}"]
        vecs = np.column_stack([np.ones(len(x_vals)), x_vals])  # (n_sub, 2)
        vec_mean = np.average(vecs, axis=0, weights=w)
        idx = [coef_names.index(s) for s in target_slice] if coef_names else []
        if len(idx) == 2 and model_vcov is not None:
            sub_vcov = model_vcov[np.ix_(idx, idx)]
            sd = float(np.sqrt(vec_mean @ sub_vcov @ vec_mean))
        else:
            sd = np.nan
        return {"ATE": ATE, "sd": sd}

    else:  # continuous FE
        w = data[weight_col].values
        x_vals = data[X].values
        ME = _c(D) + _c("DX") * x_vals
        AME = float(np.average(ME, weights=w))
        if not delta:
            return AME

        target_slice = [D, "DX"]
        vecs = np.column_stack([np.ones(len(x_vals)), x_vals])
        vec_mean = np.average(vecs, axis=0, weights=w)
        idx = [coef_names.index(s) for s in target_slice] if coef_names else []
        if len(idx) == 2 and model_vcov is not None:
            sub_vcov = model_vcov[np.ix_(idx, idx)]
            sd = float(np.sqrt(vec_mean @ sub_vcov @ vec_mean))
        else:
            sd = np.nan
        return {"AME": AME, "sd": sd}


# ---------------------------------------------------------------------------
# Delta method helpers (no FE)
# ---------------------------------------------------------------------------

def _build_target_slice_discrete(
    X, Z, Z_X, full_moderate, char,
):
    """Return the ordered list of coefficient names for discrete delta method."""
    names = ["(Intercept)", X, f"D.{char}", f"DX.{char}"]
    if Z is not None:
        names.extend(Z)
        if full_moderate and Z_X is not None:
            names.extend(Z_X)
    return names


def _build_target_slice_continuous(
    X, D, Z, Z_X, full_moderate,
):
    names = ["(Intercept)", X, D, "DX"]
    if Z is not None:
        names.extend(Z)
        if full_moderate and Z_X is not None:
            names.extend(Z_X)
    return names


def _sub_vcov(model_vcov, coef_names, target_slice):
    """Extract sub-matrix of model_vcov for target_slice names."""
    idx = []
    for s in target_slice:
        try:
            idx.append(coef_names.index(s))
        except ValueError:
            idx.append(-1)
    # Build sub-matrix, using 0 for missing
    p = len(target_slice)
    sub = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if idx[i] >= 0 and idx[j] >= 0:
                sub[i, j] = model_vcov[idx[i], idx[j]]
    return sub


def _ate_delta_discrete(
    sub_data, model_coef, model_vcov, X, D, Z, Z_X,
    method, full_moderate, char, w, ATE, coef_names,
):
    """Compute ATE with delta-method SE for discrete no-FE case."""
    _coef = model_coef
    def _c(name: str) -> float:
        return _coef.get(name, 0.0)

    target_slice = _build_target_slice_discrete(X, Z, Z_X, full_moderate, char)
    n_sub = len(sub_data)

    all_vecs = []
    for i in range(n_sub):
        row = sub_data.iloc[i]
        x = float(row[X])
        # Build link values
        link_1 = _c("(Intercept)") + x * _c(X) + _c(f"D.{char}") + x * _c(f"DX.{char}")
        link_0 = _c("(Intercept)") + x * _c(X)

        # Build gradient vectors
        vec_1 = [1.0, x, 1.0, x]
        vec_0 = [1.0, x, 0.0, 0.0]

        if Z is not None:
            z_vals = [float(row[a]) for a in Z]
            for a_idx, a in enumerate(Z):
                z_val = z_vals[a_idx]
                link_1 += z_val * _c(a)
                link_0 += z_val * _c(a)
                vec_1.append(z_val)
                vec_0.append(z_val)
                if full_moderate:
                    link_1 += z_val * _c(f"{a}.X") * x
                    link_0 += z_val * _c(f"{a}.X") * x
            if full_moderate and Z_X is not None:
                for a_idx, a in enumerate(Z):
                    vec_1.append(z_vals[a_idx] * x)
                    vec_0.append(z_vals[a_idx] * x)

        vec_1 = np.array(vec_1)
        vec_0 = np.array(vec_0)

        if method == "logit":
            vec = vec_1 * np.exp(link_1) / (1 + np.exp(link_1)) ** 2 - vec_0 * np.exp(link_0) / (1 + np.exp(link_0)) ** 2
        elif method == "probit":
            vec = vec_1 * norm.pdf(link_1) - vec_0 * norm.pdf(link_0)
        elif method in ("poisson", "nbinom"):
            vec = vec_1 * np.exp(link_1) - vec_0 * np.exp(link_0)
        else:  # linear
            vec = vec_1 - vec_0

        all_vecs.append(vec)

    all_vecs = np.array(all_vecs)  # (n_sub, p)
    vec_mean = np.average(all_vecs, axis=0, weights=w)

    sub = _sub_vcov(model_vcov, coef_names, target_slice)
    sd = float(np.sqrt(vec_mean @ sub @ vec_mean))
    return {"ATE": ATE, "sd": sd}


def _ame_delta_continuous(
    data, model_coef, model_vcov, X, D, Z, Z_X,
    method, full_moderate, w, AME, coef_names,
):
    """Compute AME with delta-method SE for continuous no-FE case."""
    _coef = model_coef
    def _c(name: str) -> float:
        return _coef.get(name, 0.0)

    target_slice = _build_target_slice_continuous(X, D, Z, Z_X, full_moderate)
    n = len(data)

    all_vecs = []
    for i in range(n):
        row = data.iloc[i]
        x = float(row[X])
        d = float(row[D])
        link = _c("(Intercept)") + x * _c(X) + _c(D) * d + _c("DX") * x * d

        vec1 = [1.0, x, d, d * x]
        vec0 = [0.0, 0.0, 1.0, x]

        if Z is not None:
            z_vals = [float(row[a]) for a in Z]
            for a_idx, a in enumerate(Z):
                z_val = z_vals[a_idx]
                link += z_val * _c(a)
                vec1.append(z_val)
                vec0.append(0.0)
                if full_moderate:
                    link += z_val * _c(f"{a}.X") * x
            if full_moderate and Z_X is not None:
                for a_idx, a in enumerate(Z):
                    vec1.append(z_vals[a_idx] * x)
                    vec0.append(0.0)

        vec1 = np.array(vec1)
        vec0 = np.array(vec0)

        if method == "logit":
            vec = (
                -(_c(D) + x * _c("DX"))
                * (np.exp(link) - np.exp(-link))
                / (2 + np.exp(link) + np.exp(-link)) ** 2
                * vec1
                + np.exp(link) / (1 + np.exp(link)) ** 2 * vec0
            )
        elif method == "probit":
            vec = norm.pdf(link) * vec0 - (_c(D) + x * _c("DX")) * link * norm.pdf(link) * vec1
        elif method in ("poisson", "nbinom"):
            vec = (_c(D) + x * _c("DX")) * np.exp(link) * vec1 + np.exp(link) * vec0
        else:  # linear
            vec = vec0

        all_vecs.append(vec)

    all_vecs = np.array(all_vecs)
    vec_mean = np.average(all_vecs, axis=0, weights=w)

    sub = _sub_vcov(model_vcov, coef_names, target_slice)
    sd = float(np.sqrt(vec_mean @ sub @ vec_mean))
    return {"AME": AME, "sd": sd}

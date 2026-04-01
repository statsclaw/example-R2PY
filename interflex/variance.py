"""Variance estimation: simulation, delta method, and bootstrap.

Direct translation of the three variance paths in ``linear.R``.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np
from scipy.stats import norm, t as t_dist

from ._typing import Method, TreatType, VcovType
from .effects import gen_general_te, gen_ate


# ======================================================================
# Simulation variance
# ======================================================================

def variance_simu(
    model_coef: dict[str, float],
    model_vcov: np.ndarray,
    coef_names: list[str],
    nsimu: int,
    X_eval: np.ndarray,
    # Context for gen_general_te
    X: str,
    D: str,
    Z: list[str] | None,
    Z_ref: Any,
    Z_X: list[str] | None,
    full_moderate: bool,
    method: Method,
    treat_type: TreatType,
    use_fe: bool,
    base: str | None,
    D_sample: dict | None,
    diff_values: np.ndarray | None,
    difference_name: list[str] | None,
    # Data for ATE
    data: Any,
    weight_col: str = "WEIGHTS",
) -> dict[str, Any]:
    """Simulation-based variance estimation.

    Returns
    -------
    dict with keys structured per treatment type:
        discrete: est_lin, pred_lin, link_lin, diff_estimate, vcov_matrix, avg_estimate
        continuous: est_lin, pred_lin, link_lin, diff_estimate, vcov_matrix, avg_estimate
    Each keyed by treatment arm label.
    """
    rng = np.random.default_rng()
    coef_array = np.array([model_coef[k] for k in coef_names])

    # Draw simulated coefficients
    try:
        simu_coef = rng.multivariate_normal(coef_array, model_vcov, size=nsimu)
    except np.linalg.LinAlgError:
        model_vcov_reg = model_vcov + np.eye(len(coef_array)) * 1e-10
        simu_coef = rng.multivariate_normal(coef_array, model_vcov_reg, size=nsimu)

    neval = len(X_eval)

    def _make_coef_dict(row: np.ndarray) -> dict[str, float]:
        return dict(zip(coef_names, row))

    results: dict[str, Any] = {}

    if treat_type == "discrete":
        other_treat = [k for k in (D_sample or {}).keys()]
        # Actually for discrete, the keys come from treat_info
        # We'll get them from the caller. For now, use D_sample if provided
        # or fall back to extracting from model_coef
        if D_sample is not None:
            # D_sample for discrete is other_treat list
            arm_list = list(D_sample.keys()) if isinstance(D_sample, dict) else D_sample
        else:
            arm_list = [k.split("D.")[1] for k in model_coef if k.startswith("D.")]

        est_lin = {}
        pred_lin = {}
        link_lin = {}
        diff_est = {}
        vcov_matrix = {}
        avg_est = {}

        for char in arm_list:
            # Point estimates
            te_out = gen_general_te(
                model_coef, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                method, treat_type, use_fe, base,
                char=char, diff_values=diff_values, difference_name=difference_name,
            )
            TE = te_out["TE"]
            E_pred = te_out["E_pred"]
            E_base = te_out["E_base"]
            link_1 = te_out["link_1"]
            link_0 = te_out["link_0"]
            diff_estimate_output = te_out["diff_estimate"]

            ATE_estimate = gen_ate(
                data, model_coef, model_vcov, X, D, Z, Z_X,
                method, treat_type, use_fe, full_moderate,
                char=char, weight_col=weight_col, coef_names=coef_names,
            )

            # Simulation
            te_simu = np.zeros((neval, nsimu))
            pred_simu = np.zeros((neval, nsimu))
            base_simu = np.zeros((neval, nsimu))
            link1_simu = np.zeros((neval, nsimu))
            link0_simu = np.zeros((neval, nsimu))
            n_diff = len(diff_estimate_output) if diff_estimate_output is not None else 0
            diff_simu = np.zeros((n_diff, nsimu)) if n_diff > 0 else None
            ate_simu = np.zeros(nsimu)

            for s in range(nsimu):
                sc = _make_coef_dict(simu_coef[s])
                out = gen_general_te(
                    sc, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                    method, treat_type, use_fe, base,
                    char=char, diff_values=diff_values, difference_name=difference_name,
                )
                te_simu[:, s] = out["TE"]
                pred_simu[:, s] = out["E_pred"]
                base_simu[:, s] = out["E_base"]
                link1_simu[:, s] = out["link_1"]
                link0_simu[:, s] = out["link_0"]
                if diff_simu is not None and out["diff_estimate"] is not None:
                    diff_simu[:, s] = out["diff_estimate"]
                ate_simu[s] = gen_ate(
                    data, sc, model_vcov, X, D, Z, Z_X,
                    method, treat_type, use_fe, full_moderate,
                    char=char, weight_col=weight_col, coef_names=coef_names,
                )

            # Statistics
            te_sd = np.nanstd(te_simu, axis=1, ddof=0)
            te_ci = np.nanpercentile(te_simu, [2.5, 97.5], axis=1).T  # (neval, 2)
            pred_sd = np.nanstd(pred_simu, axis=1, ddof=0)
            pred_ci = np.nanpercentile(pred_simu, [2.5, 97.5], axis=1).T
            base_sd = np.nanstd(base_simu, axis=1, ddof=0)
            base_ci = np.nanpercentile(base_simu, [2.5, 97.5], axis=1).T
            link1_sd = np.nanstd(link1_simu, axis=1, ddof=0)
            link1_ci = np.nanpercentile(link1_simu, [2.5, 97.5], axis=1).T
            link0_sd = np.nanstd(link0_simu, axis=1, ddof=0)
            link0_ci = np.nanpercentile(link0_simu, [2.5, 97.5], axis=1).T

            te_vcov = np.cov(te_simu) if neval > 1 else np.array([[np.nanvar(te_simu)]])

            # TE table: (k, 5)
            est_table = np.column_stack([X_eval, TE, te_sd, te_ci[:, 0], te_ci[:, 1]])
            est_lin[char] = est_table

            pred_table = np.column_stack([X_eval, E_pred, pred_sd, pred_ci[:, 0], pred_ci[:, 1]])
            pred_lin[char] = pred_table

            link_table = np.column_stack([X_eval, link_1, link1_sd, link1_ci[:, 0], link1_ci[:, 1]])
            link_lin[char] = link_table

            vcov_matrix[char] = te_vcov

            # Diff table
            if diff_simu is not None and diff_estimate_output is not None:
                diff_sd = np.nanstd(diff_simu, axis=1, ddof=0)
                diff_ci = np.nanpercentile(diff_simu, [2.5, 97.5], axis=1).T
                z_val = diff_estimate_output / np.where(diff_sd > 0, diff_sd, np.nan)
                p_val = 2 * norm.sf(np.abs(z_val))
                diff_table = np.column_stack([diff_estimate_output, diff_sd, z_val, p_val, diff_ci[:, 0], diff_ci[:, 1]])
                diff_est[char] = diff_table
            else:
                diff_est[char] = None

            # ATE
            ate_sd = np.nanstd(ate_simu, ddof=0)
            ate_ci = np.nanpercentile(ate_simu, [2.5, 97.5])
            z_val = ATE_estimate / ate_sd if ate_sd > 0 else np.nan
            p_val = 2 * norm.sf(np.abs(z_val))
            avg_est[char] = {
                "ATE": ATE_estimate, "sd": ate_sd, "z-value": z_val,
                "p-value": p_val, "lower": ate_ci[0], "upper": ate_ci[1],
            }

        # Base group prediction (last char's E_base is base group)
        if base is not None:
            pred_lin[base] = np.column_stack([X_eval, E_base, base_sd, base_ci[:, 0], base_ci[:, 1]])
            link_lin[base] = np.column_stack([X_eval, link_0, link0_sd, link0_ci[:, 0], link0_ci[:, 1]])

        results = {
            "est_lin": est_lin, "pred_lin": pred_lin, "link_lin": link_lin,
            "diff_estimate": diff_est, "vcov_matrix": vcov_matrix, "avg_estimate": avg_est,
        }

    else:  # continuous
        label_names = list(D_sample.keys()) if D_sample else []
        D_values = list(D_sample.values()) if D_sample else []

        est_lin = {}
        pred_lin = {}
        link_lin = {}
        diff_est = {}
        vcov_matrix = {}

        for k_idx, label in enumerate(label_names):
            D_ref_val = D_values[k_idx]

            me_out = gen_general_te(
                model_coef, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                method, treat_type, use_fe, base,
                D_ref=D_ref_val, D_sample=D_sample,
                diff_values=diff_values, difference_name=difference_name,
            )
            ME = me_out["ME"]
            E_pred = me_out["E_pred"]
            link_val = me_out["link"]
            diff_estimate_output = me_out["diff_estimate"]

            # Simulation
            me_simu = np.zeros((neval, nsimu))
            pred_simu = np.zeros((neval, nsimu))
            link_simu = np.zeros((neval, nsimu))
            n_diff = len(diff_estimate_output) if diff_estimate_output is not None else 0
            diff_simu = np.zeros((n_diff, nsimu)) if n_diff > 0 else None

            for s in range(nsimu):
                sc = _make_coef_dict(simu_coef[s])
                out = gen_general_te(
                    sc, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                    method, treat_type, use_fe, base,
                    D_ref=D_ref_val, D_sample=D_sample,
                    diff_values=diff_values, difference_name=difference_name,
                )
                me_simu[:, s] = out["ME"]
                pred_simu[:, s] = out["E_pred"]
                link_simu[:, s] = out["link"]
                if diff_simu is not None and out["diff_estimate"] is not None:
                    diff_simu[:, s] = out["diff_estimate"]

            me_sd = np.nanstd(me_simu, axis=1, ddof=0)
            me_ci = np.nanpercentile(me_simu, [2.5, 97.5], axis=1).T
            pred_sd = np.nanstd(pred_simu, axis=1, ddof=0)
            pred_ci = np.nanpercentile(pred_simu, [2.5, 97.5], axis=1).T
            link_sd = np.nanstd(link_simu, axis=1, ddof=0)
            link_ci = np.nanpercentile(link_simu, [2.5, 97.5], axis=1).T

            me_vcov = np.cov(me_simu) if neval > 1 else np.array([[np.nanvar(me_simu)]])

            est_table = np.column_stack([X_eval, ME, me_sd, me_ci[:, 0], me_ci[:, 1]])
            est_lin[label] = est_table

            pred_table = np.column_stack([X_eval, E_pred, pred_sd, pred_ci[:, 0], pred_ci[:, 1]])
            pred_lin[label] = pred_table

            link_table = np.column_stack([X_eval, link_val, link_sd, link_ci[:, 0], link_ci[:, 1]])
            link_lin[label] = link_table

            vcov_matrix[label] = me_vcov

            if diff_simu is not None and diff_estimate_output is not None:
                diff_sd_arr = np.nanstd(diff_simu, axis=1, ddof=0)
                diff_ci_arr = np.nanpercentile(diff_simu, [2.5, 97.5], axis=1).T
                z_val = diff_estimate_output / np.where(diff_sd_arr > 0, diff_sd_arr, np.nan)
                p_val = 2 * norm.sf(np.abs(z_val))
                diff_table = np.column_stack([diff_estimate_output, diff_sd_arr, z_val, p_val, diff_ci_arr[:, 0], diff_ci_arr[:, 1]])
                diff_est[label] = diff_table
            else:
                diff_est[label] = None

        # AME (computed once, not per D_ref)
        AME_estimate = gen_ate(
            data, model_coef, model_vcov, X, D, Z, Z_X,
            method, treat_type, use_fe, full_moderate,
            weight_col=weight_col, coef_names=coef_names,
        )
        ame_simu = np.array([
            gen_ate(
                data, _make_coef_dict(simu_coef[s]), model_vcov, X, D, Z, Z_X,
                method, treat_type, use_fe, full_moderate,
                weight_col=weight_col, coef_names=coef_names,
            )
            for s in range(nsimu)
        ])
        ame_sd = np.nanstd(ame_simu, ddof=0)
        ame_ci = np.nanpercentile(ame_simu, [2.5, 97.5])
        z_val = AME_estimate / ame_sd if ame_sd > 0 else np.nan
        p_val = 2 * norm.sf(np.abs(z_val))
        avg_estimate = {
            "AME": AME_estimate, "sd": ame_sd, "z-value": z_val,
            "p-value": p_val, "lower": ame_ci[0], "upper": ame_ci[1],
        }

        results = {
            "est_lin": est_lin, "pred_lin": pred_lin, "link_lin": link_lin,
            "diff_estimate": diff_est, "vcov_matrix": vcov_matrix, "avg_estimate": avg_estimate,
        }

    return results


# ======================================================================
# Delta-method variance
# ======================================================================

def variance_delta(
    model_coef: dict[str, float],
    model_vcov: np.ndarray,
    coef_names: list[str],
    X_eval: np.ndarray,
    model_df: int,
    X: str,
    D: str,
    Z: list[str] | None,
    Z_ref: Any,
    Z_X: list[str] | None,
    full_moderate: bool,
    method: Method,
    treat_type: TreatType,
    use_fe: bool,
    base: str | None,
    D_sample: dict | None,
    diff_values: np.ndarray | None,
    difference_name: list[str] | None,
    data: Any,
    weight_col: str = "WEIGHTS",
) -> dict[str, Any]:
    """Delta-method variance estimation.

    Returns same structure as variance_simu.
    """
    from .uniform import calculate_delta_uniform_ci

    crit = abs(t_dist.ppf(0.025, df=model_df))
    neval = len(X_eval)

    def _c(name: str) -> float:
        return model_coef.get(name, 0.0)

    def _zref_val(name: str) -> float:
        if Z_ref is None:
            return 0.0
        if isinstance(Z_ref, dict):
            return Z_ref.get(name, 0.0)
        if Z is not None:
            try:
                return float(Z_ref[Z.index(name)])
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    def _sub_vcov(target_slice: list[str]) -> np.ndarray:
        p = len(target_slice)
        sub = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                ii = coef_names.index(target_slice[i]) if target_slice[i] in coef_names else -1
                jj = coef_names.index(target_slice[j]) if target_slice[j] in coef_names else -1
                if ii >= 0 and jj >= 0:
                    sub[i, j] = model_vcov[ii, jj]
        return sub

    # ------------------------------------------------------------------
    # gen_sd: compute delta-method SE at point x
    # ------------------------------------------------------------------
    def gen_sd_fe(x: float, treat_t: str, char: str | None = None, D_ref_val: float | None = None, to_diff: bool = False):
        if treat_t == "discrete":
            target_slice = [f"D.{char}", f"DX.{char}"]
            vec = np.array([1.0, x])
            temp_vcov = _sub_vcov(target_slice)
            if to_diff:
                return {"vec": vec, "temp_vcov": temp_vcov}
            return float(np.sqrt(vec @ temp_vcov @ vec))

        else:
            target_slice = [D, "DX"]
            vec = np.array([1.0, x])
            temp_vcov = _sub_vcov(target_slice)
            if to_diff:
                return {"vec": vec, "temp_vcov": temp_vcov}
            return float(np.sqrt(vec @ temp_vcov @ vec))

    def gen_sd(x: float, treat_t: str, char: str | None = None, D_ref_val: float | None = None, to_diff: bool = False):
        if use_fe:
            return gen_sd_fe(x, treat_t, char, D_ref_val, to_diff)

        if treat_t == "discrete":
            assert char is not None
            link_1 = _c("(Intercept)") + x * _c(X) + _c(f"D.{char}") + x * _c(f"DX.{char}")
            link_0 = _c("(Intercept)") + x * _c(X)

            if Z is not None:
                vec_1 = [1.0, x, 1.0, x]
                vec_0 = [1.0, x, 0.0, 0.0]
                target_slice = ["(Intercept)", X, f"D.{char}", f"DX.{char}"]
                z_vals_ref = []
                for a in Z:
                    zv = _zref_val(a)
                    z_vals_ref.append(zv)
                    link_1 += zv * _c(a)
                    link_0 += zv * _c(a)
                    vec_1.append(zv)
                    vec_0.append(zv)
                    target_slice.append(a)
                    if full_moderate:
                        link_1 += zv * _c(f"{a}.X") * x
                        link_0 += zv * _c(f"{a}.X") * x
                if full_moderate and Z_X:
                    for i_a, a in enumerate(Z):
                        vec_1.append(z_vals_ref[i_a] * x)
                        vec_0.append(z_vals_ref[i_a] * x)
                        target_slice.append(Z_X[i_a])
            else:
                vec_1 = [1.0, x, 1.0, x]
                vec_0 = [1.0, x, 0.0, 0.0]
                target_slice = ["(Intercept)", X, f"D.{char}", f"DX.{char}"]

            vec_1 = np.array(vec_1, dtype=float)
            vec_0 = np.array(vec_0, dtype=float)
            temp_vcov = _sub_vcov(target_slice)

            if method == "logit":
                vec = vec_1 * np.exp(link_1) / (1 + np.exp(link_1)) ** 2 - vec_0 * np.exp(link_0) / (1 + np.exp(link_0)) ** 2
            elif method == "probit":
                from scipy.stats import norm as norm_dist
                vec = vec_1 * norm_dist.pdf(link_1) - vec_0 * norm_dist.pdf(link_0)
            elif method in ("poisson", "nbinom"):
                vec = vec_1 * np.exp(link_1) - vec_0 * np.exp(link_0)
            else:
                vec = vec_1 - vec_0

            if to_diff:
                return {"vec": vec, "temp_vcov": temp_vcov}
            return float(np.sqrt(vec @ temp_vcov @ vec))

        else:  # continuous
            assert D_ref_val is not None
            link = _c("(Intercept)") + x * _c(X) + _c(D) * D_ref_val + _c("DX") * x * D_ref_val

            if Z is not None:
                if full_moderate and Z_X:
                    vec1 = [1.0, x, D_ref_val, D_ref_val * x]
                    vec0 = [0.0, 0.0, 1.0, x]
                    target_slice = ["(Intercept)", X, D, "DX"]
                    z_vals_ref = []
                    for a in Z:
                        zv = _zref_val(a)
                        z_vals_ref.append(zv)
                        link += zv * _c(a) + zv * _c(f"{a}.X") * x
                        vec1.append(zv)
                        vec0.append(0.0)
                        target_slice.append(a)
                    for i_a, a in enumerate(Z):
                        vec1.append(z_vals_ref[i_a] * x)
                        vec0.append(0.0)
                        target_slice.append(Z_X[i_a])
                else:
                    vec1 = [1.0, x, D_ref_val, D_ref_val * x]
                    vec0 = [0.0, 0.0, 1.0, x]
                    target_slice = ["(Intercept)", X, D, "DX"]
                    for a in Z:
                        zv = _zref_val(a)
                        link += zv * _c(a)
                        vec1.append(zv)
                        vec0.append(0.0)
                        target_slice.append(a)
            else:
                vec1 = [1.0, x, D_ref_val, D_ref_val * x]
                vec0 = [0.0, 0.0, 1.0, x]
                target_slice = ["(Intercept)", X, D, "DX"]

            vec1 = np.array(vec1, dtype=float)
            vec0 = np.array(vec0, dtype=float)
            temp_vcov = _sub_vcov(target_slice)

            if method == "logit":
                vec = (
                    -(_c(D) + x * _c("DX"))
                    * (np.exp(link) - np.exp(-link))
                    / (2 + np.exp(link) + np.exp(-link)) ** 2
                    * vec1
                    + np.exp(link) / (1 + np.exp(link)) ** 2 * vec0
                )
            elif method == "probit":
                from scipy.stats import norm as norm_dist
                vec = norm_dist.pdf(link) * vec0 - (_c(D) + x * _c("DX")) * link * norm_dist.pdf(link) * vec1
            elif method in ("poisson", "nbinom"):
                vec = (_c(D) + x * _c("DX")) * np.exp(link) * vec1 + np.exp(link) * vec0
            else:
                vec = vec0

            if to_diff:
                return {"vec": vec, "temp_vcov": temp_vcov}
            return float(np.sqrt(vec @ temp_vcov @ vec))

    # ------------------------------------------------------------------
    # gen_link_sd (for prediction SE)
    # ------------------------------------------------------------------
    def gen_link_sd(x: float, treat_t: str, char: str | None = None, D_ref_val: float | None = None, is_base: bool = False):
        if use_fe:
            return 0.0

        if treat_t == "discrete":
            if is_base:
                if Z is not None:
                    vec = [1.0, x]
                    target_slice = ["(Intercept)", X]
                    for a in Z:
                        vec.append(_zref_val(a))
                        target_slice.append(a)
                    if full_moderate and Z_X:
                        for a in Z:
                            vec.append(_zref_val(a) * x)
                        target_slice.extend(Z_X)
                else:
                    vec = [1.0, x]
                    target_slice = ["(Intercept)", X]
            else:
                if Z is not None:
                    vec = [1.0, x, 1.0, x]
                    target_slice = ["(Intercept)", X, f"D.{char}", f"DX.{char}"]
                    for a in Z:
                        vec.append(_zref_val(a))
                        target_slice.append(a)
                    if full_moderate and Z_X:
                        for a in Z:
                            vec.append(_zref_val(a) * x)
                        target_slice.extend(Z_X)
                else:
                    vec = [1.0, x, 1.0, x]
                    target_slice = ["(Intercept)", X, f"D.{char}", f"DX.{char}"]

            vec = np.array(vec, dtype=float)
            temp_vcov = _sub_vcov(target_slice)
            return float(np.sqrt(vec @ temp_vcov @ vec))

        else:  # continuous
            if Z is not None:
                vec = [1.0, x, D_ref_val, D_ref_val * x]
                target_slice = ["(Intercept)", X, D, "DX"]
                for a in Z:
                    vec.append(_zref_val(a))
                    target_slice.append(a)
                if full_moderate and Z_X:
                    for a in Z:
                        vec.append(_zref_val(a) * x)
                    target_slice.extend(Z_X)
            else:
                vec = [1.0, x, D_ref_val, D_ref_val * x]
                target_slice = ["(Intercept)", X, D, "DX"]

            vec = np.array(vec, dtype=float)
            temp_vcov = _sub_vcov(target_slice)
            return float(np.sqrt(vec @ temp_vcov @ vec))

    # ------------------------------------------------------------------
    # gen_predict_sd
    # ------------------------------------------------------------------
    def gen_predict_sd(x: float, treat_t: str, char: str | None = None, D_ref_val: float | None = None, is_base: bool = False):
        if use_fe:
            return 0.0

        if treat_t == "discrete":
            if is_base:
                link = _c("(Intercept)") + x * _c(X)
            else:
                link = _c("(Intercept)") + x * _c(X) + _c(f"D.{char}") + x * _c(f"DX.{char}")
            if Z is not None:
                for a in Z:
                    link += _zref_val(a) * _c(a)
                    if full_moderate:
                        link += _zref_val(a) * _c(f"{a}.X") * x

            if is_base:
                if Z is not None:
                    vec = [1.0, x]
                    target_slice = ["(Intercept)", X]
                    for a in Z:
                        vec.append(_zref_val(a))
                        target_slice.append(a)
                    if full_moderate and Z_X:
                        for a in Z:
                            vec.append(_zref_val(a) * x)
                        target_slice.extend(Z_X)
                else:
                    vec = [1.0, x]
                    target_slice = ["(Intercept)", X]
            else:
                if Z is not None:
                    vec = [1.0, x, 1.0, x]
                    target_slice = ["(Intercept)", X, f"D.{char}", f"DX.{char}"]
                    for a in Z:
                        vec.append(_zref_val(a))
                        target_slice.append(a)
                    if full_moderate and Z_X:
                        for a in Z:
                            vec.append(_zref_val(a) * x)
                        target_slice.extend(Z_X)
                else:
                    vec = [1.0, x, 1.0, x]
                    target_slice = ["(Intercept)", X, f"D.{char}", f"DX.{char}"]

            vec = np.array(vec, dtype=float)
            temp_vcov = _sub_vcov(target_slice)

            if method == "logit":
                vec = vec * np.exp(link) / (1 + np.exp(link)) ** 2
            elif method == "probit":
                vec = vec * norm.pdf(link)
            elif method in ("poisson", "nbinom"):
                vec = vec * np.exp(link)
            return float(np.sqrt(vec @ temp_vcov @ vec))

        else:  # continuous
            link = _c("(Intercept)") + x * _c(X) + _c(D) * D_ref_val + _c("DX") * x * D_ref_val
            if Z is not None:
                for a in Z:
                    link += _zref_val(a) * _c(a)
                    if full_moderate:
                        link += _zref_val(a) * _c(f"{a}.X") * x

            if Z is not None:
                vec = [1.0, x, D_ref_val, D_ref_val * x]
                target_slice = ["(Intercept)", X, D, "DX"]
                for a in Z:
                    vec.append(_zref_val(a))
                    target_slice.append(a)
                if full_moderate and Z_X:
                    for a in Z:
                        vec.append(_zref_val(a) * x)
                    target_slice.extend(Z_X)
            else:
                vec = [1.0, x, D_ref_val, D_ref_val * x]
                target_slice = ["(Intercept)", X, D, "DX"]

            vec = np.array(vec, dtype=float)
            temp_vcov = _sub_vcov(target_slice)

            if method == "logit":
                vec = vec * np.exp(link) / (1 + np.exp(link)) ** 2
            elif method == "probit":
                vec = vec * norm.pdf(link)
            elif method in ("poisson", "nbinom"):
                vec = vec * np.exp(link)
            return float(np.sqrt(vec @ temp_vcov @ vec))

    # ------------------------------------------------------------------
    # Build results
    # ------------------------------------------------------------------
    results: dict[str, Any] = {}

    if treat_type == "discrete":
        arm_list = list(D_sample.keys()) if D_sample else [k.split("D.")[1] for k in model_coef if k.startswith("D.")]
        est_lin = {}
        pred_lin = {}
        link_lin = {}
        diff_est = {}
        vcov_matrix = {}
        avg_est = {}

        for char_arm in arm_list:
            te_out = gen_general_te(
                model_coef, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                method, treat_type, use_fe, base,
                char=char_arm, diff_values=diff_values, difference_name=difference_name,
            )
            TE = te_out["TE"]
            E_pred = te_out["E_pred"]
            E_base = te_out["E_base"]
            link_1 = te_out["link_1"]
            link_0 = te_out["link_0"]
            diff_estimate_output = te_out["diff_estimate"]

            # ATE with delta SE
            ate_result = gen_ate(
                data, model_coef, model_vcov, X, D, Z, Z_X,
                method, treat_type, use_fe, full_moderate,
                char=char_arm, delta=True, weight_col=weight_col, coef_names=coef_names,
            )

            # Delta SEs
            te_sd = np.array([gen_sd(x, "discrete", char=char_arm) for x in X_eval])
            pred_sd = np.array([gen_predict_sd(x, "discrete", char=char_arm) for x in X_eval])
            link1_sd = np.array([gen_link_sd(x, "discrete", char=char_arm) for x in X_eval])

            # Vcov matrix
            te_vcov = np.zeros((neval, neval))
            for i in range(neval):
                vi = gen_sd(X_eval[i], "discrete", char=char_arm, to_diff=True)
                for j in range(neval):
                    vj = gen_sd(X_eval[j], "discrete", char=char_arm, to_diff=True)
                    te_vcov[i, j] = float(vi["vec"] @ vi["temp_vcov"] @ vj["vec"])

            vcov_matrix[char_arm] = te_vcov

            # Uniform CI
            try:
                uniform_q = calculate_delta_uniform_ci(te_vcov, alpha=0.05, N=2000)
            except Exception:
                uniform_q = crit

            est_table = np.column_stack([
                X_eval, TE, te_sd,
                TE - crit * te_sd, TE + crit * te_sd,
                TE - uniform_q * te_sd, TE + uniform_q * te_sd,
            ])
            est_lin[char_arm] = est_table

            pred_table = np.column_stack([
                X_eval, E_pred, pred_sd,
                E_pred - crit * pred_sd, E_pred + crit * pred_sd,
            ])
            pred_lin[char_arm] = pred_table

            link_table = np.column_stack([
                X_eval, link_1, link1_sd,
                link_1 - crit * link1_sd, link_1 + crit * link1_sd,
            ])
            link_lin[char_arm] = link_table

            # Diff
            if diff_estimate_output is not None and diff_values is not None:
                n_d = len(diff_estimate_output)
                diff_sd_arr = np.zeros(n_d)
                vecs_diff = [gen_sd(dv, "discrete", char=char_arm, to_diff=True) for dv in diff_values]
                if len(diff_values) == 2:
                    v = vecs_diff[1]["vec"] - vecs_diff[0]["vec"]
                    diff_sd_arr[0] = float(np.sqrt(v @ vecs_diff[0]["temp_vcov"] @ v))
                elif len(diff_values) == 3:
                    for idx, (a, b) in enumerate([(1, 0), (2, 1), (2, 0)]):
                        v = vecs_diff[b]["vec"] - vecs_diff[a]["vec"]
                        diff_sd_arr[idx] = float(np.sqrt(v @ vecs_diff[0]["temp_vcov"] @ v))

                z_val = diff_estimate_output / np.where(diff_sd_arr > 0, diff_sd_arr, np.nan)
                p_val = 2 * norm.sf(np.abs(z_val))
                diff_table = np.column_stack([
                    diff_estimate_output, diff_sd_arr, z_val, p_val,
                    diff_estimate_output - crit * diff_sd_arr,
                    diff_estimate_output + crit * diff_sd_arr,
                ])
                diff_est[char_arm] = diff_table
            else:
                diff_est[char_arm] = None

            # ATE
            if isinstance(ate_result, dict):
                ate_val = ate_result["ATE"]
                ate_sd_val = ate_result["sd"]
            else:
                ate_val = ate_result
                ate_sd_val = 0.0
            z_val = ate_val / ate_sd_val if ate_sd_val > 0 else np.nan
            p_val = 2 * norm.sf(np.abs(z_val))
            avg_est[char_arm] = {
                "ATE": ate_val, "sd": ate_sd_val, "z-value": z_val, "p-value": p_val,
                "lower": ate_val - crit * ate_sd_val, "upper": ate_val + crit * ate_sd_val,
            }

        # Base group predictions
        if base is not None:
            base_pred_sd = np.array([gen_predict_sd(x, "discrete", char=base, is_base=True) for x in X_eval])
            base_link_sd = np.array([gen_link_sd(x, "discrete", char=base, is_base=True) for x in X_eval])
            pred_lin[base] = np.column_stack([
                X_eval, E_base, base_pred_sd,
                E_base - crit * base_pred_sd, E_base + crit * base_pred_sd,
            ])
            link_lin[base] = np.column_stack([
                X_eval, link_0, base_link_sd,
                link_0 - crit * base_link_sd, link_0 + crit * base_link_sd,
            ])

        results = {
            "est_lin": est_lin, "pred_lin": pred_lin, "link_lin": link_lin,
            "diff_estimate": diff_est, "vcov_matrix": vcov_matrix, "avg_estimate": avg_est,
        }

    else:  # continuous delta
        label_names = list(D_sample.keys()) if D_sample else []
        D_values = list(D_sample.values()) if D_sample else []

        est_lin = {}
        pred_lin = {}
        link_lin = {}
        diff_est = {}
        vcov_matrix = {}

        for k_idx, label in enumerate(label_names):
            D_ref_val = D_values[k_idx]

            me_out = gen_general_te(
                model_coef, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                method, treat_type, use_fe, base,
                D_ref=D_ref_val, D_sample=D_sample,
                diff_values=diff_values, difference_name=difference_name,
            )
            ME = me_out["ME"]
            E_pred = me_out["E_pred"]
            link_val = me_out["link"]
            diff_estimate_output = me_out["diff_estimate"]

            me_sd = np.array([gen_sd(x, "continuous", D_ref_val=D_ref_val) for x in X_eval])
            pred_sd = np.array([gen_predict_sd(x, "continuous", D_ref_val=D_ref_val) for x in X_eval])
            link_sd_arr = np.array([gen_link_sd(x, "continuous", D_ref_val=D_ref_val) for x in X_eval])

            # ME vcov
            me_vcov = np.zeros((neval, neval))
            for i in range(neval):
                vi = gen_sd(X_eval[i], "continuous", D_ref_val=D_ref_val, to_diff=True)
                for j in range(neval):
                    vj = gen_sd(X_eval[j], "continuous", D_ref_val=D_ref_val, to_diff=True)
                    me_vcov[i, j] = float(vi["vec"] @ vi["temp_vcov"] @ vj["vec"])

            vcov_matrix[label] = me_vcov

            try:
                uniform_q = calculate_delta_uniform_ci(me_vcov, alpha=0.05, N=2000)
            except Exception:
                uniform_q = crit

            est_table = np.column_stack([
                X_eval, ME, me_sd,
                ME - crit * me_sd, ME + crit * me_sd,
                ME - uniform_q * me_sd, ME + uniform_q * me_sd,
            ])
            est_lin[label] = est_table

            pred_table = np.column_stack([
                X_eval, E_pred, pred_sd,
                E_pred - crit * pred_sd, E_pred + crit * pred_sd,
            ])
            pred_lin[label] = pred_table

            link_table = np.column_stack([
                X_eval, link_val, link_sd_arr,
                link_val - crit * link_sd_arr, link_val + crit * link_sd_arr,
            ])
            link_lin[label] = link_table

            if diff_estimate_output is not None and diff_values is not None:
                n_d = len(diff_estimate_output)
                diff_sd_arr2 = np.zeros(n_d)
                vecs_diff = [gen_sd(dv, "continuous", D_ref_val=D_ref_val, to_diff=True) for dv in diff_values]
                if len(diff_values) == 2:
                    v = vecs_diff[1]["vec"] - vecs_diff[0]["vec"]
                    diff_sd_arr2[0] = float(np.sqrt(v @ vecs_diff[0]["temp_vcov"] @ v))
                elif len(diff_values) == 3:
                    for idx, (a, b) in enumerate([(1, 0), (2, 1), (2, 0)]):
                        v = vecs_diff[b]["vec"] - vecs_diff[a]["vec"]
                        diff_sd_arr2[idx] = float(np.sqrt(v @ vecs_diff[0]["temp_vcov"] @ v))

                z_val = diff_estimate_output / np.where(diff_sd_arr2 > 0, diff_sd_arr2, np.nan)
                p_val = 2 * norm.sf(np.abs(z_val))
                diff_table = np.column_stack([
                    diff_estimate_output, diff_sd_arr2, z_val, p_val,
                    diff_estimate_output - crit * diff_sd_arr2,
                    diff_estimate_output + crit * diff_sd_arr2,
                ])
                diff_est[label] = diff_table
            else:
                diff_est[label] = None

        # AME with delta
        ame_result = gen_ate(
            data, model_coef, model_vcov, X, D, Z, Z_X,
            method, treat_type, use_fe, full_moderate,
            delta=True, weight_col=weight_col, coef_names=coef_names,
        )
        if isinstance(ame_result, dict):
            ame_val = ame_result["AME"]
            ame_sd_val = ame_result["sd"]
        else:
            ame_val = ame_result
            ame_sd_val = 0.0
        z_val = ame_val / ame_sd_val if ame_sd_val > 0 else np.nan
        p_val = 2 * norm.sf(np.abs(z_val))
        avg_estimate = {
            "AME": ame_val, "sd": ame_sd_val, "z-value": z_val, "p-value": p_val,
            "lower": ame_val - crit * ame_sd_val, "upper": ame_val + crit * ame_sd_val,
        }

        results = {
            "est_lin": est_lin, "pred_lin": pred_lin, "link_lin": link_lin,
            "diff_estimate": diff_est, "vcov_matrix": vcov_matrix, "avg_estimate": avg_estimate,
        }

    return results


# ======================================================================
# Bootstrap variance
# ======================================================================

def variance_bootstrap(
    data: Any,
    formula_info: dict[str, Any],
    nboots: int,
    cl: str | None,
    X_eval: np.ndarray,
    X: str,
    D: str,
    Z: list[str] | None,
    Z_ref: Any,
    Z_X: list[str] | None,
    full_moderate: bool,
    method: Method,
    treat_type: TreatType,
    use_fe: bool,
    base: str | None,
    D_sample: dict | None,
    diff_values: np.ndarray | None,
    difference_name: list[str] | None,
    parallel: bool = False,
    cores: int = 4,
    weight_col: str = "WEIGHTS",
    coef_names: list[str] | None = None,
    FE: list[str] | None = None,
    IV: list[str] | None = None,
    vcov_type: VcovType = "robust",
    time: str | None = None,
    pairwise: bool = True,
) -> dict[str, Any]:
    """Bootstrap variance estimation.

    Returns same structure as variance_simu.
    """
    from .uniform import calculate_uniform_quantiles
    import pandas as pd

    neval = len(X_eval)
    n = len(data)

    # Cluster info
    if cl is not None:
        clusters = data[cl].unique()
        id_list = {c: data.index[data[cl] == c].tolist() for c in clusters}

    # Determine arm list
    if treat_type == "discrete":
        arm_list = list(D_sample.keys()) if D_sample else []
    else:
        label_names = list(D_sample.keys()) if D_sample else []
        D_values = list(D_sample.values()) if D_sample else []

    def _one_boot(seed: int) -> dict | None:
        """Execute one bootstrap replicate."""
        rng = np.random.default_rng(seed)

        if cl is None:
            smp = rng.choice(n, n, replace=True)
        else:
            boot_clusters = rng.choice(clusters, len(clusters), replace=True)
            smp_list = []
            for bc in boot_clusters:
                smp_list.extend(id_list[bc])
            smp = np.array(smp_list)

        data_boot = data.iloc[smp].reset_index(drop=True)

        # Check validity
        if treat_type == "discrete":
            if len(data_boot[D].unique()) != len(data[D].unique()):
                return None

        # Refit model
        try:
            boot_coef, boot_vcov, boot_model = _fit_model(
                data_boot, formula_info, method, use_fe, FE, IV, weight_col,
            )
        except Exception:
            return None

        if boot_coef is None:
            return None

        boot_coef_dict = dict(zip(coef_names, boot_coef)) if coef_names else {}
        # Fill missing with 0
        for k in (coef_names or []):
            if k not in boot_coef_dict:
                boot_coef_dict[k] = 0.0

        result = {}
        if treat_type == "discrete":
            for char in arm_list:
                out = gen_general_te(
                    boot_coef_dict, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                    method, treat_type, use_fe, base,
                    char=char, diff_values=diff_values, difference_name=difference_name,
                )
                ate = gen_ate(
                    data_boot, boot_coef_dict, None, X, D, Z, Z_X,
                    method, treat_type, use_fe, full_moderate,
                    char=char, weight_col=weight_col, coef_names=coef_names,
                )
                result[char] = {
                    "TE": out["TE"], "E_pred": out["E_pred"], "E_base": out["E_base"],
                    "link_1": out["link_1"], "link_0": out["link_0"],
                    "diff_estimate": out["diff_estimate"], "ATE": ate,
                }
        else:
            for k_idx, label in enumerate(label_names):
                D_ref_val = D_values[k_idx]
                out = gen_general_te(
                    boot_coef_dict, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                    method, treat_type, use_fe, base,
                    D_ref=D_ref_val, D_sample=D_sample,
                    diff_values=diff_values, difference_name=difference_name,
                )
                result[label] = {
                    "ME": out["ME"], "E_pred": out["E_pred"], "link": out["link"],
                    "diff_estimate": out["diff_estimate"],
                }
            # AME
            ame = gen_ate(
                data_boot, boot_coef_dict, None, X, D, Z, Z_X,
                method, treat_type, use_fe, full_moderate,
                weight_col=weight_col, coef_names=coef_names,
            )
            result["_AME"] = ame

        return result

    # Run bootstrap
    seeds = np.random.default_rng().integers(0, 2**31, size=nboots)
    boot_results = []

    if parallel and cores > 1:
        with ProcessPoolExecutor(max_workers=cores) as pool:
            futures = [pool.submit(_one_boot, int(s)) for s in seeds]
            for f in futures:
                r = f.result()
                if r is not None:
                    boot_results.append(r)
    else:
        for s in seeds:
            r = _one_boot(int(s))
            if r is not None:
                boot_results.append(r)

    n_valid = len(boot_results)
    if n_valid == 0:
        warnings.warn("No valid bootstrap replicates.")
        return {"est_lin": {}, "pred_lin": {}, "link_lin": {},
                "diff_estimate": {}, "vcov_matrix": {}, "avg_estimate": {}}

    # Aggregate results
    # Point estimates (from original data)
    results: dict[str, Any] = {}

    if treat_type == "discrete":
        est_lin = {}
        pred_lin = {}
        link_lin = {}
        diff_est_out = {}
        vcov_matrix = {}
        avg_est = {}

        for char in arm_list:
            te_out = gen_general_te(
                dict(zip(coef_names, [formula_info["model_coef"].get(k, 0.0) for k in coef_names])) if coef_names else formula_info.get("model_coef_dict", {}),
                X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                method, treat_type, use_fe, base,
                char=char, diff_values=diff_values, difference_name=difference_name,
            )
            TE = te_out["TE"]
            E_pred = te_out["E_pred"]
            E_base = te_out["E_base"]
            link_1 = te_out["link_1"]
            link_0 = te_out["link_0"]
            diff_estimate_output = te_out["diff_estimate"]

            ATE_estimate = gen_ate(
                data,
                dict(zip(coef_names, [formula_info["model_coef"].get(k, 0.0) for k in coef_names])) if coef_names else formula_info.get("model_coef_dict", {}),
                None, X, D, Z, Z_X,
                method, treat_type, use_fe, full_moderate,
                char=char, weight_col=weight_col, coef_names=coef_names,
            )

            # Collect bootstrap matrices
            te_boot = np.column_stack([br[char]["TE"] for br in boot_results if char in br])
            pred_boot = np.column_stack([br[char]["E_pred"] for br in boot_results if char in br])
            base_boot = np.column_stack([br[char]["E_base"] for br in boot_results if char in br])
            link1_boot = np.column_stack([br[char]["link_1"] for br in boot_results if char in br])
            link0_boot = np.column_stack([br[char]["link_0"] for br in boot_results if char in br])
            ate_boot = np.array([br[char]["ATE"] for br in boot_results if char in br])

            te_sd = np.nanstd(te_boot, axis=1, ddof=0)
            te_ci = np.nanpercentile(te_boot, [2.5, 97.5], axis=1).T
            pred_sd = np.nanstd(pred_boot, axis=1, ddof=0)
            pred_ci = np.nanpercentile(pred_boot, [2.5, 97.5], axis=1).T
            base_sd = np.nanstd(base_boot, axis=1, ddof=0)
            base_ci = np.nanpercentile(base_boot, [2.5, 97.5], axis=1).T
            link1_sd = np.nanstd(link1_boot, axis=1, ddof=0)
            link1_ci = np.nanpercentile(link1_boot, [2.5, 97.5], axis=1).T
            link0_sd = np.nanstd(link0_boot, axis=1, ddof=0)
            link0_ci = np.nanpercentile(link0_boot, [2.5, 97.5], axis=1).T

            te_vcov = np.cov(te_boot) if neval > 1 else np.array([[np.nanvar(te_boot)]])
            vcov_matrix[char] = te_vcov

            # Uniform CI
            try:
                Q_j, _, _ = calculate_uniform_quantiles(te_boot, alpha=0.05)
                est_table = np.column_stack([
                    X_eval, TE, te_sd, te_ci[:, 0], te_ci[:, 1], Q_j[:, 0], Q_j[:, 1],
                ])
            except Exception:
                est_table = np.column_stack([X_eval, TE, te_sd, te_ci[:, 0], te_ci[:, 1]])
            est_lin[char] = est_table

            pred_lin[char] = np.column_stack([X_eval, E_pred, pred_sd, pred_ci[:, 0], pred_ci[:, 1]])
            link_lin[char] = np.column_stack([X_eval, link_1, link1_sd, link1_ci[:, 0], link1_ci[:, 1]])

            # Diff
            if diff_estimate_output is not None:
                n_d = len(diff_estimate_output)
                diff_boot_all = []
                for br in boot_results:
                    if char in br and br[char]["diff_estimate"] is not None:
                        diff_boot_all.append(br[char]["diff_estimate"])
                if diff_boot_all:
                    diff_boot_mat = np.column_stack(diff_boot_all)
                    diff_sd_arr = np.nanstd(diff_boot_mat, axis=1, ddof=0)
                    diff_ci_arr = np.nanpercentile(diff_boot_mat, [2.5, 97.5], axis=1).T
                    z_val = diff_estimate_output / np.where(diff_sd_arr > 0, diff_sd_arr, np.nan)
                    p_val = 2 * norm.sf(np.abs(z_val))
                    diff_est_out[char] = np.column_stack([
                        diff_estimate_output, diff_sd_arr, z_val, p_val,
                        diff_ci_arr[:, 0], diff_ci_arr[:, 1],
                    ])
                else:
                    diff_est_out[char] = None
            else:
                diff_est_out[char] = None

            ate_sd = np.nanstd(ate_boot, ddof=0)
            ate_ci = np.nanpercentile(ate_boot, [2.5, 97.5])
            z_val = ATE_estimate / ate_sd if ate_sd > 0 else np.nan
            p_val = 2 * norm.sf(np.abs(z_val))
            avg_est[char] = {
                "ATE": ATE_estimate, "sd": ate_sd, "z-value": z_val,
                "p-value": p_val, "lower": ate_ci[0], "upper": ate_ci[1],
            }

        if base is not None:
            pred_lin[base] = np.column_stack([X_eval, E_base, base_sd, base_ci[:, 0], base_ci[:, 1]])
            link_lin[base] = np.column_stack([X_eval, link_0, link0_sd, link0_ci[:, 0], link0_ci[:, 1]])

        results = {
            "est_lin": est_lin, "pred_lin": pred_lin, "link_lin": link_lin,
            "diff_estimate": diff_est_out, "vcov_matrix": vcov_matrix, "avg_estimate": avg_est,
        }

    else:  # continuous bootstrap
        est_lin = {}
        pred_lin = {}
        link_lin = {}
        diff_est_out = {}
        vcov_matrix = {}

        model_coef_dict = dict(zip(coef_names, [formula_info["model_coef"].get(k, 0.0) for k in coef_names])) if coef_names else formula_info.get("model_coef_dict", {})

        for k_idx, label in enumerate(label_names):
            D_ref_val = D_values[k_idx]

            me_out = gen_general_te(
                model_coef_dict, X_eval, X, D, Z, Z_ref, Z_X, full_moderate,
                method, treat_type, use_fe, base,
                D_ref=D_ref_val, D_sample=D_sample,
                diff_values=diff_values, difference_name=difference_name,
            )
            ME = me_out["ME"]
            E_pred = me_out["E_pred"]
            link_val = me_out["link"]
            diff_estimate_output = me_out["diff_estimate"]

            me_boot = np.column_stack([br[label]["ME"] for br in boot_results if label in br])
            pred_boot = np.column_stack([br[label]["E_pred"] for br in boot_results if label in br])
            link_boot = np.column_stack([br[label]["link"] for br in boot_results if label in br])

            me_sd = np.nanstd(me_boot, axis=1, ddof=0)
            me_ci = np.nanpercentile(me_boot, [2.5, 97.5], axis=1).T
            pred_sd = np.nanstd(pred_boot, axis=1, ddof=0)
            pred_ci = np.nanpercentile(pred_boot, [2.5, 97.5], axis=1).T
            link_sd_arr = np.nanstd(link_boot, axis=1, ddof=0)
            link_ci = np.nanpercentile(link_boot, [2.5, 97.5], axis=1).T

            me_vcov = np.cov(me_boot) if neval > 1 else np.array([[np.nanvar(me_boot)]])
            vcov_matrix[label] = me_vcov

            try:
                Q_j, _, _ = calculate_uniform_quantiles(me_boot, alpha=0.05)
                est_table = np.column_stack([
                    X_eval, ME, me_sd, me_ci[:, 0], me_ci[:, 1], Q_j[:, 0], Q_j[:, 1],
                ])
            except Exception:
                est_table = np.column_stack([X_eval, ME, me_sd, me_ci[:, 0], me_ci[:, 1]])
            est_lin[label] = est_table

            pred_lin[label] = np.column_stack([X_eval, E_pred, pred_sd, pred_ci[:, 0], pred_ci[:, 1]])
            link_lin[label] = np.column_stack([X_eval, link_val, link_sd_arr, link_ci[:, 0], link_ci[:, 1]])

            if diff_estimate_output is not None:
                diff_boot_all = []
                for br in boot_results:
                    if label in br and br[label]["diff_estimate"] is not None:
                        diff_boot_all.append(br[label]["diff_estimate"])
                if diff_boot_all:
                    diff_boot_mat = np.column_stack(diff_boot_all)
                    diff_sd_arr2 = np.nanstd(diff_boot_mat, axis=1, ddof=0)
                    diff_ci_arr2 = np.nanpercentile(diff_boot_mat, [2.5, 97.5], axis=1).T
                    z_val = diff_estimate_output / np.where(diff_sd_arr2 > 0, diff_sd_arr2, np.nan)
                    p_val = 2 * norm.sf(np.abs(z_val))
                    diff_est_out[label] = np.column_stack([
                        diff_estimate_output, diff_sd_arr2, z_val, p_val,
                        diff_ci_arr2[:, 0], diff_ci_arr2[:, 1],
                    ])
                else:
                    diff_est_out[label] = None
            else:
                diff_est_out[label] = None

        # AME
        AME_estimate = gen_ate(
            data, model_coef_dict, None, X, D, Z, Z_X,
            method, treat_type, use_fe, full_moderate,
            weight_col=weight_col, coef_names=coef_names,
        )
        ame_boot = np.array([br["_AME"] for br in boot_results if "_AME" in br])
        ame_sd = np.nanstd(ame_boot, ddof=0) if len(ame_boot) > 0 else 0.0
        ame_ci = np.nanpercentile(ame_boot, [2.5, 97.5]) if len(ame_boot) > 0 else [np.nan, np.nan]
        z_val = AME_estimate / ame_sd if ame_sd > 0 else np.nan
        p_val = 2 * norm.sf(np.abs(z_val))
        avg_estimate = {
            "AME": AME_estimate, "sd": ame_sd, "z-value": z_val,
            "p-value": p_val, "lower": ame_ci[0], "upper": ame_ci[1],
        }

        results = {
            "est_lin": est_lin, "pred_lin": pred_lin, "link_lin": link_lin,
            "diff_estimate": diff_est_out, "vcov_matrix": vcov_matrix, "avg_estimate": avg_estimate,
        }

    return results


# ======================================================================
# Helper: refit model for bootstrap
# ======================================================================

def _fit_model(data_boot, formula_info, method, use_fe, FE, IV, weight_col):
    """Refit the model on a bootstrap sample. Returns (coef_array, vcov, model)."""
    import statsmodels.api as sm
    from statsmodels.genmod.families import Gaussian, Binomial, Poisson
    from statsmodels.genmod.families.links import Logit, Probit
    from .fwl import fwl_demean, iv_fwl

    y = data_boot[formula_info["y_col"]].values
    X_design = data_boot[formula_info["x_cols"]].values
    w = data_boot[weight_col].values if weight_col in data_boot.columns else np.ones(len(data_boot))

    if use_fe and FE is not None:
        FE_mat = data_boot[FE].values.astype(int)

        if IV is not None:
            iv_cols = formula_info.get("iv_cols", [])
            endog_cols = formula_info.get("endog_cols", [])
            exog_cols = formula_info.get("exog_cols", [])
            Y_mat = y
            X_endog = data_boot[endog_cols].values if endog_cols else np.empty((len(data_boot), 0))
            Z_exog = data_boot[exog_cols].values if exog_cols else np.empty((len(data_boot), 0))
            IV_mat = data_boot[iv_cols].values if iv_cols else np.empty((len(data_boot), 0))
            coef, resid, _, _ = iv_fwl(Y_mat, X_endog, Z_exog, IV_mat, FE_mat, w)
        else:
            data_mat = np.column_stack([y, X_design])
            coef, resid, _, _ = fwl_demean(data_mat, FE_mat, w)

        return coef, None, None

    else:
        X_with_const = sm.add_constant(X_design)

        if IV is not None:
            # Simple 2SLS
            iv_cols = formula_info.get("iv_cols", [])
            endog_idx = formula_info.get("endog_idx", [])
            Z_iv = data_boot[iv_cols].values if iv_cols else np.empty((len(data_boot), 0))
            Z_full = np.hstack([X_with_const, Z_iv])
            inv_ZZ = np.linalg.inv(Z_full.T @ Z_full)
            PzX = Z_full @ inv_ZZ @ (Z_full.T @ X_with_const)
            PzY = Z_full @ inv_ZZ @ (Z_full.T @ y)
            coef, _, _, _ = np.linalg.lstsq(PzX, PzY, rcond=None)
            return coef.ravel(), None, None

        if method == "linear":
            model = sm.GLM(y, X_with_const, family=Gaussian(), freq_weights=w).fit()
        elif method == "logit":
            model = sm.GLM(y, X_with_const, family=Binomial(link=Logit()), freq_weights=w).fit()
        elif method == "probit":
            model = sm.GLM(y, X_with_const, family=Binomial(link=Probit()), freq_weights=w).fit()
        elif method == "poisson":
            model = sm.GLM(y, X_with_const, family=Poisson(), freq_weights=w).fit()
        elif method == "nbinom":
            from statsmodels.discrete.discrete_model import NegativeBinomial
            model = NegativeBinomial(y, X_with_const).fit(disp=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        if hasattr(model, 'converged') and not model.converged:
            return None, None, None

        return model.params, model.cov_params(), model

"""Main linear estimator: model fitting, vcov, and variance dispatch.

Direct translation of ``interflex.linear()`` in ``linear.R``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from statsmodels.genmod.families import Gaussian, Binomial, Poisson
from statsmodels.genmod.families.links import Logit as SmLogit, Probit as SmProbit

from ._typing import Method, VarType, VcovType, TreatType, XdistrType
from .result import InterflexResult


def interflex_linear(
    data: pd.DataFrame,
    Y: str, D: str, X: str,
    treat_info: dict, diff_info: dict,
    Z: list[str] | None = None,
    weights: str | None = None,
    full_moderate: bool = False,
    Z_X: list[str] | None = None,
    FE: list[str] | None = None,
    IV: list[str] | None = None,
    neval: int = 50,
    X_eval: np.ndarray | None = None,
    method: Method = "linear",
    vartype: VarType = "simu",
    vcov_type: VcovType = "robust",
    time: str | None = None,
    pairwise: bool = True,
    nboots: int = 200,
    nsimu: int = 1000,
    parallel: bool = False,
    cores: int = 4,
    cl: str | None = None,
    Z_ref: Any = None,
    CI: bool = True,
    figure: bool = True,
    # plotting kwargs
    order: list[str] | None = None,
    subtitles: list[str] | None = None,
    show_subtitles: bool | None = None,
    Xdistr: XdistrType = "histogram",
    main: str | None = None,
    Ylabel: str | None = None,
    Dlabel: str | None = None,
    Xlabel: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    theme_bw: bool = False,
    show_grid: bool = True,
    cex_main: float | None = None,
    cex_sub: float | None = None,
    cex_lab: float | None = None,
    cex_axis: float | None = None,
    interval: np.ndarray | None = None,
    file: str | None = None,
    ncols: int | None = None,
    pool: bool = False,
    color: list[str] | None = None,
    show_all: bool = False,
    scale: float = 1.1,
    height: float = 7.0,
    width: float = 10.0,
) -> InterflexResult:
    """Fit the linear estimator and compute treatment/marginal effects."""
    from .fwl import fwl_demean, iv_fwl
    from .vcov import vcov_cluster, robust_vcov, pcse_vcov
    from .variance import variance_simu, variance_delta, variance_bootstrap

    n = len(data)
    treat_type: TreatType = treat_info["treat_type"]
    diff_values = diff_info.get("diff_values")
    difference_name = diff_info.get("difference_name", [])

    use_fe = FE is not None

    if treat_type == "discrete":
        other_treat = treat_info["other_treat"]
        all_treat = treat_info["all_treat"]
        base = treat_info["base"]
        other_treat_origin = treat_info.get("other_treat_origin", {})
        all_treat_origin = treat_info.get("all_treat_origin", {})
    else:
        D_sample = treat_info["D_sample"]
        base = None

    # Weights
    if weights is None:
        w = np.ones(n)
    else:
        w = data[weights].values.astype(float)
    data = data.copy()
    data["WEIGHTS"] = w

    # ---- Step 1: Evaluation points ----
    X_eval_user = X_eval
    X_eval = np.linspace(data[X].min(), data[X].max(), neval)
    if X_eval_user is not None:
        X_eval = np.sort(np.unique(np.concatenate([X_eval, np.asarray(X_eval_user)])))
    neval = len(X_eval)

    # ---- Step 2: Construct design matrix columns ----
    regressor_names: list[str] = [X]

    if treat_type == "discrete":
        for char in other_treat.values():
            d_col = f"D.{char}"
            dx_col = f"DX.{char}"
            data[d_col] = (data[D] == char).astype(float)
            data[dx_col] = data[d_col] * data[X]
            regressor_names.extend([d_col, dx_col])
    else:
        data["DX"] = data[D] * data[X]
        regressor_names.extend([D, "DX"])

    if Z is not None:
        regressor_names.extend(Z)
        if full_moderate and Z_X is not None:
            regressor_names.extend(Z_X)

    # IV instruments
    iv_instrument_cols: list[str] = []
    if IV is not None:
        for sub_iv in IV:
            xiv_name = f"{X}.{sub_iv}"
            data[xiv_name] = data[sub_iv] * data[X]
            iv_instrument_cols.extend([sub_iv, xiv_name])

    # ---- Step 3: Model fitting ----
    y = data[Y].values.astype(float)
    X_design = data[regressor_names].values.astype(float)

    model = None
    model_coef_dict: dict[str, float] = {}
    coef_names: list[str] = []
    model_df: int = max(n - len(regressor_names) - 1, 1)

    if method == "linear":
        if IV is None:
            if not use_fe:
                X_with_const = sm.add_constant(X_design)
                coef_names = ["(Intercept)"] + regressor_names
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.GLM(y, X_with_const, family=Gaussian(), freq_weights=w).fit()
                if not model.converged:
                    raise RuntimeError("Linear estimator failed to converge.")
                model_df = int(model.df_resid)
            else:
                data_mat = np.column_stack([y, X_design])
                FE_mat = data[FE].values.astype(int)
                coefs, resid, niter, mu = fwl_demean(data_mat, FE_mat, w)
                coef_names = list(regressor_names)

                # Count FE groups for df correction
                n_fe_groups = sum(len(np.unique(FE_mat[:, j])) for j in range(FE_mat.shape[1]))
                model_df = max(n - len(regressor_names) - n_fe_groups, 1)

                # Build model-like object
                class _FEModel:
                    def __init__(self, params, residuals, df_resid, X_dm, y_dm, w, coef_names):
                        self.params = params
                        self.resid = residuals
                        self.df_resid = df_resid
                        self.converged = True
                        self._X_dm = X_dm
                        self._y_dm = y_dm
                        self._w = w
                        self._coef_names = coef_names

                # Store demeaned X for vcov computation
                # Re-demean to get the demeaned design matrix
                data_mat2 = np.column_stack([y, X_design]).copy()
                data_old2 = np.zeros_like(data_mat2)
                diff_val = 100.0
                niter2 = 0
                while diff_val > 1e-5 and niter2 < 50:
                    for col in range(FE_mat.shape[1]):
                        data_wei2 = data_mat2 * w[:, None]
                        for g in np.unique(FE_mat[:, col]):
                            mask = FE_mat[:, col] == g
                            sw = w[mask].sum()
                            if sw > 0:
                                gm = data_wei2[mask].sum(axis=0) / sw
                            else:
                                gm = np.zeros(data_mat2.shape[1])
                            data_mat2[mask] -= gm
                    diff_val = np.abs(data_mat2 - data_old2).sum()
                    data_old2 = data_mat2.copy()
                    niter2 += 1

                sqrt_w = np.sqrt(w)
                X_dm = data_mat2[:, 1:] * sqrt_w[:, None]
                y_dm = data_mat2[:, 0] * sqrt_w

                model = _FEModel(coefs, resid, model_df, X_dm, y_dm, w, coef_names)
        else:
            # IV case
            if not use_fe:
                # Exogenous regressors
                exog_cols = [X]
                if Z is not None:
                    exog_cols.extend(Z)
                    if full_moderate and Z_X is not None:
                        exog_cols.extend(Z_X)

                # Endogenous regressors
                if treat_type == "discrete":
                    endog_cols = []
                    for char in other_treat.values():
                        endog_cols.extend([f"D.{char}", f"DX.{char}"])
                else:
                    endog_cols = [D, "DX"]

                # Instruments
                all_exog = exog_cols + iv_instrument_cols
                X_exog = sm.add_constant(data[all_exog].values.astype(float))
                X_endog = data[endog_cols].values.astype(float)
                X_full = sm.add_constant(data[exog_cols + endog_cols].values.astype(float))

                # 2SLS
                inv_ZZ = np.linalg.inv(X_exog.T @ X_exog)
                PzX = X_exog @ inv_ZZ @ (X_exog.T @ X_full)
                coefs_iv, _, _, _ = np.linalg.lstsq(PzX, y, rcond=None)

                coef_names = ["(Intercept)"] + exog_cols + endog_cols

                class _IVModel:
                    def __init__(self, params, X_full, y, coef_names):
                        self.params = params
                        self.resid = y - X_full @ params
                        self.df_resid = max(len(y) - len(params), 1)
                        self.converged = True
                        self._X = X_full
                        self._coef_names = coef_names

                model = _IVModel(coefs_iv, X_full, y, coef_names)
                model_df = model.df_resid
            else:
                # FE + IV
                exog_cols = [X]
                if Z is not None:
                    exog_cols.extend(Z)
                    if full_moderate and Z_X is not None:
                        exog_cols.extend(Z_X)

                if treat_type == "discrete":
                    endog_cols = []
                    for char in other_treat.values():
                        endog_cols.extend([f"D.{char}", f"DX.{char}"])
                else:
                    endog_cols = [D, "DX"]

                Y_mat = y
                X_endog_mat = data[endog_cols].values.astype(float)
                Z_exog_mat = data[exog_cols].values.astype(float)
                IV_mat = data[iv_instrument_cols].values.astype(float)
                FE_mat = data[FE].values.astype(int)

                coefs_ivfe, resid_ivfe, niter_iv, mu_iv = iv_fwl(
                    Y_mat, X_endog_mat, Z_exog_mat, IV_mat, FE_mat, w,
                )

                coef_names = exog_cols + endog_cols

                class _IVFEModel:
                    def __init__(self, params, residuals, df_resid, coef_names):
                        self.params = params
                        self.resid = residuals
                        self.df_resid = df_resid
                        self.converged = True
                        self._coef_names = coef_names

                n_fe_groups = sum(len(np.unique(FE_mat[:, j])) for j in range(FE_mat.shape[1]))
                model_df = max(n - len(coef_names) - n_fe_groups, 1)
                model = _IVFEModel(coefs_ivfe, resid_ivfe, model_df, coef_names)

    else:  # GLM methods (logit, probit, poisson, nbinom) — no FE, no IV
        X_with_const = sm.add_constant(X_design)
        coef_names = ["(Intercept)"] + regressor_names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "logit":
                model = sm.GLM(y, X_with_const, family=Binomial(link=SmLogit()), freq_weights=w).fit()
            elif method == "probit":
                model = sm.GLM(y, X_with_const, family=Binomial(link=SmProbit()), freq_weights=w).fit()
            elif method == "poisson":
                model = sm.GLM(y, X_with_const, family=Poisson(), freq_weights=w).fit()
            elif method == "nbinom":
                from statsmodels.discrete.discrete_model import NegativeBinomial
                model = NegativeBinomial(y, X_with_const).fit(disp=0)

        if hasattr(model, 'converged') and not model.converged:
            raise RuntimeError("Linear estimator failed to converge.")
        model_df = int(getattr(model, 'df_resid', max(n - len(coef_names), 1)))

    # ---- Step 4: Extract coefficients and vcov ----
    if hasattr(model, 'params'):
        raw_params = model.params
        if isinstance(raw_params, pd.Series):
            raw_params = raw_params.values
        raw_params = np.asarray(raw_params, dtype=float)

        if len(raw_params) == len(coef_names):
            model_coef_array = raw_params
        else:
            # Pad or truncate
            model_coef_array = np.zeros(len(coef_names))
            model_coef_array[:min(len(raw_params), len(coef_names))] = raw_params[:len(coef_names)]
    else:
        model_coef_array = np.zeros(len(coef_names))

    # Replace NaN with 0
    model_coef_array = np.where(np.isnan(model_coef_array), 0.0, model_coef_array)
    model_coef_dict = dict(zip(coef_names, model_coef_array))

    # ---- Vcov estimation ----
    # Helper to extract residuals from various model types
    def _get_residuals(mdl):
        if hasattr(mdl, 'resid_response'):
            return np.asarray(mdl.resid_response)
        if hasattr(mdl, 'resid'):
            return np.asarray(mdl.resid)
        # Fallback: compute manually
        X_f = sm.add_constant(X_design) if "(Intercept)" in coef_names else X_design
        return y - X_f @ model_coef_array

    p = len(coef_names)
    if not use_fe:
        X_for_vcov = sm.add_constant(X_design) if "(Intercept)" in coef_names else X_design
        resid_vec = _get_residuals(model)

        if vcov_type == "homoscedastic":
            if hasattr(model, 'cov_params'):
                raw_vcov = np.asarray(model.cov_params())
            else:
                raw_vcov = np.eye(p)
        elif vcov_type == "robust":
            raw_vcov = robust_vcov(X_for_vcov, resid_vec, len(coef_names))
        elif vcov_type == "cluster":
            raw_vcov = vcov_cluster(X_for_vcov, resid_vec, data[cl].values, len(coef_names))
        elif vcov_type == "pcse":
            raw_vcov = pcse_vcov(resid_vec, X_for_vcov, data[cl].values, data[time].values, pairwise)
            raw_vcov = pcse_vcov(resid_vec, X_for_vcov, data[cl].values, data[time].values, pairwise)
        else:
            raw_vcov = np.eye(p)
    else:
        # FE vcov
        if hasattr(model, '_X_dm'):
            X_dm = model._X_dm
            resid = model.resid
            sqrt_w = np.sqrt(w)
            # Residuals in weighted space
            resid_w = resid * sqrt_w

            if vcov_type == "homoscedastic":
                sigma2 = np.sum(resid_w ** 2) / model_df
                XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
                raw_vcov = sigma2 * XtX_inv
            elif vcov_type == "robust":
                raw_vcov = robust_vcov(X_dm, resid_w, len(coef_names))
            elif vcov_type == "cluster":
                raw_vcov = vcov_cluster(X_dm, resid_w, data[cl].values, len(coef_names))
            else:
                sigma2 = np.sum(resid_w ** 2) / model_df
                XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
                raw_vcov = sigma2 * XtX_inv
        else:
            raw_vcov = np.eye(p)

    # Replace NaN with 0 in vcov
    raw_vcov = np.where(np.isnan(raw_vcov), 0.0, raw_vcov)

    # Expand to full size (coef_names x coef_names)
    model_vcov = np.zeros((p, p))
    vcov_size = min(raw_vcov.shape[0], p)
    model_vcov[:vcov_size, :vcov_size] = raw_vcov[:vcov_size, :vcov_size]

    # Symmetry check
    if not np.allclose(model_vcov, model_vcov.T, atol=1e-6):
        warnings.warn(
            f"vcov_type={vcov_type} leads to unstable standard error; "
            "using homoscedastic SE as fallback."
        )
        if hasattr(model, 'cov_params'):
            model_vcov_fallback = np.asarray(model.cov_params())
            model_vcov_fallback = np.where(np.isnan(model_vcov_fallback), 0.0, model_vcov_fallback)
            model_vcov = np.zeros((p, p))
            s = min(model_vcov_fallback.shape[0], p)
            model_vcov[:s, :s] = model_vcov_fallback[:s, :s]
        else:
            model_vcov = (model_vcov + model_vcov.T) / 2

    # ---- Step 5: Compute TE/ME, predictions, differences, ATE/AME ----
    # Build context for variance functions
    if treat_type == "discrete":
        D_sample_for_var = {v: v for v in other_treat.values()}
    else:
        D_sample_for_var = D_sample

    common_kwargs = dict(
        X=X, D=D, Z=Z, Z_ref=Z_ref, Z_X=Z_X, full_moderate=full_moderate,
        method=method, treat_type=treat_type, use_fe=use_fe, base=base if treat_type == "discrete" else None,
        D_sample=D_sample_for_var if treat_type == "discrete" else D_sample,
        diff_values=diff_values, difference_name=difference_name,
        data=data, weight_col="WEIGHTS",
    )

    if vartype == "simu":
        var_results = variance_simu(
            model_coef=model_coef_dict,
            model_vcov=model_vcov,
            coef_names=coef_names,
            nsimu=nsimu,
            X_eval=X_eval,
            **common_kwargs,
        )
    elif vartype == "delta":
        var_results = variance_delta(
            model_coef=model_coef_dict,
            model_vcov=model_vcov,
            coef_names=coef_names,
            X_eval=X_eval,
            model_df=model_df,
            **common_kwargs,
        )
    elif vartype == "bootstrap":
        # Build formula_info for bootstrap refit
        formula_info = {
            "y_col": Y,
            "x_cols": regressor_names,
            "model_coef": model_coef_dict,
            "model_coef_dict": model_coef_dict,
        }
        if IV is not None:
            formula_info["iv_cols"] = iv_instrument_cols
            if treat_type == "discrete":
                formula_info["endog_cols"] = [
                    c for char in other_treat.values() for c in [f"D.{char}", f"DX.{char}"]
                ]
            else:
                formula_info["endog_cols"] = [D, "DX"]
            exog_cols = [X]
            if Z is not None:
                exog_cols.extend(Z)
                if full_moderate and Z_X is not None:
                    exog_cols.extend(Z_X)
            formula_info["exog_cols"] = exog_cols

        # Remove keys that are passed explicitly to avoid duplicates
        boot_kwargs = {k: v for k, v in common_kwargs.items()
                       if k not in ("data", "weight_col")}
        var_results = variance_bootstrap(
            data=data,
            formula_info=formula_info,
            nboots=nboots,
            cl=cl,
            X_eval=X_eval,
            parallel=parallel,
            cores=cores,
            coef_names=coef_names,
            FE=FE,
            IV=IV,
            vcov_type=vcov_type,
            time=time,
            pairwise=pairwise,
            weight_col=common_kwargs.get("weight_col", "WEIGHTS"),
            **boot_kwargs,
        )
    else:
        raise ValueError(f"Unknown vartype: {vartype}")

    est_lin = var_results["est_lin"]
    pred_lin = var_results["pred_lin"]
    link_lin = var_results["link_lin"]
    diff_estimate = var_results["diff_estimate"]
    vcov_matrix = var_results["vcov_matrix"]
    avg_estimate = var_results["avg_estimate"]

    # ---- Step 6: Histogram and density data ----
    de = None
    de_tr = None
    hist_out = None
    count_tr = None

    try:
        from scipy.stats import gaussian_kde

        x_vals = data[X].values
        if weights is not None:
            de = gaussian_kde(x_vals, weights=w)
        else:
            de = gaussian_kde(x_vals)

        hist_out = np.histogram(x_vals, bins=80)

        if treat_type == "discrete":
            de_tr = {}
            count_tr = {}
            for label, char in all_treat.items():
                mask = data[D] == char
                x_sub = x_vals[mask]
                if len(x_sub) > 1:
                    de_tr[label] = gaussian_kde(x_sub)
                    count_tr[label] = np.histogram(x_sub, bins=hist_out[1])
    except Exception:
        pass

    # ---- Remap keys from internal labels to original labels ----
    if treat_type == "discrete":
        est_lin_mapped = {}
        pred_lin_mapped = {}
        link_lin_mapped = {}
        diff_est_mapped = {}
        vcov_mapped = {}
        avg_est_mapped = {}

        for char_internal, orig_label in other_treat_origin.items():
            if char_internal in est_lin:
                est_lin_mapped[orig_label] = est_lin[char_internal]
            if char_internal in pred_lin:
                pred_lin_mapped[orig_label] = pred_lin[char_internal]
            if char_internal in link_lin:
                link_lin_mapped[orig_label] = link_lin[char_internal]
            if char_internal in diff_estimate:
                diff_est_mapped[orig_label] = diff_estimate[char_internal]
            if char_internal in vcov_matrix:
                vcov_mapped[orig_label] = vcov_matrix[char_internal]
            if char_internal in avg_estimate:
                avg_est_mapped[orig_label] = avg_estimate[char_internal]

        # Base group predictions
        if base in pred_lin:
            base_orig = all_treat_origin.get(base, base)
            pred_lin_mapped[base_orig] = pred_lin[base]
        if base in link_lin:
            base_orig = all_treat_origin.get(base, base)
            link_lin_mapped[base_orig] = link_lin[base]

        est_lin = est_lin_mapped
        pred_lin = pred_lin_mapped
        link_lin = link_lin_mapped
        diff_estimate = diff_est_mapped
        vcov_matrix = vcov_mapped
        avg_estimate = avg_est_mapped

    # Convert diff_estimate arrays to DataFrames
    diff_estimate_df = {}
    for key, val in diff_estimate.items():
        if val is not None and isinstance(val, np.ndarray):
            cols = ["diff.estimate", "sd", "z-value", "p-value", "lower CI(95%)", "upper CI(95%)"]
            rows = difference_name[:val.shape[0]] if difference_name else [f"diff.{i}" for i in range(val.shape[0])]
            diff_estimate_df[key] = pd.DataFrame(val, columns=cols[:val.shape[1]], index=rows)
        else:
            diff_estimate_df[key] = pd.DataFrame()

    # Convert avg_estimate to DataFrames
    if isinstance(avg_estimate, dict):
        avg_df = {}
        for key, val in avg_estimate.items():
            if isinstance(val, dict):
                avg_df[key] = pd.DataFrame([val])
            else:
                avg_df[key] = pd.DataFrame([{"value": val}])
        avg_estimate = avg_df

    # ---- Step 7: Assemble InterflexResult ----
    result = InterflexResult(
        est_lin=est_lin,
        pred_lin=pred_lin,
        link_lin=link_lin,
        diff_estimate=diff_estimate_df,
        vcov_matrix=vcov_matrix,
        avg_estimate=avg_estimate,
        treat_info=treat_info,
        diff_info=diff_info,
        xlabel=Xlabel or X,
        dlabel=Dlabel or D,
        ylabel=Ylabel or Y,
        de=de,
        de_tr=de_tr,
        hist_out=hist_out,
        count_tr=count_tr,
        estimator="linear",
        model_linear=model,
        use_fe=use_fe,
    )

    # ---- Generate figure if requested ----
    if figure:
        try:
            from .plotting import plot_interflex
            fig = plot_interflex(
                result,
                order=order,
                subtitles=subtitles,
                show_subtitles=show_subtitles,
                CI=CI,
                diff_values=diff_values,
                Xdistr=Xdistr,
                main=main,
                Ylabel=Ylabel,
                Dlabel=Dlabel,
                Xlabel=Xlabel,
                xlab=xlab,
                ylab=ylab,
                xlim=xlim,
                ylim=ylim,
                theme_bw=theme_bw,
                show_grid=show_grid,
                cex_main=cex_main,
                cex_sub=cex_sub,
                cex_lab=cex_lab,
                cex_axis=cex_axis,
                interval=interval,
                file=file,
                ncols=ncols,
                pool=pool,
                color=color,
                show_all=show_all,
                scale=scale,
                height=height,
                width=width,
            )
            result.figure = fig
        except Exception as e:
            warnings.warn(f"Plotting failed: {e}")

    return result

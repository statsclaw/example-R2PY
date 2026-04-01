"""Entry point for the interflex linear estimator.

``interflex()`` handles input validation and preprocessing, then delegates
to ``interflex_linear()`` in ``linear.py``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ._typing import Method, VarType, VcovType, TreatType, XdistrType
from .result import InterflexResult


def interflex(
    estimator: str = "linear",
    data: pd.DataFrame | None = None,
    Y: str = "",
    D: str = "",
    X: str = "",
    treat_type: TreatType | None = None,
    base: str | None = None,
    Z: list[str] | None = None,
    IV: list[str] | None = None,
    FE: list[str] | None = None,
    full_moderate: bool = False,
    weights: str | None = None,
    na_rm: bool = False,
    Xunif: bool = False,
    CI: bool = True,
    neval: int = 50,
    X_eval: np.ndarray | None = None,
    method: Method = "linear",
    vartype: VarType = "delta",
    vcov_type: VcovType = "robust",
    time: str | None = None,
    pairwise: bool = True,
    nboots: int = 200,
    nsimu: int = 1000,
    parallel: bool = False,
    cores: int = 4,
    cl: str | None = None,
    Z_ref: np.ndarray | list | None = None,
    D_ref: np.ndarray | list[float] | None = None,
    diff_values: np.ndarray | None = None,
    percentile: bool = False,
    figure: bool = True,
    # Plotting parameters
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
    """Estimate interaction effects using the linear estimator.

    Parameters
    ----------
    See spec.md Section 4.3 for full parameter documentation.

    Returns
    -------
    InterflexResult
        Dataclass containing estimation results, predictions, and optionally a figure.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if data is None:
        raise ValueError("data must be provided.")
    data = pd.DataFrame(data).copy()

    if estimator != "linear":
        raise ValueError("Only estimator='linear' is supported.")

    for col_name, col_label in [(Y, "Y"), (D, "D"), (X, "X")]:
        if not isinstance(col_name, str) or col_name == "":
            raise ValueError(f"{col_label} must be a non-empty string.")
        if col_name not in data.columns:
            raise ValueError(f"{col_label}='{col_name}' not found in data columns.")

    if Z is not None:
        for z in Z:
            if z not in data.columns:
                raise ValueError(f"Z element '{z}' not found in data columns.")

    if FE is not None:
        if method != "linear":
            raise ValueError("FE is only allowed with method='linear'.")
        for fe in FE:
            if fe not in data.columns:
                raise ValueError(f"FE element '{fe}' not found in data columns.")

    if IV is not None:
        if method != "linear":
            raise ValueError("IV is only allowed with method='linear'.")
        if vcov_type == "pcse":
            raise ValueError("IV is not compatible with vcov_type='pcse'.")
        for iv in IV:
            if iv not in data.columns:
                raise ValueError(f"IV element '{iv}' not found in data columns.")

    if method not in ("linear", "logit", "probit", "poisson", "nbinom"):
        raise ValueError(f"method must be one of linear/logit/probit/poisson/nbinom, got '{method}'.")

    if vartype not in ("delta", "simu", "bootstrap"):
        raise ValueError(f"vartype must be one of delta/simu/bootstrap, got '{vartype}'.")

    if vcov_type not in ("robust", "homoscedastic", "cluster", "pcse"):
        raise ValueError(f"vcov_type must be one of robust/homoscedastic/cluster/pcse, got '{vcov_type}'.")

    if vcov_type == "pcse":
        if cl is None or time is None:
            raise ValueError("vcov_type='pcse' requires cl and time.")
        if FE is not None:
            warnings.warn("FE not compatible with vcov_type='pcse'; switching to cluster.")
            vcov_type = "cluster"
        if method != "linear":
            raise ValueError("vcov_type='pcse' is only supported for method='linear'.")

    if vcov_type == "cluster":
        if cl is None:
            raise ValueError("vcov_type='cluster' requires cl.")

    if nboots < 1:
        raise ValueError("nboots must be a positive integer.")
    if nsimu < 1:
        raise ValueError("nsimu must be a positive integer.")

    # ------------------------------------------------------------------
    # 2. Handle missing values
    # ------------------------------------------------------------------
    relevant_cols = [Y, D, X]
    if Z is not None:
        relevant_cols.extend(Z)
    if FE is not None:
        relevant_cols.extend(FE)
    if cl is not None:
        relevant_cols.append(cl)
    if time is not None:
        relevant_cols.append(time)
    if IV is not None:
        relevant_cols.extend(IV)
    if weights is not None:
        relevant_cols.append(weights)

    relevant_cols = list(dict.fromkeys(relevant_cols))  # unique, preserve order

    if na_rm:
        data = data.dropna(subset=relevant_cols).reset_index(drop=True)
    else:
        if data[relevant_cols].isnull().any().any():
            raise ValueError(
                "Data contains missing values. Set na_rm=True to drop rows with NAs."
            )

    # Validate outcome for binary/count methods
    if method in ("logit", "probit"):
        unique_y = data[Y].unique()
        if not set(unique_y).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"method='{method}' requires Y to be binary (0/1).")

    if method in ("poisson", "nbinom"):
        if (data[Y] < 0).any() or not np.allclose(data[Y], data[Y].astype(int)):
            raise ValueError(f"method='{method}' requires Y to be non-negative integers.")

    # ------------------------------------------------------------------
    # 3. Xunif: replace X with rank percentiles
    # ------------------------------------------------------------------
    if Xunif:
        data[X] = data[X].rank(pct=True)

    # ------------------------------------------------------------------
    # 4. Factor covariates: sum-contrast dummies
    # ------------------------------------------------------------------
    Z_original = Z
    Z_X_cols: list[str] | None = None
    dummy_count = 0

    if Z is not None:
        new_Z = []
        for z in Z:
            if data[z].dtype.name == "category" or data[z].dtype == object:
                categories = sorted(data[z].unique())
                n_cat = len(categories)
                if n_cat < 2:
                    continue
                # Sum-contrast coding: n_cat - 1 dummies
                for i in range(n_cat - 1):
                    dummy_name = f"Dummy.Covariate.{dummy_count + 1}"
                    dummy_count += 1
                    data[dummy_name] = (data[z] == categories[i]).astype(float)
                    # Sum-contrast: set last category to -1
                    data.loc[data[z] == categories[-1], dummy_name] = -1.0
                    new_Z.append(dummy_name)
            else:
                new_Z.append(z)
        Z = new_Z

    # Compute default Z_ref
    if Z is not None and Z_ref is None:
        z_ref_dict = {}
        for z in Z:
            if z.startswith("Dummy.Covariate."):
                z_ref_dict[z] = 0.0
            else:
                z_ref_dict[z] = float(data[z].mean())
        Z_ref = z_ref_dict
    elif Z is not None and Z_ref is not None:
        if isinstance(Z_ref, (list, np.ndarray)):
            if len(Z_ref) != len(Z):
                raise ValueError(f"Z_ref length ({len(Z_ref)}) must match Z length ({len(Z)}).")
            Z_ref = dict(zip(Z, Z_ref))

    # ------------------------------------------------------------------
    # 5. Z.X interaction (full_moderate)
    # ------------------------------------------------------------------
    if full_moderate and Z is not None:
        Z_X_cols = []
        for z in Z:
            zx_name = f"{z}.X"
            data[zx_name] = data[z] * data[X]
            Z_X_cols.append(zx_name)

    # ------------------------------------------------------------------
    # 6. Treat type detection and encoding
    # ------------------------------------------------------------------
    if treat_type is None:
        n_unique = data[D].nunique()
        treat_type = "continuous" if n_unique > 5 else "discrete"

    treat_info: dict[str, Any] = {"treat_type": treat_type, "ncols": ncols}
    diff_info: dict[str, Any] = {}

    if treat_type == "discrete":
        all_values = sorted(data[D].unique())
        if base is None:
            base = str(all_values[0])
        else:
            base = str(base)

        # Create internal label mapping
        all_treat = {}  # original_label -> internal_label
        for i, v in enumerate(all_values):
            all_treat[str(v)] = f"Group.{i + 1}"

        other_treat_labels = [v for v in all_values if str(v) != base]
        other_treat = {}  # original_label -> internal_label
        for v in other_treat_labels:
            other_treat[str(v)] = all_treat[str(v)]

        treat_info["base"] = all_treat[base]
        treat_info["all_treat"] = {str(v): all_treat[str(v)] for v in all_values}
        treat_info["other_treat"] = other_treat
        treat_info["all_treat_origin"] = {all_treat[str(v)]: str(v) for v in all_values}
        treat_info["other_treat_origin"] = {other_treat[str(v)]: str(v) for v in other_treat_labels}

        # Recode D to internal labels
        data[D] = data[D].astype(str).map(all_treat)

    else:  # continuous
        if D_ref is not None:
            if isinstance(D_ref, (int, float)):
                D_ref = [D_ref]
            D_ref = list(D_ref)
            if len(D_ref) > 9:
                raise ValueError("D_ref must have at most 9 values.")
        else:
            D_ref = [float(data[D].median())]

        D_sample = {}
        for d in D_ref:
            D_sample[f"D={d:.4g}"] = d
        treat_info["D_sample"] = D_sample

    # ------------------------------------------------------------------
    # 7. diff_values
    # ------------------------------------------------------------------
    if diff_values is not None:
        diff_values = np.asarray(diff_values, dtype=float)
        if len(diff_values) not in (2, 3):
            raise ValueError("diff_values must have 2 or 3 elements.")
    else:
        q25 = float(np.percentile(data[X], 25))
        q50 = float(np.percentile(data[X], 50))
        q75 = float(np.percentile(data[X], 75))
        diff_values = np.array([q25, q50, q75])

    # Build difference names
    if len(diff_values) == 2:
        difference_name = [f"X={diff_values[1]:.3g} vs X={diff_values[0]:.3g}"]
    else:
        difference_name = [
            f"X={diff_values[1]:.3g} vs X={diff_values[0]:.3g}",
            f"X={diff_values[2]:.3g} vs X={diff_values[1]:.3g}",
            f"X={diff_values[2]:.3g} vs X={diff_values[0]:.3g}",
        ]

    diff_info["diff_values"] = diff_values
    diff_info["diff_values_plot"] = diff_values
    diff_info["difference_name"] = difference_name

    # ------------------------------------------------------------------
    # 8. FE/cl/time factorization
    # ------------------------------------------------------------------
    if FE is not None:
        for fe in FE:
            data[fe] = pd.Categorical(data[fe]).codes.astype(int)

    if cl is not None and cl in data.columns:
        # Keep original for clustering, but also create int-coded version
        pass

    if time is not None and time in data.columns:
        pass

    # ------------------------------------------------------------------
    # 9. Call interflex_linear
    # ------------------------------------------------------------------
    from .linear import interflex_linear

    result = interflex_linear(
        data=data,
        Y=Y, D=D, X=X,
        treat_info=treat_info,
        diff_info=diff_info,
        Z=Z,
        weights=weights,
        full_moderate=full_moderate,
        Z_X=Z_X_cols,
        FE=FE,
        IV=IV,
        neval=neval,
        X_eval=X_eval,
        method=method,
        vartype=vartype,
        vcov_type=vcov_type,
        time=time,
        pairwise=pairwise,
        nboots=nboots,
        nsimu=nsimu,
        parallel=parallel,
        cores=cores,
        cl=cl,
        Z_ref=Z_ref,
        CI=CI,
        figure=figure,
        # plotting
        order=order,
        subtitles=subtitles,
        show_subtitles=show_subtitles,
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

    return result

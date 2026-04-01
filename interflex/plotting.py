"""Plotting functions for interflex results.

Translates the R ggplot-based plotting to matplotlib/seaborn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

from ._typing import XdistrType
from .result import InterflexResult


# Default color palette (similar to Dark2)
_DEFAULT_COLORS = [
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666",
]


def plot_interflex(
    result: InterflexResult,
    order: list[str] | None = None,
    subtitles: list[str] | None = None,
    show_subtitles: bool | None = None,
    CI: bool | None = None,
    diff_values: np.ndarray | None = None,
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
) -> matplotlib.figure.Figure:
    """Plot interflex estimation results.

    Parameters
    ----------
    result : InterflexResult
        The estimation result object.
    See spec.md Section 4.10 for full parameter docs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if CI is None:
        CI = True
    if show_subtitles is None:
        show_subtitles = True

    colors = color or _DEFAULT_COLORS
    est_lin = result.est_lin
    treat_info = result.treat_info
    treat_type = treat_info["treat_type"]

    # Determine panel keys and labels
    keys = list(est_lin.keys())
    if order is not None:
        keys = [k for k in order if k in est_lin]

    n_panels = len(keys)
    if n_panels == 0:
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        ax.text(0.5, 0.5, "No estimation results to plot",
                ha='center', va='center', transform=ax.transAxes)
        return fig

    if pool:
        n_panels = 1

    if ncols is None:
        ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols / 3, height * nrows / 2),
                              squeeze=False)

    # Flatten axes
    ax_list = axes.ravel()

    xlabel_text = xlab or Xlabel or result.xlabel
    if treat_type == "discrete":
        ylabel_text = ylab or Ylabel or f"Treatment Effect on {result.ylabel}"
    else:
        ylabel_text = ylab or Ylabel or f"Marginal Effect on {result.ylabel}"

    if pool:
        ax = ax_list[0]
        for i, key in enumerate(keys):
            _plot_one_panel(
                ax, est_lin[key], key, colors[i % len(colors)],
                CI=CI, Xdistr="none", result=result, key=key,
                treat_type=treat_type,
            )
        ax.set_xlabel(xlabel_text)
        ax.set_ylabel(ylabel_text)
        if main:
            ax.set_title(main, fontsize=cex_main or 14)
        ax.legend(fontsize=9)
        if show_grid:
            ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    else:
        for i, key in enumerate(keys):
            if i >= len(ax_list):
                break
            ax = ax_list[i]
            clr = colors[i % len(colors)]

            _plot_one_panel(
                ax, est_lin[key], key, clr,
                CI=CI, Xdistr=Xdistr, result=result, key=key,
                treat_type=treat_type,
                diff_values=diff_values,
            )

            ax.set_xlabel(xlabel_text, fontsize=cex_lab or 11)
            ax.set_ylabel(ylabel_text, fontsize=cex_lab or 11)
            if show_subtitles and subtitles is not None and i < len(subtitles):
                ax.set_title(subtitles[i], fontsize=cex_sub or 12)
            elif show_subtitles:
                ax.set_title(key, fontsize=cex_sub or 12)
            if show_grid:
                ax.grid(True, alpha=0.3)
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if cex_axis:
                ax.tick_params(labelsize=cex_axis)

        # Hide unused axes
        for j in range(i + 1, len(ax_list)):
            ax_list[j].set_visible(False)

    if main and not pool:
        fig.suptitle(main, fontsize=cex_main or 14)

    fig.tight_layout()

    if file is not None:
        fig.savefig(file, dpi=100, bbox_inches='tight')

    return fig


def _plot_one_panel(
    ax: matplotlib.axes.Axes,
    est_table: np.ndarray,
    label: str,
    color: str,
    CI: bool,
    Xdistr: str,
    result: InterflexResult,
    key: str,
    treat_type: str,
    diff_values: np.ndarray | None = None,
):
    """Plot a single panel (one treatment arm or D_ref value)."""
    x_vals = est_table[:, 0]
    estimate = est_table[:, 1]

    # Main line
    ax.plot(x_vals, estimate, color=color, linewidth=1.5, label=label)

    # CI ribbon
    if CI and est_table.shape[1] >= 5:
        lower = est_table[:, 3]
        upper = est_table[:, 4]
        ax.fill_between(x_vals, lower, upper, color=color, alpha=0.2)

        # Uniform CI (if available, columns 5 and 6)
        if est_table.shape[1] >= 7:
            u_lower = est_table[:, 5]
            u_upper = est_table[:, 6]
            ax.plot(x_vals, u_lower, color=color, linewidth=0.8, linestyle='--', alpha=0.7)
            ax.plot(x_vals, u_upper, color=color, linewidth=0.8, linestyle='--', alpha=0.7)

    # Zero line for TE/ME
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')

    # Diff values vertical lines
    if diff_values is not None:
        for dv in diff_values:
            ax.axvline(x=dv, color='gray', linewidth=0.5, linestyle=':')

    # X distribution (histogram at bottom)
    if Xdistr == "histogram" and result.hist_out is not None:
        counts, bin_edges = result.hist_out
        y_range = ax.get_ylim()
        y_span = y_range[1] - y_range[0]
        # Scale histogram to bottom 1/5 of plot
        max_count = counts.max() if counts.max() > 0 else 1
        scaled_counts = counts / max_count * y_span * 0.15 + y_range[0]

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, scaled_counts - y_range[0], bottom=y_range[0],
               width=bin_width, color='gray', alpha=0.3, zorder=0)

    elif Xdistr == "density" and result.de is not None:
        y_range = ax.get_ylim()
        y_span = y_range[1] - y_range[0]
        x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
        density_vals = result.de(x_grid)
        max_d = density_vals.max() if density_vals.max() > 0 else 1
        scaled_density = density_vals / max_d * y_span * 0.15 + y_range[0]
        ax.fill_between(x_grid, y_range[0], scaled_density, color='gray', alpha=0.2, zorder=0)


def plot_interflex_pool(
    result: InterflexResult,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Plot all treatment arms on a single panel."""
    return plot_interflex(result, pool=True, **kwargs)


def plot_predict(
    result: InterflexResult,
    type: str = "response",
    **kwargs,
) -> matplotlib.figure.Figure | None:
    """Plot predicted values or link values."""
    if result.use_fe:
        return None

    if type == "response":
        data_dict = result.pred_lin
    elif type == "link":
        data_dict = result.link_lin
    else:
        raise ValueError(f"type must be 'response' or 'link', got '{type}'.")

    # Create a temporary result with pred/link data in est_lin slot
    temp_result = InterflexResult(
        est_lin=data_dict,
        pred_lin=result.pred_lin,
        link_lin=result.link_lin,
        diff_estimate=result.diff_estimate,
        vcov_matrix=result.vcov_matrix,
        avg_estimate=result.avg_estimate,
        treat_info=result.treat_info,
        diff_info=result.diff_info,
        xlabel=result.xlabel,
        dlabel=result.dlabel,
        ylabel=f"E({result.ylabel})" if type == "response" else f"Link({result.ylabel})",
        de=result.de,
        de_tr=result.de_tr,
        hist_out=result.hist_out,
        count_tr=result.count_tr,
    )

    return plot_interflex(temp_result, **kwargs)

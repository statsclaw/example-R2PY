"""Predict method for InterflexResult."""

from __future__ import annotations

import matplotlib.figure

from .result import InterflexResult


def predict_interflex(
    result: InterflexResult,
    type: str = "response",
    order: list[str] | None = None,
    subtitles: list[str] | None = None,
    show_subtitles: bool | None = None,
    CI: bool | None = None,
    pool: bool = False,
    **plot_kwargs,
) -> matplotlib.figure.Figure | None:
    """Predict method for InterflexResult.

    Parameters
    ----------
    result : InterflexResult
        The estimation result.
    type : str
        ``"response"`` for predicted values or ``"link"`` for link-scale values.

    Returns
    -------
    Figure or None
        None if the model uses fixed effects (cannot predict levels with FE).
    """
    if result.use_fe:
        return None

    from .plotting import plot_predict

    return plot_predict(
        result,
        type=type,
        order=order,
        subtitles=subtitles,
        show_subtitles=show_subtitles,
        CI=CI,
        pool=pool,
        **plot_kwargs,
    )

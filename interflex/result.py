"""InterflexResult dataclass — container for interflex linear estimator output."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.figure


@dataclass
class InterflexResult:
    """Container for interflex linear estimator output.

    Mirrors the R S3 'interflex' class list structure.
    """

    # Core results (keyed by treatment arm label)
    est_lin: dict[str, np.ndarray]  # TE/ME table: (k, 5 or 7)
    pred_lin: dict[str, np.ndarray]  # Predicted values table
    link_lin: dict[str, np.ndarray]  # Link values table
    diff_estimate: dict[str, pd.DataFrame]  # Difference estimates table
    vcov_matrix: dict[str, np.ndarray]  # TE/ME covariance matrix (k x k) per arm
    avg_estimate: dict[str, pd.DataFrame] | pd.DataFrame  # ATE or AME

    # Metadata
    treat_info: dict[str, Any]
    diff_info: dict[str, Any]

    # Labels
    xlabel: str = ""
    dlabel: str = ""
    ylabel: str = ""

    # Distribution data (for plotting)
    de: Any = None  # KDE density of X
    de_tr: Any = None  # Per-treatment KDE (discrete) or None
    hist_out: Any = None  # Histogram data
    count_tr: Any = None  # Per-treatment histogram counts (discrete) or None

    # Tests and diagnostics
    tests: dict[str, Any] = field(default_factory=dict)

    # Model object
    estimator: str = "linear"
    model_linear: Any = None  # Fitted statsmodels model
    use_fe: bool = False

    # Figure
    figure: matplotlib.figure.Figure | None = None

    def predict(
        self, type: str = "response", **plot_kwargs
    ) -> matplotlib.figure.Figure | None:
        """Predict method -- delegates to predict_interflex()."""
        from .predict import predict_interflex

        return predict_interflex(self, type=type, **plot_kwargs)

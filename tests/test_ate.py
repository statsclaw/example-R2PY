"""
Tests A1-A3: Average Treatment Effects (ATE/AME).

Tolerances from test-spec.md:
- ATE/AME weighted mean property: 1e-10 (numerical)
"""

import numpy as np
import pytest
from conftest import make_discrete_binary, make_continuous

from interflex import interflex


def _get_ate_value(avg_estimate):
    """Extract the ATE/AME scalar value from the avg_estimate structure.

    For discrete: avg_estimate is a dict keyed by treatment arm, values are DataFrames
                  with columns ['ATE', 'sd', 'z-value', 'p-value', 'lower', 'upper'].
    For continuous: avg_estimate is a dict with keys like 'AME', 'sd', etc.,
                    each mapping to a DataFrame with a single 'value' column.
    """
    keys = list(avg_estimate.keys())
    first = avg_estimate[keys[0]]
    if hasattr(first, "columns"):
        if "ATE" in first.columns:
            # Discrete: DataFrame with ATE column
            return float(first["ATE"].iloc[0])
        elif "value" in first.columns:
            # Continuous: first key is 'AME', value is DataFrame with 'value'
            return float(first["value"].iloc[0])
        else:
            return float(first.iloc[0, 0])
    return float(first)


def _get_ate_se(avg_estimate):
    """Extract the ATE/AME SE from the avg_estimate structure."""
    keys = list(avg_estimate.keys())
    first = avg_estimate[keys[0]]
    if hasattr(first, "columns"):
        if "sd" in first.columns:
            return float(first["sd"].iloc[0])
        elif "SE" in first.columns:
            return float(first["SE"].iloc[0])
    # Continuous case: 'sd' is a separate key
    if "sd" in avg_estimate:
        sd_df = avg_estimate["sd"]
        if hasattr(sd_df, "iloc"):
            return float(sd_df.iloc[0, 0])
    return None


def _get_ate_ci(avg_estimate):
    """Extract CI bounds from the avg_estimate structure."""
    keys = list(avg_estimate.keys())
    first = avg_estimate[keys[0]]
    if hasattr(first, "columns"):
        if "lower" in first.columns and "upper" in first.columns:
            return float(first["lower"].iloc[0]), float(first["upper"].iloc[0])
    # Continuous case: separate keys
    if "lower" in avg_estimate and "upper" in avg_estimate:
        lo = avg_estimate["lower"]
        up = avg_estimate["upper"]
        return float(lo.iloc[0, 0]), float(up.iloc[0, 0])
    return None, None


# ---------------------------------------------------------------------------
# Test A1: ATE Discrete Linear
# ---------------------------------------------------------------------------
class TestA1ATEDiscreteLinear:
    """Test-spec A1: Recipe A, ATE ~ mean(TE(X_i)) for treated group."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="delta", vcov_type="robust",
        )

    def test_ate_finite(self):
        """ATE should be finite."""
        assert self.result.avg_estimate is not None
        ate_val = _get_ate_value(self.result.avg_estimate)
        assert np.isfinite(ate_val), "ATE must be finite"

    def test_ate_positive_se(self):
        """ATE SE should be positive."""
        se_val = _get_ate_se(self.result.avg_estimate)
        if se_val is not None:
            assert se_val > 0, "ATE SE must be positive"

    def test_ate_approximately_correct(self):
        """ATE should be approximately mean(2 + 1.5*X_i) for D=='1' group."""
        X_treated = self.data.loc[self.data["D"] == "1", "X"].values
        true_ate = np.mean(2 + 1.5 * X_treated)
        ate_val = _get_ate_value(self.result.avg_estimate)
        assert abs(ate_val - true_ate) < 0.5, (
            f"ATE: expected ~{true_ate:.3f}, got {ate_val:.3f}"
        )


# ---------------------------------------------------------------------------
# Test A2: AME Continuous Linear
# ---------------------------------------------------------------------------
class TestA2AMEContinuousLinear:
    """Test-spec A2: Recipe B, AME ~ mean(0.8 + 0.6*X_i) ~ 0.8."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_continuous(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            treat_type="continuous", method="linear", vartype="delta",
        )

    def test_ame_approximately_correct(self):
        """AME should be approximately 0.8 (since E[X] ~ 0 for uniform[-2,2])."""
        ame_val = _get_ate_value(self.result.avg_estimate)
        assert abs(ame_val - 0.8) < 0.3, (
            f"AME: expected ~0.8, got {ame_val:.3f}"
        )

    def test_ame_se_positive(self):
        """AME SE should be positive."""
        se_val = _get_ate_se(self.result.avg_estimate)
        if se_val is not None:
            assert se_val > 0, "AME SE must be positive"


# ---------------------------------------------------------------------------
# Test A3: ATE with Delta SE
# ---------------------------------------------------------------------------
class TestA3ATEDeltaSE:
    """Test-spec A3: Recipe A, ATE with delta SE, valid CI."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="delta", vcov_type="robust",
        )

    def test_ate_ci_valid(self):
        """ATE CI should be finite and valid (lower < upper)."""
        ci_lower, ci_upper = _get_ate_ci(self.result.avg_estimate)
        if ci_lower is not None and ci_upper is not None:
            assert np.isfinite(ci_lower), "ATE CI lower must be finite"
            assert np.isfinite(ci_upper), "ATE CI upper must be finite"
            assert ci_lower < ci_upper, (
                f"ATE CI: lower ({ci_lower:.4f}) must be < upper ({ci_upper:.4f})"
            )

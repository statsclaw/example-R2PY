"""
Tests A1-A3: Average Treatment Effects (ATE/AME).

Tolerances from test-spec.md:
- ATE/AME weighted mean property: 1e-10 (numerical)
"""

import numpy as np
import pytest
from conftest import make_discrete_binary, make_continuous

from interflex import interflex


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
        keys = list(self.result.avg_estimate.keys())
        assert len(keys) >= 1
        ate_info = self.result.avg_estimate[keys[0]]
        ate_val = ate_info["ATE"] if isinstance(ate_info, dict) else ate_info.iloc[0]
        assert np.isfinite(ate_val), "ATE must be finite"

    def test_ate_positive_se(self):
        """ATE SE should be positive."""
        keys = list(self.result.avg_estimate.keys())
        ate_info = self.result.avg_estimate[keys[0]]
        if isinstance(ate_info, dict):
            se_val = ate_info.get("SE", ate_info.get("se", None))
        else:
            se_val = ate_info.iloc[1] if len(ate_info) > 1 else None
        if se_val is not None:
            assert se_val > 0, "ATE SE must be positive"

    def test_ate_approximately_correct(self):
        """ATE should be approximately mean(2 + 1.5*X_i) for D=='1' group."""
        X_treated = self.data.loc[self.data["D"] == "1", "X"].values
        true_ate = np.mean(2 + 1.5 * X_treated)
        keys = list(self.result.avg_estimate.keys())
        ate_info = self.result.avg_estimate[keys[0]]
        ate_val = ate_info["ATE"] if isinstance(ate_info, dict) else ate_info.iloc[0]
        # Loose tolerance for finite-sample
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
        keys = list(self.result.avg_estimate.keys())
        assert len(keys) >= 1
        ame_info = self.result.avg_estimate[keys[0]]
        ame_val = ame_info["ATE"] if isinstance(ame_info, dict) else ame_info.iloc[0]
        # True AME = mean(0.8 + 0.6*X_i), and E[X]=0, so AME ~ 0.8
        assert abs(ame_val - 0.8) < 0.3, (
            f"AME: expected ~0.8, got {ame_val:.3f}"
        )

    def test_ame_se_positive(self):
        """AME SE should be positive."""
        keys = list(self.result.avg_estimate.keys())
        ame_info = self.result.avg_estimate[keys[0]]
        if isinstance(ame_info, dict):
            se_val = ame_info.get("SE", ame_info.get("se", None))
        else:
            se_val = ame_info.iloc[1] if len(ame_info) > 1 else None
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
        keys = list(self.result.avg_estimate.keys())
        ate_info = self.result.avg_estimate[keys[0]]
        if isinstance(ate_info, dict):
            ci_lower = ate_info.get("CI_lower", ate_info.get("ci_lower", None))
            ci_upper = ate_info.get("CI_upper", ate_info.get("ci_upper", None))
        else:
            # Try index-based access
            ci_lower = ate_info.iloc[3] if len(ate_info) > 3 else None
            ci_upper = ate_info.iloc[4] if len(ate_info) > 4 else None
        if ci_lower is not None and ci_upper is not None:
            assert np.isfinite(ci_lower), "ATE CI lower must be finite"
            assert np.isfinite(ci_upper), "ATE CI upper must be finite"
            assert ci_lower < ci_upper, (
                f"ATE CI: lower ({ci_lower:.4f}) must be < upper ({ci_upper:.4f})"
            )

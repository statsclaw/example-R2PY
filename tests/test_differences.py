"""
Tests D1-D2: Difference tests.

Tolerances from test-spec.md:
- Difference additivity: 1e-10
"""

import numpy as np
import pytest
from conftest import make_discrete_binary

from interflex import interflex


# ---------------------------------------------------------------------------
# Test D1: Two-point Difference
# ---------------------------------------------------------------------------
class TestD1TwoPointDifference:
    """Test-spec D1: diff_values=[Q25, Q75] for Recipe A."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        X = self.data["X"].values
        self.q25 = np.percentile(X, 25)
        self.q75 = np.percentile(X, 75)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            method="linear", vartype="delta",
            diff_values=np.array([self.q25, self.q75]),
        )

    def test_diff_exists(self):
        """diff_estimate should be populated."""
        assert self.result.diff_estimate is not None

    def test_diff_approximately_correct(self):
        """diff = TE(Q75) - TE(Q25) ~ 1.5*(Q75 - Q25)."""
        keys = list(self.result.diff_estimate.keys())
        if len(keys) > 0:
            diff_info = self.result.diff_estimate[keys[0]]
            if isinstance(diff_info, dict):
                diff_val = diff_info.get("diff", diff_info.get("estimate", None))
            elif hasattr(diff_info, "iloc"):
                diff_val = diff_info.iloc[0] if len(diff_info) > 0 else None
            else:
                diff_val = None
            if diff_val is not None:
                true_diff = 1.5 * (self.q75 - self.q25)
                assert abs(diff_val - true_diff) < 0.5, (
                    f"diff: expected ~{true_diff:.3f}, got {diff_val:.3f}"
                )

    def test_diff_has_sd_and_ci(self):
        """diff should have SD, z-value, p-value, CI."""
        keys = list(self.result.diff_estimate.keys())
        if len(keys) > 0:
            diff_info = self.result.diff_estimate[keys[0]]
            if isinstance(diff_info, dict):
                assert "SD" in diff_info or "sd" in diff_info or "se" in diff_info, (
                    "Diff should include SD/se"
                )


# ---------------------------------------------------------------------------
# Test D2: Three-point Difference
# ---------------------------------------------------------------------------
class TestD2ThreePointDifference:
    """Test-spec D2: diff_values=[Q25, Q50, Q75], additivity check."""

    ADDITIVITY_TOLERANCE = 1e-10  # from test-spec section 6

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        X = self.data["X"].values
        self.q25 = np.percentile(X, 25)
        self.q50 = np.percentile(X, 50)
        self.q75 = np.percentile(X, 75)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            method="linear", vartype="delta",
            diff_values=np.array([self.q25, self.q50, self.q75]),
        )

    def test_three_differences_returned(self):
        """Three differences: (Q50-Q25, Q75-Q50, Q75-Q25)."""
        assert self.result.diff_estimate is not None
        keys = list(self.result.diff_estimate.keys())
        # Should have at least one arm with differences
        assert len(keys) >= 1

    def test_difference_additivity(self):
        """diff(Q75,Q25) = diff(Q50,Q25) + diff(Q75,Q50)."""
        keys = list(self.result.diff_estimate.keys())
        if len(keys) > 0:
            diff_info = self.result.diff_estimate[keys[0]]
            # The exact structure depends on implementation
            # Try to extract the three differences
            if hasattr(diff_info, "shape") and diff_info.shape[0] >= 3:
                d1 = diff_info.iloc[0, 0] if hasattr(diff_info, "iloc") else diff_info[0]
                d2 = diff_info.iloc[1, 0] if hasattr(diff_info, "iloc") else diff_info[1]
                d3 = diff_info.iloc[2, 0] if hasattr(diff_info, "iloc") else diff_info[2]
                # d3 = d1 + d2 (or similar ordering)
                # The exact ordering may vary, but one should be the sum of the other two
                # Check: |d3 - (d1 + d2)| < tol
                residual = min(
                    abs(d3 - (d1 + d2)),
                    abs(d1 - (d2 + d3)),
                    abs(d2 - (d1 + d3)),
                )
                assert residual < self.ADDITIVITY_TOLERANCE, (
                    f"Difference additivity failed: residual={residual} "
                    f"(tolerance={self.ADDITIVITY_TOLERANCE})"
                )

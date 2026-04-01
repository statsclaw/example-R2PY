"""
Tests R1-R2: InterflexResult structure.
"""

import pytest
from conftest import make_discrete_binary

from interflex import interflex


# ---------------------------------------------------------------------------
# Test R1: Result Has All Expected Fields
# ---------------------------------------------------------------------------
class TestR1ResultFields:
    """Test-spec R1: Result must have all expected attributes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="delta", vcov_type="robust",
        )

    def test_has_est_lin(self):
        assert hasattr(self.result, "est_lin")

    def test_has_pred_lin(self):
        assert hasattr(self.result, "pred_lin")

    def test_has_link_lin(self):
        assert hasattr(self.result, "link_lin")

    def test_has_diff_estimate(self):
        assert hasattr(self.result, "diff_estimate")

    def test_has_vcov_matrix(self):
        assert hasattr(self.result, "vcov_matrix")

    def test_has_avg_estimate(self):
        assert hasattr(self.result, "avg_estimate")

    def test_has_treat_info(self):
        assert hasattr(self.result, "treat_info")

    def test_has_estimator(self):
        assert hasattr(self.result, "estimator")

    def test_estimator_value(self):
        assert self.result.estimator == "linear"


# ---------------------------------------------------------------------------
# Test R2: est_lin Structure
# ---------------------------------------------------------------------------
class TestR2EstLinStructure:
    """Test-spec R2: est_lin structure for discrete binary."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            base="0", method="linear", vartype="delta",
        )

    def test_one_non_base_arm(self):
        """Discrete binary with base='0' should have one non-base arm."""
        assert len(self.result.est_lin) == 1, (
            f"Expected 1 arm, got {len(self.result.est_lin)}"
        )

    def test_te_table_columns(self):
        """TE table should have >= 5 columns: X, TE, sd, lower, upper."""
        key = list(self.result.est_lin.keys())[0]
        te_table = self.result.est_lin[key]
        assert te_table.shape[1] >= 5, (
            f"TE table should have >= 5 columns, got {te_table.shape[1]}"
        )

    def test_te_table_rows(self):
        """TE table should have at least one evaluation point."""
        key = list(self.result.est_lin.keys())[0]
        te_table = self.result.est_lin[key]
        assert te_table.shape[0] > 0, "TE table should have > 0 rows"

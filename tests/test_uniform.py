"""
Tests U1-U3: Uniform confidence intervals.

Tolerances from test-spec.md:
- Uniform CI vs pointwise CI: uniform width >= pointwise width
"""

import numpy as np
import pytest
from conftest import make_discrete_binary

from interflex import interflex


# ---------------------------------------------------------------------------
# Test U1: Bootstrap Uniform CI -- Coverage Property
# ---------------------------------------------------------------------------
class TestU1BootstrapUniformCI:
    """Test-spec U1: vartype=bootstrap, nboots=200. Uniform CI >= pointwise CI."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            method="linear", vartype="bootstrap", nboots=200,
        )

    def test_uniform_wider_than_pointwise(self):
        """Uniform CI must be wider than or equal to pointwise CI at every point."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        # Expect 7 columns: X, TE, sd, lower, upper, uniform_lower, uniform_upper
        if te_table.shape[1] >= 7:
            lower_pw = te_table[:, 3]
            upper_pw = te_table[:, 4]
            lower_uni = te_table[:, 5]
            upper_uni = te_table[:, 6]
            assert np.all(lower_uni <= lower_pw + 1e-10), (
                "Uniform lower must be <= pointwise lower"
            )
            assert np.all(upper_uni >= upper_pw - 1e-10), (
                "Uniform upper must be >= pointwise upper"
            )

    def test_result_has_seven_columns(self):
        """Result table should have 7 columns when uniform CI is computed."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        assert te_table.shape[1] >= 7, (
            f"Expected >= 7 columns (with uniform CI), got {te_table.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Test U2: Delta Uniform CI -- Width Property
# ---------------------------------------------------------------------------
class TestU2DeltaUniformCI:
    """Test-spec U2: vartype=delta. Uniform critical value > 1.96."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            method="linear", vartype="delta",
        )

    def test_uniform_wider_than_pointwise(self):
        """Uniform CI should be wider than pointwise CI at every point."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        if te_table.shape[1] >= 7:
            lower_pw = te_table[:, 3]
            upper_pw = te_table[:, 4]
            lower_uni = te_table[:, 5]
            upper_uni = te_table[:, 6]
            pw_width = upper_pw - lower_pw
            uni_width = upper_uni - lower_uni
            assert np.all(uni_width >= pw_width - 1e-10), (
                "Uniform CI width must be >= pointwise CI width"
            )

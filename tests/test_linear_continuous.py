"""
Test N2: Linear continuous treatment -- marginal effect recovery.

Tolerances from test-spec.md:
- ME point estimates: abs < 0.2
"""

import numpy as np
import pytest
from conftest import make_continuous

from interflex import interflex


class TestN2LinearContinuousME:
    """Test-spec N2: Recipe B, treat_type=continuous, method=linear, vartype=delta."""

    TOLERANCE = 0.2  # from test-spec section 2.2/N2

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_continuous(n=500, seed=42)
        self.result = interflex(
            "linear",
            self.data,
            "Y",
            "D",
            "X",
            treat_type="continuous",
            method="linear",
            vartype="delta",
        )

    def test_me_at_x0(self):
        """ME at x=0 should be ~0.8 (true value)."""
        keys = list(self.result.est_lin.keys())
        assert len(keys) >= 1
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table.iloc[:, 0].values
        te_vals = te_table.iloc[:, 1].values
        idx_0 = np.argmin(np.abs(x_vals - 0.0))
        me_at_0 = te_vals[idx_0]
        assert abs(me_at_0 - 0.8) < self.TOLERANCE, (
            f"ME at x=0: expected ~0.8, got {me_at_0} (tolerance={self.TOLERANCE})"
        )

    def test_me_at_x1(self):
        """ME at x=1 should be ~1.4 (true: 0.8 + 0.6*1)."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table.iloc[:, 0].values
        te_vals = te_table.iloc[:, 1].values
        idx_1 = np.argmin(np.abs(x_vals - 1.0))
        me_at_1 = te_vals[idx_1]
        assert abs(me_at_1 - 1.4) < self.TOLERANCE, (
            f"ME at x=1: expected ~1.4, got {me_at_1} (tolerance={self.TOLERANCE})"
        )

    def test_me_linear_in_x(self):
        """ME should be linear in x (R-squared > 0.9999)."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table.iloc[:, 0].values
        te_vals = te_table.iloc[:, 1].values
        # Fit line: TE = a + b*x
        coeffs = np.polyfit(x_vals, te_vals, 1)
        te_fitted = np.polyval(coeffs, x_vals)
        ss_res = np.sum((te_vals - te_fitted) ** 2)
        ss_tot = np.sum((te_vals - np.mean(te_vals)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        assert r_squared > 0.9999, (
            f"ME should be linear in x, R-squared={r_squared:.6f} (need > 0.9999)"
        )

    def test_ses_positive(self):
        """All SEs must be positive."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        sd_vals = te_table.iloc[:, 2].values
        assert np.all(sd_vals > 0), "All SEs must be positive"

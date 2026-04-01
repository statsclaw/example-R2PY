"""
Tests N1, N3, N4, N5, N6: Linear discrete treatment effects.

Tolerances from test-spec.md:
- Coefficient recovery (n=500): abs < 0.3
- TE/ME point estimates: abs < 0.3
- TE bounds (logit): -1 < TE < 1
- Predictions (poisson): > 0
"""

import numpy as np
import pytest
from conftest import (
    make_discrete_binary,
    make_discrete_multi,
    make_binary_outcome,
    make_count_outcome,
    make_panel_fe,
)

from interflex import interflex


# ---------------------------------------------------------------------------
# Test N1: Linear Discrete -- Coefficient Recovery
# ---------------------------------------------------------------------------
class TestN1LinearDiscreteCoefficientRecovery:
    """Test-spec N1: Recipe A, method=linear, vartype=delta, vcov_type=robust."""

    TOLERANCE = 0.3  # from test-spec section 4

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result = interflex(
            "linear",
            self.data,
            "Y",
            "D",
            "X",
            Z=["Z1"],
            method="linear",
            vartype="delta",
            vcov_type="robust",
        )

    def test_te_at_x0(self):
        """TE at x=0 should be ~2.0 (true value)."""
        keys = list(self.result.est_lin.keys())
        assert len(keys) >= 1, "Should have at least one treatment arm"
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table[:, 0]
        te_vals = te_table[:, 1]
        idx_0 = np.argmin(np.abs(x_vals - 0.0))
        te_at_0 = te_vals[idx_0]
        assert abs(te_at_0 - 2.0) < self.TOLERANCE, (
            f"TE at x=0: expected ~2.0, got {te_at_0} "
            f"(tolerance={self.TOLERANCE})"
        )

    def test_te_at_x1(self):
        """TE at x=1 should be ~3.5 (true value: 2 + 1.5*1)."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table[:, 0]
        te_vals = te_table[:, 1]
        idx_1 = np.argmin(np.abs(x_vals - 1.0))
        te_at_1 = te_vals[idx_1]
        assert abs(te_at_1 - 3.5) < self.TOLERANCE, (
            f"TE at x=1: expected ~3.5, got {te_at_1} "
            f"(tolerance={self.TOLERANCE})"
        )

    def test_te_monotonically_increasing(self):
        """TE should be monotonically increasing (delta_1=1.5 > 0)."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table[:, 0]
        te_vals = te_table[:, 1]
        sorted_idx = np.argsort(x_vals)
        te_sorted = te_vals[sorted_idx]
        diffs = np.diff(te_sorted)
        assert np.all(diffs >= -1e-10), (
            "TE should be monotonically increasing"
        )

    def test_ses_positive_finite(self):
        """All SEs must be positive and finite."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        sd_vals = te_table[:, 2]
        assert np.all(sd_vals > 0), "All SEs must be positive"
        assert np.all(np.isfinite(sd_vals)), "All SEs must be finite"

    def test_ci_contains_true_te_at_x0(self):
        """95% CI at x=0 should contain true TE=2.0."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table[:, 0]
        idx_0 = np.argmin(np.abs(x_vals - 0.0))
        lower = te_table[idx_0, 3]
        upper = te_table[idx_0, 4]
        assert lower < 2.0 < upper, (
            f"95% CI [{lower:.4f}, {upper:.4f}] should contain true TE=2.0"
        )


# ---------------------------------------------------------------------------
# Test N3: Fixed Effects -- TE Recovery
# ---------------------------------------------------------------------------
class TestN3FixedEffects:
    """Test-spec N3: Recipe C, method=linear, vartype=delta, FE=[unit,period]."""

    TOLERANCE = 0.3

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_panel_fe(n_units=50, n_periods=10, seed=42)
        self.result = interflex(
            "linear",
            self.data,
            "Y",
            "D",
            "X",
            FE=["unit", "period"],
            method="linear",
            vartype="delta",
        )

    def test_te_at_x0(self):
        """TE at x=0 should be ~2.0."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table[:, 0]
        te_vals = te_table[:, 1]
        idx_0 = np.argmin(np.abs(x_vals - 0.0))
        te_at_0 = te_vals[idx_0]
        assert abs(te_at_0 - 2.0) < self.TOLERANCE, (
            f"TE at x=0 with FE: expected ~2.0, got {te_at_0}"
        )

    def test_te_at_x1(self):
        """TE at x=1 should be ~3.5."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table[:, 0]
        te_vals = te_table[:, 1]
        idx_1 = np.argmin(np.abs(x_vals - 1.0))
        te_at_1 = te_vals[idx_1]
        assert abs(te_at_1 - 3.5) < self.TOLERANCE, (
            f"TE at x=1 with FE: expected ~3.5, got {te_at_1}"
        )

    def test_use_fe_flag(self):
        """result.use_fe should be True."""
        assert self.result.use_fe is True, "use_fe should be True when FE specified"


# ---------------------------------------------------------------------------
# Test N4: Multi-arm Discrete -- Per-arm TE
# ---------------------------------------------------------------------------
class TestN4MultiArmDiscrete:
    """Test-spec N4: Recipe D, base='A', method=linear, vartype=delta."""

    TOLERANCE_B = 0.3
    TOLERANCE_C = 0.5

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_multi(n=600, seed=42)
        self.result = interflex(
            "linear",
            self.data,
            "Y",
            "D",
            "X",
            base="A",
            method="linear",
            vartype="delta",
        )

    def test_treatment_arms_present(self):
        """est_lin should have keys for arms B and C (excluding base A)."""
        keys = set(self.result.est_lin.keys())
        assert "B" in keys, "Treatment arm 'B' should be in est_lin"
        assert "C" in keys, "Treatment arm 'C' should be in est_lin"
        assert "A" not in keys, "Base arm 'A' should NOT be in est_lin"

    def test_te_b_at_x0(self):
        """TE_B at x=0 should be ~1.0."""
        te_table = self.result.est_lin["B"]
        x_vals = te_table[:, 0]
        te_vals = te_table[:, 1]
        idx_0 = np.argmin(np.abs(x_vals - 0.0))
        te_at_0 = te_vals[idx_0]
        assert abs(te_at_0 - 1.0) < self.TOLERANCE_B, (
            f"TE_B at x=0: expected ~1.0, got {te_at_0}"
        )

    def test_te_c_at_x0(self):
        """TE_C at x=0 should be ~3.0."""
        te_table = self.result.est_lin["C"]
        x_vals = te_table[:, 0]
        te_vals = te_table[:, 1]
        idx_0 = np.argmin(np.abs(x_vals - 0.0))
        te_at_0 = te_vals[idx_0]
        assert abs(te_at_0 - 3.0) < self.TOLERANCE_C, (
            f"TE_C at x=0: expected ~3.0, got {te_at_0}"
        )


# ---------------------------------------------------------------------------
# Test N5: Logit Method -- TE Bounds
# ---------------------------------------------------------------------------
class TestN5LogitBounds:
    """Test-spec N5: Recipe E, method=logit, vartype=delta."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_binary_outcome(n=1000, seed=42)
        self.result = interflex(
            "linear",
            self.data,
            "Y",
            "D",
            "X",
            method="logit",
            vartype="delta",
        )

    def test_te_bounded(self):
        """All TE values must be in (-1, 1)."""
        for key, te_table in self.result.est_lin.items():
            te_vals = te_table[:, 1]
            assert np.all(te_vals > -1.0), (
                f"Arm {key}: TE values must be > -1"
            )
            assert np.all(te_vals < 1.0), (
                f"Arm {key}: TE values must be < 1"
            )

    def test_predictions_bounded(self):
        """All predicted values must be in (0, 1)."""
        for key, pred_table in self.result.pred_lin.items():
            pred_vals = pred_table[:, 1]
            assert np.all(pred_vals > 0), (
                f"Arm {key}: predictions must be > 0"
            )
            assert np.all(pred_vals < 1), (
                f"Arm {key}: predictions must be < 1"
            )

    def test_ses_positive(self):
        """SEs must be positive."""
        for key, te_table in self.result.est_lin.items():
            sd_vals = te_table[:, 2]
            assert np.all(sd_vals > 0), f"Arm {key}: SEs must be positive"


# ---------------------------------------------------------------------------
# Test N6: Poisson Method -- Non-negative Predictions
# ---------------------------------------------------------------------------
class TestN6PoissonPredictions:
    """Test-spec N6: Recipe F, method=poisson, vartype=delta."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_count_outcome(n=1000, seed=42)
        self.result = interflex(
            "linear",
            self.data,
            "Y",
            "D",
            "X",
            method="poisson",
            vartype="delta",
        )

    def test_predictions_positive(self):
        """All predicted values must be > 0 (exp link)."""
        for key, pred_table in self.result.pred_lin.items():
            pred_vals = pred_table[:, 1]
            assert np.all(pred_vals > 0), (
                f"Arm {key}: Poisson predictions must be > 0"
            )

    def test_ses_positive(self):
        """SEs must be positive."""
        for key, te_table in self.result.est_lin.items():
            sd_vals = te_table[:, 2]
            assert np.all(sd_vals > 0), f"Arm {key}: SEs must be positive"

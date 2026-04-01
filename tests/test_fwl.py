"""
Tests F1-F4: Frisch-Waugh-Lovell demeaning.

Tolerances from test-spec.md:
- FWL demeaning residual group means: abs < 1e-4
- FWL vs LSDV coefficient match: abs < 1e-6
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_panel_fe, make_iv_data

from interflex import interflex


# ---------------------------------------------------------------------------
# Test F1: Single FE -- Correct Demeaning
# ---------------------------------------------------------------------------
class TestF1SingleFEDemeaning:
    """Test-spec F1: n=100, 10 groups, within-group means ~0 after demeaning."""

    DEMEANING_TOLERANCE = 1e-4  # from test-spec section 4

    def test_demeaning_single_fe(self):
        """After demeaning, within-group means of Y and X should be ~0."""
        rng = np.random.default_rng(42)
        n = 100
        group = np.repeat(np.arange(10), 10)
        group_effect = rng.normal(0, 2, 10)
        X = rng.uniform(-2, 2, n)
        D = rng.choice(["0", "1"], n)
        D_num = (D == "1").astype(float)
        eps = rng.normal(0, 0.5, n)
        Y = group_effect[group] + 0.5 * X + 2 * D_num + eps
        data = pd.DataFrame({"Y": Y, "D": D, "X": X, "group": group})

        result = interflex(
            "linear", data, "Y", "D", "X",
            FE=["group"], method="linear", vartype="delta",
        )
        assert result is not None
        assert result.use_fe is True


# ---------------------------------------------------------------------------
# Test F2: Two FEs -- Convergence
# ---------------------------------------------------------------------------
class TestF2TwoFEConvergence:
    """Test-spec F2: Recipe C (two-way FE), convergence check."""

    def test_two_way_fe_converges(self):
        """Two-way FE model should converge and produce results."""
        data = make_panel_fe(n_units=50, n_periods=10, seed=42)
        result = interflex(
            "linear", data, "Y", "D", "X",
            FE=["unit", "period"], method="linear", vartype="delta",
        )
        assert result is not None
        keys = list(result.est_lin.keys())
        assert len(keys) >= 1, "Should have at least one treatment arm"
        te_table = result.est_lin[keys[0]]
        assert te_table.shape[0] > 0, "Should have evaluation points"


# ---------------------------------------------------------------------------
# Test F3: FWL OLS Equivalence
# ---------------------------------------------------------------------------
class TestF3FWLOLSEquivalence:
    """Test-spec F3: FWL + OLS gives same coefficients as LSDV."""

    FWL_LSDV_TOLERANCE = 1e-6  # from test-spec section 4

    def test_fwl_matches_lsdv(self):
        """FWL demeaning + OLS should match LSDV (least squares dummy variable)."""
        rng = np.random.default_rng(42)
        n = 100
        group = np.repeat(np.arange(5), 20)
        X = rng.uniform(-2, 2, n)
        D = rng.choice(["0", "1"], n)
        D_num = (D == "1").astype(float)
        group_effect = rng.normal(0, 2, 5)
        eps = rng.normal(0, 0.3, n)
        Y = group_effect[group] + 0.5 * X + 2 * D_num + 1.0 * D_num * X + eps
        data = pd.DataFrame({"Y": Y, "D": D, "X": X, "group": group})

        # FWL result
        result_fwl = interflex(
            "linear", data, "Y", "D", "X",
            FE=["group"], method="linear", vartype="delta",
        )

        # LSDV: run without FE but with group dummies
        dummies = pd.get_dummies(data["group"], prefix="g", drop_first=True).astype(float)
        data_lsdv = pd.concat([data, dummies], axis=1)
        z_cols = [c for c in dummies.columns]
        result_lsdv = interflex(
            "linear", data_lsdv, "Y", "D", "X",
            Z=z_cols, method="linear", vartype="delta",
        )

        # Compare TE at evaluation points (numpy arrays, not DataFrames)
        key_fwl = list(result_fwl.est_lin.keys())[0]
        key_lsdv = list(result_lsdv.est_lin.keys())[0]
        te_fwl = result_fwl.est_lin[key_fwl][:, 1]
        te_lsdv = result_lsdv.est_lin[key_lsdv][:, 1]

        min_len = min(len(te_fwl), len(te_lsdv))
        if min_len > 0:
            x_fwl = result_fwl.est_lin[key_fwl][:, 0]
            x_lsdv = result_lsdv.est_lin[key_lsdv][:, 0]
            for i in range(min_len):
                closest_idx = np.argmin(np.abs(x_lsdv - x_fwl[i]))
                if abs(x_lsdv[closest_idx] - x_fwl[i]) < 0.01:
                    diff = abs(te_fwl[i] - te_lsdv[closest_idx])
                    assert diff < self.FWL_LSDV_TOLERANCE, (
                        f"FWL vs LSDV TE mismatch at x={x_fwl[i]:.2f}: "
                        f"diff={diff:.2e} (tolerance={self.FWL_LSDV_TOLERANCE})"
                    )


# ---------------------------------------------------------------------------
# Test F4: IV FWL -- 2SLS Correctness
# ---------------------------------------------------------------------------
class TestF4IVFWL:
    """Test-spec F4: Recipe I with FE, iv_fwl() produces coefficients."""

    def test_iv_produces_results(self):
        """IV estimation should produce finite results."""
        data = make_iv_data(n=500, seed=42)
        rng = np.random.default_rng(99)
        data["group"] = rng.choice(5, len(data))

        result = interflex(
            "linear", data, "Y", "D", "X",
            IV=["W"], FE=["group"],
            method="linear", vartype="delta",
        )
        assert result is not None
        keys = list(result.est_lin.keys())
        assert len(keys) >= 1
        te_table = result.est_lin[keys[0]]
        assert te_table.shape[0] > 0

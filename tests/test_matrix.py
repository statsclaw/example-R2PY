"""
Test Matrix: method x vartype x vcov_type x treat_type combinations.

From test-spec section 3. Each combination must run without error and produce valid output.
"""

import numpy as np
import pytest
from conftest import (
    make_discrete_binary,
    make_continuous,
    make_binary_outcome,
    make_count_outcome,
    make_clustered,
    make_panel_pcse,
    make_panel_fe,
    make_iv_data,
)

from interflex import interflex


# ---------------------------------------------------------------------------
# HIGH priority combinations
# ---------------------------------------------------------------------------

class TestMatrixLinearDeltaRobustDiscrete:
    """method=linear, vartype=delta, vcov_type=robust, treat_type=discrete (Recipe A)."""

    def test_runs(self):
        data = make_discrete_binary(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="delta", vcov_type="robust")
        assert result is not None
        assert len(result.est_lin) >= 1


class TestMatrixLinearDeltaHomoDiscrete:
    """method=linear, vartype=delta, vcov_type=homoscedastic, treat_type=discrete (Recipe A)."""

    def test_runs(self):
        data = make_discrete_binary(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="delta", vcov_type="homoscedastic")
        assert result is not None


class TestMatrixLinearDeltaClusterDiscrete:
    """method=linear, vartype=delta, vcov_type=cluster, treat_type=discrete (Recipe G)."""

    def test_runs(self):
        data = make_clustered(n_clusters=30, cluster_size=20, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="delta", vcov_type="cluster", cl="cl")
        assert result is not None


class TestMatrixLinearSimuRobustDiscrete:
    """method=linear, vartype=simu, vcov_type=robust, treat_type=discrete (Recipe A)."""

    def test_runs(self):
        data = make_discrete_binary(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="simu", vcov_type="robust")
        assert result is not None


class TestMatrixLinearSimuRobustContinuous:
    """method=linear, vartype=simu, vcov_type=robust, treat_type=continuous (Recipe B)."""

    def test_runs(self):
        data = make_continuous(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           treat_type="continuous",
                           method="linear", vartype="simu", vcov_type="robust")
        assert result is not None


class TestMatrixLinearBootstrapRobustDiscrete:
    """method=linear, vartype=bootstrap, vcov_type=robust, treat_type=discrete (Recipe A)."""

    def test_runs(self):
        data = make_discrete_binary(n=200, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="bootstrap", vcov_type="robust",
                           nboots=50)
        assert result is not None


class TestMatrixLogitDeltaRobustDiscrete:
    """method=logit, vartype=delta, vcov_type=robust, treat_type=discrete (Recipe E)."""

    def test_runs(self):
        data = make_binary_outcome(n=1000, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="logit", vartype="delta", vcov_type="robust")
        assert result is not None


class TestMatrixLinearDeltaRobustDiscreteFE:
    """method=linear, vartype=delta, vcov_type=robust, discrete + FE (Recipe C)."""

    def test_runs(self):
        data = make_panel_fe(n_units=50, n_periods=10, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           FE=["unit", "period"],
                           method="linear", vartype="delta", vcov_type="robust")
        assert result is not None


# ---------------------------------------------------------------------------
# MEDIUM priority combinations
# ---------------------------------------------------------------------------

class TestMatrixLinearDeltaPCSEContinuous:
    """method=linear, vartype=delta, vcov_type=pcse, continuous (Recipe H)."""

    def test_runs(self):
        data = make_panel_pcse(n_units=30, n_periods=20, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           treat_type="continuous",
                           method="linear", vartype="delta", vcov_type="pcse",
                           cl="unit", time="period")
        assert result is not None


class TestMatrixLinearBootstrapClusterDiscrete:
    """method=linear, vartype=bootstrap, vcov_type=cluster, discrete (Recipe G)."""

    def test_runs(self):
        data = make_clustered(n_clusters=30, cluster_size=20, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="bootstrap", vcov_type="cluster",
                           cl="cl", nboots=50)
        assert result is not None


class TestMatrixLogitSimuRobustDiscrete:
    """method=logit, vartype=simu, vcov_type=robust, discrete (Recipe E)."""

    def test_runs(self):
        data = make_binary_outcome(n=1000, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="logit", vartype="simu", vcov_type="robust")
        assert result is not None


class TestMatrixProbitDeltaRobustDiscrete:
    """method=probit, vartype=delta, vcov_type=robust, discrete (Recipe E)."""

    def test_runs(self):
        data = make_binary_outcome(n=1000, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="probit", vartype="delta", vcov_type="robust")
        assert result is not None


class TestMatrixPoissonDeltaRobustDiscrete:
    """method=poisson, vartype=delta, vcov_type=robust, discrete (Recipe F)."""

    def test_runs(self):
        data = make_count_outcome(n=1000, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="poisson", vartype="delta", vcov_type="robust")
        assert result is not None


class TestMatrixLinearDeltaClusterDiscreteFE:
    """method=linear, vartype=delta, vcov_type=cluster, discrete + FE (Recipe C)."""

    def test_runs(self):
        data = make_panel_fe(n_units=50, n_periods=10, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           FE=["unit", "period"],
                           method="linear", vartype="delta", vcov_type="cluster",
                           cl="unit")
        assert result is not None


class TestMatrixLinearDeltaRobustDiscreteIV:
    """method=linear, vartype=delta, vcov_type=robust, discrete + IV (Recipe I)."""

    def test_runs(self):
        data = make_iv_data(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           IV=["W"],
                           method="linear", vartype="delta", vcov_type="robust")
        assert result is not None


# ---------------------------------------------------------------------------
# LOW priority combinations
# ---------------------------------------------------------------------------

class TestMatrixNbinomDeltaRobustDiscrete:
    """method=nbinom, vartype=delta, vcov_type=robust, discrete (Recipe F)."""

    def test_runs(self):
        data = make_count_outcome(n=1000, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="nbinom", vartype="delta", vcov_type="robust")
        assert result is not None


# ---------------------------------------------------------------------------
# Property-based invariants (from test-spec section 6)
# ---------------------------------------------------------------------------

class TestPropertyTELinearity:
    """Invariant 1: TE(x) = gamma + delta*x must be exactly linear (R^2 > 0.9999)."""

    def test_te_linearity_discrete(self):
        data = make_discrete_binary(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="delta")
        keys = list(result.est_lin.keys())
        te_table = result.est_lin[keys[0]]
        x_vals = te_table.iloc[:, 0].values
        te_vals = te_table.iloc[:, 1].values
        coeffs = np.polyfit(x_vals, te_vals, 1)
        te_fitted = np.polyval(coeffs, x_vals)
        ss_res = np.sum((te_vals - te_fitted) ** 2)
        ss_tot = np.sum((te_vals - np.mean(te_vals)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        assert r_sq > 0.9999, f"TE linearity: R^2 = {r_sq:.6f} (need > 0.9999)"


class TestPropertyVcovPSD:
    """Invariant 8: All eigenvalues of model_vcov >= -1e-10."""

    def test_vcov_psd(self):
        data = make_discrete_binary(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="delta", vcov_type="robust")
        vcov = result.vcov_matrix
        eigvals = np.linalg.eigvalsh(vcov)
        assert np.all(eigvals >= -1e-10), (
            f"Vcov not PSD: min eigenvalue = {eigvals.min()}"
        )


class TestPropertyDeltaCISymmetry:
    """Invariant 7: For linear method, CI is symmetric around point estimate."""

    SYMMETRY_TOLERANCE = 1e-10  # from test-spec section 6

    def test_ci_symmetry(self):
        data = make_discrete_binary(n=500, seed=42)
        result = interflex("linear", data, "Y", "D", "X",
                           method="linear", vartype="delta")
        keys = list(result.est_lin.keys())
        te_table = result.est_lin[keys[0]]
        te = te_table.iloc[:, 1].values
        lower = te_table.iloc[:, 3].values
        upper = te_table.iloc[:, 4].values
        upper_gap = upper - te
        lower_gap = te - lower
        max_asym = np.max(np.abs(upper_gap - lower_gap))
        assert max_asym < self.SYMMETRY_TOLERANCE, (
            f"CI asymmetry: {max_asym} (tolerance={self.SYMMETRY_TOLERANCE})"
        )

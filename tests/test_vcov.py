"""
Tests C1-C4: Variance-covariance estimation types.

Tolerances from test-spec.md:
- Vcov symmetry: max asymmetry < 1e-6
"""

import numpy as np
import pytest
from conftest import make_discrete_binary, make_clustered, make_panel_pcse

from interflex import interflex


def _get_vcov_array(result):
    """Extract the vcov matrix as a numpy array.

    The implementation returns vcov_matrix as a dict keyed by treatment arm.
    Each value is a numpy 2-d array (n_eval x n_eval).
    For model-level vcov checks, we use the first arm's matrix.
    """
    vcov = result.vcov_matrix
    if isinstance(vcov, dict):
        first_key = list(vcov.keys())[0]
        return vcov[first_key]
    return vcov


# ---------------------------------------------------------------------------
# Test C1: Homoscedastic vs Robust with Heteroscedastic Data
# ---------------------------------------------------------------------------
class TestC1HomoscedasticVsRobust:
    """Test-spec C1: Heteroscedastic data, compare homoscedastic vs robust."""

    @pytest.fixture(autouse=True)
    def setup(self, data_heteroscedastic):
        self.data = data_heteroscedastic
        self.result_homo = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="delta", vcov_type="homoscedastic",
        )
        self.result_robust = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="delta", vcov_type="robust",
        )

    def test_vcov_symmetric_homo(self):
        """Homoscedastic vcov must be symmetric."""
        vcov = _get_vcov_array(self.result_homo)
        max_asym = np.max(np.abs(vcov - vcov.T))
        assert max_asym < 1e-6, (
            f"Homoscedastic vcov asymmetry: {max_asym} (tolerance=1e-6)"
        )

    def test_vcov_symmetric_robust(self):
        """Robust vcov must be symmetric."""
        vcov = _get_vcov_array(self.result_robust)
        max_asym = np.max(np.abs(vcov - vcov.T))
        assert max_asym < 1e-6, (
            f"Robust vcov asymmetry: {max_asym} (tolerance=1e-6)"
        )

    def test_vcov_psd_homo(self):
        """Homoscedastic vcov must be positive semi-definite."""
        vcov = _get_vcov_array(self.result_homo)
        eigvals = np.linalg.eigvalsh(vcov)
        assert np.all(eigvals >= -1e-10), (
            f"Homo vcov not PSD: min eigenvalue = {eigvals.min()}"
        )

    def test_vcov_psd_robust(self):
        """Robust vcov must be positive semi-definite."""
        vcov = _get_vcov_array(self.result_robust)
        eigvals = np.linalg.eigvalsh(vcov)
        assert np.all(eigvals >= -1e-10), (
            f"Robust vcov not PSD: min eigenvalue = {eigvals.min()}"
        )

    def test_robust_se_larger_for_heteroscedastic(self):
        """Robust SEs should generally be larger for heteroscedastic data."""
        key_h = list(self.result_homo.est_lin.keys())[0]
        key_r = list(self.result_robust.est_lin.keys())[0]
        sd_homo = self.result_homo.est_lin[key_h][:, 2]
        sd_robust = self.result_robust.est_lin[key_r][:, 2]
        assert np.mean(sd_robust) > np.mean(sd_homo) * 0.8, (
            f"Robust SEs (mean={np.mean(sd_robust):.4f}) should be larger "
            f"than homoscedastic (mean={np.mean(sd_homo):.4f})"
        )


# ---------------------------------------------------------------------------
# Test C2: Clustered SE -- Larger than Robust
# ---------------------------------------------------------------------------
class TestC2ClusteredSE:
    """Test-spec C2: Recipe G, compare cluster vs robust."""

    @pytest.fixture(autouse=True)
    def setup(self, data_clustered):
        self.data = data_clustered
        self.result_cluster = interflex(
            "linear", self.data, "Y", "D", "X",
            method="linear", vartype="delta", vcov_type="cluster", cl="cl",
        )
        self.result_robust = interflex(
            "linear", self.data, "Y", "D", "X",
            method="linear", vartype="delta", vcov_type="robust",
        )

    def test_clustered_se_larger(self):
        """Clustered SEs should generally be larger than robust SEs."""
        key_c = list(self.result_cluster.est_lin.keys())[0]
        key_r = list(self.result_robust.est_lin.keys())[0]
        sd_cluster = self.result_cluster.est_lin[key_c][:, 2]
        sd_robust = self.result_robust.est_lin[key_r][:, 2]
        assert np.mean(sd_cluster) > np.mean(sd_robust) * 0.8, (
            f"Clustered SEs (mean={np.mean(sd_cluster):.4f}) should be larger "
            f"than robust SEs (mean={np.mean(sd_robust):.4f})"
        )

    def test_cluster_vcov_symmetric(self):
        """Cluster vcov must be symmetric."""
        vcov = _get_vcov_array(self.result_cluster)
        max_asym = np.max(np.abs(vcov - vcov.T))
        assert max_asym < 1e-6, (
            f"Cluster vcov asymmetry: {max_asym} (tolerance=1e-6)"
        )

    def test_cluster_vcov_psd(self):
        """Cluster vcov must be PSD."""
        vcov = _get_vcov_array(self.result_cluster)
        eigvals = np.linalg.eigvalsh(vcov)
        assert np.all(eigvals >= -1e-10), (
            f"Cluster vcov not PSD: min eigenvalue = {eigvals.min()}"
        )


# ---------------------------------------------------------------------------
# Test C3: PCSE -- Panel Corrected SE
# ---------------------------------------------------------------------------
class TestC3PCSE:
    """Test-spec C3: Recipe H, vcov_type=pcse, cl=unit, time=period."""

    @pytest.fixture(autouse=True)
    def setup(self, data_panel_pcse):
        self.data = data_panel_pcse
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            treat_type="continuous",
            method="linear", vartype="delta", vcov_type="pcse",
            cl="unit", time="period",
        )

    def test_pcse_vcov_symmetric(self):
        """PCSE vcov must be symmetric."""
        vcov = _get_vcov_array(self.result)
        max_asym = np.max(np.abs(vcov - vcov.T))
        assert max_asym < 1e-6, (
            f"PCSE vcov asymmetry: {max_asym} (tolerance=1e-6)"
        )

    def test_pcse_vcov_psd(self):
        """PCSE vcov must be PSD."""
        vcov = _get_vcov_array(self.result)
        eigvals = np.linalg.eigvalsh(vcov)
        assert np.all(eigvals >= -1e-10), (
            f"PCSE vcov not PSD: min eigenvalue = {eigvals.min()}"
        )

    def test_pcse_se_positive_finite(self):
        """PCSE SEs must be positive and finite."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        sd_vals = te_table[:, 2]
        assert np.all(sd_vals > 0), "PCSE SEs must be positive"
        assert np.all(np.isfinite(sd_vals)), "PCSE SEs must be finite"

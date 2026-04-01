"""
Tests V1-V4: Variance estimation methods.

Tolerances from test-spec.md:
- Delta vs simulation SD ratio: within factor of 2
- Bootstrap vs delta SD ratio: within factor of 3
- CI validity: lower < estimate < upper
"""

import numpy as np
import pytest
from conftest import make_discrete_binary, make_continuous

from interflex import interflex


# ---------------------------------------------------------------------------
# Test V1: Simulation Variance -- SD Consistency
# ---------------------------------------------------------------------------
class TestV1SimulationVariance:
    """Test-spec V1: Recipe A, compare vartype=simu vs vartype=delta."""

    SD_RATIO_TOLERANCE = 2.0  # within factor of 2

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=500, seed=42)
        self.result_delta = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="delta", vcov_type="robust",
        )
        self.result_simu = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="simu", vcov_type="robust",
        )

    def test_sd_consistency(self):
        """Simulation SDs and delta SDs should agree within factor of 2."""
        key_d = list(self.result_delta.est_lin.keys())[0]
        key_s = list(self.result_simu.est_lin.keys())[0]
        sd_delta = self.result_delta.est_lin[key_d].iloc[:, 2].values
        sd_simu = self.result_simu.est_lin[key_s].iloc[:, 2].values
        # Check ratio
        ratio = sd_simu / sd_delta
        assert np.all(ratio > 1.0 / self.SD_RATIO_TOLERANCE), (
            f"Simu/delta SD ratio too small: min={ratio.min():.4f}"
        )
        assert np.all(ratio < self.SD_RATIO_TOLERANCE), (
            f"Simu/delta SD ratio too large: max={ratio.max():.4f}"
        )

    def test_simu_ci_valid(self):
        """Simulation CIs should satisfy lower < estimate < upper."""
        key = list(self.result_simu.est_lin.keys())[0]
        te_table = self.result_simu.est_lin[key]
        te = te_table.iloc[:, 1].values
        lower = te_table.iloc[:, 3].values
        upper = te_table.iloc[:, 4].values
        assert np.all(lower < te), "CI lower must be < TE estimate"
        assert np.all(te < upper), "CI upper must be > TE estimate"


# ---------------------------------------------------------------------------
# Test V2: Bootstrap Variance -- SD Consistency
# ---------------------------------------------------------------------------
class TestV2BootstrapVariance:
    """Test-spec V2: Recipe A (n=200), vartype=bootstrap, nboots=100."""

    SD_RATIO_TOLERANCE = 3.0  # within factor of 3

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=200, seed=42)
        self.result_delta = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="delta", vcov_type="robust",
        )
        self.result_boot = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="bootstrap", vcov_type="robust",
            nboots=100,
        )

    def test_bootstrap_sd_positive_finite(self):
        """Bootstrap SDs must be positive and finite."""
        key = list(self.result_boot.est_lin.keys())[0]
        sd = self.result_boot.est_lin[key].iloc[:, 2].values
        assert np.all(sd > 0), "Bootstrap SDs must be positive"
        assert np.all(np.isfinite(sd)), "Bootstrap SDs must be finite"

    def test_bootstrap_ci_contains_estimate(self):
        """Bootstrap CIs must contain the point estimate."""
        key = list(self.result_boot.est_lin.keys())[0]
        te_table = self.result_boot.est_lin[key]
        te = te_table.iloc[:, 1].values
        lower = te_table.iloc[:, 3].values
        upper = te_table.iloc[:, 4].values
        assert np.all(lower < te), "Bootstrap CI lower must be < estimate"
        assert np.all(te < upper), "Bootstrap CI upper must be > estimate"

    def test_bootstrap_delta_sd_same_magnitude(self):
        """Bootstrap SDs should be in the same order of magnitude as delta SDs."""
        key_d = list(self.result_delta.est_lin.keys())[0]
        key_b = list(self.result_boot.est_lin.keys())[0]
        sd_delta = self.result_delta.est_lin[key_d].iloc[:, 2].values
        sd_boot = self.result_boot.est_lin[key_b].iloc[:, 2].values
        ratio = sd_boot / sd_delta
        assert np.all(ratio > 1.0 / self.SD_RATIO_TOLERANCE), (
            f"Boot/delta SD ratio too small: min={ratio.min():.4f}"
        )
        assert np.all(ratio < self.SD_RATIO_TOLERANCE), (
            f"Boot/delta SD ratio too large: max={ratio.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Test V3: Delta Method -- Analytical Properties
# ---------------------------------------------------------------------------
class TestV3DeltaMethodProperties:
    """Test-spec V3: Recipe B continuous, vartype=delta, method=linear."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_continuous(n=500, seed=42)
        self.result = interflex(
            "linear", self.data, "Y", "D", "X",
            treat_type="continuous", method="linear", vartype="delta",
        )

    def test_se_increases_with_abs_x(self):
        """SE should generally increase as |x| increases (due to DX term)."""
        keys = list(self.result.est_lin.keys())
        te_table = self.result.est_lin[keys[0]]
        x_vals = te_table.iloc[:, 0].values
        sd_vals = te_table.iloc[:, 2].values
        # SE at extreme |x| should be >= SE near 0 on average
        near_zero = np.abs(x_vals) < 0.5
        far_from_zero = np.abs(x_vals) > 1.5
        if np.any(near_zero) and np.any(far_from_zero):
            mean_se_near = np.mean(sd_vals[near_zero])
            mean_se_far = np.mean(sd_vals[far_from_zero])
            assert mean_se_far >= mean_se_near * 0.8, (
                f"SE far from zero ({mean_se_far:.4f}) should be >= "
                f"SE near zero ({mean_se_near:.4f}) * 0.8"
            )


# ---------------------------------------------------------------------------
# Test V4: Simulation Variance -- Reproducibility
# ---------------------------------------------------------------------------
class TestV4SimulationReproducibility:
    """Test-spec V4: Same setup twice with same seed -> identical results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = make_discrete_binary(n=200, seed=42)

    def test_reproducibility(self):
        """Two calls with same seed must produce identical results."""
        result1 = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="simu", vcov_type="robust",
        )
        result2 = interflex(
            "linear", self.data, "Y", "D", "X", Z=["Z1"],
            method="linear", vartype="simu", vcov_type="robust",
        )
        key1 = list(result1.est_lin.keys())[0]
        key2 = list(result2.est_lin.keys())[0]
        te1 = result1.est_lin[key1].iloc[:, 1].values
        te2 = result2.est_lin[key2].iloc[:, 1].values
        np.testing.assert_array_equal(te1, te2, err_msg="Simulation should be reproducible")

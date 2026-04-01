"""
Tests I1-I8: Input validation.

All should raise ValueError with descriptive messages.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_discrete_binary

from interflex import interflex


# ---------------------------------------------------------------------------
# Test I1: Missing Column Raises ValueError
# ---------------------------------------------------------------------------
class TestI1MissingColumn:
    def test_missing_y_column(self, data_discrete_binary):
        with pytest.raises(ValueError):
            interflex("linear", data_discrete_binary, "NONEXISTENT", "D", "X")

    def test_missing_d_column(self, data_discrete_binary):
        with pytest.raises(ValueError):
            interflex("linear", data_discrete_binary, "Y", "NONEXISTENT", "X")

    def test_missing_x_column(self, data_discrete_binary):
        with pytest.raises(ValueError):
            interflex("linear", data_discrete_binary, "Y", "D", "NONEXISTENT")


# ---------------------------------------------------------------------------
# Test I2: Invalid Method Raises ValueError
# ---------------------------------------------------------------------------
class TestI2InvalidMethod:
    def test_invalid_method(self, data_discrete_binary):
        with pytest.raises(ValueError):
            interflex("linear", data_discrete_binary, "Y", "D", "X", method="invalid")


# ---------------------------------------------------------------------------
# Test I3: FE with Non-linear Method Raises ValueError
# ---------------------------------------------------------------------------
class TestI3FEWithNonlinearMethod:
    def test_fe_with_logit(self):
        data = make_discrete_binary(n=100, seed=42)
        data["unit"] = np.repeat(np.arange(10), 10)
        with pytest.raises(ValueError):
            interflex("linear", data, "Y", "D", "X", FE=["unit"], method="logit")

    def test_fe_with_probit(self):
        data = make_discrete_binary(n=100, seed=42)
        data["unit"] = np.repeat(np.arange(10), 10)
        with pytest.raises(ValueError):
            interflex("linear", data, "Y", "D", "X", FE=["unit"], method="probit")


# ---------------------------------------------------------------------------
# Test I4: Cluster Without cl Raises ValueError
# ---------------------------------------------------------------------------
class TestI4ClusterWithoutCl:
    def test_cluster_no_cl(self, data_discrete_binary):
        with pytest.raises(ValueError):
            interflex("linear", data_discrete_binary, "Y", "D", "X", vcov_type="cluster")


# ---------------------------------------------------------------------------
# Test I5: PCSE Requires cl and time
# ---------------------------------------------------------------------------
class TestI5PCSERequirements:
    def test_pcse_no_cl_no_time(self, data_discrete_binary):
        with pytest.raises(ValueError):
            interflex("linear", data_discrete_binary, "Y", "D", "X", vcov_type="pcse")

    def test_pcse_no_time(self, data_discrete_binary):
        data = data_discrete_binary.copy()
        data["cl"] = np.repeat(np.arange(50), 10)
        with pytest.raises(ValueError):
            interflex("linear", data, "Y", "D", "X", vcov_type="pcse", cl="cl")


# ---------------------------------------------------------------------------
# Test I6: Binary Outcome for Logit
# ---------------------------------------------------------------------------
class TestI6BinaryOutcomeLogit:
    def test_non_binary_y_for_logit(self):
        data_bad = pd.DataFrame({
            "Y": [0, 1, 2],
            "D": ["a", "a", "b"],
            "X": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError):
            interflex("linear", data_bad, "Y", "D", "X", method="logit")


# ---------------------------------------------------------------------------
# Test I7: NA Handling
# ---------------------------------------------------------------------------
class TestI7NAHandling:
    def test_na_rm_false_raises(self):
        data_na = make_discrete_binary()
        data_na.loc[0, "Y"] = np.nan
        with pytest.raises(ValueError):
            interflex("linear", data_na, "Y", "D", "X", na_rm=False)

    def test_na_rm_true_succeeds(self):
        data_na = make_discrete_binary()
        data_na.loc[0, "Y"] = np.nan
        result = interflex("linear", data_na, "Y", "D", "X", na_rm=True)
        assert result is not None


# ---------------------------------------------------------------------------
# Test I8: diff_values Length
# ---------------------------------------------------------------------------
class TestI8DiffValuesLength:
    def test_diff_values_length_1(self, data_discrete_binary):
        with pytest.raises(ValueError):
            interflex(
                "linear", data_discrete_binary, "Y", "D", "X",
                diff_values=np.array([1.0]),
            )

"""
Test R3: Predict method.
"""

import pytest
from conftest import make_discrete_binary, make_panel_fe

from interflex import interflex


# ---------------------------------------------------------------------------
# Test R3: Predict Method Works
# ---------------------------------------------------------------------------
class TestR3PredictMethod:
    """Test-spec R3: result.predict() should work."""

    def test_predict_non_fe(self, data_discrete_binary):
        """predict() on non-FE result should return a Figure or predictions."""
        result = interflex(
            "linear", data_discrete_binary, "Y", "D", "X",
            method="linear", vartype="delta", figure=False,
        )
        # predict should be callable
        pred = result.predict(type="response")
        # For non-FE: may return a Figure or predictions object
        # Just verify it does not raise an error

    def test_predict_fe_returns_none(self):
        """predict() on FE result should return None."""
        data = make_panel_fe(n_units=50, n_periods=10, seed=42)
        result = interflex(
            "linear", data, "Y", "D", "X",
            FE=["unit", "period"],
            method="linear", vartype="delta", figure=False,
        )
        pred = result.predict(type="response")
        # For FE case: returns None (since predictions are 0)
        assert pred is None, "predict() with FE should return None"

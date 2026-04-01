"""
Tests P1-P3: Plotting functionality.
"""

import os
import tempfile

import matplotlib
import matplotlib.figure
import pytest
from conftest import make_discrete_binary, make_discrete_multi

from interflex import interflex

# Use non-interactive backend for testing
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Test P1: Figure Is Created
# ---------------------------------------------------------------------------
class TestP1FigureCreated:
    """Test-spec P1: figure=True produces a matplotlib Figure."""

    def test_figure_created(self, data_discrete_binary):
        result = interflex(
            "linear", data_discrete_binary, "Y", "D", "X",
            method="linear", vartype="delta", figure=True,
        )
        assert result.figure is not None, "figure should not be None when figure=True"
        assert isinstance(result.figure, matplotlib.figure.Figure), (
            f"figure should be a matplotlib Figure, got {type(result.figure)}"
        )


# ---------------------------------------------------------------------------
# Test P2: Pool Plot
# ---------------------------------------------------------------------------
class TestP2PoolPlot:
    """Test-spec P2: Multi-arm discrete with pool=True."""

    def test_pool_plot(self, data_discrete_multi):
        result = interflex(
            "linear", data_discrete_multi, "Y", "D", "X",
            base="A", figure=True, pool=True,
            method="linear", vartype="delta",
        )
        assert result.figure is not None, "Pool plot figure should not be None"


# ---------------------------------------------------------------------------
# Test P3: Save to File
# ---------------------------------------------------------------------------
class TestP3SaveToFile:
    """Test-spec P3: Saving figure to a file."""

    def test_save_figure(self, data_discrete_binary):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            result = interflex(
                "linear", data_discrete_binary, "Y", "D", "X",
                method="linear", vartype="delta",
                figure=True, file=fname,
            )
            assert os.path.exists(fname), f"Figure file should exist at {fname}"
            assert os.path.getsize(fname) > 0, "Figure file should not be empty"
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

"""
Shared fixtures and synthetic data generators for interflex test suite.

All generators use fixed seeds for reproducibility.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Recipe A: Simple Discrete Treatment (Binary)
# ---------------------------------------------------------------------------
def make_discrete_binary(n=500, seed=42):
    """
    True DGP: Y = 1 + 0.5*X + 2*(D=="1") + 1.5*(D=="1")*X + 0.3*Z1 + eps
    True TE(x) = 2 + 1.5*x
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    D = rng.choice([0, 1], n, p=[0.5, 0.5]).astype(str)
    Z1 = rng.normal(0, 1, n)
    D_num = (D == "1").astype(float)
    eps = rng.normal(0, 0.5, n)
    Y = 1 + 0.5 * X + 2 * D_num + 1.5 * D_num * X + 0.3 * Z1 + eps
    return pd.DataFrame({"Y": Y, "D": D, "X": X, "Z1": Z1})


# ---------------------------------------------------------------------------
# Recipe B: Simple Continuous Treatment
# ---------------------------------------------------------------------------
def make_continuous(n=500, seed=42):
    """
    True DGP: Y = 1 + 0.5*X + 0.8*D + 0.6*D*X + eps
    True ME(x) = 0.8 + 0.6*x
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    D = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    Y = 1 + 0.5 * X + 0.8 * D + 0.6 * D * X + eps
    return pd.DataFrame({"Y": Y, "D": D, "X": X})


# ---------------------------------------------------------------------------
# Recipe C: Panel Data with Fixed Effects
# ---------------------------------------------------------------------------
def make_panel_fe(n_units=50, n_periods=10, seed=42):
    rng = np.random.default_rng(seed)
    n = n_units * n_periods
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    unit_fe = rng.normal(0, 2, n_units)
    period_fe = rng.normal(0, 1, n_periods)
    X = rng.uniform(-2, 2, n)
    D = rng.choice(["0", "1"], n, p=[0.5, 0.5])
    D_num = (D == "1").astype(float)
    eps = rng.normal(0, 0.5, n)
    Y = unit_fe[unit] + period_fe[period] + 0.5 * X + 2 * D_num + 1.5 * D_num * X + eps
    return pd.DataFrame({"Y": Y, "D": D, "X": X, "unit": unit, "period": period})


# ---------------------------------------------------------------------------
# Recipe D: Multi-arm Discrete Treatment (3 arms)
# ---------------------------------------------------------------------------
def make_discrete_multi(n=600, seed=42):
    """
    Base = "A". TE_B(x) = 1 + 0.5*x, TE_C(x) = 3 + 2*x
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    D = rng.choice(["A", "B", "C"], n, p=[1 / 3, 1 / 3, 1 / 3])
    eps = rng.normal(0, 0.5, n)
    Y = 1 + 0.5 * X + eps
    Y[D == "B"] += 1 + 0.5 * X[D == "B"]
    Y[D == "C"] += 3 + 2 * X[D == "C"]
    return pd.DataFrame({"Y": Y, "D": D, "X": X})


# ---------------------------------------------------------------------------
# Recipe E: Binary Outcome (Logit/Probit)
# ---------------------------------------------------------------------------
def make_binary_outcome(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    D = rng.choice(["0", "1"], n)
    D_num = (D == "1").astype(float)
    eta = -0.5 + 0.3 * X + 1.0 * D_num + 0.5 * D_num * X
    p = 1 / (1 + np.exp(-eta))
    Y = rng.binomial(1, p, n)
    return pd.DataFrame({"Y": Y, "D": D, "X": X})


# ---------------------------------------------------------------------------
# Recipe F: Count Outcome (Poisson)
# ---------------------------------------------------------------------------
def make_count_outcome(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 2, n)
    D = rng.choice(["0", "1"], n)
    D_num = (D == "1").astype(float)
    eta = 0.5 + 0.3 * X + 0.5 * D_num + 0.2 * D_num * X
    mu = np.exp(eta)
    Y = rng.poisson(mu, n)
    return pd.DataFrame({"Y": Y, "D": D, "X": X})


# ---------------------------------------------------------------------------
# Recipe G: Clustered Data
# ---------------------------------------------------------------------------
def make_clustered(n_clusters=30, cluster_size=20, seed=42):
    rng = np.random.default_rng(seed)
    n = n_clusters * cluster_size
    cl = np.repeat(np.arange(n_clusters), cluster_size)
    cluster_effect = rng.normal(0, 1, n_clusters)
    X = rng.uniform(-2, 2, n)
    D = rng.choice(["0", "1"], n)
    D_num = (D == "1").astype(float)
    eps = rng.normal(0, 0.3, n) + cluster_effect[cl]
    Y = 1 + 0.5 * X + 2 * D_num + 1.5 * D_num * X + eps
    return pd.DataFrame({"Y": Y, "D": D, "X": X, "cl": cl})


# ---------------------------------------------------------------------------
# Recipe H: Panel Data for PCSE
# ---------------------------------------------------------------------------
def make_panel_pcse(n_units=30, n_periods=20, seed=42):
    rng = np.random.default_rng(seed)
    n = n_units * n_periods
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    X = rng.uniform(-2, 2, n)
    D = rng.normal(0, 1, n)  # continuous treatment
    eps = rng.normal(0, 0.5, n)
    Y = 1 + 0.5 * X + 0.8 * D + 0.6 * D * X + eps
    return pd.DataFrame({"Y": Y, "D": D, "X": X, "unit": unit, "period": period})


# ---------------------------------------------------------------------------
# Recipe I: IV Data
# ---------------------------------------------------------------------------
def make_iv_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    W = rng.normal(0, 1, n)  # instrument
    v = rng.normal(0, 1, n)
    D_num = 0.5 * W + 0.3 * W * X + v
    D = (D_num > 0).astype(str)  # make discrete
    eps = 0.5 * v + rng.normal(0, 0.3, n)  # correlated with v
    D_dum = (D == "True").astype(float)
    Y = 1 + 0.5 * X + 2 * D_dum + 1.5 * D_dum * X + eps
    return pd.DataFrame({"Y": Y, "D": D, "X": X, "W": W})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def data_discrete_binary():
    return make_discrete_binary()


@pytest.fixture
def data_continuous():
    return make_continuous()


@pytest.fixture
def data_panel_fe():
    return make_panel_fe()


@pytest.fixture
def data_discrete_multi():
    return make_discrete_multi()


@pytest.fixture
def data_binary_outcome():
    return make_binary_outcome()


@pytest.fixture
def data_count_outcome():
    return make_count_outcome()


@pytest.fixture
def data_clustered():
    return make_clustered()


@pytest.fixture
def data_panel_pcse():
    return make_panel_pcse()


@pytest.fixture
def data_iv():
    return make_iv_data()


@pytest.fixture
def data_discrete_binary_small():
    """Small sample (n=200) for bootstrap tests."""
    return make_discrete_binary(n=200, seed=42)


@pytest.fixture
def data_heteroscedastic():
    """Recipe A with heteroscedastic errors for vcov tests."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.uniform(-2, 2, n)
    D = rng.choice([0, 1], n, p=[0.5, 0.5]).astype(str)
    Z1 = rng.normal(0, 1, n)
    D_num = (D == "1").astype(float)
    eps = rng.normal(0, 0.1 + np.abs(X), n)
    Y = 1 + 0.5 * X + 2 * D_num + 1.5 * D_num * X + 0.3 * Z1 + eps
    return pd.DataFrame({"Y": Y, "D": D, "X": X, "Z1": Z1})

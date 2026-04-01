"""Type aliases for the interflex package."""

from __future__ import annotations
from typing import Literal

Method = Literal["linear", "logit", "probit", "poisson", "nbinom"]
VarType = Literal["delta", "simu", "bootstrap"]
VcovType = Literal["homoscedastic", "robust", "cluster", "pcse"]
TreatType = Literal["discrete", "continuous"]
XdistrType = Literal["histogram", "density", "none"]

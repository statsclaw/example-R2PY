"""interflex: Python implementation of the interflex linear estimator for interaction effects."""

from .core import interflex
from .result import InterflexResult

__all__ = ["interflex", "InterflexResult"]
__version__ = "0.1.0"

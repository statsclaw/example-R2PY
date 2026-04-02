"""interflex: Python implementation of the interflex linear estimator for interaction effects."""

import sys
import types

from .core import interflex as _interflex
from .result import InterflexResult

__all__ = ["interflex", "InterflexResult"]
__version__ = "0.1.0"


class _CallableModule(types.ModuleType):
    """Make the module callable: ``import interflex; interflex(data, ...)``."""

    def __call__(self, *args, **kwargs):
        return _interflex(*args, **kwargs)


# Replace this module in sys.modules with a callable wrapper
_mod = _CallableModule(__name__)
_mod.__dict__.update({
    k: v for k, v in globals().items()
    if not k.startswith("_CallableModule")
})
_mod.interflex = _interflex
_mod.InterflexResult = InterflexResult
_mod.__file__ = __file__
_mod.__package__ = __package__
_mod.__path__ = __path__
_mod.__spec__ = __spec__
_mod.__version__ = __version__
_mod.__all__ = __all__
sys.modules[__name__] = _mod

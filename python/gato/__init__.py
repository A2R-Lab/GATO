"""gato — GPU-accelerated batched trajectory optimization.

Core solver (`BSQP`) imports with numpy only; heavier layers (MPC controller,
force estimators, gym env) pull in pinocchio / gymnasium lazily on first access.
"""
try:
    from importlib.metadata import version as _version

    __version__ = _version("gato")
except Exception:  # not installed (e.g. sys.path use from a checkout)
    __version__ = "0.0.2"

from .interface import BSQP, available

# Heavy-dependency exports resolved lazily (PEP 562) so `import gato` works in a
# numpy-only environment.
_LAZY = {
    "MPC_GATO": ".mpc_controller",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module 'gato' has no attribute {name!r}")


__all__ = ["BSQP", "available", "__version__", *sorted(_LAZY)]

"""gato — GPU-accelerated batched trajectory optimization.

Core solver (`BSQP`) imports with numpy only; heavier layers (MPC controller,
force estimators, gym env) pull in pinocchio / gymnasium lazily on first access.
"""
try:
    from importlib.metadata import version as _version

    __version__ = _version("gato")
except Exception:  # not installed (e.g. sys.path use from a checkout)
    __version__ = "0.0.2"

from .interface import BSQP, SolveResult, SolverStats, available, robot_info

# Heavy-dependency exports resolved lazily (PEP 562) so `import gato` works in a
# numpy-only environment. NOTE: "build"/"codegen" resolve to the FUNCTIONS in
# gato.builder (call gato.build(urdf, ...)); the module is named builder.py so
# `from gato.builder import ...` can never shadow the gato.build callable.
_LAZY = {
    "build": ".builder",
    "codegen": ".builder",
    "MPC_GATO": ".mpc_gato",
    "MPCController": ".controller",
    "StepResult": ".controller",
    "HypothesisBatch": ".hypotheses",
    "ForceHypothesisBatch": ".hypotheses",
    "MPCPolicy": ".policy",
    "TrajectoryReference": ".policy",
    "GoalReference": ".policy",
    "ArmTrackEnv": ".envs",
    "ForceEstimator": ".estimators",
    "CEMForceEstimator": ".estimators",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module 'gato' has no attribute {name!r}")


__all__ = ["BSQP", "SolveResult", "SolverStats", "available", "robot_info",
           "__version__", *sorted(_LAZY)]

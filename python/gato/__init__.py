"""gato — GPU-accelerated batched trajectory optimization.

Core solver (`BSQP`) imports with numpy only; heavier layers (MPC controller,
force estimators, gym env) pull in pinocchio / gymnasium lazily on first access.
"""
try:
    from importlib.metadata import version as _version

    __version__ = _version("gato-mpc")
except Exception:  # not installed (e.g. sys.path use from a checkout)
    __version__ = "0.1.6"

from .interface import BSQP, SolveResult, SolverStats, available, module_name, robot_info, robot_urdf
from .config import SolverParams

# Heavy-dependency exports resolved lazily (PEP 562) so `import gato` works in a
# numpy-only environment. NOTE: "build"/"codegen" resolve to the FUNCTIONS in
# gato.builder (call gato.build(urdf, ...)); the module is named builder.py so
# `from gato.builder import ...` can never shadow the gato.build callable.
_LAZY = {
    "build": ".builder",
    "codegen": ".builder",
    "register_robot": "._cache",
    "CostMenu": ".cost_menu",
    "Term": ".cost_menu",
    "LinearTerm": ".cost_menu",
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
    "GaitSchedule": ".gait",
    "GaitProgrammer": ".gait",
    "MuJoCoWorld": ".worlds",
    "PinocchioWorld": ".worlds",
}
# submodules reachable as attributes (gato.fingerprint.check, gato.worlds, ...) — plain
# `import gato` must not pull their optional dependencies, so they resolve lazily too
_LAZY_MODULES = {"fingerprint", "worlds", "gait", "certificate", "linsys_autotune", "envs", "rowkinds"}


def __getattr__(name):
    import importlib

    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    if name in _LAZY_MODULES:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module 'gato' has no attribute {name!r}")


__all__ = ["BSQP", "SolveResult", "SolverStats", "SolverParams", "available", "module_name", "robot_info", "robot_urdf",
           "__version__", *sorted(_LAZY)]

"""Shared helpers for the GATO paper figure-regeneration scripts.

Every ``reproduce_figN_*.py`` script in this directory imports from here so they
share a consistent look, model/path resolution, data I/O, and CLI flags. Run any
script from the repo root, e.g.::

    python examples/paper-figures/reproduce_fig4_hparam.py --quick

Reproducibility tiers (see paper-figures/README.md):
  A (no GPU)  : ``--replot`` loads bundled/recovered data and only renders.
  B (GPU)     : default invocation re-runs the experiment on the GPU.
  C (hardware): Fig-6 / Fig-8 / Table-II are NOT reproducible in software.
"""
import os
import sys
import pickle
import argparse

import numpy as np

# --- repo layout -----------------------------------------------------------
# this file lives at <repo>/examples/paper-figures/_common.py
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(HERE, "data")            # regenerated pkls land here
FIG_DIR = HERE                                    # figures render next to scripts
BENCH_DIR = os.path.join(REPO, "examples", "benchmarks")   # benchmark scripts + baselines + data
BENCH_DATA = os.path.join(BENCH_DIR, "data")               # benchmark pkls (GATO sweep, recovered)

# make the GATO python package importable no matter the CWD
for _p in (os.path.join(REPO, "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# robot URDFs (note iiwa14 lives under iiwa_description/, not iiwa14_description/)
URDFS = {
    "indy7": os.path.join(REPO, "examples", "indy7_description", "indy7.urdf"),
    "iiwa14": os.path.join(REPO, "examples", "iiwa_description", "iiwa14.urdf"),
}


def resolve_model(plant):
    """Return (urdf_path, model_dir, pin_model) for a plant ('indy7'|'iiwa14')."""
    import pinocchio as pin
    urdf = URDFS[plant]
    if not os.path.exists(urdf):
        raise FileNotFoundError(f"URDF for {plant} not found at {urdf}")
    model_dir = os.path.dirname(urdf) + "/"
    model, _, _ = pin.buildModelsFromUrdf(urdf, model_dir)
    return urdf, model_dir, model


def require_module(plant, N):
    """Assert the compiled bsqpN{N}_{plant} extension exists; else a clear error.

    Batch sizes / (plant, N) are compile-time, so the module must be pre-built::

        cmake -S . -B build -DPLANT="indy7;iiwa14" -DKNOTS="8;16;32;64;128" \\
              -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel 4
    """
    import importlib
    try:
        importlib.import_module(f"gato.bsqpN{N}_{plant}")
    except ImportError as e:
        raise SystemExit(
            f"\n[paper-figures] Missing compiled module gato.bsqpN{N}_{plant} "
            f"({e}).\nBuild it first, e.g.:\n"
            f"  cmake -S . -B build -DPLANT=\"{plant}\" -DKNOTS=\"{N}\" "
            f"-DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel 4\n"
        )


# --- plotting --------------------------------------------------------------
def set_paper_rcParams():
    """Apply the serif/font-size style shared across the paper figures."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; scripts save PNGs
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })
    return plt


def batch_color(batch_size):
    """Color for a batch size (falls back through config.BATCH_COLORS)."""
    from gato.config import BATCH_COLORS
    return BATCH_COLORS.get(int(batch_size), None)


def savefig(fig, name, dpi=150):
    """Save a figure as paper-figures/<name>.png and return the path."""
    out = os.path.join(FIG_DIR, name if name.endswith(".png") else name + ".png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[paper-figures] saved {os.path.relpath(out, REPO)}")
    return out


# --- data I/O --------------------------------------------------------------
def save_data(obj, name):
    """Pickle ``obj`` to paper-figures/data/<name>.pkl (regenerated artifact)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, name if name.endswith(".pkl") else name + ".pkl")
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    print(f"[paper-figures] wrote {os.path.relpath(p, REPO)}")
    return p


def load_data(name, recovered=None):
    """Load a regenerated pkl from data/, else fall back to a recovered path.

    Prefers a freshly regenerated paper-figures/data/<name>.pkl; if absent and a
    ``recovered`` path (repo-relative or absolute) is given, loads that instead.
    Prints which source was used so the data lineage is explicit at runtime.
    """
    regen = os.path.join(DATA_DIR, name if name.endswith(".pkl") else name + ".pkl")
    if os.path.exists(regen):
        print(f"[paper-figures] loading regenerated {os.path.relpath(regen, REPO)}")
        with open(regen, "rb") as f:
            return pickle.load(f)
    if recovered is not None:
        rp = recovered if os.path.isabs(recovered) else os.path.join(REPO, recovered)
        if os.path.exists(rp):
            print(f"[paper-figures] loading recovered {os.path.relpath(rp, REPO)}")
            with open(rp, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(
        f"No data for '{name}': neither {regen} nor recovered={recovered} exists. "
        f"Run without --replot to regenerate."
    )


# --- CLI -------------------------------------------------------------------
def add_repro_args(parser):
    """Add the standard reproducibility flags shared by all figure scripts."""
    parser.add_argument("--replot", action="store_true",
                        help="skip GPU compute; load bundled/recovered data and only render")
    parser.add_argument("--quick", action="store_true",
                        help="tiny subset for a fast wiring smoke (NOT paper numbers)")
    parser.add_argument("--regen", action="store_true",
                        help="force re-running compute even if regenerated data exists")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    return parser


def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if x != ""]


# ---- paper-experiment constants (moved from gato.config/gato.common: these are
# CS3 pick-place experiment settings, not solver API) ----

PICKPLACE_SOLVER_PARAMS = {
    'max_sqp_iters': 5,
    'kkt_tol': 0.0,
    'max_pcg_iters': 100,
    'pcg_tol': 1e-6,
    'solve_ratio': 1.0,
    'mu': 10.0,
    'q_cost': 5.0,
    'qd_cost': 1e-2,
    'u_cost': 5e-7,
    'N_cost': 50.0,
    'q_lim_cost': 0.0,
    'vel_lim_cost': 0.0,
    'ctrl_lim_cost': 0.0,
    'rho': 0.001
}

PICKPLACE_MPC_DEFAULTS = {
    'goal_timeout': 5.0,
    'goal_threshold': 0.05,
    'velocity_threshold': 1.0
}

# Pendulum parameter defaults
PENDULUM_DEFAULT_PARAMS = {
    'mass': 15.0,           # kg
    'length': 0.3,          # m
    'damping': 0.4,         # Nms/rad
    'initial_angle': np.array([0.3, 0.0, 0.0])  # axis-angle (radians)
}

# Default pick&place goal sequence (IIWA14 workspace)
PICKPLACE_DEFAULT_GOALS = [
    np.array([0.5, -0.1865, 0.5]),
    np.array([0.5, 0.5, 0.2]),
    np.array([0.3, 0.3, 0.8]),
    np.array([0.6, -0.5, 0.2]),
    np.array([0.0, -0.5, 0.8])
]


def sample_axis_angle(mag_range=(0.0, 0.6)):
    """
    Sample random axis-angle vector for pendulum initial condition.
    
    Args:
        mag_range: Tuple of (min_magnitude, max_magnitude) in radians
        
    Returns:
        3D axis-angle vector
    """
    mag = np.random.uniform(*mag_range)
    # Random direction on unit sphere
    v = np.random.normal(size=3)
    n = np.linalg.norm(v) + 1e-12
    axis = v / n
    return axis * mag


def sample_pendulum_params(length_range=(0.3, 0.7), damping_range=(0.1, 0.6), 
                          angle_range=(0.0, 0.6), mass=15.0):
    """
    Sample random pendulum configuration for parameter sweeps.
    
    Args:
        length_range: Tuple of (min, max) pendulum length in meters
        damping_range: Tuple of (min, max) damping coefficient in Nms/rad
        angle_range: Tuple of (min, max) initial angle magnitude in radians
        mass: Fixed mass in kg
        
    Returns:
        Dictionary with pendulum configuration
    """
    return {
        'mass': mass,
        'length': np.random.uniform(*length_range),
        'damping': np.random.uniform(*damping_range),
        'initial_angle': sample_axis_angle(angle_range)
    }

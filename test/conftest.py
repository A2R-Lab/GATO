"""Shared pytest config: repo-checkout import path + gpu/slow marker gating.

Tier selection (CI-ready; no CI wiring here):
    pytest -m "not gpu"          # host-only: packaging, math, codegen determinism
    pytest -m gpu                # needs a CUDA GPU with built solver modules
    pytest -m "not slow"         # skip codegen/build-heavy tests
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

import gato  # noqa: E402
from gato import SolverParams  # noqa: E402

# The suite's solver configuration. These are the OLD BSQP constructor
# defaults (pre-2026-09-20): every gate, oracle tolerance and golden npz was
# established under them, so pinning them here keeps the suite's numerics
# unchanged while the library default (gato.SolverParams()) is the paper/MPC
# set. Tests that want the library default say so explicitly.
TEST_PARAMS = SolverParams(max_sqp_iters=10, max_pcg_iters=100, pcg_tol=1e-4, solve_ratio=1.0,
                           mu=1.0, rho=1e-3, q_cost=2.0, qd_cost=1e-4, u_cost=1e-6, N_cost=50.0,
                           q_lim_cost=1e-3, vel_lim_cost=0.0, ctrl_lim_cost=0.0)

HAVE_MODULES = bool(gato.available())
HAVE_PIN = importlib.util.find_spec("pinocchio") is not None

INDY7_URDF = REPO / "examples" / "indy7_description" / "indy7.urdf"
IIWA14_URDF = REPO / "examples" / "iiwa_description" / "iiwa14.urdf"
GO2_URDF = REPO / "external" / "GRiD" / "config" / "robot_assets" / "go2.urdf"
URDFS = {"indy7": INDY7_URDF, "iiwa14": IIWA14_URDF}

# ---- go2 (floating base, N16-only module) shared helpers ----
GO2_N, GO2_DT = 16, 0.01
GO2_NQ, GO2_NV, GO2_NU = 19, 18, 12
GO2_NX = GO2_NQ + GO2_NV
GO2_XU_STRIDE = GO2_NX + GO2_NU
# the go2 gates run the barrier weights at ZERO so row-group violation deltas
# are attributable (the plant barriers are not the mechanism under test)
GO2_TEST_PARAMS = TEST_PARAMS.replace(q_cost=1.0, qd_cost=1e-2, u_cost=1e-4, N_cost=5.0,
                                      q_lim_cost=0.0, vel_lim_cost=0.0, ctrl_lim_cost=0.0)


def go2_standing_q():
    """Nominal stance (STORED layout [p; quat xyzw; 12 joints]) — the fingerprint rest posture."""
    import numpy as np
    from gato.config import GO2_START_CONFIGS
    return np.asarray(GO2_START_CONFIGS["standing"], dtype=float).copy()


def go2_standing_x(**base):
    """Standing state [q; qd=0]; keyword overrides px/py/pz move the base."""
    import numpy as np
    x = np.zeros(GO2_NX)
    x[:GO2_NQ] = go2_standing_q()
    for k, v in base.items():
        x[{"px": 0, "py": 1, "pz": 2}[k]] = v
    return x


def go2_solver(B, **kw):
    """go2 N16 solver on the shared test params (+ field overrides)."""
    return gato.BSQP(model_path=str(GO2_URDF), batch_size=B, N=GO2_N, dt=GO2_DT,
                     params=GO2_TEST_PARAMS.replace(**kw), plant_type="go2")


def go2_goals_at(model, x, B):
    """EE (imu) goal pinned at the CURRENT pose so the solve stays quiet."""
    import numpy as np
    import pinocchio as pin
    data = model.createData()
    pin.framesForwardKinematics(model, data, x[:GO2_NQ])
    p = data.oMf[model.getFrameId("imu_joint")].translation
    goals = np.zeros((B, GO2_N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = p[0], p[1], p[2]
    return goals


def mujoco_world(urdf, **kw):
    """MuJoCoWorld for a URDF (skips the test if mujoco is not installed)."""
    pytest.importorskip("mujoco")
    from gato.worlds import MuJoCoWorld
    return MuJoCoWorld(str(urdf), **kw)


KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U = 0, 1, 2


def oracle_box_violations(xu, groups, nq, nv, nu, floating=False):
    """numpy {max, sum} true violation per BOX group from the flat STORED
    trajectory row and each group's own bounds (f32 like the kernel). Floating
    base: the box rows cover the ACTUATED slots only (q[7:], qd[6:], u)."""
    import numpy as np
    nx = nq + nv
    step = nx + nu
    out = []
    for grp in groups:
        lo = np.asarray(grp["lo"], dtype=np.float32)
        hi = np.asarray(grp["hi"], dtype=np.float32)
        viols = []
        for k in range(grp["knot_lo"], grp["knot_hi"]):
            base = k * step
            if grp["kind"] == KIND_BOX_Q:
                g = xu[base + 7:base + 7 + len(lo)] if floating else xu[base:base + nq]
            elif grp["kind"] == KIND_BOX_QD:
                g = xu[base + nq + 6:base + nq + 6 + len(lo)] if floating else xu[base + nq:base + nx]
            else:
                # BOX_U rows = ACTUATED_SIZE (fc builds: nu = actuated + fc)
                g = xu[base + nx:base + nx + len(lo)]
            g = g.astype(np.float32)
            viols.append(np.maximum(0, g - hi) + np.maximum(0, lo - g))
        v = np.concatenate(viols)
        out.append((v.max(), v.sum(dtype=np.float64)))
    return out


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="no built solver modules (cmake or gato.build first)")
    for item in items:
        # The signed pytest-gpu-proof receipt attests the FULL suite (GATO's
        # whole run is ~30s warm — no need for GRiD-style marker scoping), so
        # every item gets the plugin's gpu_proof marker.
        item.add_marker(pytest.mark.gpu_proof)
        if not HAVE_MODULES and "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root():
    return REPO


@pytest.fixture(scope="session")
def urdfs():
    return URDFS


@pytest.fixture(scope="session")
def smallest_module():
    """(plant, N) of the cheapest built module that has a vendored URDF."""
    combos = [k for k in gato.available() if k[0] in URDFS]
    if not combos:
        pytest.skip("no built modules for the vendored robots")
    return min(combos, key=lambda k: k[1])


@pytest.fixture
def make_solver():
    """Factory: fresh BSQP for a (plant, N) with example-01 default params.

    Construction is pinocchio-free (dims come from the module); tests that
    need FK/oracles import pinocchio themselves (never skip: a missing dep is
    a broken environment, not an expected outcome)."""
    def _make(plant, N, batch_size=1, variant=None, **kw):
        return gato.BSQP(model_path=str(URDFS[plant]), batch_size=batch_size,
                         N=N, dt=0.01, params=TEST_PARAMS, plant_type=plant,
                         variant=variant, **kw)

    return _make


@pytest.fixture(scope="session")
def test_params():
    return TEST_PARAMS


@pytest.fixture(scope="session")
def go2_model():
    import pinocchio as pin
    return pin.buildModelFromUrdf(str(GO2_URDF), pin.JointModelFreeFlyer())

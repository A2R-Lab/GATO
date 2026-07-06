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

# The vendored test/pytest-gpu-proof submodule ships its OWN test suite (the
# plugin's internals) — never collect it as part of GATO's suite (its conftest
# would also shadow this one).
collect_ignore = ["pytest-gpu-proof"]

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

import gato  # noqa: E402

HAVE_MODULES = bool(gato.available())
HAVE_PIN = importlib.util.find_spec("pinocchio") is not None

INDY7_URDF = REPO / "examples" / "indy7_description" / "indy7.urdf"
IIWA14_URDF = REPO / "examples" / "iiwa_description" / "iiwa14.urdf"
URDFS = {"indy7": INDY7_URDF, "iiwa14": IIWA14_URDF}


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
    """Factory: fresh BSQP for a (plant, N) with example-01 default params."""
    if not HAVE_PIN:
        pytest.skip("pinocchio required to construct BSQP")

    def _make(plant, N, batch_size=1, **kw):
        return gato.BSQP(model_path=str(URDFS[plant]), batch_size=batch_size,
                         N=N, dt=0.01, plant_type=plant, **kw)

    return _make

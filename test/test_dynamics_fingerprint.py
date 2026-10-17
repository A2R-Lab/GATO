"""Model-alignment fingerprint gates (2026-08-02, PDDP round-5 aftermath).

test/dynamics_fingerprint.json pins qdd at canonical probes per plant. The gpu
gate replays the probes on the attested device dynamics — a drift here means
the MODEL changed (URDF edit, GRiD dynamics change, gravity/convention flip)
and every consumer's fingerprint check is stale: regenerate via
tools/gen_dynamics_fingerprint.py and flag consumers. The host gate keeps the
table well-formed and the pinned URDFs byte-identical.
"""
import hashlib
import json
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(REPO, "test", "dynamics_fingerprint.json")


def test_fingerprint_table_wellformed_and_urdfs_pinned():
    tab = json.load(open(JSON))
    assert tab["schema"] == 2
    for plant, t in tab["plants"].items():
        nv, nu = t["nv"], t["nu"]
        assert len(t["probes"]) == nu + 2  # gravity + inertia_j x nu + coriolis
        assert t["probes"][0]["name"] == "gravity"
        for p in t["probes"]:
            assert len(p["q"]) == t["nq"] and len(p["qd"]) == nv
            assert len(p["u"]) == nv and len(p["qdd"]) == nv
            # base rows carry no probe torque/velocity (convention-immunity)
            assert not np.any(np.asarray(p["u"])[:nv - nu])
            assert not np.any(np.asarray(p["qd"])[:nv - nu])
        urdf = os.path.join(REPO, t["urdf"])
        sha = hashlib.sha256(open(urdf, "rb").read()).hexdigest()
        assert sha == t["urdf_sha256"], (
            f"{plant} URDF changed vs the committed fingerprint — regenerate "
            f"tools/gen_dynamics_fingerprint.py and notify consumers")
        for p in t["probes"]:
            assert np.isfinite(np.asarray(p["qdd"])).all()


def _receipt_arms():
    """(plant, smallest N) per fixed-base arm of the receipt profile (collection
    must not depend on which binaries exist on the host)."""
    from conftest import URDFS
    rows = open(os.path.join(REPO, "test", "receipt_modules.txt")).read().splitlines()
    best = {}
    for line in rows:
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) > 2 or parts[0] not in URDFS:
            continue
        best[parts[0]] = min(best.get(parts[0], 10 ** 9), int(parts[1]))
    return sorted(best.items())


@pytest.mark.gpu
@pytest.mark.parametrize("plant,N", _receipt_arms(), ids=lambda v: str(v))
def test_device_dynamics_match_fingerprint(make_solver, plant, N):
    """Every arm in the receipt profile: the attested device dynamics replay the
    pinned probes (the 0.1.2 arms included)."""
    from gato import fingerprint
    s = make_solver(plant, N, batch_size=1)
    res = fingerprint.check_solver(s, rtol=1e-4)
    assert res["ok"], fingerprint.report(res)
    np.testing.assert_allclose(res["inertia_ratio"], 1.0, atol=1e-4)

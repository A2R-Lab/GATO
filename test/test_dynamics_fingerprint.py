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
        if not os.path.exists(urdf):
            # submodule-vendored URDF (go2 lives in external/GRiD) on a
            # submodule-less checkout (the CI cpu-lane): the sha pin is still
            # enforced by the local gpu-proof receipt run + check_solver
            assert t["urdf"].startswith("external/"), t["urdf"]
        else:
            sha = hashlib.sha256(open(urdf, "rb").read()).hexdigest()
            assert sha == t["urdf_sha256"], (
                f"{plant} URDF changed vs the committed fingerprint — regenerate "
                f"tools/gen_dynamics_fingerprint.py and notify consumers")
        for p in t["probes"]:
            assert np.isfinite(np.asarray(p["qdd"])).all()


@pytest.mark.gpu
def test_device_dynamics_match_fingerprint(make_solver, smallest_module):
    from gato import fingerprint
    plant, N = smallest_module
    s = make_solver(plant, N, batch_size=1)
    res = fingerprint.check_solver(s, rtol=1e-4)
    assert res["ok"], fingerprint.report(res)
    np.testing.assert_allclose(res["inertia_ratio"], 1.0, atol=1e-4)

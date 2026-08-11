#!/usr/bin/env python
"""Generate test/dynamics_fingerprint.json — the model-alignment probe table.

The fingerprint pins THE PLANT MODEL (URDF + dynamics semantics), not the
solver: for each plant, forward-dynamics accelerations qdd at a small set of
canonical (q, qd, u) probes, evaluated by the attested device dynamics
(f_ext = 0). Any consumer — another solver, a MuJoCo/Isaac harness, a
hardware pipeline — can evaluate ITS model at the same probes and compare per
joint in seconds (gato.fingerprint.check).

Probe design (deliberately damping-immune where it matters):
  - "gravity":   qd = 0, u = 0 at the rest posture     -> gravity/bias vector
  - "inertia_j": qd = 0, u = 10 N m on ACTUATED dof j alone -> ~10 * column j
                 of Minv + bias; qd = 0 keeps viscous damping OUT of the probe,
                 so a per-joint response ratio directly exposes effective-
                 inertia mismatch (armature, rotor, missing link mass — the
                 PDDP round-5 q7 class, MuJoCo-vs-URDF ratio 2.6-3.2 there)
  - "coriolis":  bent posture, qd != 0 on the ACTUATED dofs, u = 0
                 -> velocity terms (NOTE: includes joint damping if a consumer
                 models it; disagreement HERE with agreement on inertia_j
                 probes points at damping/friction, not inertia)

SCHEMA 2 (floating base): plants carry explicit nv/nu; q is the STORED
configuration (quat xyzw), qd/qdd/u are nv-sized with u the FULL generalized
force (consumer applies as qfrc; base rows zero). Probes keep the base
velocity zero and orientation identity, so the base qdd rows are
frame-convention-immune (MuJoCo world-linear == pin local-linear there).
Floating device eval rides sim_forward: SI-EULER v' = v + dt*aba means
(v'-v)/dt IS aba exactly (debug_contact_dynamics is fixed-base-shaped).

Conventions pinned in the metadata: gravity -9.81 z-down (world), zero
external wrench, URDF identified by sha256. Run on the lab box with the
default modules installed; the JSON is committed + fingerprinted by the
gpu-proof receipt, so the table is attested alongside the code.
"""
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from gato.interface import BSQP  # noqa: E402
from gato.config import (INDY7_START_CONFIGS, IIWA14_START_CONFIGS,  # noqa: E402
                         GO2_START_CONFIGS)
from gato import fingerprint as fp  # noqa: E402

URDFS = {
    "indy7": os.path.join(REPO, "examples", "indy7_description", "indy7.urdf"),
    "iiwa14": os.path.join(REPO, "examples", "iiwa_description", "iiwa14.urdf"),
    "go2": os.path.join(REPO, "external", "GRiD", "config", "robot_assets",
                        "go2.urdf"),
}
MODULE_N = {"indy7": 8, "iiwa14": 8, "go2": 16}  # smallest built module per plant
REST = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"],
        "go2": GO2_START_CONFIGS["standing"]}
# bent posture for the coriolis probe: the PDDP t2 operating posture (iiwa14),
# a generic elbow-bent indy7, and a crouch-asymmetric go2 (base pose = REST's);
# fixed qd pattern, no RNG anywhere.
BENT = {
    "indy7": [0.4, 0.6, -1.1, 0.3, -0.9, 0.2],
    "iiwa14": [-0.2028, 0.5336, -0.0110, -1.6286, 1.7575, -1.9313, 0.4153],
    "go2": [0.15, 1.2, -1.5, -0.15, 1.1, -1.6, 0.15, 0.7, -2.0, -0.15, 0.8, -2.1],
}
TORQUE = 10.0


def probes_for(plant, nq, nv, nu):
    """(name, q(nq), qd(nv), u_full(nv)) probes; base rows stay zero."""
    off = nv - nu  # 0 fixed base, 6 free-flyer
    rest = np.asarray(REST[plant], dtype=np.float32).reshape(nq)
    bent = rest.copy()
    bent[nq - nu:] = np.asarray(BENT[plant], dtype=np.float32)
    qd_bent = np.zeros(nv, np.float32)
    qd_bent[off:] = [0.5 * ((-1) ** j) * (1 + 0.1 * j) for j in range(nu)]
    out = [("gravity", rest, np.zeros(nv, np.float32), np.zeros(nv, np.float32))]
    for j in range(nu):
        u = np.zeros(nv, np.float32)
        u[off + j] = TORQUE
        out.append((f"inertia_{off + j}", rest, np.zeros(nv, np.float32), u))
    out.append(("coriolis", bent, qd_bent, np.zeros(nv, np.float32)))
    return out


def main():
    table = {
        "schema": 2,
        "semantics": {
            "quantity": "forward-dynamics qdd(q, qd, u), f_ext = 0; u is the "
                        "FULL nv generalized force (apply as qfrc; base rows "
                        "zero on floating plants); q is the stored "
                        "configuration (quat xyzw on a free-flyer)",
            "gravity": -9.81,
            "source": "GATO attested device dynamics (GRiD-generated grid.cuh)",
            "guidance": "per-joint |qdd_yours/qdd_ref| on the inertia_j probes: "
                        "1.00+-0.05 = aligned; >1.5 on any joint = effective-inertia "
                        "mismatch (check armature/rotor/link inertia). inertia_j "
                        "probes are qd=0 hence damping-immune; coriolis-only "
                        "disagreement points at damping/friction modeling. j "
                        "indexes the nv dof vector (actuated dofs only; 6+ on "
                        "floating plants).",
        },
        "plants": {},
    }
    for plant, urdf in URDFS.items():
        s = BSQP(model_path=urdf, batch_size=1, N=MODULE_N[plant], dt=0.01,
                 plant_type=plant)
        nq, nv, nu = s.nq, s.nv, s.n_actuated
        if s.floating_base:
            dev_qdd = fp.solver_qdd_fn(s)
        else:
            def dev_qdd(q, qd, u):
                d = s.solver.debug_contact_dynamics(
                    np.asarray(q, np.float32), np.asarray(qd, np.float32),
                    np.asarray(u, np.float32), np.zeros(6, np.float32))
                return np.asarray(d["qdd"])
        entries = []
        for name, q, qd, u in probes_for(plant, nq, nv, nu):
            entries.append({
                "name": name,
                "q": np.asarray(q, dtype=np.float64).tolist(),
                "qd": np.asarray(qd, dtype=np.float64).tolist(),
                "u": np.asarray(u, dtype=np.float64).tolist(),
                "qdd": np.asarray(dev_qdd(q, qd, u), dtype=np.float64).tolist(),
            })
        table["plants"][plant] = {
            "nq": nq,
            "nv": nv,
            "nu": nu,
            "urdf": os.path.relpath(urdf, REPO),
            "urdf_sha256": hashlib.sha256(open(urdf, "rb").read()).hexdigest(),
            "probes": entries,
        }
        print(f"[{plant}] {len(entries)} probes "
              f"(gravity qdd[0..2] = {entries[0]['qdd'][:3]})")

    out = os.path.join(REPO, "test", "dynamics_fingerprint.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=1)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Dynamics-fingerprint checker: is YOUR plant model the same robot as ours?

``test/dynamics_fingerprint.json`` pins forward-dynamics accelerations
qdd(q, qd, u) at canonical probes per plant (gravity -9.81, f_ext = 0),
evaluated by the attested device dynamics. Any consumer evaluates its own
model at the same probes and compares per joint — the two models disagreeing
HERE will disagree in every closed loop built on them (a 2.6-3.2x per-joint
response ratio was the entire PDDP round-5 "anchor instability").

Schema 2 (floating-base plants): ``q`` is the STORED configuration (nq, quat
xyzw for a free-flyer), ``qd``/``qdd`` are nv-sized, and ``u`` is the FULL
generalized force (nv — a consumer applies it as qfrc, zeros on the base
rows). Probes keep the base velocity at zero and the base orientation at
identity, so the base qdd rows are frame-convention-immune (MuJoCo's
world-frame linear acceleration == pin's local one there).

MuJoCo example (their side, one screen)::

    import gato.fingerprint as fp
    def mj_qdd(q, qd, u):
        data.qpos[:] = q; data.qvel[:] = qd; data.ctrl[:] = 0
        data.qfrc_applied[:] = u
        mujoco.mj_forward(model, data)
        return data.qacc.copy()
    print(fp.report(fp.check(mj_qdd, "iiwa14")))

``check`` needs no GPU and no compiled modules — just the JSON."""
import json
import os

import numpy as np

_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "test", "dynamics_fingerprint.json")


def load(path=None):
    with open(path or _JSON) as f:
        return json.load(f)


def check(qdd_fn, plant, path=None, rtol=0.05):
    """Evaluate ``qdd_fn(q, qd, u) -> qdd`` at every probe of ``plant``.

    Returns a dict: per-probe reference/actual/relative error, plus the
    per-joint response RATIO over the damping-immune inertia probes (the
    effective-inertia comparison — the number that catches armature/rotor/
    missing-mass mismatches). ``ok`` is True when every probe agrees within
    ``rtol`` of its scale."""
    tab = load(path)["plants"][plant]
    nv = tab.get("nv", tab["nq"])  # schema 1 tables are fixed base (nv == nq)
    rows, ratios = [], np.full(nv, np.nan)
    ok = True
    for p in tab["probes"]:
        q, qd, u = (np.asarray(p[k], dtype=np.float64) for k in ("q", "qd", "u"))
        ref = np.asarray(p["qdd"], dtype=np.float64)
        got = np.asarray(qdd_fn(q, qd, u), dtype=np.float64).reshape(nv)
        scale = max(1.0, np.abs(ref).max())
        rel = float(np.abs(got - ref).max() / scale)
        rows.append({"name": p["name"], "ref": ref, "got": got, "rel_err": rel})
        if rel > rtol:
            ok = False
        if p["name"].startswith("inertia_"):
            j = int(p["name"].split("_")[1])
            # response of joint j to its own unit torque, bias-corrected by the
            # gravity probe -> ratio of effective inverse inertia
            grav = next(r for r in rows if r["name"] == "gravity")
            num = got[j] - grav["got"][j]
            den = ref[j] - grav["ref"][j]
            ratios[j] = num / den if abs(den) > 1e-9 else np.nan
    return {"plant": plant, "ok": ok, "rtol": rtol, "rows": rows,
            "inertia_ratio": ratios}


def report(res):
    """Human-readable summary of a check() result."""
    lines = [f"dynamics fingerprint [{res['plant']}]: "
             f"{'ALIGNED' if res['ok'] else 'MISMATCH'} (rtol {res['rtol']})"]
    for r in res["rows"]:
        lines.append(f"  {r['name']:<10} rel_err {r['rel_err']:.3e}")
    lines.append("  per-joint inertia-response ratio (yours/ref, 1.0 = aligned): "
                 + np.array2string(res["inertia_ratio"], precision=2))
    bad = [j for j, x in enumerate(res["inertia_ratio"])
           if np.isfinite(x) and abs(x - 1) > 0.5]
    if bad:
        lines.append(f"  >> joints {bad}: effective-inertia mismatch — check "
                     f"armature / rotor inertia / link mass in YOUR model")
    return "\n".join(lines)


def check_solver(solver, path=None, rtol=1e-4):
    """Self-check: a BSQP instance's device dynamics vs the committed table
    (regression gate; also verifies the vendored URDF is the pinned one)."""
    tab = load(path)["plants"][solver.plant_type]
    import hashlib
    urdf = os.path.join(os.path.dirname(_JSON), "..", tab["urdf"])
    sha = hashlib.sha256(open(urdf, "rb").read()).hexdigest()
    if sha != tab["urdf_sha256"]:
        raise AssertionError(f"URDF {tab['urdf']} sha mismatch vs fingerprint "
                             f"— regenerate tools/gen_dynamics_fingerprint.py")

    if getattr(solver, "floating_base", False):
        return check(solver_qdd_fn(solver), solver.plant_type, path=path, rtol=rtol)

    def dev_qdd(q, qd, u):
        d = solver.solver.debug_contact_dynamics(
            np.asarray(q, np.float32), np.asarray(qd, np.float32),
            np.asarray(u, np.float32), np.zeros(6, np.float32))
        return np.asarray(d["qdd"])

    return check(dev_qdd, solver.plant_type, path=path, rtol=rtol)


def solver_qdd_fn(solver):
    """Device qdd(q, qd, u_full) via one ``sim_forward`` SI-EULER step:
    v' = v + dt * aba(q, v, tau), so (v' - v)/dt recovers aba EXACTLY (no
    integration approximation — only f32 rounding). Used for floating plants,
    whose debug_contact_dynamics path is fixed-base-shaped; ``u_full`` is the
    nv generalized force and only its ACTUATED tail reaches the device."""
    n_act, dt = solver.n_actuated, solver.dt

    def dev_qdd(q, qd, u_full):
        x = np.concatenate([q, qd]).astype(np.float32)
        u = np.asarray(u_full, dtype=np.float32)[-n_act:]
        out = np.asarray(solver.solver.sim_forward(x, u, dt))[0]
        return (out[len(q):] - np.asarray(qd, dtype=np.float64)) / dt

    return dev_qdd

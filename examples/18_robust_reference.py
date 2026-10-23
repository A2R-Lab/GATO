"""Intro example 18: tracking a reference with outliers, quadratic versus pseudo-Huber.

A per-knot end-effector reference with a few corrupted knots (a sensor glitch,
a bad waypoint) pulls a quadratic tracking cost toward every outlier. The
pseudo-Huber activation, w (sqrt(r^2 + eps^2) - eps), is quadratic below eps
and linear above it, so a far target costs its distance instead of its square
and the plan stays with the clean knots. Both runs solve the same problem from
the same start; the table reports the planned end-effector path against the
clean reference and against the corrupted one.

Needs one built arm module and pinocchio (the ``[test]`` extra) for the frame
check; runs from any cwd:
    python examples/18_robust_reference.py
"""
import numpy as np

import gato
from gato.cost_menu import CostMenu, Term
from gato.config import FR3_START_CONFIGS, GEN3_START_CONFIGS, IIWA14_START_CONFIGS, INDY7_START_CONFIGS, UR5E_START_CONFIGS

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["ready"],
         "fr3": FR3_START_CONFIGS["ready"], "gen3": GEN3_START_CONFIGS["ready"], "ur5e": UR5E_START_CONFIGS["ready"]}
plant, N = min((k for k in gato.available() if k[0] in START), key=lambda k: k[1])
DT = 0.01
params = gato.SolverParams(max_sqp_iters=10, qd_cost=1e-3, u_cost=1e-6)


def make():
    return gato.BSQP(str(gato.robot_urdf(plant)), batch_size=1, N=N, dt=DT, plant_type=plant, params=params)


probe = make()
if not probe.cost_menu_available:
    raise SystemExit(f"{plant}: the module does not carry the cost menu")
nq, nv, nx, nu = probe.nq, probe.nv, probe.nx, probe.nu
q0 = np.asarray(START[plant], np.float64)
p0 = probe.ee_pos(q0)
# the clean reference: a straight line of 12 cm over the horizon; the corrupted
# one jumps 25 cm at four knots
clean = p0[None, :] + np.linspace(0.0, 1.0, N)[:, None] * np.array([0.08, 0.08, 0.04])
noisy = clean.copy()
rng = np.random.default_rng(3)
bad = rng.choice(np.arange(2, N - 1), size=4, replace=False)
noisy[bad] += rng.normal(0.0, 0.25, (4, 3))

x = np.concatenate([q0, np.zeros(nv)]).astype(np.float32)[None]
goals = np.zeros((1, N * 6), np.float32)
rows = []
for label, activation, eps in (("quadratic", "quadratic", 0.0), ("pseudo-Huber eps 2 cm", "pseudo_huber", 0.02)):
    s = make()
    s.set_cost_menu(CostMenu(frame_pos=Term(weight=[2.0, 2.0, 2.0], target=noisy, activation=activation, eps=eps),
                             state=Term(weight=[0.0] * nq + [params.qd_cost] * nv),
                             input=Term(weight=params.u_cost)))
    res = s.solve(x, goals)
    xu = res.xu_b(0)
    path = np.stack([s.ee_pos(xu[k * (nx + nu):k * (nx + nu) + nq].astype(np.float64)) for k in range(N)])
    d_clean = np.linalg.norm(path - clean, axis=1)
    d_noisy = np.linalg.norm(path - noisy, axis=1)
    rows.append((label, int(res.stats.sqp_iters[0]), d_clean.mean(), d_clean.max(), d_noisy[bad].mean()))

print(f"{plant} N{N}: {len(bad)} corrupted knots {sorted(bad.tolist())}, planned end-effector path over the horizon")
print(f"  {'activation':24s} {'iters':>5s} {'mean to clean':>14s} {'max to clean':>13s} {'mean to outliers':>17s}")
for label, iters, m_c, x_c, m_o in rows:
    print(f"  {label:24s} {iters:5d} {m_c:11.4f} m  {x_c:10.4f} m  {m_o:14.4f} m")
print("  the pseudo-Huber plan stays nearer the clean line and further from the outliers; its tail is linear, so an")
print("  outlier pulls with a bounded force (w per meter) instead of w times its distance")

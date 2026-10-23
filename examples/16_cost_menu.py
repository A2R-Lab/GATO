"""Intro example 16: the cost menu.

The cost menu (``gato.cost_menu``) replaces the tracking objective with up to
eight terms, each a residual under an activation: frame position and
orientation of the end effector, state, input, linear features on x and u, the
end-effector twist, and a second frame position when the plant bakes one.
Every target, weight, bound and activation is per knot, shared across the
batch or per solve. This example spells the default objective as a menu and
checks the two agree, then changes what the default cannot: per-solve targets,
a pseudo-Huber posture term, targets that move from tick to tick under the
real-time iteration, and the per-term readout.

Needs one built arm module; runs from any cwd:
    python examples/16_cost_menu.py
"""
import numpy as np

import gato
from gato.cost_menu import CostMenu, Term, TERMS
from gato.config import FR3_START_CONFIGS, GEN3_START_CONFIGS, IIWA14_START_CONFIGS, INDY7_START_CONFIGS, UR5E_START_CONFIGS

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["ready"],
         "fr3": FR3_START_CONFIGS["ready"], "gen3": GEN3_START_CONFIGS["ready"], "ur5e": UR5E_START_CONFIGS["ready"]}
plant, N = min((k for k in gato.available() if k[0] in START), key=lambda k: k[1])
B, DT = 2, 0.01
params = gato.SolverParams(max_sqp_iters=1, q_cost=2.0, qd_cost=1e-3, u_cost=1e-6, N_cost=20.0)   # the real-time iteration
solver = gato.BSQP(str(gato.robot_urdf(plant)), batch_size=B, N=N, dt=DT, plant_type=plant, params=params)
if not solver.cost_menu_available:
    raise SystemExit(f"{plant}: the module does not carry the cost menu (rebuild against grid-rbd 0.5.5)")
nq, nv, nx, nu = solver.nq, solver.nv, solver.nx, solver.nu

q0 = np.asarray(START[plant], np.float32)
x = np.tile(np.concatenate([q0, np.zeros_like(q0)]), (B, 1))
goal = np.array([0.35, 0.25, 0.5], np.float32)
goals = np.zeros((B, N * 6), np.float32)
goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = goal

# 1. the tracking objective, as shipped
default = solver.solve(x, goals)

# 2. the same objective spelled as a menu: the end-effector position under q_cost
#    (N_cost at the last knot), the velocities under qd_cost, the torques under u_cost
W = np.tile([params.q_cost] * 3, (N, 1)).astype(np.float32)
W[N - 1] = params.N_cost
menu = CostMenu(frame_pos=Term(weight=W, target=goal),
                state=Term(weight=[0.0] * nq + [params.qd_cost] * nv),
                input=Term(weight=params.u_cost))
solver.set_cost_menu(menu)
same = solver.solve(x, goals)
print(f"{plant} N{N}: the tracking objective as a menu")
print(f"  initial merit, default vs menu: {default.stats.initial_merit[0]:.6f} vs {same.stats.initial_merit[0]:.6f}")
rel = np.abs(default.xu - same.xu).max() / np.abs(default.xu).max()
print(f"  plans after one iteration: accepted steps {default.stats.step_size[0, 0]:.3f} vs {same.stats.step_size[0, 0]:.3f}, "
      f"largest difference {rel:.1e} of the largest entry (rounding: the end-effector Jacobian takes another arithmetic path)")

# 3. what the default cannot do: one goal per solve, and a pseudo-Huber posture term
#    that holds the arm near its start without a quadratic pull at large errors
targets = np.stack([np.tile(goal, (N, 1)), np.tile(goal + [0.0, -0.3, 0.1], (N, 1))]).astype(np.float32)   # (B, N, 3)
menu.frame_pos.target = targets
menu.state = Term(weight=[1.0] * nq + [params.qd_cost] * nv, target=np.concatenate([q0, np.zeros_like(q0)]),
                  activation="pseudo_huber", eps=0.1)
solver.set_cost_menu(menu)            # the same shapes: no new device allocation
res = solver.solve(x, goals)
terms = solver.cost_menu_terms(res.xu)   # (B, N, 8), each term alone
print("  per-solve goals and a pseudo-Huber posture term; cost per term summed over the horizon:")
for b in range(B):
    row = ", ".join(f"{name} {terms[b, :, i].sum():.4f}" for i, name in enumerate(TERMS) if terms[b, :, i].any())
    print(f"    solve {b}: {row}")

# 4. a goal that moves from tick to tick under the real-time iteration: apply the
#    menu again each tick (a memcpy, not a rebuild) and warm-start from the last plan
print("  a goal moving 1 cm per tick along x, one SQP iteration per tick from the last plan:")
for tick in range(6):
    menu.frame_pos.target = targets + np.array([0.01 * (tick + 1), 0.0, 0.0], np.float32)
    solver.set_cost_menu(menu)
    res = solver.solve(x, goals, xu_warm=res.xu)
    q_last = res.xu_b(0)[(N - 1) * (nx + nu):(N - 1) * (nx + nu) + nq].astype(np.float64)
    gap = np.linalg.norm(solver.ee_pos(q_last) - menu.frame_pos.target[0, N - 1])
    print(f"    tick {tick}: goal x {menu.frame_pos.target[0, 0, 0]:.3f}, last-knot end effector {gap * 100:.1f} cm from it")

# 5. back to the tracking objective
solver.clear_cost_menu()
print("  menu cleared:", solver.cost_menu is None)

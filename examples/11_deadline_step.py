"""Intro example 11: the deadline-aware step.

``MPCController.step(x, ref, budget_s=...)`` runs SQP iterations while the time
spent plus the next iteration's predicted cost (its measured cost in the
previous budgeted solve) still fits the budget, then returns the best plan so
far (at least one iteration; the cut is at the iteration boundary, never inside
a kernel, so a budget is exceeded by at most one iteration). ``budget_s=None`` runs
``max_sqp_iters`` iterations as before. ``StepResult.iterations``,
``budget_hit`` and ``wall_s`` report what each tick did.

Tracks the figure-8 on the Indy7 (N64) in closed loop with fixed simulation
pacing: unbudgeted with max_sqp_iters=8 to measure the per-step wall time,
then with budgets of 100, 80 and 60 percent of that median. Prints iterations
per tick, how often the budget stopped the solve, wall time, how often and by
how much the step overran its budget, and tracking error. The iteration counts come from the clock, so they depend on the machine
and on what else runs on the GPU: a correctness demo, not a benchmark.

Needs the bsqpN64_indy7 module and a python with pinocchio; runs from any cwd:
    python examples/11_deadline_step.py
"""
import numpy as np
import pinocchio as pin

import gato
from gato import MPC_GATO
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS
from gato.policy import TrajectoryReference

URDF = str(gato.robot_urdf("indy7"))
N, DT, SIM_DT, SIM_TIME = 64, 0.01, 0.001, 2.0
ITERS = 8   # enough SQP iterations per tick that a time budget has something to cut


def run(budget_s):
    model = pin.buildModelFromUrdf(URDF)
    mpc = MPC_GATO(model, model_path=URDF, N=N, dt=DT, batch_size=1, plant_type="indy7",
                   params=dict(max_sqp_iters=ITERS), linsys="pcg",
                   limits="off")   # the unconstrained step (the controller default adds ADMM limit rows)
    ctrl, world, solver = mpc.controller, mpc.world, mpc.solver
    ref = TrajectoryReference(figure8(DT, **FIG8_DEFAULT_PARAMS), DT, N)

    q = np.asarray(INDY7_START_CONFIGS["ready"], dtype=np.float64)
    dq = np.zeros(model.nv)
    x = np.hstack((q, dq)).astype(np.float32)
    ctrl.reset(x)
    ctrl.warmup(x, ref.window(0.0))

    iters, hits, walls, errs = [], [], [], []
    t = 0.0
    while t < SIM_TIME and not ref.done(t):
        goal = ref.window(t)
        r = ctrl.step(x, goal, budget_s=budget_s)
        iters.append(r.iterations)
        hits.append(r.budget_hit)
        walls.append(r.wall_s)
        errs.append(np.linalg.norm(solver.ee_pos(q) - goal[:3]))
        for _ in range(int(round(DT / SIM_DT))):          # apply u for one control period
            q, dq = world.step(q, dq, r.u, SIM_DT)
        x = np.hstack((q, dq)).astype(np.float32)
        t += DT
    walls = np.asarray(walls)
    over = (walls - budget_s) if budget_s is not None else np.zeros_like(walls)
    return dict(iters=np.asarray(iters), hit=float(np.mean(hits)), wall_ms=1e3 * float(np.median(walls)),
                over_frac=float((over > 0).mean()), over_max_ms=1e3 * max(float(over.max()), 0.0),
                err=float(np.mean(errs)))


def report(label, r):
    counts = ", ".join(f"{k}: {int((r['iters'] == k).sum())}" for k in np.unique(r["iters"]))
    print(f"{label}")
    print(f"  ticks              : {len(r['iters'])}")
    print(f"  SQP iterations/tick: {r['iters'].mean():.2f} mean  (histogram {counts})")
    print(f"  budget stopped     : {100 * r['hit']:.0f}% of ticks")
    print(f"  step wall time     : {r['wall_ms']:.3f} ms median (diagnostic, not a benchmark)")
    if r["over_frac"] or r["over_max_ms"]:
        print(f"  over budget        : {100 * r['over_frac']:.0f}% of ticks, by at most {r['over_max_ms']:.3f} ms")
    print(f"  mean tracking err  : {r['err']:.4f} m")


base = run(None)
report(f"Unbudgeted (max_sqp_iters={ITERS}):", base)
for frac in (1.0, 0.8, 0.6):
    budget = frac * base["wall_ms"] * 1e-3
    report(f"\nBudget {1e3 * budget:.3f} ms per step ({100 * frac:.0f}% of the unbudgeted median):", run(budget))
print("\nA budgeted tick returns the best plan its iterations reached. The cut is at an"
      " iteration boundary, so a step overruns its budget by at most one iteration;"
      "\nthe first iteration of a warm-started tick carries the PCG work and the later"
      " ones are cheap, which is why a budget cuts iterations faster than it cuts time"
      "\nand a budget below one iteration's cost buys exactly one.")

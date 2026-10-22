"""Intro example 15: reading the solver statistics.

``SolveResult.stats`` reports what one batched solve did: iterations, the
linear-system path and its PCG iterations, the accepted merit and step per
iteration, the constraint telemetry when limit rows are on, and, under
``set_collect_stats(True)`` (the default), the device time of every stage per
iteration: KKT setup, Schur assembly, the linear system, the merit evaluation
and the line search. ``SolveResult.summary()`` prints the split as a share of
the wall solve time together with what each stage scales with, so a slow solve
on a new GPU can be read off directly instead of profiled.

Solves the reach problem on the smallest built arm module cold (hold-at-x seed)
and then warm (the cold solution as the seed), prints both summaries, and ends
with the knobs that move each stage. The first solve of a process also pays
the one-time launch setup (kernel attributes, module load) inside its initial
merit stage; the warm solve shows the steady state. Times depend on the machine
and on what else runs on the GPU: a diagnostic, not a benchmark.

Needs one built arm module; runs from any cwd:
    python examples/15_solver_stats.py
"""
import numpy as np

import gato

ARMS = ("indy7", "iiwa14", "fr3", "gen3", "ur5e")
plant, N = min((k for k in gato.available() if k[0] in ARMS), key=lambda k: k[1])
B, DT = 4, 0.01
solver = gato.BSQP(str(gato.robot_urdf(plant)), batch_size=B, N=N, dt=DT, plant_type=plant,
                   params=gato.SolverParams(max_sqp_iters=8))

x = np.zeros((B, solver.nx), np.float32)
goals = np.zeros((B, N * 6), np.float32)
goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5

cold = solver.solve(x, goals)
print(f"{plant} N{N}, cold solve (hold-at-x seed)")
print(cold.summary())
warm = solver.solve(x, goals, xu_warm=cold.xu)
print(f"\n{plant} N{N}, warm solve (the cold solution as the seed)")
print(warm.summary())
print(f"\nper-iteration PCG iterations (cold): {cold.stats.pcg_iters.max(axis=1).tolist()}")
print(f"accepted step sizes (cold): {cold.stats.step_size[:, 0].tolist()}")

print("""
What moves each stage:
  kkt, schur    one block per (knot, solve); the horizon N and the batch B set the grid.
  linsys        PCG iterations per solve (pcg_tol, max_pcg_iters, the warm start) or, on the
                bdsv path (SolverParams.linsys, the controller's auto policy), one factorization.
  merit         N x B x NUM_ALPHAS candidate blocks at the generated block size; on a GPU with
                few SMs the solver fuses the candidates into N x B blocks (GATO_MERIT_FUSED=0/1
                overrides), and GATO_MERIT_THREADS sets the block size at compile time.
  line search   B blocks; its cost is the eight-candidate reduction.
  iterations    SolverParams.max_sqp_iters, step(budget_s=...) under a deadline, or
                SolverParams.defect_exit to stop once the plan is dynamically consistent.
""")

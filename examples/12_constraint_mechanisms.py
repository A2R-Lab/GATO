"""Intro example 12: the four limit mechanisms on one task, on shared axes.

The same pick-and-place task (three goals, step reference: every knot targets
the goal at once, so the arm is asked for its maximal acceleration and the
URDF velocity box binds) on the Indy7, run under each limit mechanism of
``MPCController(limits=...)``:

  * none       telemetry rows: the limits are reported, never enforced (the paper setting)
  * barrier    relaxed log barrier (soft)
  * al         augmented Lagrangian (the warm-started ticks are its outer loop)
  * admm       ADMM projection with the residual exit (the controller default)

and at three horizons (N16, N32, N64). Per cell it reports the SIMULATED
plant's peak joint velocity and torque against the URDF limits (the plant is
integrated by pinocchio from the applied torques, so the planner does not
grade itself), the planner's own max violation of its velocity rows, the
goals reached and the task time, and the per-tick solve time as a diagnostic
(fixed simulation pacing, so a correctness demo, not a benchmark; the quiet-box
times are in docs/constraints.md). Mechanisms, defaults and provenance:
docs/constraints.md. The task comes from gato.tasks (seeded scenario 0 of the
dev-set generator with the step reference).

Needs the bsqpN16/32/64_indy7 modules and a python with pinocchio:
    python examples/12_constraint_mechanisms.py [--plot mechanisms.png]
"""
import argparse
import contextlib
import io

import numpy as np
import pinocchio as pin

import gato
from gato import MPC_GATO, tasks

PLANT = "indy7"
URDF = str(gato.robot_urdf(PLANT))
DT, SIM_DT = 0.01, 0.001
HORIZONS = (16, 32, 64)
MECHS = {"none": dict(limits="telemetry"), "barrier": dict(limits="barrier"),
         "al": dict(limits="al"), "admm": dict(limits="admm")}
GOALS = dict(goal_timeout=4.0, goal_threshold=0.05, velocity_threshold=1.0, velocity_norm=2,
             settle_time=0.1, goal_ramp=0.0, pace_by_solve_time=False)


def run(model, scenario, N, mech):
    lim = tasks.urdf_limits(model)
    mpc = MPC_GATO(model, model_path=URDF, N=N, dt=DT, batch_size=1, plant_type=PLANT,
                   linsys="pcg", **MECHS[mech])
    us, viols, solve_us = [], [], []
    orig = mpc.controller.step

    def step(x, g, **kw):
        r = orig(x, g, **kw)
        us.append(np.asarray(r.u, dtype=float)); viols.append(r.row_viol[1]); solve_us.append(r.solve.solve_time_us)
        return r
    mpc.controller.step = step
    x0 = np.hstack((scenario["q0"], np.zeros(model.nv)))
    with contextlib.redirect_stdout(io.StringIO()):
        _, st = mpc.run_mpc_goals(x0, [np.asarray(g) for g in scenario["goals"]], sim_dt=SIM_DT, **GOALS)
    qd = np.abs(np.asarray(st["joint_velocities"])) / lim["qd"]
    u = np.abs(np.asarray(us)) / lim["u"]
    return dict(qd=qd.max(), qd_over=float((qd.max(axis=1) > 1.0).mean()), u=u.max(),
                plan_viol=float(max(viols)), reached=sum(o == "reached" for o in st["goal_outcomes"]),
                t_all=st["time_to_all_reached"], solve_ms=float(np.median(solve_us)) / 1e3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plot", default=None, help="save the shared-axes figure to this PNG (needs matplotlib)")
    args = ap.parse_args()
    model = pin.buildModelFromUrdf(URDF)
    sc = tasks.pickplace_scenarios(model, PLANT, 1, seed=1, ramps=(0.0,))[0]
    print(f"Indy7 pick-and-place, step reference, goals {np.round(sc['goals'], 2).tolist()}")
    print(f"{'N':>4} {'mechanism':<9} {'sim |qd|/lim':>12} {'ticks over':>10} {'sim |u|/lim':>11} "
          f"{'plan viol qd':>12} {'reached':>7} {'task time':>9} {'solve ms':>9}")
    results = {}
    for N in HORIZONS:
        for mech in MECHS:
            r = results[(N, mech)] = run(model, sc, N, mech)
            t_all = "-" if r["t_all"] is None else f"{r['t_all']:.2f} s"
            print(f"{N:>4} {mech:<9} {r['qd']:>12.2f} {100 * r['qd_over']:>9.0f}% {r['u']:>11.2f} "
                  f"{r['plan_viol']:>12.1e} {r['reached']:>5d}/3 {t_all:>9} {r['solve_ms']:>9.3f}")
    print("\nThe simulated plant is graded against the URDF limits (1.0 = at the limit). Telemetry rows"
          " report the violation and leave it;\nthe barrier and the augmented Lagrangian soften it"
          " (one SQP iteration per tick is one outer AL step); the ADMM projection with the\n"
          "controller's per-box rho holds the plan inside the box to within its residual, and the"
          " simulated peak follows the plan to within\nthe tracking error of one control period,"
          " at no cost in goals reached or task time.")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        x = np.arange(len(HORIZONS))
        w = 0.8 / len(MECHS)
        for i, mech in enumerate(MECHS):
            for ax, key, label in zip(axes, ("qd", "u", "plan_viol", "solve_ms"),
                                      ("simulated peak |qd| / limit", "simulated peak |u| / limit",
                                       "planner max violation, qd rows (rad/s)", "solve time per tick (ms, diagnostic)")):
                ax.bar(x + (i - 1.5) * w, [results[(N, mech)][key] for N in HORIZONS], w, label=mech)
                ax.set_xticks(x, [f"N{N}" for N in HORIZONS]); ax.set_title(label, fontsize=9)
        axes[0].axhline(1.0, color="k", lw=0.8, ls="--"); axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
        axes[2].set_yscale("symlog", linthresh=1e-3)
        axes[0].legend(fontsize=8)
        fig.suptitle("Indy7 pick-and-place (step reference): limit mechanisms vs horizon", fontsize=10)
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print(f"figure saved to {args.plot}")


if __name__ == "__main__":
    main()

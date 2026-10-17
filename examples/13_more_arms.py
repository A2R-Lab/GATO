"""Intro example 13: the controller defaults on the 0.1.2 arms (FR3, Gen3, UR5e).

The three arms added in 0.1.2 ship as vendored descriptions
(examples/<robot>_description, provenance and edits in each README), generated
dynamics (gato/dynamics/<robot>) and N64 receipt modules. This example runs
each of them in closed loop on a figure-8 sized from its own URDF limits
(gato.tasks.fig8_scenarios: the period is set so the demanded peak joint speed
is a chosen fraction of the velocity limit) with ``MPCController`` as
constructed by default: limit rows on under the ADMM projection, the residual
exit, the inner linear solver by horizon. Four runs per arm:

  * paper weights, ratio 0.8   the solver's default cost weights (the paper's
                               Indy7 setting, one SQP iteration per tick) on
                               a reference the arm can follow: they track a
                               fast 15 cm figure-8 to several centimetres on
                               every arm, as they do on the Indy7
  * tracking weights, 0.8      q_cost 10, qd_cost 1e-3, three SQP iterations
                               per tick: the knobs a tracking task turns
                               (SolverParams); the error drops by an order of
                               magnitude and the arm uses its speed
  * tracking weights, 1.3      the reference asks for more than the velocity
                               limit: a feasible plan must lag it, and the
                               simulated plant should stay at the limit rather
                               than follow the reference over it
  * telemetry, 1.3             the same run with ``limits="telemetry"``: what
                               the unconstrained controller does instead

Per run it reports the mean end-effector tracking error, the SIMULATED plant's
peak joint velocity and torque against the URDF limits (pinocchio integrates
the applied torques; the planner does not grade itself), the planner's own max
violation of its velocity rows, the SQP iterations and the ADMM inner
iterations per tick. Fixed simulation pacing: a correctness demo, not a
benchmark. Which robots are available: ``gato.available()``; your own arm:
example 05.

Needs the bsqpN64_{fr3,gen3,ur5e} modules and a python with pinocchio:
    python examples/13_more_arms.py [--robots fr3 gen3 ur5e] [--periods 2]
"""
import argparse
import contextlib
import io

import numpy as np
import pinocchio as pin

import gato
from gato import MPC_GATO, tasks

N, DT, SIM_DT = 64, 0.01, 0.001
TRACKING = dict(max_sqp_iters=3, q_cost=10.0, qd_cost=1e-3)     # SolverParams overrides for a tracking task
RUNS = (("paper weights", 0.8, {}),
        ("tracking weights", 0.8, dict(params=TRACKING)),
        ("tracking weights", 1.3, dict(params=TRACKING)),
        ("telemetry", 1.3, dict(params=TRACKING, limits="telemetry")))


def run(plant, model, ratio, periods, **kw):
    urdf = str(gato.robot_urdf(plant))
    sc = tasks.fig8_scenarios(model, plant, 1, seed=7, ratio_range=(ratio, ratio))[0]
    traj, sim_time = tasks.fig8_reference(sc, N, periods=periods)
    lim = tasks.urdf_limits(model)
    mpc = MPC_GATO(model, model_path=urdf, N=N, dt=DT, batch_size=1, plant_type=plant, **kw)
    us, viols, inner, its = [], [], [], []
    orig = mpc.controller.step

    def step(x, g, **skw):
        r = orig(x, g, **skw)
        us.append(np.asarray(r.u, dtype=float))
        viols.append(r.row_viol[1])
        its.append(r.iterations)
        if r.solve.stats.admm_iters is not None:
            inner.append(int(r.solve.stats.admm_iters[0]))
        return r
    mpc.controller.step = step
    x0 = np.hstack((sc["q0"], np.zeros(model.nv)))
    with contextlib.redirect_stdout(io.StringIO()):
        _, st = mpc.run_mpc_fig8(x0, traj, sim_dt=SIM_DT, sim_time=sim_time, pace_by_solve_time=False)
    qd = np.abs(np.asarray(st["joint_velocities"])) / lim["qd"]
    u = np.abs(np.asarray(us)) / lim["u"]
    return dict(limits=mpc.controller.limits, ratio_est=sc["ratio_est"], period=sc["params"]["period"],
                err=float(np.mean(st["goal_distances_knot0"])), qd=float(qd.max()),
                qd_over=float((qd.max(axis=1) > 1.0).mean()), u=float(u.max()),
                plan_viol=float(max(viols)), inner=float(np.mean(inner)) if inner else float("nan"),
                sqp=float(np.mean(its)), ticks=len(us))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robots", nargs="+", default=["fr3", "gen3", "ur5e"])
    ap.add_argument("--periods", type=int, default=2, help="figure-8 periods to simulate per run")
    args = ap.parse_args()
    print(f"{'robot':<6} {'run':<17} {'limits':<10} {'ratio':>5} {'period s':>8} {'track err m':>11} {'sim |qd|/lim':>12} "
          f"{'ticks over':>10} {'sim |u|/lim':>11} {'plan viol qd':>12} {'sqp it':>6} {'inner it':>8}")
    for plant in args.robots:
        model = pin.buildModelFromUrdf(str(gato.robot_urdf(plant)))
        info = gato.robot_info(plant)
        print(f"-- {plant}: {model.nq} joints, EE frame {info['ee_frame']}, "
              f"velocity limits {np.round(model.velocityLimit, 2).tolist()} rad/s")
        for label, ratio, kw in RUNS:
            r = run(plant, model, ratio, args.periods, **kw)
            print(f"{plant:<6} {label:<17} {r['limits']:<10} {r['ratio_est']:>5.2f} {r['period']:>8.2f} {r['err']:>11.4f} {r['qd']:>12.2f} "
                  f"{100 * r['qd_over']:>9.0f}% {r['u']:>11.2f} {r['plan_viol']:>12.1e} {r['sqp']:>6.1f} {r['inner']:>8.1f}")
    print("\nratio = the demanded peak joint speed over the URDF velocity limit (differential IK along the path);"
          " the simulated plant is graded against\nthe URDF limits (1.0 = at the limit). The paper's weights are"
          " the solver defaults on every arm; a tracking task raises q_cost, lowers\nqd_cost and spends more SQP"
          " iterations. Under the controller's limit rows the plan stays inside the velocity box to within its"
          " residual and the\nplant's peak follows it; telemetry rows report the excursion and leave it.")


if __name__ == "__main__":
    main()

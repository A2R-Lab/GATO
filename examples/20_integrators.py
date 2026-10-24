"""The trapezoidal and the RK4 step on the same arm.

Every GATO module integrates with one step, chosen when it is built: the
default fixed-base modules use the trapezoidal update (q += dt qd + dt^2 qdd / 2,
qd += dt qdd), the ``rk4`` variant the classical fourth-order step on the
generated grid integrator. Both see the same dynamics; the difference is the
truncation error and the stability region of the step at the solver's dt.

This script puts the two side by side on the iiwa14:

1. one-step error against a fine reference (RK4 with 200 substeps on
   Pinocchio) as the step grows from 2 ms to 50 ms;
2. an open-loop swing under gravity, zero torque, one second, comparing the
   final state of each module's step chain with the fine reference;
3. the same reach problem solved by both modules, with the per-iteration
   cost of the four dynamics evaluations read from the solver's stage times.

Needs bsqpN16_iiwa14 and bsqpN16_iiwa14_rk4 (``gato.build(..., integrator="rk4")``
or ``MODULES="iiwa14:16:rk4" ./tools/build.sh``) and pinocchio.
"""
import os
import sys

import numpy as np

import gato
from gato.common import rk4
from gato.config import IIWA14_START_CONFIGS, SolverParams

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "iiwa_description", "iiwa14.urdf")


def fine_reference(model, data, q, qd, u, dt, substeps=200):
    h = dt / substeps
    for _ in range(substeps):
        q, qd = rk4(model, data, q, qd, u, h)
    return np.concatenate([q, qd])


def main():
    import pinocchio as pin
    urdf = URDF
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    N = 16
    solvers = {}
    for name, variant in (("trapezoidal", "default"), ("rk4", "rk4")):
        try:
            solvers[name] = gato.BSQP(model_path=urdf, batch_size=1, N=N, dt=0.01,
                                      params=SolverParams(max_sqp_iters=20, solve_ratio=1.0),
                                      plant_type="iiwa14", variant=variant)
        except ValueError as e:
            sys.exit(f"missing module for the {name} step: {e}")
    nq = model.nq
    rng = np.random.default_rng(0)
    q0 = np.asarray(IIWA14_START_CONFIGS["home"], dtype=np.float64) + rng.normal(0, 0.3, nq)
    qd0 = rng.normal(0, 1.0, nq)
    u0 = rng.normal(0, 10.0, nq)

    print("1. one-step error against the fine reference (max abs over [q; qd])")
    print(f"   {'dt [ms]':>8} {'trapezoidal':>14} {'rk4':>14}")
    for dt in (0.002, 0.005, 0.01, 0.02, 0.05):
        ref = fine_reference(model, data, q0, qd0, u0, dt)
        row = []
        for name, s in solvers.items():
            x = np.concatenate([q0, qd0]).astype(np.float32)[None]
            xn = np.asarray(s.sim_forward(x, u0.astype(np.float32)[None], dt)).ravel()
            row.append(np.abs(xn - ref).max())
        print(f"   {1e3 * dt:8.0f} {row[0]:14.3e} {row[1]:14.3e}")

    print("\n2. open-loop swing under gravity, zero torque, one second: mechanical energy drift")
    print("   (the exact flow conserves it; the arm's motion is chaotic, so states are not compared)")

    def energy(x):
        return pin.computeMechanicalEnergy(model, data, x[:nq], x[nq:])

    for dt in (0.005, 0.01, 0.02):
        steps = int(round(1.0 / dt))
        e0 = energy(np.concatenate([q0, np.zeros(nq)]))
        row = []
        for name, s in solvers.items():
            x = np.concatenate([q0, np.zeros(nq)]).astype(np.float32)[None]
            u = np.zeros((1, nq), dtype=np.float32)
            worst = 0.0
            for _ in range(steps):
                x = np.asarray(s.sim_forward(x, u, dt)).reshape(1, -1)
                if not np.all(np.isfinite(x)):
                    worst = np.inf
                    break
                worst = max(worst, abs(energy(x.ravel().astype(np.float64)) - e0))
            row.append(worst)
        print(f"   dt {1e3 * dt:3.0f} ms: trapezoidal {row[0]:.3e} J   rk4 {row[1]:.3e} J   (largest |E(t) - E(0)| over the second)")

    print("\n3. the same reach problem on both modules: the step's error along the plan it returns")
    print("   (one step of the module from each planned knot against the fine reference from the same knot)")
    goals = np.zeros((1, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.45, 0.25, 0.55
    q_home = np.asarray(IIWA14_START_CONFIGS["home"], dtype=np.float64)
    for dt in (0.01, 0.02):
        x = np.concatenate([q_home, np.zeros(nq)]).astype(np.float32)[None]
        row = []
        for name, s in solvers.items():
            s = gato.BSQP(model_path=urdf, batch_size=1, N=N, dt=dt,
                          params=SolverParams(max_sqp_iters=20, solve_ratio=1.0),
                          plant_type="iiwa14", variant=s.variant)
            r = s.solve(x, goals)
            xu = r.xu[0].astype(np.float64)
            step = s.nx + s.nu
            worst = 0.0
            for k in range(N - 1):
                xk = xu[k * step:k * step + s.nx]
                uk = xu[k * step + s.nx:(k + 1) * step]
                x_mod = np.asarray(s.sim_forward(xk.astype(np.float32)[None], uk.astype(np.float32)[None], dt)).ravel()
                x_fine = fine_reference(model, data, xk[:nq], xk[nq:], uk, dt, substeps=50)
                worst = max(worst, np.abs(x_mod - x_fine).max())
            row.append((worst, float(np.asarray(r.stats.final_merit).ravel()[0])))
        print(f"   dt {1e3 * dt:3.0f} ms: trapezoidal {row[0][0]:.3e} (merit {row[0][1]:.3f})"
              f"   rk4 {row[1][0]:.3e} (merit {row[1][1]:.3f})   (largest one-step error along the plan)")

    print("\nThe step is a module property (BSQP.integrator); the solver's linearization and the merit's\n"
          "integrator error always use the module's own step. Timing of the two is a quiet-window measurement,\n"
          "not part of this script.")


if __name__ == "__main__":
    main()

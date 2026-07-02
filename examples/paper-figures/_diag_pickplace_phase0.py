"""Phase-0 diagnostic for the iiwa14 pick-place case (Fig-7 / Table-I, CS3).

NOT a paper figure. Localizes WHY closed-loop pick-place fails before any tuning,
per docs/open-tasks/fig3_mpcgpu_fig7_pickplace_plan.md.

It runs the 4-way diagnostic cross on a FIXED scenario and logs per-step solver
health (sqp/pcg iters, goal distance, joint velocity, NaN flags) plus the
ForceEstimator magnitude trajectory:

    run | payload | batch + force-est | isolates
    a   | off     | batch=1           | does the iiwa14 N16 solver track at all?
    b   | on      | batch=1 (no FE)   | the unmodeled-payload divergence baseline
    c   | on      | batch=8 + FE      | the paper's actual robustness method
    d   | off     | batch=8 + FE      | FE overhead/regression with no disturbance

Plus a 5th run (c_wide) = run c with the FE force range widened to cover the
~147 N (15 kg) payload, testing the hypothesis that the default FE max_radius=20 N
is far too small for the disturbance magnitude.

These are SUCCESS-RATE / solver-health runs (not timing), so they are safe to run
under light GPU load. Run with the GRiD venv (has pinocchio):

    ~/Desktop/GRiD/.venv/bin/python examples/paper-figures/_diag_pickplace_phase0.py
    ~/Desktop/GRiD/.venv/bin/python examples/paper-figures/_diag_pickplace_phase0.py --quick
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

N = 16
DT = 0.01


def _summarize(stats, fe_log, label):
    """Reduce a run_mpc_goals stats dict to a compact health record."""
    outcomes = stats.get("goal_outcomes", [])
    reached = sum(1 for o in outcomes if o == "reached")
    gd = np.asarray(stats.get("goal_distances", []), dtype=float)
    jv = stats.get("joint_velocities", [])
    jv = np.asarray(jv, dtype=float) if len(jv) else np.zeros((0,))
    ee = np.asarray(stats.get("ee_actual", []), dtype=float)
    sqp = np.asarray(stats.get("sqp_iters", []), dtype=float)
    pcg = np.asarray(stats.get("pcg_iters", []), dtype=float)

    nan_gd = bool(np.isnan(gd).any()) if gd.size else False
    nan_ee = bool(np.isnan(ee).any()) if ee.size else False
    max_vel = float(np.nanmax(np.linalg.norm(jv, ord=1, axis=1))) if jv.size else float("nan")

    rec = {
        "label": label,
        "goals_reached": reached,
        "goals_total": len(outcomes),
        "outcomes": outcomes,
        "n_steps": int(gd.size),
        "gd_min": float(np.nanmin(gd)) if gd.size else float("nan"),
        "gd_final": float(gd[-1]) if gd.size else float("nan"),
        "gd_max": float(np.nanmax(gd)) if gd.size else float("nan"),
        "max_l1_vel": max_vel,
        "sqp_max": float(sqp.max()) if sqp.size else float("nan"),
        "sqp_mean": float(sqp.mean()) if sqp.size else float("nan"),
        "pcg_max": float(pcg.max()) if pcg.size else float("nan"),
        "pcg_mean": float(pcg.mean()) if pcg.size else float("nan"),
        "nan_goal_dist": nan_gd,
        "nan_ee": nan_ee,
        "diverged": bool((gd.size and np.nanmax(gd) > 2.0) or nan_gd or nan_ee),
    }
    if fe_log:
        mags = np.asarray([m for (m, r, b) in fe_log], dtype=float)
        rads = np.asarray([r for (m, r, b) in fe_log], dtype=float)
        rec["fe_est_final_N"] = float(mags[-1])
        rec["fe_est_max_N"] = float(np.nanmax(mags))
        rec["fe_radius_max_N"] = float(np.nanmax(rads))
    return rec


def _make_mpc(model, urdf, batch_size, pendulum_config, fe_override=None, solver_override=None):
    """Build an MPC_GATO that logs the FE estimate magnitude each control step."""
    from gato.mpc_controller import MPC_GATO  # puts <repo>/examples on sys.path on import
    from gato.config import PICKPLACE_SOLVER_PARAMS
    from force_estimator import ForceEstimator

    sp = dict(PICKPLACE_SOLVER_PARAMS)
    if solver_override:
        sp.update(solver_override)

    class _DiagMPC(MPC_GATO):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.fe_log = []

        def evaluate_best_trajectory(self, *a, **k):
            bid = super().evaluate_best_trajectory(*a, **k)
            if self.force_estimator is not None:
                st = self.force_estimator.get_stats()
                self.fe_log.append(
                    (float(np.linalg.norm(st["smoothed_estimate"][:3])),
                     float(st["radius"]), int(bid))
                )
            return bid

    mpc = _DiagMPC(
        model, model_path=urdf, N=N, dt=DT, batch_size=batch_size,
        plant_type="iiwa14", pendulum_config=pendulum_config,
        solver_params=sp, track_full_stats=True,
    )
    if fe_override is not None and mpc.force_estimator is not None:
        mpc.force_estimator = ForceEstimator(batch_size=batch_size, **fe_override)
    return mpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 goal instead of the full sequence")
    ap.add_argument("--batch", type=int, default=8, help="batch size for the FE runs")
    ap.add_argument("--sweep-rho", action="store_true",
                    help="isolate payload+batch case: sweep rho to test if KKT conditioning "
                         "fixes the FE divergence (Phase-1 tuning, success-rate only)")
    args = ap.parse_args()

    from gato.experiment_runner import ExperimentRunner
    from gato.config import (PICKPLACE_DEFAULT_GOALS, PICKPLACE_MPC_DEFAULTS,
                             PENDULUM_DEFAULT_PARAMS, IIWA14_START_CONFIGS)

    urdf = C.URDFS["iiwa14"]
    C.require_module("iiwa14", N)
    runner = ExperimentRunner(urdf)
    model = runner.model

    goals = PICKPLACE_DEFAULT_GOALS[:1] if args.quick else PICKPLACE_DEFAULT_GOALS
    x_start = np.hstack((IIWA14_START_CONFIGS["home"], np.zeros(model.nv)))
    md = PICKPLACE_MPC_DEFAULTS
    pend = dict(PENDULUM_DEFAULT_PARAMS)  # mass=15 kg -> ~147 N static -Z
    B = args.batch

    # FE force range widened to cover ~147 N payload (default is max_radius=20 N).
    wide_fe = dict(initial_radius=50.0, min_radius=10.0, max_radius=200.0, smoothing_factor=0.5)

    if args.sweep_rho:
        # Isolate the diverging payload+batch case: does KKT regularization (rho) stop the
        # FE-induced NaN divergence? (rho, fe_override) per run.
        runs = [(f"rho{rho}_b{B}", pend, B, None, {"rho": rho})
                for rho in (0.001, 0.01, 0.05, 0.1, 0.3)]
    else:
        runs = [
            ("a_off_b1",   None, 1,  None, None),
            ("b_on_b1",    pend, 1,  None, None),
            (f"c_on_b{B}", pend, B,  None, None),
            (f"d_off_b{B}", None, B, None, None),
            (f"c_wide_b{B}", pend, B, wide_fe, None),
        ]

    records = []
    for label, pcfg, bs, fe_ovr, sv_ovr in runs:
        print(f"\n{'='*60}\n[{label}] payload={'on' if pcfg else 'off'} batch={bs} "
              f"FE={'wide' if fe_ovr else ('default' if bs > 1 else 'none')}"
              f"{' rho='+str(sv_ovr['rho']) if sv_ovr and 'rho' in sv_ovr else ''}\n{'='*60}")
        mpc = _make_mpc(model, urdf, bs, pcfg, fe_ovr, sv_ovr)
        _, stats = mpc.run_mpc_goals(
            x_start, goals, sim_dt=0.001,
            goal_timeout=md["goal_timeout"], goal_threshold=md["goal_threshold"],
            velocity_threshold=md["velocity_threshold"],
        )
        rec = _summarize(stats, getattr(mpc, "fe_log", None), label)
        records.append(rec)
        print(f"  -> reached {rec['goals_reached']}/{rec['goals_total']}  "
              f"gd[min/final/max]={rec['gd_min']:.3f}/{rec['gd_final']:.3f}/{rec['gd_max']:.3f}m  "
              f"maxL1vel={rec['max_l1_vel']:.2f}  sqp[max/mean]={rec['sqp_max']:.0f}/{rec['sqp_mean']:.1f}  "
              f"pcg[max/mean]={rec['pcg_max']:.0f}/{rec['pcg_mean']:.1f}  "
              f"NaN={rec['nan_goal_dist'] or rec['nan_ee']}  diverged={rec['diverged']}")
        if "fe_est_final_N" in rec:
            print(f"     FE: est_final={rec['fe_est_final_N']:.1f}N est_max={rec['fe_est_max_N']:.1f}N "
                  f"radius_max={rec['fe_radius_max_N']:.1f}N  (payload ~147N)")

    print(f"\n{'#'*60}\nPHASE-0 SUMMARY (goals={len(goals)}, batch={B})\n{'#'*60}")
    hdr = f"{'run':<14}{'reached':<9}{'gd_final':<10}{'gd_max':<9}{'diverged':<10}{'fe_est_N':<10}"
    print(hdr)
    for r in records:
        fe = f"{r.get('fe_est_final_N', float('nan')):.1f}" if "fe_est_final_N" in r else "-"
        print(f"{r['label']:<14}{r['goals_reached']}/{r['goals_total']:<7}"
              f"{r['gd_final']:<10.3f}{r['gd_max']:<9.3f}{str(r['diverged']):<10}{fe:<10}")

    print("\nINTERPRETATION GUIDE:")
    print("  a fails        -> SOLVER/config bound on iiwa14 N16 (attack rho/conditioning), not payload")
    print("  a ok, c ok     -> harness was just under-powered at batch=1; Fig-7 reproduces (likely the fix)")
    print("  a ok, c fails, c_wide ok -> FE force RANGE was too small (max_radius=20N << 147N) -> tuning fix")
    print("  a ok, c & c_wide fail    -> genuine payload-robustness gap -> escalate (Phase-2 territory)")

    out = os.path.join(C.DATA_DIR, "pickplace_phase0_diag.pkl") if hasattr(C, "DATA_DIR") else "/tmp/pickplace_phase0_diag.pkl"
    try:
        import pickle
        with open(out, "wb") as fh:
            pickle.dump(records, fh)
        print(f"\nsaved -> {out}")
    except Exception as e:
        print(f"(save skipped: {e})")


if __name__ == "__main__":
    main()

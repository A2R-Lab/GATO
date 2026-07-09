"""Regenerate Fig-7 + Table-I (Case Study 3): planning under uncertainty.

Paper IV-E: a 7-DoF KUKA iiwa14 runs a multi-point pick-and-place task with an
unmodeled 15 kg suspended pendulum (swinging payload). At each control step GATO
warm-starts from the previous solution and solves a batch of disturbance-hypothesis
problems, selecting the control most consistent with the observed motion. Over 100
randomized scenarios (pendulum length 0.3-0.7 m, initial angle 0-0.6 rad, damping
0.1-0.6 Nms/rad) we report success rate + mean completion time vs batch size
(Table-I) and the CDF of episode completion times (Fig-7). N=16, h=0.01, 5 SQP
iters, PCG tol 1e-6, 1 kHz RK4 sim. Success = EE within 5 cm of each goal in <5 s
with total joint velocity < 1.0 rad/s.

>>> STATUS (2026-06-25, see docs/baselines.md "pick-place Phase-0"):
The dominant Table-I bug is FIXED — the ForceEstimator (the CS3 batched-robustness
mechanism) was silently disabled by a CWD-relative import, so every batch size behaved
like batch-1 (all-zero Table-I). With it live, batching now demonstrably helps (batch>1
succeeds where batch=1 fails), and the FE sampling is now SEEDED (reproducible runs).
>>> CAVEAT: a residual FE-robustness gap remains (parked Phase-2 R&D): the estimate is
high-variance and doesn't fully converge to a large *swinging* payload, so the success
*magnitudes* may not match the paper exactly. The curve SHAPE (success rising with batch)
and the CDF are correct; ship with this caveat. The script runs and produces the figure
regardless (NaN scenarios are caught per-batch and counted as failures, not crashes).
>>> METRICS (2026-07-08): PICKPLACE_MPC_DEFAULTS now uses the paper-comparable
success gate + clock — Euclidean velocity norm (the old L1-sum gate was stricter and
capped success) and fixed dt pacing (completion time = physical task time; the old
wall-clock pacing reported cumulative wall time and was non-reproducible). Table-I
data generated before this change is on the old metrics — do not mix pools.

Examples::
    python examples/paper-figures/reproduce_fig7_pickplace.py            # 100 scenarios (slow)
    python examples/paper-figures/reproduce_fig7_pickplace.py --quick    # fast smoke
    python examples/paper-figures/reproduce_fig7_pickplace.py --replot   # plot saved data
"""
import argparse
import numpy as np

import _common as C

N = 16
DT = 0.01


def run(n_scenarios, batch_sizes, max_time):
    from _pickplace_runner import ExperimentRunner
    from _common import (PICKPLACE_DEFAULT_GOALS, PICKPLACE_SOLVER_PARAMS,
                         PICKPLACE_MPC_DEFAULTS, sample_pendulum_params)

    urdf = C.URDFS["iiwa14"]
    C.require_module("iiwa14", N)
    runner = ExperimentRunner(urdf)

    # per-batch pools of episode completion times (None == failed/timeout)
    pool = {b: [] for b in batch_sizes}
    for s in range(n_scenarios):
        pend = sample_pendulum_params()  # random length/angle/damping, 15 kg
        print(f"scenario {s + 1}/{n_scenarios}  (L={pend['length']:.2f} d={pend['damping']:.2f})")
        res = runner.run_pickplace_sweep(
            batch_sizes=batch_sizes, N=N, dt=DT, sim_dt=0.001, plant_type="iiwa14",
            goal_sequences=[PICKPLACE_DEFAULT_GOALS], pendulum_config=pend,
            solver_params=PICKPLACE_SOLVER_PARAMS, mpc_defaults=PICKPLACE_MPC_DEFAULTS,
            verbose=False,
        )
        for b in batch_sizes:
            r = res.get(b, {})
            seq = (r.get("per_sequence") or [{}])[0]
            pool[b].append(seq.get("time_to_all_reached"))  # seconds, or None
    return {"batch_sizes": batch_sizes, "n_scenarios": n_scenarios, "pool": pool}


def table_I(data):
    lines = ["", "=" * 48, "TABLE I — pick-place success vs batch size", "=" * 48,
             f"{'Batch':>6} {'Success [%]':>12} {'Mean time [s]':>14}"]
    for b in data["batch_sizes"]:
        times = data["pool"][b]
        done = [t for t in times if t is not None]
        sr = 100.0 * len(done) / len(times) if times else 0.0
        mt = float(np.mean(done)) if done else float("nan")
        lines.append(f"{b:>6} {sr:>12.1f} {mt:>14.2f}")
    txt = "\n".join(lines)
    print(txt)
    import os
    with open(os.path.join(C.FIG_DIR, "table_I.txt"), "w") as f:
        f.write(txt + "\n")


def plot_cdf(data, max_time):
    plt = C.set_paper_rcParams()
    fig = plt.figure(figsize=(8, 5))
    for b in data["batch_sizes"]:
        times = data["pool"][b]
        done = sorted(t for t in times if t is not None)
        n = len(times)
        # step CDF: fraction of scenarios completed by time t (failures never complete)
        xs = [0.0] + done + [max_time]
        ys = [0.0] + [(i + 1) / n for i in range(len(done))] + [len(done) / n]
        plt.step(xs, ys, where="post", color=C.batch_color(b), label=f"M={b}")
    plt.xlabel("Time [s]")
    plt.ylabel("Fraction completed")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Batch Size", fontsize=9)
    plt.tight_layout()
    C.savefig(fig, "fig7_pickplace_cdf")


def main():
    p = argparse.ArgumentParser(description="Regenerate Fig-7 + Table-I (CS3 pick-place).")
    C.add_repro_args(p)
    p.add_argument("--n-scenarios", type=int, default=100)
    p.add_argument("--batch-sizes", default="1,4,8,16,32,64,128")
    p.add_argument("--max-time", type=float, default=25.0, help="CDF x-axis cap [s]")
    args = p.parse_args()
    np.random.seed(args.seed)

    if args.replot:
        data = C.load_data("fig7_pickplace")
    else:
        n_scenarios = args.n_scenarios
        batch_sizes = C.parse_int_list(args.batch_sizes)
        if args.quick:
            n_scenarios, batch_sizes = 2, [1, 8]
            print("[quick] tiny subset — NOT paper numbers")
        data = run(n_scenarios, batch_sizes, args.max_time)
        C.save_data(data, "fig7_pickplace")

    table_I(data)
    plot_cdf(data, args.max_time)


if __name__ == "__main__":
    main()

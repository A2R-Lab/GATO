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
>>> PROTOCOL (2026-08-02): the randomized 100-scenario 15 kg protocol above has NO
committed paper-era implementation (archaeology: the Sep-2025 experiment branches run
single fixed-seed tuned scenarios — indy7 10 kg/0.6 m, iiwa14 as light as 0-8 kg;
`sample_pendulum_params` first appears June 2026 on ICRA-26). It is a June-2026
protocol that is strictly harder than anything the paper code demonstrably ran.
The pendulum-distribution knobs below (--pend-mass, --length-range, --damping-range,
--angle-range) select the protocol; the pkl + Table-I header record it. Pools from
different protocols must never be mixed. Taxonomy (2026-08-02, seed-0 pool): SHORT
length is the dominant hardness axis (r=+0.43 success~length; swing frequency vs FE
bandwidth), angle is uncorrelated; per-(scenario,B) success is knife-edge (94/100
scenarios flip outcome across batch sizes).

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


def run(n_scenarios, batch_sizes, max_time, protocol, fc_config=None, wrench_id=None,
        start_config='ready'):
    from _pickplace_runner import ExperimentRunner
    from _common import (PICKPLACE_DEFAULT_GOALS, PICKPLACE_SOLVER_PARAMS,
                         PICKPLACE_MPC_DEFAULTS, sample_pendulum_params)

    urdf = C.URDFS["iiwa14"]
    C.require_module("iiwa14", N)
    runner = ExperimentRunner(urdf)

    # per-batch pools of episode completion times (None == failed/timeout) +
    # per-goal outcomes ('reached'/'timeout' per goal — the failure taxonomy)
    pool = {b: [] for b in batch_sizes}
    goal_outcomes = {b: [] for b in batch_sizes}
    scenarios = []
    for s in range(n_scenarios):
        pend = sample_pendulum_params(length_range=protocol["length_range"],
                                      damping_range=protocol["damping_range"],
                                      angle_range=protocol["angle_range"],
                                      mass=protocol["mass"])
        scenarios.append({k: (v.copy() if hasattr(v, "copy") else v) for k, v in pend.items()})
        print(f"scenario {s + 1}/{n_scenarios}  (L={pend['length']:.2f} d={pend['damping']:.2f})")
        res = runner.run_pickplace_sweep(
            batch_sizes=batch_sizes, N=N, dt=DT, sim_dt=0.001, plant_type="iiwa14",
            goal_sequences=[PICKPLACE_DEFAULT_GOALS], pendulum_config=pend,
            solver_params=PICKPLACE_SOLVER_PARAMS, mpc_defaults=PICKPLACE_MPC_DEFAULTS,
            fc_config=fc_config, wrench_id=wrench_id, start_config=start_config,
            verbose=False,
        )
        for b in batch_sizes:
            r = res.get(b, {})
            seq = (r.get("per_sequence") or [{}])[0]
            pool[b].append(seq.get("time_to_all_reached"))  # seconds, or None
            goal_outcomes[b].append(seq.get("goal_outcomes"))
    return {"batch_sizes": batch_sizes, "n_scenarios": n_scenarios, "pool": pool,
            "goal_outcomes": goal_outcomes, "scenarios": scenarios, "protocol": protocol,
            "fc_config": fc_config, "wrench_id": wrench_id}


def table_I(data):
    proto = data.get("protocol")
    plines = []
    if proto:
        plines = [f"protocol: mass={proto['mass']}kg L={proto['length_range']} "
                  f"d={proto['damping_range']} |th|={proto['angle_range']} "
                  f"start={proto.get('start_config', 'home')}"]
    fc, wid = data.get("fc_config"), data.get("wrench_id")
    if wid is None and not fc:
        plines.append("arm: ForceEstimator hypothesis batch (no fc slots)")
    else:
        # the combined arm is a real configuration (identified wrench sets the
        # f_ext bias, fc absorbs the residual) — record BOTH, never just one
        if wid is not None:
            plines.append(f"arm: least-squares wrench identification, wrench_id={wid}")
        if fc:
            plines.append(f"arm: solver contact-force slots, fc_config={fc}")
    lines = ["", "=" * 48, "TABLE I — pick-place success vs batch size", "=" * 48] + plines + [
             f"{'Batch':>6} {'Success [%]':>12} {'Mean time* [s]':>15}   (*successes only)"]
    for b in data["batch_sizes"]:
        times = data["pool"][b]
        done = [t for t in times if t is not None]
        sr = 100.0 * len(done) / len(times) if times else 0.0
        mt = float(np.mean(done)) if done else float("nan")
        row = f"{b:>6} {sr:>12.1f} {mt:>15.2f}"
        gos = (data.get("goal_outcomes") or {}).get(b)
        if gos and any(g for g in gos):
            # failure taxonomy: distribution of goals reached among FAILED episodes
            fails = [g for t, g in zip(times, gos) if t is None and g]
            if fails:
                hist = {}
                for g in fails:
                    k = sum(1 for o in g if o == "reached")
                    hist[k] = hist.get(k, 0) + 1
                row += "   failed@goals-reached " + " ".join(
                    f"{k}:{hist[k]}" for k in sorted(hist))
        lines.append(row)
    txt = "\n".join(lines)
    print(txt)
    import os
    with open(os.path.join(C.FIG_DIR, f"{data.get('tag', 'fig7_pickplace')}_table_I.txt"
                           if data.get("tag", "fig7_pickplace") != "fig7_pickplace"
                           else "table_I.txt"), "w") as f:
        f.write(txt + "\n")


def plot_cdf(data, max_time, tag="fig7_pickplace"):
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
    C.savefig(fig, f"{tag}_cdf")


def main():
    p = argparse.ArgumentParser(description="Regenerate Fig-7 + Table-I (CS3 pick-place).")
    C.add_repro_args(p)
    p.add_argument("--n-scenarios", type=int, default=100)
    p.add_argument("--batch-sizes", default="1,4,8,16,32,64,128")
    p.add_argument("--max-time", type=float, default=25.0, help="CDF x-axis cap [s]")
    p.add_argument("--pend-mass", type=float, default=15.0, help="pendulum mass [kg]")
    p.add_argument("--length-range", default="0.3,0.7", help="pendulum length range [m]")
    p.add_argument("--damping-range", default="0.1,0.6", help="damping range [Nms/rad]")
    p.add_argument("--angle-range", default="0.0,0.6", help="initial |axis-angle| range [rad]")
    p.add_argument("--tag", default="fig7_pickplace",
                   help="data/plot basename (use a distinct tag per protocol — never mix pools)")
    p.add_argument("--start-config", default="ready",
                   help="IIWA14_START_CONFIGS key for the initial pose. Default 'ready' is a "
                        "mid-workspace elbow pose; 'zero'/'home' are all-zeros, where the arm "
                        "is vertical and a hanging payload is UNOBSERVABLE (|J^T w| = 0).")
    p.add_argument("--wrench-id", action="store_true",
                   help="wrench-IDENTIFICATION arm: least-squares fit of the disturbance "
                        "wrench from sensor-rate motion, injected as f_ext. B=1 only "
                        "(replaces the ForceEstimator batch).")
    p.add_argument("--wrench-id-alpha", type=float, default=None,
                   help="EMA smoothing for --wrench-id (default = identifier default)")
    p.add_argument("--wrench-id-tau", type=float, default=None,
                   help="weight-filter time constant [s] for --wrench-id-mode weight")
    p.add_argument("--wrench-id-mode", default=None, choices=["wrench", "weight"],
                   help="--wrench-id disturbance model: full wrench, or only its "
                        "gravity-aligned (horizon-constant) component")
    p.add_argument("--fc", action="store_true",
                   help="contact-force arm: the SOLVER's fc slots explain the payload "
                        "(needs a GATO_CONTACT_FORCES module; no ForceEstimator). "
                        "Pools from the fc and FE arms share a protocol but not a solver "
                        "— tag them apart.")
    p.add_argument("--fc-cost", type=float, default=1e-2,
                   help="fc regularization weight for --fc (default = the build default)")
    p.add_argument("--fc-free-torque", action="store_true",
                   help="with --fc, leave the wrench moment rows free (default pins "
                        "them to zero: a point-mass payload exerts pure force)")
    args = p.parse_args()
    np.random.seed(args.seed)

    if args.replot:
        data = C.load_data(args.tag)
    else:
        n_scenarios = args.n_scenarios
        batch_sizes = C.parse_int_list(args.batch_sizes)
        if args.quick:
            n_scenarios, batch_sizes = 2, [1, 8]
            print("[quick] tiny subset — NOT paper numbers")
        rng = lambda s: tuple(float(x) for x in s.split(","))
        protocol = {"mass": args.pend_mass, "length_range": rng(args.length_range),
                    "damping_range": rng(args.damping_range),
                    "angle_range": rng(args.angle_range), "seed": args.seed,
                    "start_config": args.start_config}
        wrench_id = None
        if args.wrench_id:
            wrench_id = {}
            if args.wrench_id_alpha is not None:
                wrench_id["alpha"] = args.wrench_id_alpha
            if args.wrench_id_mode is not None:
                wrench_id["mode"] = args.wrench_id_mode
            if args.wrench_id_tau is not None:
                wrench_id["weight_tau"] = args.wrench_id_tau
            print(f"[wrench-id arm] least-squares wrench identification: {wrench_id}")
        fc_config = None
        if args.fc:
            fc_config = {"cost": args.fc_cost,
                         "pin_torque_rows": not args.fc_free_torque}
            print(f"[fc arm] solver contact-wrench slots active: {fc_config}")
        data = run(n_scenarios, batch_sizes, args.max_time, protocol, fc_config, wrench_id,
                   start_config=args.start_config)
        data["tag"] = args.tag
        C.save_data(data, args.tag)

    table_I(data)
    plot_cdf(data, args.max_time, tag=data.get("tag", "fig7_pickplace"))


if __name__ == "__main__":
    main()

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

Reproduction status (2026-09-27): runnable, not an exact paper reproduction.
The current loop uses fixed simulation pacing and a Euclidean joint-velocity
norm. Historical local 100-scenario results exist, but their success magnitudes
differ from Table I. Reconcile success aggregation, initial conditions, pacing
and force-estimator settings before attributing the gap to solver performance
or claiming this is a strictly harder protocol. Earlier force/frame/metric bugs
were fixed; older pools are not interchangeable with present runs. See README.md
for the current protocol checklist. Full Fig-7 refresh remains deferred; the
merge checkpoint is only ten seeded B128 scenarios, not a success-rate estimate.

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
        start_config='ready', estimator='fe', solver_params=None, mpc_defaults=None):
    from _pickplace_runner import (ExperimentRunner, PICKPLACE_DEFAULT_GOALS,
                                   PICKPLACE_SOLVER_PARAMS, PICKPLACE_MPC_DEFAULTS,
                                   sample_pendulum_params)

    C.require_module("iiwa14", N)
    runner = ExperimentRunner("iiwa14")
    solver_params = {**PICKPLACE_SOLVER_PARAMS, **(solver_params or {})}
    mpc_defaults = {**PICKPLACE_MPC_DEFAULTS, **(mpc_defaults or {})}

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
            solver_params=solver_params, mpc_defaults=mpc_defaults,
            fc_config=fc_config, wrench_id=wrench_id, start_config=start_config,
            verbose=False, estimator=estimator,
        )
        for b in batch_sizes:
            r = res.get(b, {})
            seq = (r.get("per_sequence") or [{}])[0]
            pool[b].append(seq.get("time_to_all_reached"))  # seconds, or None
            goal_outcomes[b].append(seq.get("goal_outcomes"))
    return {"simulation_protocol": "unit-quaternion-pendulum-v2",
            "source": C.bench.git_provenance(),
            "solver_params": solver_params, "mpc_defaults": mpc_defaults,
            "batch_sizes": batch_sizes, "n_scenarios": n_scenarios, "pool": pool,
            "goal_outcomes": goal_outcomes, "scenarios": scenarios, "protocol": protocol,
            "fc_config": fc_config, "wrench_id": wrench_id, "estimator": estimator}


def table_I(data):
    proto = data.get("protocol")
    plines = []
    if proto:
        plines = [f"protocol: mass={proto['mass']}kg L={proto['length_range']} "
                  f"d={proto['damping_range']} |th|={proto['angle_range']} "
                  f"start={proto.get('start_config', 'home')}"]
    plines.append(f"simulation: {data.get('simulation_protocol', 'legacy/unversioned (not v2)')}")
    md, sp = data.get("mpc_defaults") or {}, data.get("solver_params") or {}
    if md:
        plines.append("task: " + ("stop at each goal (reference ramps, gates hold)" if md.get("settle_time", 0.0) > 0
                                  else "pass through each goal (the paper's step reference, instantaneous gate)"))
    if md or sp:
        plines.append(f"gates: {md.get('goal_threshold')} m, {md.get('velocity_threshold')} rad/s "
                      f"(L{md.get('velocity_norm')}), dwell {md.get('settle_time', 0.0)} s, ramp {md.get('goal_ramp', 0.0)} s, "
                      f"timeout {md.get('goal_timeout')} s; qd_cost={sp.get('qd_cost')}")
    fc, wid = data.get("fc_config"), data.get("wrench_id")
    if wid is None and not fc:
        plines.append("arm: IdentifiedWrenchSampler hypothesis batch (identified payload weight + bounded exploration)"
                      if data.get("estimator") == "wid" else
                      "arm: ForceEstimator hypothesis batch (no fc slots)")
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


def plot_success_vs_batch(panels, name="fig7_success_vs_batch"):
    """Success rate against batch size, one panel per task; each panel draws the task's
    chosen sampler solid and the other sampler dashed when its pool exists.
    panels = [(title, [(label, tag, style), ...]), ...]; missing pools are skipped."""
    plt = C.set_paper_rcParams()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels):
        for label, tag, style in series:
            try:
                d = C.load_data(tag)
            except FileNotFoundError:
                continue
            Bs = d["batch_sizes"]
            rate = [100.0 * np.mean([t is not None for t in d["pool"][b]]) for b in Bs]
            ax.plot(Bs, rate, style, label=label)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 8, 32, 128]); ax.set_xticklabels(["1", "8", "32", "128"])
        ax.set_ylim(0, 102); ax.set_xlabel("Batch size B"); ax.set_ylabel("Episodes with all 5 goals [%]")
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    C.savefig(fig, name)


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
    p.add_argument("--task", default="stop", choices=["stop", "pass-through"],
                   help="protocol preset (2026-10-03). 'stop': the payload must be set down — minimum-jerk "
                        "reference between goals (1.5 s), gates hold 100 ms, 15 kg, the paper's exploration "
                        "sampler (fe). 'pass-through': the paper's protocol verbatim — step goals, instantaneous "
                        "gate — with the identified-weight sampler (wid). Explicit --estimator/--goal-ramp/"
                        "--settle-time/--pend-mass override the preset.")
    p.add_argument("--tag", default=None,
                   help="data/plot basename (default fig7_<task>; use a distinct tag per protocol — never mix pools)")
    p.add_argument("--start-config", default="ready",
                   help="IIWA14_START_CONFIGS key for the initial pose. Default 'ready' is a "
                        "mid-workspace elbow pose; 'zero'/'home' are all-zeros, where the arm "
                        "is vertical and a hanging payload is UNOBSERVABLE (|J^T w| = 0).")
    p.add_argument("--settle-time", type=float, default=None,
                   help="dwell [s] both success gates must hold (default: PICKPLACE_MPC_DEFAULTS; "
                        "0 = the paper's instantaneous gate)")
    p.add_argument("--goal-ramp", type=float, default=None,
                   help="minimum-jerk EE reference travel time [s] between goals (default: "
                        "PICKPLACE_MPC_DEFAULTS; 0 = the paper's step reference)")
    p.add_argument("--qd-cost", type=float, default=None,
                   help="joint-velocity cost override (default: PICKPLACE_SOLVER_PARAMS)")
    p.add_argument("--estimator", default=None, choices=["fe", "wid"],
                   help="hypothesis sampler behind the batch (default: the --task preset): 'fe' = the paper's "
                        "ForceEstimator (bounded exploration, never identifies the load — the right fill when "
                        "the arm must stop with the load swinging); 'wid' = IdentifiedWrenchSampler (identified "
                        "payload weight + bounded exploration — the right fill when flying through the goals).")
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
                        "(runs the fc module variant bsqpN16_iiwa14_fc; no ForceEstimator). "
                        "Pools from the fc and FE arms share a protocol but not a solver "
                        "— tag them apart.")
    p.add_argument("--fc-cost", type=float, default=1e-2,
                   help="fc regularization weight for --fc (default = the build default)")
    p.add_argument("--fc-free-torque", action="store_true",
                   help="with --fc, leave the wrench moment rows free (default pins "
                        "them to zero: a point-mass payload exerts pure force)")
    p.add_argument("--success-plot", action="store_true",
                   help="also render fig7_success_vs_batch.png from the two preset pools (and the secondary pools when present)")
    args = p.parse_args()
    preset = {"stop": dict(estimator="fe", goal_ramp=1.5, settle_time=0.1),
              "pass-through": dict(estimator="wid", goal_ramp=0.0, settle_time=0.0)}[args.task]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.tag is None:
        args.tag = "fig7_" + args.task.replace("-", "_")
    np.random.seed(args.seed)

    if args.replot:
        data = C.load_data(args.tag)
        data["tag"] = args.tag
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
        overrides_s = {"qd_cost": args.qd_cost} if args.qd_cost is not None else {}
        overrides_m = {k: v for k, v in (("settle_time", args.settle_time), ("goal_ramp", args.goal_ramp)) if v is not None}
        data = run(n_scenarios, batch_sizes, args.max_time, protocol, fc_config, wrench_id,
                   start_config=args.start_config, estimator=args.estimator,
                   solver_params=overrides_s, mpc_defaults=overrides_m)
        data["tag"] = args.tag
        C.save_data(data, args.tag)

    table_I(data)
    plot_cdf(data, args.max_time, tag=data.get("tag", "fig7_pickplace"))
    if args.success_plot:
        plot_success_vs_batch([
            ("Stop at each goal\n(15 kg, 1.5 s reference, gates hold 0.1 s)",
             [("exploration sampler (paper's)", "fig7_stop", "o-"),
              ("identified weight + exploration", "fig7_stop_wid", "s--")]),
            ("Pass through each goal\n(paper protocol, 15 kg)",
             [("identified weight + exploration", "fig7_pass_through", "o-"),
              ("exploration sampler (paper's)", "fig7_pickplace_v2_fe", "s--")]),
        ])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Joint-feasible arm MPC: the dev set and the frozen holdout (plan H2.5).

Closed-loop episodes on tasks sized to the URDF limits (gato.tasks: figure-8
tracking with the period set from the demanded joint speed, pick-and-place
arrival with a step or ramped reference), on the Indy7 and the iiwa14, at
N32 and N64, under each limit mechanism of MPCController(limits=...):

    none        telemetry rows (the paper setting; limits reported, not enforced)
    barrier     relaxed log barrier (soft)
    al          augmented Lagrangian (outer loop = the warm-started ticks)
    admm        ADMM projection, the controller default (per-box rho 0.1 /
                0.03 / 1e-5 since the dev set of October 16, residual exit 1e-2)
    and, on the dev set only, the strength sweeps (MECHS below: ADMM rho 1e-3
    to 3e-2, AL rho 10, barrier mu 1e-2, ADMM at eps_rel 1e-3 and with the
    fixed budget) and three SQP iterations per tick (none / admm / al).

Every episode is fixed pacing (one solve per 10 ms control period, rk4 plant
at 1 ms, pinocchio) and bit-deterministic, so the results do not depend on
what else runs on the GPU; the recorded solve times are diagnostics and are
never quotable. Graded on the SIMULATED plant against the URDF limits (the
planner's own telemetry is recorded beside it):

    sim_qd_ratio   max over ticks and joints of |qd| / qd_limit  (1 = at the limit)
    sim_u_ratio    the same for the applied torque
    sim_q_out      max position excursion beyond [q_lo, q_hi], rad
    plan_viol      the planner's max violation per row group (q, qd, u rows)
    track_mean     figure-8 mean end-effector error (m, knot-0 convention)
    reached / time_to_all   pick-and-place goals reached of 3, task time (s)

Protocol (Fig-7 holdout discipline): the DEV set (seed 1, 20 scenarios per
task and plant) compares the mechanisms and settles the controller defaults;
the HOLDOUT (seed 20261016, 100 scenarios) is drawn fresh and run once with
the frozen defaults, published as-is, no retuning.

    python examples/benchmarks/limits_holdout.py --set dev --workers 4
    python examples/benchmarks/limits_holdout.py --set holdout --workers 4
    python examples/benchmarks/limits_holdout.py --report data/limits_holdout/dev.jsonl

Rows: one JSON line per episode in --out (default
examples/benchmarks/data/limits_holdout/<set>.jsonl), with scenario, config,
metrics, git SHA and date. --workers runs the (plant, N) shards as parallel
subprocesses appending to the same file.
"""
import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from _bench import git_provenance, pct, pin_model, urdf_path

DATA = Path(__file__).resolve().parent / "data" / "limits_holdout"
DT, SIM_DT = 0.01, 0.001
PLANTS, NS, TASKS = ("indy7", "iiwa14"), (32, 64), ("fig8", "pickplace")
# Configurations: MPC_GATO keyword arguments; "params" overrides SolverParams
# (the MPC default is one SQP iteration per tick). The dev set sweeps the
# mechanisms' strength knobs (ADMM and AL rho, barrier mu) and the SQP
# iterations per tick; the holdout runs the four frozen defaults.
MECHS = {
    "none": dict(limits="telemetry"),
    "barrier": dict(limits="barrier"),
    "barrier-mu1e-2": dict(limits="barrier", limits_kw=dict(mu=1e-2)),
    "al": dict(limits="al"),
    "al-r10": dict(limits="al", limits_kw=dict(rho=10.0)),
    "admm": dict(limits="admm"),
    "admm-r3e-3": dict(limits="admm", limits_kw=dict(rho=3e-3)),
    "admm-r1e-3": dict(limits="admm", limits_kw=dict(rho=1e-3)),
    "admm-r3e-2": dict(limits="admm", limits_kw=dict(rho=3e-2)),
    "admm-1e-3": dict(limits="admm", limits_kw=dict(eps_rel=1e-3)),
    "admm-fixed": dict(limits="admm", limits_kw=dict(eps_rel=None)),
    # per-box rho at each block's cost scale (set_row_group_rho; q rows ~ the
    # EE tracking Hessian, qd rows ~ qd_cost 1e-2, u rows ~ u_cost 2e-6)
    "admm-kind": dict(limits="admm", limits_kw=dict(rho_q=0.1, rho_qd=0.01, rho_u=1e-5)),
    "admm-kind-u1e-4": dict(limits="admm", limits_kw=dict(rho_q=0.1, rho_qd=0.01, rho_u=1e-4)),
    "admm-kind-u2e-6": dict(limits="admm", limits_kw=dict(rho_q=0.1, rho_qd=0.01, rho_u=2e-6)),
    "admm-kind-qd3e-2": dict(limits="admm", limits_kw=dict(rho_q=0.1, rho_qd=0.03, rho_u=1e-5)),
    "admm-kind-sqp3": dict(limits="admm", limits_kw=dict(rho_q=0.1, rho_qd=0.01, rho_u=1e-5),
                           params=dict(max_sqp_iters=3)),
    "al-kind": dict(limits="al", limits_kw=dict(rho_q=10.0, rho_qd=1.0, rho_u=1e-3)),
    "none-sqp3": dict(limits="telemetry", params=dict(max_sqp_iters=3)),
    "admm-sqp3": dict(limits="admm", params=dict(max_sqp_iters=3)),
    "al-sqp3": dict(limits="al", params=dict(max_sqp_iters=3)),
}
SETS = {"dev": dict(seed=1, n=20, mechs=list(MECHS)),
        "holdout": dict(seed=20261016, n=100, mechs=["none", "barrier", "al", "admm"])}
PICKPLACE = dict(goal_timeout=4.0, goal_threshold=0.05, velocity_threshold=1.0, velocity_norm=2,
                 settle_time=0.1, pace_by_solve_time=False)


def scenarios(model, plant, task, n, seed):
    from gato import tasks
    if task == "fig8":
        return tasks.fig8_scenarios(model, plant, n, seed=seed, dt=DT)
    return tasks.pickplace_scenarios(model, plant, n, seed=seed)


def run_episode(model, plant, N, sc, mech):
    """One closed-loop episode; returns the metrics dict."""
    from gato import tasks
    from gato.mpc_gato import MPC_GATO
    lim = tasks.urdf_limits(model)
    cfg = dict(MECHS[mech])
    params = cfg.pop("params", None)
    mpc = MPC_GATO(model, model_path=urdf_path(plant), N=N, dt=DT, batch_size=1, plant_type=plant,
                   linsys="pcg", params=params, **cfg)
    rec = dict(u=[], viol=[], inner=[], solve_us=[])
    orig = mpc.controller.step

    def step(x, g, **kw):
        r = orig(x, g, **kw)
        rec["u"].append(np.asarray(r.u, dtype=float))
        rec["viol"].append(np.asarray(r.row_viol, dtype=float))
        rec["solve_us"].append(r.solve.solve_time_us)
        if r.solve.stats.admm_iters is not None:
            rec["inner"].append(int(r.solve.stats.admm_iters.max()))
        return r
    mpc.controller.step = step

    x0 = np.hstack((np.asarray(sc["q0"], dtype=float), np.zeros(model.nv)))
    out = dict(mech=mech, config=MECHS[mech], limits=mpc.controller.limits,
               admm_linsys=mpc.controller.admm_linsys)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if sc["task"] == "fig8":
            traj, sim_time = tasks.fig8_reference(sc, N)
            _, st = mpc.run_mpc_fig8(x0, traj, sim_dt=SIM_DT, sim_time=sim_time, pace_by_solve_time=False)
            err = np.asarray(st["goal_distances_knot0"], dtype=float)
            out.update(track_mean=float(err.mean()), track_max=float(err.max()))
        else:
            goals = [np.asarray(g, dtype=float) for g in sc["goals"]]
            _, st = mpc.run_mpc_goals(x0, goals, sim_dt=SIM_DT, goal_ramp=sc["ramp"], **PICKPLACE)
            out.update(reached=int(sum(o == "reached" for o in st["goal_outcomes"])),
                       time_to_all=st["time_to_all_reached"],
                       final_dist=float(st["goal_events"][-1]["distance"]) if st["goal_events"] else None)
    qd = np.abs(np.asarray(st["joint_velocities"], dtype=float)) / lim["qd"]
    q = np.asarray(st["joint_positions"], dtype=float)
    q_out = np.maximum(np.maximum(q - lim["q_hi"], lim["q_lo"] - q), 0.0)
    u = np.abs(np.asarray(rec["u"], dtype=float)) / lim["u"]
    viol = np.asarray(rec["viol"], dtype=float)
    out.update(steps=int(len(qd)), wall_s=time.perf_counter() - t0,
               sim_qd_ratio=float(qd.max()), sim_qd_over_frac=float((qd.max(axis=1) > 1.0).mean()),
               sim_u_ratio=float(u.max()), sim_u_over_frac=float((u.max(axis=1) > 1.0).mean()),
               sim_q_out=float(q_out.max()),
               plan_viol=viol.max(axis=0).tolist(), plan_viol_mean=viol.mean(axis=0).tolist(),
               inner_mean=float(np.mean(rec["inner"])) if rec["inner"] else None,
               solve_ms=pct(np.asarray(rec["solve_us"], dtype=float) / 1e3))
    return out


def _append(out, row):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def run_shard(args, plant, N):
    cfg = SETS[args.set]
    n, seed = args.n or cfg["n"], args.seed or cfg["seed"]
    mechs = args.mechs.split(",") if args.mechs else cfg["mechs"]
    model = pin_model(plant)
    prov = git_provenance()
    out = Path(args.out)
    for task in args.tasks.split(","):
        scs = scenarios(model, plant, task, n, seed)
        for sc in scs:
            for mech in mechs:
                m = run_episode(model, plant, N, sc, mech)
                row = dict(set=args.set, seed=seed, scenario=sc["id"], task=task, plant=plant, N=N,
                           scenario_def={k: v for k, v in sc.items() if k not in ("id", "plant", "task")},
                           sha=prov["short"] + ("+dirty" if prov["dirty"] else ""), date=prov["date"],
                           quotable_times=False, **m)
                _append(out, row)
                extra = (f"track {m['track_mean']:.4f}" if task == "fig8" else
                         f"reached {m['reached']}/3 t_all {m['time_to_all']}")
                print(f"{plant} N{N} {task} #{sc['id']:3d} {mech:<10} qd {m['sim_qd_ratio']:.2f} u {m['sim_u_ratio']:.2f}"
                      f" viol_qd {m['plan_viol'][1]:.1e} {extra}  ({m['wall_s']:.1f}s)", flush=True)


# ---- report ----------------------------------------------------------------

def _q(v, p):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, p)) if v else float("nan")


def report(paths):
    rows = [json.loads(ln) for p in paths for ln in Path(p).read_text().splitlines() if ln.strip()]
    keys = sorted({(r["set"], r["task"], r["plant"], r["N"]) for r in rows}, key=str)
    mech_order = [m for m in MECHS if any(r["mech"] == m for r in rows)]
    for key in keys:
        sub = [r for r in rows if (r["set"], r["task"], r["plant"], r["N"]) == key]
        task = key[1]
        print(f"\n### {key[0]} set, {task}, {key[2]} N{key[3]}  ({len({r['scenario'] for r in sub})} scenarios)\n")
        head = "| mechanism | n | sim qd/limit p50 / p90 / max | ticks over qd limit | sim u/limit p50 / max | plan viol qd p50 / max | plan viol u max | "
        head += "track mean p50 / p90 (m) |" if task == "fig8" else "reached mean (of 3) | time to all p50 (s) |"
        head += " inner it/tick | solve ms p50 (diag) |"
        print(head)
        print("|" + "---|" * (head.count("|") - 1))
        for mech in mech_order:
            g = [r for r in sub if r["mech"] == mech]
            if not g:
                continue
            qd = [r["sim_qd_ratio"] for r in g]
            line = (f"| {mech} | {len(g)} | {_q(qd, 50):.2f} / {_q(qd, 90):.2f} / {max(qd):.2f} "
                    f"| {100 * np.mean([r['sim_qd_over_frac'] for r in g]):.1f} % "
                    f"| {_q([r['sim_u_ratio'] for r in g], 50):.2f} / {max(r['sim_u_ratio'] for r in g):.2f} "
                    f"| {_q([r['plan_viol'][1] for r in g], 50):.1e} / {max(r['plan_viol'][1] for r in g):.1e} "
                    f"| {max(r['plan_viol'][2] for r in g):.1e} | ")
            if task == "fig8":
                tm = [r["track_mean"] for r in g]
                line += f"{_q(tm, 50):.4f} / {_q(tm, 90):.4f} |"
            else:
                line += (f"{np.mean([r['reached'] for r in g]):.2f} | "
                         f"{_q([r['time_to_all'] for r in g], 50):.2f} |")
            inner = [r["inner_mean"] for r in g if r["inner_mean"] is not None]
            line += f" {np.mean(inner):.1f} |" if inner else " |"
            line += f" {_q([r['solve_ms']['p50'] for r in g], 50):.2f} |"
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev", choices=list(SETS))
    ap.add_argument("--n", type=int, default=None, help="scenarios per task and plant (default: the set's)")
    ap.add_argument("--seed", type=int, default=None, help="scenario seed (default: the set's)")
    ap.add_argument("--plants", default=",".join(PLANTS))
    ap.add_argument("--Ns", default=",".join(map(str, NS)))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--mechs", default=None, help="comma list (default: the set's)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1, help="parallel (plant, N) shard subprocesses")
    ap.add_argument("--shard", default=None, help="internal: run one plant:N shard in this process")
    ap.add_argument("--report", nargs="*", help="render the tables from these jsonl files and exit")
    args = ap.parse_args()
    if args.report is not None:
        report(args.report or [DATA / f"{args.set}.jsonl"])
        return
    args.out = args.out or str(DATA / f"{args.set}.jsonl")
    if args.shard:
        plant, N = args.shard.split(":")
        run_shard(args, plant, int(N))
        return
    shards = [(p, int(N)) for p in args.plants.split(",") for N in args.Ns.split(",")]
    if args.workers <= 1:
        for plant, N in shards:
            run_shard(args, plant, N)
        return
    base = [sys.executable, str(Path(__file__).resolve()), "--set", args.set, "--tasks", args.tasks, "--out", args.out]
    for k in ("n", "seed", "mechs"):
        if getattr(args, k) is not None:
            base += [f"--{k}", str(getattr(args, k))]
    running, failed = [], 0
    for plant, N in shards:
        while len(running) >= args.workers:
            for pr in list(running):
                if pr.poll() is not None:
                    running.remove(pr); failed += pr.returncode != 0
            time.sleep(1.0)
        print(f"== shard {plant}:{N}", flush=True)
        running.append(subprocess.Popen(base + ["--shard", f"{plant}:{N}"]))
    for pr in running:
        failed += pr.wait() != 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

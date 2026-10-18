#!/usr/bin/env python3
"""Fixed-pacing torque saturation A/B on development scenarios, not a benchmark.

Reuse limits_holdout's tasks and unchanged ADMM controller settings at N64.
Only the simulated actuator changes: pass through versus clip to URDF effort
limits. Sample initial state and every 1 ms substep endpoint, including motion
from warmup and the final goal transition. This is not a continuous-time or
hardware certificate. No controller or default simulator behavior is changed.

    python examples/benchmarks/torque_clip_ab.py --out new-clip-ab.jsonl
    python examples/benchmarks/torque_clip_ab.py --report new-clip-ab.jsonl

One worker; existing output files are refused. Partial files are preserved,
not silently resumed. Runtime diagnostics are omitted from saved results.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from _bench import git_provenance, pin_model
from limits_holdout import DT, SIM_DT, PICKPLACE, run_episode, scenarios

PLANTS = ("fr3", "gen3", "ur5e")
TASKS = ("fig8", "pickplace")


class ActuationAudit:
    """Experiment-only fixed-base actuator wrapper; never modifies a plan in place."""
    def __init__(self, model, clip, *, effort_scale=1.0):
        from gato.tasks import urdf_limits
        if model.nq != model.nv:
            raise ValueError("this audit requires a fixed-base arm")
        self.limits = urdf_limits(model)
        if any(not np.isfinite(self.limits[k]).all() or np.any(self.limits[k] <= 0)
               for k in ("u", "qd")):
            raise ValueError("finite positive effort and velocity limits required")
        self.clip = bool(clip)
        if not np.isfinite(effort_scale) or not 0 < effort_scale <= 1:
            raise ValueError("effort_scale must be in (0, 1]")
        self.actuator_limit = self.limits["u"] * effort_scale
        self.samples = self.clipped = self.velocity_over = 0
        self.command_peak = self.applied_peak = self.velocity_peak = self.position_out = 0.0
        self.sim_time = 0.0
        self.last_applied = None

    def _state(self, q, dq):
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise FloatingPointError("nonfinite simulated state")
        velocity = float(np.max(np.abs(dq) / self.limits["qd"]))
        self.velocity_peak = max(self.velocity_peak, velocity)
        self.position_out = max(self.position_out, float(np.max(np.maximum(
            self.limits["q_lo"] - q, q - self.limits["q_hi"]))), 0.0)
        return velocity

    def attach(self, mpc):
        if self.last_applied is not None:
            raise ValueError("use a fresh audit for each episode")
        original_step, original_observe = mpc.world.step, mpc._observe_substep

        def step(q, dq, u, dt):
            if not np.isfinite(u).all():
                raise FloatingPointError("nonfinite command")
            if not self.samples:
                self._state(q, dq)  # include the initial condition
            applied = np.clip(u, -self.actuator_limit, self.actuator_limit) if self.clip else u
            self.last_applied = np.asarray(applied).copy()
            self.command_peak = max(self.command_peak, float(np.max(np.abs(u) / self.limits["u"])))
            self.applied_peak = max(self.applied_peak, float(np.max(np.abs(applied) / self.limits["u"])))
            self.clipped += int(np.any(applied != u))
            q_next, dq_next = original_step(q, dq, applied, dt)
            self.velocity_over += int(self._state(q_next, dq_next) > 1.0)
            self.samples += 1
            self.sim_time += dt
            return q_next, dq_next

        def observe(q_pre, dq_pre, q_post, dq_post, tau, dt):
            # An observer must see the applied torque, not the unclipped command.
            return original_observe(q_pre, dq_pre, q_post, dq_post, self.last_applied, dt)

        mpc.world.step, mpc._observe_substep = step, observe

    def metrics(self):
        return dict(samples=self.samples, simulated_s=self.sim_time,
                    commanded_u_ratio=self.command_peak, applied_u_ratio=self.applied_peak,
                    qd_ratio=self.velocity_peak, q_out_rad=self.position_out,
                    clipped_samples=self.clipped, velocity_over_samples=self.velocity_over)


def validate(rows):
    """Require the complete paired inventory before reporting aggregate results."""
    if not rows:
        raise ValueError("empty collection")
    protocol, source = rows[0]["protocol"], rows[0]["source"]
    expected = set(itertools.product(protocol["plants"], protocol["tasks"],
                                     range(protocol["n"]), (False, True)))
    keys = [(r["plant"], r["task"], r["scenario"], r["clip"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("incomplete or duplicate paired inventory")
    paired = {}
    for r in rows:
        if r["protocol"] != protocol or r["source"] != source or r["quotable_times"] is not False:
            raise ValueError("mixed provenance or protocol")
        key = r["plant"], r["task"], r["scenario"]
        if paired.setdefault(key, r["scenario_def"]) != r["scenario_def"]:
            raise ValueError("paired scenarios differ")
    return protocol


def report(rows):
    protocol = validate(rows)
    print("Correctness only; times below are simulated task durations, not runtimes.")
    print("plant task clip ok/total goals-complete u-command-max u-applied-max qd-max q-out-max clipped-episodes task-metric-p50")
    for plant, task, clip in itertools.product(protocol["plants"], protocol["tasks"], (False, True)):
        group = [r for r in rows if (r["plant"], r["task"], r["clip"]) == (plant, task, clip)]
        good = [r["result"] for r in group if r["status"] == "ok"]
        if not good:
            print(plant, task, clip, "all episodes failed")
            continue
        metric = "track_mean" if task == "fig8" else "time_to_all"
        values = [r[metric] for r in good if r[metric] is not None]
        peaks = [max(r["substeps"][k] for r in good) for k in
                 ("commanded_u_ratio", "applied_u_ratio", "qd_ratio", "q_out_rad")]
        print(plant, task, clip, f"{len(good)}/{len(group)}",
              sum(r.get("reached") == 3 for r in good) if task == "pickplace" else "-",
              *(f"{v:.6f}" for v in peaks), sum(r["substeps"]["clipped_samples"] > 0 for r in good),
              f"{np.median(values):.6f}" if values else "-")
    print("Paired changes (clipped minus unclipped): figure-8 mean error in m; completion in simulated s.")
    for plant, task in itertools.product(protocol["plants"], protocol["tasks"]):
        group = [r for r in rows if (r["plant"], r["task"]) == (plant, task) and r["status"] == "ok"]
        a = {r["scenario"]: r["result"] for r in group if not r["clip"]}
        b = {r["scenario"]: r["result"] for r in group if r["clip"]}
        metric = "track_mean" if task == "fig8" else "time_to_all"
        deltas = [b[i][metric] - a[i][metric] for i in a.keys() & b.keys()
                  if a[i][metric] is not None and b[i][metric] is not None]
        print(plant, task, f"{len(deltas)}/{protocol['n']} pairs with both metrics",
              f"min/median/max {min(deltas):.6f} / {np.median(deltas):.6f} / {max(deltas):.6f}"
              if deltas else "no paired metric")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.report:
        report([json.loads(line) for line in args.report.read_text().splitlines()])
        return
    if args.out is None or args.n < 1 or args.seed < 0:
        ap.error("--out, positive --n and nonnegative --seed required")
    source = git_provenance()
    if source["dirty"]:
        ap.error("commit source changes before collecting the paired experiment")
    from gato.controller import MPCController
    protocol = dict(schema=1, set="dev", plants=list(PLANTS), tasks=list(TASKS),
                    n=args.n, seed=args.seed, N=64, dt=DT, sim_dt=SIM_DT,
                    mechanism="admm", limit_settings=MPCController.LIMIT_DEFAULTS["admm"],
                    linsys="pcg", admm_linsys="bdsv_factor", pickplace=PICKPLACE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.out.open("x") as out:
        for plant in PLANTS:
            model = pin_model(plant)
            for task in TASKS:
                for sc in scenarios(model, plant, task, args.n, args.seed):
                    for clip in (False, True):
                        row = dict(protocol=protocol, source=source, quotable_times=False,
                                   plant=plant, task=task, scenario=sc["id"], scenario_def=sc, clip=clip)
                        try:
                            result = run_episode(model, plant, 64, sc, "admm", audit=ActuationAudit(model, clip))
                            for key in ("wall_s", "solve_ms"):
                                result.pop(key)
                            if result["admm_linsys"] != protocol["admm_linsys"]:
                                raise ValueError("unexpected ADMM policy")
                            json.dumps(result, allow_nan=False)
                            row.update(status="ok", result=result)
                        except Exception as exc:
                            failed += 1
                            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print(plant, task, sc["id"], "clipped" if clip else "unclipped", row["status"], flush=True)
    report([json.loads(line) for line in args.out.read_text().splitlines()])
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

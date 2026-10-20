"""Host/device controller parity and latency on an identical moving-state replay.

Default mode is correctness: no duration samples are read, saved or reported.
Timing requires GATO_QUIET_WINDOW=1 and a clean source tree. Use the guarded
run_controller_replay.sh launcher in an assigned exclusive window.

This is a kinematic input replay, NOT a closed-loop simulation or task-success
experiment. All paths receive the same states and future FK references; B>1
duplicates the problem without hypotheses. Input creation/upload, controller
construction/reset and parity downloads are outside the measured boundary.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from _bench import REPO, git_provenance, pct

PLANTS = ("indy7", "iiwa14", "fr3", "gen3", "ur5e")
KINDS = ("numpy", "torch", "jax")
N, DT = 64, 0.01


def replay_inputs(plant, ticks):
    """Small sinusoidal joint motion about READY; reference = its future FK.

    No inverse dynamics, integration or controller is used to generate inputs.
    The hash covers actual float32 inputs, not just the generation recipe.
    """
    import gato
    import pinocchio as pin
    from gato.tasks import READY, ee_frame_of
    model = pin.buildModelFromUrdf(str(gato.robot_urdf(plant)))
    data = model.createData()
    phase = np.arange(model.nq) * 0.3
    t = np.arange(ticks + N)[:, None] * DT
    q = READY[plant] + 0.04 * (np.sin(0.5 * t + phase) - np.sin(phase))
    qd = 0.02 * np.cos(0.5 * t + phase)
    x = np.ascontiguousarray(np.concatenate((q[:ticks], qd[:ticks]), axis=1), dtype=np.float32)
    goals = np.zeros((ticks + N, 6), dtype=np.float32)
    frame = model.getFrameId(ee_frame_of(plant))
    if frame >= model.nframes:
        raise ValueError(f"unknown EE frame for {plant}")
    for k, pose in enumerate(q):
        pin.forwardKinematics(model, data, pose)
        pin.updateFramePlacements(model, data)
        goals[k, :3] = data.oMf[frame].translation
    refs = np.ascontiguousarray([goals[k:k + N].reshape(-1) for k in range(ticks)])
    digest = hashlib.sha256(x.tobytes() + refs.tobytes()).hexdigest()
    return x, refs, digest


def make_controller(plant, batch, limits):
    import gato
    # Require an in-tree receipt module; never launch an implicit compile.
    if not list((REPO / "python/gato").glob(f"bsqpN{N}_{plant}.*.so")):
        raise RuntimeError(f"build the receipt module {plant}:N{N} first")
    solver = gato.BSQP(gato.robot_urdf(plant), batch_size=batch, N=N, dt=DT,
                       plant_type=plant, max_sqp_iters=1)
    return gato.MPCController(solver, limits=limits)


def stage_inputs(kind, x, refs):
    if kind == "numpy":
        return x, refs
    if kind == "torch":
        import torch
        return torch.from_numpy(x).cuda(), torch.from_numpy(refs).cuda()
    import jax
    device = jax.devices("gpu")[0]
    return jax.device_put(x, device), jax.device_put(refs, device)


def run_path(plant, batch, limits, kind, inputs, warmup, *, timing=False):
    """Run from fresh controller state; hash complete plans and row telemetry.

    Timing is external host wall through device completion, plus the native
    solver's own elapsed metric. The external boundary includes the final
    synchronization, unlike StepResult.wall_s. No deadline budget is supplied.
    """
    import torch
    from gato import _arrays
    ctrl = make_controller(plant, batch, limits)
    x, refs = stage_inputs(kind, *inputs)
    ctrl.reset(x[0])
    torch.cuda.synchronize()  # finish staging before the replay
    digest = hashlib.sha256()
    walls, native, iterations, pred_err = [], [], [], []
    max_violation = 0.0
    for k in range(len(x)):
        # Indexing/device view construction is outside the call boundary.
        state, ref = x[k], refs[k]
        if kind == "jax":
            state.block_until_ready()
            ref.block_until_ready()
        if timing and k >= warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        result = ctrl.step(state, ref)
        if timing and k >= warmup:
            torch.cuda.synchronize()
            walls.append(1e3 * (time.perf_counter() - start))
            native.append(result.solve.solve_time_us / 1000.0)
        plan = np.ascontiguousarray(_arrays.to_numpy(result.solve.xu))
        if not np.isfinite(plan).all() or not np.isfinite(result.pred_err):
            raise RuntimeError(f"nonfinite result: {plant}/{batch}/{limits}/{kind} tick {k}")
        digest.update(plan.tobytes())
        viol = np.asarray(result.row_viol if result.row_viol is not None else [], dtype=np.float32)
        if not np.isfinite(viol).all():
            raise RuntimeError("nonfinite row telemetry")
        digest.update(viol.tobytes())
        digest.update(np.asarray([result.iterations, result.reseeded], dtype=np.int64).tobytes())
        max_violation = max(max_violation, float(viol.max(initial=0)))
        iterations.append(result.iterations)
        pred_err.append(result.pred_err)
        expected_limits = "" if limits == "off" else "admm"
        if result.limits != expected_limits:
            raise RuntimeError(f"unexpected effective limits: {result.limits!r}")
        if ctrl._xu_where != ("host" if kind == "numpy" else "device"):
            raise RuntimeError("controller took the wrong array path")
    row = dict(kind=kind, plan_telemetry_sha256=digest.hexdigest(),
               effective_limits=ctrl.limits, linsys=ctrl.linsys,
               bdsv_threshold=ctrl.bdsv_threshold, admm_linsys=ctrl.admm_linsys,
               admm_exit=[ctrl.solver.admm_eps_abs, ctrl.solver.admm_eps_rel,
                          ctrl.solver.admm_check_every],
               native_source_id=ctrl.solver.lib.NATIVE_SOURCE_ID,
               row_violation_max=max_violation, iterations=iterations,
               pred_err=pred_err)
    if timing:
        row.update(call_complete_ms=pct(walls), native_solve_ms=pct(native),
                   call_complete_samples_ms=walls, native_solve_samples_ms=native)
    return row


def check_parity(rows):
    baseline = next(r for r in rows if r["kind"] == "numpy")
    for row in rows:
        if row["plan_telemetry_sha256"] != baseline["plan_telemetry_sha256"]:
            raise RuntimeError(f"plan/telemetry parity failed for {row['kind']}")
        np.testing.assert_allclose(row["pred_err"], baseline["pred_err"], rtol=1e-5, atol=1e-6,
                                   err_msg=f"{row['kind']} warm-start signal")


def box_snapshot():
    """Fail closed on unavailable GPU telemetry; exclude only this process."""
    def query(field):
        return subprocess.run(["nvidia-smi", field, "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True).stdout.strip()
    pids = [int(p) for p in query("--query-compute-apps=pid").splitlines()]
    return dict(gpu=query("--query-gpu=name,uuid,driver_version,clocks.sm,temperature.gpu,power.limit"),
                foreign_pids=[p for p in pids if p != os.getpid()], load1=os.getloadavg()[0])


def validate_timing_state(provenance):
    if os.environ.get("GATO_QUIET_WINDOW") != "1":
        raise RuntimeError("timing requires an assigned window and GATO_QUIET_WINDOW=1")
    if provenance["dirty"] or git_provenance()["sha"] != provenance["sha"]:
        raise RuntimeError("timing requires unchanged, clean source")
    snap = box_snapshot()
    if snap["foreign_pids"] or snap["load1"] > 2.0:
        raise RuntimeError(f"box is not exclusive: {snap}")
    return snap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("correctness", "timing"), default="correctness")
    ap.add_argument("--plants", nargs="+", choices=PLANTS, default=list(PLANTS))
    ap.add_argument("--batches", nargs="+", type=int, choices=(1, 8), default=[1, 8])
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True, help="new JSONL path; never overwrite or append")
    args = ap.parse_args()
    if args.samples < 1 or args.warmup < 1 or args.repeats < 1:
        ap.error("samples, warmup and repeats must be positive")
    if args.out.exists():
        ap.error("output exists; use a fresh path")
    timing = args.mode == "timing"
    provenance = git_provenance()
    if timing:
        validate_timing_state(provenance)
        from _bench import require_quiet_gpu
        require_quiet_gpu()
    # JAX must not reserve the shared GPU's memory; set before its first import.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import torch
    import jax
    if not torch.cuda.is_available() or not jax.devices("gpu"):
        raise RuntimeError("CUDA PyTorch and GPU JAX are required; no skipped paths")
    versions = {p: importlib.metadata.version(p) for p in ("torch", "jax", "grid-rbd", "numpy")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("x") as out:
        for plant in args.plants:
            x, refs, digest = replay_inputs(plant, args.samples + args.warmup)
            for batch, limits, repeat in itertools.product(args.batches, ("off", "admm"), range(args.repeats)):
                if (args.out.parent / "STOP").exists():
                    raise SystemExit("STOP requested; partial collection is not a complete batch")
                before = validate_timing_state(git_provenance()) if timing else None
                if timing and git_provenance()["sha"] != provenance["sha"]:
                    raise RuntimeError("HEAD changed during collection")
                order = KINDS[repeat % 3:] + KINDS[:repeat % 3]
                rows = [run_path(plant, batch, limits, kind, (x, refs), args.warmup, timing=timing)
                        for kind in order]
                check_parity(rows)
                after = validate_timing_state(git_provenance()) if timing else None
                if timing and git_provenance()["sha"] != provenance["sha"]:
                    raise RuntimeError("HEAD changed during collection")
                for row in rows:
                    row.update(mode=args.mode, quotable=timing, provenance=provenance,
                               versions=versions, plant=plant, N=N, dt=DT, batch=batch,
                               limits=limits, repeat=repeat, order=list(order),
                               samples=args.samples, warmup=args.warmup,
                               input_sha256=digest, parity="PASS", box_before=before, box_after=after)
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                    count += 1
                print(f"PASS {plant} B{batch} {limits} repeat {repeat + 1}: all three paths", flush=True)
    expected = len(args.plants) * len(args.batches) * 2 * args.repeats * len(KINDS)
    if count != expected:
        raise RuntimeError(f"incomplete inventory: {count}/{expected}")
    print(f"COMPLETE {count} rows, mode={args.mode}; replay is not closed-loop task evidence", flush=True)


if __name__ == "__main__":
    main()

"""Shared helpers for the GATO benchmark / timing scripts (examples/benchmarks).

One home for what every harness used to re-implement: robot URDF lookup
(the registry is the source of truth), pinocchio model construction, the
quiet-GPU guard for timing runs, git provenance stamps, the percentile
line printer, the kicked-arm MPC probe behind the linsys studies, and the
sibling-repo (MPCGPU) location.

Scripts in this directory import it directly (``from _bench import ...``:
python puts the script's own directory on the import path). Scripts that
live elsewhere (tools/, paper-figures/, contact-task/) load it by file path
with importlib — there is no package here and nothing edits the import path.
"""
import datetime
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG8_PLANTS = ("indy7", "iiwa14")   # plants with a wired fig8 MPC task (fig8_task)


# ---- repo layout -----------------------------------------------------------

def import_sibling(name):
    """Import examples/benchmarks/<name>.py by path (for callers outside this
    directory). Idempotent: returns the already-loaded module if present."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mpcgpu_root():
    """The sibling MPCGPU checkout: $MPCGPU_ROOT, default <repo>/../MPCGPU."""
    return Path(os.environ.get("MPCGPU_ROOT", REPO.parent / "MPCGPU"))


# ---- robots ----------------------------------------------------------------

def urdf_path(plant):
    """Absolute URDF path of a registered plant (python/gato/_registry.json,
    written by tools/regen_grid.py / gato.build; paths are repo-relative)."""
    import gato
    info = gato.robot_info(plant)
    if "urdf" not in info:
        raise KeyError(f"plant {plant!r} is not in the registry (known: "
                       f"{sorted(gato.builder.load_registry())}) — run tools/regen_grid.py")
    return str(REPO / info["urdf"])


def pin_model(plant):
    """pinocchio Model for a registered plant (free-flyer root for floating-base
    plants, matching the solver's stored [p; quat xyzw; joints] layout)."""
    import pinocchio as pin
    import gato
    urdf = urdf_path(plant)
    if gato.robot_info(plant).get("floating_base"):
        return pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    return pin.buildModelFromUrdf(urdf)


# ---- quiet-box guard + provenance -----------------------------------------

def _smi(*query):
    try:
        return subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def gpu_busy():
    """Comma-joined compute pids on the GPU ('' when idle / no nvidia-smi)."""
    return ",".join(_smi("--query-compute-apps=pid").split())


def gpu_util():
    """GPU utilization percent (0 when unknown)."""
    out = _smi("--query-gpu=utilization.gpu").splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else 0


def gpu_info():
    """One-line GPU descriptor (name, SM clock, temperature, driver) for provenance."""
    return _smi("--query-gpu=name,clocks.sm,temperature.gpu,driver_version")


def require_quiet_gpu(allow_busy=False, util_max=5):
    """The timing guard: refuse to run while other compute processes hold the
    GPU (or utilization > util_max %). Returns True when quiet.

    allow_busy=True downgrades the refusal to a printed warning — for plumbing
    smoke tests only; numbers measured under contention are never kept.
    Correctness-class harnesses (fixed pacing, bit-deterministic) call it with
    allow_busy=True just to flag that their recorded wall times are not quotable.
    """
    pids, util = gpu_busy(), gpu_util()
    if not pids and util <= util_max:
        return True
    msg = f"GPU busy (compute pids: {pids or 'none'}, util {util}%)"
    if not allow_busy:
        sys.exit(f"REFUSING to time: {msg}")
    print(f"WARNING: {msg} — timing numbers from this run are NOT quotable",
          file=sys.stderr, flush=True)
    return False


def git_provenance():
    """{'sha', 'short', 'dirty', 'date'} of the GATO checkout (submodule
    untracked content ignored)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "HEAD")
    return {
        "sha": sha,
        "short": sha[:7],
        "dirty": bool(git("status", "--porcelain", "--ignore-submodules=untracked")),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ---- percentiles -----------------------------------------------------------

def pct(samples):
    """{p50, p90, p99, max, mean} of a sample list (floats)."""
    t = np.asarray(samples, dtype=float)
    return {k: float(np.percentile(t, q)) for k, q in
            (("p50", 50), ("p90", 90), ("p99", 99))} | \
           {"max": float(t.max()), "mean": float(t.mean())}


def pct_line(name, samples, extra=""):
    """One aligned stats line (ms): p50 / p90 / p99 / max / mean."""
    s = pct(samples)
    return (f"{name:<14} p50 {s['p50']:6.3f}  p90 {s['p90']:6.3f}  "
            f"p99 {s['p99']:6.3f}  max {s['max']:6.3f}  mean {s['mean']:6.3f} ms{extra}")


# ---- the kicked-arm fig8 MPC probe (linsys studies) ------------------------

def fig8_task(plant, N, dt):
    """(q0, fig8_flat) for plant — iiwa14 uses the fair 3-way harness definition
    (iiwa_fig8_shared), indy7 the package figure8 + 'ready' start."""
    from gato.common import figure8
    if plant == "indy7":
        from gato.config import INDY7_START_CONFIGS, FIG8_DEFAULT_PARAMS
        return INDY7_START_CONFIGS["ready"], figure8(dt, **FIG8_DEFAULT_PARAMS)
    if plant == "iiwa14":
        S = import_sibling("iiwa_fig8_shared")
        n_steps = int(round(S.FIG8_PERIOD * 5 / dt))
        return S.Q0_READYC, S.figure8_goal(n_steps)
    raise ValueError(f"no fig8 task wired for plant {plant!r} (known: {FIG8_PLANTS})")


def run_kicked_arm(mode, tau=0.0, *, plant="indy7", N=64, dt=0.01, sim_time=6.0,
                   kick_every=25, seed=7):
    """One fixed-pacing fig8 MPC run under linsys policy `mode` with a seeded,
    stream-aligned kick schedule (the 08-12 CDF study rig; also the
    tools/autotune_linsys.py probe). TIMING-CLASS: callers own the quiet-box guard.

    Every `kick_every` steps the measured state is kicked, cycling through
    mild (q+=N(0,.03)) / medium (q+=N(0,.10)) / severe (q+=N(0,.20),
    qd+=N(0,.5)). Non-kick steps burn the identical number of rng draws so
    the disturbance stream stays aligned across arms. Returns a trace dict
    (solve_ms / pred_err / pcg_iters per solve + tracking summary).
    """
    from gato.mpc_gato import MPC_GATO
    from gato.controller import MPCController

    urdf = urdf_path(plant)
    mpc = MPC_GATO(pin_model(plant), model_path=urdf, N=N, dt=dt, batch_size=1,
                   plant_type=plant)
    # always pass linsys explicitly: since 08-12 the controller DEFAULT is auto
    # (fixed-base), so an omitted arg would not give the pure-pcg arm
    kw = {"linsys": mode}
    if mode == "auto":
        kw["bdsv_threshold"] = tau
    mpc.controller = MPCController(mpc.solver, hypotheses=mpc.controller.hypotheses,
                                   warm_start="shift", reset_rho_each_step=True, **kw)

    pred_errs, iters, colds = [], [], []
    rng = np.random.default_rng(seed)
    orig_step = mpc.controller.step
    nq, nx = mpc.solver.nq, mpc.solver.nx
    k = [0]

    def step(x, g, **skw):
        k[0] += 1
        if k[0] % kick_every == 0:
            x = x.copy()
            c = (k[0] // kick_every) % 3
            if c == 0:
                x[:nq] += rng.normal(0.0, 0.03, nq)
            elif c == 1:
                x[:nq] += rng.normal(0.0, 0.10, nq)
            else:
                x[:nq] += rng.normal(0.0, 0.20, nq)
                x[nq:nx] += rng.normal(0.0, 0.5, nx - nq)
        else:
            rng.normal(0.0, 1.0, 2 * nx - nq)   # keep the stream aligned across arms
        r = orig_step(x, g, **skw)
        pred_errs.append(r.pred_err)
        iters.append(int(np.asarray(r.solve.stats.pcg_iters).reshape(-1).max()))
        if mode == "auto":
            colds.append(bool(r.pred_err > tau))
        return r

    mpc.controller.step = step
    q0, fig8 = fig8_task(plant, N, dt)
    xs = np.hstack((q0, np.zeros(nx - nq)))
    _, stats = mpc.run_mpc_fig8(xs, fig8, sim_dt=0.001, sim_time=sim_time,
                                pace_by_solve_time=False)
    st = np.asarray(stats["solve_times"], dtype=float)
    gd = np.asarray(stats["goal_distances"], dtype=float)
    n = min(len(st), len(pred_errs))
    return {
        "mode": mode, "tau": tau, "steps": n,
        "solve_ms": st[:n].tolist(),
        "pred_err": pred_errs[:n],
        "pcg_iters": iters[:n],
        "cold": colds[:n] if mode == "auto" else None,
        "track_mean": float(gd.mean()), "track_max": float(gd.max()),
    }

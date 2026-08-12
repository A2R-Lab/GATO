"""Shared linsys probe machinery: run one MPC arm under a pinned (or auto)
linsys policy on the fig8 task, collecting per-solve (solve ms, pred_err,
pcg iters) traces.

Consumers: linsys_auto_cdf.py (the CDF study rig) and tools/autotune_linsys.py
(the per-problem policy autotuner). The kick schedule is seeded and
stream-aligned across arms so every arm sees the identical disturbance
sequence. TIMING-CLASS: callers own the quiet-box guard.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(ROOT / "examples" / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples" / "benchmarks"))

URDFS = {
    "indy7": str(ROOT / "examples" / "indy7_description" / "indy7.urdf"),
    # canonical iiwa14 (fig3-fair harness URDF, md5 eeb7d4ff — NOT GRiD/robot_assets)
    "iiwa14": str(ROOT / "examples" / "iiwa_description" / "iiwa14.urdf"),
}


def start_and_ref(plant, N, dt):
    """(q0, fig8_flat) for plant — iiwa14 uses the shared fig3-fair harness."""
    from gato.common import figure8
    if plant == "indy7":
        from gato.config import INDY7_START_CONFIGS, FIG8_DEFAULT_PARAMS
        return INDY7_START_CONFIGS["ready"], figure8(dt, **FIG8_DEFAULT_PARAMS)
    if plant == "iiwa14":
        import iiwa_fig8_shared as S
        n_steps = int(round(S.FIG8_PERIOD * 5 / dt))
        return S.Q0_READYC, S.figure8_goal(n_steps)
    raise ValueError(f"no fig8 task wired for plant {plant!r} "
                     f"(known: {sorted(URDFS)})")


def run_arm(mode, tau=0.0, *, plant="indy7", N=64, dt=0.01, sim_time=6.0,
            kick_every=25, seed=7):
    """One fixed-pacing fig8 MPC run under linsys policy `mode`.

    Every `kick_every` steps the measured state is kicked, cycling through
    mild (q+=N(0,.03)) / medium (q+=N(0,.10)) / severe (q+=N(0,.20),
    qd+=N(0,.5)). Non-kick steps burn the identical number of rng draws so
    the disturbance stream stays aligned across arms. Returns a trace dict
    (solve_ms / pred_err / pcg_iters per solve + tracking summary).
    """
    import pinocchio as pin
    from gato.mpc_gato import MPC_GATO
    from gato.controller import MPCController

    urdf = URDFS[plant]
    model = pin.buildModelFromUrdf(urdf)
    mpc = MPC_GATO(model, model_path=urdf, N=N, dt=dt, batch_size=1,
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
    q0, fig8 = start_and_ref(plant, N, dt)
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


def gpu_busy():
    """Comma-joined compute pids on the GPU ('' when idle)."""
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                           "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()

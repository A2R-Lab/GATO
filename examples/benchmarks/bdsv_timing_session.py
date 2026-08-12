#!/usr/bin/env python
"""Single-command driver for the hybrid pcg/bdsv TIMING session
(`docs/open-tasks/hybrid_pcg_bdsv_plan_2026-07-07.md` §6.2-6.4).

Run from the repo root with a python that has pinocchio (the GRiD venv):

    python examples/benchmarks/bdsv_timing_session.py            # all phases, in order
    python examples/benchmarks/bdsv_timing_session.py --build    # GATO_BDSV_THREADS variants
    python examples/benchmarks/bdsv_timing_session.py --kernel   # §6.2 per-SQP-iter linsys A/B
    python examples/benchmarks/bdsv_timing_session.py --mpc      # §6.3/6.4 fig8 modes × τ
    python examples/benchmarks/bdsv_timing_session.py --report   # aggregate → markdown
    python examples/benchmarks/bdsv_timing_session.py --restore  # put the pre-session .so back

§6.1 (GLASS-level characterization) is separate — standalone GLASS + the bs14 shapes patch;
measured 2026-07-09 in `docs/open-tasks/glass_bs14_solvers_results_2026-07-09.md`.

Methodology guards baked in: refuses to time unless the GPU is idle; builds and timed legs
never overlap (build phase completes and stashes .so variants first; timing only copies
files); provenance (GPU clocks/temp, git SHA) logged per leg; every timed cell reports its
spread so sub-noise deltas can't be over-read. The per-iteration linsys time comes from
`SolverStats.pcg_times_us` (cudaEvent pair around the solvePCG/solveBDSV launch, collected
only when stats collection is on). The default-flip decision is a HUMAN call — this script
measures and reports, it does not change any default.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "examples" / "benchmarks" / "data" / "bdsv_timing"
PLANT = "indy7"
KNOTS = [32, 64]
BATCHES = [1, 16, 64, 128]
THREADS = [128, 256, 512]          # GATO_BDSV_THREADS candidates (256 = current default)
TAUS = [0.05, 0.10, 0.17, 0.35]    # around the measured anchor 5×median(pred_err) ≈ 0.17
URDF = str(ROOT / "examples" / "indy7_description" / "indy7.urdf")


# ─── shared helpers ──────────────────────────────────────────────────────────

def sh(cmd, **kw):
    print(f"  $ {' '.join(map(str, cmd))}")
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def module_sos(knots=KNOTS):
    pats = [f"bsqpN{n}_{PLANT}.*.so" for n in knots]
    return [p for pat in pats for p in sorted((ROOT / "python" / "gato").glob(pat))]


def gpu_idle_or_die():
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if apps:
        sys.exit(f"REFUSING to time: GPU busy (compute pids: {apps.replace(chr(10), ' ')})")


def provenance(tag):
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,clocks.sm,temperature.gpu,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "--ignore-submodules=untracked"],
                           capture_output=True, text=True).stdout.strip()
    rec = {"tag": tag, "when": time.strftime("%F %T"), "gpu": smi,
           "gato": sha + ("+dirty" if dirty else "")}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "provenance.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  [{tag}] {smi} | gato @{rec['gato']}")


def child(args_list):
    """Run this script as a subprocess child; return its parsed JSON stdout."""
    r = subprocess.run([sys.executable, __file__] + [str(a) for a in args_list],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"child {args_list} FAILED:\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.splitlines()[-1])


# ─── phase: build the GATO_BDSV_THREADS variants ─────────────────────────────

def phase_build():
    orig = OUT / "so_orig"
    if not orig.exists():
        orig.mkdir(parents=True)
        for so in module_sos():
            shutil.copy2(so, orig)
        print(f"  stashed pre-session modules -> {orig}")
    for t in THREADS:
        bdir = ROOT / f"build_bdsv_t{t}"
        import pybind11
        sh(["cmake", "-S", ROOT, "-B", bdir, f"-DKNOTS={';'.join(map(str, KNOTS))}",
            f"-DPLANT={PLANT}", "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DCMAKE_CUDA_ARCHITECTURES=120",
            f"-DCMAKE_CUDA_FLAGS=-DGATO_BDSV_THREADS={t}"],
           stdout=subprocess.DEVNULL)
        sh(["cmake", "--build", bdir, "--parallel", "4"])
        stash = OUT / f"so_t{t}"
        stash.mkdir(parents=True, exist_ok=True)
        for so in module_sos():
            shutil.copy2(so, stash)
        print(f"  T={t}: stashed {len(module_sos())} modules -> {stash}")


def install_variant(name):
    src = OUT / name
    for so in sorted(src.glob("*.so")):
        shutil.copy2(so, ROOT / "python" / "gato")


# ─── phase: kernel-level A/B (child does one (N, B) cell, both modes) ────────

def phase_kernel():
    gpu_idle_or_die()
    provenance("kernel")
    rows = []
    variants = [f"so_t{t}" for t in THREADS if (OUT / f"so_t{t}").exists()] or ["so_orig"]
    for var in variants:
        install_variant(var)
        for n in KNOTS:
            for b in BATCHES:
                row = child(["--child-kernel", n, b])
                row.update(variant=var)
                rows.append(row)
                print(f"  {var} N={n:>3} B={b:>3}: "
                      f"pcg {row['pcg_us_med']:.1f}us (it~{row['pcg_iters_med']:.0f}) "
                      f"vs bdsv {row['bdsv_us_med']:.1f}us  "
                      f"[spreads {row['pcg_us_iqr']:.1f}/{row['bdsv_us_iqr']:.1f}]")
    install_variant("so_t256" if (OUT / "so_t256").exists() else "so_orig")
    (OUT / "kernel_results.json").write_text(json.dumps(rows, indent=1))
    print(f"==> wrote {OUT / 'kernel_results.json'}")


def child_kernel(n, b):
    import numpy as np
    from gato.interface import BSQP
    from gato.config import DEFAULT_SOLVER_PARAMS as SP
    solver = BSQP(model_path=URDF, batch_size=b, N=n, dt=0.01, plant_type=PLANT,
                  **{**SP, "max_sqp_iters": 5, "rho": 1e-3})
    rng = np.random.default_rng(0)
    x = np.zeros((b, solver.nx), dtype=np.float32)
    x[:, :solver.nq] = rng.uniform(-0.4, 0.4, (b, solver.nq)).astype(np.float32)
    g = np.tile(np.concatenate([rng.uniform(0.2, 0.5, 3), np.zeros(3)])
                .astype(np.float32), (b, n))
    out = {"N": n, "B": b}
    for mode in ("pcg", "bdsv"):
        solver.set_linsys(mode)
        times, iters = [], []
        for r in range(13):                      # 3 warmup + 10 measured
            solver.solver.reset_dual(); solver.solver.reset_rho()
            solver.XU_B = np.zeros_like(solver.XU_B)
            res = solver.solve(x.copy(), g.copy())
            if r >= 3:
                times += list(res.stats.pcg_times_us)   # one entry per SQP iter
                iters += list(res.stats.pcg_iters.reshape(-1))
        t = np.asarray(times)
        out[f"{mode}_us_med"] = float(np.median(t))
        out[f"{mode}_us_iqr"] = float(np.percentile(t, 75) - np.percentile(t, 25))
        out[f"{mode}_iters_med"] = float(np.median(iters))
    print(json.dumps(out))


# ─── phase: end-to-end MPC fig8, modes × τ, nominal + perturbed ──────────────

def phase_mpc():
    gpu_idle_or_die()
    provenance("mpc")
    install_variant("so_t256" if (OUT / "so_t256").exists() else "so_orig")
    rows = []
    cells = [("pcg", 0.0), ("bdsv", 0.0), ("bdsv_first", 0.0)] + \
            [("auto", tau) for tau in TAUS]
    for perturb in (0, 25):                      # 0 = nominal; else perturb every K steps
        for mode, tau in cells:
            row = child(["--child-mpc", mode, tau, perturb])
            rows.append(row)
            print(f"  {'nominal' if not perturb else f'perturb/{perturb}'} "
                  f"{mode}{f'(t={tau})' if mode == 'auto' else '':<8}: "
                  f"solve p50 {row['solve_ms_p50']:.3f}ms p95 {row['solve_ms_p95']:.3f}ms | "
                  f"track mean {row['track_mean']:.4f} max {row['track_max']:.4f} | "
                  f"iters p50 {row['iters_p50']:.0f}")
    (OUT / "mpc_results.json").write_text(json.dumps(rows, indent=1))
    print(f"==> wrote {OUT / 'mpc_results.json'}")


def child_mpc(mode, tau, perturb_every):
    import numpy as np
    import pinocchio as pin
    from gato.mpc_gato import MPC_GATO
    from gato.controller import MPCController
    from gato.common import figure8
    from gato.config import INDY7_START_CONFIGS, FIG8_DEFAULT_PARAMS
    N, DT = 64, 0.01
    model = pin.buildModelFromUrdf(URDF)
    mpc = MPC_GATO(model, model_path=URDF, N=N, dt=DT, batch_size=1, plant_type=PLANT)
    # always pass linsys explicitly: since 08-12 the controller DEFAULT is auto
    # (fixed-base), so an omitted arg would not give the pure-pcg arm
    kw = {"linsys": mode}
    if mode == "auto":
        kw["bdsv_threshold"] = tau
    mpc.controller = MPCController(mpc.solver, hypotheses=mpc.controller.hypotheses,
                                   warm_start="shift", reset_rho_each_step=True, **kw)

    iters, pred_errs = [], []
    rng = np.random.default_rng(7)
    orig_step = mpc.controller.step
    nq = mpc.solver.nq
    state = {"k": 0}

    def step(x, g, **skw):
        state["k"] += 1
        if perturb_every and state["k"] % perturb_every == 0:
            x = x.copy()
            x[:nq] += rng.normal(0.0, 0.05, nq)      # seeded joint-position kick
        r = orig_step(x, g, **skw)
        iters.append(np.asarray(r.solve.stats.pcg_iters).reshape(-1))
        pred_errs.append(r.pred_err)
        return r

    mpc.controller.step = step
    xs = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(mpc.solver.nx - 6)))
    fig8 = figure8(DT, **FIG8_DEFAULT_PARAMS)
    _, stats = mpc.run_mpc_fig8(xs, fig8, sim_dt=0.001, sim_time=3.0,
                                pace_by_solve_time=False)   # fixed pacing: deterministic
    st = np.asarray(stats["solve_times"], dtype=float)
    gd = np.asarray(stats["goal_distances"], dtype=float)
    it = np.concatenate(iters)
    print(json.dumps({
        "mode": mode, "tau": tau, "perturb_every": perturb_every,
        "steps": int(len(st)),
        "solve_ms_p50": float(np.percentile(st, 50)),
        "solve_ms_p95": float(np.percentile(st, 95)),
        "track_mean": float(gd.mean()), "track_max": float(gd.max()),
        "iters_p50": float(np.percentile(it, 50)),
        "iters_p95": float(np.percentile(it, 95)),
        "iters_hist": {str(k): int((it == k).sum()) for k in np.unique(it)[:12]},
        "pred_err_med": float(np.median(pred_errs)),
    }))


# ─── phase: report ────────────────────────────────────────────────────────────

def phase_report():
    lines = ["# bdsv timing session — measured results", "",
             f"_Generated {time.strftime('%F %T')} by bdsv_timing_session.py; "
             f"provenance in `data/bdsv_timing/provenance.jsonl`. Default-flip is a "
             f"human decision — see plan §6.5._", ""]
    kj = OUT / "kernel_results.json"
    if kj.exists():
        rows = json.loads(kj.read_text())
        lines += ["## §6.2 kernel-level: per-SQP-iteration linsys time (µs, median of "
                  "50 iters; IQR in parens)", "",
                  "| variant | N | B | pcg µs | pcg iters | bdsv µs | pcg/bdsv |",
                  "|---------|---|---|--------|-----------|---------|----------|"]
        for r in rows:
            lines.append(
                f"| {r['variant']} | {r['N']} | {r['B']} "
                f"| {r['pcg_us_med']:.1f} ({r['pcg_us_iqr']:.1f}) "
                f"| {r['pcg_iters_med']:.0f} "
                f"| {r['bdsv_us_med']:.1f} ({r['bdsv_us_iqr']:.1f}) "
                f"| {r['pcg_us_med'] / r['bdsv_us_med']:.2f} |")
        lines.append("")
    mj = OUT / "mpc_results.json"
    if mj.exists():
        rows = json.loads(mj.read_text())
        lines += ["## §6.3/6.4 end-to-end fig8 (indy7 N=64 M=1, fixed pacing, 3s)", "",
                  "| run | mode | τ | solve p50 ms | p95 ms | track mean | track max "
                  "| iters p50/p95 |",
                  "|-----|------|---|--------------|--------|------------|-----------"
                  "|---------------|"]
        for r in rows:
            run = "nominal" if not r["perturb_every"] else f"perturb/{r['perturb_every']}"
            lines.append(
                f"| {run} | {r['mode']} | {r['tau'] or ''} "
                f"| {r['solve_ms_p50']:.3f} | {r['solve_ms_p95']:.3f} "
                f"| {r['track_mean']:.4f} | {r['track_max']:.4f} "
                f"| {r['iters_p50']:.0f}/{r['iters_p95']:.0f} |")
        lines.append("")
    md = OUT / "BDSV_TIMING_RESULTS.md"
    md.write_text("\n".join(lines))
    print(f"==> wrote {md}")


def phase_restore():
    if (OUT / "so_orig").exists():
        install_variant("so_orig")
        print("restored pre-session modules")


# ─── entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("build", "kernel", "mpc", "report", "restore"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--child-kernel", nargs=2, type=int)
    ap.add_argument("--child-mpc", nargs=3)
    a = ap.parse_args()
    if a.child_kernel:
        child_kernel(*a.child_kernel)
    elif a.child_mpc:
        child_mpc(a.child_mpc[0], float(a.child_mpc[1]), int(a.child_mpc[2]))
    elif not any((a.build, a.kernel, a.mpc, a.report, a.restore)):
        phase_build(); phase_kernel(); phase_mpc(); phase_report()
    else:
        if a.build:
            phase_build()
        if a.kernel:
            phase_kernel()
        if a.mpc:
            phase_mpc()
        if a.report:
            phase_report()
        if a.restore:
            phase_restore()

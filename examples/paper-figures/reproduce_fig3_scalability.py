"""Regenerate Fig-3 (left): Indy7 figure-8 batch-size scalability.

Paper IV-B: at each control step solve a batch of M trajectory-optimization
problems (N=64, h=0.01, warm-started from the previous step) and compare GATO's
per-step solve time against the OSQP CPU baseline and the MPCGPU GPU baseline as
M grows over [1, 2, 4, ..., 128].

This is a thin orchestrator over the canonical generators:
  - GATO:   examples/benchmark_fig8.py        -> benchmark_fig8_64N.pkl
  - OSQP:   baselines/run_osqp_fig8.py         -> baselines/osqp_fig8_results.pkl
  - MPCGPU: baselines/mpcgpu_indy7_fig8_N64.csv (built separately; OPTIONAL — the
            plot degrades gracefully if absent. Its tracking fix + PR are deferred
            until Fig-3 is otherwise complete; see docs/baselines.md.)

DEFAULT regenerates GATO + OSQP on the GPU/CPU then assembles the plot. ``--replot``
just assembles from existing pkls. NOTE: the OSQP baseline is the single-solve
Python `Thneed` (Python-interpreter-bound, conservative); the paper's CPU bar is a
multi-threaded C++ `BatchThneed` (backlog — needs osqp/OsqpEigen/pinocchio-C++).

Examples::
    python examples/paper-figures/reproduce_fig3_scalability.py            # regen + plot
    python examples/paper-figures/reproduce_fig3_scalability.py --quick    # fast smoke
    python examples/paper-figures/reproduce_fig3_scalability.py --replot   # assemble only
"""
import os
import argparse
import pickle
import subprocess

import numpy as np

import _common as C

N = 64
GATO_PKL = os.path.join(C.REPO, f"benchmark_fig8_{N}N.pkl")
OSQP_PKL = os.path.join(C.REPO, "baselines", "osqp_fig8_results.pkl")
MPCGPU_CSV = os.path.join(C.REPO, "baselines", "mpcgpu_indy7_fig8_N64.csv")
VENV = os.path.join(os.path.dirname(C.REPO), "GRiD", ".venv", "bin", "python")
PY = VENV if os.path.exists(VENV) else "python"


def _run(cmd):
    print(f"[paper-figures] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=C.REPO, check=True)


def regen_gato(batch_sizes, sim_time):
    _run([PY, "examples/benchmark_fig8.py", "--plant", "indy7", "--N", str(N),
          "--batch-sizes", batch_sizes, "--sim-time", str(sim_time)])


def regen_osqp(sim_time):
    _run([PY, "baselines/run_osqp_fig8.py", "--N", str(N), "--sim-time", str(sim_time)])


def load_gato():
    with open(GATO_PKL, "rb") as f:
        rows = pickle.load(f)
    return {int(r["batch_size"]): float(r["avg_gpu_time_ms"])
            for r in rows if r.get("success") and r.get("avg_gpu_time_ms") is not None}


def load_osqp():
    if not os.path.exists(OSQP_PKL):
        return None
    with open(OSQP_PKL, "rb") as f:
        rows = pickle.load(f)
    for r in rows:
        if int(r["N"]) == N:
            return float(r["avg_cpu_time_ms"])
    return None


def load_mpcgpu():
    if not os.path.exists(MPCGPU_CSV):
        return None
    with open(MPCGPU_CSV) as f:
        lines = [l.strip() for l in f if l.strip()]
    return float(lines[-1].split(",")[0]) / 1000.0  # avg us -> ms


def report(gato, osqp_ms, mpcgpu_ms):
    lines = [f"=== Fig-3 (left): Indy7 fig8, N={N}, solve time vs batch ===",
             f"{'M':>5} {'GATO ms':>9} {'vs OSQP(CPU)':>13} {'vs MPCGPU(GPU)':>15}"]
    for m in sorted(gato):
        g = gato[m]
        cpu = f"{osqp_ms / g:6.1f}x" if osqp_ms else "n/a"
        gpu = f"{mpcgpu_ms / g:6.1f}x" if mpcgpu_ms else "n/a"
        lines.append(f"{m:>5} {g:>9.3f} {cpu:>13} {gpu:>15}")
    lines.append(f"OSQP-CPU (single, batch=1): {osqp_ms:.3f} ms" if osqp_ms else "OSQP: missing")
    lines.append(f"MPCGPU (single, batch=1):   {mpcgpu_ms:.3f} ms" if mpcgpu_ms else "MPCGPU: missing")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(C.FIG_DIR, "fig3_scalability.txt"), "w") as f:
        f.write(txt + "\n")


def plot(gato, osqp_ms, mpcgpu_ms):
    plt = C.set_paper_rcParams()
    M = sorted(gato)
    ms = [gato[m] for m in M]
    fig = plt.figure(figsize=(7, 5))
    plt.plot(M, ms, "o-", color="#00693E", label="GATO (GPU, batched)")
    if osqp_ms:
        plt.axhline(osqp_ms, ls="--", color="#C90016", label=f"OSQP CPU: {osqp_ms:.1f} ms")
    if mpcgpu_ms:
        plt.axhline(mpcgpu_ms, ls="--", color="#003192", label=f"MPCGPU GPU: {mpcgpu_ms:.2f} ms")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Batch Size")
    plt.ylabel("Solve Time (ms)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    C.savefig(fig, "fig3_scalability")


def main():
    p = argparse.ArgumentParser(description="Regenerate Fig-3 left (scalability).")
    C.add_repro_args(p)
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    p.add_argument("--sim-time", type=float, default=5.0)
    p.add_argument("--skip-osqp", action="store_true")
    args = p.parse_args()

    if not args.replot:
        C.require_module("indy7", N)
        batch_sizes, sim_time = args.batch_sizes, args.sim_time
        if args.quick:
            batch_sizes, sim_time = "1,4,32,128", 2.0
            print("[quick] tiny subset — NOT paper numbers")
        if args.regen or not os.path.exists(GATO_PKL):
            regen_gato(batch_sizes, sim_time)
        if not args.skip_osqp and (args.regen or not os.path.exists(OSQP_PKL)):
            regen_osqp(sim_time)

    gato = load_gato()
    osqp_ms = load_osqp()
    mpcgpu_ms = load_mpcgpu()
    report(gato, osqp_ms, mpcgpu_ms)
    plot(gato, osqp_ms, mpcgpu_ms)


if __name__ == "__main__":
    main()

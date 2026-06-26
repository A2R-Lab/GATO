"""Regenerate Fig-3 (left): Indy7 figure-8 batch-size scalability.

Paper IV-B: at each control step solve a batch of M trajectory-optimization
problems (N=64, h=0.01, warm-started from the previous step) and compare GATO's
per-step solve time against the BatchThneed CPU baseline and the MPCGPU GPU
baseline as M grows over [1, 2, 4, ..., 128].

This is a thin orchestrator over the canonical generators:
  - GATO:        examples/benchmarks/benchmark_fig8.py -> data/benchmark_fig8_64N.pkl (batched GPU)
  - BatchThneed: benchmarks/baselines/run_batchthneed_fig8.py -> baselines/batchthneed_fig8_results.pkl
                 The paper's CPU competitor: threaded C++ BatchThneed solves M problems across
                 cores -> SUB-LINEAR (flat to core count, then linear), the fair CPU line. Build
                 via benchmarks/baselines/build_cpu_baseline.sh (no ROS; reuses cmeel pinocchio).
                 (Single-solve OSQP was dropped: BatchThneed is faster even at M=1, so OSQP added
                 nothing — see docs/baselines.md.)
  - MPCGPU:      benchmarks/baselines/mpcgpu_indy7_fig8_N64.csv (built separately; OPTIONAL). Single-solve
                 (whole-GPU cooperative), so a batch of M costs M x per-solve -> drawn as a LINEAR
                 line (~0.2 -> ~28 ms over M=1..128). Per-solve = MEDIAN of the CSV.
All baselines degrade gracefully if absent; GATO's sub-linear curve under both batched baselines
is the figure's point. See docs/baselines.md.

DEFAULT regenerates GATO then assembles the plot (BatchThneed/MPCGPU read if present).
``--replot`` just assembles from existing pkls.

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
GATO_PKL = os.path.join(C.BENCH_DATA, f"benchmark_fig8_{N}N.pkl")
BATCHTHNEED_PKL = os.path.join(C.BENCH_DIR, "baselines", "batchthneed_fig8_results.pkl")
MPCGPU_CSV = os.path.join(C.BENCH_DIR, "baselines", "mpcgpu_indy7_fig8_N64.csv")
VENV = os.path.join(os.path.dirname(C.REPO), "GRiD", ".venv", "bin", "python")
PY = VENV if os.path.exists(VENV) else "python"


def _run(cmd):
    print(f"[paper-figures] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=C.REPO, check=True)


def regen_gato(batch_sizes, sim_time):
    _run([PY, "examples/benchmarks/benchmark_fig8.py", "--plant", "indy7", "--N", str(N),
          "--batch-sizes", batch_sizes, "--sim-time", str(sim_time)])


def load_gato():
    with open(GATO_PKL, "rb") as f:
        rows = pickle.load(f)
    return {int(r["batch_size"]): float(r["avg_gpu_time_ms"])
            for r in rows if r.get("success") and r.get("avg_gpu_time_ms") is not None}


def load_batchthneed():
    """Batched-CPU per-step time vs M (the paper's threaded CPU competitor, pysqpcpu.BatchThneed).
    {batch_size: batched_cpu_ms}. Built/run via baselines/build_cpu_baseline.sh +
    run_batchthneed_fig8.py; absent until then (plot degrades gracefully)."""
    if not os.path.exists(BATCHTHNEED_PKL):
        return None
    with open(BATCHTHNEED_PKL, "rb") as f:
        rows = pickle.load(f)
    return {int(r["batch_size"]): float(r["batched_cpu_ms"])
            for r in rows if r.get("success") and r.get("batched_cpu_ms") is not None}


def load_mpcgpu():
    """Per-solve MPCGPU time in ms (MEDIAN, col 4) — the representative stat; the average
    (col 0) is inflated by first-step JIT/warmup outliers. Last CSV line = solve-time stats
    row 'Average,Std,Min,Max,Median,Q1,Q3' in microseconds."""
    if not os.path.exists(MPCGPU_CSV):
        return None
    with open(MPCGPU_CSV) as f:
        lines = [l.strip() for l in f if l.strip()]
    cols = lines[-1].split(",")
    median_us = float(cols[4]) if len(cols) > 4 else float(cols[0])
    return median_us / 1000.0  # us -> ms (per single solve)


def report(gato, mpcgpu_ms, batchcpu=None):
    lines = [f"=== Fig-3 (left): Indy7 fig8, N={N}, solve time vs batch ===",
             f"{'M':>5} {'GATO ms':>9} {'vs BatchCPU':>12} {'vs MPCGPU(GPU)':>15}"]
    for m in sorted(gato):
        g = gato[m]
        # BatchThneed at batch m (threaded) vs GATO
        cpu = f"{batchcpu[m] / g:6.1f}x" if (batchcpu and m in batchcpu) else "n/a"
        # MPCGPU at batch m = m sequential single-solves -> mpcgpu_ms * m
        gpu = f"{(mpcgpu_ms * m) / g:6.1f}x" if mpcgpu_ms else "n/a"
        lines.append(f"{m:>5} {g:>9.3f} {cpu:>12} {gpu:>15}")
    if batchcpu:
        bm = sorted(batchcpu)
        lines.append(f"BatchThneed CPU (threaded): {batchcpu[bm[0]]:.2f}..{batchcpu[bm[-1]]:.1f} ms "
                     f"over M={bm[0]}..{bm[-1]}")
    if mpcgpu_ms:
        lines.append(f"MPCGPU (single solve): {mpcgpu_ms:.3f} ms/solve -> xM sequential "
                     f"({mpcgpu_ms:.2f}..{mpcgpu_ms*max(gato):.1f} ms over M=1..{max(gato)})")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(C.FIG_DIR, "fig3_scalability.txt"), "w") as f:
        f.write(txt + "\n")


def plot(gato, mpcgpu_ms, batchcpu=None):
    plt = C.set_paper_rcParams()
    M = sorted(gato)
    ms = [gato[m] for m in M]
    fig = plt.figure(figsize=(7, 5))
    plt.plot(M, ms, "o-", color="#00693E", label="GATO (GPU, batched)")
    if batchcpu:
        # The paper's CPU competitor: threaded BatchThneed solves M problems across cores ->
        # sub-linear (flat to core count, then linear). The fair CPU line.
        bm = sorted(batchcpu)
        plt.plot(bm, [batchcpu[m] for m in bm], "^-", color="#C90016",
                 label="BatchThneed (CPU, batched/threaded)")
    if mpcgpu_ms:
        # MPCGPU is single-solve (whole-GPU cooperative), so a batch of M problems costs
        # M x per-solve (sequential) -> a LINEAR line, not flat. This is exactly GATO's
        # batching advantage: GATO stays sub-linear while MPCGPU scales 1:1 with M.
        plt.plot(M, [mpcgpu_ms * m for m in M], "s--", color="#003192",
                 label=f"MPCGPU GPU (xM seq): {mpcgpu_ms:.2f} ms/solve")
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
    args = p.parse_args()

    if not args.replot:
        C.require_module("indy7", N)
        batch_sizes, sim_time = args.batch_sizes, args.sim_time
        if args.quick:
            batch_sizes, sim_time = "1,4,32,128", 2.0
            print("[quick] tiny subset — NOT paper numbers")
        if args.regen or not os.path.exists(GATO_PKL):
            regen_gato(batch_sizes, sim_time)

    gato = load_gato()
    mpcgpu_ms = load_mpcgpu()
    batchcpu = load_batchthneed()
    report(gato, mpcgpu_ms, batchcpu)
    plot(gato, mpcgpu_ms, batchcpu)


if __name__ == "__main__":
    main()

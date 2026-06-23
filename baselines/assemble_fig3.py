"""Assemble the paper's Fig-3 comparison (IV-B scalability) from the three solvers'
results, all on the SAME problem: Indy7 figure-8 tracking, N=64, h=0.01, 1 SQP iter.

  - GATO (GPU, batched): solve time vs batch M in [1,2,4,...,128]   -> benchmark_fig8_64N.pkl
  - OSQP (CPU, single):  solve time, batch=1 (flat baseline)         -> osqp_fig8_results.pkl (N=64 entry)
  - MPCGPU (GPU, single):full per-step SQP solve time, batch=1 (flat)-> baselines/mpcgpu_indy7_fig8_N64.csv

Produces baselines/fig3_comparison.{png,txt}: solve time vs batch (log-y) with the two
single-solve baselines as horizontal lines, plus the speedup of GATO(M) over each baseline.
This is the apples-to-apples version of the paper's 18-21x CPU / 1.4-16x GPU claim.

Run AFTER collecting all three (on a quiet GPU):
    python baselines/assemble_fig3.py
"""
import os, pickle, csv
import numpy as np

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 64

# --- GATO: list of per-batch dicts ---
def load_gato():
    p = os.path.join(G, f"benchmark_fig8_{N}N.pkl")
    with open(p, "rb") as f:
        rows = pickle.load(f)
    out = {}
    for r in rows:
        if r.get("success") and r.get("avg_gpu_time_ms") is not None:
            out[int(r["batch_size"])] = float(r["avg_gpu_time_ms"])
    return out  # {batch: ms}

# --- OSQP: list of per-N dicts (batch=1) ---
def load_osqp():
    p = os.path.join(G, "baselines", "osqp_fig8_results.pkl")
    with open(p, "rb") as f:
        rows = pickle.load(f)
    for r in rows:
        if int(r["N"]) == N:
            return float(r["avg_cpu_time_ms"])
    return None

# --- MPCGPU: overall_stats.csv, row 2 = solve-time stats (Average first col) ---
def load_mpcgpu():
    p = os.path.join(G, "baselines", "mpcgpu_indy7_fig8_N64.csv")
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        lines = [l.strip() for l in f if l.strip()]
    # rows: header, tracking-stats, solve-time-stats  (Average,Std,Min,Max,Median,Q1,Q3) in us
    vals = [float(x) for x in lines[-1].split(",")]
    avg_us, median_us = vals[0], vals[4]
    return avg_us / 1000.0, median_us / 1000.0  # -> ms

def main():
    gato = load_gato()
    osqp_ms = load_osqp()
    mpcgpu_avg_ms, mpcgpu_med_ms = load_mpcgpu()

    batches = sorted(gato)
    lines = []
    lines.append(f"=== Fig-3 comparison: Indy7 fig8, N={N}, 1 SQP iter, h=0.01 ===")
    lines.append(f"{'batch M':>8} {'GATO ms':>10} {'vs OSQP(CPU)':>14} {'vs MPCGPU(GPU)':>16}")
    for m in batches:
        g = gato[m]
        cpu_x = f"{osqp_ms/g:6.1f}x" if osqp_ms else "  n/a"
        gpu_x = f"{mpcgpu_avg_ms/g:6.1f}x" if mpcgpu_avg_ms else "  n/a"
        lines.append(f"{m:>8} {g:>10.4f} {cpu_x:>14} {gpu_x:>16}")
    lines.append("")
    lines.append(f"OSQP   (CPU, batch=1): {osqp_ms:.4f} ms" if osqp_ms else "OSQP: missing")
    if mpcgpu_avg_ms:
        lines.append(f"MPCGPU (GPU, batch=1): {mpcgpu_avg_ms:.4f} ms avg / {mpcgpu_med_ms:.4f} ms median")
    else:
        lines.append("MPCGPU: missing")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(G, "baselines", "fig3_comparison.txt"), "w") as f:
        f.write(report + "\n")

    # plot (optional; skips cleanly if matplotlib absent)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ms = [gato[m] for m in batches]
        plt.figure(figsize=(7, 5))
        plt.plot(batches, ms, "o-", label="GATO (GPU, batched)", color="#003192")
        if osqp_ms:
            plt.axhline(osqp_ms, ls="--", color="#C90016", label=f"OSQP (CPU, 1): {osqp_ms:.1f} ms")
        if mpcgpu_avg_ms:
            plt.axhline(mpcgpu_avg_ms, ls="--", color="#00693E",
                        label=f"MPCGPU (GPU, 1): {mpcgpu_avg_ms:.2f} ms")
        plt.xscale("log", base=2); plt.yscale("log")
        plt.xlabel("batch size M"); plt.ylabel("solve time per MPC step (ms)")
        plt.title(f"Indy7 fig-8, N={N}, 1 SQP iter — GATO vs baselines")
        plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
        out = os.path.join(G, "baselines", "fig3_comparison.png")
        plt.savefig(out, dpi=130)
        print(f"\nsaved {out}")
    except ImportError:
        print("\n(matplotlib absent — text report only)")

if __name__ == "__main__":
    main()

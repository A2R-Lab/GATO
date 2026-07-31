"""Regenerate Fig-3 DATA from the FAIR iiwa14 fig8 parity harness (2026-07 config).

Supersedes the June indy7 DATA path of reproduce_fig3_{scalability,heatmap}.py: all three
solvers solve the IDENTICAL problem (examples/benchmarks/iiwa_fig8_shared.py — same fig8,
same EE frame, same costs, same zero-control warm start) under the 2026-07-07 benchmark
config: SQP=1 (RTI), PCG cap 200 / rel 1e-4, rho 0.01, and MPCGPU running GATO_REG_PATTERN
with its native eta-exit. Full provenance: MPCGPU docs/benchmark_3way_2026-07-06.md.

Fig-3 left = the N=64 row: batched total solve time at B in [1..128] for GATO (batched GPU),
BatchThneed (threaded CPU), MPCGPU (single-solve GPU -> B x per-solve), plus GATO's speedup
over each baseline at every B. Fig-3 right = the GATO N x B heat map (N in {8..128}, B up
to 512 — B>128 is GATO-only: MPCGPU cannot batch and BT is past core saturation).

Data stages (each appends CSVs under examples/benchmarks/data/; TIMING — quiet box only):
  --run-gato     sweep_batch_iiwa_fig8.py per N        -> sweep_fig8_gato.csv
  --run-bt       track_iiwa_fig8_bt.py per (N, B)      -> sweep_fig8_bt.csv
  --run-mpcgpu   MPCGPU tools/time_persolve.sh per N   -> sweep_fig8_mpcgpu.csv
Default (no --run-*) assembles the table + figures from existing CSVs. Stages run
sequentially (never overlap timing). Re-runs append; assembly takes the LAST row per (N,B).

Examples::
    # full regeneration on a quiet box (GATO grid + BT B-sweep at N=64 + MPCGPU N=64)
    python examples/paper-figures/reproduce_fig3_fair.py --run-gato --run-bt --run-mpcgpu
    # assemble only
    python examples/paper-figures/reproduce_fig3_fair.py
"""
import os
import csv
import argparse
import subprocess

import numpy as np

import _common as C

MPCGPU_REPO = os.path.join(os.path.dirname(C.REPO), "MPCGPU")
GRIDVENV = os.path.join(os.path.dirname(C.REPO), "GRiD", ".venv")
PY = os.path.join(GRIDVENV, "bin", "python")
if not os.path.exists(PY):
    PY = "python"

GATO_CSV = os.path.join(C.BENCH_DATA, "sweep_fig8_gato.csv")
BT_CSV = os.path.join(C.BENCH_DATA, "sweep_fig8_bt.csv")
MPCGPU_CSV = os.path.join(C.BENCH_DATA, "sweep_fig8_mpcgpu.csv")


def _run(cmd, cwd=C.REPO, env=None):
    print(f"[fig3-fair] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, env=env)


def bt_env():
    """LD_LIBRARY_PATH/PYTHONPATH for pysqpcpu (mirrors MPCGPU tools/run_3way_iiwa.sh)."""
    sqpcpu = os.path.join(C.BENCH_DIR, "baselines", "sqpcpu")
    prefix = os.path.join(sqpcpu, "deps", "install")
    cmeel = os.path.join(GRIDVENV, "lib", "python3.12", "site-packages", "cmeel.prefix", "lib")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        [os.path.join(sqpcpu, "build"), os.path.join(prefix, "lib"), cmeel,
         env.get("LD_LIBRARY_PATH", "")])
    env["PYTHONPATH"] = ":".join(
        [os.path.join(sqpcpu, "build"), os.path.join(C.REPO, "python"),
         env.get("PYTHONPATH", "")])
    return env


def run_gato(N_list, batches, extra, solves):
    for N in N_list:
        C.require_module("iiwa14", N)
        blist = batches + [b for b in extra if b not in batches]
        _run([PY, "examples/benchmarks/sweep_batch_iiwa_fig8.py", "--N", N,
              "--batches", ",".join(map(str, blist)), "--solves", solves, "--out", GATO_CSV])


def run_bt(N_list, batches, sim_time):
    env = bt_env()
    script = os.path.join(C.BENCH_DIR, "baselines", "track_iiwa_fig8_bt.py")
    for N in N_list:
        for B in batches:
            _run([PY, script, sim_time, B, N, BT_CSV], env=env)


def run_mpcgpu(N_list, cycles=3):
    for N in N_list:
        _run(["bash", "tools/time_persolve.sh", N, "pcg", cycles, MPCGPU_CSV], cwd=MPCGPU_REPO)


def read_cells(path):
    """{(N, B): median_ms} from a sweep CSV; last row wins so re-runs supersede."""
    if not os.path.exists(path):
        return {}
    cells = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cells[(int(row["N"]), int(row["B"]))] = float(row["median_ms"])
    return cells


def report_fig3_left(N, batches, gato, bt, mpc):
    mpc1 = mpc.get((N, 1))
    lines = [f"=== Fig-3 (left, FAIR): iiwa14 fig8, N={N}, batched total solve time vs B ===",
             "config: SQP=1, PCG<=200 rel 1e-4, rho 0.01, shared fig8/EE-frame/costs; "
             "MPCGPU = GATO_REG_PATTERN + native exit (docs/benchmark_3way_2026-07-06.md)",
             f"{'B':>4} {'GATO_ms':>9} {'BT_ms':>9} {'MPCGPUxB_ms':>12} {'GATOvsBT':>9} {'GATOvsMPCGPU':>13}"]
    for B in batches:
        g = gato.get((N, B))
        b = bt.get((N, B))
        m = mpc1 * B if mpc1 else None
        if g is None:
            continue
        lines.append(f"{B:>4} {g:>9.3f} "
                     f"{(f'{b:9.3f}' if b else '      n/a')} "
                     f"{(f'{m:12.3f}' if m else '         n/a')} "
                     f"{(f'{b/g:8.1f}x' if b else '      n/a')} "
                     f"{(f'{m/g:12.1f}x' if m else '          n/a')}")
    if mpc1:
        lines.append(f"MPCGPU (GBD-PCG, GATO_REG_PATTERN) per-solve median = {mpc1:.3f} ms; "
                     "no batch axis -> B x per-solve (sequential).")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(C.FIG_DIR, "fig3_fair_scalability.txt"), "w") as f:
        f.write(txt + "\n")


def plot_fig3_left(N, batches, gato, bt, mpc):
    plt = C.set_paper_rcParams()
    fig = plt.figure(figsize=(7, 5))
    Bs = [B for B in batches if (N, B) in gato]
    plt.plot(Bs, [gato[(N, B)] for B in Bs], "o-", color="#00693E", label="GATO (GPU, batched)")
    bBs = [B for B in batches if (N, B) in bt]
    if bBs:
        plt.plot(bBs, [bt[(N, B)] for B in bBs], "^-", color="#C90016",
                 label="BatchThneed (CPU, threaded)")
    mpc1 = mpc.get((N, 1))
    if mpc1:
        plt.plot(Bs, [mpc1 * B for B in Bs], "s--", color="#003192",
                 label=f"MPCGPU GPU (xB seq): {mpc1:.3f} ms/solve")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Batch Size")
    plt.ylabel("Total Solve Time (ms)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    C.savefig(fig, "fig3_fair_scalability")


def report_heatmap(gato, N_list, all_batches):
    Ns = [n for n in N_list if any((n, b) in gato for b in all_batches)]
    Bs = [b for b in all_batches if any((n, b) in gato for n in Ns)]
    if not Ns or not Bs:
        print("[fig3-fair] no GATO heatmap cells yet — run --run-gato first")
        return None, None, None
    lines = ["=== Fig-3 (right, FAIR): GATO iiwa14 fig8 total batched solve time (ms) ===",
             "N\\B " + " ".join(f"{b:>8}" for b in Bs)]
    Z = np.full((len(Ns), len(Bs)), np.nan)
    for i, n in enumerate(Ns):
        cells = []
        for j, b in enumerate(Bs):
            v = gato.get((n, b))
            if v is not None:
                Z[i, j] = v
            cells.append(f"{v:8.3f}" if v is not None else "     n/a")
        lines.append(f"{n:>4} " + " ".join(cells))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(C.FIG_DIR, "fig3_fair_heatmap.txt"), "w") as f:
        f.write(txt + "\n")
    return Ns, Bs, Z


def plot_heatmap(Ns, Bs, Z):
    plt = C.set_paper_rcParams()
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdYlGn_r",
                   norm=LogNorm(vmin=max(np.nanmin(Z), 0.05), vmax=np.nanmax(Z)),
                   extent=[-0.5, len(Bs) - 0.5, -0.5, len(Ns) - 0.5],
                   interpolation="nearest")
    for i in range(len(Ns)):
        for j in range(len(Bs)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white" if Z[i, j] > 2.0 else "black")
    levels = [lvl for lvl in (0.105, 0.2, 1, 4) if np.nanmin(Z) <= lvl <= np.nanmax(Z)]
    if levels and min(Z.shape) >= 2:  # contour needs a real 2-D grid
        CS = ax.contour(np.round(Z, 2), levels=levels, colors="blue", linewidths=1.5)
        ax.clabel(CS, inline=True, fontsize=14,
                  fmt=lambda t: f"{1.0/t:.0f}kHz" if 1.0 / t >= 1 else f"{1000.0/t:.0f}Hz")
    ax.set_xticks(range(len(Bs)))
    ax.set_xticklabels([str(b) for b in Bs])
    ax.set_yticks(range(len(Ns)))
    ax.set_yticklabels([str(n) for n in Ns])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Trajectory Length (N)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("GPU Solve Time (ms)")
    plt.tight_layout()
    C.savefig(fig, "fig3_fair_heatmap")


def main():
    p = argparse.ArgumentParser(description="Fig-3 data from the FAIR iiwa14 parity harness.")
    p.add_argument("--run-gato", action="store_true", help="TIMING: GATO N x B sweep (quiet box)")
    p.add_argument("--run-bt", action="store_true", help="TIMING: BatchThneed B sweep (quiet box)")
    p.add_argument("--run-mpcgpu", action="store_true", help="TIMING: MPCGPU per-solve (quiet box)")
    p.add_argument("--fig3-N", type=int, default=64, help="the fig3-left horizon (paper: 64)")
    p.add_argument("--N-list", default="8,16,32,64,128", help="heatmap horizons (GATO)")
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128", help="shared batch sizes")
    p.add_argument("--gato-extra-batches", default="256,512", help="GATO-only extra batch sizes")
    p.add_argument("--bt-N-list", default="64", help="BT horizons (BT N is a runtime arg)")
    p.add_argument("--mpcgpu-N-list", default="64", help="MPCGPU horizons (rebuild per N)")
    p.add_argument("--solves", type=int, default=400, help="GATO solves per config")
    p.add_argument("--sim-time", type=float, default=6.0, help="BT closed-loop sim seconds")
    p.add_argument("--quick", action="store_true", help="tiny wiring smoke (NOT paper numbers)")
    args = p.parse_args()

    N_list = C.parse_int_list(args.N_list)
    batches = C.parse_int_list(args.batches)
    extra = C.parse_int_list(args.gato_extra_batches)
    solves, sim_time = args.solves, args.sim_time
    if args.quick:
        N_list, batches, extra, solves, sim_time = [16], [1, 8], [], 30, 1.0
        print("[quick] tiny subset — NOT paper numbers")

    if args.run_gato:
        run_gato(N_list, batches, extra, solves)
    if args.run_bt:
        run_bt(C.parse_int_list(args.bt_N_list) if not args.quick else [16], batches, sim_time)
    if args.run_mpcgpu:
        run_mpcgpu(C.parse_int_list(args.mpcgpu_N_list) if not args.quick else [16])

    gato, bt, mpc = read_cells(GATO_CSV), read_cells(BT_CSV), read_cells(MPCGPU_CSV)
    if not gato:
        raise SystemExit("[fig3-fair] no GATO data — run with --run-gato on a quiet box first.")
    report_fig3_left(args.fig3_N, batches, gato, bt, mpc)
    plot_fig3_left(args.fig3_N, batches, gato, bt, mpc)
    Ns, Bs, Z = report_heatmap(gato, N_list, batches + extra)
    if Ns:
        plot_heatmap(Ns, Bs, Z)


if __name__ == "__main__":
    main()

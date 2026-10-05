"""Regenerate Fig-3 DATA from the FAIR fig8 parity harness (2026-07 config; iiwa14 default, --robot indy7).

Supersedes the June indy7 DATA path of reproduce_fig3_{scalability,heatmap}.py: all three
solvers solve the IDENTICAL problem (examples/benchmarks/iiwa_fig8_shared.py — same fig8,
same EE frame, same costs, same zero-control warm start) under the 2026-07-07 benchmark
config: SQP=1 (RTI), PCG cap 200 / rel 1e-4, rho 0.01, and MPCGPU running GATO_REG_PATTERN
with its native eta-exit (MPCGPU's tools/build.py figure-eight flags).

Fig-3 left = the N=64 row: batched total solve time at B in [1..128] for GATO (batched GPU),
a multi-threaded QDLDL-based CPU solver (pysqpcpu), MPCGPU (single-solve GPU -> B x per-solve), plus GATO's speedup
over each baseline at every B. Fig-3 right = the GATO N x B heat map (N in {8..128}, B up
to 512 — B>128 is GATO-only: MPCGPU cannot batch and BT is past core saturation).

Data stages (each appends CSVs under examples/benchmarks/data/; TIMING — quiet box only):
  --run-gato     sweep_batch_iiwa_fig8.py per N        -> sweep_fig8_gato.csv
  --run-bt       track_iiwa_fig8_bt.py per (N, B)      -> sweep_fig8_bt.csv
  --mpcgpu-timing-dir DIR   import MPCGPU's own timing-harness output (its figure-eight
                 plan, run from the MPCGPU repository) -> sweep_fig8_mpcgpu.csv
Default (no --run-*) assembles the table + figures from existing CSVs. Stages run
sequentially (never overlap timing). Re-runs append; assembly takes the LAST row per (N,B).

--robot indy7 runs the same harness on the paper's Indy7 (GATO + CPU lanes only — MPCGPU has
no Indy7 build or trajfile, so the goal is synthesized): CSVs and figures get an `_indy7`
suffix (sweep_fig8_gato_indy7.csv, fig3_fair_scalability_indy7.*); iiwa14 names are unchanged.

Examples::
    # full regeneration on a quiet box (GATO grid + BT B-sweep at N=64), MPCGPU imported
    python examples/paper-figures/reproduce_fig3_fair.py --run-gato --run-bt \
        --mpcgpu-timing-dir ../MPCGPU/tmp/timing/<fig8-run>
    # assemble only
    python examples/paper-figures/reproduce_fig3_fair.py
"""
import os
import csv
import argparse
import subprocess
import sys

import numpy as np

import _common as C

PY = sys.executable                        # the data stages run under THIS python

ROBOT = "iiwa14"                           # set by main(); the CSV/figure names carry non-iiwa14 robots
SUFFIX = ""
GATO_CSV = os.path.join(C.BENCH_DATA, "sweep_fig8_gato.csv")
BT_CSV = os.path.join(C.BENCH_DATA, "sweep_fig8_bt.csv")
MPCGPU_CSV = os.path.join(C.BENCH_DATA, "sweep_fig8_mpcgpu.csv")


def set_robot(robot):
    global ROBOT, SUFFIX, GATO_CSV, BT_CSV, MPCGPU_CSV
    ROBOT = robot
    SUFFIX = "" if robot == "iiwa14" else f"_{robot}"
    GATO_CSV = os.path.join(C.BENCH_DATA, f"sweep_fig8_gato{SUFFIX}.csv")
    BT_CSV = os.path.join(C.BENCH_DATA, f"sweep_fig8_bt{SUFFIX}.csv")
    MPCGPU_CSV = os.path.join(C.BENCH_DATA, f"sweep_fig8_mpcgpu{SUFFIX}.csv")


def _run(cmd, cwd=C.REPO, env=None):
    print(f"[fig3-fair] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, env=env)


def run_gato(N_list, batches, extra, solves):
    for N in N_list:
        C.require_module(ROBOT, N)
        blist = batches + [b for b in extra if b not in batches]
        _run([PY, "examples/benchmarks/sweep_batch_iiwa_fig8.py", "--robot", ROBOT, "--N", N,
              "--batches", ",".join(map(str, blist)), "--solves", solves, "--out", GATO_CSV])


def run_bt(N_list, batches, sim_time):
    # Use the build helper's selected Python and libraries, not GATO's venv.
    helper = os.path.join(C.BENCH_DIR, 'baselines', 'sqpcpu_env.sh')
    if not os.path.isfile(helper):
        raise SystemExit('Build the optional CPU baseline first; see baselines/README.md')
    script = os.path.join(C.BENCH_DIR, "baselines", "track_iiwa_fig8_bt.py")
    for N in N_list:
        for B in batches:
            _run(['bash','-c',
                  'set -euo pipefail; source "$1"; shift; '
                  ': "${GATO_CPU_PYTHON:?Rebuild the CPU baseline environment helper}"; '
                  'exec "$GATO_CPU_PYTHON" "$@"',
                  'cpu-baseline',helper,script,sim_time,B,N,BT_CSV,'--robot',ROBOT])


def import_mpcgpu(timing_dir):
    """Append MPCGPU single-solve cells from its own timing harness (tools/timing.py run).

    Uses the figure-eight plan's PCG workloads with reused workspaces, whose build flags match
    iiwa_fig8_shared (SQP=1, PCG cap 200 / rel 1e-4, rho 0.01). One aggregated row per N: the
    median of the per-repeat medians. Samples are MPCGPU's internal SQP time per control update.
    """
    import glob, json, statistics
    if ROBOT != "iiwa14":
        raise SystemExit(f"[fig3-fair] MPCGPU has no {ROBOT} lane — import only applies to iiwa14")
    cells = {}
    for path in sorted(glob.glob(os.path.join(timing_dir, "pcg-*-reuse-r*", "verdict.json"))):
        v = json.load(open(path))
        if not v.get("ok"):
            continue
        cells.setdefault(int(v["workload"]["knots"]), []).append(v)
    if not cells:
        raise SystemExit(f"[fig3-fair] no MPCGPU pcg reuse verdicts under {timing_dir}")
    new = not os.path.exists(MPCGPU_CSV)
    with open(MPCGPU_CSV, "a") as f:
        if new:
            f.write("N,B,median_ms,p90_ms,per_traj_us,n_solves,L2_mean\n")
        for N, runs in sorted(cells.items()):
            med = statistics.median(r["median_us"] for r in runs) / 1000
            p90 = statistics.median(r["p90_us"] for r in runs) / 1000
            l2 = statistics.mean(r["tracking_mean_l2"] for r in runs)
            f.write(f"{N},1,{med:.4f},{p90:.4f},{med*1000:.1f},{sum(r['samples'] for r in runs)},{l2:.6f}\n")
    print(f"[fig3-fair] imported MPCGPU horizons {sorted(cells)} from {timing_dir}")


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
    lines = [f"=== Fig-3 (left, FAIR): {ROBOT} fig8, N={N}, batched total solve time vs B ===",
             "config: SQP=1, PCG<=200 rel 1e-4, rho 0.01, shared fig8/EE-frame/costs; "
             "MPCGPU = its figure-eight build flags (GATO_REG_PATTERN, native exit), imported from its timing harness",
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
    with open(os.path.join(C.FIG_DIR, f"fig3_fair_scalability{SUFFIX}.txt"), "w") as f:
        f.write(txt + "\n")


def plot_fig3_left(N, batches, gato, bt, mpc):
    Bs = [B for B in batches if (N, B) in gato]
    if not Bs:
        print(f"[fig3-fair] no GATO cells at N={N} — skipping the fig3-left plot")
        return
    plt = C.set_paper_rcParams()
    fig = plt.figure(figsize=(7, 5))
    plt.plot(Bs, [gato[(N, B)] for B in Bs], "o-", color="#00693E", label="GATO (GPU, batched)")
    bBs = [B for B in batches if (N, B) in bt]
    if bBs:
        plt.plot(bBs, [bt[(N, B)] for B in bBs], "^-", color="#C90016",
                 label="Multi-threaded CPU (QDLDL)")
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
    C.savefig(fig, f"fig3_fair_scalability{SUFFIX}")


def report_heatmap(gato, N_list, all_batches):
    Ns = [n for n in N_list if any((n, b) in gato for b in all_batches)]
    Bs = [b for b in all_batches if any((n, b) in gato for n in Ns)]
    if not Ns or not Bs:
        print("[fig3-fair] no GATO heatmap cells yet — run --run-gato first")
        return None, None, None
    lines = [f"=== Fig-3 (right, FAIR): GATO {ROBOT} fig8 total batched solve time (ms) ===",
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
    with open(os.path.join(C.FIG_DIR, f"fig3_fair_heatmap{SUFFIX}.txt"), "w") as f:
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
                r, g, b, _ = im.cmap(im.norm(Z[i, j]))   # dark text on light cells, white on dark
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                        color="black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white")
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
    C.savefig(fig, f"fig3_fair_heatmap{SUFFIX}")


def main():
    p = argparse.ArgumentParser(description="Fig-3 data from the FAIR fig8 parity harness.")
    p.add_argument("--robot", choices=("iiwa14", "indy7"), default="iiwa14",
                   help="iiwa14 = the FAIR 3-way lane; indy7 = the paper's robot (GATO + CPU only)")
    p.add_argument("--run-gato", action="store_true", help="TIMING: GATO N x B sweep (quiet box)")
    p.add_argument("--run-bt", action="store_true", help="TIMING: multi-threaded QDLDL-based CPU solver B sweep (quiet box)")
    p.add_argument("--mpcgpu-timing-dir", help="import MPCGPU cells from its tools/timing.py output "
                   "(figure-eight plan); MPCGPU timing runs from its own harness")
    p.add_argument("--fig3-N", type=int, default=64, help="the fig3-left horizon (paper: 64)")
    p.add_argument("--N-list", default="8,16,32,64,128", help="heatmap horizons (GATO)")
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128", help="shared batch sizes")
    p.add_argument("--gato-extra-batches", default="256,512", help="GATO-only extra batch sizes")
    p.add_argument("--bt-N-list", default="64", help="BT horizons (BT N is a runtime arg)")
    p.add_argument("--solves", type=int, default=400, help="GATO solves per config")
    p.add_argument("--sim-time", type=float, default=6.0, help="BT closed-loop sim seconds")
    p.add_argument("--quick", action="store_true", help="tiny wiring smoke (NOT paper numbers)")
    args = p.parse_args()
    set_robot(args.robot)

    N_list = C.parse_int_list(args.N_list)
    batches = C.parse_int_list(args.batches)
    extra = C.parse_int_list(args.gato_extra_batches)
    solves, sim_time = args.solves, args.sim_time
    if args.quick:
        N_list, batches, extra, solves, sim_time = [16], [1, 8], [], 30, 1.0
        args.fig3_N = 16  # fig3-left must use a horizon the quick subset ran
        print("[quick] tiny subset — NOT paper numbers")

    if args.run_gato or args.run_bt:
        C.bench.require_quiet_gpu(allow_busy=args.quick)   # the --run-* stages are TIMING
    if args.run_gato:
        run_gato(N_list, batches, extra, solves)
    if args.run_bt:
        run_bt(C.parse_int_list(args.bt_N_list) if not args.quick else [16], batches, sim_time)
    if args.mpcgpu_timing_dir:
        import_mpcgpu(args.mpcgpu_timing_dir)

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

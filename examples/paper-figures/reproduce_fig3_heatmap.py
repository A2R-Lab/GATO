"""Regenerate Fig-3 (right): GATO solve-time heat map over batch M x horizon N.

Paper IV-B: the same Indy7 figure-8 task as Fig-3 left, sweeping both batch size
M in [1,2,...,512] and trajectory length N in {8,16,32,64,128}, showing GATO can
hit kHz control rates across a flexible (M, N) envelope (scalability tracks the
total knot count N*M).

Each (N) row comes from the canonical GATO generator examples/benchmark_fig8.py
(writes benchmark_fig8_{N}N.pkl). This script ensures those pkls exist (regenerating
per-N if the module is built) then renders the LogNorm heat map.

NOTE: the paper's original fig-8 heat-map data did not survive the repo migration
(see docs/archaeology.md), so a faithful figure requires a full re-run; ``--replot``
renders whatever benchmark_fig8_*.pkl are present. The heat map needs indy7 modules
for every N in the sweep (build with -DKNOTS="8;16;32;64;128").

Examples::
    python examples/paper-figures/reproduce_fig3_heatmap.py            # regen all N + plot
    python examples/paper-figures/reproduce_fig3_heatmap.py --quick    # tiny smoke
    python examples/paper-figures/reproduce_fig3_heatmap.py --replot   # plot existing pkls
"""
import os
import glob
import argparse
import pickle
import subprocess
from collections import defaultdict

import numpy as np

import _common as C

VENV = os.path.join(os.path.dirname(C.REPO), "GRiD", ".venv", "bin", "python")
PY = VENV if os.path.exists(VENV) else "python"


def regen_N(N, batch_sizes, sim_time):
    print(f"[paper-figures] $ benchmark_fig8.py --N {N}")
    subprocess.run([PY, "examples/benchmark_fig8.py", "--plant", "indy7", "--N", str(N),
                    "--batch-sizes", batch_sizes, "--sim-time", str(sim_time)],
                   cwd=C.REPO, check=True)


def load_all():
    rows = []
    for pf in sorted(glob.glob(os.path.join(C.REPO, "benchmark_fig8_*.pkl"))):
        with open(pf, "rb") as f:
            for r in pickle.load(f):
                if "error" not in r and r.get("avg_gpu_time_ms") is not None:
                    rows.append(r)
    return rows


def aggregate(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["batch_size"], r["N"])].append(r["avg_gpu_time_ms"])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def plot(cells):
    plt = C.set_paper_rcParams()
    from matplotlib.colors import LogNorm
    batch_sizes = sorted({b for (b, _) in cells if b != 1024})
    knots = sorted({n for (_, n) in cells})
    Z = np.full((len(knots), len(batch_sizes)), np.nan)
    for (b, n), t in cells.items():
        if b in batch_sizes:
            Z[knots.index(n), batch_sizes.index(b)] = t

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdYlGn_r",
                   norm=LogNorm(vmin=0.09, vmax=20.0),
                   extent=[-0.5, len(batch_sizes) - 0.5, -0.5, len(knots) - 0.5],
                   interpolation="nearest")
    for i in range(len(knots)):
        for j in range(len(batch_sizes)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white" if Z[i, j] > 2.0 else "black")
    # contours at ~10kHz/5kHz/1kHz/250Hz (robust auto-placed labels)
    levels = [lvl for lvl in (0.105, 0.2, 1, 4) if np.nanmin(Z) <= lvl <= np.nanmax(Z)]
    if levels:
        CS = ax.contour(np.round(Z, 2), levels=levels, colors="blue", linewidths=1.5)
        ax.clabel(CS, inline=True, fontsize=14,
                  fmt=lambda t: f"{1.0/t:.0f}kHz" if 1.0/t >= 1 else f"{1000.0/t:.0f}Hz")
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_yticks(range(len(knots)))
    ax.set_yticklabels([str(n) for n in knots])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Trajectory Length (N)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("GPU Solve Time (ms)")
    plt.tight_layout()
    C.savefig(fig, "fig3_heatmap")


def main():
    p = argparse.ArgumentParser(description="Regenerate Fig-3 right (heat map).")
    C.add_repro_args(p)
    p.add_argument("--N", default="8,16,32,64,128")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256,512")
    p.add_argument("--sim-time", type=float, default=5.0)
    args = p.parse_args()

    N_list = C.parse_int_list(args.N)
    batch_sizes, sim_time = args.batch_sizes, args.sim_time
    if args.quick:
        N_list, batch_sizes, sim_time = [16, 32], "1,8,64", 2.0
        print("[quick] tiny subset — NOT paper numbers")

    if not args.replot:
        for N in N_list:
            pkl = os.path.join(C.REPO, f"benchmark_fig8_{N}N.pkl")
            if args.regen or not os.path.exists(pkl):
                C.require_module("indy7", N)
                regen_N(N, batch_sizes, sim_time)

    cells = aggregate(load_all())
    if not cells:
        raise SystemExit("[paper-figures] no benchmark_fig8_*.pkl found — run without --replot.")
    plot(cells)


if __name__ == "__main__":
    main()

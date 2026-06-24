"""Regenerate every GATO paper figure in one go.

Runs each reproduce_figN_*.py script in turn (as a subprocess so a missing module
or a failing figure doesn't abort the rest), forwarding --replot / --quick, then
prints a tier summary. Use --only to select a subset.

Examples::
    python examples/paper-figures/make_all.py --quick     # smoke every figure
    python examples/paper-figures/make_all.py --replot    # re-render from saved data
    python examples/paper-figures/make_all.py --only fig4,fig5
"""
import os
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# (key, script, note)
FIGURES = [
    ("fig3", "reproduce_fig3_scalability.py", "Fig-3 left: Indy7 fig-8 scalability (GATO vs CPU/GPU)"),
    ("fig3heatmap", "reproduce_fig3_heatmap.py", "Fig-3 right: GATO solve-time heat map (needs indy7 N8..128)"),
    ("fig4", "reproduce_fig4_hparam.py", "Fig-4: CS1 online rho hyperparameter convergence"),
    ("fig5", "reproduce_fig5_disturbance.py", "Fig-5: CS2 disturbance rejection"),
    ("fig7", "reproduce_fig7_pickplace.py", "Fig-7 + Table-I: CS3 pick-place (GATED: iiwa14 instability)"),
]


def main():
    p = argparse.ArgumentParser(description="Regenerate all GATO paper figures.")
    p.add_argument("--replot", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--regen", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated keys, e.g. fig4,fig5")
    args = p.parse_args()

    keys = set(args.only.split(",")) if args.only else None
    fwd = [f for f, on in (("--replot", args.replot), ("--quick", args.quick),
                           ("--regen", args.regen)) if on]

    results = []
    for key, script, note in FIGURES:
        if keys and key not in keys:
            continue
        print(f"\n{'=' * 70}\n[make_all] {key}: {note}\n{'=' * 70}")
        cmd = [sys.executable, os.path.join(HERE, script), *fwd]
        rc = subprocess.run(cmd).returncode
        results.append((key, "OK" if rc == 0 else f"FAILED (rc={rc})", note))

    print(f"\n{'=' * 70}\n[make_all] SUMMARY\n{'=' * 70}")
    for key, status, note in results:
        print(f"  {key:12} {status:18} {note}")


if __name__ == "__main__":
    main()

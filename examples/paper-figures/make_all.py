"""Regenerate every GATO paper figure in one go.

Runs each reproduce_figN_*.py script in turn (as a subprocess so a missing module
or a failing figure doesn't abort the rest), forwarding --replot / --quick / --regen
where the script takes them, then prints a summary. Use --only to select a subset.

fig3 is the FAIR iiwa14 parity harness (reproduce_fig3_fair.py): by default it
assembles the table + plots from the committed sweep CSVs (no GPU); its TIMING data
stages (--run-gato/--run-bt/--run-mpcgpu) are quiet-box legs of
examples/benchmarks/run_timing_night.sh, not something make_all runs. The June
indy7 fig3 chain is archived under examples/archive/.

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

REPRO_FLAGS = ("--replot", "--quick", "--regen")
# (key, script, flags the script accepts, note)
FIGURES = [
    ("fig3", "reproduce_fig3_fair.py", ("--quick",),
     "Fig-3 (both): FAIR iiwa14 fig8 — table + heat map from the sweep CSVs (timing legs: run_timing_night.sh)"),
    ("fig4", "reproduce_fig4_hparam.py", REPRO_FLAGS, "Fig-4: CS1 online rho hyperparameter convergence"),
    ("fig5", "reproduce_fig5_disturbance.py", REPRO_FLAGS, "Fig-5: CS2 disturbance rejection"),
    ("fig7", "reproduce_fig7_pickplace.py", REPRO_FLAGS,
     "Fig-7 + Table-I: CS3 pick-place (100 scenarios; magnitudes carry the FE caveat in the script header)"),
]


def main():
    p = argparse.ArgumentParser(description="Regenerate all GATO paper figures.")
    p.add_argument("--replot", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--regen", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated keys, e.g. fig4,fig5")
    args = p.parse_args()

    keys = set(args.only.split(",")) if args.only else None
    on = [f for f, v in zip(REPRO_FLAGS, (args.replot, args.quick, args.regen)) if v]

    results = []
    for key, script, accepts, note in FIGURES:
        if keys and key not in keys:
            continue
        fwd = [f for f in on if f in accepts]
        print(f"\n{'=' * 70}\n[make_all] {key}: {note}\n{'=' * 70}")
        cmd = [sys.executable, os.path.join(HERE, script), *fwd]
        rc = subprocess.run(cmd).returncode
        results.append((key, "OK" if rc == 0 else f"FAILED (rc={rc})", note))

    print(f"\n{'=' * 70}\n[make_all] SUMMARY\n{'=' * 70}")
    for key, status, note in results:
        print(f"  {key:12} {status:18} {note}")


if __name__ == "__main__":
    main()

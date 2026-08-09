"""Paired summary table for a contact-wipe pool directory.

    python summarize_wipe.py data/wipe_20260809_...

Per-arm means + paired Wilcoxon (fc vs each baseline) on the continuous
metrics — the W3 lesson: no binary gates, continuous everywhere.
"""
import glob
import os
import pickle
import sys

import numpy as np

METRICS = ["fn_rms_err", "path_rms", "cone_viol_mean", "contact_loss_frac",
           "solve_ms_mean"]
UNITS = {"fn_rms_err": "N", "path_rms": "mm", "cone_viol_mean": "N",
         "contact_loss_frac": "%", "solve_ms_mean": "ms"}
SCALE = {"path_rms": 1000.0, "contact_loss_frac": 100.0}


def load(outdir):
    pools = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*_s*.pkl"))):
        d = pickle.load(open(p, "rb"))
        arm = d["protocol"]["arm"]
        sid = d["protocol"]["scenario"]["id"]
        pools.setdefault(arm, {})[sid] = d["metrics"]
    return pools


def main():
    outdir = sys.argv[1]
    pools = load(outdir)
    arms = [a for a in ("pos", "ucone", "fc") if a in pools]
    common = sorted(set.intersection(*[set(pools[a]) for a in arms])) if arms else []
    print(f"arms: {arms}   paired scenarios: {len(common)}")

    hdr = f"{'metric':<20}" + "".join(f"{a:>12}" for a in arms)
    print(hdr + "   (paired means)")
    for m in METRICS:
        sc = SCALE.get(m, 1.0)
        row = f"{m:<20}"
        for a in arms:
            v = np.array([pools[a][s][m] for s in common]) * sc
            row += f"{v.mean():>12.3f}"
        print(row + f"   [{UNITS[m]}]")

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("(scipy absent: skipping paired tests)")
        return
    if "fc" not in pools:
        return
    for base in [a for a in ("pos", "ucone") if a in pools]:
        print(f"\npaired Wilcoxon fc vs {base} (n={len(common)}):")
        for m in METRICS:
            x = np.array([pools["fc"][s][m] for s in common])
            y = np.array([pools[base][s][m] for s in common])
            d = x - y
            if np.allclose(d, 0):
                print(f"  {m:<20} identical")
                continue
            stat, p = wilcoxon(x, y)
            print(f"  {m:<20} fc-{base} median delta {np.median(d):+9.4f}  p={p:.2e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Camera-ready table + figure for the contact-wipe study (pure CPU).

Reads the committed n=24 paired pool (data/wipe_20260809_181705/, three arms:
pos = position-reference baseline, ucone = frozen-pinv EE cone rows on
torques, fc = contact-force slots + fc_ref + cone on the fc columns) and
emits, under <pool>/paper/:
  - wipe_table.md / wipe_table.tex — paired means/medians + Wilcoxon p's
  - wipe_scenarios.md              — full per-scenario appendix table
  - wipe_figure.png / .pdf         — fn(t) trace + paired per-scenario slopes

Solve-time caveat: the pool ran on a shared box; the quiet-box quotes
(08-12 overnight, GPU idle: pos 0.152 ms, fc 0.370 ms) are the numbers to
put in text. The paired solve-time COMPARISON (fc ≈ 2.3-2.4× pos) is stable
across both. ucone was not re-quoted (lost the bake-off).
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

POOL = Path(__file__).resolve().parent / "data" / "wipe_20260809_181705"
OUT = POOL / "paper"
ARMS = ["pos", "ucone", "fc"]
QUIET_MS = {"pos": 0.152, "fc": 0.370}   # 08-12 quiet-box quote (s00)
METRICS = [
    ("fn_rms_err", "Normal-force RMS error", "N"),
    ("path_rms", "Path RMS error", "mm"),
    ("cone_viol_mean", "Friction-cone violation (mean)", "N"),
    ("contact_loss_frac", "Contact loss", "%"),
    ("solve_ms_mean", "Solve time (mean)", "ms"),
]
SCALE = {"path_rms": 1e3, "contact_loss_frac": 1e2}   # m->mm, frac->%


def load_pool():
    data = {a: {} for a in ARMS}
    for a in ARMS:
        for p in sorted(POOL.glob(f"{a}_s*.pkl")):
            sid = int(p.stem.split("_s")[1])
            data[a][sid] = pickle.load(open(p, "rb"))
    sids = sorted(set.intersection(*(set(data[a]) for a in ARMS)))
    return data, sids


def metric_matrix(data, sids, key):
    s = SCALE.get(key, 1.0)
    return {a: np.array([data[a][i]["metrics"][key] * s for i in sids])
            for a in ARMS}


def main():
    OUT.mkdir(exist_ok=True)
    data, sids = load_pool()
    n = len(sids)
    fset = data["fc"][sids[0]]["protocol"]["F_set"]

    # ---------- table ----------
    md = [f"# Contact-wipe paired pool (n={n} scenarios, F_set={fset:g} N)\n",
          "| Metric | pos | ucone | fc | fc vs pos p | fc vs ucone p |",
          "|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Metric & pos & ucone & \\textbf{fc} & $p$ (fc/pos) & $p$ (fc/ucone) \\\\",
           "\\midrule"]
    for key, label, unit in METRICS:
        m = metric_matrix(data, sids, key)
        pp = wilcoxon(m["fc"], m["pos"]).pvalue
        pu = wilcoxon(m["fc"], m["ucone"]).pvalue
        row = {a: f"{m[a].mean():.3f}" for a in ARMS}
        md.append(f"| {label} [{unit}] | {row['pos']} | {row['ucone']} | "
                  f"**{row['fc']}** | {pp:.1e} | {pu:.1e} |")
        tex.append(f"{label} [{unit}] & {row['pos']} & {row['ucone']} & "
                   f"\\textbf{{{row['fc']}}} & {pp:.1e} & {pu:.1e} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    md.append(
        f"\nPaired means over the {n}-scenario pool; Wilcoxon signed-rank "
        "p-values on the paired per-scenario differences. Solve times are "
        "pool means from a SHARED box — quote the quiet-box numbers in text: "
        f"pos {QUIET_MS['pos']} ms, fc {QUIET_MS['fc']} ms per solve "
        "(fc ≈ 2.4× pos, both > 2.7 kHz).")
    (OUT / "wipe_table.md").write_text("\n".join(md) + "\n")
    (OUT / "wipe_table.tex").write_text("\n".join(tex) + "\n")

    # ---------- appendix: per-scenario ----------
    keys = [k for k, _, _ in METRICS]
    app = [f"# Per-scenario metrics (n={n})\n"]
    for key, label, unit in METRICS:
        m = metric_matrix(data, sids, key)
        app.append(f"\n## {label} [{unit}]\n")
        app.append("| scenario | theta | stroke | pos | ucone | fc |")
        app.append("|---|---|---|---|---|---|")
        for j, i in enumerate(sids):
            sc = data["fc"][i]["protocol"]["scenario"]
            app.append(f"| s{i:02d} | {sc['theta']:.2f} | {sc['stroke']:.2f} | "
                       f"{m['pos'][j]:.3f} | {m['ucone'][j]:.3f} | {m['fc'][j]:.3f} |")
    (OUT / "wipe_scenarios.md").write_text("\n".join(app) + "\n")

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    ax = axes[0]
    sim_dt = data["fc"][sids[0]]["protocol"]["sim_dt"]
    colors = {"pos": "tab:blue", "ucone": "tab:orange", "fc": "tab:green"}
    for a in ARMS:
        ch = np.asarray(data[a][sids[0]]["contact_history"])   # [ncon, fn, ft]
        t = np.arange(len(ch)) * sim_dt
        ax.plot(t, ch[:, 1], lw=0.9, color=colors[a], label=a)
    ax.axhline(fset, color="k", ls="--", lw=1, label=f"$F_{{set}}$ = {fset:g} N")
    ax.set_xlabel("time [s]"); ax.set_ylabel("normal force $f_n$ [N]")
    ax.set_title("scenario s00"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for ax, (key, label, unit) in zip(axes[1:], [METRICS[0], METRICS[1]]):
        m = metric_matrix(data, sids, key)
        xs = np.arange(len(ARMS))
        for j in range(n):
            ax.plot(xs, [m[a][j] for a in ARMS], "-o", color="gray",
                    alpha=0.35, ms=3, lw=0.8)
        ax.plot(xs, [m[a].mean() for a in ARMS], "-o", color="tab:red",
                lw=2.2, ms=6, label="mean")
        ax.set_xticks(xs, ARMS)
        ax.set_ylabel(f"{label} [{unit}]"); ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "wipe_figure.png", dpi=200)
    fig.savefig(OUT / "wipe_figure.pdf")
    print(f"wrote {OUT}/wipe_table.{{md,tex}}, wipe_scenarios.md, "
          f"wipe_figure.{{png,pdf}}  (n={n})")


if __name__ == "__main__":
    main()

"""Per-problem linsys policy autotuning: fit / decide / persist / resolve.

The pcg-vs-bdsv verdict is WORKLOAD-specific: it is decided by the task's
warm-startedness distribution (how often, and how far, the measured state
lands off the shifted prediction), not by the robot alone. The 08-12 CDF
study proved linsys="auto" (per-step selection on pred_err) matches or beats
both pure arms on the fig8 arms tasks — but a different disturbance profile
moves the right threshold, and a workload that is (almost) always cold or
always warm is better served by a pure arm with no switching at all.

Pipeline (probe legs are TIMING — quiet box only):
  1. tools/autotune_linsys.py runs the task once under pure pcg (collecting
     per-solve pred_err + solve ms) and once under pure bdsv (flat cost).
  2. fit_tau() estimates the pred_err where predicted pcg cost crosses the
     bdsv cost; decide_policy() turns that + the measured pred_err
     distribution into {pure pcg | pure bdsv | auto@tau}.
  3. save_tuning() persists the entry (keyed plant|N|task_tag) to
     linsys_tuning.json next to this file (registry-adjacent, like
     _registry.json); override the location with $GATO_LINSYS_TUNING.
  4. MPCController(task_tag=...) resolves at construction:
     explicit linsys arg > tuned entry > wired per-base default.

Everything in this module is host-only numpy (no GPU, no solver import) so
the logic is testable in the cpu lane.
"""
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / "linsys_tuning.json"


def _tuning_path(path=None):
    if path is not None:
        return Path(path)
    env = os.environ.get("GATO_LINSYS_TUNING")
    return Path(env) if env else DEFAULT_PATH


def _key(plant, N, task_tag):
    return f"{plant}|N{int(N)}|{task_tag}"


def fit_tau(pred_err, solve_ms, bdsv_ms, n_bins=12):
    """Fit the pred_err threshold where predicted pcg cost crosses bdsv cost.

    Args:
        pred_err, solve_ms: per-solve traces from a PURE-PCG probe run.
        bdsv_ms: the flat per-solve bdsv cost (median of the bdsv probe run).
        n_bins: quantile bins over pred_err for the cost fit.

    Returns (tau, diag):
        tau: crossing threshold, linearly interpolated between the bracketing
            bin centers. None if the (monotone) pcg cost never reaches bdsv_ms
            (pcg wins everywhere probed); 0.0 if it exceeds bdsv_ms already in
            the first bin (bdsv wins everywhere probed).
        diag: dict with the binned fit (bin_center / bin_cost arrays, n per
            bin) for reporting.

    The pcg-cost-vs-pred_err relation is fit as binned medians made monotone
    non-decreasing with a running max (iterations grow with how cold the warm
    start is; the running max removes small-sample inversions without
    assuming a functional form).
    """
    pe = np.asarray(pred_err, dtype=float)
    ms = np.asarray(solve_ms, dtype=float)
    if pe.shape != ms.shape or pe.ndim != 1 or pe.size < n_bins:
        raise ValueError(f"need matching 1-D traces with >= {n_bins} solves; "
                         f"got {pe.shape} / {ms.shape}")
    bdsv_ms = float(bdsv_ms)
    # quantile bin edges: equal occupancy, so sparse cold tails still get bins
    edges = np.quantile(pe, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)          # ties (many identical pred_err) collapse
    if edges.size < 3:
        raise ValueError("pred_err trace is (nearly) constant — probe a task "
                         "with disturbances before fitting")
    idx = np.clip(np.searchsorted(edges, pe, side="right") - 1, 0, edges.size - 2)
    centers, costs, counts = [], [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if not sel.any():
            continue
        centers.append(float(np.median(pe[sel])))
        costs.append(float(np.median(ms[sel])))
        counts.append(int(sel.sum()))
    centers = np.asarray(centers)
    costs = np.maximum.accumulate(np.asarray(costs))   # enforce monotone
    diag = {"bin_center": centers.tolist(), "bin_cost": costs.tolist(),
            "bin_n": counts, "bdsv_ms": bdsv_ms}
    above = costs >= bdsv_ms
    if not above.any():
        return None, diag
    j = int(np.argmax(above))
    if j == 0:
        return 0.0, diag
    # interpolate the crossing between the bracketing bin centers
    c0, c1 = costs[j - 1], costs[j]
    x0, x1 = centers[j - 1], centers[j]
    frac = 0.0 if c1 == c0 else (bdsv_ms - c0) / (c1 - c0)
    return float(x0 + frac * (x1 - x0)), diag


def decide_policy(pred_err, tau, lo=0.02, hi=0.98):
    """Turn a fitted tau + the probe's pred_err distribution into a policy.

    Fraction of probe solves that would run cold (pred_err > tau) decides:
    <= lo -> pure "pcg" (cold solves too rare to be worth switching),
    >= hi -> pure "bdsv" (warm solves too rare), else "auto"@tau.
    tau None (pcg never crossed bdsv) -> pure "pcg"; tau 0.0 (bdsv cheaper
    even warm) -> pure "bdsv".

    Returns {"policy": mode, "tau": float | None, "cold_frac": float}.
    """
    if tau is None:
        return {"policy": "pcg", "tau": None, "cold_frac": 0.0}
    pe = np.asarray(pred_err, dtype=float)
    cold_frac = float(np.mean(pe > tau)) if pe.size else 1.0
    if tau <= 0.0 or cold_frac >= hi:
        return {"policy": "bdsv", "tau": None, "cold_frac": cold_frac}
    if cold_frac <= lo:
        return {"policy": "pcg", "tau": None, "cold_frac": cold_frac}
    return {"policy": "auto", "tau": float(tau), "cold_frac": cold_frac}


def load_tuning(path=None):
    """The whole tuning table ({} when absent)."""
    p = _tuning_path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_tuning(plant, N, task_tag, entry, path=None):
    """Insert/replace the (plant, N, task_tag) entry; returns the file path.

    `entry` should carry at least {"policy", "tau"}; the autotune CLI adds
    provenance (date, SHA, probe stats)."""
    p = _tuning_path(path)
    table = load_tuning(p)
    table[_key(plant, N, task_tag)] = entry
    p.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return p


def lookup(plant, N, task_tag, path=None):
    """The tuned entry for (plant, N, task_tag), or None."""
    return load_tuning(path).get(_key(plant, N, task_tag))


def resolve_linsys(floating_base, linsys=None, bdsv_threshold=None,
                   plant=None, N=None, task_tag=None, path=None):
    """Controller-side resolution: explicit > tuned > wired default.

    Returns (linsys, bdsv_threshold) with linsys in
    {"pcg", "bdsv", "bdsv_first", "auto"} and bdsv_threshold set iff auto.
    """
    if linsys is None and task_tag is not None and plant is not None:
        entry = lookup(plant, N, task_tag, path)
        if entry is not None:
            linsys = entry["policy"]
            if linsys == "auto" and bdsv_threshold is None:
                bdsv_threshold = entry.get("tau")
    if linsys is None:
        # wired per-base defaults (2026-08-12): see MPCController docstring
        linsys = "bdsv" if floating_base else "auto"
    if linsys == "auto" and bdsv_threshold is None:
        bdsv_threshold = 0.1
    return linsys, bdsv_threshold

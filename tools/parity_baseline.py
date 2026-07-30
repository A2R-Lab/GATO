#!/usr/bin/env python
"""Capture / check a bitwise solve() parity baseline across built modules.

Usage:
  python tools/parity_baseline.py capture <out.npz>
  python tools/parity_baseline.py check   <base.npz>

Covers, per built (plant, N): cold + 3-solve warm chain, pcg + bdsv linsys,
and (largest N only) the barrier/admm/al row-group mechanisms. Arrays compared
with np.array_equal (bitwise for xu/iters; final_merit has +-1 ulp atomicAdd
jitter so it is stored but compared with rtol=1e-5).

The baseline pins the DEFAULT build (all flags off): any refactor that claims
"default path untouched" must reproduce it exactly. Used as the Phase-1
(USE_EXACT_HESSIAN) and R2 gates, 2026-07-30.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

import gato
from gato.config import INDY7_START_CONFIGS, IIWA14_START_CONFIGS

START = {"indy7": INDY7_START_CONFIGS["ready"], "iiwa14": IIWA14_START_CONFIGS["home"]}
URDFS = {
    "indy7": REPO / "examples" / "indy7_description" / "indy7.urdf",
    "iiwa14": REPO / "examples" / "iiwa_description" / "iiwa14.urdf",
}


def _inputs(plant, N, B):
    q0 = np.asarray(START[plant], dtype=np.float32)
    nq = q0.size
    x0 = np.concatenate([q0, np.zeros(nq, dtype=np.float32)])
    rng = np.random.default_rng(20260730)
    X = np.tile(x0, (B, 1)) + rng.normal(0, 0.01, (B, 2 * nq)).astype(np.float32)
    goals = np.zeros((B, N * 6), dtype=np.float32)
    goals[:, 0::6], goals[:, 1::6], goals[:, 2::6] = 0.35, 0.25, 0.5
    return X, goals


def _solver(plant, N, B, **kw):
    return gato.BSQP(model_path=str(URDFS[plant]), batch_size=B, N=N, dt=0.01,
                     max_sqp_iters=6, plant_type=plant, **kw)


def _cases():
    combos = sorted(k for k in gato.available() if k[0] in START)
    for plant, N in combos:
        for linsys in ("pcg", "bdsv"):
            yield f"{plant}_N{N}_{linsys}", plant, N, dict(linsys=linsys), None
    # row-group mechanisms on each plant's largest built N
    for plant in sorted({p for p, _ in combos}):
        N = max(n for p, n in combos if p == plant)
        for mech, setup in (
            ("barrier", lambda s: s.solver.enable_limit_barrier(1e-2, 0.1)),
            ("admm", lambda s: s.solver.enable_limit_admm(0.01, 10)),
            ("al", lambda s: s.solver.enable_limit_al(1.0)),
        ):
            yield f"{plant}_N{N}_{mech}", plant, N, dict(), setup


def run_all():
    out = {}
    B = 8
    for name, plant, N, kw, setup in _cases():
        s = _solver(plant, N, B, **kw)
        if setup is not None:
            setup(s)
        X, goals = _inputs(plant, N, B)
        # cold solve + 2 warm re-solves (warm chain exercises persistent state)
        for i in range(3):
            res = s.solve(X, goals)
        out[f"{name}_xu"] = res.xu
        out[f"{name}_iters"] = res.stats.sqp_iters
        out[f"{name}_merit"] = res.stats.final_merit
        print(f"  {name}: iters={res.stats.sqp_iters.tolist()}")
    return out


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("capture", "check"):
        sys.exit(__doc__)
    mode, path = sys.argv[1], sys.argv[2]
    got = run_all()
    if mode == "capture":
        np.savez_compressed(path, **got)
        print(f"baseline written: {path} ({len(got)} arrays)")
        return
    base = np.load(path)
    missing = sorted(set(base.files) - set(got))
    fails = []
    for k in sorted(got):
        if k not in base.files:
            continue
        if k.endswith("_merit"):
            ok = np.allclose(got[k], base[k], rtol=1e-5, atol=0)
        else:
            ok = np.array_equal(got[k], base[k])
        if not ok:
            d = np.abs(np.asarray(got[k], dtype=np.float64) - np.asarray(base[k], dtype=np.float64)).max()
            fails.append(f"{k}: maxdiff={d}")
    for f in fails:
        print(f"MISMATCH {f}")
    if missing:
        print(f"missing cases (baseline has, current run lacks): {missing}")
    if fails or missing:
        sys.exit(f"PARITY FAIL: {len(fails)} mismatches, {len(missing)} missing")
    print(f"PARITY OK: {sum(1 for k in got if k in base.files)} arrays bitwise-equal (merit rtol 1e-5)")


if __name__ == "__main__":
    main()

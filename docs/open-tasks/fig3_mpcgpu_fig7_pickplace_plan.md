# Plan: Fig-3 MPCGPU line (item 1) + Fig-7 iiwa14 pick-place (item 3)

Detailed, launch-ready plan for the two backlog items the user chose to explore (2026-06-24).
Grounded in recon of `fig3/indy7-mpcgpu`, `~/Desktop/MPCGPU`, `docs/baselines.md` (local),
`reproduce_fig3_scalability.py`, `reproduce_fig7_pickplace.py`, and the pick-place runner.
Held-local, unpushed convention still applies; perf findings stay in `docs/baselines.md`.

---

## ITEM 1 — MPCGPU GPU baseline for Fig-3 (Indy7 fig-8, N=64)

### Current state (recon)
- Branch **`fig3/indy7-mpcgpu`** has the substantive port committed: `94fdbe3` (iiwa14→indy7,
  N=64, h=0.01, 1 SQP iter; `mpcsim` returns full per-step sqp time) + `d5efd0c`/`e9e2e57`
  (sm_120 / CUDA-13 build fixes). **Builds + runs on sm_120.**
- MPCGPU repo at `~/Desktop/MPCGPU` (separate, frozen). Submodule pins recorded in
  `docs/baselines.md:139`. Run via the `/tmp/gato_mpcgpu` worktree recipe (`baselines.md:156-167`).
- **Timing RESOLVED:** the "293 µs linsys" was a mislabel — it was already the full per-step SQP
  time (`mpcsim.cuh` slot 0 under `TIME_LINSYS==1` times all of `sqpSolvePcg`). Clean gravity-off
  cost = **293 µs median**.
- **CSV exists + wired:** `baselines/mpcgpu_indy7_fig8_N64.csv` (gravity-off run). `assemble_fig3.py`
  / `reproduce_fig3_scalability.py::load_mpcgpu` read the last row (solve-time stats, µs) → median;
  `plot()` draws a **horizontal dashed line** at 0.293 ms (MPCGPU is single-solve → batch-invariant).
  So the Fig-3 MPCGPU **timing line already renders.**
- **The open problem is tracking fairness, not timing.** Gravity-off tracking is stuck at ~0.47 m
  and **weight-invariant**: MPCGPU's indy7 EE-tracking weight is hardcoded `1.0`
  (`indy7_plant.cuh:291`, no Q multiplier), dominating QD/R. Gravity-on tracks 0.065 m but is
  unstable (1.6 m excursions) + 4× slower (1128 µs, a KKT-conditioning artifact).

### DECISION POINT 1a (gravity config for the published bar)
- **(A) gravity-off** — clean 293 µs solve cost, but tracking 0.47 m and physics-mismatched vs
  GATO/OSQP (which run with gravity). Honest for a *timing* axis; flag the caveat.
- **(B) gravity-on** — fair physics, 0.065 m final, but unstable + 1128 µs (inflated by
  conditioning, not representative).
- **Recommended: (A) for the timing line + the fairness fix below** so the reported MPCGPU tracking
  isn't an embarrassing 0.47 m. **DECIDED 2026-06-24 (user): (A) gravity-off + EE-weight fix.**

### Work
1. **MPCGPU indy7 EE-weight fairness fix (the "tracking fix").** In MPCGPU `indy7_plant.cuh:291`,
   thread a `q_cost` multiplier into the EE-tracking cost term (mirror GATO's Q weighting) so the
   tracking weight is tunable instead of hardcoded `1.0`. Re-run the `/tmp/gato_mpcgpu` worktree,
   regenerate `baselines/mpcgpu_indy7_fig8_N64.csv` (gravity-off). Target: tracking materially
   below 0.47 m so the baseline is a legitimate (if weaker, 1-SQP-iter) competitor. **Bounded
   MPCGPU-side change; verify it builds + runs on sm_120 myself.**
2. **Re-render Fig-3** (`reproduce_fig3_scalability.py`) with the refreshed CSV; confirm the
   GATO curve + OSQP line + MPCGPU flat line all draw and the report table prints `vs MPCGPU` ratios.
   Update the Fig-3 caveat text (single-solve → flat line; gravity config; 1 SQP iter).
3. **PR the MPCGPU indy7 port upstream** (A2R-Lab/MPCGPU) once the fix lands — the port + sm_120
   build fixes + the EE-weight multiplier. Frozen-repo etiquette: small PR, note it's for the GATO
   Fig-3 fair comparison. (Per standing rule, PR only on explicit go-ahead.)

### Effort / risk
Low–medium. The hard parts (port, sm_120 build, timing) are done. The EE-weight fix is small and
local to MPCGPU's indy7 plant. Main judgment call is 1a (gravity config). **This item is ~80% done.**

---

## ITEM 3 — iiwa14 pick-place instability (Fig-7 + Table-I, CS3)

### Current state (recon)
- `reproduce_fig7_pickplace.py` runs; **Table-I = 0% success, NaN times** at both batch=1 and
  batch=8 (`examples/paper-figures/table_I.txt`, empty CDF). Docstring already carries the caveat.
- **Task:** iiwa14 (7-DoF) + **15 kg pendulum payload** modeled in *sim* as a 3-DoF spherical joint
  (length 0.3–0.7 m, damping 0.1–0.6, randomized) but **fully UNMODELED in the solver** — a hidden,
  swinging ~75 N disturbance with ~2 s swing period vs the 160 ms MPC horizon.
- **Config** (`config.py:52-73`): N=16, h=0.01, max_sqp_iters=5, pcg_tol=1e-6, **rho=0.001**,
  q_cost=5.0. Module `bsqpN16_iiwa14.so` built.
- **The paper's CS3 robustness mechanism is BATCHED force-hypothesis sampling** (the ForceEstimator:
  sphere-sample candidate wrenches → evaluate_best_trajectory). **batch=1 has no force estimator**,
  so 0% at batch=1 is half-expected. The real question is whether **batched + force-estimator with a
  sample range covering ~75 N -Z** succeeds.
- Related but UNVERIFIED: a documented EE-Hessian/solver-conditioning non-determinism exists on the
  `vendor-grid-submodule` branch (correct GN Hessian → NaN). `cleanup-modernization` may use a
  different cost path — do **not** assume same root cause; characterize empirically first.

### Phase 0 — Characterize the failure (empirical, do FIRST; ~hours)
Goal: know *how* it fails and *which* variable controls it before tuning anything.
1. **Instrument** one scenario/one goal: log per-step `sqp_iters`, `pcg_iters`, final merit,
   `max|dz|`, NaN flags, goal distance vs time. (Add temporary stat plumbing in `mpc_controller.py`
   / `interface.py`; remove before commit.)
2. **Four controlled runs** (the diagnostic cross-product):
   | run | payload | batch + force-est | isolates |
   |---|---|---|---|
   | a | **off** | batch=1 | does the iiwa14 N16 solver track at all (no disturbance)? |
   | b | on | batch=1 (no FE) | the unmodeled-payload divergence baseline |
   | c | on | **batch≥8 + FE, sample range ⊇ 75 N -Z** | the paper's actual method |
   | d | off | batch≥8 + FE | FE overhead/regression with no disturbance |
   - **Branch logic:** if (a) fails → it's the SOLVER/config on iiwa14 N16, not the payload (attack
     rho/conditioning). If (a) passes & (c) passes → harness was just under-powered (batch=1); Fig-7
     reproduces — **possibly the whole fix.** If (a) passes & (c) fails → genuine payload-robustness
     gap → Phase 1/2.
3. **Confirm the ForceEstimator sample range** actually covers the payload magnitude/direction
   (`force_estimator.py`); a sphere that never samples ~75 N -Z can't help.

### Phase 1 — Solver/config robustness (if Phase 0 points at the solver; ~days)
Sweep on 5–10 scenarios, one knob at a time, measure success-rate + stability:
- **rho** 0.001 → 0.01, 0.05, 0.1 (condition the KKT; the documented "no single rho" risk lives here).
- **pcg_tol** 1e-6 → 1e-4, 1e-3; **max_sqp_iters** 5 → 8, 10; **q_cost** 5 → 2, 1 (less aggressive).
- **horizon N** 16 → 24, 32 (more foresight vs the swing; needs those modules built).
- Pick the cheapest combo that stabilizes; record in `docs/baselines.md`.

### Phase 2 — Structural (only if Phase 1 insufficient; ~week, uncertain)
- Extend/retune the ForceEstimator to track the swinging payload (time-correlated sampling, not iid).
- Augment the solver model with an estimated constant -Z wrench (partial payload modeling).
- These are real R&D with uncertain payoff — gate on Phase 0/1 results before committing.

### Honest exit criteria
- **Best case** (Phase 0c passes): Fig-7/Table-I reproduce with batching → done.
- **Likely case:** Phase 1 tuning gets a non-zero, batch-increasing success curve (paper *shape*),
  numbers may not match exactly → ship with the documented caveat.
- **Worst case:** structural gap → record findings, keep the caveat, escalate to user. **Do not
  fake numbers; a faithful "shape-correct, magnitude-caveated" Fig-7 is the floor.**

### DECISION POINT 3a (scope / aggressiveness)
- **(A) Phase 0 + tuning only** (bounded; accept caveat if tuning can't fully close it). *Recommended.*
- **(B) allow Phase 2 structural** (force-estimator R&D / state augmentation; open-ended).
**DECIDED 2026-06-24 (user): (A) Phase 0 + tuning only.** If tuning yields a shape-correct,
batch-increasing success curve, ship Fig-7 with the documented caveat; do NOT enter open-ended
Phase 2 structural R&D without escalating back to the user first.

---

## Sequencing & mechanics
- **Independent items** → can run in parallel, but both touch the GPU (serialize builds/timing per
  the no-contention rule; correctness runs can overlap). Recommend: **MPCGPU first** (bounded,
  ~80% done, fast win) then the **iiwa14 Phase 0 diagnostic** (which itself may resolve item 3 cheaply).
- Builds capped `--parallel 2–4`; check `free -g`/`nvidia-smi` before each; GRiD venv for pinocchio.
- Findings → `docs/baselines.md` (local). Deliverable code/figures → committed (short-line msgs,
  no Co-Authored-By). No pushes / no PRs without explicit go-ahead.
- Validate subagent GPU work myself (compile+run) before trusting [[feedback_subagent_bash_revoked_verify_self]].

## Recon citations
MPCGPU: `docs/baselines.md:44-67,136-167`; `reproduce_fig3_scalability.py:37,74-107`;
`assemble_fig3.py:43-52`; branch `fig3/indy7-mpcgpu` `94fdbe3`; `~/Desktop/MPCGPU/.../indy7_plant.cuh:291`.
Pick-place: `reproduce_fig7_pickplace.py:13-16,28`; `config.py:52-104`; `mpc_controller.py:351-610`;
`force_estimator.py`; `table_I.txt`; plan `note-that-other-agents-async-castle.md:66-67,157-164`.

# Changelog

Unreleased on `cleanup-modernization` (no tag: installs are source-tree only, plan D10).

## 2026-09-20 — audit waves 0–2

Breaking (clean-break API, no shims):
- `gato.SolverParams` is the ONE solver configuration; `BSQP(model, B, N, dt, params=...)`
  (field keywords accepted). The constructor's second default set and `kkt_tol` are gone;
  defaults = the paper/MPC values (`max_sqp_iters=1`, `mu=10`, `rho=0.01`, `qd_cost=1e-2`, …).
- `BSQP.solve(x, ref, xu_warm=None)` is stateless in the trajectory (None = hold-at-x cold
  start; pass `res.xu` to warm-start); caller arrays are never mutated; `SolveResult.xu` is
  a fresh array. `BSQP.XU_B` / `clear_cost_weights_per_knot` removed
  (`set_cost_weights_per_knot(None)` clears).
- `MPC_GATO(params=..., linsys=..., bdsv_threshold=..., reseed_threshold=..., variant=...)`
  replaces the `solver_params` dict; `StepResult.u` is the ACTUATED control (`StepResult.fc`
  carries wrench slots on fc variants).
- Module variants have their own names: `bsqpN{N}_{plant}[_fc|_eh].so`, side by side;
  `BSQP(..., variant=)`, `gato.available(variant)`, `gato.build(contact_forces=, exact_hessian=)`,
  `-DMODULES="plant:N[:variant]"`, `./tools/build.sh --variant`. The `.so`-swap workflow is gone.
- `enable_limit_*` after an appended row group raises (used to drop the group silently).
- C++: `BSQP` constructor drops `dt`/`kkt_tol`; every hand-written device/host function is
  snake_case; raw bindings constructor has 14 args.
- `gato.config.{STANDARD_BATCH_SIZES, EXPERIMENT_BATCH_SIZES, BATCH_COLORS}` moved to
  `examples/paper-figures/_common.py`.

Fixed:
- Strided / broadcast / f64 numpy inputs were copied through their strides (garbage reads):
  every binding array parameter is now `c_style|forcecast`.
- `TrajectoryReference.done()` fired 5N knots early (knot/element unit mix).
- Max-dynamic-shared-memory attribute was latched at the first request; a later larger carve
  (collision after exact-Hessian) launched over-ceiling. Post-launch checks on every launch.
- Dead L2 persisting carve-out reservation removed; pinocchio no longer required to construct
  a solver; `gato.build` no longer silently overridden by a cached receipt-profile configure.

Infrastructure:
- pytest-gpu-proof 0.4.0 (schema 3, restricted signer, weekly CI cron); receipt module
  profile `test/receipt_modules.txt` (arms × 6 horizons + go2 N16 + fc/eh arms N16); golden
  bitwise gate (`test/golden/`, 34 cases); kernel harnesses run from pytest; distribution
  contract (pure-python wheel, buildable sdist) gated; HTTPS submodules; one `.venv`.
- GRiD → 3af782e, GLASS → e83b086; regen emits only the consumed families (headers −21%).
- CUDA: GLASS banded block movers in the Schur assembly, shared gamma/BDSV/ADMM helpers,
  shared-memory layout structs (sizer == carve), dead solver state removed; all bitwise.

## 0.0.2 (2026-06 → 2026-08)
Modernization: GRiD/GLASS pins, runtime batch size, `gato.build`, pytest + signed receipts,
constraint row-group layer (boxes, EE rows, cones, collision; barrier/ADMM/AL), exact-Hessian
SO-SQP, contact-force controls (fc), floating base (go2), hybrid pcg/bdsv + autotune plumbing,
MuJoCo worlds, wipe contact task, paper figure reproduction.

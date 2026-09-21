# Changelog

Unreleased on `cleanup-modernization` (no tag: installs are source-tree only, plan D10).

## 2026-09-21 — D14: the Fig-3 CPU baseline pin is public

- `examples/benchmarks/baselines/sqpcpu` now points at `A2R-Lab/sqpcpu` (the public
  fork of EmreAdabag/sqpcpu), branch `fig3-fair-sigma` = upstream master + the two
  fair-comparison commits GATO pins. The `update = none` opt-in gating is gone: a plain
  recursive clone and `tools/install.sh` fetch it like every other submodule.

## 2026-09-20 — CL-4 groundwork: gait oracle, per-knot row masks, per-knot fc reference

- `gato.gait.GaitSchedule`: the fixed-gait ORACLE (trot/bound/pace/walk/stand; period,
  duty, phase offsets) → rolling per-knot stance masks, fc pin masks, per-knot fn
  references, Raibert-lite footholds, swing curves, base references. Pure numpy.
- Row groups carry a per-knot row-activity mask (`BSQP.set_row_group_mask(g, mask)`,
  `get_row_groups()[g]["active"]`): an inactive (knot, row) is treated like a knot
  below the group's window (ADMM keeps its proximal fold, everything else contributes
  nothing) and its AL dual is reset. All-True is bitwise the previous behaviour.
- `set_fc_ref` accepts an (N, n_fc) per-knot table (a 1-D wrench still broadcasts,
  bitwise); the device buffer is per knot.

## 2026-09-20 — Wave F: contact forces on the floating base (go2 fc-on-feet)

- `bsqpN16_go2_fc`: the fc variant now builds on the floating base (was a
  `static_assert`). The fc tail (one world-aligned `[n; f]` wrench per baked foot
  frame, `FC_SIZE = 24`) enters the grid step as the mapped per-body wrench; the
  linearization gains the fc columns of B and the dfext/dq chain term, composed from
  the grid step's full-force B block; the floating cost composition carries the fc
  regularization/reference terms. Gated against pinocchio central FD at 25 N
  wrenches, bitwise the default module at fc = 0. `go2 16 fc` joins the receipt
  profile (+2 goldens).
- API: `BSQP.contact_frames`, `BSQP.fc_slots(frame, part)`, module attr
  `NUM_CONTACT_FRAMES`, `MuJoCoWorld.last_contact["fn_by_body"]`.
- Fixed: `debug_contact_dynamics` returned silently wrong numbers on floating
  modules (fixed-base composition) — it now raises there. Kernel launches past the
  48 KB dynamic-smem default are opted in at every launch site via one helper
  (the Schur assembly kernel had no opt-in: 64 KB on go2-fc failed to launch);
  the setup_kkt/merit collision sizers under-allocated the dedicated carve by the
  temp tail (latent — every generated robot's carve fit the tail). setup_kkt's
  terminal cost blocks overlay the dead dynamics scratch (offsets only, −10 KB:
  the device opt-in maximum is ~99 KB, go2 default sat at 94 KB).
- Fixed (found by memcheck once that slack was gone): the fixed-base setup_kkt and
  merit kernels sized their dynamics scratch from the plant adapter's arena alone,
  omitting the integrator layer's own qdd|dqdd / qdd|err prefix (114 floats on
  indy7 N8) — the ID-gradient inner wrote past the launch by up to that much, hidden
  for months by the prepended terminal chain. `linearizedDynamics_TempMemCt` /
  `integratorError_TempMemCt` / `simStep_TempMemCt` (integrator.cuh) are now the
  sizers. Results are bitwise unchanged (goldens).

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

# Changelog

Unreleased (no tag; installs are source-tree only).

## 2026-10-04 — Audit evidence and reproducibility

- Signed GPU receipts now require the full test inventory, wider source coverage
  and content identities matching all receipt-profile native modules. Missing
  receipts and partial test selections fail closed; numerical goldens are unchanged.
- A frozen Fig-7 holdout records arm-arrival outcomes separately from applied
  effort, velocity and position limits, including a torque-clamped stop arm.
  See the [figure refresh](docs/figure-refresh-2026-10-01.md) for interpretation.
- The optional CPU comparison baseline builds in its own pinned Pinocchio 3.8
  environment, with a two-arm numerical check and the matching Python launcher.
- An opt-in quiet-window lane pairs controller-call wall time with internal
  solver time. Documentation distinguishes simulated task quality, feasibility
  and timing boundaries; no new performance claim follows from correctness runs.

## 2026-10-03 — Fig-7 refresh: two task settings, identified-weight hypothesis batch

- `reproduce_fig7_pickplace.py --task {stop,pass-through}` presets. `stop` (default): the EE
  reference ramps to each goal over 1.5 s (minimum jerk) and the success gates must hold 100 ms,
  with the paper's exploration sampler — 15/100/99/98 % at B = 1/8/32/128 on 100 corrected
  scenarios. `pass-through`: the paper's protocol verbatim with the identified-weight sampler —
  26/83/97/95 % (the paper's sampler: 6/62/84/82 %). `--success-plot` renders both panels.
- `gato.estimators.IdentifiedWrenchSampler` and `MPC_GATO(estimator="wid")`: the hypothesis
  batch is the least-squares identified payload weight, a zero row and bounded Fibonacci-sphere
  perturbations with an adaptive radius (`inertial_rows=True` adds the full identified wrench
  and its blends — off by default: they fling the arm). The API default stays `estimator="fe"`.
- `MPC_GATO.run_mpc_goals(settle_time=, goal_ramp=)`: optional dwell on the success gates and a
  minimum-jerk EE reference between goals (both 0 by default = the paper's loop); exposed on
  `reproduce_fig7_pickplace.py` as `--settle-time`, `--goal-ramp`, `--qd-cost`, `--estimator`.
- Fig-3 Indy7 lane re-run in an exclusive window (October 3); Fig-7 findings and the reasons
  the gate stays instantaneous are in `docs/figure-refresh-2026-10-01.md`.

## 2026-10-02 — Fig-3 harness: `--robot indy7` + fig8 driver stop-condition fix

- `MPC_GATO.run_mpc_fig8` stopped `6N` goal steps early instead of `N`: a goal of exactly
  sim-steps + N knots produced 148 control steps at N = 32 and none at N ≥ 64. The long
  MPCGPU goal file (1202 steps) hid it on the iiwa14 lane; the wipe task padded around it.
  Fixed (`eepos_offset + N > steps`), regression test `test_mpc_gato_run_mpc_fig8_uses_whole_goal`.
  No committed number changes: every existing caller's goal is far longer than its sim time.
- The FAIR fig8 harness is robot-parameterized: `iiwa_fig8_shared.robot(name)` carries URDF,
  registry EE frame, q0 and whether MPCGPU's goal file applies; `sweep_batch_iiwa_fig8.py`,
  `track_iiwa_fig8_gato.py`, `baselines/track_iiwa_fig8_bt.py` and `reproduce_fig3_fair.py`
  take `--robot {iiwa14,indy7}`. iiwa14 outputs are unchanged (same goal hashes as every
  recorded run); Indy7 uses the same fig8 at `INDY7_START_CONFIGS["ready"]`, a synthesized
  goal (MPCGPU has no Indy7 trajfile) and `_indy7`-suffixed CSV/figure names. Indy7 numbers
  are recorded in `docs/figure-refresh-2026-10-01.md` once measured in a quiet window.

## 2026-10-01 — Release-hygiene batch (no numeric change: all 42 goldens bit-identical)

- Cooperative row kinds are described once. `gato/bsqp/rowgroups.cuh` lost the per-kind
  EE_POS / COLLISION / CONTACT_POS sections (scratch sizers, `has_*_rows`, `*_eval*`, six
  fold/value wrappers, the three telemetry branches and the collision-only AL dual branch):
  the `CooperativeRows<T, KIND>` trait now owns the FK call, the scratch carve and its sizes,
  the Jacobian layout, the bound slot and the state index, and `dispatch_cooperative_kind`
  routes the telemetry, AL dual-update and ADMM kernels to it. setup_kkt / merit call
  `cooperative_row_grad_hess<KIND>` / `cooperative_row_cost_value<KIND>` directly
  (`__noinline__` kept: the cicc expansion cliff). −180 lines across rowgroups.cuh, admm.cuh,
  setup_kkt.cuh, merit.cuh (the handoff's −250 estimate predated the 09-26 trait landing). Gated bitwise FIRST: four new arm `collision_{al,admm}` goldens
  (indy7 / iiwa14 N16, sphere parked on the reach goal so the rows bind) join the existing
  masked-CONTACT_POS and ADMM/EE cases — 38 → 42 goldens, all unchanged by the refactor.
- `python/bindings.cu` 720 → 632 lines: one `DenseArray` alias, `checked_ptr` /
  `optional_ptr` for the per-solve and per-joint setters, one `state_dict` behind the four
  dual/ADMM getters, `per_iter_array` for the (iters, B) stats. Same Python surface; the
  per-joint vector size errors now print the expected count instead of the constant's name.
- Scratch sizers follow the tree's snake_case `*_smem_ct` convention:
  `stepValueFloating_TempMemCt` → `step_value_floating_smem_ct`, `stepGradFloating_TempMemCt`
  → `step_grad_floating_smem_ct`, `simStep_TempMemCt` → `sim_step_smem_ct`,
  `linearizedDynamics_TempMemCt` → `linearized_dynamics_smem_ct`, `integratorError_TempMemCt`
  → `integrator_error_smem_ct` (integrator.cuh / grid_plant_step.cuh and their kernel callers).
- Retired `examples/archive/` (the June indy7 fig3 chain, the pinocchio-sim baseline, the
  bdsv timing session; nothing imported it — recover from git history). The bundled Fig-4
  data moved next to its only consumer: `examples/paper-figures/gato_hparam_batch_results.pkl`
  (`reproduce_fig4_hparam.py --replot`).
- New `docs/development.md`: install, capped builds, test tiers and goldens, the signed
  receipt and its local verify command, the quiet-window timing rule. Linked from
  `docs/README.md` and the README Tests section.

## 2026-10-01 — Main integration candidate and figure refresh

- Pinned GRiD `main` (0a14c0f); all vendored headers regenerate byte-identically. Merged the
  remote README citation and funding edits. Quick start, package URL and the GRiD submodule
  branch now point at `main`.
- Seed A/B timing: the benchmark's seed change explains nearly all of the September gap;
  about 3% (B8) and 4% (B128) remain unattributed (docs/figure-refresh-2026-10-01.md).
- Fixed Fig-4, which plotted nothing: best-merit curves are now padded past early SQP stops
  instead of truncated to the shortest. Refreshed Fig-3, Fig-4 and Fig-5 with the current code;
  Fig-3's MPCGPU lane now imports MPCGPU's own timing-harness output. Results and limits:
  `docs/figure-refresh-2026-10-01.md`. Fig-7 / Table I stay as published.
- Timing launchers no longer need ripgrep.
- Fixed a solver bug: the merit's terminal-knot cost read the unwritten control slot with
  zero weight, so NaN or Inf left in shared memory by an earlier kernel (for example a
  diverging tiny-rho solve) made 0 * NaN = NaN. That solve's initial merit became NaN and its
  line search never accepted a step, nondeterministically and only after such a kernel. The
  slot is now zeroed; all 38 goldens are bit-identical and a regression test replays the
  triggering sequence. Found because Fig-4 batches up to 128 with rho down to 1e-8.

## 2026-09-27 — Checkpoint follow-up fixes (no new timing claims)

- Fixed floating effort-vector uploads/checks to use actuator width, not stored
  configuration width; default/fc KKT gates cover the contract and scalar reset.
- Fixed spherical-payload initialization: axis-angle is integrated on the
  manifold into a unit quaternion, and augmented robot q/v offsets are separate.
  Both simulation task loops share this assembly. New pick-place data is marked
  `unit-quaternion-pendulum-v2`; historical pools remain preserved, not relabeled
  as corrected results. Added fixed-pacing traces and per-goal event diagnostics.
- Added explicit controller reset seeds and restored the batch benchmark's
  historical zero-tail initial guess. Hold remains an explicit comparison mode;
  correctness-only raw/controller parity and frozen reference provenance guard
  future comparisons. Added a guarded seed A/B timing suite; execution deferred.
- Recorded the overnight checkpoint and its limitations; reconciled preferred
  citation metadata with the README's ICRA entry. No production tuning changes.
- Fresh lean-install validation exposed a missing SciPy codegen dependency:
  trimesh fell back to a different collision sphere cover. SciPy is now a base
  build dependency, fallback geometry is rejected, and old codegen cache keys
  are invalidated. No upstream sources or vendored headers were changed.

## 2026-09-27 — Merge-readiness documentation and example validation

- Added a documentation index, feature-status boundaries, API migration notes
  and a bounded merge checklist. Corrected standing, ADMM dispatch, receipt
  coverage and paper-protocol claims; restored main's citation/acknowledgement
  in the README. Quick start builds one module with capped concurrency.
- Fixed the gym example's indentation error and added introductory syntax
  gates. The basic MPC example now uses fixed simulation pacing. Added a short
  masked-contact AL/barrier/ADMM gate suitable for race instrumentation.
- Added a GATO-only, opt-in quiet-window checkpoint runner with a GPU-free
  dry run, unique result paths and source/binary/environment provenance. It
  samples Fig-3 and Fig-7, not a full research reproduction or autotune run.
- Corrected the distribution contract: Git checkout plus pinned submodules
  is the supported native build; the sdist does not bundle external trees.
  Excluded local working notes from sdists and added a packaging canary gate.

## 2026-09-26 — GRiD integration and cooperative-row cleanup

- GRiD pinned to `65fd051`: public HTTPS nested dependencies, declared trimesh
  dependency, and updated integrator/plant surfaces. Regenerated all three robot
  headers. GATO's floating adapter continues to select Euler/semi-implicit Euler
  by enum name; its integration scheme is unchanged.
- Added masked floating CONTACT_POS goldens for AL and ADMM before consolidating
  the cooperative EE, collision and contact fold/merit/ADMM implementations.
  Traits retain each evaluator's scratch layout, Jacobian layout, bounds, masks
  and dual-state indexing. Existing noinline boundaries are preserved.
- Consolidated binding input uploads and row-state downloads, named owned
  scratch counts consistently, and shared the plant's linalg arena padding.
- Corrected notebook package shadowing, documented contact-wipe reproduction and
  the local-only stale data archives, and added package project URLs. Stance
  documentation now distinguishes moving prediction targets from measured slip.

## 2026-09-26 — Closed-loop foot rows: non-PD regularization bump, re-anchored stance targets; release hygiene

- Solver: a non-PD direct factor (`stats.pcg_iters == 2`, the f32 Cholesky of the Schur
  system failing once stiff row groups are folded into Q) now multiplies the trust-region
  rho by `NON_PD_RHO_FACTOR` (settings.h, 100: 1e-3 -> 0.1 in one step) independent of the
  AL mode's frozen adaptation. Measured on the go2 stance rows at AL rho 1e3: every
  iteration non-PD before, none after the first iteration of a solve now. The two
  exact-Hessian goldens moved (each hits one non-PD factor mid-solve) and end at a lower
  merit; re-baselined.
- `GaitProgrammer`: stance targets are RE-ANCHORED to the measured feet every tick (the
  no-slip contact semantics) instead of frozen at touchdown; the static stand with stance
  rows now holds height within 2 mm at 2-5 mm residual (gate:
  `test_fc_mpc_stands_with_foot_rows`). The swing itself remains open: lifting a foot without
  a support-polygon base plan tips the model (docs/constraints.md, closed-loop section).
- Bug fix: masked-off COLLISION knots no longer fold gradient/Hessian/merit (the row-mask
  contract already promised it; the fold and value sites lacked the gate).
- `test_build` compiles in its own CMake tree (it used to reconfigure the shared `build/`
  to the test plant).
- Release hygiene: `gato.rowkinds` (the row-kind / mechanism ids, shared by the API, the KKT
  certificate — now covering LIN_U / COLLISION / CONTACT_POS — and the tests);
  `gato.fingerprint` / `gato.worlds` / `gato.gait` reachable as attributes; the seven
  constraint docstrings that duplicated docs/constraints.md now summarize and point there;
  `add_contact_pos_rows` defaults to knots 1..N-1; batch accessors, the four limit enables,
  the mechanism/rho boilerplate and six copies of the test reach problem deduplicated;
  dangling references to internal notes removed from tracked code and docs; dead
  `tools/clean.sh` and unused helpers deleted (`test/expected_skips.txt` stays: it is the CI
  verifier's empty allow-list of skips, i.e. the zero-skip gate); README/docs corrected
  (import forms, submodule story, Fig-3 reproduction, contact-frame order, telemetry
  semantics); `docs/baselines.md` (orphaned, self-contradictory) removed and
  `docs/archaeology.md` trimmed to the provenance table; sdist now ships docs/ and the
  changelog; packaging metadata (authors, license, classifiers) completed.

## 2026-09-26 — CL-4: contact-frame POSITION rows (stance/swing feet), solver-level gates

- Row kind `CONTACT_POS` (`BSQP.add_contact_pos_rows`, `set_row_group_targets`,
  `contact_positions`): per-knot residual rows `p_f(q_k) - tgt_k` on the baked contact
  frames over GRiD's new `contact_frame_positions[_gradient]` surface, folded at every
  active knot (AL / relaxed barrier / ADMM / telemetry), with a per-knot target table on
  the row-group descriptor and the CL-4 masks selecting (knot, foot) rows. The dedicated
  cooperative carve of setup_kkt / merit is shared with the collision rows (sized when
  either kind is registered; defaults untouched — goldens bitwise).
- `GaitProgrammer.install_foot_rows`: a stance group (footholds frozen at touchdown, at
  ground height) and a swing group (swing-curve tracking to the planned landing);
  `apply(t, q)` retargets and re-masks both every tick beside the fc pins and the fn
  reference. Default AL rho 100 (loop-stable; see below).
- Gates (test_contact_rows.py): device residual == pinocchio, per-knot target table,
  AL/ADMM move an arm EE to a violated target within 3 mm on the active knots only,
  masks select knots, go2 keeps its feet < 1 mm under a lateral goal, solver-only S3
  (FR lifts ~3 cm at knot 12 with the stance feet held), programmer plumbing.
- OPEN (measured, docs/constraints.md, closed-loop section): the closed loop. The stiffness that holds feet within
  mm (AL rho 1e3) makes the loop's SQP reject every line-search step on 26/150 ticks of a
  static stand (base sags 4 cm); every ADMM variant collapses; AL rho 100 stands with a
  1.8 cm residual. S2 weight shift / S3 lift-a-foot closed-loop gates are not shipped.
- `pyproject`: mujoco pinned `<3.14` (3.14.0 segfaults inside MjSpec compile of the go2
  URDF with the welded root; CI cpu-lane at 2b03a38, reproduced locally 6/6).

## 2026-09-26 — GRiD pin 3af782e → 5904dbd (contact-frame positions), GLASS e83b086 → 8ce68a2

- GRiD `modernizing-tests` tip: the contact-frame position surface GATO asked for on
  2026-09-20 (`grid_plant::contact_frame_positions[_gradient]`, world positions of the
  baked contact origins + tangent Jacobians, receipt-covered upstream @8aac88e) and the raw
  `grid_plant::multi_target_position[_gradient]` evaluators (GATO nit 2). The three plants
  are regenerated (`tools/regen_grid.py`); `gato.build` idempotency key recorded for go2.
  GLASS moved by docs/figures only (headers byte-identical).
- iiwa14 goldens RE-BASELINED: URDFParser 29d3d78 no longer rationalizes transform
  coefficients, so iiwa14's emitted transforms carry ±1.6e-15 terms the old parser snapped
  to 0 (indy7/go2: bit-identical). The golden problems are unconverged 10-iteration solves
  that amplify any ULP change to O(1) (a 1e-7 input nudge moves xu by ~1%), so the drift was
  attributed with the referee gates (dynamics fingerprint, f_ext, exact Hessian, row groups
  vs the pinocchio-pinned tables: 54 passed) and a clean compute-sanitizer memcheck of
  iiwa14 N8 + both go2 modules. Sensitivity note in test_parity_golden.py.

## 2026-09-21 — D14: the Fig-3 CPU baseline pin is public

- `examples/benchmarks/baselines/sqpcpu` now points at `A2R-Lab/sqpcpu` (the public
  fork of the upstream sqpcpu), branch `fig3-fair-sigma` = upstream master + the two
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
  bitwise gate (`test/golden/`, 36 cases); kernel harnesses run from pytest; distribution
  contract (pure-python wheel, buildable sdist) gated; HTTPS submodules; one `.venv`.
- GRiD → 3af782e, GLASS → e83b086; regen emits only the consumed families (headers −21%).
- CUDA: GLASS banded block movers in the Schur assembly, shared gamma/BDSV/ADMM helpers,
  shared-memory layout structs (sizer == carve), dead solver state removed; all bitwise.

## 0.0.2 (2026-06 → 2026-08)
Modernization: GRiD/GLASS pins, runtime batch size, `gato.build`, pytest + signed receipts,
constraint row-group layer (boxes, EE rows, cones, collision; barrier/ADMM/AL), exact-Hessian
SO-SQP, contact-force controls (fc), floating base (go2), hybrid pcg/bdsv + autotune plumbing,
MuJoCo worlds, wipe contact task, paper figure reproduction.

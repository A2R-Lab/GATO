# GATO consumer contract — conventions external integrations depend on

Audience: anyone driving GATO from outside this repo (closed-loop harnesses,
other lab solvers, hardware pipelines, comparison benchmarks). Updated 2026-09-26.
For supported configurations and experimental boundaries, see [feature status](status.md).

## 0. First thing to run: the dynamics fingerprint

`test/dynamics_fingerprint.json` pins forward-dynamics qdd at canonical
(q, qd, u) probes per plant (fixed base, gravity −9.81, zero external wrench,
URDF pinned by sha256). Before comparing controllers or closing a loop against
your own simulator, verify **your** model is the same robot:

```python
import gato.fingerprint as fp
res = fp.check(my_qdd_fn, "iiwa14")   # my_qdd_fn(q, qd, u) -> qdd, no GPU needed
print(fp.report(res))                  # per-joint inertia-response ratios
```

A per-joint ratio away from 1.0 on the `inertia_j` probes means your model and
ours disagree about that joint's effective inertia — and every closed loop
built across that disagreement has a mis-designed gain. Provenance: a ×2.6–3.2
mismatch on iiwa14 joint 7 (MuJoCo harness vs URDF) presented as a "solver
instability" and took five debugging rounds to name (PDDP round-5). The
`inertia_j` probes are qd = 0, hence immune to damping/friction modeling;
disagreement isolated to the `coriolis` probe points at damping instead.

## 1. Frames and kinematics

- **EE frame**: the device FK (tracking cost and EE row-groups) targets the
  URDF `ee_frame` via named-target `*_EE` codegen — this INCLUDES the terminal
  fixed-joint origin. `BSQP.ee_pos(q)` is the same frame to f32 precision.
  Provenance: before 2026-07-30 the device frame dropped the terminal fixed
  joint (≈40 mm on iiwa14); any calibration captured against the old convention
  inverted into a constant offset after the regen. If you maintain your own
  FK, verify against `ee_pos` at a few postures, not against history.
- **Gravity is −9.81** (world z-down, matches pinocchio/MuJoCo defaults).

## 2. Limits and barriers

- **`JOINT_LIMIT_MARGIN` = −0.1** (plant.cuh; units per table — rad, rad/s, N·m): the baked barrier limits are
  the URDF limits SHRUNK by 0.1 rad on each side (position AND torque tables).
  A reference that is "in-limit" by the URDF can still sit past the barrier's
  effective limit. Provenance: this explained an entire experiment matrix of
  "barrier-on always diverges" (PDDP round-2).
- Barrier weights (`q_lim_cost`, `vel_lim_cost`, `ctrl_lim_cost`) default to
  {0.01, 0, 0} (`gato.SolverParams`): **torque limits are NOT enforced by
  default** — either enable
  `ctrl_lim_cost`, add box rows, or clamp applied torque driver-side (do the
  last one anyway if your plant is real hardware).

## 3. Cost semantics (what a "tracking cost" actually is)

- Running cost = EE-position quadratic (`q_cost`) + qd quadratic (`qd_cost`)
  + u quadratic (`u_cost`) + barriers; terminal knot swaps `N_cost` for the EE
  weight and drops the control terms. The q-position block is ZERO unless the
  posture anchor is on.
- **Posture anchor** (`set_q_pos_cost` + `set_q_nom`): adds
  0.5·w·‖q − q_nom‖² per knot. Two non-obvious consequences: (a) the +w·I
  Hessian term applies at ZERO posture error too — it regularizes the rank-3
  GN EE Hessian, so anchored solves legitimately converge deeper and command
  larger (honest) torques; "inert at q == q_nom" is true of the value and
  gradient only. (b) It closes position feedback on previously-uncontrolled
  nullspace joints — if your sim disagrees with our model there (see §0), the
  anchor is what exposes it.
- **Per-joint weights**: `set_u_cost_vec([...])` requires `n_actuated` entries
  (Go2: 12), excluding base coordinates and contact-wrench slots.
  `set_q_pos_cost([...])` uses stored configuration coordinates (`nq`; Go2: 19).
  Both default and contact-force Go2 variants have dedicated effort-vector KKT
  gates in `test/test_anchor.py`. On an arm, vector weights price one joint's
  channel without changing the others.
- Weights land literally on the KKT diagonals — `debug_setup_kkt` exposes the
  blocks, and `test/test_anchor.py` shows the pattern for verifying any cost
  claim directly instead of arguing from solve outcomes.

## 4. Solve interface

- **ONE configuration object**: `gato.SolverParams` (frozen dataclass; derive
  with `.replace(...)`). `BSQP(model_path, batch_size, N, dt, params=...)`, or
  keyword overrides of its fields. Defaults are the fixed-base real-time-iteration
  baseline, NOT every paper experiment's configuration (max_sqp_iters=1; mu=10, rho=0.01,
  qd_cost=1e-2, u_cost=2e-6, q_lim_cost=0.01, pcg_tol=1e-4, max_pcg_iters=200).
  Provenance: until 2026-09-20 the constructor carried a SECOND default set
  (mu=1, rho=1e-3, max_sqp_iters=10, ...) that no harness ever ran — if you
  copied constructor defaults, re-check against `SolverParams()`. `kkt_tol`
  is gone (it never terminated anything; convergence is merit/step based).
  `solver.params` records configuration fields; setters for those fields update
  it. Row groups, vector weights and target tables have their own state and are
  not a complete replay configuration in `SolverParams` alone.
- **linsys**: `SolverParams.linsys` None = the wired static default (pcg fixed
  base, bdsv floating base) — the static arm of `MPCController`'s per-step
  policy ("auto" = pcg warm / bdsv_first cold, same resolver:
  `gato.linsys_autotune.resolve_linsys`). An explicit mode always wins.
- **Array kinds.** `solve()` / `step()` take numpy, torch or JAX arrays and
  return the same kind. CUDA tensors (torch on a CUDA device, JAX on a GPU)
  use the device path: inputs are read in place through their device pointer
  (float32 + contiguous; converted on the device otherwise) and `xu` comes
  back on the device through DLPack — no host round trip, bit-identical to the
  host path. A device `xu_warm` must live on the same device and is used as
  given (knot-0 states replaced by `xcur_B`; no floating-base quaternion
  healing). CPU tensors take the host path. Inputs must be complete on the
  device: synchronize side streams before calling. The controller runs the
  host path (its warm-start buffer is host memory) and returns `u` in the
  caller's kind.
- `solve(xcur_B, eepos_goals_B, xu_warm=None)` — ALL batch-dim arrays; xu layout
  is `[x_0, u_0, x_1, u_1, ..., x_{N-1}]`, row length `N*(nx+nu) − nu`.
  **Stateless in the trajectory**: `xu_warm=None` seeds a hold at `xcur_B`
  (a cold start — NOT the previous solution); pass the last `SolveResult.xu`
  to warm-start. Your array is never modified; `SolveResult.xu` is a fresh
  array. AL/ADMM outer iteration = warm-started repeat solves, so thread
  `xu_warm` explicitly in such loops. The solver IS stateful in duals, ADMM
  state and the adapted trust-region rho (`reset_dual/reset_rho/reset`).
  Provenance: the pre-09-20 solver kept a hidden buffer that aliased the
  returned `xu` and mutated caller arrays in place.
- **Module variants** are separate ABIs with their own module names:
  `bsqpN{N}_{plant}` (default), `_fc` (contact-force controls appended to u),
  `_eh` (exact Hessian). `BSQP(..., variant="fc")`; `gato.available("fc")`.
  Row-group ORDER is enforced: `enable_limit_*` before any appended group
  (`add_lin_u_rows`, `enable_ee_terminal_equality`, `enable_collision`, `add_contact_pos_rows`) or
  it raises (they used to be dropped silently).
- **Control width is a MODULE property**: `nu = CONTROL_SIZE` from the module
  (on `GATO_CONTACT_FORCES` builds, `CONTROL_SIZE = ACTUATED_SIZE + FC_SIZE`).
  Use `SolveResult.u0()/control_at(k)` for the APPLIED (actuated) control —
  never slice xu by pinocchio's nv.
- **fc slots are per baked contact frame**: `BSQP.contact_frames` lists the
  frames in codegen order (`_registry.json`), `BSQP.fc_slots(frame, part=None|"n"|"f")`
  gives the 6-wide `[n; f]` block (world-aligned, about the frame origin) or
  its moment/force half — feed it to `add_fc_box`, `set_fc_ref`, `fc_at`.
  Arms bake one frame (the EE); go2 bakes the four `*_foot_joint` frames
  (`FC_SIZE = 24`, module attr `NUM_CONTACT_FRAMES = 4`). Floating base
  consumes fc through the grid step: the fc columns of B and the dfext/dq
  chain term are composed from the full-force B block, gated against
  pinocchio FD (`test_floating_go2.py`). `add_fc_box` rows are TELEMETRY unless
  a mechanism is chosen (`mech="al"` / `enable_limit_*` first) — a pin that
  does not pin is the default-mech trap.
- `MuJoCoWorld.last_contact["fn_by_body"]` splits the normal force per robot
  link (URDF link name) — the per-foot signal the standing gates read.
- **`MPCController.step()` contract.** One call = one solve from the measured
  state, warm-started from the previous tick's shifted plan, of up to
  `SolverParams.max_sqp_iters` SQP iterations (each with its PCG/ADMM budget);
  it returns when those are spent, when the merit test stops it, or when a
  time budget is hit. **Time budget**: `step(x, ref, budget_s=...)` (and
  `solve(..., budget_s=...)` on the raw solver) is measured from entry to the
  call; the solver runs the next SQP iteration only while the time spent plus
  that iteration's predicted cost still fits, then returns the best plan so
  far (the line search accepts descent steps only, so that is the current
  trajectory). The prediction is the measured cost of the same iteration in
  the previous budgeted solve, since an MPC tick repeats its predecessor's
  profile (the first iteration carries the PCG work, the later ones are cheap
  refinements); the most expensive iteration of the current solve stands in
  when there is no history. At least one iteration always runs and the
  cut is at an iteration boundary, never inside a kernel, so the budget holds
  to within one iteration. It is a budget, not a hard real-time guarantee:
  launch latency and the caller's own loop are outside it, and a GPU shared
  with other work stretches the iteration it measures. `budget_s=None`
  (default) is the unbudgeted path, unchanged by the option: the step takes as
  long as its iterations take and the caller owns the schedule (choose
  `max_sqp_iters` and `N` so the measured step fits your period; the paired
  internal-vs-wall numbers are in the [figure refresh](figure-refresh-2026-10-01.md),
  controller call boundaries). `StepResult.iterations`, `budget_hit` and
  `wall_s` (the raw solver: `SolverStats.sqp_iters`, `budget_hit`) report what
  happened. `StepResult.u` is the ACTUATED control only (what you apply;
  `StepResult.fc` carries the contact-wrench slots on fc variants);
  `StepResult.solve.stats` carries the iteration counts and, once a row group
  exists, the true violation of the returned plan (`row_max_violation`;
  `StepResult.row_viol` is the winner's, per row group). **Limits**: on a
  fixed-base arm whose solver has no row groups, the controller installs the
  URDF position, velocity and torque boxes under the ADMM mechanism with the
  measured defaults (rho 0.01, cap 10, residual exit at eps_rel 1e-2, inner
  linear solver by horizon) and `StepResult.limits` reads "admm"; a solver
  the caller configured is left as configured, floating-base solvers get
  nothing, `limits="off"` is the unconstrained step of 0.1.1 bit for bit, and
  `"admm"` / `"barrier"` / `"al"` / `"telemetry"` (with `limits_kw`) pick a
  mechanism. The plan then respects the limits to within the reported
  residual (measured: 3e-2 rad/s on a 1 rad/s velocity box at N64), in
  closed loop on the tasks of `docs/constraints.md`; it is a bounded,
  measured property, not a certificate, and the executed torque is what the
  caller applies (clamp it; the integrator's own safety layer stays).
  Solves are bit-deterministic run-to-run; batch entries are independent. A
  budgeted solve is deterministic in its result given the iteration count it
  reached, but that count depends on the clock.
- The measured state enters as a hard initial-state constraint (x_0 is data;
  don't put state limits at knot 0 — they'd be unsatisfiable under violation).

## 5. What the gpu-proof receipt does and does not attest

The signed receipt (`gpu-proof.json`, verified in CI) attests the recorded tests
at the fingerprinted sources and pinned dependencies, using the modules in
[`test/receipt_modules.txt`](../test/receipt_modules.txt). This includes default,
contact-force (`fc`) and exact-Hessian (`eh`) variants—not just default builds.
[`test/expected_skips.txt`](../test/expected_skips.txt) is an EMPTY allow-list:
any skipped test fails receipt verification.

Coverage includes bitwise goldens, determinism, finite-difference and KKT gates,
and specific simulator scenarios. It does NOT establish correctness for all
robots/settings, arbitrary variant combinations, your hardware controller, or
runtime/compile performance. Check which test exercises the feature you use.

## 6. Migrating from the paper-era API

- Import `gato`, not `bsqp`; native modules live under `python/gato/`.
- Use `SolverParams` for solver settings. Removed `kkt_tol` was not a working
  stopping criterion; do not replace it with an unrelated tolerance.
- Pass `xu_warm=previous.xu` explicitly when repeating raw solves. For a moving
  horizon, let `MPCController` own the shift and reset policy.
- Load `variant="fc"` or `variant="eh"` explicitly; never swap `.so` files.
- Apply `StepResult.u` / `SolveResult.u0()`, not the contact-wrench tail.
- Install canonical limit groups before appended groups; the reverse order raises.
- Check your simulator with the dynamics fingerprint before retuning a controller.
- `gato.build` reconfigures its selected build directory. Supply `build_dir=`
  for an isolated custom build, or restore the receipt profile afterwards.

## 7. Reporting a solver discrepancy

Open a GitHub issue with the URDF, the module name (`gato.module_name(plant, N, variant)`), the
`SolverParams`, and a replayable `.npz` of `(x, goals, xu_warm)` plus the observed vs expected numbers.

## 8. C++/CUDA interface

`BSQP<T>` (`gato/bsqp/bsqp.cuh`, header-only; `T` = float from `settings.h`)
is the solver the Python class wraps — same constructor arguments as
`gato.SolverParams` (batch, SQP/PCG budgets, `pcg_tol`, `solve_ratio`, `mu`,
cost weights, `rho`), same `enable_limit_*` / `add_*` / `set_*` surface on host
pointers, and `solve(T* d_xu_traj_batch, ProblemInputs<T>)` on device pointers
you own: `d_x_s_batch` (B, nx), `d_reference_traj_batch` (B, 6N), and
`d_xu_traj_batch` (B, N*(nx+nu) − nu) as warm start in / solution out, in the
Python layout. Dimensions are compile-time constants (`gato::constants::
XU_STATE_SIZE`, `CONTROL_SIZE`, `XU_TRAJ_SIZE`, `REFERENCE_TRAJ_SIZE`) fixed by
the defines `KNOT_POINTS`, `GATO_PLANT_NAME`, `GATO_PLANT_HEADER` (+ one per
variant). `python -m gato.paths --robot R --N N [--variant V]` prints the
include directories (GATO, the robot's generated headers, GLASS and the
collision header from the pinned grid-rbd wheel), defines and flags;
`examples/cpp/` has the template and a standalone CMake. The time budget is
`set_time_budget_us(us)` (0 = off) with the stop reported in
`SQPStats::budget_hit`, the same iteration-boundary semantics as `budget_s`
above. The ADMM residual exit is `set_admm_exit(eps_abs, eps_rel, check_every)`
with the inner work in `SQPStats::admm_iters` / `admm_linsys_iters`. Stable:
the class, `ProblemInputs`, `SQPStats` (`solve_time_us`, per-row
`sqp_iterations`, `budget_hit`, `admm_iters`, `admm_linsys_iters`), the three
defines and the array layouts. Not stable: private members, kernel
signatures, and anything under `gato/bsqp/kernels/`.

## 9. API stability

Stable (the release surface; changes are announced in the changelog and come
with a migration note):

- `gato.SolverParams`, `gato.BSQP` (`solve`, the `set_*` cost surface, the
  `enable_limit_*` / `add_*_rows` / `enable_collision` constraint surface,
  `ee_pos`, `sim_forward`), `SolveResult`, `SolverStats`.
- `gato.MPCController` / `StepResult`, `gato.MPCPolicy` with
  `TrajectoryReference` / `GoalReference`.
- `gato.build`, `gato.available`, `gato.robot_info`, `gato.fingerprint`,
  the module naming (`bsqpN{N}_{plant}[_fc|_eh]`) and `_registry.json`.
- `gato.MPC_GATO` and `gato.worlds` as the simulation drivers the paper
  figures run on (their stats dictionaries are part of the figure protocols).
- `ForceEstimator` / `ForceHypothesisBatch` (the paper's Fig-7 sampler).

Experimental (usable, tested, but shaped by ongoing research; may change
without a migration note):

- `IdentifiedWrenchSampler` (`MPC_GATO(estimator="wid")`), `CEMForceEstimator`
  and the B=1 `wrench_id` path.
- Go2 locomotion beyond standing: `GaitSchedule` / `GaitProgrammer`, foot masks
  and swing targets (closed-loop weight shift and lift are in development).
- `gato.linsys_autotune` (the pcg/bdsv policy fitter) and `gato.certificate`.
- `ArmTrackEnv` (the gymnasium wrapper) and the `contact-task` study scripts.


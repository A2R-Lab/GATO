# GATO consumer contract — conventions external integrations depend on

Audience: anyone driving GATO from outside this repo (closed-loop harnesses,
other lab solvers, hardware pipelines, comparison benchmarks). Every item here
has bitten a real integration; the provenance notes say how. Updated 2026-09-20
(SolverParams / stateless solve / module variants — see §4).

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

- **`JOINT_LIMIT_MARGIN` = −0.1 rad** (plant.cuh): the baked barrier limits are
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
- **Per-joint weights**: `set_u_cost_vec([...])` and `set_q_pos_cost([...])`
  accept length-nq arrays — the knob for pricing a single joint's channel
  (e.g. a near-massless wrist in a 100 Hz loop) without touching the rest.
- Weights land literally on the KKT diagonals — `debug_setup_kkt` exposes the
  blocks, and `test/test_anchor.py` shows the pattern for verifying any cost
  claim directly instead of arguing from solve outcomes.

## 4. Solve interface

- **ONE configuration object**: `gato.SolverParams` (frozen dataclass; derive
  with `.replace(...)`). `BSQP(model_path, batch_size, N, dt, params=...)`, or
  keyword overrides of its fields. Its defaults are the values every paper
  experiment ran with (max_sqp_iters=1 = real-time iteration; mu=10, rho=0.01,
  qd_cost=1e-2, u_cost=2e-6, q_lim_cost=0.01, pcg_tol=1e-4, max_pcg_iters=200).
  Provenance: until 2026-09-20 the constructor carried a SECOND default set
  (mu=1, rho=1e-3, max_sqp_iters=10, ...) that no harness ever ran — if you
  copied constructor defaults, re-check against `SolverParams()`. `kkt_tol`
  is gone (it never terminated anything; convergence is merit/step based).
  `solver.params` is always truthful: every `set_*` writes through.
- **linsys**: `SolverParams.linsys` None = the wired static default (pcg fixed
  base, bdsv floating base) — the static arm of `MPCController`'s per-step
  policy ("auto" = pcg warm / bdsv_first cold, same resolver:
  `gato.linsys_autotune.resolve_linsys`). An explicit mode always wins.
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
  (`add_lin_u_rows`, `enable_ee_terminal_equality`, `enable_collision`) or
  it raises (they used to be dropped silently).
- **Control width is a MODULE property**: `nu = CONTROL_SIZE` from the module
  (on `GATO_CONTACT_FORCES` builds, `CONTROL_SIZE = ACTUATED_SIZE + FC_SIZE`).
  Use `SolveResult.u0()/control_at(k)` for the APPLIED (actuated) control —
  never slice xu by pinocchio's nv.
- `MPCController.step()` returns `StepResult.u` = the ACTUATED control only
  (what you apply; `StepResult.fc` carries the contact-wrench slots on fc
  variants). Solves are bit-deterministic run-to-run; batch entries are
  independent.
- The measured state enters as a hard initial-state constraint (x_0 is data;
  don't put state limits at knot 0 — they'd be unsatisfiable under violation).

## 5. What the gpu-proof receipt does and does not attest

The signed receipt (`gpu-proof.json`, verified in CI) attests the committed
test suite ON THE DEFAULT BUILD at the fingerprinted sources: default-path
bitwise parity, determinism, FD gates, the KKT-level cost gates. It does NOT
attest: whichever variant modules are absent from the receipt profile
(`test/receipt_modules.txt`; `test/expected_skips.txt` lists exactly which
tests skip when a variant is not built),
your driver's closed-loop behavior, or timing. If you depend on a feature,
check a test exercises it — "the suite is green" is scoped by the suite.

## 6. Filing issues across agents

Cross-repo asks live as docs (`docs/open-tasks/ask_from_*.md`, relayed by the
human). What makes rounds converge fast, measured over five of them: send the
exact interface crossing (npz of x0/ref/warm/kwargs of ONE failing call) —
a replayable artifact settles in one round what config tables cannot.

# GATO constraint layer — mechanisms, measured defaults, provenance

Reference for the row-group constraint API on `gato.BSQP` (CL-0..CL-3 arcs).
The code docstrings carry the CONTRACT (what each call does); this page keeps
the measured tuning rulings, R1/R2 evaluation provenance and traps that used
to live inline. Moved out of `interface.py` on 2026-09-20 (plan 2.A).

Ordering rule (enforced, raises): `enable_limit_*` FIRST, then appended groups
(`add_lin_u_rows` / `enable_u_cone` / `add_fc_box` / `enable_ee_terminal_equality`
/ `enable_collision`) — a mechanism enable reinstalls the canonical limit groups.

Mechanisms: `telemetry` (report only) · `barrier` (relaxed log barrier, soft)
· `admm` (OSQP-style projection inner loop on the reused bdsv factor,
approximately hard) · `al` (PHR augmented Lagrangian, outer loop = warm-started
repeat solves — pass `xu_warm` explicitly).

## `set_admm_linsys(mode)`

ADMM inner-loop linear solver: "pcg" (default) | "bdsv_factor".

"pcg" runs warm-started PCG per ADMM iteration (λ carries across the
loop, no factorization); "bdsv_factor" factors the (constant-within-
the-loop) Schur matrix once per SQP iteration and re-solves per ADMM
iteration. Identical iterates up to linsys tolerance; only affects
MECH_ADMM solves.

Default BOUND "pcg" by the 2026-08-01 quiet-box A/B: 1.4-2.5x faster
per solve at identical tracking on every ADMM family (box fig8
16.5->6.6 ms indy7 / 18.9->9.7 ms iiwa14; cone 2.4x/2.2x; collision
1.6x/1.4x). The trade: PCG's looser inner residual leaves transient
box violations ~2-3x higher (same order, still enforced — e.g.
3.7e-2 -> 1.0e-1 on fig8 boxes). Pick "bdsv_factor" when tightest
transient enforcement matters more than speed.

## `set_exact_hessian(on)`

Toggle the SO-SQP stage-Hessian PSD projection for subsequent solves.

Per-TASK feature: wins on EE-terminal tasks, neutral-to-worse on
full-rank joint-terminal ones (so_sqp_prototype/RESULTS_2026-07-17).
Raises if the module was built without -DGATO_EXACT_HESSIAN=ON.

Constraint-mechanism pairing (measured): exact pairs with AL, not ADMM
(R2 2026-07-30: un-parks AL-cone both plants; admm_ee diverges). The
2026-08-01 ADMM-fold cells extend the rule: exact x cone-ADMM parks
BOTH plants (track 0.67-0.78 vs 0.02-0.03 GN) and exact x
collision-ADMM parks iiwa14; the one healthy pairing is indy7
collision-ADMM (parity with GN, tighter inner residual). Don't
combine exact with ADMM row groups by default.

## `enable_limit_telemetry()`

Install the canonical limit row-groups (position/velocity/torque boxes
from the URDF limit tables) in TELEMETRY mode: every solve() reports each
group's true violation of the returned trajectory in
``stats.row_{max,sum}_violation`` (group order BOX_Q, BOX_QD, BOX_U).
Telemetry never touches the solver path — trajectories are bit-identical
with it on or off. Part of the constraint row-group layer (CL-0).

## `enable_limit_barrier(mu, delta)`

Bind the limit row-groups to the RELAXED log-barrier mechanism: a
C² barrier with bounded Hessian (quadratic extension within ``delta``
of a bound) folded into the KKT cost and merit — infeasible-start safe,
the constraint layer's soft prior mode. Additive to grid_plant's own
clamped log barriers; zero q_lim/vel_lim/ctrl_lim_cost for a clean
comparison. Telemetry (stats.row_*_violation) stays on.

## `enable_limit_admm(rho, iters)`

Bind the limit row-groups to the ADMM-projection mechanism: an
OSQP-style fixed-budget inner loop per SQP iteration on a REUSED
direct (bdsv) factorization — the constraint layer's
"approximately hard" mode. ``rho`` is the ADMM penalty (fixed within
a solve; adapt it between solves), ``iters`` the fixed budget.
R1 default rho=0.01: the penalty must ride the COST-HESSIAN scale —
rho >= 1 swamps the u-block (natural scale u_cost=1e-6), freezing
controls at the warm start (closed-loop MPC parks); the measured
pocket is ~0.005-0.02 (r1_report_2026-07-11.md).
iters=10 BOUND by R2 (r2_report_2026-07-30.md): 2/5 park the feasible
cone cell; box cells saturate by 10 (20 = marginal viol gains at 2x
inner cost).
Duals warm-start across solves (reset_dual() reinitializes) —
EXCEPT equality rows (lo == hi, e.g. enable_ee_terminal_equality),
whose (z, y) reinit every solve: a warm-started dual on a row the
primal may not reach is an unbounded violation integrator (measured).
stats gain admm_r_prim/admm_r_dual; telemetry stays on.

## `enable_limit_al(rho)`

Bind the limit row-groups to the PHR augmented-Lagrangian mechanism:
hinge-activated grad/GN-Hessian and C¹ AL value folded into the KKT
cost and merit, with the outer dual update
``lam <- max(0, lam + rho*violation)`` run ONCE per solve on the final
trajectory (equality rows ``lo == hi`` always active) — warm-started
repeat solves are the outer loop, so feasibility converges across MPC
steps. The update is gated on TRUE-violation acceptance (feasible or
strictly improved), so a stalled primal freezes the duals instead of
drifting. ``rho`` is fixed per enable; duals persist across solves
(reset_dual() zeroes them). Violation is honestly telemetry-reported;
``get_row_duals()`` exposes the multipliers. While active, solves use
the direct (bdsv) linear solver and freeze trust-region rho
adaptation — both required for outer convergence (measured; see
bsqp.cuh dispatch comments). REQUIRES the trust-region floor
(constructor rho > 0, the default): f32 bdsv on an unregularized
Schur system returns garbage steps (R1). R1 default rho=1.0 — the
fold lands rho on ACTIVE rows whose natural Hessian scale is tiny
(qd rows ~1e-4): rho >= 10 makes the f32 factor error large enough
that closed-loop MPC destabilizes on tight-limit plants (measured:
iiwa14 pickplace spins at 100 rad/s at rho=100, final 5mm at
rho=1). Higher rho = tighter transients — raise it only within the
f32 ceiling (rho ~ 1e4 x the block's natural Hessian scale).

## `enable_ee_terminal_equality(target, rho)`

Append an EE terminal-position equality row-group: the returned
trajectory's final-knot EE position is constrained to ``target`` (xyz,
``lo == hi``). The first non-selection row kind — evaluated by
on-device FK, in the SOLVER's EE frame (``ee_pos(q, frame="solver")``
— the same frame the tracking cost optimizes; see ee_pos for the
frame-offset caveat). Mechanism follows the current mode: AL when
enable_limit_al() is active (always-active equality, signed
multiplier in lam_hi), ADMM when enable_limit_admm() is active
(linearized inner-loop projection: z pins to target, y accumulates
the equality multiplier), telemetry-only reporting otherwise. Call
AFTER enable_limit_* — mechanism enables reinstall the canonical
groups and drop appended ones. R1 binding ruling: ADMM binding measured
best for closed-loop MPC (2mm finals at rho=10); AL binding works at
SOFT rho (al rho=1, ee rho=1: ~5mm finals) — at rho=100 the equality
multiplier winds up through the f32 factor error and diverges.

## `set_row_group_bounds(g, lo, hi)`

Override group ``g``'s interval bounds (arrays of n_rows each).
``lo == hi`` rows become always-active equalities under AL. ADMM's
auxiliary state reinitializes on the next solve (re-clip).

## `set_row_group_soft(g, sigma)`

Soft/slack toggle (TurboMPC delta_xi) for group ``g``: sigma > 0
makes its rows ELASTIC — transient violation is traded against the
elastic weight instead of forced to zero. AL: L1 slack — the
effective multiplier saturates at sigma (the outer update caps
|lam| <= sigma; the principled lambda-cap for conflict regimes).
ADMM: quadratic slack — smoothed z-projection (slope
rho/(rho+sigma) past a bound; sigma -> inf recovers the hard clamp).
sigma = 0 restores the exact hard path. Telemetry always reports
the TRUE violation, slack notwithstanding.

## `set_admm_merit(on)`

R1 ablation toggle: include the AL-form ADMM constraint value
y'(g - z) + (rho/2)|g - z|^2 (current row state) in the line-search
merit. v1 ADMM's merit is tracking-only, so the line search rejects
steps that trade tracking for feasibility (measured: closed-loop MPC
parks in conservative basins). Off by default — the exact v1
semantics; only read while ADMM mode is active.

⚠ R2 measured (2026-07-30): ON + a CONFLICTED cone group DIVERGES
(NaN merit, violation blowup to 1e4..1e14 on 3/4 press-family cells);
on feasible cells it merely matches OFF. Keep OFF unless the
constraint set is known feasible-along-the-path.

## `set_admm_rho_adaptation(on)`

OSQP-style ADMM rho adaptation (opt-in; default OFF = bitwise
pre-adaptation path). One per-solve SCALAR multiplier on top of every
ADMM group's rho baseline (the bound per-group ratios — cone u-block
0.01 vs collision Q-block 1.0+ — are preserved), updated once per SQP
iteration from the inner loop's final residuals: adapt when
r_prim/r_dual is imbalanced by >5x, step by sqrt(ratio), clamp to
[1e-2, 1e2] (OSQP's rule). The dual form is unscaled, so rho changes
need no y-rescaling; the rho*G'G fold refreshes each SQP iteration.
The scale persists across solves (warm rho, like the (z, y) dual warm
start); toggling resets it to 1. Telemetry: get_admm_rho_scale().

MEASURED (2026-08-01 recovery cells, both plants): a clear WIN on
collision/Q-block rows — pillars at the bound cc_rho=1.0 tightens
iiwa14 cc_viol_max 0.021 -> 0.0031 (beats even the static 5.0
binding's 0.0099) and halves indy7's mean violation at flat tracking,
inner residual ~5x tighter. Do NOT enable on CONFLICTED cone cells:
an irreducible primal residual reads as "under-penalized", the rule
adapts UP away from the sharp 0.01 u-block pocket, and the
fixed-budget loop destabilizes (press_mild cone@1.0 got worse, not
recovered). Rule of thumb: adapt where the imbalance is a SCALE
problem (state-block rows), keep the bound static rho where the
constraint fights the task.

## `add_lin_u_rows(C, d, lo, hi, mech, rho, delta, sigma, cone, knot_lo, knot_hi, admm_iters, equilibrate, normalize)`

Append a LIN_U row-group: m rows ``g = C @ u + d`` on the control
block (C shape (m, nu), FROZEN at a host-chosen configuration — the
cross-term audit's contact-frame rule for config-dependent maps).

``cone=True`` binds SECOND-ORDER-CONE semantics to the row vector
(row 0 = axis t, rows 1.. = x-bar; feasible iff ||x-bar|| <= t;
lo/hi unused): ADMM z-update = SOC projection (``admm_soc``), AL =
conic PHR (dual vector projected onto K each outer update; hard-only),
barrier = relaxed barrier on the margin t - ||x-bar||. ``cone=False``
keeps interval semantics on the mapped rows (lo/hi required) — the
pyramid-facet path.

``mech`` is "telemetry" | "barrier" | "admm" | "al" (None = follow the
active enable_limit_* mode). Mixing mechanisms across groups composes
(e.g. AL boxes + ADMM cone). Call AFTER enable_limit_* — mechanism
enables reinstall the canonical groups and drop appended ones.
``rho`` defaults per mechanism, BOUND by the R2 round (2026-07-30,
docs/open-tasks/r2_report_2026-07-30.md): admm 0.01 (sharp optimum on
the feasible cone cell — 0.002 and 0.05 both park the closed loop),
al 1.0 (enforces in the stationary/hard regime; NO al rho tracks AND
enforces on transient cells — prefer admm there), barrier 3e-3 (soft
fallback; 1e-2 parks). The rho-scale law applies: the fold lands
rho * C^T C on the R block, so scale rho DOWN by ||C||^2 when the map
is large. Telemetry reports the cone margin violation
max(0, ||x-bar|| - t) (interval rows: interval violation).

``equilibrate=True`` (interval rows only) rescales each row of
(C, d, lo, hi) by 1/||C_i||_2 at enable time — an exact
reformulation (same feasible set) that puts the group's rho on
unit-norm rows (the TinyMPC-style normalization; the manual
rho-scale-law correction becomes automatic). Rejected for cone
rows: an SOC couples its rows, so per-row scaling would change
the cone — cone rows are instead normalized as a WHOLE map (below).

``normalize=True`` (cone rows only): scale the whole (C, d) uniformly
by 1/||C||_2 before install. An SOC is invariant under uniform positive
scaling, so the feasible set is untouched — but the admm/al fold lands
rho * C^T C on the R block, so a large map (e.g. pinv(J^T) with
||C|| ~ 1/sigma_min(J) ~ 6) silently over-regularizes the controls at
the bound-default rho (measured 2026-08-09: 2x tracking loss on the
wipe task's frozen-pinv cone). The R2 rho defaults are for unit-norm
maps; this makes that the installed contract. get_row_groups() returns
the NORMALIZED map. Pass normalize=False to install verbatim.

## `add_fc_box(lo, hi, slots)`

Box rows on contact-force slots (GATO_CONTACT_FORCES builds only):
selection LIN_U rows on control columns n_actuated+slots. ``slots``
indexes into the fc block (default: all n_fc slots); lo/hi broadcast.
Pin the wrench torque rows of a point contact with
``add_fc_box(0, 0, slots=range(3))`` (wrench layout is [n; f]).
Extra kwargs go to add_lin_u_rows (mech/rho/knot range/...).

## `enable_u_cone(C, d, mech, rho, form, facets, facet_scale)`

Cone constraint on a mapped control quantity g = C @ u + d
(CL-2 demo surface: e.g. an EE contact-force friction cone with
C = S @ pinv(J(q).T), rows [mu*f_n; f_t1; f_t2], frozen at q).

form="soc": exact second-order cone via add_lin_u_rows(cone=True).
form="pyramid": m must be 3; the cone is replaced by ``facets``
one-sided linear rows h_j = cos(th_j) g1 + sin(th_j) g2 - s*g0 <= 0
riding the ordinary interval machinery (any mechanism, slack toggle
included). facet_scale="inscribed" (s = cos(pi/facets), conservative:
facet-feasible => cone-feasible) or "circumscribed" (s = 1, outer
approximation). Returns the appended group index.

## `set_collision_environment(spheres, capsules, cuboids, planes)`

Upload the runtime obstacle set for the COLLISION clearance rows
(CL-2). Lists of tuples, one per obstacle (all in world frame, meters):

- spheres: (x, y, z, r)
- capsules: (ax, ay, az, bx, by, bz, r) — segment endpoints + radius
- cuboids: (cx, cy, cz, ux, uy, uz, hu, vx, vy, vz, hv, wx, wy, wz, hw)
  — oriented box: center, then 3 (unit axis, half-extent) pairs
- planes: (nx, ny, nz, d) — half-space n·p >= d, n a UNIT normal
  pointing into FREE space (ground floor at z0: (0, 0, 1, z0))

Deep-copied to the device; callable between solves. An empty
environment makes every clearance +1e30 (rows inert).

## `enable_collision(mech, margin, rho, delta, sigma, knot_lo, admm_iters)`

Append THE collision clearance group (one max): per-sphere rows
d_i(q_k) >= margin over knots [knot_lo, N] — d_i = signed clearance of
collision sphere i (baked at codegen, ``collision_res``) to the
nearest obstacle from set_collision_environment (call that FIRST).
The covering spheres are already conservative (inflated by the
spherizer), so margin is extra safety on top.

``mech``: "admm" (linearized one-sided intervals in the inner loop —
the transient/MPC recommendation, mirroring the R2 cone verdict; AL
under-enforces on transient avoidance even at rho 5), "al" (PHR
hinge; +L1 elastic via sigma>0), "barrier" (soft), or "telemetry"
(report-only). None follows the active enable_limit_* mode.

rho defaults: admm 1.0 — BOUND by the 2b pillars round (2026-07-30):
clearance rows fold onto the Q block (natural scale O(q_cost)), so
the admm rho pocket is WIDE AND FLAT, unlike the cone's sharp
u-block pocket at 0.01 — violation drops monotonically over rho
0.01..5.0 at flat tracking cost (indy7 2.7mm / iiwa14 21mm sphere-
margin violation at 1.0; 5.0 strictly clears both). Raise toward 5
for strict clearance; the rho-scale law applies per target block.
al 1.0 / barrier 3e-3 as elsewhere.
knot_lo >= 1 always: x_0 is data — a start pose in collision would
make knot-0 rows unsatisfiable (the R1 windup lesson).

Telemetry group slot: the clearance group's {max, sum} true violation
rides get_row_telemetry() at this group's index (see get_row_groups).
Call AFTER enable_limit_* (mechanism enables reinstall the canonical
groups, dropping appended ones).

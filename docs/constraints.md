# GATO constraint layer — mechanisms, measured defaults, provenance

Reference for the row-group constraint API on `gato.BSQP`.
The code docstrings carry the CONTRACT (what each call does); this page keeps
the measured tuning rulings, R1/R2 evaluation provenance and traps that used
to live inline; the docstrings carry a one-line summary and point here.

Ordering rule (enforced, raises): `enable_limit_*` FIRST, then appended groups
(`add_lin_u_rows` / `enable_u_cone` / `add_fc_box` / `add_fc_cone` / `add_contact_pos_rows` / `enable_ee_terminal_equality`
/ `enable_collision`) — a mechanism enable reinstalls the canonical limit groups.

Mechanisms: `telemetry` (report only) · `barrier` (relaxed log barrier, soft)
· `admm` (OSQP-style projection inner loop with PCG or reused BDSV factor,
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

Per-step policy (October 15, 2026): `MPCController(admm_linsys="auto",
admm_threshold=tau)` switches the arm per tick from the warm-startedness
signal that drives `linsys="auto"`: a cold tick (pred_err above the
threshold, a PCG cap-out on the previous tick, or a re-seed) runs
"bdsv_factor", a warm tick "pcg"; `StepResult.admm_linsys` reports the arm.
Resolution: an explicit argument, else a tuned entry's `admm_policy` /
`admm_tau` (`tools/autotune_linsys.py --admm`, a quiet-window probe), else
auto at the SQP `bdsv_threshold`. Counts behind the crossover (Indy7 N32
figure-8 with kicks, residual exit at eps_rel 1e-2): warm ticks cost 2.3
inner iterations of 13 PCG steps each per SQP iteration, kicked ticks 7.7 of
37; the factor arm costs one factorization plus 2.5 (warm) to 8.5 (kicked)
back-solves. Measured on a quiet box (October 16, 2026; the table under
`set_admm_exit`): at N32 the PCG arm is the faster inner solver on quiet and
kicked ticks alike (0.43 against 0.78 ms per solve on the Indy7, quiet ticks)
and the fitted threshold is 0.027 (factor arm on 10 % of ticks); at N64 the
factor arm wins on every statistic (1.6 to 2.3 ms mean against 2.1 to 8.2 ms,
3.3 ms worst case against 12.7 to 33 ms) and the fit returns the factor arm for
every tick. The arm follows the horizon more than the tick: auto at the SQP
threshold 0.1 ran the factor arm on 0.2 % of quiet ticks and 6 % of kicked
ticks and timed as the PCG arm. The tool fits the threshold from the pcg arm's
cost-vs-pred_err curve against the factor arm's flat cost, exactly as the SQP
policy was fitted; W2b sets the controller default from these numbers.

## `set_admm_exit(eps_abs, eps_rel, check_every)`

Residual early exit of the ADMM inner loop (added October 15, 2026, the H2 ADMM
speed study). Off by default (`eps_abs = eps_rel = 0`): the inner loop runs the
fixed budget of `enable_limit_admm(iters)` bit for bit. With a tolerance the
budget is a cap and the loop stops, per batch entry, after the first inner
iteration whose iterate passes, for every ADMM row group g,

    r_prim_g <= eps_abs + eps_rel * max|z_g|
    r_dual_g <= eps_abs + eps_rel * rho_g * max|z_g|

(OSQP's absolute-plus-relative form with the auxiliary variable as the scale;
per group because the rows of one group share units and a global scale would
let the torque rows excuse the velocity rows) and whose x-update converged:
PCG below its iteration cap on the "pcg" arm, an OK factor on "bdsv_factor".
The second condition is the inexact-ADMM guard: `r_dual = rho * max|z - z_prev|`
is the dual residual only for an exact x-update, so an unconverged inner solve
is never allowed to read as a fixed point. The test runs on the device
(a per-entry skip mask seeded from the SQP convergence mask each iteration), so
batch entries exit at their own iteration without a host sync; the host breaks
the launch loop every `check_every` inner iterations once every entry has
exited (0 = never; exited entries are still skipped on the device, so only
empty launches remain). `stats.admm_iters` / `stats.admm_linsys_iters` report
the inner iterations run and the inner linear-system work (PCG steps, or one
per back-solve). The iterates do not depend on `check_every`
(`test_admm_exit.py`).

Measured (October 15, 2026, counts only; Indy7 N32 figure-8, velocity box
±1 rad/s, rho 0.01, cap 10, 5 SQP iterations per tick, fixed pacing):

| inner linsys | eps_rel | inner iterations per SQP iteration (mean / p90) | plan violation max | simulated peak |
|---|---|---|---|---|
| pcg | off (fixed 10) | 10 / 10 | 3.5e-2 rad/s | 1.034 rad/s |
| pcg | 1e-3 | 7.0 / 10 | 3.5e-2 | 1.034 |
| pcg | 1e-2 | 1.9 / 2.6 | 2.4e-2 | 1.027 |
| bdsv_factor | 1e-3 | 3.6 / 9.2 | 5.4e-3 | 1.005 |

At N64 (example 09, 3 s, cap 10 x 5 SQP iterations = 50 per tick) the exit at
1e-2 with `admm_linsys="auto"` spends 10.9 inner iterations per tick against
the fixed 50, at the same simulated peak (1.04 rad/s), a planner violation of
3.4e-2 against 3.9e-2 and a tracking error of 0.035 m against 0.030 m.
At 1e-2 the warm tick spends two inner iterations and the enforcement is
unchanged; 1e-3 is below what the inner loop reaches within the cap on the PCG
arm (the fixed point is approached to about 1e-2 relative, and the PCG
inexactness keeps `r_dual` above 1e-3 of the scale). The factor arm reaches a
tighter iterate and exits sooner at the same tolerance; its returned violation
is 6x smaller, as the August 1 A/B found. With disturbances (q kicked by
N(0, 0.1 rad) every 25 ticks) the PCG arm averages 3.1 inner iterations per
SQP iteration with 17 PCG steps each, rising to 7.7 and 37 on the kicked
ticks; the factor arm averages 2.5 with one back-solve each. On the solver-level
reach problem (iiwa14 N8, rho 10, `test_admm_exit.py`) a warm repeat exits
after 1-1.6 inner iterations per SQP iteration at 1e-2 with zero violation.
The inner loop's fixed point is approached only to ~1e-2 relative within 10
iterations at these cost scales (rho 0.01 against a 1e-6 control weight is a
proximal step, not a Newton step): tighter tolerances buy budget-length loops,
not better enforcement.

Wall time (October 16, 2026, 02:28 local, RTX 5090, quiet box, commit 880a2d7,
`examples/benchmarks/run_timing_w2a.sh`, rows in
`examples/benchmarks/data/admm_time_matrix/results.jsonl`): closed-loop
figure-8 with the velocity box tightened to ±1 rad/s, rho 0.01, cap 10, one
SQP iteration per tick (the MPC default), fixed pacing; "kicked" adds a
measured-state kick every 25 ticks. Per-solve wall p50 / mean / max in ms:

| case (figure-8, 600 ticks) | free | fixed 10, pcg | fixed 10, factor | exit 1e-2, pcg | exit 1e-2, factor |
|---|---|---|---|---|---|
| Indy7 N32 | 0.17 / 0.17 / 0.26 | 0.59 / 0.72 / 2.6 | 1.72 / 1.72 / 1.7 | 0.43 / 0.52 / 2.7 | 0.78 / 0.79 / 1.8 |
| Indy7 N32, kicked | 0.17 / 0.18 / 0.30 | 0.95 / 1.11 / 4.3 | 1.72 / 1.72 / 1.8 | 0.57 / 0.89 / 4.5 | 0.78 / 1.00 / 1.8 |
| Indy7 N64 | 0.31 / 0.31 / 0.80 | 3.21 / 3.81 / 12.7 | 3.19 / 3.19 / 3.4 | 0.97 / 2.08 / 12.7 | 1.66 / 1.57 / 3.3 |
| Indy7 N64, kicked | 0.32 / 0.36 / 0.91 | 5.53 / 6.59 / 23.0 | 3.20 / 3.20 / 3.2 | 4.26 / 5.53 / 23.1 | 1.93 / 2.00 / 3.3 |
| iiwa14 N32 | 0.17 / 0.18 / 0.28 | 0.65 / 0.77 / 5.3 | 1.96 / 1.96 / 2.1 | 0.33 / 0.47 / 5.4 | 0.89 / 0.88 / 2.1 |
| iiwa14 N32, kicked | 0.18 / 0.20 / 0.41 | 1.04 / 1.37 / 6.1 | 1.96 / 1.96 / 2.3 | 0.70 / 1.12 / 6.2 | 1.07 / 1.28 / 2.1 |
| iiwa14 N64 | 0.32 / 0.33 / 0.72 | 4.82 / 6.11 / 30.1 | 3.74 / 3.74 / 3.9 | 1.40 / 3.43 / 30.3 | 1.94 / 1.91 / 3.9 |
| iiwa14 N64, kicked | 0.34 / 0.40 / 1.29 | 9.46 / 9.84 / 34.2 | 3.74 / 3.74 / 3.9 | 7.23 / 8.18 / 33.3 | 1.94 / 2.32 / 3.9 |

The exit runs 2.6 to 3.1 inner iterations per tick on quiet ticks and 4.5 to
5.7 on kicked ticks on both arms, 1.4 to 2.4 times faster than the fixed budget
on the same arm, at unchanged row violation and a tracking cost (Indy7 N64:
0.028 m fixed against 0.037 to 0.040 m with the exit; N32: 0.027 against 0.030).
The arm choice follows the horizon. At N32 the PCG arm costs 2.5 times the
unconstrained step (median) with 15 to 22 PCG steps per inner iteration; at N64
PCG needs 52 to 80 steps per inner iteration and caps out on a tenth of the
ticks, so the factor arm wins on mean and worst case (5 to 6 times the
unconstrained step, 3.3 ms worst case against 12.7 ms). The factor arm's inner
iteration costs 0.15 ms at N32 and 0.29 to 0.34 ms at N64 regardless of kicks:
the block-Cholesky back-solve sweeps the horizon serially, so one inner
iteration costs as much as the whole unconstrained step. The 16 ms figure of
October 12 was this fixed budget over five SQP iterations per tick.

## `set_exact_hessian(on)`

Toggle the SO-SQP stage-Hessian PSD projection for subsequent solves.

Per-TASK feature: wins on EE-terminal tasks, neutral-to-worse on
full-rank joint-terminal ones (numpy prototype study).
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

Controller default since October 16, 2026: `MPCController(limits=...)` installs
these rows itself on a fixed-base arm whose solver has none (`limits=None`),
with per-box rho 0.1 (position rows), 0.03 (velocity rows) and 1e-5 (torque
rows), cap 10, the residual exit at eps_rel 1e-2 and the inner linear
solver by horizon (`gato.linsys_autotune.ADMM_FACTOR_MIN_N` = 64: warm PCG
below, the reused factorization from there; see `set_admm_linsys`). A solver
the caller configured stands, `limits="off"` installs nothing, and the raw
`BSQP` never installs rows on its own. Example 09 runs the default against
the fixed budget and telemetry.

Bind the limit row-groups to the ADMM-projection mechanism: an
OSQP-style inner loop per SQP iteration (warm PCG by default;
optionally a reused direct BDSV factorization) — the constraint layer's
"approximately hard" mode. ``rho`` is the ADMM penalty (fixed within
a solve; adapt it between solves), ``iters`` the inner budget: fixed by
default, a cap once the residual exit is on (`set_admm_exit`, or the
``eps_abs`` / ``eps_rel`` keywords here).
R1 default rho=0.01: the penalty must ride the COST-HESSIAN scale —
rho >= 1 swamps the u-block (natural scale u_cost=1e-6), freezing
controls at the warm start (closed-loop MPC parks); the measured
pocket is ~0.005-0.02.
That pocket was measured on a binding velocity box with one rho for the
three boxes. On tasks where no limit binds, one rho still costs tracking:
rho 0.01 on the torque rows against the 2e-6 control cost is a 5000x
control regularizer that the ten-iteration inner loop never unwinds
(October 16, 2026: applied torque peaks at 0.2 of the limit under ADMM
against 0.6 under the other mechanisms, and a feasible figure-8 tracks
25 to 65 % worse). Per-box weights (`enable_limit_admm(rho_q=, rho_qd=,
rho_u=)`, `set_row_group_rho(g, rho)`) put each fold at its block's
scale; the controller default is 0.1 / 0.03 / 1e-5 for the position /
velocity / torque rows, set on the dev set below.

Dev set (October 16, 2026, `examples/benchmarks/limits_holdout.py --set dev`,
seed 1, 20 seeded scenarios per task and arm, N32 and N64, fixed pacing,
pinocchio plant, MPC default of one SQP iteration per tick; rows in
`examples/benchmarks/data/limits_holdout/dev.jsonl`). Tasks from
`gato.tasks`: figure-8s whose period makes the demanded joint speed 0.6 to
1.6 times the velocity limit (the arm lags rather than exceeds, so nothing
should bind), and three-goal pick-and-place with step or ramped references
(the step reference asks for maximal acceleration and the velocity box binds).
The simulated plant is graded against the URDF limits; "ticks over" is the
fraction of control periods with a joint above its velocity limit. Twenty
configurations were run (uniform ADMM rho 1e-3 to 3e-2, the fixed budget,
eps_rel 1e-3, barrier mu 1e-2, AL rho 10, three SQP iterations per tick,
per-box rho variants); the five that decide the default:

| task | arm, N | mechanism | sim peak qd / limit p50 / p90 / max | ticks over | tracking p50 (m) or goals, task time p50 (s) | planner viol qd max |
|---|---|---|---|---|---|---|
| fig8 | indy7 N32 | none | 0.81 / 0.95 / 1.02 | 0.0 % | 0.0618 | 2.0e-02 |
| fig8 | indy7 N32 | barrier | 0.76 / 0.95 / 1.00 | 0.0 % | 0.0649 | 0.0e+00 |
| fig8 | indy7 N32 | al | 0.70 / 0.84 / 0.86 | 0.0 % | 0.0677 | 0.0e+00 |
| fig8 | indy7 N32 | admm, one rho 0.01 | 0.83 / 0.99 / 0.99 | 0.0 % | 0.0764 | 1.1e-01 |
| fig8 | indy7 N32 | admm, per-box rho (default) | 0.71 / 0.84 / 0.87 | 0.0 % | 0.0647 | 0.0e+00 |
| fig8 | indy7 N64 | none | 0.74 / 0.85 / 0.90 | 0.0 % | 0.0709 | 0.0e+00 |
| fig8 | indy7 N64 | barrier | 0.69 / 0.72 / 0.85 | 0.0 % | 0.0756 | 0.0e+00 |
| fig8 | indy7 N64 | al | 0.67 / 0.77 / 0.83 | 0.0 % | 0.0718 | 0.0e+00 |
| fig8 | indy7 N64 | admm, one rho 0.01 | 0.64 / 0.76 / 0.78 | 0.0 % | 0.1054 | 4.6e-02 |
| fig8 | indy7 N64 | admm, per-box rho (default) | 0.67 / 0.77 / 0.83 | 0.0 % | 0.0721 | 0.0e+00 |
| fig8 | iiwa14 N32 | none | 0.78 / 0.96 / 1.04 | 0.1 % | 0.0378 | 1.4e-01 |
| fig8 | iiwa14 N32 | barrier | 0.62 / 0.72 / 0.79 | 0.0 % | 0.0454 | 0.0e+00 |
| fig8 | iiwa14 N32 | al | 0.78 / 0.95 / 1.03 | 0.0 % | 0.0377 | 1.3e-01 |
| fig8 | iiwa14 N32 | admm, one rho 0.01 | 0.93 / 1.01 / 1.02 | 0.1 % | 0.0488 | 1.7e+00 |
| fig8 | iiwa14 N32 | admm, per-box rho (default) | 0.79 / 0.92 / 0.93 | 0.0 % | 0.0377 | 7.0e-03 |
| fig8 | iiwa14 N64 | none | 0.73 / 0.96 / 1.05 | 0.1 % | 0.0438 | 1.7e-01 |
| fig8 | iiwa14 N64 | barrier | 0.60 / 0.74 / 0.79 | 0.0 % | 0.0519 | 0.0e+00 |
| fig8 | iiwa14 N64 | al | 0.73 / 0.92 / 0.93 | 0.0 % | 0.0439 | 1.7e-01 |
| fig8 | iiwa14 N64 | admm, one rho 0.01 | 1.00 / 1.09 / 1.11 | 0.3 % | 0.0716 | 5.6e-01 |
| fig8 | iiwa14 N64 | admm, per-box rho (default) | 0.75 / 0.92 / 0.93 | 0.0 % | 0.0440 | 8.2e-03 |
| pickplace | indy7 N32 | none | 0.49 / 0.95 / 1.98 | 0.6 % | 3.00 of 3, 3.07 s | 2.7e+00 |
| pickplace | indy7 N32 | barrier | 0.48 / 0.80 / 1.29 | 0.1 % | 3.00 of 3, 3.10 s | 1.1e-01 |
| pickplace | indy7 N32 | al | 0.47 / 0.83 / 1.42 | 0.1 % | 3.00 of 3, 3.07 s | 1.3e+00 |
| pickplace | indy7 N32 | admm, one rho 0.01 | 0.47 / 0.97 / 1.10 | 0.3 % | 3.00 of 3, 3.14 s | 5.4e-01 |
| pickplace | indy7 N32 | admm, per-box rho (default) | 0.47 / 0.82 / 1.02 | 0.0 % | 3.00 of 3, 3.08 s | 2.8e-02 |
| pickplace | indy7 N64 | none | 0.47 / 0.86 / 1.97 | 0.4 % | 3.00 of 3, 3.07 s | 2.6e+00 |
| pickplace | indy7 N64 | barrier | 0.44 / 0.81 / 1.26 | 0.1 % | 3.00 of 3, 3.08 s | 1.5e-01 |
| pickplace | indy7 N64 | al | 0.45 / 0.84 / 1.51 | 0.2 % | 3.00 of 3, 3.06 s | 1.7e+00 |
| pickplace | indy7 N64 | admm, one rho 0.01 | 0.49 / 0.97 / 1.09 | 0.3 % | 3.00 of 3, 3.26 s | 1.0e+00 |
| pickplace | indy7 N64 | admm, per-box rho (default) | 0.45 / 0.84 / 1.01 | 0.0 % | 3.00 of 3, 3.06 s | 2.4e-02 |
| pickplace | iiwa14 N32 | none | 1.59 / 3.00 / 3.54 | 19.2 % | 2.95 of 3, 1.70 s | 3.8e+00 |
| pickplace | iiwa14 N32 | barrier | 0.91 / 1.08 / 1.18 | 0.1 % | 2.75 of 3, 2.20 s | 7.4e-01 |
| pickplace | iiwa14 N32 | al | 1.03 / 1.35 / 1.50 | 3.7 % | 2.95 of 3, 2.08 s | 2.0e+00 |
| pickplace | iiwa14 N32 | admm, one rho 0.01 | 1.11 / 1.42 / 1.47 | 5.0 % | 2.95 of 3, 2.97 s | 1.5e+00 |
| pickplace | iiwa14 N32 | admm, per-box rho (default) | 0.96 / 1.02 / 1.05 | 0.1 % | 2.95 of 3, 2.24 s | 1.6e-01 |
| pickplace | iiwa14 N64 | none | 1.59 / 2.88 / 3.13 | 15.4 % | 2.65 of 3, 1.72 s | 3.3e+00 |
| pickplace | iiwa14 N64 | barrier | 0.91 / 1.07 / 1.13 | 0.1 % | 2.45 of 3, 1.92 s | 9.5e-01 |
| pickplace | iiwa14 N64 | al | 1.00 / 1.20 / 1.43 | 3.3 % | 2.60 of 3, 1.99 s | 1.9e+00 |
| pickplace | iiwa14 N64 | admm, one rho 0.01 | 1.11 / 1.35 / 1.44 | 4.5 % | 2.85 of 3, 3.04 s | 8.5e-01 |
| pickplace | iiwa14 N64 | admm, per-box rho (default) | 0.95 / 1.00 / 1.01 | 0.1 % | 2.70 of 3, 2.01 s | 4.5e-02 |

Reading. Where nothing binds (figure-8), the per-box ADMM tracks like the
unconstrained controller (within 2 % at N64 and 5 % at N32 on the Indy7,
equal on the iiwa14) with a planner violation below 1e-2 rad/s, where one rho
costs 25 to 65 % and on the iiwa14 lets the plan violate its own box by 0.5
to 1.7 rad/s; the barrier costs 5 to 20 % as a prior that always acts; AL
costs nothing and enforces nothing it does not have to. Where the box binds
(pick-and-place, step reference), the per-box ADMM holds the simulated peak
to 1.01 to 1.05 of the limit with at most 0.1 % of ticks over, against 1.98
to 3.54 unconstrained, 1.13 to 1.29 for the barrier and 1.42 to 1.51 for AL,
at the unconstrained controller's goals and task time on the Indy7 and 0.3 s
behind it on the iiwa14 (whose unconstrained runs reach goals faster by
running at 1.6 to 3.5 times its velocity limit). Among the per-box variants,
rho_qd 0.03 against 0.01 tightens the peak from 1.02 to 1.10 down to 0.98 to
1.02 at identical tracking; rho_u 1e-4 brings the control bias back (the
torque rows regularize again), 2e-6 equals 1e-5. Three SQP iterations per
tick make AL enforce (1.0 to 1.05) and make the unconstrained arm more
aggressive; they do not change the ADMM verdict. The residual exit costs
nothing in enforcement against the fixed budget on this set. The default is
therefore rho 0.1 / 0.03 / 1e-5 with the exit at 1e-2; the holdout below is a
fresh draw run once with it.

Holdout (October 16, 2026, `--set holdout`, seed 20261016, 100 fresh scenarios
per task and arm, N32 and N64, the four mechanisms at the frozen defaults, no
retuning; 3200 episodes in `examples/benchmarks/data/limits_holdout/holdout.jsonl`,
published as run):

| task | arm, N | mechanism | sim peak qd / limit p50 / p90 / max | ticks over | sim peak u / limit max | tracking p50 / p90 (m) or goals, task time p50 (s) | planner viol qd p50 / max |
|---|---|---|---|---|---|---|---|
| fig8 | indy7 N32 | none | 0.79 / 1.02 / 1.18 | 0.1 % | 0.46 | 0.0661 / 0.0951 | 0.0e+00 / 1.6e-01 |
| fig8 | indy7 N32 | barrier | 0.72 / 0.89 / 1.04 | 0.0 % | 0.45 | 0.0704 / 0.1008 | 0.0e+00 / 2.3e-02 |
| fig8 | indy7 N32 | al | 0.66 / 0.88 / 0.96 | 0.0 % | 0.32 | 0.0731 / 0.1021 | 0.0e+00 / 1.5e-02 |
| fig8 | indy7 N32 | admm (default) | 0.67 / 0.88 / 0.96 | 0.0 % | 0.33 | 0.0718 / 0.1009 | 0.0e+00 / 7.1e-03 |
| fig8 | indy7 N64 | none | 0.69 / 0.88 / 1.09 | 0.0 % | 0.43 | 0.0762 / 0.1027 | 0.0e+00 / 1.1e-03 |
| fig8 | indy7 N64 | barrier | 0.63 / 0.78 / 0.94 | 0.0 % | 0.42 | 0.0807 / 0.1101 | 0.0e+00 / 0.0e+00 |
| fig8 | indy7 N64 | al | 0.62 / 0.84 / 0.91 | 0.0 % | 0.30 | 0.0767 / 0.1025 | 0.0e+00 / 8.9e-03 |
| fig8 | indy7 N64 | admm (default) | 0.62 / 0.84 / 0.91 | 0.0 % | 0.30 | 0.0770 / 0.1028 | 0.0e+00 / 1.4e-04 |
| fig8 | iiwa14 N32 | none | 0.80 / 0.95 / 1.05 | 0.1 % | 0.23 | 0.0360 / 0.0475 | 0.0e+00 / 1.6e-01 |
| fig8 | iiwa14 N32 | barrier | 0.63 / 0.71 / 0.80 | 0.0 % | 0.23 | 0.0435 / 0.0548 | 0.0e+00 / 0.0e+00 |
| fig8 | iiwa14 N32 | al | 0.80 / 0.94 / 1.04 | 0.0 % | 0.23 | 0.0360 / 0.0476 | 0.0e+00 / 1.5e-01 |
| fig8 | iiwa14 N32 | admm (default) | 0.81 / 0.93 / 0.93 | 0.0 % | 0.23 | 0.0361 / 0.0476 | 0.0e+00 / 8.9e-03 |
| fig8 | iiwa14 N64 | none | 0.77 / 0.95 / 1.04 | 0.0 % | 0.36 | 0.0417 / 0.0566 | 0.0e+00 / 1.6e-01 |
| fig8 | iiwa14 N64 | barrier | 0.62 / 0.73 / 0.82 | 0.0 % | 0.34 | 0.0490 / 0.0641 | 0.0e+00 / 0.0e+00 |
| fig8 | iiwa14 N64 | al | 0.77 / 0.93 / 0.97 | 0.0 % | 0.37 | 0.0421 / 0.0569 | 0.0e+00 / 1.5e-01 |
| fig8 | iiwa14 N64 | admm (default) | 0.79 / 0.92 / 0.93 | 0.0 % | 0.36 | 0.0422 / 0.0570 | 0.0e+00 / 9.9e-03 |
| pickplace | indy7 N32 | none | 0.68 / 2.02 / 2.57 | 3.3 % | 1.16 | 3.00 of 3, 1.93 s | 0.0e+00 / 4.0e+00 |
| pickplace | indy7 N32 | barrier | 0.63 / 1.44 / 2.24 | 0.5 % | 0.80 | 3.00 of 3, 1.94 s | 0.0e+00 / 1.3e+00 |
| pickplace | indy7 N32 | al | 0.66 / 1.84 / 2.13 | 1.2 % | 0.96 | 3.00 of 3, 1.92 s | 0.0e+00 / 3.1e+00 |
| pickplace | indy7 N32 | admm (default) | 0.65 / 0.97 / 1.44 | 0.0 % | 0.79 | 3.00 of 3, 1.91 s | 0.0e+00 / 7.1e-02 |
| pickplace | indy7 N64 | none | 0.67 / 1.96 / 2.57 | 2.6 % | 1.00 | 3.00 of 3, 2.00 s | 0.0e+00 / 3.9e+00 |
| pickplace | indy7 N64 | barrier | 0.66 / 1.50 / 2.25 | 0.5 % | 0.81 | 3.00 of 3, 1.98 s | 0.0e+00 / 1.1e+00 |
| pickplace | indy7 N64 | al | 0.64 / 1.79 / 2.51 | 1.2 % | 0.95 | 3.00 of 3, 1.91 s | 0.0e+00 / 4.1e+00 |
| pickplace | indy7 N64 | admm (default) | 0.64 / 0.99 / 1.43 | 0.1 % | 0.81 | 3.00 of 3, 1.91 s | 0.0e+00 / 4.7e-02 |
| pickplace | iiwa14 N32 | none | 1.39 / 2.87 / 4.17 | 16.3 % | 1.02 | 2.93 of 3, 2.01 s | 7.0e-01 / 5.6e+00 |
| pickplace | iiwa14 N32 | barrier | 0.88 / 1.00 / 1.23 | 0.1 % | 0.98 | 2.78 of 3, 2.48 s | 0.0e+00 / 1.0e+00 |
| pickplace | iiwa14 N32 | al | 0.96 / 1.29 / 1.62 | 2.2 % | 0.91 | 2.92 of 3, 2.28 s | 3.3e-01 / 2.0e+00 |
| pickplace | iiwa14 N32 | admm (default) | 0.94 / 1.00 / 1.08 | 0.1 % | 0.99 | 2.93 of 3, 2.26 s | 1.3e-02 / 2.1e-01 |
| pickplace | iiwa14 N64 | none | 1.48 / 2.57 / 4.02 | 14.4 % | 1.11 | 2.76 of 3, 2.08 s | 8.0e-01 / 5.4e+00 |
| pickplace | iiwa14 N64 | barrier | 0.88 / 1.01 / 1.29 | 0.0 % | 0.97 | 2.56 of 3, 2.45 s | 0.0e+00 / 1.0e+00 |
| pickplace | iiwa14 N64 | al | 0.99 / 1.23 / 1.56 | 2.0 % | 0.91 | 2.73 of 3, 2.30 s | 2.6e-01 / 2.0e+00 |
| pickplace | iiwa14 N64 | admm (default) | 0.94 / 0.97 / 1.00 | 0.0 % | 0.96 | 2.73 of 3, 2.28 s | 1.2e-02 / 8.8e-02 |

Reading. On the figure-8 the default tracks within 1 % of the unconstrained
controller on both arms at N64 (0.0770 against 0.0762 m; 0.0422 against
0.0417), 9 % at N32 on the Indy7 and equal on the iiwa14, with the planner
violation of the velocity rows at most 1e-2 rad/s where the unconstrained
plan reaches 0.16. On the step-reference pick-and-place the simulated peak
velocity is at the limit at the 90th percentile (0.97 to 1.00 of the limit)
with at most 0.1 % of control periods over, against 1.96 to 2.87 at the 90th
percentile and 2.6 to 16 % of periods over unconstrained; the barrier holds
the iiwa14 (1.00 to 1.01) but not the Indy7 (1.44 to 1.50) and loses goals on
the iiwa14 (2.56 to 2.78 of 3); AL sits between (1.23 to 1.84). The worst
single scenario under the default peaks at 1.43 to 1.44 on the Indy7 (one of
100; the plan itself violates by at most 7e-2 rad/s, so the excess is the
plant overshooting a feasible plan within one control period) and at 1.00 to
1.08 on the iiwa14. Goals reached and task time match the unconstrained
controller on the Indy7 (3 of 3, 1.91 against 1.93 to 2.00 s) and come within
0.2 to 0.3 s on the iiwa14, whose unconstrained runs reach goals by running at
up to four times its velocity limit. The applied torque stays inside the
limit under the default (0.96 to 0.99 of the limit at the peak) where the
unconstrained controller exceeds it (1.00 to 1.16), and no default run leaves
the position box (the unconstrained controller leaves it by up to 0.24 rad).
The exit spends 3.7 inner iterations per tick on average. The claim this
supports: inside the URDF position, velocity and torque limits to within the
reported residual on these tasks, in closed loop on the simulated plant, with
the per-tick overshoot above bounded by one control period of plant motion;
a bounded, measured property, not a certificate (hard per-solve guarantees are
research, column 2 of the plan).
iters=10 BOUND by R2: 2/5 park the feasible
cone cell; box cells saturate by 10 (20 = marginal viol gains at 2x
inner cost).
Duals warm-start across solves (reset_dual() reinitializes) —
EXCEPT equality rows (lo == hi, e.g. enable_ee_terminal_equality),
whose (z, y) reinit every solve: a warm-started dual on a row the
primal may not reach is an unbounded violation integrator (measured).
stats gain admm_r_prim/admm_r_dual; telemetry stays on.

Closed loop (examples/09_mpc_with_limits.py, test_limits_closed_loop.py; Indy7
figure-8, velocity box tightened to ±1 rad/s, rho 0.01, 10 iterations, 5 SQP
steps per tick, fixed pacing, measured 2026-10-12): the planner's residual
violation is 0.035–0.039 rad/s at N32/N64 and the SIMULATED joint velocity
peaks at 1.03–1.04 rad/s, versus 1.6–2.0 rad/s with telemetry only; tracking
error rises from 0.024 to 0.030 m (N64). rho 0.1 tightens the plan (0.015) but
costs tracking (0.045 m); horizons of 16 knots or fewer (≤160 ms lookahead)
cannot brake in time (1.10 rad/s at N16, 1.19 at N8), so the closed-loop gate
runs on N ≥ 32. On a quiet box (October 16, 2026, RTX 5090, one SQP iteration
per tick) the fixed-budget ADMM tick at N64 is 3.2 ms on the factor arm and
3.2 to 6.6 ms mean on the PCG arm (12 to 23 ms worst case) against 0.31 ms
unconstrained; the residual exit (`set_admm_exit`) brings the factor arm to
1.6 to 2.0 ms (3.3 ms worst case), five to six times the unconstrained step,
and costs about 0.01 m of tracking (table and reading in that section). The
16 ms of October 12 was the fixed budget over five SQP iterations per tick on
a shared box.

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
measured): admm 0.01 (sharp optimum on
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
``add_fc_box(0, 0, slots=range(3))`` (wrench layout is [n; f]); on
multi-contact plants address a frame's block with
``solver.fc_slots(frame, "n"|"f")`` (frame = index or baked name from
``solver.contact_frames``; go2 bakes the four ``*_foot_joint`` frames).
Extra kwargs go to add_lin_u_rows (mech/rho/knot range/...). ⚠ Like every
appended row group the default mechanism is TELEMETRY (unless an
``enable_limit_*`` set one) — pass ``mech="al"`` (or "admm") for an
enforced pin; the standing gates pin the moment rows with AL.

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
rides `stats.row_{max,sum}_violation` at the group's index (get_row_groups() order) at this group's index (see get_row_groups).
Call AFTER enable_limit_* (mechanism enables reinstall the canonical
groups, dropping appended ones).

## `add_contact_pos_rows(targets, lo, hi, mech, rho, delta, sigma, knot_lo, knot_hi, admm_iters)`

CONTACT_POS row-group (CL-4 §1.3, 2026-09-26): for every baked contact frame f
(`contact_frames`: the go2 feet, an arm's EE) and every knot k of
`[knot_lo, knot_hi)` three rows `g = p_f(q_k) - tgt_k[f]` — the world position
of the contact ORIGIN (the point the fc wrench acts about) from on-device FK
over GRiD's `contact_frame_positions[_gradient]` surface, with the interval
`lo <= g <= hi` on the RESIDUAL. Defaults `lo = hi = 0` pin the frames to their
targets (an equality per knot); `lo = clearance, hi = inf` on a z row (the
others masked off) is a swing clearance. `targets` is `(N, n_rows)`,
`(n_frames, 3)` or `(n_rows,)` (broadcast over knots; None = zeros = the raw
position). Unlike the terminal-only EE row, these fold at EVERY active knot
(the dense J^T J Gauss-Newton term on the tangent q block, like the collision
rows; ADMM linearizes per knot in its inner loop). The per-knot target table
(`get_row_groups()[g]["tgt"]`, `set_row_group_targets`) lets one group hold a
different point per knot: a foothold frozen at touchdown, a swing curve.
Mask (knot, foot) rows with `set_row_group_mask` — knot 0 is the measured state,
so leave it masked off unless the target IS the measured foot position (the
AL multiplier would otherwise wind up on rows no step can change, the
limit-box lesson). Mechanism defaults follow `enable_collision` (the rows fold
onto the Q block: admm rho 1.0, al rho 1.0; the go2 foot gates run AL rho 1e3, GaitProgrammer defaults to the loop-stable 100).
Returns the group index.

Measured (2026-09-26). The AL outer loop converges/freezes after the first
solve on these rows, so they act as PENALTIES — rho sets the residual: on the
gentle indy7 scenario (tracking = stay, target 2 cm up) |p - tgt| is 2.3 cm at
rho 10, 1.3 cm at rho 100, 0.8 mm at rho 1e3 (the terminal `EE_POS` AL row does
not move the EE at all there); ADMM projects per inner iteration and reaches the
same at rho ~1e2. The telemetry residual at the measured state equals pinocchio's
frame position to < 2e-4 (f32 FK); on go2 fc the feet stay < 1 mm under a lateral
imu goal and a solver-only foot lift (FR +3 cm from knot 4, wrench pinned, fn on
three feet) reaches ~3 cm at AL rho 1e3 with the stance feet held < 1 cm. The
merit's row term is exact against the duals and residuals (zero-dual and dual
checks agree to 1e-6). CLOSED LOOP (open work): the stiffness the rows
need to hold feet within mm breaks the loop's SQP — on the static stand with
stance rows AL rho 1e3 rejects every line-search step on 26/150 ticks and the base
sags 4 cm, every ADMM variant collapses, AL rho 100 keeps the stand with a 1.8 cm
residual. Two fixes landed since: the solver bumps the trust-region rho x100 on a
non-PD direct factor (the f32 Cholesky failure the stiff rows cause; settings.h
`NON_PD_RHO_FACTOR`), and the programmer re-anchors stance targets to the measured feet
every tick — the static stand with stance rows now holds (`test_fc_mpc_stands_with_foot_rows`).
These are horizon-local targets: the measured feet may accumulate world-frame
drift while the model's residual stays small. The standing gate separately bounds
measured horizontal drift to 3 cm; it does not establish a no-slip locomotion gate.
Lifting a foot is still open: without a support-polygon base plan the model tilts and the
diagonal foot rises; S2 (weight shift) / S3 (lift-a-foot) closed-loop gates are not shipped.

## `set_row_group_targets(g, targets)`

Per-knot target table of a CONTACT_POS group (`(N, n_rows)`, `(n_frames, 3)` or
`(n_rows,)`); ADMM z/y re-initialize on the next solve because the residual
moved. `GaitProgrammer.apply(t, q)` writes it every tick (stance targets
re-anchored to measured feet, future touchdown targets at planned landings, and
swing targets along `GaitSchedule.swing_curve`).

# Figure refresh, October 2026

Current-code reruns of the paper's simulated experiments on an RTX 5090 with CUDA 13.2, timing
collected in exclusive windows on October 1 and 3. The published figures remain the paper's
results; these describe the current code. Protocols and their differences from the paper:
[paper-figures README](../examples/paper-figures/README.md).

| Figure | Data | Window needed | Status |
| --- | --- | --- | --- |
| Fig-3 scalability and heatmap | GATO N × B sweep and multi-threaded QDLDL-based CPU solver B sweep (Oct 1); MPCGPU imported from its Oct 1 evening harness run (main 3566358) | Yes | Refreshed |
| Fig-4 batched rho search | 50 goals × 24 cost settings | No | Refreshed after a plotting fix and a solver fix |
| Fig-5 disturbance rejection | Fixed-pacing force sweep and 50 N trajectories | No | Refreshed |
| Fig-7 / Table I pick-and-place (Oct 3) | 100 seeded scenarios × B ∈ {1, 8, 32, 128}, corrected simulator, two task settings | No (fixed pacing) | Refreshed: stop-at-goal 15 → 100 % and pass-through 26 → 97 % with batch size |
| Fig-3 on the paper's Indy7 (Oct 3) | GATO N × B sweep and CPU B sweep with `--robot indy7` | Yes | Refreshed in an exclusive window (third run; the two shared-box runs agreed within 3%) |

## Seed A/B (September 30)

The September 27 checkpoint measured iiwa14 N64 medians about 41%, 37% and 13% above saved August
values at B1/8/128, after the benchmark's initial guess changed from zero-tail to hold. One
quiet-window run on a single source and binary, three repeats per seed in alternating order with
one frozen reference, isolated that change:

| iiwa14 N64 batch | zero-tail median range (ms) | hold median range (ms) | August median (ms) | zero-tail vs August |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4495–0.4510 | 0.6330–0.6340 | 0.4490 | +0.3% |
| 8 | 0.7155–0.7190 | 0.9430–0.9530 | 0.6980 | +2.7% |
| 128 | 4.7915–4.7920 | 5.2000–5.2040 | 4.5950 | +4.3% |

The seed accounts for almost all of the gap. Residuals of about 3% at B8 and 4% at B128 remain
unattributed: the August cells are single saved runs on older source. These are internal solver
times.

## Fig-3: matched iiwa14 figure-eight benchmark

Internal solver time per batched solve at N = 64. Data: `examples/benchmarks/data/sweep_fig8_*.csv`
(last row per cell), with the GATO reference hash and seed policy in the `.runs.jsonl` companion.

| Batch | GATO (ms) | QDLDL-based CPU (ms) | MPCGPU × batch (ms) | GATO vs CPU | GATO vs MPCGPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.452 | 3.079 | 0.178 | 6.8× | 0.4× |
| 4 | 0.601 | 3.316 | 0.714 | 5.5× | 1.2× |
| 16 | 0.969 | 4.251 | 2.854 | 4.4× | 2.9× |
| 64 | 2.492 | 16.407 | 11.418 | 6.6× | 4.6× |
| 128 | 4.816 | 29.728 | 22.835 | 6.2× | 4.7× |

- GATO is 4.4–7.2× faster than the multi-threaded QDLDL-based CPU solver (OSQP with QDLDL,
  `pysqpcpu`) at every batch size from 1 to 128.
- MPCGPU solves one problem at a time. It is faster for one or two problems; GATO overtakes it from a
  batch of four and is 4.7× faster at 128.
- These ratios are smaller than the paper's 18–21× (CPU) and 1.4–16× (GPU). The paper measured the
  Indy7 arm with its own harness and timing boundary, on a different CPU/GPU system; this benchmark uses the iiwa14 with all three
  solvers on an identical problem and internal solver time. Against the previously saved cells, the
  CPU baseline moved by −2% to +22% (batch 2 is the outlier) and GATO by −12% to +7% (N = 128 at small
  batches is now up to 12% faster). Single runs per cell on a CPU baseline carry that much spread; do not
  quote these ratios as reproductions of the paper's numbers.

## Fig-3 on the paper's Indy7 (October 3)

The paper measured the Indy7; the matched benchmark above uses the iiwa14. To separate the robot
from the timing boundary and hardware, the same FAIR harness now runs on the Indy7
(`reproduce_fig3_fair.py --robot indy7`): the same figure-eight (A = 0.15 m, period 6 s) centered at
the end effector of `INDY7_START_CONFIGS["ready"]`, the same costs and budget (SQP = 1, PCG cap 200 /
rel 1e-4, rho 0.01), the same EE-frame metric. MPCGPU has no Indy7 build, so the goal is synthesized
from the formula and the comparison is GATO against the CPU solver only. Data:
`examples/benchmarks/data/sweep_fig8_{gato,bt}_indy7.csv`.

Internal solver time per batched solve at N = 64:

| Batch | GATO Indy7 (ms) | QDLDL-based CPU Indy7 (ms) | GATO vs CPU | GATO Indy7 / iiwa14 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.266 | 2.498 | 9.4× | 0.59 |
| 4 | 0.338 | 3.169 | 9.4× | 0.56 |
| 16 | 0.633 | 3.307 | 5.2× | 0.65 |
| 64 | 1.875 | 12.002 | 6.4× | 0.75 |
| 128 | 3.621 | 21.961 | 6.1× | 0.75 |

- GATO is 5.2–10.6× faster than the CPU solver on the Indy7, against 4.4–7.2× on the iiwa14. The
  six-joint arm makes GATO 25–45% faster per solve; the CPU solver's time barely changes, so the
  ratio grows by about 1.5×.
- That is still 2–3× short of the paper's 18–21×. The robot explains a minority of the gap; the rest
  has not been isolated. Measurement boundaries, the CPU baseline's build, and
  the hardware differ; matched experiments are needed to attribute their effects.
- GATO's Indy7 heat map (N = 8…128, B = 1…512) is `examples/paper-figures/fig3_fair_heatmap_indy7.png`
  after assembly; at N ≤ 32 and B ≤ 4 the solve time is 0.135–0.19 ms regardless of N, so launch
  and synchronization cost dominates there.
- Collection: the table is the October 3 11:48 run in an exclusive window (`a2rlab-timing-chain`
  run `20261003-114755`). Two earlier runs on a shared box (23:18 and 00:32, with another agent's
  test suite in the background) agreed with it within 2.7% on every GATO cell; the CPU lane moved
  by up to 17% between runs, the same spread the iiwa14 lane shows — single runs of the CPU
  baseline carry that much noise, so read its ratios to ±15%.

## Fig-7 / Table I: pick-and-place with an unmodelled 15 kg payload (October 3)

The paper's point is that a batch of disturbance hypotheses, selected each tick by consistency
with the observed motion, turns a failing single-model MPC into a working one. The refresh keeps
that claim and shows it on two versions of the task, because the task as published is a
*pass-through* (an instantaneous arrival gate) while the second task requires
the arm to remain within its arrival gates for a short dwell. Neither task models
placing or releasing the payload onto a surface.
Both use the corrected simulator (`unit-quaternion-pendulum-v2`, `ready` start), N = 16,
dt = 10 ms, 5 SQP iterations, the paper's costs, a 15 kg payload with random length / damping /
initial angle, 100 seeded scenarios (seed 0, the same list for every row), and the paper's gate:
EE within 5 cm of the goal with joint-velocity norm below 1 rad/s, 5 s per goal. Fixed pacing, so
every pool is deterministic and needs no quiet window. `reproduce_fig7_pickplace.py --task stop`
and `--task pass-through`; `--success-plot` draws `fig7_success_vs_batch.png`.

| Task | Hypothesis sampler | B = 1 | 8 | 32 | 128 | Mean completion of successes (s) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **Stop at each goal** — reference ramps to the next goal over 1.5 s (minimum jerk), gates must hold 100 ms | paper's exploration sampler (`fe`) | 15% | **100%** | **99%** | **98%** | 11.1 / 9.4 / 8.7 / 8.7 |
| **Pass through each goal** — the paper's protocol: step reference, instantaneous gate | identified weight + exploration (`wid`) | 26% | 83% | **97%** | **95%** | 11.9 / 6.8 / 5.7 / 5.2 |

Batching improves success in both tested settings: at B = 8 the stop task reaches
100% and the pass-through task 83%; the latter improves further to 97% at B = 32.
Every failure in the headline rows at B ≥ 8 is a 3- or 4-of-5 near miss;
there are no divergences. What changes between the tasks is how the batch is filled, and the
physics says which to use:

- When the arm must stop with the load still swinging, the batch should *not* chase the swing. The
  paper's sampler keeps every hypothesis within 20 N of a slowly moving estimate; the control it
  selects is smooth and the arm settles. A hypothesis carrying the identified payload weight tracks
  the reference faster, excites the swing, and the swinging bob keeps the joints above the 1 rad/s
  gate: on this task the identified-weight sampler reaches 1 / 4 / 3 / 11% at 15 kg and
  40 / 68 / 81 / 59% at 5 kg. A planner check with a constant known force
  confirms the compensation itself is right (exact hypothesis holds the goal at 3 cm, zero
  hypothesis sags 5–8 cm) — the loss is the swing it provokes.
- When the arm flies through the goals, speed is what the gate rewards and knowing the load pays:
  the identified-weight sampler lifts the paper's sampler's 6 / 62 / 84 / 82% to 26 / 83 / 97 / 95%
  (paired wins/losses at B = 1/8/32/128: 25/5, 30/9, 15/2, 16/3;
  unadjusted two-sided exact McNemar p = 0.000325, 0.001065, 0.002350, 0.004425).
  These are exploratory comparisons on scenarios used while selecting settings,
  not held-out confirmation. The previous paragraph's 35/10 and nonsignificance
  claim belonged to the earlier inertial-row sampler, not this headline variant.

These are unconstrained simulations: torque and velocity limits are disabled.
The dwell checks arm EE position and joint velocity, not payload swing or contact
with a placement surface. The saved headline stop pool records a dirty source tree;
its source SHA alone cannot reconstruct that run. Preserve the pool as exploratory
evidence; clean-source held-out validation is required before stronger claims.

Secondary rows, same scenarios: stop task at 12 kg with the paper's sampler 31 / 100 / 100 / 100%,
at 5 kg 89 / 100 / 100 / 100% (arm-arrival success is higher for the lighter payload);
pass-through with the paper's sampler
6 / 62 / 84 / 82%.

What the diagnosis found, and why the protocol has two settings:

- *The paper's estimator never finds the payload.* Scoring the true wrench against the one-step
  rollout shows the selection works (the true wrench predicts the next state 5× better than zero
  and wins 96% of ticks), but the wrench a 15 kg bob exerts on this arm averages 270 N and swings by
  hundreds of newtons per tick, while the `ForceEstimator` explores within a 20 N ball and blends 10%
  per tick: its vertical estimate averages −17 N against a 147 N weight in every scenario. The batch
  helps by selecting among small guesses each tick. These exploratory observations
  motivate the stop setting; they do not establish payload settling or placement.
- *The published protocol is a fly-through.* With the paper's costs (`u_cost` 5·10⁻⁷, no torque
  limits) and a step reference, the arm sprints at 5–10 rad/s and the payload swings 50–70°
  (near inverted at peaks); the gate is met in passing at about 1 rad/s. Requiring the same gate to
  hold for 100 ms drops every estimator to 0–5% at that setting. A minimum-jerk reference between
  goals halves the joint speeds and swing (median 1.6 rad/s, 35–45°) and makes the stop task
  solvable; that is the only change the stop task makes to the controller's inputs.
- *Settings that do not help*: 10 SQP iterations (the cap is hit at every step but the failures are
  not convergence-limited), `qd_cost` 0.1–0.5 (steady-state sag or sluggishness), torque penalties
  (the arm then cannot hold the load), longer weight filters, smaller exploration radii.
- `IdentifiedWrenchSampler` (`gato/estimators.py`, `MPC_GATO(estimator="wid")`) builds the batch
  from a least-squares identification of the payload weight (the gravity-aligned component of the
  one-step motion residual, filtered at 0.1 s) plus a zero row and Fibonacci-sphere perturbations
  with an adaptive radius. Rows carrying the full identified wrench (`inertial_rows=True`) explain
  the last tick best but extrapolate the payload's inertial reaction over the horizon and
  occasionally fling the arm (2–3 of 100 scenarios), so they are off by default.
- `MPC_GATO.run_mpc_goals(settle_time=, goal_ramp=)` carry the two protocol knobs; both default to
  the paper's loop (0).

## Fig-4: the published plotting script produced an empty figure

Each solve's best-merit curve ends when its SQP loop stops, after 2 to 101 iterations. The script
averaged by truncating every curve to the shortest one, which left a single point and an empty plot;
the August figure had the same defect. Curves are now extended with their final value, which is what a
best-merit-so-far curve means after the solver stops. The refreshed figure averages 50 goals × 24 cost
settings; the paper describes 100 runs × 81 cost choices, so it is not the paper's exact grid.

After that fix the full regeneration was still empty, which exposed a solver bug: some batch
members reported a NaN initial merit and never accepted a line-search step. The terminal knot's
cost read an unwritten shared-memory control slot with zero weight, and 0 × NaN from a previous
kernel's leftovers is NaN. Fixed in `gato/bsqp/kernels/merit.cuh` (commit `9e25778`), with a
regression test; every golden is bit-identical. Runs without NaN or Inf left in shared memory were
unaffected; earlier results can only have been affected after a solve that produced such values.

## Fig-5: unchanged behavior

Tracking error and joint velocity versus disturbance force match the August data closely. Batches of
8 or more keep tracking error near 2 cm up to 80 N; a single solve reaches 10 cm. Fixed pacing measures
control quality without the batch-latency effect shown in the paper.

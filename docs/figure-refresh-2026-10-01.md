# Figure refresh, October 1, 2026

Current-code reruns of the paper's simulated experiments on an RTX 5090 with CUDA 13.2, collected
in an exclusive overnight window. The published figures remain the paper's results; these describe
the current code. Protocols and their differences from the paper: [paper-figures README](../examples/paper-figures/README.md).

| Figure | Data | Window needed | Status |
| --- | --- | --- | --- |
| Fig-3 scalability and heatmap | GATO N × B sweep and multi-threaded QDLDL-based CPU solver B sweep (Oct 1); MPCGPU imported from its Oct 1 evening harness run (main 3566358) | Yes | Refreshed |
| Fig-4 batched rho search | 50 goals × 24 cost settings | No | Refreshed after a plotting fix and a solver fix |
| Fig-5 disturbance rejection | Fixed-pacing force sweep and 50 N trajectories | No | Refreshed |
| Fig-7 / Table I pick-and-place (Oct 3) | 100 seeded scenarios × B ∈ {1, 8, 32, 128}, corrected simulator, paper gate | No (fixed pacing) | Refreshed with the identified-weight hypothesis batch; the paper's estimator never finds the payload |
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
  is the measurement (internal solver time on an identical problem versus the paper's harness and
  timing boundary), the CPU baseline's build, and the hardware.
- GATO's Indy7 heat map (N = 8…128, B = 1…512) is `examples/paper-figures/fig3_fair_heatmap_indy7.png`
  after assembly; at N ≤ 32 and B ≤ 4 the solve time is 0.135–0.19 ms regardless of N, so launch
  and synchronization cost dominates there.
- Collection: the table is the October 3 11:48 run in an exclusive window (`a2rlab-timing-chain`
  run `20261003-114755`). Two earlier runs on a shared box (23:18 and 00:32, with another agent's
  test suite in the background) agreed with it within 2.7% on every GATO cell; the CPU lane moved
  by up to 17% between runs, the same spread the iiwa14 lane shows — single runs of the CPU
  baseline carry that much noise, so read its ratios to ±15%.

## Fig-7 / Table I: pick-and-place with an unmodelled 15 kg payload (October 3)

`reproduce_fig7_pickplace.py --n-scenarios 100 --batch-sizes 1,8,32,128`, corrected simulator
(`unit-quaternion-pendulum-v2`, `ready` start), the paper's protocol otherwise: five step goals,
N = 16, dt = 10 ms, 5 SQP iterations, 15 kg payload with random length / damping / initial angle,
success = EE within 5 cm of the goal with joint-velocity norm below 1 rad/s before a 5 s timeout.
Fixed pacing, so the pools are deterministic and need no quiet window.

| Estimator behind the hypothesis batch | B = 1 | 8 | 32 | 128 | mean completion, successes (s) |
| --- | ---: | ---: | ---: | ---: | --- |
| Paper's ForceEstimator (`--estimator fe`) | 6% | 62% | 84% | 82% | 5.8 / 6.4 / 7.1 / 6.7 |
| Identified weight + bounded exploration (`--estimator wid`, default) | 26% | 83% | 97% | 95% | 11.9 / 6.8 / 5.7 / 5.2 |

Paired on the same 100 scenarios the new sampler wins at every batch size (B = 1: 25 won / 5 lost,
B = 8: 35 / 10; exact McNemar p ≈ 3·10⁻⁴ and 2·10⁻⁴ for the bracket variant measured first).
Every failure at B ≥ 8 is a 3- or 4-of-5 near miss; there are no divergences.

Why the estimator changed. Scoring the true payload wrench against the one-step rollout shows the
selection mechanism works (the true wrench predicts the next state 5× better than zero and wins 96%
of ticks), but the wrench a 15 kg bob exerts on this arm averages 270 N and swings by hundreds of
newtons per tick, while the paper's `ForceEstimator` explores within a 20 N ball and blends 10% per
tick: its estimate averages −17 N vertical against a 147 N weight in every scenario, so the batch
helped only by selecting among weak guesses. `IdentifiedWrenchSampler` (`gato/estimators.py`)
builds the batch around a least-squares identification of the payload weight (the gravity-aligned
component of the one-step motion residual, filtered at 0.1 s) plus a zero row and Fibonacci-sphere
perturbations with an adaptive radius; `MPC_GATO(estimator="wid")`. Rows carrying the full identified
wrench (`inertial_rows=True`) explain the last tick best but extrapolate the payload's inertial
reaction to the arm's own motion and occasionally fling the arm, so they are off by default.

Why the gate stays the paper's. Requiring both gates to *hold* for 100 ms (`--settle-time 0.1`)
collapses every arm (FE 0 / 33 / 62 / 57 %, new sampler 0 / 4 / 5 / 5 %): with these costs the
arm passes through each goal at about 1 rad/s and the payload swings 50–70° (peaks near inverted),
so nothing settles. A minimum-jerk reference between goals (`--goal-ramp 1.5`) halves the joint
speeds and swing, after which the estimator ordering inverts: the compensated arm follows the
reference faster, excites the swing more, and the swinging bob keeps the joints above 1 rad/s,
while the uncompensated arm settles because it under-delivers (a planner check with a constant
known force confirms the compensation itself is right: exact hypothesis 3 cm, zero hypothesis
5–8 cm sag). "Hold at the goal with a swinging payload" is therefore a different task that needs
a settling cost and a payload model, not the paper's pass-through task; both knobs stay available
for that study. Settings that do not help: 10 SQP iterations (not convergence-limited), `qd_cost`
0.1–0.5 (steady-state sag or sluggishness), torque penalties (the arm then cannot hold the load),
smaller payloads (same swing), longer weight filters.

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

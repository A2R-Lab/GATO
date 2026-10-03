# Figure refresh, October 1, 2026

Current-code reruns of the paper's simulated experiments on an RTX 5090 with CUDA 13.2, collected
in an exclusive overnight window. The published figures remain the paper's results; these describe
the current code. Protocols and their differences from the paper: [paper-figures README](../examples/paper-figures/README.md).

| Figure | Data | Window needed | Status |
| --- | --- | --- | --- |
| Fig-3 scalability and heatmap | GATO N × B sweep and multi-threaded QDLDL-based CPU solver B sweep (Oct 1); MPCGPU imported from its Oct 1 evening harness run (main 3566358) | Yes | Refreshed |
| Fig-4 batched rho search | 50 goals × 24 cost settings | No | Refreshed after a plotting fix and a solver fix |
| Fig-5 disturbance rejection | Fixed-pacing force sweep and 50 N trajectories | No | Refreshed |
| Fig-7 / Table I pick-and-place | — | — | Kept as published; the 8/10 success gap is unresolved |
| Fig-3 on the paper's Indy7 (Oct 3) | GATO N × B sweep and CPU B sweep with `--robot indy7` | Yes | Collected twice, reproducible; another agent's test suite was active on the box (see below) |

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
| 1 | 0.259 | 2.694 | 10.4× | 0.57 |
| 4 | 0.336 | 2.740 | 8.2× | 0.56 |
| 16 | 0.634 | 3.361 | 5.3× | 0.65 |
| 64 | 1.863 | 12.945 | 6.9× | 0.75 |
| 128 | 3.625 | 23.981 | 6.6× | 0.75 |

- GATO is 5.3–10.4× faster than the CPU solver on the Indy7, against 4.4–7.2× on the iiwa14. The
  six-joint arm makes GATO 25–45% faster per solve; the CPU solver's time barely changes, so the
  ratio grows by about 1.5×.
- That is still 2–3× short of the paper's 18–21×. The robot explains a minority of the gap; the rest
  is the measurement (internal solver time on an identical problem versus the paper's harness and
  timing boundary), the CPU baseline's build, and the hardware.
- GATO's Indy7 heat map (N = 8…128, B = 1…512) is `examples/paper-figures/fig3_fair_heatmap_indy7.png`
  after assembly; at N ≤ 32 and B ≤ 4 the solve time is 0.135–0.19 ms regardless of N, so launch
  and synchronization cost dominates there.
- Collection: the leg ran twice (23:18 and 00:32 on October 2–3, `a2rlab-timing-chain` runs
  `20261002-231837` and `20261003-003201`), and every cell agreed within 1%. The quiet gate (GPU idle,
  no compute processes, load ≤ 2) passed at both starts, but another agent's receipt test suite was
  running on the box during both, and one foreign GPU process was seen for a few seconds during the
  first run. These are not exclusive-window numbers; the agreement between the two runs is the
  evidence for them.

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

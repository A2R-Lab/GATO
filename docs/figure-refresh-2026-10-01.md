# Figure refresh, October 1, 2026

Current-code reruns of the paper's simulated experiments on an RTX 5090 with CUDA 13.2, collected
in an exclusive overnight window. The published figures remain the paper's results; these describe
the current code. Protocols and their differences from the paper: [paper-figures README](../examples/paper-figures/README.md).

| Figure | Data | Window needed | Status |
| --- | --- | --- | --- |
| Fig-3 scalability and heatmap | GATO N × B sweep and BatchThneed B sweep (Oct 1); MPCGPU imported from its Sep 30 harness run | Yes | Refreshed |
| Fig-4 batched rho search | 50 goals × 24 cost settings | No | Refreshed after a plotting fix and a solver fix |
| Fig-5 disturbance rejection | Fixed-pacing force sweep and 50 N trajectories | No | Refreshed |
| Fig-7 / Table I pick-and-place | — | — | Kept as published; the 8/10 success gap is unresolved |

## Fig-3: matched iiwa14 figure-eight benchmark

Internal solver time per batched solve at N = 64. Data: `examples/benchmarks/data/sweep_fig8_*.csv`
(last row per cell), with the GATO reference hash and seed policy in the `.runs.jsonl` companion.

| Batch | GATO (ms) | BatchThneed CPU (ms) | MPCGPU × batch (ms) | GATO vs CPU | GATO vs MPCGPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.452 | 3.079 | 0.178 | 6.8× | 0.4× |
| 4 | 0.601 | 3.316 | 0.711 | 5.5× | 1.2× |
| 16 | 0.969 | 4.251 | 2.843 | 4.4× | 2.9× |
| 64 | 2.492 | 16.407 | 11.373 | 6.6× | 4.6× |
| 128 | 4.816 | 29.728 | 22.746 | 6.2× | 4.7× |

- GATO is 4.4–7.2× faster than the threaded CPU solver at every batch size from 1 to 128.
- MPCGPU solves one problem at a time. It is faster for one or two problems; GATO overtakes it from a
  batch of four and is 4.7× faster at 128.
- These ratios are smaller than the paper's 18–21× (CPU) and 1.4–16× (GPU). The paper measured the
  Indy7 arm with its own harness and timing boundary; this benchmark uses the iiwa14 with all three
  solvers on an identical problem and internal solver time. Against the previously saved cells, the
  CPU baseline moved by −2% to +22% (batch 2 is the outlier) and GATO by −12% to +7% (N = 128 at small
  batches is now up to 12% faster). Single runs per cell on a CPU baseline carry that much spread; do not
  quote these ratios as reproductions of the paper's numbers.

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

# Merit launch timing, October 22, 2026

The merit kernel evaluates the eight line-search candidates of every knot and
solve. Two questions from issue #19 (Felix Mueh, GTX 1070): whether fusing the
eight candidates into one block per knot helps on a GPU with few SMs, and what
block size the kernel should run at. Both were measured in a user-assigned
quiet window on the RTX 5090, source `a791af7`, grid-rbd 0.5.4, with
`examples/benchmarks/merit_fusion_timing.py`: Indy7 and iiwa14 at N32 and N64,
Go2 at N16, batch 1, 8 and 16, two SQP iterations per cold solve from the
hold-at-x seed with the trust region reset, 20 warm-up and 200 timed solves per
cell, the merit time per launch from `SolverStats.stage_times_us`. Three
conditions: the 5090 as is, and the 5090 throttled through MPS
(`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) to 10 and 25 percent of its SMs, the
stand-in for a 14-SM Jetson Orin or a 15-SM GTX 1070. Every row is a median of
200 solves; the p90 is within a few percent of it in every cell. Raw rows:
[merit_timing_20261022.jsonl](../examples/benchmarks/data/merit_timing/merit_timing_20261022.jsonl)
(SHA256 `2299d15ea2a9`).

Columns: merit device time per launch in microseconds at the generated
per-robot maximum block size (the former default: Indy7 256, iiwa14 352, Go2
448), at 128 and at 64 threads; the whole solve's median at the former default
and at 64; and the fused launch's time relative to the one-block-per-candidate
launch at the former default block size.

## RTX 5090

| plant | N | B | merit, generated max | merit, 128 | merit, 64 | solve, generated max | solve, 64 | fused / unfused at the max |
|---|---|---|---|---|---|---|---|---|
| go2 | 16 | 1 | 61 | 62 | 63 | 2296 | 2303 | 7.22 |
| go2 | 16 | 8 | 175 | 114 | 119 | 2529 | 2422 | 2.50 |
| go2 | 16 | 16 | 298 | 169 | 173 | 3072 | 2817 | 1.50 |
| iiwa14 | 32 | 1 | 31 | 31 | 32 | 698 | 701 | 4.80 |
| iiwa14 | 32 | 8 | 112 | 65 | 46 | 863 | 733 | 1.31 |
| iiwa14 | 32 | 16 | 207 | 98 | 72 | 1081 | 798 | 1.33 |
| iiwa14 | 64 | 1 | 46 | 32 | 33 | 2287 | 2273 | 3.18 |
| iiwa14 | 64 | 8 | 208 | 97 | 72 | 3160 | 2874 | 1.35 |
| iiwa14 | 64 | 16 | 397 | 179 | 119 | 3646 | 3057 | 1.10 |
| indy7 | 32 | 1 | 28 | 28 | 31 | 591 | 609 | 3.98 |
| indy7 | 32 | 8 | 96 | 57 | 53 | 734 | 651 | 1.19 |
| indy7 | 32 | 16 | 172 | 98 | 68 | 899 | 687 | 1.18 |
| indy7 | 64 | 1 | 42 | 29 | 30 | 1397 | 1373 | 2.63 |
| indy7 | 64 | 8 | 173 | 98 | 71 | 1881 | 1669 | 1.18 |
| indy7 | 64 | 16 | 327 | 177 | 123 | 2285 | 1849 | 0.97 |

## RTX 5090 at 10 percent of its SMs (MPS)

| plant | N | B | merit, generated max | merit, 128 | merit, 64 | solve, generated max | solve, 64 | fused / unfused at the max |
|---|---|---|---|---|---|---|---|---|
| go2 | 16 | 1 | 148 | 73 | 73 | 2461 | 2310 | 2.91 |
| go2 | 16 | 8 | 841 | 387 | 396 | 5692 | 4708 | 1.43 |
| go2 | 16 | 16 | 1645 | 736 | 752 | 9271 | 7340 | 1.23 |
| iiwa14 | 32 | 1 | 94 | 48 | 41 | 821 | 716 | 1.52 |
| iiwa14 | 32 | 8 | 574 | 249 | 159 | 2011 | 1123 | 1.15 |
| iiwa14 | 32 | 16 | 1127 | 468 | 287 | 3430 | 1657 | 1.07 |
| iiwa14 | 64 | 1 | 163 | 80 | 63 | 2551 | 2335 | 1.68 |
| iiwa14 | 64 | 8 | 1141 | 476 | 289 | 5561 | 3766 | 1.08 |
| iiwa14 | 64 | 16 | 2259 | 924 | 547 | 8469 | 4846 | 1.08 |
| indy7 | 32 | 1 | 81 | 58 | 46 | 707 | 634 | 1.36 |
| indy7 | 32 | 8 | 469 | 248 | 173 | 1655 | 1032 | 1.03 |
| indy7 | 32 | 16 | 918 | 478 | 322 | 2822 | 1564 | 0.96 |
| indy7 | 64 | 1 | 135 | 81 | 63 | 1587 | 1440 | 1.48 |
| indy7 | 64 | 8 | 926 | 484 | 328 | 3836 | 2566 | 0.98 |
| indy7 | 64 | 16 | 1834 | 944 | 633 | 6195 | 3649 | 0.96 |

## RTX 5090 at 25 percent of its SMs (MPS)

| plant | N | B | merit, generated max | merit, 128 | merit, 64 | solve, generated max | solve, 64 | fused / unfused at the max |
|---|---|---|---|---|---|---|---|---|
| go2 | 16 | 1 | 107 | 63 | 63 | 2381 | 2301 | 4.07 |
| go2 | 16 | 8 | 574 | 267 | 275 | 4255 | 3602 | 1.44 |
| go2 | 16 | 16 | 1111 | 505 | 518 | 6730 | 5474 | 1.27 |
| iiwa14 | 32 | 1 | 66 | 47 | 36 | 765 | 701 | 2.20 |
| iiwa14 | 32 | 8 | 390 | 174 | 115 | 1533 | 958 | 1.11 |
| iiwa14 | 32 | 16 | 759 | 323 | 204 | 2494 | 1323 | 1.09 |
| iiwa14 | 64 | 1 | 113 | 61 | 57 | 2437 | 2329 | 1.27 |
| iiwa14 | 64 | 8 | 769 | 328 | 205 | 4633 | 3449 | 1.11 |
| iiwa14 | 64 | 16 | 1516 | 625 | 376 | 6570 | 4154 | 1.09 |
| indy7 | 32 | 1 | 58 | 45 | 32 | 658 | 609 | 1.90 |
| indy7 | 32 | 8 | 321 | 174 | 122 | 1268 | 855 | 0.99 |
| indy7 | 32 | 16 | 617 | 324 | 225 | 2049 | 1208 | 0.96 |
| indy7 | 64 | 1 | 96 | 59 | 50 | 1503 | 1409 | 1.16 |
| indy7 | 64 | 8 | 625 | 330 | 227 | 3060 | 2228 | 0.98 |
| indy7 | 64 | 16 | 1230 | 638 | 430 | 4652 | 2955 | 0.96 |

## Reading

The block size is the lever. At 64 threads the merit launch takes two to four
times less than at the generated maximum at batch 8 and 16 in every condition,
and the whole solve 15 to 25 percent less on the 5090 and up to 45 percent less
at 10 percent of its SMs; at batch 1 the sizes tie within a few microseconds.
Go2 is the same at 64 and 128. The merit kernel is latency bound, one knot's
integrator and cost per block, so the smaller block raises the number of
resident blocks without starving the loops; GRiD's widest-loop count
(`TRACKING_COST_SUGGESTED_THREADS`, 224 on iiwa14) is not the operating point.
The default is therefore 64 threads (`GATO_MERIT_THREADS`), bit-identical to
the former default (goldens unchanged, the thread-invariance gate in the
register tests).

The fused launch does not help. One block per knot evaluating the eight
candidates in turn never won: within the repeat spread on Indy7 at batch 8 and 16
(0.96 to 0.99 of the unfused time) and slower everywhere else, up to seven times at
batch 1 (Go2), in all three conditions: the candidates
serialize inside the block while the GPU, even at 10 percent of its SMs, still
had room for the candidate blocks. The SM-count rule that selected it is
removed; the path stays as `GATO_MERIT_FUSED=1` for a measurement on hardware
that behaves differently, and the kernel gate keeps it bit-identical.

## Jetson AGX Orin, October 24, 2026

Felix Mueh ran the same breakdown on a Jetson AGX Orin 32 GB at 40 W (JetPack
6.2, CUDA 12.6, GATO 0.1.5 at `4d5f8c2`, the GPU clock between 306 and 816 MHz
during the run) and posted it on issue #19: eight SQP iterations, batch 1, 4 and
16, ten interleaved repetitions per cell, the per-stage times from
`SolverStats.stage_times_us`, mean and sample standard deviation in
milliseconds per solve. The four launches are 32, 64 and 256 threads per block
and the fused launch at 64.

Warm solve (the previous solution as the seed):

| launch | B | solve | merit | linsys | kkt | schur | merit share |
|---|---|---|---|---|---|---|---|
| 32 | 1 | 22.73 ± 1.89 | 2.73 ± 0.21 | 16.34 ± 1.46 | 1.77 | 1.28 | 12.0 % |
| 64 | 1 | 22.76 ± 2.06 | 2.94 ± 0.28 | 16.18 ± 1.63 | 1.75 | 1.28 | 12.9 % |
| 256 | 1 | 25.07 ± 3.35 | 6.05 ± 0.88 | 15.45 ± 2.10 | 1.67 | 1.22 | 24.1 % |
| 64 fused | 1 | 26.23 ± 1.60 | 5.72 ± 0.29 | 16.75 ± 1.21 | 1.79 | 1.31 | 21.8 % |
| 32 | 4 | 26.16 ± 3.82 | 5.76 ± 0.90 | 15.29 ± 2.29 | 2.89 | 1.55 | 22.0 % |
| 64 | 4 | 26.19 ± 4.18 | 6.79 ± 1.17 | 14.52 ± 2.37 | 2.74 | 1.45 | 25.9 % |
| 256 | 4 | 26.26 ± 3.86 | 13.07 ± 1.62 | 9.60 ± 1.61 | 1.85 | 1.00 | 50.0 % |
| 64 fused | 4 | 27.56 ± 5.17 | 9.13 ± 1.67 | 13.77 ± 2.72 | 2.61 | 1.40 | 33.2 % |
| 32 | 16 | 34.24 ± 0.48 | 10.30 ± 0.16 | 15.68 ± 0.22 | 5.32 | 2.31 | 30.1 % |
| 64 | 16 | 37.97 ± 0.58 | 14.04 ± 0.26 | 15.60 ± 0.22 | 5.33 | 2.30 | 37.0 % |
| 256 | 16 | 63.94 ± 1.32 | 40.82 ± 0.83 | 14.61 ± 0.26 | 5.08 | 2.16 | 63.9 % |
| 64 fused | 16 | 38.08 ± 0.32 | 14.20 ± 0.18 | 15.54 ± 0.08 | 5.28 | 2.30 | 37.3 % |

Cold solve (the hold-at-x seed):

| launch | B | solve | merit | linsys | kkt | schur | merit share |
|---|---|---|---|---|---|---|---|
| 32 | 1 | 25.30 ± 0.71 | 3.11 ± 0.03 | 12.32 ± 0.01 | 2.32 | 1.36 | 12.3 % |
| 64 | 1 | 25.32 ± 1.11 | 3.42 ± 0.04 | 12.32 ± 0.01 | 2.38 | 1.36 | 13.5 % |
| 256 | 1 | 29.12 ± 0.86 | 7.17 ± 0.04 | 12.32 ± 0.01 | 2.24 | 1.36 | 24.6 % |
| 64 fused | 1 | 27.77 ± 0.71 | 6.14 ± 0.05 | 12.33 ± 0.01 | 2.41 | 1.36 | 22.1 % |
| 32 | 4 | 31.33 ± 1.41 | 7.01 ± 0.05 | 12.41 ± 0.01 | 3.73 | 1.74 | 22.4 % |
| 64 | 4 | 32.98 ± 0.73 | 8.85 ± 0.04 | 12.41 ± 0.01 | 3.64 | 1.74 | 26.9 % |
| 256 | 4 | 48.67 ± 1.75 | 24.64 ± 1.05 | 12.27 ± 0.35 | 3.47 | 1.74 | 50.6 % |
| 64 fused | 4 | 36.74 ± 1.29 | 12.51 ± 0.07 | 12.42 ± 0.01 | 3.68 | 1.75 | 34.1 % |
| 32 | 16 | 61.21 ± 5.05 | 19.08 ± 1.72 | 20.89 ± 2.41 | 10.38 | 4.58 | 31.2 % |
| 64 | 16 | 65.40 ± 5.85 | 25.23 ± 2.25 | 19.58 ± 2.39 | 9.97 | 4.40 | 38.6 % |
| 256 | 16 | 93.68 ± 5.76 | 60.47 ± 4.01 | 14.56 ± 0.85 | 7.80 | 3.39 | 64.5 % |
| 64 fused | 16 | 67.90 ± 3.42 | 26.12 ± 1.50 | 20.57 ± 1.33 | 10.21 | 4.53 | 38.5 % |

The Orin agrees with the throttled 5090 on the two questions above and adds a
third answer. The merit launch at 64 threads grows 4.8 times from batch 1 to
16 on the warm solve, against 15.75 times reported for the old default on the
GTX 1070, and 256 threads costs two to three times the merit time of 64 at
every batch. The fused launch gains nothing: level with the unfused launch at
batch 16 and 15 percent slower at batch 1. The third answer is that 32 threads
beats 64 on this GPU: level within the spread at batch 1 and 4, and at batch 16
the merit time falls from 14.0 to 10.3 ms and the whole solve by 10 percent,
warm and cold. The 5090 tables did not include 32 threads. From 0.1.6 the
configure-time default is therefore 32 on the Jetson family (sm_72 and sm_87,
`CMakeLists.txt`) and 64 elsewhere; `GATO_MERIT_THREADS` still overrides
either, and results are bit-identical at any size.

The tables also show where the time goes on this GPU once the merit launch is
sized: the linear system. At batch 1 it is 70 percent of the solve, and it does
not grow with the batch (0.96 from batch 1 to 16 on the warm solve); the
cold-solve linsys time is 12.32 ms with a spread of 0.01 ms in every launch
configuration. The October 26 sweep below shows that this is a fixed PCG
iteration count per problem, not the iteration cap.

## Jetson AGX Orin, October 26, 2026: longer horizons

Felix Mueh ran a second sweep on the same Orin at GATO 0.1.6 (the build picked
32 merit threads on sm_87 on its own; JetPack 6.2.1, CUDA 12.6, the GPU clock
between 306 and 816 MHz, ten interleaved repetitions, each run in its own
process; scripts on his fork under `issue19/`). Example 15's problem on the
Indy7, dt 0.01, eight SQP iterations, cold then warm.

Merit block size 32 against 64 at longer horizons, warm solve, mean and sample
standard deviation in milliseconds; the plans are bit-identical between the two
block sizes, and the cold solves show the same ratios (1.16 to 1.44):

| N | B | merit, 32 | merit, 64 | merit 64 / 32 | solve 64 / 32 |
|---|---|---|---|---|---|
| 64 | 1 | 2.49 ± 0.42 | 3.14 ± 0.39 | 1.26 | 1.09 |
| 64 | 4 | 6.39 ± 1.77 | 7.48 ± 0.64 | 1.17 | 0.98 |
| 64 | 16 | 17.98 ± 0.55 | 24.66 ± 0.89 | 1.37 | 1.12 |
| 128 | 1 | 3.10 ± 0.19 | 3.87 ± 0.21 | 1.25 | 0.98 |
| 128 | 4 | 9.59 ± 0.49 | 13.01 ± 0.57 | 1.36 | 1.07 |
| 128 | 16 | 33.19 ± 0.42 | 45.13 ± 0.28 | 1.36 | 1.09 |
| 256 | 1 | 4.92 ± 0.10 | 6.48 ± 0.11 | 1.32 | 1.02 |
| 256 | 4 | 16.83 ± 0.24 | 23.14 ± 0.21 | 1.38 | 1.07 |
| 256 | 16 | 66.07 ± 0.58 | 90.46 ± 0.82 | 1.37 | 1.09 |
| 512 | 1 | 9.04 ± 0.11 | 12.15 ± 0.13 | 1.34 | 1.01 |
| 512 | 4 | 33.28 ± 0.36 | 44.93 ± 0.43 | 1.35 | 1.02 |
| 512 | 16 | 130.35 ± 1.36 | 173.78 ± 0.98 | 1.33 | 1.03 |

The merit launch is faster at 32 threads in all 24 cells; the whole solve gains
up to 12 percent at N64 and batch 16 and 1 to 3 percent at N512, where the
linear system dominates (the ratios below 1 are within the spread). The sm_87
default of 0.1.6 holds at every horizon tried.

PCG iterations per SQP iteration at the library default `max_pcg_iters = 200`,
identical in all ten runs and at batch 1, 4 and 16:

| N | cold | warm |
|---|---|---|
| 32 | 27, 29, 32, 34, 50, 50, 56, 54 | 59, 59, 60, 60, 60, 61, 60, 58 |
| 64 | 30, 44, 53, 56, 85, 88, 58, 57 | 89, 30, 85, 29, 86, 55, 29, 30 |
| 128 | 30, 79, 85, 92, 99, 98, 89, 90 | 85, 77, 76, 47, 89, 82, 60, 84 |
| 256 | 30, 131, 120, 132, 132, 131, 128, 150 | 111, 101, 93, 88, 84, 39, 44, 97 |
| 512 | 31, 144, 118, 128, 130, 159, 152, 200 | 127, 126, 192, 115, 112, 105, 103, 96 |

One SQP iteration in the whole sweep reaches the cap (N512, cold, the last
one). The 12.32 ms of the October 24 cold solve is therefore a fixed iteration
count, and the warm solve at N32 runs more PCG iterations than the cold one
(477 against 332), which is why its linear system took longer in the first
tables. The linear system's share of the warm batch-1 solve grows with the
horizon: 72 percent at N32, 78 at N64, 85 at N128, 89 at N256 and 96 at N512
(13.9, 17.2, 32.3, 69.6 and 427.1 ms); from N256 to N512 the PCG time grows
6.1 times for 1.5 times the iterations.

The same sweep found that the direct path returned its seed on the Orin on
every horizon (every line search failed; `bdsv_first` then ran PCG to the cap
on each later iteration). That was a GATO defect on every target other than
sm_120, fixed in 0.1.7 (CHANGELOG); the bdsv timings of the sweep measured
solves that did no work and are not reported. Which of the two paths the
embedded default should be is the next measurement for this page, on 0.1.7.

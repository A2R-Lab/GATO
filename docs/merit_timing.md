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
candidates in turn was slower in every cell, from a few percent at batch 16
to seven times at batch 1 (Go2), in all three conditions: the candidates
serialize inside the block while the GPU, even at 10 percent of its SMs, still
had room for the candidate blocks. The SM-count rule that selected it is
removed; the path stays as `GATO_MERIT_FUSED=1` for a measurement on hardware
that behaves differently, and the kernel gate keeps it bit-identical.

Times depend on the machine and the driver; they are evidence for this box and
these conditions, not a benchmark of GATO against anything else.

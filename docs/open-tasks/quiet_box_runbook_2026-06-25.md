# Quiet-box runbook — Fig-3 + Fig-7/Table-I timing collection (2026-06-25)

All CODE + CORRECTNESS work is DONE & committed (local/unpushed, branch `cleanup-modernization`).
What remains needs a QUIET GPU/CPU (no other agents' load) because it's TIMING. Run these one at a
time; do NOT run heavy CPU work during a GPU timing sweep or vice-versa. Findings → docs/baselines.md.

## Environment
- GATO repo: `~/Desktop/GATO`, branch `cleanup-modernization`.
- Python with pinocchio: `~/Desktop/GRiD/.venv/bin/python` (GATO venv is lean codegen-only).
- Check first: `free -g` (>~15 GB free) and `nvidia-smi` (no compute apps) before each run.

## Commits this session (all LOCAL, unpushed)
- GATO `cleanup-modernization`: 443aead (FE import fix + diag), 5e0ffb4 (FE seed), dd6243f (Fig-3
  MPCGPU linear), 8a9f199 (Fig-7 caveat), fdee0f6 (BatchThneed baseline). docs/baselines.md is
  intentionally UNCOMMITTED (local findings).
- MPCGPU worktree `/tmp/gato_mpcgpu` (branch `fig3/indy7-mpcgpu`): 7726ccb (tunable Q_COST, set to 1.0).
  Loose: `track_iiwa_pcg.cu` "Full per-step SQP" relabel still uncommitted (fold into the PR).

## 1. GATO fig8 batched-GPU sweep (GPU) — the main Fig-3 curve
Current `benchmark_fig8_64N.pkl` has only M=16. Regenerate full batch range:
```
cd ~/Desktop/GATO
~/Desktop/GRiD/.venv/bin/python examples/benchmark_fig8.py --plant indy7 --N 64 \
    --batch-sizes 1,2,4,8,16,32,64,128 --sim-time 5
```
(Module bsqpN64_indy7 must be built. Tune threads UP for big batches per the jax/torch thread
pathology note — but benchmark_fig8 already handles its own launch config.)

## 2. BatchThneed batched-CPU sweep (CPU) — the fair CPU line
```
cd ~/Desktop/GATO
# one-time build if deps/build are gone: ./baselines/build_cpu_baseline.sh
source baselines/sqpcpu_env.sh
~/Desktop/GRiD/.venv/bin/python baselines/run_batchthneed_fig8.py \
    --batch-sizes 1,2,4,8,16,32,64,128            # -> baselines/batchthneed_fig8_results.pkl
```
Provisional (non-quiet): M=1→2.5ms, 8→3.2, 16→3.5, 64→11.8 ms. Sub-linear (threaded), ~paper shape.

## 3. OSQP single-solve reference (CPU) — faint reference line only
```
~/Desktop/GRiD/.venv/bin/python baselines/run_osqp_fig8.py --N 64 --sim-time 5
```

## 4. MPCGPU per-solve (GPU) — the linear xM line
Q_COST is set to 1.0 (clean 293us; the tradeoff at Q=10/100 inflates time — see baselines.md). Rebuild
+ run in the worktree, then refresh the CSV that Fig-3 reads (median = col 4 of the last line):
```
cd /tmp/gato_mpcgpu
cmake --build build_mpcgpu --target MPCGPU-pcg -j4
mkdir -p build/results && ./build_mpcgpu/MPCGPU-pcg          # prints tracking + "Full per-step SQP solve times (us)"
# Take the LAST "Full per-step SQP solve times" stats row (Average,Std,Min,Max,Median,Q1,Q3 in us)
# and write it as the last line of: ~/Desktop/GATO/baselines/mpcgpu_indy7_fig8_N64.csv
```
Expect median ~293 us (gravity-off, 1 SQP iter). load_mpcgpu() uses col 4 (median).

## 5. Render Fig-3 (no GPU)
```
~/Desktop/GRiD/.venv/bin/python examples/paper-figures/reproduce_fig3_scalability.py --replot
```
Should draw: GATO (sub-linear), BatchThneed (sub-linear CPU), MPCGPU (linear xM), OSQP (faint ref).

## 6. Fig-7 + Table-I pick-place (GPU) — success is correctness, mean-time is timing
FE import bug FIXED + FE seeded (seed=0 default). Batching now helps (batch8 3/5 vs batch1 0/5 on the
fixed scenario). Run the full 100-scenario sweep (LONG; the timing column wants the quiet box):
```
~/Desktop/GRiD/.venv/bin/python examples/paper-figures/reproduce_fig7_pickplace.py     # 100 scenarios
# or --quick for a smoke. NaN scenarios are caught per-batch (won't crash the sweep).
```
Expect a shape-correct, batch-increasing success curve; magnitudes may not match paper (residual
FE-robustness = parked Phase-2). Caveat text already updated.

## Held for explicit user go-ahead (do NOT do without it)
- MPCGPU upstream PR (port + sm_120 + Q_COST + relabel) to A2R-Lab/MPCGPU.
- PUSH all local commits (GATO `cleanup-modernization` + MPCGPU worktree).

## Open small decisions
- Fig-3 OSQP line: currently a faint single-solve reference. Fine as-is now that BatchThneed is the
  real CPU line; drop it entirely if it clutters.
- Whether build_cpu_baseline.sh should move INTO the sqpcpu submodule (to travel / be PR'd upstream).

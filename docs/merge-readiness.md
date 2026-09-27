# Merge-readiness checklist

This is the bounded path to merging the maintained solver, not a requirement to
finish every research idea. No merge, website deployment, release tag or timing
window is authorized merely by this checklist. Updated 2026-09-27.

## 1. User-facing truth and onboarding

- [x] Add a documentation index, explicit feature-status boundaries and API
  migration notes; correct stale standing and receipt-scope descriptions.
- [x] Use a minimal one-module quick start and document example dependencies.
- [x] Correct the gym example's syntax; add a host-only introductory syntax gate.
- [x] GPU smoke: examples 01/02/03/04/06 and 07 `--fc --steps 100` passed on
  2026-09-27; 03 uses fixed pacing. Example 08's timing display remains deferred;
  its syntax and the underlying linsys paths are tested. Custom-robot build is
  covered by the receipt. Shared-box durations are not performance evidence.
- [ ] Repeat the beginner install/build path from a fresh checkout of the final
  candidate. CI's anonymous recursive clone/host install complements but does
  not replace a fresh CUDA build.
- [ ] Finish the small website quick-start/links update alongside the merge,
  preserving the paper plots and identifying their numbers as published results.
  Do not publish new-API instructions linked to `main` before that API lands.
- [ ] Reconcile preferred citation metadata with main's ICRA citation; preserve
  the funding acknowledgement. Remove the temporary quick-start branch selector
  and switch package documentation URLs to main as part of final integration.

## 2. Correctness and integration

- [x] Bounded masked-contact racecheck: AL/barrier/ADMM cases, one SQP/ADMM
  iteration each, zero errors/warnings (2026-09-27). The earlier full ADMM
  golden diagnostic remains incomplete; do not claim exhaustive race coverage.
- [ ] Reconcile floating effort-vector width/documentation before advertising
  that API; scalar-weight standing is the covered configuration meanwhile.
- [x] Full signed receipt for source `78fc5d4`: 341 passed, zero skips on the
  18-module profile (2026-09-27); all 38 goldens passed. Refresh again if later
  fingerprinted sources change, and require green receipt/CPU CI before merge.
- [ ] Compare with the latest remote main, preserve its independent changes,
  review API removals and generated/pinned dependency changes, and test the
  resulting merge candidate. Do not blindly squash/rebase away the receipt's
  attested ancestor; re-attest the resulting commit if history/sources change.

## 3. Focused quiet-window checkpoint

For an overnight coordinator, use the [timing handoff](timing-handoff.md): its
wrapper adds a shared advisory lock and separately selectable compile/calibration
legs. Reserve provisionally two hours for the full bundle, or one hour for the
runtime checkpoint alone; release the box early when finished.

Ready to preview; NOT scheduled or run by this document:

```bash
examples/benchmarks/run_merge_checkpoint.sh --dry-run
# Only after the user declares an exclusive slot and correctness is green:
GATO_QUIET_WINDOW=1 nohup examples/benchmarks/run_merge_checkpoint.sh > /tmp/gato-merge-checkpoint.log 2>&1 &
```

The runner is GATO-only: three independent Fig-3 N64 / B1,8,128 repeats and
ten seeded Fig-7 B128 scenarios. It refuses a dirty source tree, requires an
idle box before/between legs, does not kill other agents, and writes unique
logs/CSVs/tags. Preserve the output directory and verify the box stayed quiet
throughout each leg (boundary checks cannot detect every transient workload).
It records binary hashes, source/pins, receipt and environment; no builds or
autotuning are hidden inside this runner.

- [ ] Compare matching iiwa14 cells with the matching prior iiwa14 CSVs, including
  repeated-run variation. Do not use the Indy7 point-to-point archive.
- [ ] Check numerical quality alongside latency. The Fig-3 sweep follows the
  predicted state, not an independent plant; use the fixed-pacing regression
  gates and matched tracking harness when diagnosing changed results.
- [ ] Report internal solver latency as such. Full Python/controller latency
  and new cross-solver speedup claims need a matched measurement boundary.
- [ ] Separately measure cold/no-op compilation, peak RSS and binary size in a
  fresh temporary CMake tree with separate `GATO_MODULE_OUTPUT_DIR`, explicit
  arch 120 on this box, receipt profile OFF and demo OFF. Use the project Python
  and pybind11; MODULES=`indy7:16;iiwa14:64`; one target at a time under the
  36 GiB / no-swap / two-job systemd cap. Preserve production binaries.
- [ ] Optional calibration: `tools/autotune_linsys.py` for indy7 and iiwa14 N64,
  using a NEW `--tuning-path` in the result directory; retain PCG/BDSV and auto
  validation. Do not silently change controller defaults during baseline runs.
  The Fig-3 harness pins PCG and therefore does not validate tuned auto behavior.

Allow provisionally up to two hours for the focused batch plus separately
agreed compiler/calibration legs; this is not a measured duration or timeout.
Split work into exclusive slots around other agents rather than overlapping
timing jobs. Full figure regeneration is NOT included.

## 4. Research work that does not block this core merge

- Full Fig-3 sweep and matched competitor refresh, Fig-4 grid reconciliation,
  and Fig-5 quality/latency protocol work before new public result claims.
- Full Fig-7 / Table-I refresh and its unresolved success/metric gap.
- Go2 base-tracking cost, pre-liftoff support plan, S2 weight shift and S3 lift.
- One-TU compile-layout experiments, stream/graph optimizations and other
  performance A/Bs. Each needs its own correctness gate and measured decision.
- Hardware re-evaluation, binary distribution and a release tag/version decision.
- Standalone sdist builds: the current archive contains owned sources but not
  external GRiD/GLASS dependency trees. Support this only after a fresh unpacked
  archive actually compiles and runs, or retain the Git-checkout-only contract.

See [feature status](status.md) and [figure protocols](../examples/paper-figures/README.md).

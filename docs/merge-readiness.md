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
- [x] Fresh development candidate `d91ab5a`: separate checkout + lean venv,
  anonymous pinned recursive submodules, byte-identical regeneration, capped
  indy7 N64 CUDA build and examples 01/02 passed. No Pinocchio installed.
  Repeat if final integration changes the install/build path; this is not a
  pre-emptive validation of a future merge commit.
- [x] Website deployed October 1 after the main merge (published results labeled,
  current-code results added, quick start on `main`); see docs/website.md.
- [x] Reconcile preferred citation metadata with main's ICRA citation; preserve
  the funding acknowledgement.
- [x] Removed the temporary quick-start branch selector and switched package
  documentation URLs and the GRiD submodule branch to `main` (October 1).

## 2. Correctness and integration

- [x] Bounded masked-contact racecheck: AL/barrier/ADMM cases, one SQP/ADMM
  iteration each, zero errors/warnings (2026-09-27). The earlier full ADMM
  golden diagnostic remains incomplete; do not claim exhaustive race coverage.
- [x] Reconcile floating effort-vector width/documentation: native upload and
  binding now use ACTUATED_SIZE; default/fc KKT and scalar-reset gates pass.
- [x] Fix invalid spherical-payload quaternion initialization and augmented
  robot-velocity layout; preserve old pools and label corrected Fig-7 protocol
  `unit-quaternion-pendulum-v2`. The two overnight failures reproduce with the
  old initialization and both succeed with the correction, twice deterministically.
- [x] Full signed receipt for source `78fc5d4`: 341 passed, zero skips on the
  18-module profile (2026-09-27); all 38 goldens passed. Refresh again if later
  fingerprinted sources change, and require green receipt/CPU CI before merge.
- [x] Follow-up source `d91ab5a`, receipt `3b1a051`: 356 passed, zero skips,
  all 38 goldens unchanged. Floating effort-vector memcheck: two passes, zero
  errors. No new timing measurements; see the September 27 checkpoint review.
- [x] Merged remote main (`eb025dd`, citation and funding README text) on October 1
  and re-attested; GRiD pinned to its `main`. Original item: compare with the latest remote main, preserve its independent changes,
  review API removals and generated/pinned dependency changes, and test the
  resulting merge candidate. Do not blindly squash/rebase away the receipt's
  attested ancestor; re-attest the resulting commit if history/sources change.

## 3. Focused quiet-window checkpoint

The first batch completed on September 27: see the
[checkpoint review](checkpoint-2026-09-27.md). The next targeted measurement is
`--suite seed-ab` from the [timing handoff](timing-handoff.md), not a blind rerun
of all suites. Reserve provisionally 15–30 minutes after correctness preparation.

Ready to preview; follow-up timing is NOT scheduled or authorized here:

```bash
examples/benchmarks/run_timing_handoff.sh --suite seed-ab --dry-run
# Only after the user declares an exclusive slot and correctness is green:
GATO_QUIET_WINDOW=1 bash examples/benchmarks/run_timing_handoff.sh --suite seed-ab
```

The runner is GATO-only: three independent Fig-3 N64 / B1,8,128 repeats and
ten seeded Fig-7 B128 scenarios. It refuses a dirty source tree, requires an
idle box before/between legs, does not kill other agents, and writes unique
logs/CSVs/tags. Preserve the output directory and verify the box stayed quiet
throughout each leg (boundary checks cannot detect every transient workload).
It records binary hashes, source/pins, receipt and environment; no builds or
autotuning are hidden inside this runner.

- [x] Compare matching iiwa14 cells with the matching prior iiwa14 CSVs, including
  repeated-run variation: September medians +41%/+37%/+13% at B1/8/128, but the
  seed changed from zero-tail to hold during harness migration. Attribution open.
- [x] Run controlled seed-policy A/B on frozen inputs (September 30; see the
  checkpoint review). The seed explains nearly all of the gap; residuals of about
  3% at B8 and 4% at B128 stay unattributed until a matched source A/B.
- [x] Check numerical quality alongside latency: both seeds pass 400-step raw/
  controller parity at B1/8/128. Corrected pick-place v2 is deterministic 8/10
  (scenarios 3/9 fail at goal five); preserve those limitations. The Fig-3 sweep follows the
  predicted state, not an independent plant; use the fixed-pacing regression
  gates and matched tracking harness when diagnosing changed results.
- [x] Report internal solver latency as such. Full Python/controller latency
  and new cross-solver speedup claims need a matched measurement boundary.
- [x] Separately measure cold/no-op compilation, peak process RSS and binary size in a
  fresh temporary CMake tree with separate `GATO_MODULE_OUTPUT_DIR`, explicit
  arch 120 on this box, receipt profile OFF and demo OFF. Use the project Python
  and pybind11; MODULES=`indy7:16;iiwa14:64`; one target at a time under the
  36 GiB / no-swap / two-job systemd cap. Preserve production binaries.
- [x] Optional calibration: `tools/autotune_linsys.py` for indy7 and iiwa14 N64,
  using a NEW `--tuning-path` in the result directory; retain PCG/BDSV and auto
  validation. Do not silently change controller defaults during baseline runs.
  The Fig-3 harness pins PCG and therefore does not validate tuned auto behavior.

The completed suites took 127.4 s / 64.3 s / 45.3 s. Future reservations still
include headroom and preflight; split exclusive slots around other agents.
No timings have been collected for the follow-up fixes. Full figure
regeneration is NOT included.

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

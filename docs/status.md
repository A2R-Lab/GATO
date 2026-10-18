# Feature status and limitations

Scope: the source revision containing this document. “Covered” means specific committed tests exercise a capability; it is not a guarantee for every problem,
hardware configuration or combination of features. This remains source-installed
research software, not a certified robot controller.

| Capability | Evidence / supported scope | Boundary |
|---|---|---|
| Batched fixed-base SQP, PCG/BDSV, explicit warm starts | Indy7 and iiwa14 receipt modules at every paper horizon, FR3, Gen3 and UR5e at N64; numerical goldens and dynamics fingerprints on all five arms; KKT and solve tests | The three new arms have an N64 frozen holdout in [constraints](constraints.md#new-arm-holdout-on-october-18). Other horizons and tuning need their own validation |
| Python solver/controller/policy APIs | Shape, controller, hypothesis and policy tests; introductory examples; the controller's device path (CUDA torch / JAX inputs, resident warm start) gated bit-identical to the host path over a closed loop | Raw solves retain dual/rho state but do not implicitly retain the trajectory; a hypothesis batch scores on the host |
| URDF code generation and native builds | Codegen freshness and custom-robot build test | NVIDIA CUDA required; native modules are not shipped in a universal wheel |
| Boxes, EE rows, linear/conic control rows, collision and contact-position rows | Finite-difference, residual, mask and mechanism gates; closed-loop ADMM velocity-box gate on the simulated plant (`test_limits_closed_loop.py`, example 09) | Finite-budget AL/ADMM solves need not be feasible; inspect violations. Enforcement needs a horizon that can brake (N ≥ 32 on the figure-8) |
| Contact-force (`fc`) and exact-Hessian (`eh`) variants | Named N16 arm variants in the receipt | Separate variants, not an arbitrary combined fc+eh configuration; exact Hessian is workload-dependent |
| Go2 floating-base dynamics and contact-force standing | N16 default/fc modules, manifold/derivative tests, MuJoCo standing gates | Longer horizons are not supported by the receipt; standing is not walking |
| Gait schedule, foot masks and swing targets | Solver-level foot-lift and programmer-plumbing tests | Schedule is supplied, not discovered; closed-loop S2 weight shift / S3 lift remain in development |
| Runtime and task performance | [Figure refresh](figure-refresh-2026-10-01.md): seed A/B, Fig-3 (iiwa14 and Indy7), Fig-4, Fig-5, Fig-7 at their recorded revisions | Those ratios are not the paper's speedups or measurements of the latest controller defaults; Fig-7 is reported on two task settings |

## Important operating limits

- **Go2 locomotion:** the next design work is base position/orientation/velocity
  tracking and a support-polygon plan before liftoff, then S2 and S3 gates.
  Re-anchoring stance targets to measured feet prevents stale prediction targets;
  independent world-frame drift/slip measurements are still necessary.
- **Floating vector weights:** effort weights have length `n_actuated` (Go2:
  12), excluding base pose and contact-wrench slots. Posture targets/weights
  remain stored-q indexed (`nq`, Go2: 19); these are different contracts.
- **Pick-place results:** pools before the September 27 simulator fix
  (`unit-quaternion-pendulum-v2`) are not comparable with current runs or the
  paper. Current numbers are in the figure refresh. Its October 4 holdout
  confirms strong batched arrival rates but zero episodes satisfying all
  measured joint limits, including with applied-torque clamping.
- **Actuation:** optimized contact wrenches explain model contact forces; they
  are not actuator commands. Apply `StepResult.u` or `SolveResult.u0()` only.
  The raw solver installs no limit rows. The fixed-base controller installs
  ADMM boxes by default when its solver has no row groups, but this is not a
  torque clamp or a hardware-side safety layer.
  An [experiment-only clipping comparison](constraints.md#torque-saturation-development-comparison)
  preserved goal completion under brief FR3 and UR5e saturation; Gen3 did
  not saturate in that development sample. Clipping is not enabled by default.
  A [persistent Gen3 derating study](constraints.md#sustained-gen3-actuator-derating)
  reaches goals at half nominal torque but includes a fourfold velocity
  excursion; stronger derating triggers numerical escape guards.
- **Constraints:** the controller installs the URDF limit boxes under ADMM
  by default on fixed-base arms (`MPCController(limits=...)`; the raw solver
  installs nothing); follow row-group ordering and check feasibility
  telemetry (`StepResult.row_viol`). A report-only row is not an enforced
  bound, and a finite-budget ADMM plan can still violate bounds. Inspect
  the returned row violations and residuals. The October 16 Indy7/iiwa14 and October 18
  FR3/Gen3/UR5e holdouts in [constraints](constraints.md) show lower velocity
  peaks than unconstrained control, not universal feasibility. At N64 the
  new-arm default reaches every pick-and-place goal, but FR3 commanded torque
  peaks at 1.31 times its limit and Gen3 sampled velocity at 1.42 times its
  limit. Position and velocity are sampled at control ticks; these aggregates
  do not bound violations between samples or their duration.
  [Substep diagnostics](constraints.md#velocity-prediction-diagnosis) identify
  both finite-solve dynamics inconsistency and planner/simulator integration
  mismatch. Small box-row violations do not bound either error.
- **Integration:** default-stream execution is the supported path; do not
  assume concurrent solves on a shared solver instance are supported.
- **Timing:** internal solve time and `MPCController.step` wall time have
  different boundaries. Neither includes the entire robot cycle; small median
  durations do not establish deadline compliance. `step(..., budget_s=...)`
  bounds the solver's SQP iterations to the budget (to within one iteration,
  measured on the host); it is not a real-time guarantee for the loop around it. The dated October 6 results
  describe their recorded source and dependency revisions, not the current
  dependency pin, default constraints or device-resident controller path.
- **Dependencies:** MuJoCo remains pinned below 3.14 pending verification of a
  version fixing the observed URDF-compilation crash. Do not bypass that pin
  based solely on a newer version number.

## What the receipt proves

The committed `gpu-proof.json` attests the full suite on the receipt module set
at the commit it names. The full test inventory is tracked in
`test/expected_tests.txt`; the numerical contract includes 48 bitwise goldens.
`CHANGELOG.md` records major changes. Sanitizer evidence is bounded:
memcheck on the effort-vector and masked-contact gates and racecheck on a
one-iteration masked-contact gate passed with zero errors, which covers the
exercised kernels, not every solver execution. This is correctness and install
evidence, not a performance result.

See the [receipt contract](consumer_contract.md#5-what-the-gpu-proof-receipt-does-and-does-not-attest)
and [paper-reproduction policy](../examples/paper-figures/README.md) before
making broader correctness, performance or reproduction claims.

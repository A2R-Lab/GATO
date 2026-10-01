# Feature status and limitations

Scope: the modernization branch, reviewed 2026-09-27. “Covered” means specific
committed tests exercise a capability; it is not a guarantee for every problem,
hardware configuration or combination of features. This remains source-installed
research software, not a certified robot controller.

| Capability | Evidence / supported scope | Boundary |
|---|---|---|
| Batched fixed-base SQP, PCG/BDSV, explicit warm starts | Indy7 and iiwa14 receipt modules, numerical goldens, KKT and solve tests | New robots, horizons and tuning need their own validation |
| Python solver/controller/policy APIs | Shape, controller, hypothesis and policy tests; introductory examples | Raw solves retain dual/rho state but do not implicitly retain the trajectory |
| URDF code generation and native builds | Codegen freshness and custom-robot build test | NVIDIA CUDA required; native modules are not shipped in a universal wheel |
| Boxes, EE rows, linear/conic control rows, collision and contact-position rows | Finite-difference, residual, mask and mechanism gates | Finite-budget AL/ADMM solves need not be feasible; inspect violations |
| Contact-force (`fc`) and exact-Hessian (`eh`) variants | Named N16 arm variants in the receipt | Separate variants, not an arbitrary combined fc+eh configuration; exact Hessian is workload-dependent |
| Go2 floating-base dynamics and contact-force standing | N16 default/fc modules, manifold/derivative tests, MuJoCo standing gates | Longer horizons are not supported by the receipt; standing is not walking |
| Gait schedule, foot masks and swing targets | Solver-level foot-lift and programmer-plumbing tests | Schedule is supplied, not discovered; closed-loop S2 weight shift / S3 lift remain in development |
| Runtime and compilation performance | [October 1 figure refresh](figure-refresh-2026-10-01.md): seed A/B, Fig-3/4/5 on the current code | Seed policy explains the September gap except about 3–4% at B8/B128; current ratios are not the paper's speedups |

## Important operating limits

- **Go2 locomotion:** the next design work is base position/orientation/velocity
  tracking and a support-polygon plan before liftoff, then S2 and S3 gates.
  Re-anchoring stance targets to measured feet prevents stale prediction targets;
  independent world-frame drift/slip measurements are still necessary.
- **Floating vector weights:** effort weights have length `n_actuated` (Go2:
  12), excluding base pose and contact-wrench slots. Posture targets/weights
  remain stored-q indexed (`nq`, Go2: 19); these are different contracts.
- **Pick-place results:** the September 27 review found invalid historical
  pendulum quaternion initialization. Corrected pools use protocol
  `unit-quaternion-pendulum-v2`; old success rates are not interchangeable with
  corrected simulations or the paper. The corrected ten-scenario sample succeeds 8/10
  (scenarios 3 and 9 fail at goal five).
- **Actuation:** optimized contact wrenches explain model contact forces; they
  are not actuator commands. Apply `StepResult.u` or `SolveResult.u0()` only.
  Torque limits are not enforced by default; explicitly choose limits and a
  hardware-side safety layer before driving a robot.
- **Constraints:** select a mechanism explicitly, follow row-group ordering,
  and check feasibility telemetry. A report-only row is not an enforced bound.
- **Integration:** default-stream execution is the supported path; do not
  assume concurrent solves on a shared solver instance are supported.
- **Dependencies:** MuJoCo remains pinned below 3.14 pending verification of a
  version fixing the observed URDF-compilation crash. Do not bypass that pin
  based solely on a newer version number.

## What the receipt proves

Latest follow-up: source `d91ab5a`, receipt commit `3b1a051`, **356 passed,
zero skips**, with all 38 existing bitwise goldens unchanged. New gates cover
explicit controller seeds, pendulum state assembly, floating effort-vector
width and codegen fallback refusal. Bounded effort-vector memcheck passed both
default/fc cases with zero errors. Fresh lean onboarding also built indy7 N64
and ran examples 01/02 without Pinocchio. This is correctness/install evidence,
not a new performance result. Corrected pick-place v2 remains 8/10 on the ten
seeded diagnostic cases (different failures from the old invalid simulation).

The source commit `4c8ab22` was attested by receipt commit `5b309d3`: 328 tests,
zero skips, including 38 bitwise goldens across the 18-module profile. This is a
dated checkpoint. The 2026-09-27 documentation/example/test update (`78fc5d4`)
passed 341 tests with zero skips and the same 38 goldens; inspect the current
`gpu-proof.json` for the latest coverage.
The two masked-contact golden cases passed memory checking. The original full
ADMM race check was interrupted. A subsequent bounded masked-contact gate (one
SQP/ADMM iteration, AL/barrier/ADMM) passed racecheck with zero errors/warnings;
this covers the exercised kernels, not every possible solver execution.

See the [receipt contract](consumer_contract.md#5-what-the-gpu-proof-receipt-does-and-does-not-attest)
and [paper-reproduction policy](../examples/paper-figures/README.md) before
making broader correctness, performance or reproduction claims.

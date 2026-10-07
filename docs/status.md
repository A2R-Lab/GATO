# Feature status and limitations

Scope: the source revision containing this document. “Covered” means specific committed tests exercise a capability; it is not a guarantee for every problem,
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
| Runtime and task performance | [Figure refresh](figure-refresh-2026-10-01.md): seed A/B, Fig-3 (iiwa14 and Indy7), Fig-4, Fig-5, Fig-7 on the current code | Current ratios are not the paper's speedups; Fig-7 is reported on two task settings |

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
  Torque limits are not enforced by default; explicitly choose limits and a
  hardware-side safety layer before driving a robot.
- **Constraints:** select a mechanism explicitly, follow row-group ordering,
  and check feasibility telemetry. A report-only row is not an enforced bound.
- **Integration:** default-stream execution is the supported path; do not
  assume concurrent solves on a shared solver instance are supported.
- **Timing:** internal solve time and `MPCController.step` wall time have
  different boundaries. Neither includes the entire robot cycle; small median
  durations do not establish deadline compliance. The dated October 6 results
  were collected on the current GRiD/GLASS pins.
- **Dependencies:** MuJoCo remains pinned below 3.14 pending verification of a
  version fixing the observed URDF-compilation crash. Do not bypass that pin
  based solely on a newer version number.

## What the receipt proves

The committed `gpu-proof.json` attests the full suite on the receipt module set
at the commit it names. The full test inventory is tracked in
`test/expected_tests.txt`; the numerical contract includes 42 bitwise goldens.
`CHANGELOG.md` records major changes. Sanitizer evidence is bounded:
memcheck on the effort-vector and masked-contact gates and racecheck on a
one-iteration masked-contact gate passed with zero errors, which covers the
exercised kernels, not every solver execution. This is correctness and install
evidence, not a performance result.

See the [receipt contract](consumer_contract.md#5-what-the-gpu-proof-receipt-does-and-does-not-attest)
and [paper-reproduction policy](../examples/paper-figures/README.md) before
making broader correctness, performance or reproduction claims.

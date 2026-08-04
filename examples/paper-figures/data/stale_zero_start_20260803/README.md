# STALE: pick-place pools started from the all-zeros (singular) configuration

Rotated 2026-08-03. Every pool here ran with `start_config='home'`, which for
iiwa14 is **all-zeros** — the arm extended straight up. That pose is a kinematic
singularity, and a consequential one for these experiments:

- a VERTICAL end-effector force produces zero joint torque there
  (`|J^T w|` = 1.4e-6 vs > 1 once the arm bends), so a hanging payload is
  **completely unobservable** to any wrench estimator until the arm moves away;
- manipulability is 0.0 and cond(J) ~ 2e10, so the arm starts with no usable
  Cartesian authority in any direction.

`IIWA14_START_CONFIGS['ready']` (a mid-workspace elbow pose, signs alternating
down the arm) replaces it for closed-loop experiments. `'zero'`/`'home'` are
unchanged — they remain the inputs of record for the bitwise parity baseline and
the test suite, which is why a new key was added rather than the old ones
redefined.

## Contents — the W3 arm comparison of record for the zero start

Protocol: eased (mass 10 kg, L 0.5-0.7, damping 0.3-0.6, angle 0.0-0.6, seed 0),
100 scenarios, identical scenario list across pools (paired).

| pool | arm | success | mean goals / 5 |
|------|-----|---------|----------------|
| `fig7_pickplace_eased.pkl` | ForceEstimator, B = 1..128 | 4 / 11 / 82 / 87 / 87 / 87 / 86 | B=1 2.55, B=128 4.86 |
| `fig7_fc.pkl` | fc-as-control, B=1 | 6/100 | 2.58 |
| `fig7_wid.pkl` | wrench-ID, instantaneous | 10/100 | 3.10 |
| `fig7_wid_tau01.pkl` | wrench-ID, weight_tau=0.1 | 27/100 | 3.80 |

Those relative comparisons are internally valid (all four share the start config
and the scenario list) and the conclusions drawn from them stand. What is stale
is their absolute difficulty: **do not mix with post-'ready' pools.**

## Why the numbers move

At the `ready` start the task is easier at high batch — the arm begins closer to
every goal (0.41-0.94 m vs 0.56-1.19 m) and with 6x the manipulability. On a
25-scenario probe at the eased protocol, B=128 went 92% -> 96%, out of the
80-90% target band.

★ Which exposed the more interesting fact: the protocol easing (15 kg -> 10 kg)
was substantially compensating for the bad start pose. At `ready`, the
**unmodified 15 kg paper protocol** (L 0.3-0.7, damping 0.1-0.6) scores 0 / 64 /
88% at B = 1 / 8 / 128 on 25 scenarios — back in the target band with no easing
at all, versus 59% at B=128 from the zero start. Post-rotation pools therefore
run the stated paper protocol, and the "eased protocol has no paper-era
implementation" caveat no longer needs to carry the result.

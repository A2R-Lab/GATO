# The cost menu

GATO's cost API since 0.2.0. A `CostMenu` replaces the tracking objective of
the solver (the scalar weights `q_cost`, `qd_cost`, `u_cost`, `N_cost`, the
per-knot weight triples, the posture anchor and the end-effector reference)
with up to eight terms, each a residual under an activation, evaluated by
GRiD's generated `grid_plant::tracking_cost_menu` (grid-rbd 0.5.4). The menu
is runtime data, like the constraint row groups: it is built in Python, applied
to a solver, updated by applying it again, and cleared. The solver's
`solve(x, ref, ...)` keeps its reference argument; the menu does not read it.

## Terms

| term | residual | size | frame |
|---|---|---|---|
| `frame_pos` | frame position minus target | 3 | cost frame 0, the end effector |
| `state` | x minus target, x = [q; qd] | nx | |
| `input` | u minus target, the actuated torques | nu | |
| `linear_x` | A x + b minus target, A of shape nx by nx | nx | |
| `linear_u` | A u + b minus target, A of shape nu by nu | nu | |
| `orientation` | log(R_des^T R_frame), the rotation error as a rotation vector | 3 | cost frame 0 |
| `velocity` | the frame twist J(q) qd minus target, linear then angular, world axes at the frame point | 6 | cost frame 0 |
| `frame_pos_2` | a second frame position minus target | 3 | cost frame 1 |

A term is a `Term(weight, target, activation, eps, lower, upper)`; the two
linear features are `LinearTerm`s with their `A` and `b`. The weight is per
entry and nonnegative; a zero weight turns an entry off. The target of
`orientation` is a rotation matrix; the target of `input` may be the string
`"gravity"` together with `CostMenu.q_ref`, the reference configuration per
knot, and then u_des is the generalized gravity g(q_ref), evaluated once on
the device when the menu is applied, so holding the arm against gravity is not
priced as effort.

Every target, weight, bound and parameter is per knot. A value given with the
term's size is used at every knot, one of shape (N, size) is per knot and
shared across the batch, one of shape (B, N, size) is per solve; only the
fields given per solve are stored per solve. Matrices take the shapes
(size, size), (N, size, size) or (B, N, size, size).

Activations, per term and per knot (one name, or one name per knot):

| activation | a(r) | parameter |
|---|---|---|
| `quadratic` | w r^2 / 2 | |
| `pseudo_huber` | w (sqrt(r^2 + eps^2) - eps): quadratic below eps, linear above | `eps` > 0, the scale |
| `relaxed_log_barrier` | the log barrier on the feature value inside [lower, upper], with the quadratic extension of Grandia et al. below the relaxation distance | `eps` > 0, the relaxation distance; `w` the barrier weight |
| `quadratic_barrier` | w (max(0, lower - v)^2 + max(0, v - upper)^2) / 2 on the feature value v | |
| `none` | the term is off at that knot | |

The two barrier kinds act on the feature value, not on the residual, and need
`lower` and `upper` (an infinite bound disables that side); `frame_pos` and
`orientation` have no bounds. The gradient of every term is J^T a'(r) and its
Hessian the Gauss-Newton J^T diag(a''(r)) J, so the KKT blocks stay positive
semidefinite under every activation.

## What stays

The joint-limit barriers of the tracking objective (`q_lim_cost`,
`vel_lim_cost`, `ctrl_lim_cost`, with the plant's limit tables) stay in force
under the menu, as do the constraint row groups (`docs/constraints.md`): the
menu's barrier activations are cost terms, the row groups are the constraint
layer. The terminal knot has no control, so `input` and `linear_u` are off
there and the torque barrier too, as in the tracking objective. While a menu is
on, the setters of the tracking objective (`set_cost_weights`,
`set_cost_weights_per_knot`, `set_q_pos_cost`, `set_q_nom`, `set_u_cost_vec`)
raise, so a weight cannot be silently ignored or counted
twice; `clear_cost_menu()` returns to the tracking objective with those
settings intact.

## Where it runs

The setup-KKT and merit kernels carry a second instantiation for the menu. The
default instantiation keeps its code, its register allocation and its launch,
and the host picks by the menu state, so a solver without a menu runs the
kernels it ran before (the goldens are unchanged). With the menu on, the
kernels evaluate the eight terms through the generated preset in the same
block; the shared-memory carve grows to the frame evaluation arena. The
equivalence of the menu with quadratic activations and the tracking objective
holds to rounding, not bitwise: the end-effector Jacobian comes from
`frame_jacobian`, a different arithmetic order than the pose gradient, so the
KKT blocks agree to about 1e-6 relative and a solve can pick another
line-search candidate on a tie after a few iterations
(`test/test_cost_menu.py`).

## Availability

grid-rbd 0.5.4 emits the menu for fixed-base plants, so the five vendored arms
and any registered fixed-base robot carry it (`BSQP.cost_menu_available`,
module attribute `COST_MENU_AVAILABLE`). The contact-force variant widens the
control past the menu's input size and does not carry it. The Go2 and other
floating-base plants wait for the floating composition of the next GRiD batch;
on them a standing or base-pose reference goes through the tracking
objective's posture anchor, whose first six weights are the base-pose tangent
weights (`set_q_pos_cost` with an array, `set_q_nom`; example 19). The menu was first
measured on October 23 for correctness only; its cost per iteration against
the tracking objective is a quiet-window item.

## Cost frames

The end-effector target of a plant is always cost frame 0. Extra frames (an
elbow, a camera, any articulated or fixed joint of the URDF, by its URDF name)
are baked at codegen time: `gato.build(..., cost_frames=["A4"])` or
`gato.register_robot(..., cost_frames=[...])`; they become cost frame 1, 2,
... in the order given, and the `frame_pos_2` term addresses cost frame 1. The
vendored arms carry frame 0 only. A joint attached to the world root cannot be
a cost frame (it does not move). The frame list is part of the codegen key.

## Reading the cost

`BSQP.cost_menu_terms(xu)` returns the value of every term along a trajectory,
an array (B, N, 8) in `gato.cost_menu.TERMS` order with each term evaluated
alone, for reporting and tests. It is a readout, never part of the solve.

## Examples and tests

Examples 16 to 19 (`examples/16_cost_menu.py`, the tracking objective as a
menu and what the default cannot do; `17_tool_pose_and_posture.py`, an
orientation target with a held elbow and a gravity-relative effort cost;
`18_robust_reference.py`, quadratic against pseudo-Huber on a reference with
outliers; `19_go2_standing.py`, the base-pose weights on the Go2).
`test/test_cost_menu.py` holds the shape and validation rules, the equivalence
gate, every term against RBDReference's menu oracle in the assembled KKT
blocks and in the per-term values, the per-solve independence of the batch,
the setter refusal and the gravity target against Pinocchio;
`test/test_codegen.py` the cost-frame baking.

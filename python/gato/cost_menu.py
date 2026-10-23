"""The cost menu: GATO's cost API over grid_plant::tracking_cost_menu (grid-rbd 0.5.4).

A :class:`CostMenu` replaces the solver's tracking objective (the scalar
``q_cost``/``qd_cost``/``u_cost`` weights, the per-knot weight triples, the
posture anchor and the EE reference trajectory) with up to eight terms, each a
residual under an activation:

==============  =============================================  ======  =======
term            residual                                       size    frame
==============  =============================================  ======  =======
``frame_pos``   frame position minus target                    3       0 (EE)
``state``       x minus target, x = [q; qd]                    nx
``input``       u minus target (the actuated torques)          nu
``linear_x``    A x + b minus target (A nx by nx)              nx
``linear_u``    A u + b minus target (A nu by nu)              nu
``orientation`` log(R_des^T R_frame)                           3       0 (EE)
``velocity``    frame twist J(q) qd minus target               6       0 (EE)
``frame_pos_2`` second frame position minus target             3       1
==============  =============================================  ======  =======

Activations (per term, per knot): ``quadratic`` (1/2 w r^2), ``pseudo_huber``
(w (sqrt(r^2 + eps^2) - eps)), ``relaxed_log_barrier`` (the log barrier on the
feature value inside ``[lower, upper]`` with the quadratic extension below
``eps``) and ``quadratic_barrier`` (1/2 w times the squared bound violation).
The two barrier kinds act on the feature value, not on the residual, and
need ``lower``/``upper``. Every target, weight, bound and parameter is per
knot: pass one value for every knot (shape ``(size,)``), one per knot
(``(N, size)``) or one per solve and knot (``(B, N, size)``); only the fields
given per solve are stored per solve. The joint-limit barriers of the default
objective (``q_lim_cost``, ``vel_lim_cost``, ``ctrl_lim_cost``) stay in force
under the menu with the plant's limit tables. The terminal knot has no
control: ``input`` and ``linear_u`` are off there.

The menu is runtime data like the constraint row groups: build it, apply it
with :meth:`gato.BSQP.set_cost_menu`, update targets by applying again (no
allocation after the first application for the same shapes), and remove it
with :meth:`gato.BSQP.clear_cost_menu`. The solver's ``solve(x, ref, ...)``
still takes the EE reference argument; the menu does not read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

ACTIVATIONS = {"none": -1, "quadratic": 0, "pseudo_huber": 1, "relaxed_log_barrier": 2, "quadratic_barrier": 3}
BARRIER_KINDS = ("relaxed_log_barrier", "quadratic_barrier")
EPS_KINDS = ("pseudo_huber", "relaxed_log_barrier")
TERMS = ("frame_pos", "state", "input", "linear_x", "linear_u", "orientation", "velocity", "frame_pos_2")

# field names of each term (gato::menu::FIELD_NAMES): target, weight, lower, upper; the linear
# features add their A and b, the orientation has no bounds
_FIELDS = {
    "frame_pos":   dict(target="ee_des", weight="W"),
    "state":       dict(target="x_des", weight="Q", lower="x_lower", upper="x_upper"),
    "input":       dict(target="u_des", weight="R", lower="ub_lower", upper="ub_upper"),
    "linear_x":    dict(target="tx", weight="Wx", lower="lx_lower", upper="lx_upper", A="Ax", b="bx"),
    "linear_u":    dict(target="tu", weight="Wu", lower="lu_lower", upper="lu_upper", A="Au", b="bu"),
    "orientation": dict(target="R_des", weight="Wo"),
    "velocity":    dict(target="v_des", weight="Wv", lower="v_lower", upper="v_upper"),
    "frame_pos_2": dict(target="cf_des", weight="Wcf", lower="cf_lower", upper="cf_upper"),
}


@dataclass
class Term:
    """One menu term.

    weight: per-entry weight (nonnegative), broadcast to the term's size.
    target: the residual's target; None means zeros. For ``orientation`` the
        target is a rotation matrix ``R_des`` (3 by 3, per knot or per solve),
        for ``input`` it may be the string ``"gravity"`` together with
        :attr:`CostMenu.q_ref` (``u_des = g(q_ref)`` per knot).
    activation: one activation name for every knot or one name per knot (N,).
    eps: the activation parameter (pseudo-Huber scale, relaxed-barrier
        relaxation distance), a scalar or one per knot; positive where used.
    lower / upper: the barrier activations' bounds on the feature value
        (``+-inf`` disables one side); None for the other activations.
    """
    weight: Any
    target: Any = None
    activation: str | Sequence[str] = "quadratic"
    eps: Any = 0.0
    lower: Any = None
    upper: Any = None


@dataclass
class LinearTerm(Term):
    """A linear feature ``A v + b`` on x (``linear_x``) or u (``linear_u``):
    ``A`` is (size, size) for every knot, (N, size, size) or (B, N, size, size);
    ``b`` broadcasts like a target (None = zeros). A row is off with a zero weight."""
    A: Any = None
    b: Any = None


@dataclass
class CostMenu:
    """The eight terms (None = off) and the gravity reference ``q_ref`` (N, nq)
    for ``input.target == "gravity"``."""
    frame_pos: Term | None = None
    state: Term | None = None
    input: Term | None = None
    linear_x: LinearTerm | None = None
    linear_u: LinearTerm | None = None
    orientation: Term | None = None
    velocity: Term | None = None
    frame_pos_2: Term | None = None
    q_ref: Any = None

    def terms(self) -> dict:
        return {t: getattr(self, t) for t in TERMS if getattr(self, t) is not None}

    # ---- upload ------------------------------------------------------------
    def apply(self, solver) -> None:
        """Validate the menu against ``solver`` and upload it (the solver's
        ``set_cost_menu`` calls this). Shapes: see the module docstring."""
        lib = solver.lib
        if not bool(getattr(lib, "COST_MENU_AVAILABLE", False)):
            raise ValueError("the cost menu is not available in this module: grid-rbd 0.5.4 emits it for "
                             "fixed-base plants, and the contact-force variant does not carry it")
        sizes = dict(lib.COST_MENU_FIELDS)
        N, B = int(solver.N), int(solver.batch_size)
        n_frames = int(getattr(lib, "NUM_COST_FRAMES", 0))
        terms = self.terms()
        if not terms:
            raise ValueError("an empty cost menu: give at least one term, or clear_cost_menu() to use the tracking objective")
        if self.frame_pos_2 is not None and n_frames < 2:
            raise ValueError("frame_pos_2 needs a second baked cost frame: build the plant with cost_frames=[...]")
        activation = np.full((N, len(TERMS)), ACTIVATIONS["none"], dtype=np.int32)
        eps = np.zeros((N, len(TERMS)), dtype=np.float32)
        uploads = []   # (field, flat, per_solve)
        for ti, name in enumerate(TERMS):
            term = terms.get(name)
            if term is None:
                continue
            f = _FIELDS[name]
            size = int(sizes[f["weight"]])
            kinds = _activation_table(name, term.activation, N)
            activation[:, ti] = [ACTIVATIONS[k] for k in kinds]
            eps[:, ti] = _per_knot_scalar(name + ".eps", term.eps, N)
            needs_eps = [k in EPS_KINDS for k in kinds]
            if any(needs_eps) and np.any(eps[needs_eps, ti] <= 0):
                raise ValueError(f"{name}: eps must be positive at every knot using pseudo_huber or relaxed_log_barrier")
            is_barrier = [k in BARRIER_KINDS for k in kinds]
            if any(is_barrier):
                if "lower" not in f:
                    raise ValueError(f"{name}: the barrier activations are not defined for this term (it has no bounds)")
                if term.lower is None or term.upper is None:
                    raise ValueError(f"{name}: the barrier activations need lower and upper bounds")
            w, w_ps = _broadcast(name + ".weight", term.weight, size, N, B)
            if np.any(~np.isfinite(w)) or np.any(w < 0):
                raise ValueError(f"{name}: weights must be finite and nonnegative")
            uploads.append((f["weight"], w, w_ps))
            if name == "orientation":
                tgt, t_ps = _rotations(name + ".target", term.target, N, B)
            elif name == "input" and isinstance(term.target, str):
                if term.target != "gravity":
                    raise ValueError("input.target must be an array or the string 'gravity'")
                if self.q_ref is None:
                    raise ValueError("input.target == 'gravity' needs CostMenu.q_ref (N, nq): the reference configuration")
                tgt = None
            else:
                tgt, t_ps = _broadcast(name + ".target", 0.0 if term.target is None else term.target, size, N, B)
            if tgt is not None:
                if np.any(~np.isfinite(tgt)):
                    raise ValueError(f"{name}: targets must be finite")
                uploads.append((f["target"], tgt, t_ps))
            if term.lower is not None or term.upper is not None:
                if "lower" not in f:
                    raise ValueError(f"{name}: this term has no bounds")
                lo, lo_ps = _broadcast(name + ".lower", -np.inf if term.lower is None else term.lower, size, N, B)
                hi, hi_ps = _broadcast(name + ".upper", np.inf if term.upper is None else term.upper, size, N, B)
                lo_b, hi_b = np.broadcast_arrays(lo.reshape(-1, N, size), hi.reshape(-1, N, size))
                if np.any(lo_b > hi_b):
                    raise ValueError(f"{name}: every lower bound must be at most its upper bound")
                uploads.append((f["lower"], lo, lo_ps))
                uploads.append((f["upper"], hi, hi_ps))
            if isinstance(term, LinearTerm) or "A" in f:
                if not isinstance(term, LinearTerm) or term.A is None:
                    raise ValueError(f"{name}: pass a LinearTerm with its matrix A")
                A, A_ps = _matrices(name + ".A", term.A, size, N, B)
                if np.any(~np.isfinite(A)):
                    raise ValueError(f"{name}: A must be finite")
                uploads.append((f["A"], A, A_ps))
                b, b_ps = _broadcast(name + ".b", 0.0 if term.b is None else term.b, size, N, B)
                uploads.append((f["b"], b, b_ps))
        raw = solver.solver
        raw.set_cost_menu_on(False)
        for fname, flat, per_solve in uploads:
            raw.set_cost_menu_field(fname, np.ascontiguousarray(flat, dtype=np.float32).reshape(-1), bool(per_solve))
        raw.set_cost_menu_field("eps", eps.reshape(-1), False)
        raw.set_cost_menu_activation(activation.reshape(-1))
        if self.q_ref is not None:
            q_ref = np.asarray(self.q_ref, dtype=np.float32)
            nq = int(solver.nq)
            if q_ref.shape == (nq,):
                q_ref = np.broadcast_to(q_ref, (N, nq))
            if q_ref.shape != (N, nq):
                raise ValueError(f"q_ref: expected shape ({nq},) or ({N}, {nq}), got {q_ref.shape}")
            raw.set_cost_menu_gravity_target(np.ascontiguousarray(q_ref).reshape(-1))
        raw.set_cost_menu_on(True)


# ---- shape helpers ---------------------------------------------------------

def _broadcast(what, value, size, N, B):
    """-> (flat float32, per_solve): (size,) and (N, size) are shared across the batch, (B, N, size) per solve."""
    a = np.asarray(value, dtype=np.float32)
    if a.ndim == 0:
        a = np.full((size,), float(a), dtype=np.float32)
    if a.shape == (size,):
        return np.ascontiguousarray(np.broadcast_to(a, (N, size))).reshape(-1), False
    if a.shape == (N, size):
        return np.ascontiguousarray(a).reshape(-1), False
    if a.shape == (B, N, size):
        return np.ascontiguousarray(a).reshape(-1), True
    raise ValueError(f"{what}: expected shape ({size},), ({N}, {size}) or ({B}, {N}, {size}), got {a.shape}")


def _matrices(what, value, size, N, B):
    """(size, size), (N, size, size) or (B, N, size, size) -> column-major per knot (GRiD's layout)."""
    a = np.asarray(value, dtype=np.float32)
    if a.shape[-2:] != (size, size) or a.ndim not in (2, 3, 4):
        raise ValueError(f"{what}: expected shape ({size}, {size}), ({N}, {size}, {size}) or ({B}, {N}, {size}, {size}), got {a.shape}")
    cm = np.swapaxes(a, -1, -2).reshape(a.shape[:-2] + (size * size,))
    return _broadcast(what, cm, size * size, N, B)


def _rotations(what, value, N, B):
    if value is None:
        raise ValueError(f"{what}: the orientation term needs a rotation matrix target")
    a = np.asarray(value, dtype=np.float64)
    if a.shape[-2:] != (3, 3) or a.ndim not in (2, 3, 4):
        raise ValueError(f"{what}: expected shape (3, 3), ({N}, 3, 3) or ({B}, {N}, 3, 3), got {a.shape}")
    flat = a.reshape(-1, 3, 3)
    err = np.max(np.abs(np.einsum("kij,kil->kjl", flat, flat) - np.eye(3)))
    if err > 1e-4 or np.any(np.linalg.det(flat) < 0):
        raise ValueError(f"{what}: every target must be a rotation matrix (orthonormal, determinant +1); max |R^T R - I| = {err:.2e}")
    return _matrices(what, a.astype(np.float32), 3, N, B)


def _activation_table(name, activation, N):
    if isinstance(activation, str):
        kinds = [activation] * N
    else:
        kinds = [str(k) for k in activation]
        if len(kinds) != N:
            raise ValueError(f"{name}: activation needs one name or one per knot ({N}), got {len(kinds)}")
    for k in kinds:
        if k not in ACTIVATIONS:
            raise ValueError(f"{name}: unknown activation {k!r}; one of {sorted(ACTIVATIONS)}")
    return kinds


def _per_knot_scalar(what, value, N):
    a = np.asarray(value, dtype=np.float32).reshape(-1)
    if a.size == 1:
        return np.full((N,), float(a[0]), dtype=np.float32)
    if a.size != N:
        raise ValueError(f"{what}: expected a scalar or one value per knot ({N}), got {a.size}")
    return a

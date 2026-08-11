"""Simulation worlds for the closed-loop harness.

The MPC evaluation loop (``MPC_GATO._play_control``) advances the plant one
sensor-rate substep at a time through a WORLD object::

    q_next, dq_next = world.step(q, dq, u_robot, sim_dt)

- :class:`PinocchioWorld` — the historical in-process world: pinocchio RK4 with
  optional pendulum payload and constant external wrench. It is a thin adapter
  over ``MPC_GATO``'s own machinery, so the default behavior (and every
  existing pool) is bit-identical to the pre-worlds harness.
- :class:`MuJoCoWorld` — an INDEPENDENT simulator for contact experiments: the
  same URDF loaded by MuJoCo (verified the same robot by
  ``gato.fingerprint`` — see ``test_worlds.py``), plus an optional contact
  plane. Uses MuJoCo's own integrator/contact solver: plant-model mismatch vs
  the solver's rigid fc model is deliberate.

MuJoCo gotchas this module encodes (found 2026-08-04):
- the vendored URDF's ``<mujoco><compiler meshdir=...>`` block points at a
  nonexistent dir — ``MjSpec.meshdir`` overrides it to the URDF's own dir;
- MuJoCo does NOT exclude contacts between a world-welded base and its child
  link (the parent filter skips the world body), so the intersecting base/L1
  meshes produce phantom self-contacts at some configurations that corrupt the
  dynamics (inertia-response off 100x at q=0). Robot geoms therefore get
  contype=1/conaffinity=2 and the plane contype=2/conaffinity=1: robot-robot
  never collides, robot-plane does.
"""
import os
import re

import numpy as np

from .common import rk4, _quat_to_rot


class PinocchioWorld:
    """The historical pinocchio-RK4 world, bit-identical to the pre-worlds loop.

    Constructed by ``MPC_GATO`` itself (it owns the model/data/pendulum/f_ext
    state); ``step`` preserves the exact call order of the old inline code:
    augment control -> refresh world-wrench -> rk4.
    """

    def __init__(self, mpc):
        self._mpc = mpc

    def step(self, q, dq, u, sim_dt):
        m = self._mpc
        u_aug = m._augment_control(u, dq)
        m._update_sim_f_ext(q)
        return rk4(m.model, m.data, q, dq, u_aug, sim_dt, m.actual_f_ext)


def probe_tip_drop(urdf_path, q, ref_point, span=0.20, iters=24,
                   size_xy=(0.3, 0.3)):
    """Height of a reference point (e.g. the EE FRAME origin) above the
    robot's lowest CONTACT point at configuration q, probed with MuJoCo's own
    collision checker.

    Why this exists: the EE frame origin is NOT the tool tip (~33 mm apart on
    the iiwa14 mesh). Geometry placed relative to the frame starts the tool
    in collision, and a "surface" reference is physically unreachable — a
    hidden press bias for every controller (measured 2026-08-09: +13-15 N on
    the wipe task). Probe once, define all contact geometry at the TIP.

    Bisects the highest horizontal table top (finite box under ref_point's
    xy) that does NOT touch the robot at q; returns ref_point[2] - that z.
    Exact for the evaluation world, since the probe uses the same collision
    model/masks as MuJoCoWorld. ``span``: search window below ref_point (the
    tool must clear a table span below the reference and touch one at it).
    """
    import mujoco

    def touches(z):
        w = MuJoCoWorld(urdf_path, plane={"z": float(z),
                                          "pos_xy": (float(ref_point[0]),
                                                     float(ref_point[1])),
                                          "size_xy": size_xy})
        w.data.qpos[:] = q
        w.data.qvel[:] = 0
        mujoco.mj_forward(w.model, w.data)
        return w.data.ncon > 0

    lo, hi = float(ref_point[2]) - span, float(ref_point[2])
    if touches(lo) or not touches(hi):
        raise ValueError(
            f"probe_tip_drop: bisection bracket invalid (span={span}): the "
            f"robot must clear a table at z={lo:.3f} and touch one at "
            f"z={hi:.3f} — widen span or check q/ref_point")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if touches(mid):
            hi = mid
        else:
            lo = mid
    return float(ref_point[2]) - lo


class MuJoCoWorld:
    """MuJoCo world: our URDF + an optional contact plane, MuJoCo's integrator.

    Parameters
    ----------
    urdf_path : str
        The SAME URDF the solver/pinocchio use. Model alignment is a test gate
        (``gato.fingerprint.check`` on the contact-disabled model: aligned to
        1e-7 on iiwa14).
    plane : None | dict
        ``None`` = no contact geometry (free-space experiments).
        A dict adds a horizontal TABLE surface (a finite box — an infinite
        plane at EE height would swallow the robot base 0.63 m below it and
        produce MN-scale penetration forces; found the hard way):
        ``{"z": top_surface_height}`` plus optional ``friction`` (sliding
        coefficient, default 0.6), ``pos_xy`` (center, default (0.6, 0.2) —
        under the iiwa14 'ready' workspace), ``size_xy`` (half-extents,
        default (0.25, 0.25)).
    timestep : float
        MuJoCo integrator step; ``step(sim_dt)`` asserts sim_dt matches.

    Notes
    -----
    - ``step`` writes (q, dq) into MuJoCo state each call (the caller's state
      is authoritative — it lets the harness inject disturbances), applies u as
      joint-space torque via ``qfrc_applied``, and returns copies.
    - Deterministic: same (q, dq, u) sequence -> same trajectory (MuJoCo is
      deterministic on a fixed binary/machine; the warmstart cache is reset
      each call for state-purity).
    - ``last_contact`` after each step: dict with ``ncon``, ``fn`` (total
      normal force, plane frame = world z), ``ft`` (tangential magnitude) —
      the contact-task metrics feed.
    - No pendulum / no constant-wrench support: those are PinocchioWorld
      features; MuJoCo experiments model disturbances as geometry.
    """

    def __init__(self, urdf_path, plane=None, timestep=1e-3, record_contact=False,
                 floating=False):
        import mujoco  # lazy: mujoco is an optional dependency
        self._mujoco = mujoco
        self.floating = bool(floating)

        xml = open(urdf_path).read()
        # URDFs whose VISUAL meshes are package:// references with no vendored
        # files (go2) fail MjSpec compile — strip the visuals (collision
        # geometry is untouched; go2 collisions are all primitives).
        if 'package://' in xml:
            xml = re.sub(r"<visual\b.*?</visual>", "", xml, flags=re.DOTALL)
        spec = mujoco.MjSpec.from_string(xml)
        spec.meshdir = os.path.dirname(os.path.abspath(urdf_path))
        if self.floating:
            # MuJoCo's URDF import WELDS the root body to the world; give the
            # base back its 6 dofs (freejoint => qpos [p; quat wxyz; joints])
            spec.worldbody.bodies[0].add_freejoint()
        # URDF links with NO <inertial> are MASSLESS to pinocchio/GRiD, but
        # MuJoCo synthesizes their inertia from the collision geometry
        # (density 1000) — +0.19 kg of phantom calf mass on go2 (calflower
        # cylinders), a 10-20% effective-inertia mismatch vs the solver's
        # model. Pin those bodies to explicit ~zero inertia so this world is
        # the SAME robot the solver sees (no-op on URDFs with full inertials).
        for name in re.findall(r'<link name="([^"]+)"\s*/>', xml) + [
                m.group(1) for m in re.finditer(r'<link name="([^"]+)"\s*>', xml)
                if "<inertial>" not in xml[m.end():xml.find("</link>", m.end())]]:
            body = spec.body(name)
            if body is not None:
                body.explicitinertial = True
                body.mass = 1e-9
                body.inertia = [1e-12, 1e-12, 1e-12]

        self.plane_cfg = dict(plane) if plane else None
        if self.plane_cfg is not None:
            g = spec.worldbody.add_geom()
            g.name = "contact_plane"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            xy = self.plane_cfg.get("pos_xy", (0.6, 0.2))
            sx, sy = self.plane_cfg.get("size_xy", (0.25, 0.25))
            thick = 0.02
            g.size = [sx, sy, thick]
            g.pos = [xy[0], xy[1], self.plane_cfg["z"] - thick]  # top face at z
            mu = self.plane_cfg.get("friction", 0.6)
            g.friction = [mu, 0.005, 0.0001]
            g.contype = 2
            g.conaffinity = 1

        self.model = spec.compile()
        # robot geoms: collide with the plane only, never with each other
        # (masks are set post-compile so the plane's 2/1 assignment above is
        # left untouched)
        plane_id = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      "contact_plane")
                    if self.plane_cfg is not None else -1)
        for gid in range(self.model.ngeom):
            if gid != plane_id:
                self.model.geom_contype[gid] = 1
                self.model.geom_conaffinity[gid] = 2

        self.model.opt.timestep = float(timestep)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.last_contact = {"ncon": 0, "fn": 0.0, "ft": 0.0}
        # opt-in SUBSTEP-rate contact trace (the control tick is ~30 substeps —
        # force metrics sampled per tick would alias chatter): one (ncon, fn, ft)
        # row per step() call. None when recording is off (zero overhead).
        self.contact_history = [] if record_contact else None

    @property
    def params(self):
        """World parameters for the protocol stamp (do-not-mix discipline)."""
        o = self.model.opt
        return {
            "world": "mujoco",
            "mujoco_version": self._mujoco.__version__,
            "integrator": int(o.integrator),
            "timestep": float(o.timestep),
            "solref": np.asarray(self.model.geom_solref[0]).tolist(),
            "solimp": np.asarray(self.model.geom_solimp[0]).tolist(),
            "plane": self.plane_cfg,
        }

    # ---- stored-state (pin) <-> MuJoCo conversions (floating base) --------
    #
    # Our stored state uses the pin free-flyer conventions: quat xyzw at
    # q[3:7], base velocity [v_lin; omega] BOTH in the BODY frame. MuJoCo's
    # freejoint uses quat wxyz and qvel = [v_lin in WORLD frame; omega in
    # body frame; joints]. step() converts at the boundary so the world keeps
    # the stored-state duck-type contract.

    def _pin_to_mj(self, q, dq):
        qm, vm = np.array(q, dtype=np.float64), np.array(dq, dtype=np.float64)
        quat = np.asarray(q[3:7], dtype=np.float64)
        qm[3] = quat[3]
        qm[4:7] = quat[:3]
        vm[:3] = _quat_to_rot(quat / np.linalg.norm(quat)) @ np.asarray(dq[:3])
        return qm, vm

    def _mj_to_pin(self, qm, vm):
        q, dq = qm.copy(), vm.copy()
        quat = np.array([qm[4], qm[5], qm[6], qm[3]])  # wxyz -> xyzw
        q[3:6] = quat[:3]
        q[6] = quat[3]
        dq[:3] = _quat_to_rot(quat / np.linalg.norm(quat)).T @ vm[:3]
        return q, dq

    def step(self, q, dq, u, sim_dt):
        mujoco = self._mujoco
        d = self.data
        assert abs(sim_dt - self.model.opt.timestep) < 1e-12, \
            f"sim_dt {sim_dt} != mujoco timestep {self.model.opt.timestep}"
        if self.floating:
            qm, vm = self._pin_to_mj(q, dq)
            d.qpos[:] = qm
            d.qvel[:] = vm
        else:
            d.qpos[:] = q
            d.qvel[:] = dq
        d.qacc_warmstart[:] = 0.0            # state-purity: no hidden carry-over
        d.ctrl[:] = 0.0
        d.qfrc_applied[:] = 0.0
        # actuated torques land on the trailing dofs (floating: past the 6
        # base dofs — the same scatter as common.rk4 / the device load)
        off = self.model.nv - len(u)
        d.qfrc_applied[off:off + len(u)] = u
        mujoco.mj_step(self.model, d)
        self._read_contact()
        if self.contact_history is not None:
            c = self.last_contact
            self.contact_history.append((c["ncon"], c["fn"], c["ft"]))
        if self.floating:
            return self._mj_to_pin(d.qpos.copy(), d.qvel.copy())
        return d.qpos.copy(), d.qvel.copy()

    def _read_contact(self):
        mujoco = self._mujoco
        d = self.data
        fn, ft = 0.0, 0.0
        force = np.zeros(6)
        for i in range(d.ncon):
            mujoco.mj_contactForce(self.model, d, i, force)
            # contact frame: x = normal; rotate into world for the plane metric
            R = d.contact[i].frame.reshape(3, 3)
            f_world = R.T @ force[:3]
            fn += abs(f_world[2])
            ft += float(np.hypot(f_world[0], f_world[1]))
        self.last_contact = {"ncon": int(d.ncon), "fn": float(fn), "ft": float(ft)}

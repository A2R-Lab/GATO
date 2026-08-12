"""MPC_GATO: the closed-loop simulation driver used by the paper experiments.

This is a *driver*, not the controller: it owns the pinocchio RK4 simulation
(including the unmodeled pendulum payload and ground-truth disturbance), the
wall-clock pacing, task logic (figure-8 windowing / goal switching), and stat
collection. The per-tick solver interaction lives in gato.controller.MPCController
(warm start, hypothesis batch, best-of-batch selection, time shift) — users with
their own sim or hardware loop should use MPCController/MPCPolicy directly.
"""
import time

import numpy as np
import pinocchio as pin

from .controller import MPCController
from .estimators import ForceEstimator, OneStepWrenchIdentifier
from .hypotheses import ForceHypothesisBatch, IdentifiedWrenchBatch
from .interface import BSQP
from .common import world_wrench_to_joint_local
from .config import DEFAULT_SOLVER_PARAMS
from .worlds import PinocchioWorld


class MPC_GATO:

    def __init__(
        self,
        model,
        model_path,
        N=32,
        dt=0.03125,
        batch_size=1,
        constant_f_ext=None,
        track_full_stats=False,
        plant_type=None,  # None => BSQP auto-detects from model_path
        pendulum_config=None,
        solver_params=None,
        fe_seed=0,
        fc_config=None,
        wrench_id=None,
        world=None,
    ):
        """
        Initialize MPC driver.

        Args:
            model: Pinocchio model (for the SIM; a pendulum may be appended to a copy)
            model_path: Path to URDF file
            N: Prediction horizon (knot points)
            dt: Time step
            batch_size: Number of parallel trajectories (>1 enables force hypotheses)
            constant_f_ext: Ground-truth constant external EE wrench applied in the sim
            track_full_stats: If True, track all stats; if False, only essential ones
            plant_type: Plant identifier selecting the CUDA module (e.g., 'indy7',
                'iiwa14'). None auto-detects from model_path.
            pendulum_config: Optional dict with keys: mass, length, damping, initial_angle
            solver_params: overrides merged onto config.DEFAULT_SOLVER_PARAMS.
                Extra keys 'linsys' and 'bdsv_threshold' pin the linear-system
                policy for BOTH the solver and the controller (absent -> the
                wired per-base defaults; see MPCController)
            fe_seed: seed for the force-estimator hypothesis sampling
            fc_config: GATO_CONTACT_FORCES modules only — program the solver's
                contact-wrench slots so the SOLVER explains an unmodeled wrench
                with decision variables. Keys: 'cost' (fc regularization weight)
                and 'pin_torque_rows' (bool). This is the B=1 counterpart to the
                ForceEstimator hypothesis batch; raises on a default module
                rather than silently degrading to the no-estimator baseline.
            wrench_id: B=1 wrench IDENTIFICATION (dict, or {} for defaults):
                least-squares fit of the disturbance wrench from the one-step
                motion residual, injected through the same f_ext path the
                ForceEstimator uses. Keys: 'alpha' (EMA smoothing),
                'damping' (lstsq Tikhonov), 'max_wrench' (clamp). Requires
                batch_size == 1 — it replaces the hypothesis batch rather than
                augmenting it, and silently coexisting would confound the A/B.
            world: simulation world advancing the plant each substep
                (``worlds.PinocchioWorld`` / ``worlds.MuJoCoWorld``). None =
                the historical pinocchio-RK4 loop, bit-identical. A MuJoCo
                world is an INDEPENDENT simulator (own integrator + contact);
                it supports neither pendulum_config nor constant_f_ext —
                disturbances there are modeled as geometry.
        """
        # Store original model for solver (without pendulum)
        self.solver_model = model

        # Add pendulum to simulation model if configured
        if pendulum_config is not None:
            self.model = self._add_pendulum_to_model(model.copy(), pendulum_config)
            self.pendulum_config = pendulum_config
            self.has_pendulum = True
            self.nq_robot = self.solver_model.nq
            self.nv_robot = self.solver_model.nv
        else:
            self.model = model
            self.pendulum_config = None
            self.has_pendulum = False
            self.nq_robot = model.nq
            self.nv_robot = model.nv

        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()

        # Initialize solver with configurable parameters
        solver_cfg = DEFAULT_SOLVER_PARAMS.copy()
        if solver_params is not None:
            solver_cfg.update(solver_params)

        self.solver = BSQP(
            model_path=model_path,
            batch_size=batch_size,
            N=N,
            dt=dt,
            plant_type=plant_type,
            max_sqp_iters=solver_cfg['max_sqp_iters'],
            kkt_tol=solver_cfg['kkt_tol'],
            max_pcg_iters=solver_cfg['max_pcg_iters'],
            pcg_tol=solver_cfg['pcg_tol'],
            solve_ratio=solver_cfg['solve_ratio'],
            mu=solver_cfg['mu'],
            q_cost=solver_cfg['q_cost'],
            qd_cost=solver_cfg['qd_cost'],
            u_cost=solver_cfg['u_cost'],
            N_cost=solver_cfg['N_cost'],
            q_lim_cost=solver_cfg['q_lim_cost'],
            vel_lim_cost=solver_cfg['vel_lim_cost'],
            ctrl_lim_cost=solver_cfg['ctrl_lim_cost'],
            rho=solver_cfg['rho'],
            linsys=solver_cfg.get('linsys'),
        )

        self.solver_params = solver_cfg

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq_robot + self.nv_robot  # Solver state dimension (robot only)
        # The xu stride and the applied-torque slice diverge on GATO_CONTACT_FORCES
        # builds, where the control is [tau; fc]: only the first n_actuated entries
        # are actuator commands — fc is the solver's own contact explanation and is
        # never played into the sim. Both equal solver_model.nv on default builds.
        self.nu = self.solver.nu                 # xu control stride
        self.nu_act = self.solver.n_actuated     # torques applied to the plant
        self.N = N
        self.dt = dt
        self.batch_size = batch_size
        self.track_full_stats = track_full_stats
        self.fe_seed = fe_seed

        self.wrench_id = wrench_id
        self.fc_config = fc_config
        if fc_config is not None:
            self._configure_fc(fc_config)

        # Ground-truth external force applied in the SIM (what the batch hypothesizes about)
        self.setup_external_forces(constant_f_ext)

        # Hypothesis batch (B > 1): fibonacci-sphere force estimator, same tuning +
        # alpha/beta as the paper experiments
        hypotheses = None
        if wrench_id is not None:
            if batch_size != 1:
                raise ValueError(
                    "wrench_id requires batch_size == 1: it REPLACES the "
                    f"ForceEstimator hypothesis batch (got batch_size={batch_size}). "
                    "Running both would confound which mechanism explains the wrench.")
            # forward only the keys the caller set, so the identifier's own
            # (measured) defaults stay the single source of truth
            id_kw = {k: wrench_id[k] for k in ('alpha', 'damping', 'max_wrench', 'mode', 'weight_tau')
                     if k in wrench_id}
            identifier = OneStepWrenchIdentifier(
                self.solver_model, ee_frame=self.solver.ee_frame, **id_kw)
            hypotheses = IdentifiedWrenchBatch(
                identifier, self.solver_model, ee_frame=self.solver.ee_frame,
                n_actuated=self.solver.n_actuated)
        elif batch_size > 1:
            estimator = ForceEstimator(
                batch_size=batch_size,
                initial_radius=5.0,
                min_radius=2.0,
                max_radius=20.0,
                smoothing_factor=0.5,
                seed=fe_seed,
                alpha=0.6,
                beta=0.5,
            )
            hypotheses = ForceHypothesisBatch(estimator, self.solver_model,
                                              ee_frame=self.solver.ee_frame)

        # 'linsys'/'bdsv_threshold' in solver_params pin the controller's
        # per-step policy too; None -> the controller's wired per-base default
        # (fixed-base "auto"@0.1, floating "bdsv" — see MPCController)
        self.controller = MPCController(self.solver, hypotheses=hypotheses,
                                        warm_start="shift", reset_rho_each_step=True,
                                        linsys=solver_cfg.get('linsys'),
                                        bdsv_threshold=solver_cfg.get('bdsv_threshold'))

        if world is None:
            self.world = PinocchioWorld(self)
        else:
            if self.has_pendulum or np.any(self.constant_f_ext_world):
                raise ValueError(
                    "custom worlds do not support pendulum_config/constant_f_ext "
                    "(PinocchioWorld features — model such disturbances in the "
                    "world itself)")
            if getattr(world, "nq", self.nq) != self.nq:
                raise ValueError(f"world nq {world.nq} != sim model nq {self.nq}")
            self.world = world

    def _configure_fc(self, cfg):
        """Program the solver's contact-wrench slots from an fc_config dict."""
        if self.solver.n_fc == 0:
            raise RuntimeError(
                "fc_config requires a GATO_CONTACT_FORCES module, but this plant "
                "has no fc slots (n_fc == 0) — rebuild with -DGATO_CONTACT_FORCES=ON "
                "and install the build_fc modules before running the fc arm.")
        cost = cfg.get('cost')
        if cost is not None:
            self.solver.set_fc_cost(float(cost))
        if cfg.get('pin_torque_rows', False):
            # The world-aligned wrench layout is [n; f]; a point-mass payload
            # exerts pure force at the EE, so pinning the moment rows to zero
            # halves the fc decision space instead of letting the solve spend
            # it on a moment the physics can't produce.
            self.solver.add_fc_box(0.0, 0.0, slots=range(3))

    @property
    def force_estimator(self):
        """The hypothesis estimator (None at batch_size == 1)."""
        h = self.controller.hypotheses
        return getattr(h, "estimator", None) if h is not None else None

    @property
    def wrench_identifier(self):
        """The least-squares wrench identifier (None unless wrench_id was set)."""
        h = self.controller.hypotheses
        return getattr(h, "identifier", None) if h is not None else None

    def setup_external_forces(self, constant_f_ext):
        """Setup ground-truth external forces for the simulation.

        `constant_f_ext` = [force(3), torque(3)] in WORLD axes acting at the solver's EE
        frame origin. It is re-expressed into the EE parent joint's local frame at every
        sim substep (pin.aba's fext[j] is joint-local, so a world-constant force must
        rotate with the configuration — the old code set it once, which silently turned
        it into a wrist-local force)."""
        self.constant_f_ext_world = constant_f_ext if constant_f_ext is not None else np.zeros(6)

        self.actual_f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.actual_f_ext.append(pin.Force.Zero())

        # resolve on the SIM model (may be pendulum-augmented; robot frames are preserved,
        # so this targets the ROBOT's EE parent joint, never the appended pendulum body)
        self._f_ext_frame_id = self.model.getFrameId(self.solver.ee_frame)

    def _update_sim_f_ext(self, q):
        """Re-express the constant WORLD wrench into the EE parent joint frame at q."""
        if not np.any(self.constant_f_ext_world):
            return
        jid, Fj = world_wrench_to_joint_local(self.model, self.data, q,
                                              self.constant_f_ext_world,
                                              self._f_ext_frame_id)
        self.actual_f_ext[jid] = Fj

    def _observe_substep(self, q_pre, dq_pre, q_post, dq_post, tau, sim_dt):
        """Feed one SENSOR-RATE motion sample to the wrench identifier.

        Called per sim substep, not per control tick, and that is the whole
        point: the fit needs the shortest measurement interval available. The
        average acceleration over a 10 ms control tick is a poor match for any
        single-point evaluation of the dynamics (9.4% median wrench error,
        which destabilizes the loop once the estimate is fed back); over the
        1 ms substep it is 0.5%. A real robot reads encoders at sensor rate for
        the same reason. The identifier never sees the payload — only the
        robot-only model, the applied torque, and the measured motion.
        """
        ident = self.wrench_identifier
        if ident is None:
            return
        nq_r, nv_r = self.nq_robot, self.nv_robot
        ident.identify(q=q_pre[:nq_r], dq=dq_pre[:nv_r], dq_next=dq_post[:nv_r],
                       tau_applied=tau, dt=sim_dt, q_next=q_post[:nq_r])

    def _play_control(self, xu_best, q, dq, timestep, sim_dt, total_sim_time,
                      accumulated_time):
        """Advance the sim by ~timestep using the trajectory's per-knot controls.

        Returns (q, dq, total_sim_time, accumulated_time). When timestep spans
        several knots (wall-clock pacing slower than dt), each sub-step plays the
        control of the knot it falls in.
        """
        nsteps = int(timestep / sim_dt)
        for i in range(nsteps):
            offset = int(i / (self.dt / sim_dt))
            u_idx = self.nx + (self.nx + self.nu) * min(offset, self.N - 1)
            u = xu_best[u_idx:u_idx + self.nu_act]
            q_pre, dq_pre = q.copy(), dq.copy()
            q, dq = self.world.step(q, dq, u, sim_dt)
            self._observe_substep(q_pre, dq_pre, q, dq, u, sim_dt)
            total_sim_time += sim_dt

        # Handle residual time
        if timestep % sim_dt > 1e-5:
            accumulated_time += timestep % sim_dt
            if accumulated_time >= sim_dt:
                accumulated_time = 0.0
                offset = int(nsteps / (self.dt / sim_dt))
                u_idx = self.nx + (self.nx + self.nu) * min(offset, self.N - 1)
                u = xu_best[u_idx:u_idx + self.nu_act]
                q_pre, dq_pre = q.copy(), dq.copy()
                q, dq = self.world.step(q, dq, u, sim_dt)
                self._observe_substep(q_pre, dq_pre, q, dq, u, sim_dt)
                total_sim_time += sim_dt

        return q, dq, total_sim_time, accumulated_time

    def _augment_control(self, u, dq):
        """Robot torques + passive pendulum damping when the payload is attached.

        The actuated torques land at the tail of the ROBOT's dof block
        (offset nv_robot - nu_act: zero on fixed base, past the 6 base dofs
        on floating — common.rk4 applies the same scatter for the plain
        actuated-only case)."""
        if not self.has_pendulum:
            return u
        damping = self.pendulum_config.get('damping', 0.4)
        u_aug = np.zeros(self.nv)
        off = self.nv_robot - self.nu_act
        u_aug[off:off + self.nu_act] = u
        u_aug[self.nv_robot:] = -damping * dq[self.nv_robot:]
        return u_aug

    def run_mpc_fig8(self, x_start, fig8_traj, sim_dt=0.001, sim_time=5.0,
                     pace_by_solve_time=True):
        """
        Run MPC tracking a figure-8 trajectory.

        pace_by_solve_time: if True (default, real-time MPC), advance the sim by the
            measured wall-clock solve time each control step (a faster solver re-plans
            more often). If False, advance a fixed `dt` per solve — matches the CPU
            baseline harness for an apples-to-apples per-solve tracking comparison.

        Returns (None, stats). `goal_distances` uses goal knot 1 (one step ahead);
        `goal_distances_knot0` uses goal knot 0 (the baseline's convention).
        """
        stats = {
            'timestamps': [],
            'solve_times': [],        # GPU solve time in ms
            'goal_distances': [],     # Tracking error in meters
            'ee_actual': [],          # Actual end-effector positions
            'joint_positions': [],    # Joint positions over time
            'joint_velocities': [],   # Joint velocities over time
        }
        if self.track_full_stats:
            stats['sqp_iters'] = []

        total_sim_time = 0.0
        accumulated_time = 0.0

        x_curr = x_start
        q = x_start[:self.nq]
        dq = x_start[self.nq:self.nx]

        ee_g = fig8_traj[:6 * self.N]

        # Reset + warm-up solve (each batch row keeps its own solution)
        self.controller.reset(x_curr)
        res = self.controller.warmup(x_curr, ee_g)
        xu_best = res.xu[0, :]

        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, time={sim_time}s")
        if np.any(self.constant_f_ext_world):
            print(f"External force: {self.constant_f_ext_world[:3]}")

        solve_time = self.dt
        while total_sim_time < sim_time:

            # Simulate forward with the current best trajectory's controls
            timestep = solve_time if pace_by_solve_time else self.dt
            q, dq, total_sim_time, accumulated_time = self._play_control(
                xu_best, q, dq, timestep, sim_dt, total_sim_time, accumulated_time)
            x_curr = np.concatenate([q, dq])

            # Check if trajectory is complete
            eepos_offset = int(total_sim_time / self.dt)
            if eepos_offset >= len(fig8_traj) / 6 - 6 * self.N:
                break
            ee_g = fig8_traj[6 * eepos_offset:6 * (eepos_offset + self.N)]

            # One controller tick (hypotheses -> solve -> select winner -> time shift)
            dt_realized = max(sim_dt, round(timestep / sim_dt) * sim_dt)
            start = time.time()
            r = self.controller.step(x_curr, ee_g, dt_elapsed=dt_realized)
            solve_time = time.time() - start
            xu_best = r.xu_best

            # Collect essential statistics
            ee_pos = self.solver.ee_pos(q)
            goal_dist = np.linalg.norm(ee_pos[:3] - ee_g[6:9])
            goal_dist0 = np.linalg.norm(ee_pos[:3] - ee_g[0:3])  # baseline convention (knot 0)

            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(r.solve.solve_time_us / 1000.0)  # ms
            stats['goal_distances'].append(goal_dist)
            stats.setdefault('goal_distances_knot0', []).append(goal_dist0)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q.copy())
            stats['joint_velocities'].append(dq.copy())

            if self.track_full_stats:
                stats['sqp_iters'].append(int(r.solve.stats.sqp_iters[0]))
                pcg_iters = r.solve.stats.pcg_iters
                if pcg_iters.size > 0:
                    stats.setdefault('pcg_iters', []).append(
                        [int(v) for v in pcg_iters[:, 0]])

        # Convert to numpy arrays
        for key in stats:
            if stats[key]:
                try:
                    stats[key] = np.array(stats[key])
                except (ValueError, TypeError):
                    pass

        print(f"Avg error: {np.mean(stats['goal_distances']):.4f}m")
        print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")

        return None, stats

    def _add_pendulum_to_model(self, model, config):
        """Add a 3D pendulum (spherical joint) to the end-effector."""
        mass = config.get('mass', 15.0)
        length = config.get('length', 0.3)

        ee_joint_id = model.njoints - 1  # Last joint is EE
        pendulum_joint_id = model.addJoint(
            ee_joint_id,
            pin.JointModelSpherical(),
            pin.SE3.Identity(),
            "pendulum_joint"
        )

        # Create inertia for pendulum bob (point mass at distance)
        com = np.array([0.0, 0.0, -length])  # Center of mass along -Z
        inertia_matrix = np.diag([0.001, 0.001, 0.001])  # Small inertia at COM
        pendulum_inertia = pin.Inertia(mass, com, inertia_matrix)
        model.appendBodyToJoint(pendulum_joint_id, pendulum_inertia, pin.SE3.Identity())

        return model

    def run_mpc_goals(
        self,
        x_start,
        goals,
        sim_dt=0.001,
        goal_timeout=5.0,
        goal_threshold=0.05,
        velocity_threshold=1.0,
        velocity_norm=1,
        pace_by_solve_time=True
    ):
        """
        Run MPC tracking discrete goal positions (pick-place style).

        Args:
            x_start: Initial state (robot only, no pendulum)
            goals: List of 3D goal positions np.array([x, y, z])
            sim_dt: Simulation timestep
            goal_timeout: Max time per goal before timeout
            goal_threshold: Distance threshold for goal reached (m)
            velocity_threshold: Velocity threshold for goal reached (rad/s)
            velocity_norm: norm order for the settling gate on dq (np.linalg.norm
                ``ord``): 1 = L1 sum over joints (historic default, strictest),
                2 = Euclidean, np.inf = worst joint.
            pace_by_solve_time: if True (default, real-time MPC), advance the sim
                by the measured wall-clock solve time each control step — so the
                sim clock (timeouts, ``goal_reached_times``, ``time_to_all_reached``)
                is really cumulative WALL time and runs are not reproducible. If
                False, advance a fixed ``dt`` per solve: the clock becomes physical
                task time (paper-comparable completion times, deterministic runs).

        Returns (None, stats) with per-goal outcomes/reach times + tracking stats.
        """
        stats = {
            'timestamps': [],
            'solve_times': [],
            'goal_distances': [],
            'ee_actual': [],
            'joint_positions': [],
            'joint_velocities': [],
            'best_trajectory_id': []
        }
        if self.track_full_stats:
            stats['sqp_iters'] = []
            stats['pcg_iters'] = []

        stats['goal_outcomes'] = ['not_reached'] * len(goals)
        stats['goal_reached_times'] = [None] * len(goals)
        stats['time_to_all_reached'] = None

        total_sim_time = 0.0
        accumulated_time = 0.0

        # Initialize augmented state with pendulum if configured
        if self.has_pendulum:
            x_start_aug = np.zeros(self.nq + self.nv)
            x_start_aug[:self.nx] = x_start  # Robot state
            pendulum_init = self.pendulum_config.get('initial_angle', np.array([0.3, 0.0, 0.0]))
            x_start_aug[self.nq_robot:self.nq_robot + 3] = pendulum_init
            q = x_start_aug[:self.nq]
            dq = x_start_aug[self.nq:]
        else:
            q = x_start[:self.nq]
            dq = x_start[self.nq:]

        # Solver uses robot-only state
        x_curr = x_start

        current_goal_idx = 0
        current_goal = goals[current_goal_idx]
        ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)

        # Reset + warm-up solve
        self.controller.reset(x_curr)
        res = self.controller.warmup(x_curr, ee_g)
        xu_best = res.xu[0, :]

        print(f"\nRunning MPC: N={self.N}, batch={self.batch_size}, {len(goals)} goals")
        if self.has_pendulum:
            print(f"Pendulum: mass={self.pendulum_config['mass']}kg, length={self.pendulum_config['length']}m")

        goal_start_time = total_sim_time
        solve_time = self.dt

        while total_sim_time < goal_timeout * len(goals):

            # Simulate forward (wall-clock paced unless fixed pacing requested)
            timestep = solve_time if pace_by_solve_time else self.dt
            q, dq, total_sim_time, accumulated_time = self._play_control(
                xu_best, q, dq, timestep, sim_dt, total_sim_time, accumulated_time)

            # Update solver state (robot only)
            q_robot = q[:self.nq_robot] if self.has_pendulum else q
            dq_robot = dq[:self.nv_robot] if self.has_pendulum else dq
            x_curr = np.concatenate([q_robot, dq_robot])

            # Check goal reached or timeout
            ee_pos = self.solver.ee_pos(q_robot)
            current_dist = np.linalg.norm(ee_pos - current_goal)
            current_vel = np.linalg.norm(dq_robot, ord=velocity_norm)
            reached = (current_dist < goal_threshold) and (current_vel < velocity_threshold)
            timeout = (total_sim_time - goal_start_time) >= goal_timeout

            if reached or timeout:
                if reached:
                    stats['goal_outcomes'][current_goal_idx] = 'reached'
                    stats['goal_reached_times'][current_goal_idx] = total_sim_time
                else:
                    stats['goal_outcomes'][current_goal_idx] = 'timeout'

                current_goal_idx += 1
                if current_goal_idx >= len(goals):
                    break

                current_goal = goals[current_goal_idx]
                ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
                goal_start_time = total_sim_time

            # One controller tick
            dt_realized = max(sim_dt, round(timestep / sim_dt) * sim_dt)
            start = time.time()
            r = self.controller.step(x_curr, ee_g, dt_elapsed=dt_realized)
            solve_time = time.time() - start
            xu_best = r.xu_best

            # Collect statistics
            stats['timestamps'].append(total_sim_time)
            stats['solve_times'].append(r.solve.solve_time_us / 1000.0)  # ms
            stats['goal_distances'].append(current_dist)
            stats['ee_actual'].append(ee_pos.copy())
            stats['joint_positions'].append(q_robot.copy())
            stats['joint_velocities'].append(dq_robot.copy())
            stats['best_trajectory_id'].append(r.best_id)

            if self.track_full_stats:
                stats['sqp_iters'].append(int(r.solve.stats.sqp_iters[0]))
                pcg_iters = r.solve.stats.pcg_iters
                stats['pcg_iters'].append(int(pcg_iters[0, 0]) if pcg_iters.size else 0)

        # Convert to numpy arrays
        for key in stats:
            if isinstance(stats[key], list) and len(stats[key]) > 0:
                if key not in ['goal_outcomes', 'goal_reached_times', 'time_to_all_reached']:
                    try:
                        stats[key] = np.array(stats[key])
                    except (ValueError, TypeError):
                        pass

        # Compute time to all-goals-reached if all were reached
        if all([o == 'reached' for o in stats['goal_outcomes']]):
            reached_times = [t for t in stats['goal_reached_times'] if t is not None]
            if len(reached_times) == len(goals):
                stats['time_to_all_reached'] = float(np.max(reached_times))

        goals_reached = sum(1 for o in stats['goal_outcomes'] if o == 'reached')
        print(f"Goals reached: {goals_reached}/{len(goals)}")
        if len(stats['solve_times']) > 0:
            print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")

        return None, stats

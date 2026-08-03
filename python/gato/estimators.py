"""Force-disturbance estimators: sample B wrench hypotheses, refit on feedback.

Both classes share the duck-typed estimator contract consumed by
gato.hypotheses.ForceHypothesisBatch:
    generate_batch() -> (B, 6) world-frame wrenches
    update(best_idx, prediction_errors, batch_used)
    reset(); get_stats() -> dict
"""
import numpy as np


class ForceEstimator:
    
    def __init__(self, batch_size, initial_radius=10.0, min_radius=1.0, max_radius=100.0,
                 smoothing_factor=0.3, seed=0, alpha=0.5, beta=0.8):

        assert batch_size > 3, "Batch size must be > 3 for exploitation + exploration strategy"

        # Dedicated RNG so the disturbance-hypothesis sampling is REPRODUCIBLE. With the old
        # global np.random the pick-place closed loop was non-deterministic run-to-run (and
        # occasionally diverged); seed=None falls back to fresh entropy if non-determinism is wanted.
        self._rng = np.random.default_rng(seed)

        self.batch_size = batch_size
        self.dim = 6  # 6D force/torque vector
        
        self.radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radius_increase_factor = 1.05  
        self.radius_decrease_factor = 0.95  
        
        self.estimate = np.zeros(self.dim, dtype=np.float32)
        self.momentum = np.zeros(self.dim, dtype=np.float32)
        self.smoothed_estimate = np.zeros(self.dim, dtype=np.float32)
        self.confidence = 0.0
        self.error_history = []
        self.smoothing_factor = smoothing_factor
        self.alpha = alpha  # blend toward the winning hypothesis
        self.beta = beta    # momentum retention
        
        num_exploration = batch_size - 3
        self.sphere_dirs = self._fibonacci_sphere(num_exploration)
        
        self.current_rotation = np.eye(3, dtype=np.float32)
        
    def _fibonacci_sphere(self, n):
        """
        Generate n uniformly distributed points on unit sphere using Fibonacci spiral.
        
        Args:
            n: Number of points to generate
            
        Returns:
            Array of shape (n, 3) with unit vectors
        """
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
            
        points = np.zeros((n, 3), dtype=np.float32)
        
        phi = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            y = 1 - (2 * i / (n - 1)) if n > 1 else 0
            
            radius = np.sqrt(1 - y * y)
            
            theta = 2 * np.pi * i / phi
            
            points[i, 0] = radius * np.cos(theta)
            points[i, 1] = y
            points[i, 2] = radius * np.sin(theta)
            
        return points
    
    def _random_rotation_matrix(self):
        """
        Generate a random 3x3 rotation matrix using a uniformly random unit quaternion.
        """
        u1, u2, u3 = self._rng.random(3)
        q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
        q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
        x, y, z, w = q1, q2, q3, q4
        # Quaternion to rotation matrix
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = np.array([
            [1.0 - 2.0 * (yy + zz),     2.0 * (xy - wz),           2.0 * (xz + wy)],
            [2.0 * (xy + wz),           1.0 - 2.0 * (xx + zz),     2.0 * (yz - wx)],
            [2.0 * (xz - wy),           2.0 * (yz + wx),           1.0 - 2.0 * (xx + yy)]
        ], dtype=np.float32)
        return R
    
    def generate_batch(self):

        batch = np.zeros((self.batch_size, 6), dtype=np.float32)
        
        batch[0, :] = self.smoothed_estimate
        
        batch[1, :] = 0.0
        
        batch[2, :] = self.smoothed_estimate + 0.5 * self.momentum  
        for i in range(3, self.batch_size):
            base_direction = self.sphere_dirs[i - 3]
            direction = self.current_rotation @ base_direction
            base = 0.7 * self.smoothed_estimate[:3] + 0.3 * self.estimate[:3]
            batch[i, :3] = base + self.radius * direction
            batch[i, 3:] = self.smoothed_estimate[3:]
            
        return batch
    
    def update(self, best_idx, prediction_errors, batch_used):
        # Refit on feedback. batch_used = the WORLD-FRAME batch the errors were
        # scored under (the exact array generate_batch() returned this tick) —
        # regenerating it here instead was order-fragile (only worked because the
        # random rotation refreshes at the END of update).
        min_error = np.min(prediction_errors)
        self.error_history.append(min_error)

        best_force = np.asarray(batch_used)[best_idx, :]

        delta = best_force - self.estimate
        self.momentum = self.beta * self.momentum + (1 - self.beta) * delta

        raw_update = self.alpha * best_force + (1 - self.alpha) * self.estimate
        self.estimate = 0.8 * self.estimate + 0.2 * (raw_update + 0.5 * self.momentum)
        
        self.smoothed_estimate = (1 - self.smoothing_factor) * self.smoothed_estimate + self.smoothing_factor * self.estimate
        
        if best_idx < 3:
            self.radius *= self.radius_decrease_factor
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.radius *= self.radius_increase_factor
            self.confidence = max(0.0, self.confidence - 0.1)
            
        self.radius = np.clip(self.radius, self.min_radius, self.max_radius)
        
        if len(self.error_history) > 5:
            recent_errors = self.error_history[-5:]
            error_std = np.std(recent_errors)
            
            if error_std < 0.01:
                self.radius *= 0.9  
            elif recent_errors[-1] > 1.5 * np.mean(recent_errors[:-1]):
                self.radius *= 1.3  
                self.confidence *= 0.5  
                
            self.radius = np.clip(self.radius, self.min_radius, self.max_radius)
        
        self.current_rotation = self._random_rotation_matrix()
    
    def reset(self):
        self.estimate = np.zeros(self.dim, dtype=np.float32)
        self.momentum = np.zeros(self.dim, dtype=np.float32)
        self.smoothed_estimate = np.zeros(self.dim, dtype=np.float32)
        self.radius = 10.0 
        self.confidence = 0.0
        self.error_history = []
        self.current_rotation = np.eye(3, dtype=np.float32)
    
    def get_stats(self):
        return {
            'current_estimate': self.estimate.copy(),
            'smoothed_estimate': self.smoothed_estimate.copy(),
            'momentum': self.momentum.copy(),
            'radius': self.radius,
            'confidence': self.confidence,
            'recent_error': self.error_history[-1] if self.error_history else np.inf
        }


"""CEM (cross-entropy method) external-force estimator — an alternative to the
fibonacci-sphere `ForceEstimator` in force_estimator.py. Recovered from the
pre-migration CS3a pick-place work (origin/hardware:examples/force_estimator_cem.py);
see docs/archaeology.md. Self-contained (numpy only); drop-in for batched MPC force
estimation (generate_batch / update / reset / get_stats)."""


class CEMForceEstimator:
    """
    Cross-Entropy Method (CEM) force/torque estimator for 6D wrenches.

    - Maintains a diagonal Gaussian N(mu, diag(sigma^2)) over 6D wrench
      (fx, fy, fz, mx, my, mz)
    - Generates a batch with exploitation seeds + exploration samples
    - Refits to elite samples based on prediction error
    """

    def __init__(
        self,
        batch_size: int,
        force_sigma: float = 10.0,
        torque_sigma: float = 0.0,
        elite_frac: float = 0.25,
        min_sigma: float = 0.5,
        max_sigma: float = 100.0,
        alpha_mean: float = 0.6,
        alpha_cov: float = 0.3,
        process_noise: float = 0.1,
        momentum_beta: float = 0.4,
        seed: int | None = None,
        axial_seeds: bool = True,
        shrink_on_exploit: float = 0.7,
        plateau_shrink: float = 0.85,
        spike_expand: float = 1.25,
    ):
        assert batch_size > 3, "Batch size must be > 3 (exploitation + exploration)"
        self.batch_size = batch_size
        self.dim = 6

        # Distribution parameters (diagonal covariance stored as std vector)
        self.mu = np.zeros(self.dim, dtype=np.float32)
        init_sigma_vec = np.array(
            [force_sigma] * 3 + [torque_sigma] * 3, dtype=np.float32
        )
        self._sigma = init_sigma_vec.astype(np.float32)  # std vector
        self._init_sigma = self._sigma.copy()

        # Bounds for diag elements
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)

        # Update smoothing
        self.alpha_mean = float(alpha_mean)
        self.alpha_cov = float(alpha_cov)
        self.process_noise = float(process_noise)

        # Momentum on best direction
        self.momentum_beta = float(momentum_beta)
        self.momentum = np.zeros(self.dim, dtype=np.float32)

        # Elite configuration
        self.elite_frac = float(elite_frac)
        self.elite_k = max(1, int(np.floor(self.elite_frac * self.batch_size)))

        self.last_batch = None
        self.last_best = np.zeros(self.dim, dtype=np.float32)
        self.error_history: list[float] = []
        self.rng = np.random.default_rng(seed)

        self.axial_seeds = bool(axial_seeds)
        self.shrink_on_exploit = float(shrink_on_exploit)
        self.plateau_shrink = float(plateau_shrink)
        self.spike_expand = float(spike_expand)

    def _clamp_sigma(self, sigma: np.ndarray) -> np.ndarray:
        # Clamp diagonal standard deviations
        return np.clip(sigma, self.min_sigma, self.max_sigma).astype(np.float32)

    @property
    def Sigma(self) -> np.ndarray:
        return np.diag((self._sigma ** 2).astype(np.float32))

    def _softmax_weights(self, elite_err: np.ndarray) -> np.ndarray:
        tau = max(1e-6, float(np.median(elite_err) - float(np.min(elite_err)) + 1e-6))
        shifted = -(elite_err - float(np.min(elite_err))) / tau
        exps = np.exp(shifted - float(np.max(shifted)))
        return exps / np.sum(exps)

    def _adapt_sigma_from_error_trend(self) -> None:
        if len(self.error_history) < 6:
            return
        recent = np.array(self.error_history[-6:], dtype=np.float32)
        err_std = float(np.std(recent[:-1]))
        if err_std < 1e-2:
            self._sigma *= self.plateau_shrink
        elif recent[-1] > 1.5 * float(np.mean(recent[:-1])):
            self._sigma *= self.spike_expand
        self._sigma = self._clamp_sigma(self._sigma)

    def generate_batch(self) -> np.ndarray:
        """
        Returns array of shape (batch_size, 6).

        Slots:
        - [0] current mean (mu)
        - [1] zero wrench
        - [2] momentum seed (mu + 0.5 * momentum)
        - [3:] samples from N(mu, Sigma)
        """
        B = self.batch_size
        batch = np.zeros((B, self.dim), dtype=np.float32)

        # Exploitation seeds
        batch[0, :] = self.mu
        batch[1, :] = 0.0
        batch[2, :] = self.mu + 0.5 * self.momentum

        # Exploration samples
        std = self._sigma.astype(np.float32)

        idx = 3
        if self.axial_seeds and B - idx >= 1:
            # Deterministic axial probes: mu ± std along each axis
            eye = np.eye(self.dim, dtype=np.float32)
            for d in range(self.dim):
                if idx < B:
                    batch[idx, :] = self.mu + std[d] * eye[d]
                    idx += 1
                if idx < B:
                    batch[idx, :] = self.mu - std[d] * eye[d]
                    idx += 1

        remaining = B - idx
        if remaining > 0:
            samples = self.rng.normal(loc=self.mu, scale=std, size=(remaining, self.dim)).astype(
                np.float32
            )
            batch[idx:, :] = samples

        self.last_batch = batch
        return batch

    def update(
        self,
        best_idx: int,
        errors: np.ndarray,
        batch_used: np.ndarray | None = None,
    ) -> None:
        """
        Update distribution parameters using elite re-fitting.

        Args:
            best_idx: index of the best-performing hypothesis
            errors: array of shape (B,) with lower = better
            batch_used: the exact batch used for evaluating errors (B x 6)
        """
        if batch_used is None:
            if self.last_batch is None:
                raise ValueError("No batch available for update; pass batch_used explicitly.")
            batch_used = self.last_batch

        B = batch_used.shape[0]
        assert errors.shape[0] == B, "errors must match batch size"

        self.error_history.append(float(np.min(errors)))
        self.last_best = batch_used[best_idx].astype(np.float32)

        elite_k = self.elite_k
        elite_ids = np.argsort(errors)[:elite_k]
        elites = batch_used[elite_ids]

        # Weighted refit using softmax over negative errors
        elite_err = errors[elite_ids]
        w = self._softmax_weights(elite_err)
        elite_mean = np.sum(elites * w[:, None], axis=0)

        # Covariance with noise
        diff = elites - elite_mean[None, :]
        elite_var = np.sum((diff**2) * w[:, None], axis=0)
        elite_var += (self.process_noise**2)

        # Smooth updates
        self.mu = ((1.0 - self.alpha_mean) * self.mu + self.alpha_mean * elite_mean.astype(np.float32))

        target_sigma = np.sqrt(elite_var.astype(np.float32))
        self._sigma = (1.0 - self.alpha_cov) * self._sigma + self.alpha_cov * target_sigma
        self._sigma = self._clamp_sigma(self._sigma)

        # Momentum update
        delta = self.last_best - self.mu
        self.momentum = self.momentum_beta * self.momentum + (1.0 - self.momentum_beta) * delta

        # Adaptive exploration: if recent errors plateau, shrink; if spike, expand
        self._adapt_sigma_from_error_trend()

        # If an exploitation seed won, shrink covariance
        if best_idx < 3:
            self._sigma = self._clamp_sigma(self._sigma * self.shrink_on_exploit)

    def reset(self) -> None:
        self.mu[:] = 0.0
        self._sigma[:] = self._init_sigma
        self.momentum[:] = 0.0
        self.last_batch = None
        self.last_best[:] = 0.0
        self.error_history.clear()

    def get_stats(self) -> dict:
        return {
            "mu": self.mu.copy(),
            "sigma_diag": self._sigma.copy(),
            "momentum": self.momentum.copy(),
            "last_error": self.error_history[-1] if self.error_history else float("inf"),
        }


class OneStepWrenchIdentifier:
    """Least-squares wrench identification from the one-step motion residual.

    The sampling estimators above search for the wrench that best explains the
    observed motion; this one SOLVES for it. Over an interval where a constant
    torque tau was applied, the nominal (payload-free) model needs generalized
    force ``rnea(q, dq, ddq_measured)`` to produce the motion that was actually
    observed, so

        tau_resid = rnea(q, dq, ddq_meas) - tau_applied

    is exactly the generalized force contributed by everything the solver's
    model omits. An external wrench w at the EE frame contributes ``J^T w``, so
    w is the least-squares solution of ``J^T w = tau_resid`` — 6 unknowns from
    nv equations, one linear solve, NO batch dimension.

    That is the point: the hypothesis batch spends B parallel solves searching
    a 6-D space that a single lstsq resolves directly. Accuracy is limited by
    the finite-difference ddq and by wrenches outside the EE-Jacobian range
    (which no EE wrench can explain), not by sample count.

    Frame conventions match ForceEstimator so the two are interchangeable
    behind gato.hypotheses: ``generate_batch()`` returns (1, 6) WORLD-frame
    wrenches ordered [force(3); torque(3)] at the EE frame origin.

    Args:
        model: the SOLVER's pinocchio model (robot only — the whole point is
            that the payload is unmodeled).
        ee_frame: URDF frame the wrench acts at.
        alpha: EMA smoothing on the estimate (1.0 = raw per-tick fit). The
            one-step fit is unbiased but noisy — ddq is a finite difference.
        damping: Tikhonov damping on the lstsq (0 = plain pseudo-inverse);
            regularizes wrench directions the Jacobian barely observes. Near
            kinematic singularities an undamped pseudo-inverse amplifies the
            measurement residual without bound, so the default is nonzero.
        mode: 'wrench' (default) uses the full identified wrench; 'weight'
            estimates the payload WEIGHT — the gravity-aligned component
            filtered on weight_tau — which is the part a constant-over-the-
            horizon wrench model can legitimately extrapolate.
        weight_tau: time constant [s] of the weight filter (mode='weight').
            ★ Measured on the pick-place task (100 scenarios at the default,
            15-scenario probes elsewhere; success / mean goals per 5):
                instantaneous  10/100, 3.10
                tau = 0.05      -     , 3.47
                tau = 0.10   ★ 27/100, 3.80   <- default
                tau = 0.15      -     , 3.20
                tau = 0.25      -     , 2.27
                tau = 0.50      -     , 1.73  <- WORSE THAN NO ESTIMATOR (2.55)
            The optimum is sharp and the penalty for over-filtering is severe:
            long constants lag a payload the arm is actively swinging, and the
            lagged weight is worse than no correction at all. Do not raise this
            toward the swing period on the intuition that more averaging is
            safer -- it is not.
        max_wrench: magnitude clamp on force/torque halves, a divergence guard.
            NOT a physical prior: on this task the true wrench swings over
            34-1041 N (a swinging payload on an accelerating arm is mostly
            inertial reaction, not weight), so clamp well above the static
            weight or the observer will saturate exactly when it matters.
    """

    batch_size = 1

    def __init__(self, model, ee_frame="EE", alpha=0.5, damping=1e-3,
                 max_wrench=2000.0, mode="wrench", weight_tau=0.1):
        from .common import _require_pin
        self._pin = _require_pin()
        self.model = model
        self._data = model.createData()
        self._ee_frame_id = model.getFrameId(ee_frame)
        if self._ee_frame_id >= model.nframes:
            raise ValueError(f"URDF has no frame named {ee_frame!r}")
        if mode not in ("wrench", "weight"):
            raise ValueError(f"mode must be 'wrench' or 'weight', got {mode!r}")
        self.alpha = float(alpha)
        self.damping = float(damping)
        self.max_wrench = float(max_wrench)
        self.mode = mode
        self.weight_tau = float(weight_tau)
        g_vec = np.asarray(model.gravity.linear, dtype=np.float64)
        if not np.any(g_vec):
            g_vec = np.array([0.0, 0.0, -9.81])
        # unit vector along gravity: the direction a hanging payload pulls
        self._g_hat = g_vec / np.linalg.norm(g_vec)
        self.dim = 6
        self.reset()

    def reset(self):
        self.estimate = np.zeros(self.dim, dtype=np.float32)
        self.raw = np.zeros(self.dim, dtype=np.float32)
        self._weight_raw = 0.0
        self.weight = 0.0
        self.residual_norm = 0.0     # ||J^T w - tau_resid||: what NO EE wrench explains
        self.tau_resid_norm = 0.0
        self.n_updates = 0

    def generate_batch(self):
        """(1, 6) world-frame wrench — the current identified estimate."""
        return self.estimate.reshape(1, self.dim).copy()

    def identify(self, q, dq, dq_next, tau_applied, dt, q_next=None):
        """Fit the wrench explaining (q, dq) -> (q_next, dq_next) under tau_applied.

        q/dq/dq_next/q_next are ROBOT-only (the solver model's coordinates);
        tau_applied is the actuated torque held over the interval. Returns the
        raw fit (unsmoothed); the smoothed value lands in ``self.estimate``.

        ★ ACCURACY IS ALL ABOUT THE INTERVAL. ``ddq`` is the AVERAGE
        acceleration over [t, t+dt], so the dynamics must be evaluated at the
        interval MIDPOINT (second-order) rather than its start (first-order),
        and dt must be the SENSOR period, not the control period. Measured on
        the closed-loop pick-place task against an exact-acceleration observer:

            1 ms interval, midpoint  ->  0.5% median error   (usable)
           10 ms interval, start     ->  9.4% median error   (diverges in the
                                         loop: a wrong wrench drives worse
                                         control, which corrupts the next fit)

        so pass the shortest measurement interval available and give q_next.
        """
        pin = self._pin
        nv = self.model.nv
        nq = self.model.nq
        q = np.asarray(q, dtype=np.float64)[:nq]
        dq = np.asarray(dq, dtype=np.float64)[:nv]
        dq_next = np.asarray(dq_next, dtype=np.float64)[:nv]
        tau_applied = np.asarray(tau_applied, dtype=np.float64)[:nv]

        ddq = (dq_next - dq) / float(dt)
        if q_next is not None:
            q_eval = pin.interpolate(self.model, q,
                                     np.asarray(q_next, dtype=np.float64)[:nq], 0.5)
            dq_eval = 0.5 * (dq + dq_next)
        else:
            q_eval, dq_eval = q, dq
        tau_needed = pin.rnea(self.model, self._data, q_eval, dq_eval, ddq)
        tau_resid = tau_needed - tau_applied
        q = q_eval
        dq = dq_eval

        # J maps joint velocity -> EE spatial velocity in world-aligned axes at
        # the frame origin; its transpose maps a wrench there to joint torques.
        # pinocchio orders spatial quantities [linear; angular] == [force; torque]
        # for the dual, which is the world-wrench convention the injection path
        # (world_wrench_to_joint_local) expects.
        pin.computeJointJacobians(self.model, self._data, q)
        pin.updateFramePlacements(self.model, self._data)
        J = pin.getFrameJacobian(self.model, self._data, self._ee_frame_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        A = J.T                                   # (nv, 6)
        if self.damping > 0:
            AtA = A.T @ A + self.damping * np.eye(6)
            w = np.linalg.solve(AtA, A.T @ tau_resid)
        else:
            w, *_ = np.linalg.lstsq(A, tau_resid, rcond=None)

        for half in (slice(0, 3), slice(3, 6)):
            n = np.linalg.norm(w[half])
            if n > self.max_wrench:
                w[half] *= self.max_wrench / n

        self.tau_resid_norm = float(np.linalg.norm(tau_resid))
        self.residual_norm = float(np.linalg.norm(A @ w - tau_resid))

        if self.mode == "weight":
            # Keep only the part a constant-wrench model can honestly
            # EXTRAPOLATE over the horizon. The measured wrench is dominated by
            # the payload's inertial reaction to the arm's OWN acceleration —
            # state-dependent, so holding it fixed across 16 knots plans against
            # a force the plan itself changes.
            #
            # ★ Estimate the WEIGHT, not the instantaneous downward component.
            # The payload weight is CONSTANT; the inertial reaction oscillates
            # and also projects onto gravity, so an instantaneous read is not a
            # weight at all (measured: it sat at 0 through a transient, then
            # read 270-556 N against a true 98.1 N). Filter it on a time
            # constant longer than the payload's swing period (T = 2*pi*sqrt(L/g)
            # ~ 1.5 s here) so the oscillating part averages out.
            #
            # ⚠ Lag is harmless HERE and only here: the target does not move.
            # Filtering the full wrench this slowly is much worse than not
            # filtering at all (mode="wrench" mean goals/5: 1.87 at alpha=0.5,
            # 1.27 at 0.1, 0.93 at 0.02) because that signal is genuinely fast.
            # Slow-filter the constant part; never the varying part.
            a = float(dt) / (self.weight_tau + float(dt))   # sample-rate independent
            self._weight_raw = (1.0 - a) * self._weight_raw + a * float(w[:3] @ self._g_hat)
            self.weight = max(0.0, self._weight_raw)        # a negative weight is unphysical
            self.raw = np.concatenate(
                [self._g_hat * self.weight, np.zeros(3)]).astype(np.float32)
            self.estimate = self.raw                        # filtering already done
        else:
            self.raw = w.astype(np.float32)
            self.estimate = ((1.0 - self.alpha) * self.estimate
                             + self.alpha * self.raw).astype(np.float32)
        self.n_updates += 1
        return self.raw

    def update(self, best_idx, prediction_errors, batch_used=None):
        """No-op: the fit is driven by ``identify``, not by batch feedback."""

    def get_stats(self):
        return {
            "estimate": self.estimate.copy(),
            "raw": self.raw.copy(),
            "force_norm": float(np.linalg.norm(self.estimate[:3])),
            "torque_norm": float(np.linalg.norm(self.estimate[3:])),
            "tau_resid_norm": self.tau_resid_norm,
            "unexplained_norm": self.residual_norm,
            "weight": self.weight,
            "n_updates": self.n_updates,
        }

import numpy as np


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

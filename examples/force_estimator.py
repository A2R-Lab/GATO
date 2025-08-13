import numpy as np


class ImprovedForceEstimator:
    
    def __init__(self, batch_size, initial_radius=10.0, min_radius=1.0, max_radius=100.0, smoothing_factor=0.3):
        """
        Initialize the force estimator.
        
        Args:
            batch_size: Number of parallel force hypotheses to test
            initial_radius: Starting search radius for exploration
            min_radius: Minimum allowed search radius
            max_radius: Maximum allowed search radius
            smoothing_factor: Exponential smoothing factor for estimate updates (0-1, lower=smoother)
        """
        assert batch_size > 3, "Batch size must be > 3 for exploitation + exploration strategy"
        
        self.batch_size = batch_size
        self.dim = 6  # 6D force/torque vector
        
        self.radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radius_increase_factor = 1.05  # Gentler increase when exploration wins
        self.radius_decrease_factor = 0.95  # Gentler decrease when exploitation wins
        
        self.estimate = np.zeros(self.dim, dtype=np.float32)
        self.momentum = np.zeros(self.dim, dtype=np.float32)
        self.smoothed_estimate = np.zeros(self.dim, dtype=np.float32)
        self.confidence = 0.0
        self.error_history = []
        self.smoothing_factor = smoothing_factor
        
        # Reserve 3 slots for exploitation strategies
        num_exploration = batch_size - 3
        self.sphere_dirs = self._fibonacci_sphere(num_exploration)
        
        # Rotation applied to exploration directions. Updated each iteration
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
        
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            # Map to [-1, 1] for y coordinate
            y = 1 - (2 * i / (n - 1)) if n > 1 else 0
            
            # Radius in x-z plane
            radius = np.sqrt(1 - y * y)
            
            # Golden angle increment
            theta = 2 * np.pi * i / phi
            
            # Convert to Cartesian coordinates
            points[i, 0] = radius * np.cos(theta)
            points[i, 1] = y
            points[i, 2] = radius * np.sin(theta)
            
        return points
    
    def _random_rotation_matrix(self):
        """
        Generate a random 3x3 rotation matrix using a uniformly random unit quaternion.
        """
        u1, u2, u3 = np.random.rand(3)
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
        """
        Generate batch of force hypotheses for testing.
        
        Returns:
            Array of shape (batch_size, 6)
        """
        batch = np.zeros((self.batch_size, 6), dtype=np.float32)
        
        # === Exploitation slots (first 3) ===
        # 1. Current smoothed estimate
        batch[0, :] = self.smoothed_estimate
        
        # 2. Zero force hypothesis (useful for detecting when force stops)
        batch[1, :] = 0.0
        
        # 3. Momentum-based prediction (for tracking time-varying forces)
        batch[2, :] = self.smoothed_estimate + 0.5 * self.momentum  # Reduced momentum influence
        
        # === Exploration slots (remaining) ===
        # Sample points on sphere at current radius with some noise reduction
        # Only explore the force components (first 3), keep torques at 0
        for i in range(3, self.batch_size):
            base_direction = self.sphere_dirs[i - 3]
            # Rotate base Fibonacci directions to change exploration orientation each iteration
            direction = self.current_rotation @ base_direction
            # Mix between estimate and smoothed estimate for exploration base
            base = 0.7 * self.smoothed_estimate[:3] + 0.3 * self.estimate[:3]
            batch[i, :3] = base + self.radius * direction
            batch[i, 3:] = self.smoothed_estimate[3:]  # Keep torque components from estimate
            
        return batch
    
    def update(self, best_idx, prediction_errors, alpha=0.5, beta=0.8):
        """
        Update force estimate based on simulation results.
        
        Args:
            best_idx: Index of best performing hypothesis
            prediction_errors: Array of prediction errors for all hypotheses
            alpha: Blending factor for estimate update (0=keep old, 1=use new)
            beta: Momentum decay factor (0=no momentum, 1=full momentum)
        """
        # Track error history
        min_error = np.min(prediction_errors)
        self.error_history.append(min_error)
        
        # Get the winning force estimate
        batch = self.generate_batch()  # Regenerate to get same batch
        best_force = batch[best_idx, :]  # Full 6D force/torque
        
        # Update momentum with exponential decay
        delta = best_force - self.estimate
        self.momentum = beta * self.momentum + (1 - beta) * delta
        
        # Update estimate
        raw_update = alpha * best_force + (1 - alpha) * self.estimate
        self.estimate = 0.8 * self.estimate + 0.2 * (raw_update + 0.5 * self.momentum)
        
        self.smoothed_estimate = (1 - self.smoothing_factor) * self.smoothed_estimate + self.smoothing_factor * self.estimate
        
        # Adaptive radius control based on which strategy won
        if best_idx < 3:  # Exploitation sample won
            self.radius *= self.radius_decrease_factor
            self.confidence = min(1.0, self.confidence + 0.05)  # Slower confidence growth
        else:  # Exploration sample won
            self.radius *= self.radius_increase_factor
            self.confidence = max(0.0, self.confidence - 0.1)  # Slower confidence decay
            
        # Clamp radius to reasonable bounds
        self.radius = np.clip(self.radius, self.min_radius, self.max_radius)
        
        # Additional radius adjustment based on convergence
        if len(self.error_history) > 5:
            recent_errors = self.error_history[-5:]
            error_std = np.std(recent_errors)
            
            if error_std < 0.01:  # Converged - reduce exploration
                self.radius *= 0.9  
            elif recent_errors[-1] > 1.5 * np.mean(recent_errors[:-1]):  # Diverging
                self.radius *= 1.3  
                self.confidence *= 0.5  
                
            # Keep radius in bounds after adjustments
            self.radius = np.clip(self.radius, self.min_radius, self.max_radius)
        
        # Rotate exploration directions for the next iteration
        self.current_rotation = self._random_rotation_matrix()
    
    def reset(self):
        """Reset estimator to initial state."""
        self.estimate = np.zeros(self.dim, dtype=np.float32)
        self.momentum = np.zeros(self.dim, dtype=np.float32)
        self.smoothed_estimate = np.zeros(self.dim, dtype=np.float32)
        self.radius = 10.0 
        self.confidence = 0.0
        self.error_history = []
        self.current_rotation = np.eye(3, dtype=np.float32)
    
    def get_stats(self):
        """Get current estimator statistics."""
        return {
            'current_estimate': self.estimate.copy(),
            'smoothed_estimate': self.smoothed_estimate.copy(),
            'momentum': self.momentum.copy(),
            'radius': self.radius,
            'confidence': self.confidence,
            'recent_error': self.error_history[-1] if self.error_history else np.inf
        }

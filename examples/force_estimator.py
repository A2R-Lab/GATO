import numpy as np


class ImprovedForceEstimator:
    
    def __init__(self, batch_size, initial_radius=10.0, min_radius=1.0, max_radius=100.0, smoothing_factor=0.3):

        assert batch_size > 3, "Batch size must be > 3 for exploitation + exploration strategy"
        
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
    
    def update(self, best_idx, prediction_errors, alpha=0.5, beta=0.8):
        min_error = np.min(prediction_errors)
        self.error_history.append(min_error)
        
        batch = self.generate_batch()  
        best_force = batch[best_idx, :] 
        
        delta = best_force - self.estimate
        self.momentum = beta * self.momentum + (1 - beta) * delta
        
        raw_update = alpha * best_force + (1 - alpha) * self.estimate
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

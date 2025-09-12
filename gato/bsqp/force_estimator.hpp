#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <cstdint>

namespace gato {

// Lightweight host-side port of examples/force_estimator.ImprovedForceEstimator
template <typename T=float>
class ImprovedForceEstimator {
public:
    explicit ImprovedForceEstimator(uint32_t batch_size,
                                    T initial_radius = T(5.0),
                                    T min_radius = T(2.0),
                                    T max_radius = T(20.0),
                                    T smoothing = T(0.5))
        : batch_size_(batch_size),
          radius_(initial_radius), min_radius_(min_radius), max_radius_(max_radius),
          radius_inc_(T(1.05)), radius_dec_(T(0.95)),
          smoothing_(smoothing), confidence_(T(0.0)), rng_(std::random_device{}())
    {
        if (batch_size_ <= 3) throw std::runtime_error("Batch size must be > 3");
        estimate_.fill(T(0));
        momentum_.fill(T(0));
        smoothed_.fill(T(0));
        current_rot_ = identity3();
        const uint32_t num_explore = batch_size_ - 3;
        sphere_dirs_.resize(num_explore * 3);
        fibonacci_sphere(num_explore, sphere_dirs_.data());
    }

    // Returns (batch_size x 6) row-major [fx, fy, fz, mx, my, mz]
    std::vector<T> generate_batch()
    {
        std::vector<T> batch(static_cast<size_t>(batch_size_) * 6, T(0));
        // exploitation seeds
        for (int i = 0; i < 6; ++i) batch[i] = smoothed_[i]; // [0]
        // [1] zero already
        for (int i = 0; i < 6; ++i) batch[2*6 + i] = smoothed_[i] + T(0.5) * momentum_[i];

        // exploration on force components only
        for (uint32_t bi = 3; bi < batch_size_; ++bi) {
            const T* dir0 = &sphere_dirs_[(bi - 3) * 3];
            // rotate
            T dir[3] = {
                current_rot_[0]*dir0[0] + current_rot_[1]*dir0[1] + current_rot_[2]*dir0[2],
                current_rot_[3]*dir0[0] + current_rot_[4]*dir0[1] + current_rot_[5]*dir0[2],
                current_rot_[6]*dir0[0] + current_rot_[7]*dir0[1] + current_rot_[8]*dir0[2]
            };
            T base[3] = {
                T(0.7)*smoothed_[0] + T(0.3)*estimate_[0],
                T(0.7)*smoothed_[1] + T(0.3)*estimate_[1],
                T(0.7)*smoothed_[2] + T(0.3)*estimate_[2]
            };
            T* dst = &batch[bi * 6];
            dst[0] = base[0] + radius_ * dir[0];
            dst[1] = base[1] + radius_ * dir[1];
            dst[2] = base[2] + radius_ * dir[2];
            // keep torques from estimate
            dst[3] = smoothed_[3];
            dst[4] = smoothed_[4];
            dst[5] = smoothed_[5];
        }
        last_batch_ = batch; // copy
        return batch;
    }

    // errors is length-B vector; updates internal state
    void update(uint32_t best_idx, const std::vector<T>& errors)
    {
        if (errors.size() != batch_size_) return;
        const T min_err = *std::min_element(errors.begin(), errors.end());
        error_hist_.push_back(min_err);

        const T* best = &last_batch_[static_cast<size_t>(best_idx) * 6];
        T delta[6];
        for (int i = 0; i < 6; ++i) delta[i] = best[i] - estimate_[i];
        // momentum
        for (int i = 0; i < 6; ++i) momentum_[i] = T(0.8) * momentum_[i] + T(0.2) * delta[i];
        // estimate
        for (int i = 0; i < 6; ++i) estimate_[i] = T(0.8) * estimate_[i] + T(0.2) * (T(0.5) * momentum_[i] + (T(0.5) * best[i] + T(0.5) * estimate_[i]));
        // smoothed
        for (int i = 0; i < 6; ++i) smoothed_[i] = (T(1) - smoothing_) * smoothed_[i] + smoothing_ * estimate_[i];

        // adapt radius/confidence
        if (best_idx < 3) { radius_ *= radius_dec_; confidence_ = std::min(T(1), confidence_ + T(0.05)); }
        else { radius_ *= radius_inc_; confidence_ = std::max(T(0), confidence_ - T(0.1)); }
        radius_ = std::min(max_radius_, std::max(min_radius_, radius_));

        if (error_hist_.size() > 5) {
            const size_t n = error_hist_.size();
            T mean = T(0);
            for (size_t i = n - 5; i < n - 1; ++i) mean += error_hist_[i];
            mean /= T(4);
            T stdv = T(0);
            for (size_t i = n - 5; i < n - 1; ++i) { T d = error_hist_[i] - mean; stdv += d * d; }
            stdv = std::sqrt(stdv / T(4));
            if (stdv < T(0.01)) radius_ *= T(0.9);
            else if (error_hist_.back() > T(1.5) * mean) { radius_ *= T(1.3); confidence_ *= T(0.5); }
            radius_ = std::min(max_radius_, std::max(min_radius_, radius_));
        }

        // random rotation for next exploration
        current_rot_ = random_rotation();
    }

    std::array<T,6> smoothed() const { return smoothed_; }
    T radius() const { return radius_; }
    T confidence() const { return confidence_; }

private:
    uint32_t batch_size_;
    std::array<T,6> estimate_{};
    std::array<T,6> momentum_{};
    std::array<T,6> smoothed_{};
    T radius_, min_radius_, max_radius_;
    T radius_inc_, radius_dec_;
    T smoothing_;
    T confidence_;

    // rotation matrix (row-major 3x3)
    std::array<T,9> current_rot_{};
    std::vector<T> sphere_dirs_; // (num_explore x 3)
    std::vector<T> last_batch_;
    std::vector<T> error_hist_;
    std::mt19937 rng_;

    static std::array<T,9> identity3()
    { return {T(1),T(0),T(0), T(0),T(1),T(0), T(0),T(0),T(1)}; }

    void fibonacci_sphere(uint32_t n, T* out)
    {
        if (n == 0) return;
        const T phi = (T(1) + std::sqrt(T(5))) / T(2);
        for (uint32_t i = 0; i < n; ++i) {
            T y = (n > 1) ? (T(1) - (T(2) * T(i) / T(n - 1))) : T(0);
            T r = std::sqrt(T(1) - y * y);
            T theta = T(2) * T(M_PI) * T(i) / phi;
            out[i*3 + 0] = r * std::cos(theta);
            out[i*3 + 1] = y;
            out[i*3 + 2] = r * std::sin(theta);
        }
    }

    std::array<T,9> random_rotation()
    {
        std::uniform_real_distribution<T> unif(T(0), T(1));
        T u1 = unif(rng_), u2 = unif(rng_), u3 = unif(rng_);
        T q1 = std::sqrt(T(1)-u1) * std::sin(T(2)*T(M_PI)*u2);
        T q2 = std::sqrt(T(1)-u1) * std::cos(T(2)*T(M_PI)*u2);
        T q3 = std::sqrt(u1) * std::sin(T(2)*T(M_PI)*u3);
        T q4 = std::sqrt(u1) * std::cos(T(2)*T(M_PI)*u3);
        T x=q1,y=q2,z=q3,w=q4;
        T xx=x*x, yy=y*y, zz=z*z;
        T xy=x*y, xz=x*z, yz=y*z;
        T wx=w*x, wy=w*y, wz=w*z;
        return {
            T(1) - T(2)*(yy + zz),     T(2)*(xy - wz),           T(2)*(xz + wy),
            T(2)*(xy + wz),           T(1) - T(2)*(xx + zz),     T(2)*(yz - wx),
            T(2)*(xz - wy),           T(2)*(yz + wx),            T(1) - T(2)*(xx + yy)
        };
    }
};

} // namespace gato


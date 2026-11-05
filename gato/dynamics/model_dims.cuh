#pragma once
// Dimensions GATO reads from the generated model descriptor (grid::model,
// grid-rbd 0.7.0) rather than from a rule of its own. Included by the plant
// adapter and by constants.h, which precede each other depending on the
// translation unit, so this header depends on the generated header only.
// grid.cuh carries no include guard; the plant adapter includes it first and
// this header is read after it on every path (plant.cuh line 4, then here).

namespace gato {
namespace plant {
        // The actuated torque width: the tangent size less the rows of the
        // floating coordinate block. A fixed-base plant actuates every row; a
        // free-flyer actuates every row but the base twist.
        __host__ __device__ constexpr int actuated_width()
        {
                int n = grid::model::nv;
                for (int k = 0; k < grid::model::coordinate_block_count; ++k) {
                        grid::model::coordinate_block_record b{};
                        if (grid::model::coordinate_block(k, b) && b.kind == grid::model::coordinate_kind::floating) n -= b.nv;
                }
                return n;
        }
        static_assert(grid::model::nv == grid::NUM_VEL, "the model descriptor and the generated constants must agree on the tangent size");
        static_assert(actuated_width() > 0 && actuated_width() <= grid::NUM_VEL, "the actuated width is the tangent size less the floating block");
}  // namespace plant
}  // namespace gato

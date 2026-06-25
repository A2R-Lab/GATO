# GLASS feature requests (from GATO)

Wishlist of GLASS primitives that would let GATO delete hand-rolled block-linear-algebra.
Written 2026-06-24 against GLASS `7e2354f` (post contraction-parallel family) and GATO
`cleanup-modernization`. Share into GLASS `docs/open-tasks/`.

## Framing (read first — honest scope)

After the GATO→GLASS unification (P4), GATO defers ~95% of its block linear algebra to
`glass::` (gemm/gemm_ex/copy/axpy/axpby/addI_partial/loadIdentity, the fused K-way
`invertMatrix`, `bdmv`, `pcg`). **GATO's runtime hot path is the line-search merit kernel
(~69% of GPU time at M=128) — that's dynamics rollouts (GRiD), not linear algebra — so none
of the requests below are headline-speed wins.** They are **maintainability + correctness +
modest-fusion** wins: each one deletes a chunk of GATO-specific hand-rolled CUDA that is
currently a silent-bug surface (the unification plan flagged the row-major-dense ↔ banded
boundary as "the highest-risk silent bug" in the whole migration).

Tiers below are by value-to-GATO. Each item cites where GATO hand-rolls it today.

---

## Tier 1 — Banded block scatter/gather (the one that matters)

**Gap.** GLASS owns the block-tridiagonal `[L|D|R]` layout *internally* for `bdmv`/`pcg`, but
exposes **no primitive to move a dense square block in/out of a banded strip**. So GATO
hand-rolls every write of the Schur system `S` and preconditioner `P_inv`, and every read
back — 6 thread-strided loops in `schur_linsys.cuh`, each doing an index remap between a dense
`STATE×STATE` block (row- or col-major) and a `BLOCK_ROW_R_DIM (= 3·STATE)`-strided banded
slot, usually **with a transpose and sometimes a negate**:

- `schur_linsys.cuh:142-151` — write `phi_kᵀ`, `phi_k`, `-theta_k` into three banded `S` slots (transpose).
- `schur_linsys.cuh:162-166` — write `-theta_k_inv` into the `P_inv` main diagonal (transpose + negate).
- `schur_linsys.cuh:186-199` — first-block-row variants.
- `schur_linsys.cuh:234-241` — gather `theta_k_inv`, `theta_km1_inv`, `phi_k` *out of* banded `P_inv`/`S` into dense blocks (transpose).
- `schur_linsys.cuh:255-...` — write left/right off-diagonals back into `P_inv` (transpose).

These index remaps (`y*BLOCK_ROW_R_DIM + x` ↔ `x*STATE_SIZE + y`) are exactly the kind of
thing a `glass::banded::` helper should own, and getting one wrong is an invisible-at-32-threads
race or a silent transpose bug.

**Proposed.** A small banded block-access family in `src/base/banded/` (alongside `bdmv`):

```cpp
// Store a dense d×d block into banded strip `dst` at (block_row, slot ∈ {LEFT,MAIN,RIGHT}),
// with optional transpose + scale. Strip stride = band_width (e.g. 3*d).
template <typename T, uint32_t d, uint32_t band_width, bool TRANSPOSE = false>
__device__ void store_block(T *dst_strip, BandSlot slot, const T *src, T scale = 1);

// Inverse: gather a banded slot into a dense d×d block (optional transpose + scale).
template <typename T, uint32_t d, uint32_t band_width, bool TRANSPOSE = false>
__device__ void load_block(T *dst, const T *src_strip, BandSlot slot, T scale = 1);
```

**Why it helps GATO.** Deletes all 6 hand-rolled remap loops and removes the riskiest part of
the Schur assembly. **This is the highest-value request** — it's squarely in GLASS's banded
module, it's where GATO's remaining hand-rolled linalg actually lives, and it closes the one
layout boundary the unification called highest-risk.

---

## Tier 2 — Ergonomic / fusion gaps

### 2a. `gemm` with a **non-square transposed** operand
**Gap.** `glass::gemm<...,TRANSPOSE_B=true>` requires **square B**. GATO's Schur has one
genuine non-square transposed product (`B_R_inv · B_kᵀ`, shapes S×C · (S×C)ᵀ), so it must drop
to `gemm_ex` and manually reinterpret B col-major as Bᵀ row-major:

```cpp
// schur_linsys.cuh:119 — wants gemm<...,TRANSPOSE_B>, forced into gemm_ex gymnastics
glass::gemm_ex<T,false,false,true,false>(STATE_SIZE,CONTROL_SIZE,STATE_SIZE, 1, s_B_R_inv, s_B_k, 1, s_theta_k);
```

**Proposed.** Let `gemm<...,TRANSPOSE_B>` (and `TRANSPOSE_A`) accept rectangular operands —
i.e. fold the `gemm_ex` rectangular-transpose path into the friendly `gemm` template so callers
don't reason about col-major-as-transpose. Pure ergonomics; `gemm_ex` stays as the explicit
escape hatch.

### 2b. Fused congruence-accumulate `C += α · G · M · Gᵀ`
**Gap.** GATO forms each Schur diagonal block as `theta_k = A·Q_inv·Aᵀ + B·R_inv·Bᵀ + Q_kp1_inv`
via **two temporaries + an extra barrier**:

```cpp
// schur_linsys.cuh:111-119
glass::gemm(1, s_A_k, s_Q_k_inv, s_A_Q_inv);     // temp A·Q_inv
glass::gemm(1, s_B_k, s_R_k_inv, s_B_R_inv);     // temp B·R_inv
__syncthreads();
glass::gemm<...,true>(1, s_A_Q_inv, s_A_k, 1, s_theta_k);            // + A_Q_inv·Aᵀ
glass::gemm_ex<...>(...,1, s_B_R_inv, s_B_k, 1, s_theta_k);          // + B_R_inv·Bᵀ
```

The new `congruence_sym` (XᵀMX) / `bilinear` (XᵀMY) are close cousins but the orientation is
`G·M·Gᵀ` with M **already inverted** and G **general/rectangular** (not the symmetric XᵀMX
form). A `congruence_accum<...>(scale, G, M, C, beta)` covering `C = β·C + scale·G·M·Gᵀ`
(M symmetric, G rectangular) would collapse the temp+gemm pair into one call and drop a barrier.

**Caveat:** modest win (saves one temp buffer + one barrier per knot); list it as nice-to-have,
not urgent. Only worth it if it composes cleanly with 2a (the rectangular-transpose support).

### 2c. `asum` / fused L1-norm-of-difference
**Gap.** The merit kernel's initial-state constraint error hand-rolls abs-diff + reduce:

```cpp
// merit.cuh:79-83
for (i = threadIdx.x; i < STATE_SIZE; i += blockDim.x)
    s_temp[i] = abs(d_xu_k[i] + alpha*d_dz_k[i] - d_x_initial_k[i]);
__syncthreads();
block::reduce<T>(STATE_SIZE, s_temp);   // GATO's own reduce
```

**Proposed.** A BLAS-style `glass::asum<T,N>(x)` (Σ|xᵢ|), and optionally a fused
`nrm1_diff<T,N>(a, b)` (Σ|aᵢ−bᵢ|). Small, but `asum` is a standard L1 primitive GLASS lacks
and it removes a hand-rolled reduce on the hot kernel.

---

## Tier 3 — Already shipped in `7e2354f`; GATO should *adopt* (no GLASS work)

Noting these so they're not mistaken for gaps — the recent GLASS surface already covers them;
the action is on the GATO side:

- **`glass::warp::reduce`** satisfies `merit.cuh:81`'s explicit `// TODO: use warp reduce
  instead` (GATO still calls its block `reduce` there).
- **`glass::axpby`** already covers the line-search trial step `xu + α·dz`
  (`merit.cuh:58-60`, hand-rolled) and `s_temp[i] = abs(...)`'s axpb part.
- **`glass::set_const<T,N>(0, …)`** covers the trivial zero-fills at `schur_linsys.cuh:358,388`.
- **`invertMatrix_pivoted`** is available as a robustness drop-in for the Schur inverts if
  near-singular regularized blocks ever cause PCG instability (same augmented convention,
  `3·dim+1` scratch).

These are tracked as GATO-side cleanup, not requests.

---

## Explicitly NOT requesting (and why)

- **Block-tridiagonal direct solver** (block-Thomas / cyclic reduction). GATO deliberately uses
  **PCG** for the batched, warm-started regime; a direct factorization doesn't fit that design.
- **`*_reduced` contraction-parallel ops** for the Schur blocks. GLASS's own measurement
  (`bench/REDUCED_SWEEP_RESULTS.md`, sm_120) shows `*_reduced` is **slower than serial** at
  GATO's tiny block sizes (STATE=12, CONTROL=6); `suggested_use_reduced` is false there.
- **Anything for the merit/line-search rollout.** That's dynamics (GRiD), GATO's actual
  bottleneck, and out of GLASS's linear-algebra scope.

---

## Priority summary

| # | Request | Value | Effort (GLASS) |
|---|---------|-------|----------------|
| 1 | `glass::banded::{store,load}_block` (transpose+scale) | **High** — deletes GATO's riskiest hand-rolled code | Small–medium |
| 2a | `gemm<...,TRANSPOSE>` for rectangular operands | Medium (ergonomics) | Small |
| 2b | `congruence_accum` `C += α·G·M·Gᵀ` (G rectangular) | Low–medium (1 temp + 1 barrier/knot) | Medium |
| 2c | `asum` / `nrm1_diff` | Low | Trivial |

Everything else GATO needs from GLASS is already shipped.

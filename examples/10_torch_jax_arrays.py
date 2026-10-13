"""Intro example 10: the solver as a batched tensor op (torch / JAX in, same kind out).

`solve()` takes numpy, torch or JAX arrays. CUDA tensors never leave the device:
the solver reads them in place through their device pointer and returns the
trajectory batch through DLPack, bit-identical to the numpy path (cold and
warm-started). That is the primitive a learning or sampling loop wants: B
trajectory-optimization problems with per-row initial states as one op on
tensors that already live on the GPU.

Needs the bsqpN64_indy7 module (built in the checkout, or compiled into the
cache on first use) and torch with CUDA; the JAX half runs when a CUDA JAX is
installed. Runs from any cwd:
    python examples/10_torch_jax_arrays.py
"""
import numpy as np
import torch

import gato
from gato import BSQP, SolverParams
from gato.common import figure8
from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS

N, DT, B = 64, 0.01, 16
URDF = gato.robot_urdf("indy7")
solver = BSQP(URDF, batch_size=B, N=N, dt=DT, params=SolverParams(max_sqp_iters=5), plant_type="indy7")

# B initial states: the ready pose with a random perturbation per batch entry
rng = np.random.default_rng(0)
x0 = np.tile(np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(solver.nv))), (B, 1)).astype(np.float32)
x0[:, : solver.nq] += rng.normal(0.0, 0.05, size=(B, solver.nq)).astype(np.float32)
ref = np.tile(figure8(DT, **FIG8_DEFAULT_PARAMS)[: 6 * N].astype(np.float32), (B, 1))

ref_np = solver.solve(x0, ref)                                  # numpy path, for comparison

assert torch.cuda.is_available(), "this example needs a CUDA torch build"
x0_t, ref_t = torch.from_numpy(x0).cuda(), torch.from_numpy(ref).cuda()
solver_t = BSQP(URDF, batch_size=B, N=N, dt=DT, params=SolverParams(max_sqp_iters=5), plant_type="indy7")
res = solver_t.solve(x0_t, ref_t)                               # device path: no host round trip
print(f"torch: xu is a {type(res.xu).__name__} on {res.xu.device}, shape {tuple(res.xu.shape)}, "
      f"max |Δ| vs numpy path = {(res.xu.cpu().numpy() - ref_np.xu).__abs__().max():.1e}")
warm = solver_t.solve(x0_t, ref_t, xu_warm=res.xu)              # warm start stays on the device
print(f"torch warm start: sqp_iters {warm.stats.sqp_iters.tolist()[:4]}..., first controls {warm.u0(0).cpu().numpy().round(2)}")

try:
    import jax
    import jax.numpy as jnp
    if any(d.platform == "gpu" for d in jax.devices()):
        solver_j = BSQP(URDF, batch_size=B, N=N, dt=DT, params=SolverParams(max_sqp_iters=5), plant_type="indy7")
        res_j = solver_j.solve(jnp.asarray(x0), jnp.asarray(ref))
        print(f"jax: xu is a {type(res_j.xu).__name__} on {next(iter(res_j.xu.devices()))}, "
              f"max |Δ| vs numpy path = {float(jnp.abs(res_j.xu - jnp.asarray(ref_np.xu)).max()):.1e}")
    else:
        print("jax: installed without a GPU backend — JAX CPU arrays take the host path")
except ImportError:
    print("jax: not installed (pip install 'jax[cuda13]' to run the JAX half)")

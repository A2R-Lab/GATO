"""Array-framework boundary: numpy, torch and JAX in, the same kind out.

The solver has two entry points: the host path (numpy / CPU tensors, staged
through pinned memory) and the device path (CUDA tensors read in place through
their device pointer, results returned through DLPack without a host round
trip). ``kind()`` classifies an input, ``device_f32()`` validates a CUDA array
for the device path, ``from_device()`` wraps a solver DeviceArray in the
caller's framework.
"""
from __future__ import annotations

import sys

import numpy as np


def kind(x) -> str:
    """'torch' | 'jax' | 'numpy' (anything array-like that numpy accepts)."""
    mod = type(x).__module__.split(".")[0]
    if mod == "torch":
        return "torch"
    if mod in ("jax", "jaxlib"):
        return "jax"
    return "numpy"


def is_cuda(x) -> bool:
    """True for a torch CUDA tensor or a JAX array on a CUDA device."""
    k = kind(x)
    if k == "torch":
        return bool(x.is_cuda)
    if k == "jax":
        try:
            dev = next(iter(x.devices()))
        except Exception:
            return False
        return getattr(dev, "platform", "") == "gpu"
    return False


def to_numpy(x, dtype=np.float32) -> np.ndarray:
    """Host copy as a contiguous numpy array (CPU or CUDA input of any kind)."""
    k = kind(x)
    if k == "torch":
        return np.ascontiguousarray(x.detach().cpu().numpy(), dtype=dtype)
    return np.ascontiguousarray(np.asarray(x, dtype=dtype))


def device_f32(x, numel: int, name: str):
    """Validate a CUDA array for the device path; return (device_pointer, keepalive).

    The array must be float32 and contiguous — torch inputs are converted when
    they are not (a copy, still on the device); JAX arrays are immutable and
    already contiguous, so a wrong dtype is converted with ``astype``."""
    k = kind(x)
    if k == "torch":
        t = x.detach()
        if t.dtype != torch_module().float32:
            t = t.to(torch_module().float32)
        t = t.contiguous()
        if t.numel() != numel:
            raise ValueError(f"{name}: expected {numel} elements, got {t.numel()} (shape {tuple(t.shape)})")
        return t.data_ptr(), t
    if k == "jax":
        jnp = sys.modules["jax"].numpy
        a = x if x.dtype == jnp.float32 else x.astype(jnp.float32)
        if a.size != numel:
            raise ValueError(f"{name}: expected {numel} elements, got {a.size} (shape {tuple(a.shape)})")
        # JAX dispatches producers (including slices/casts) asynchronously on
        # its own stream. Our pointer-only native entry points run on the CUDA
        # default stream and do not consume the CAI producer-stream field.
        # Wait for this input, not the whole device, before handing it off.
        a.block_until_ready()
        cai = a.__cuda_array_interface__
        return cai["data"][0], a
    raise TypeError(f"{name}: device path needs a torch CUDA tensor or a JAX CUDA array, got {type(x).__name__}")


def device_rows(x, row: int, batch: int, name: str):
    """Validate a CUDA array holding one row of ``row`` elements (broadcast to the
    batch) or exactly ``batch`` rows; return (device_pointer, keepalive, batched)."""
    k = kind(x)
    n = int(x.numel()) if k == "torch" else int(x.size)
    if n == row:
        batched = False
    elif n == row * batch:
        batched = True
    else:
        raise ValueError(f"{name}: expected {row} elements (one row) or {row * batch} (batch_size {batch} rows), got {n}")
    ptr, keep = device_f32(x, n, name)
    return ptr, keep, batched


def from_device(device_array, like_kind: str):
    """Wrap a solver DeviceArray (DLPack exporter) in the caller's framework, zero-copy."""
    if like_kind == "torch":
        return torch_module().from_dlpack(device_array)
    if like_kind == "jax":
        import jax.dlpack
        return jax.dlpack.from_dlpack(device_array)
    return device_array.numpy()


def like(x_np: np.ndarray, like_kind: str, device=None):
    """A host result in the caller's framework (CPU tensors for CPU inputs)."""
    if like_kind == "torch":
        t = torch_module().from_numpy(np.ascontiguousarray(x_np))
        return t.to(device) if device is not None else t
    if like_kind == "jax":
        jnp = sys.modules["jax"].numpy
        return jnp.asarray(x_np)
    return x_np


def torch_module():
    import torch

    return torch

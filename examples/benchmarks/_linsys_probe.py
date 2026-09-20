"""Linsys probe machinery — thin re-export of the shared `_bench` helpers.

The kicked-arm fig8 MPC probe (`run_arm`), the fig8 task definition
(`start_and_ref`) and the GPU-busy check live in `_bench` since 2026-09-20;
this module keeps the historical names for linsys_auto_cdf.py and
tools/autotune_linsys.py. TIMING-CLASS: callers own the quiet-box guard.
"""
from _bench import (FIG8_PLANTS, gpu_busy, urdf_path,  # noqa: F401
                    fig8_task as start_and_ref, run_kicked_arm as run_arm)

URDFS = {p: urdf_path(p) for p in FIG8_PLANTS}

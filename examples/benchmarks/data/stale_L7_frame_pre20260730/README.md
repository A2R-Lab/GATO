# STALE: pre-2026-07-30 L7-frame data — DO NOT MIX

Everything here was measured BEFORE the 2026-07-30 named-target regen
(`fixed_target_name="EE"`): the solver cost, the MPCGPU trajfiles, and the
tracking metric all moved from the L7 link frame to the URDF "EE" fixed joint
(+0.04 m), and the GLASS/GRiD bump shifted dynamics ULPs besides. The fig3
assembler (`reproduce_fig3_fair.py`) takes the LAST row per (N,B) from the
sweep CSVs, so stale rows silently coexisting with fresh ones would corrupt
the table — hence this rotation, not deletion.

- `sweep_fig8_{gato,bt,mpcgpu}.csv` — 2026-07-08 fig3-fair sweeps (L7 frame;
  `L7_mean` column). Fresh CSVs regenerate at `../sweep_fig8_*.csv` with an
  `EE_mean` column via the Phase-3 night runner.
- `benchmark_fig8_*.pkl` — 2026-06 benchmark_fig8.py pkls, stale-in-kind.

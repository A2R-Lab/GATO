# STALE: pre-Drake-swap iiwa14 fig7 pools (rotated 2026-08-03)

This is an archive manifest for local, gitignored experiment outputs. The named
pickles and plots are not distributed with the repository. Numbers below record
the historical analysis and should not be treated as current benchmark results.

These pools were generated BEFORE the Wave-H iiwa14 Drake-family inertial swap
(GATO @74df831 / attest @67f461a, 2026-08-02 night). The arm is ~5 kg heavier with
a different mass distribution, so every closed-loop success/time number here is for
a different robot. **Do not mix with post-swap pools** (the do-not-mix rule that
already applies to the pre-EE-frame data in `stale_L7_frame_pre20260730/`).

Contents (eased protocol: mass 10 kg, length 0.5-0.7 m, damping 0.3-0.6, angle
0.0-0.6 rad, seed 0, B = 1,4,8,16,32,64,128, 100 scenarios):
- `fig7_pickplace_eased.pkl` — the 100x7 record pool; Table I read
  11/21/74/76/81/85/85% success, monotone, all high-B failures 4/5-goal near-misses.
- `fig7_eased_probe.pkl` — the 25-scenario probe that picked the eased protocol
  (8/68/84% at B = 1/16/128).
- the matching CDF plots + Table-I text.

The post-swap re-run uses the SAME tag (`fig7_pickplace_eased`) and the SAME
protocol, so the CL-3a W3 comparison is apples-to-apples on the Drake model.

## SUPERSEDED 2026-08-03 by the post-swap re-run

`../fig7_pickplace_eased.pkl` (2026-08-03 12:46) is the FE baseline of record:
**4/11/82/87/87/87/86%** vs this pool's 11/21/74/76/81/85/85%. The 100 scenario dicts
are identical (seed 0) so the two are properly paired; 82/100 scenarios flipped outcome
at ≥1 batch size, low B degraded and B≥8 improved, and batching now saturates at B≈16
rather than B≈64. No single-B delta was significant on its own in that analysis.
Nothing here should be quoted except as the pre-swap contrast.

# Contact-wipe paired pool (n=24 scenarios, F_set=25 N)

| Metric | pos | ucone | fc | fc vs pos p | fc vs ucone p |
|---|---|---|---|---|---|
| Normal-force RMS error [N] | 9.440 | 19.856 | **4.538** | 1.2e-07 | 1.2e-07 |
| Path RMS error [mm] | 25.595 | 40.634 | **21.292** | 1.2e-02 | 1.2e-07 |
| Friction-cone violation (mean) [N] | 1.375 | 0.274 | **0.203** | 1.2e-07 | 2.9e-01 |
| Contact loss [%] | 0.443 | 11.346 | **0.459** | 3.9e-01 | 1.2e-07 |
| Solve time (mean) [ms] | 0.161 | 0.572 | **0.378** | 1.2e-07 | 1.2e-07 |

Paired means over the 24-scenario pool; Wilcoxon signed-rank p-values on the paired per-scenario differences. Solve times are pool means from a SHARED box — quote the quiet-box numbers in text: pos 0.152 ms, fc 0.37 ms per solve (fc ≈ 2.4× pos, both > 2.7 kHz).

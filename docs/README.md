# GATO documentation

These pages are also published as the reference tier of the project site,
<https://a2r-lab.org/GATO/docs/>, with the API reference and the current receipt.

Start with the [quick start](../README.md#quick-start),
then run the [numbered examples](../examples/README.md). Build only the module
needed by your example; the full receipt profile is for validation and research.

| Question | Reference |
|---|---|
| What works, and what is experimental? | [Feature status and limitations](status.md); [API stability](consumer_contract.md#9-api-stability) |
| How do I use the Python API or migrate old code? | [Consumer contract and migration notes](consumer_contract.md) |
| How do I add limits, cones, collision or foot rows? | [Constraint mechanisms and conventions](constraints.md) |
| How do I install from a wheel and use my own URDF? | [register_robot and the per-host cache](development.md#installing-from-a-wheel-register_robot-and-the-per-host-cache), [example 05](../examples/05_build_your_robot.py) |
| How do I use torch or JAX arrays? | [Array kinds](consumer_contract.md#4-solve-interface) |
| How do I use the solver from C++/CUDA? | [C++/CUDA interface](consumer_contract.md#8-ccuda-interface), [examples/cpp](../examples/cpp/) |
| How do I add a robot or read the dynamics layer? | [Dynamics adapter](../gato/dynamics/README.md) |
| How do I build, test, sign a receipt or time a change? | [Development and validation](development.md) |
| What block size and launch does the merit kernel run at, and why? | [Merit launch timing](merit_timing.md) |
| Which results reproduce the paper? | [Figure protocols, provenance and caveats](../examples/paper-figures/README.md) |
| What do the current-code figures show? | [Figure refresh, October 2026](figure-refresh-2026-10-01.md) |
| How is the project page updated? | [Website](website.md) |
| Where did historical datasets come from? | [Dated archaeology](archaeology.md) — history, not current API guidance |

The [project website](https://a2r-lab.org/GATO/) presents the published experiments and the
latest current-code results, each labeled. It is not a live benchmark dashboard. A green correctness receipt does not
certify runtime speed, arbitrary robot models or safety on physical hardware.

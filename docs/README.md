# GATO documentation

Start with the [quick start](../README.md#quick-start-host-native--no-docker-needed),
then run the [numbered examples](../examples/README.md). Build only the module
needed by your example; the full receipt profile is for validation and research.

| Question | Reference |
|---|---|
| What works, and what is experimental? | [Feature status and limitations](status.md) |
| How do I use the Python API or migrate old code? | [Consumer contract and migration notes](consumer_contract.md) |
| How do I add limits, cones, collision or foot rows? | [Constraint mechanisms and conventions](constraints.md) |
| How do I add a robot or use the native solver? | [Dynamics adapter](../gato/dynamics/README.md), [C++/CUDA example](../examples/bsqp.cu) |
| How do I build, test, sign a receipt or time a change? | [Development and validation](development.md) |
| Which results reproduce the paper? | [Figure protocols, provenance and caveats](../examples/paper-figures/README.md) |
| What do the current-code figures show? | [Figure refresh, October 2026](figure-refresh-2026-10-01.md) |
| How is the project page updated? | [Website](website.md) |
| Where did historical datasets come from? | [Dated archaeology](archaeology.md) — history, not current API guidance |

The [project website](https://a2r-lab.org/GATO/) presents the published experiments and the
latest current-code results, each labeled. It is not a live benchmark dashboard. A green correctness receipt does not
certify runtime speed, arbitrary robot models or safety on physical hardware.

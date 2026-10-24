# GATO documentation

GATO is a GPU batched trajectory optimizer and MPC controller for robot arms and
floating-base robots: a batched SQP with PCG or block-Cholesky linear solves,
rigid-body dynamics generated per robot by [GRiD](https://a2r-lab.org/GRiD/),
numpy, torch, JAX and C++/CUDA interfaces, and a constraint layer that keeps
plans inside the URDF limits. The [project page](https://a2r-lab.org/GATO/)
presents the paper and the latest results; this site is the reference: how to
install and use the library, what it guarantees, and what it does not.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Quick start and usage
:link: readme
:link-type: doc
`pip install gato-mpc`, the first solve, the controller, the constraint layer,
adding a robot, the C++/CUDA interface.
:::

:::{grid-item-card} Examples
:link: examples
:link-type: doc
Fourteen numbered scripts from one solve to MuJoCo in the loop; the paper
figures and the benchmark harnesses.
:::

:::{grid-item-card} Consumer contract
:link: consumer_contract
:link-type: doc
The conventions an integration depends on: frames, limits, cost semantics,
the solve interface, array kinds, API stability.
:::

:::{grid-item-card} Feature status and limits
:link: status
:link-type: doc
What is attested, what is experimental, the operating limits, and what the
receipt proves.
:::

:::{grid-item-card} Constraint layer
:link: constraints
:link-type: doc
Mechanisms, measured defaults and provenance for every row-group method:
boxes, cones, collision, contact rows, the ADMM inner loop.
:::
:::{grid-item-card} Cost menu
:link: cost_menu
:link-type: doc
Objectives beyond quadratic tracking: frame position and orientation, state,
input, linear features, twist, under per-knot activations and barriers.
:::

:::{grid-item-card} Signed receipt
:link: receipt
:link-type: doc
The current GPU test receipt: commit, outcome, module set, signer, and how
to verify it yourself.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
`BSQP`, `MPCController`, `MPCPolicy`, the hypothesis batch, tasks, worlds,
the builder and the fingerprint, from the docstrings.
:::

:::{grid-item-card} Development and validation
:link: development
:link-type: doc
Build, tests, the receipt procedure, timing windows, releasing.
:::
::::

```{toctree}
:maxdepth: 1
:hidden:

readme
examples
guide
consumer_contract
status
constraints
cost_menu
merit_timing
receipt
api
development
dynamics
paper_figures
figure_refresh
changelog
archaeology
website
```

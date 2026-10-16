# API reference

The public surface is `import gato`: the solver (`BSQP`), its configuration
(`SolverParams`), the controller and policy layers, the hypothesis batch, the
task generators, the simulation worlds, the robot builder and the dynamics
fingerprint. Everything below is rendered from the docstrings in the installed
package; the conventions they rely on are in the [consumer contract](consumer_contract.md).

## Solver

```{eval-rst}
.. automodule:: gato.interface
   :members: BSQP, SolveResult, SolverStats, available, module_name, robot_info, robot_urdf
```

```{eval-rst}
.. automodule:: gato.config
   :members: SolverParams
```

## Controller and policy

```{eval-rst}
.. automodule:: gato.controller
```

```{eval-rst}
.. automodule:: gato.policy
```

```{eval-rst}
.. automodule:: gato.hypotheses
```

```{eval-rst}
.. automodule:: gato.estimators
```

## Tasks, worlds and environments

```{eval-rst}
.. automodule:: gato.tasks
```

```{eval-rst}
.. automodule:: gato.worlds
```

```{eval-rst}
.. automodule:: gato.envs
```

```{eval-rst}
.. automodule:: gato.mpc_gato
```

## Robots: codegen, cache and fingerprint

```{eval-rst}
.. automodule:: gato.builder
   :members: build, codegen, canonicalize_inertials, rotated_inertial_origins, load_registry
```

```{eval-rst}
.. automodule:: gato._cache
   :members: register_robot
```

```{eval-rst}
.. automodule:: gato.fingerprint
```

## Linear-system policy and row kinds

```{eval-rst}
.. automodule:: gato.linsys_autotune
```

```{eval-rst}
.. automodule:: gato.rowkinds
```

# Objective architecture

The active target is implemented by `ShellThicknessTerm` in `objectives.py` and
registered under:

```text
shell_thickness
```

Configuration:

```toml
[objective]
shell_thickness_temperature = 0.002

[[objective.terms]]
name = "shell_thickness"
weight = 1.0

[[objective.terms]]
name = "equal_spacing"
weight = 0.0002

[[objective.terms]]
name = "bending"
weight = 0.00002
```

## Two thickness values

`ShellThicknessTerm.value()` returns the negative smooth thickness used by
Autograd and SLSQP. The smooth minimum is a Boltzmann-weighted average of all
inner/outer segment distances. It lets several nearly active segment pairs
contribute to the gradient instead of switching abruptly between pairs.

`exact_shell_thickness()` computes the true minimum Euclidean distance between
all closed inner and outer boundary segments. It is used for diagnostics and is
not differentiated.

Accepted states are kept topologically valid by the collision solver. This is
important because the differentiable endpoint-based segment formula assumes the
inner and outer boundaries are disjoint. If a proposed state crosses, exact
validation rejects it before it becomes the current state.

## Adding another term

An additional objective term implements:

```python
@dataclass(frozen=True)
class MyTargetTerm:
    name: str = "my_target"

    def value(self, state, assembly):
        return 0.0
```

Register it in `objective_factory.py`, then add a `[[objective.terms]]` row in
`settings.toml`. The solver remains independent of the term implementation.

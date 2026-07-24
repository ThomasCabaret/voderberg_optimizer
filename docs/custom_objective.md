# Adding a new optimization target

An objective term implements the protocol used in `objectives.py`:

```python
from dataclasses import dataclass
from typing import Any

from voderberg_optimizer.parameterization import TileAssembly
from voderberg_optimizer.state import SRN2State


@dataclass(frozen=True)
class MyTargetTerm:
    name: str = "my_target"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        # Return a scalar to minimize.
        return 0.0
```

Register it in `objective_factory.py`:

```python
TERM_FACTORIES["my_target"] = lambda settings: MyTargetTerm()
```

Then select it in `settings.toml`:

```toml
[[objective.terms]]
name = "my_target"
weight = 1.0
```

## Future minimum shell thickness

The intended target is conceptually:

```text
maximize minimum thickness of the external two-copy shell
```

Since the optimizer minimizes, the term would return the negative of a smooth or
auxiliary-variable representation of that thickness.

Before implementing it, the missing third/global contour must be constructed.
`TileAssembly.additional_contours` already allows the parameterization to expose
that contour without changing the solver or viewer.

A direct `min(distance)` is nondifferentiable when the active closest pair
changes. Better implementations include:

- an auxiliary thickness variable with separate inequalities;
- a smooth minimum during exploratory optimization, followed by exact checking;
- an active-set formulation containing only locally relevant shell pairs.

Collision validity should remain a solver constraint, not be folded into the
shell-thickness objective.

# Revision C notes

## Responsive display

The previous application invoked the complete SciPy minimization in the same
thread that owned Pygame. Pygame events were therefore processed only after a
local solve returned, making the window appear frozen.

The application now keeps Pygame in the main thread and runs the solver in a
worker thread. Accepted states are transferred through a bounded queue. The
viewer supports zoom, pan, fit, resize, and visibility controls without changing
or interrupting the numerical state.

## Strict accepted-state collision policy

The previous revision depended mainly on segment distance. Revision C first
classifies every segment pair with orientation predicates:

- `proper`: transverse crossing;
- `touch`: endpoint or tangent contact;
- `overlap`: collinear overlap;
- `none`.

Every forbidden non-adjacent pair must be `none`. This check is used for every
accepted endpoint and during adaptive path validation.

Cross-contour validation is now enabled by default. Initial zero-distance pairs
are treated as intentional contacts, but an excluded pair is still invalid if
it changes into a proper transverse crossing.

## Backtracking instead of committing an invalid proposal

SLSQP produces a proposed vector. The solver no longer treats that proposal as
an all-or-nothing step. If it is invalid, the solver tries fractions
`1, 1/2, 1/4, ...` until it finds a feasible, objective-improving prefix or the
configured minimum fraction is reached. No intersecting proposal is committed.

The final returned vector undergoes a defensive feasibility check.

## Adaptive path guard

Uniform finite sampling could miss a short-lived contact. Revision C recursively
checks midpoint states and uses a segment-clearance versus maximum-vertex-motion
bound to certify intervals far from collision. If the configured sample/depth
budget is exhausted, the step is rejected conservatively.

This is stronger than uniform sampling but is still not presented as a symbolic
continuous-collision proof for the nonlinear theta parameterization.

## Objective architecture

Objective construction moved out of `app.py` into `objective_factory.py`.
`settings.toml` now contains an array of weighted named terms. The numerical
solver only sees the scalar `OptimizationProblem` interface.

`TileAssembly` also accepts `additional_contours`, and collision/display code
iterates over an arbitrary number of contours. This prepares the project for a
third piece and a future outer-shell-thickness target.

## Compatibility

The old flat objective weights remain accepted by `config.py` when no
`[[objective.terms]]` array is present. The historical soft barrier remains
available but disabled by default.

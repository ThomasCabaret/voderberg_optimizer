# Geometry model

## Exact dependency propagation

The free state contains only `theta`, the four ordered point chains `X`, `P`,
`Q`, `Y`, and the point `B`. `SRN2Parameterization.build()` constructs all
other vertices with translations, half-turns, rotations, and order reversal.
Consequently, repeated points and transformed images are not separate decision
variables and cannot drift apart numerically.

## Additional free points

`X`, `P`, `Q`, and `Y` are variable-length chains. A point inserted into one of
these chains is initialized on a selected segment and then becomes an
independent optimization coordinate. Every occurrence of its image in the
three-piece assembly is still produced by the same exact parameterization.

The TOML `refinement` section lists segments that receive one midpoint. Repeating
a segment index inserts several evenly spaced points into that segment. For
example:

```toml
[refinement]
p_segments = [0, 0]
q_segments = [1]
```

This inserts two points in segment 0 of `P` and one point in segment 1 of `Q`.
The state layout is recomputed after refinement, so no vector slice is hardcoded.

## Separation of roles

- `state.py`: free variables and vector encoding.
- `parameterization.py`: exact assembly construction.
- `geometry.py`: local geometric primitives.
- `constraints.py`: validity margins and barrier construction.
- `objectives.py`: replaceable optimization targets.
- `problem.py`: solver-neutral scalar function and automatic gradient.
- `solvers/`: interchangeable numerical backends.

## Future additional contours

`TileAssembly` stores the currently reconstructed main and left contours and an
optional `additional_contours` tuple. Display, collision enumeration, and the
objective interface consume the generic `assembly.contours` property. A future
third/right contour can therefore be exposed by the parameterization without
changing those subsystems.

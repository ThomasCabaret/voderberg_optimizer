# Three-piece shell topology

## Poles and complete pieces

The normalized poles are:

```text
P0 = SOUTH = (0, -1)
P1 = NORTH = (0,  1)
```

`SRN2Parameterization.build()` reconstructs three complete tile contours from
every free state:

```text
assembly.main_contour   reference tile
assembly.left_contour   rotated/translated surrounding copy
assembly.right_contour  half-turn image of the reference tile
```

The right copy is the central symmetry around the midpoint of `P0` and `P1`:

```text
right_contour = -main_contour
```

## Shared seams

The surrounding copies share:

```text
lower seam: P0 -> P0'
upper seam: P1 -> P1'
```

The lower seam follows the current `Q` discretization and the upper seam follows
`P`. Both exact images are retained in `SharedChain` for validation.

## Shell boundaries

```python
assembly.shell.inner_boundary
assembly.shell.outer_boundary
```

The inner boundary is the complete reference-tile contour. The outer boundary is
assembled analytically from:

```text
P0' --left_outer_arc--> P1'
P1' --right_outer_arc-> P0'
```

No polygon union or tolerance-based boundary discovery is performed during an
objective evaluation.

## Optimization and display views

`assembly.contours` now contains all three complete physical pieces so the
collision solver validates the same geometry used by the thickness objective.
`assembly.piece_contours` provides the three-piece display view. The shell
objective reads the named inner and outer boundaries directly.

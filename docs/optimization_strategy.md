# Shell-thickness optimization strategy

The compact state remains `(theta, X, P, Q, Y, B)`. The reference tile, both
surrounding copies, both shared seams, and both shell boundaries are reconstructed
from that state at every evaluation.

## Primary objective

Let `C_in` be `assembly.shell.inner_boundary` and `C_out` be
`assembly.shell.outer_boundary`. The geometric target is:

```text
maximize min distance(C_in, C_out)
```

The local solver minimizes a smooth approximation of its negative. The exact
polyline distance is evaluated after accepted iterations and printed as
`thickness=...`.

## Regularization

The only additional active terms are weak mesh-quality energies:

```text
0.0002  * equal_spacing_energy
0.00002 * bending_energy
```

They encourage evenly distributed control points and suppress gratuitous
zigzags without replacing the shell-thickness target.

## Hard feasibility constraints

The objective contains no 45-degree angle preference and no legacy barrier.
Feasibility remains the solver's responsibility:

- no forbidden self-intersection;
- no forbidden collision among the three pieces;
- minimum generated-edge length;
- coordinate bounds;
- conservative validation of the continuous accepted path.

Initial exact interfaces and seams are registered as intentional contacts.

## Solver phases

1. Raise a small technical forbidden-segment clearance by continuation.
2. At each accepted step, rebuild separators and validate the endpoint/path.
3. Once the clearance schedule is complete, continue local optimization at
   fixed clearance for `objective_refinement_stages` stages.

The clearance is a safety constraint, not a competing geometric objective.

# Optimization strategy

The compact optimization state is `(theta, X, P, Q, Y, B)`. All repeated images
are exact consequences of `SRN2Parameterization`; equality constraints between
copies are never passed to SciPy.

## Local subproblem

At the last accepted geometry, nearby forbidden segment pairs receive fixed
separating normals. SLSQP minimizes the configured composite objective inside a
box trust region while enforcing:

- active separator inequalities;
- minimum generated-edge length;
- coordinate bounds.

The separators are only local approximations. They accelerate the solve but are
not trusted for final acceptance.

## Acceptance filter

A proposal is accepted only after:

1. exact 2D forbidden-intersection classification at the endpoint;
2. endpoint clearance validation;
3. minimum-edge and coordinate validation;
4. conservative adaptive path validation;
5. objective improvement.

If the full proposal fails, geometric backtracking tries progressively smaller
fractions. A failed pass reduces the trust radius.

## Continuation

The hard clearance target is increased in small stages. Each stage is warm
started from the last accepted geometry. When a stage fails, the clearance
increment and trust radius are reduced.

This favors slow deformation inside the valid topology instead of direct search
through invalid configurations.

## Objective composition

The objective is a weighted sum selected in TOML. The default is equivalent to:

```text
1.0    * negative_contact_length
0.01   * negative_mean_angle
0.002  * equal_spacing_energy
0.0002 * bending_energy
```

The solver has no dependency on these term names. A future shell-thickness term
can replace the contact-length term without modifying continuation or collision
handling.

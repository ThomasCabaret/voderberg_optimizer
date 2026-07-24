# Voderberg SRN2 optimizer - interactive strict-feasibility edition

This archive contains the complete refactored project, including the exact SRN2
parameterization, free-point refinement, a responsive interactive viewer, SVG/STL
export, and a topology-conscious continuation solver based on SciPy SLSQP.

The solver is intentionally local: it continuously deforms a known valid SRN2
solution instead of searching directly through arbitrary, frequently tangled
polygons.

## 1. Required input

The historical initialization file is not included because it was not supplied:

```text
voderberg_srn2_angles45_contact_optimV4.init
```

Copy it beside `settings.toml`. The default historical layout is:

```text
X = 7 points
P = 3 points
Q = 3 points
Y = 3 points
```

## 2. Windows installation

From the extracted project directory, run:

```bat
install_windows.bat
```

Equivalent PowerShell commands are:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

The dependencies are ordinary `pip` packages: NumPy, SciPy, Autograd, Pygame,
Pytest, and `tomli` for Python 3.10. No Docker image, external solver executable,
or compiler toolchain is required.

Python 3.10 or later, 64-bit, is recommended on Windows.

## 3. Run the optimizer

### Optimization with the interactive viewer

```bat
run_optimize.bat
```

Equivalent command:

```powershell
.\.venv\Scripts\python.exe -m voderberg_optimizer.cli optimize --settings settings.toml
```

### Optimization without a window

```bat
run_optimize_no_display.bat
```

Equivalent command:

```powershell
.\.venv\Scripts\python.exe -m voderberg_optimizer.cli optimize --settings settings.toml --no-display
```

### Display only

Initial state:

```bat
display_initial.bat
```

Optimized state:

```bat
display_final.bat
```

An explicit state file:

```powershell
.\.venv\Scripts\python.exe -m voderberg_optimizer.cli display --settings settings.toml --state path\to\state.init
```

### Export the optimized state

```bat
export_final.bat
```

### Run the tests

```bat
run_tests.bat
```

## 4. Interactive viewer controls

The Pygame event loop stays in the main thread. The numerical solver runs in a
worker thread, so the window remains responsive while SLSQP evaluates a local
subproblem.

Controls:

```text
Mouse wheel       zoom around the cursor
+ / -             zoom around the window center
Left/middle/right drag
                  pan the view
F, R, or Home     fit all displayed contours
C                 show/hide clearance circles
H                 show/hide the control reminder
Q or Escape       close the viewer
Window close      close the viewer
```

Closing the viewer does not discard a long computation. Optimization continues
headlessly and still writes the final state and exports.

The current zoom and pan are preserved when a new accepted solution is drawn.
The viewer no longer refits the geometry after every iteration.

## 5. Collision and step-acceptance policy

The recommended backend is:

```toml
[solver]
name = "feasible_continuation"
method = "SLSQP"
```

Each local cycle performs the following operations:

1. Build local separating constraints for nearby forbidden segment pairs.
2. Ask SLSQP for a candidate inside a bounded trust region.
3. Test the candidate with explicit 2D segment-intersection predicates.
4. Reject proper crossings, tangencies, and collinear overlap for every
   forbidden pair.
5. Check minimum edge length and the coordinate bound.
6. Adaptively validate intermediate states between the last accepted state and
   the candidate.
7. If the full candidate is invalid, repeatedly reduce the step fraction.
8. Accept only a feasible backtracked prefix that also improves the objective.
9. Rebuild the active constraints from that accepted geometry.

Therefore, an intersecting endpoint is never committed as the new solver state.
The final saved state is validated once more defensively before being returned.

### Self-collision

All pairs of non-adjacent edges of every contour are protected. Adjacent edges
are excluded because they must share a vertex.

### Collision between displayed contours

Cross-contour protection is enabled by default:

```toml
enforce_cross_contour_clearance = true
```

Pairs already touching in the initial exact assembly are recorded as intentional
contacts. Such a pair may remain tangent or collinear, but it is still rejected
if it becomes a transverse crossing. All initially separated cross-contour pairs
must remain separated.

### Path validation

Endpoint intersection testing is exact up to the configured floating-point
tolerance. Intermediate-path checking is deliberately conservative: it combines
adaptive subdivision with a displacement/clearance certificate. If the path
cannot be certified within the configured budget, the move is rejected rather
than assumed safe.

Important controls are:

```toml
path_sample_spacing = 0.001
maximum_path_samples = 256
path_max_subdivision_depth = 14
maximum_backtracking_steps = 18
backtracking_factor = 0.5
```

Reducing `path_sample_spacing` makes path checking stricter and slower.

## 6. Continuation and plateau handling

The required forbidden-segment clearance is raised progressively:

```toml
initial_clearance = 0.001
target_clearance = 0.02
clearance_increment = 0.001
```

A failed continuation stage reduces both its increment and the trust radius.
Small random escape proposals are attempted only after stagnation, and they pass
through the same collision, edge-length, coordinate, and path validation as an
ordinary solver proposal.

No random perturbation is added to every gradient step.

## 7. Additional free points and mesh regularization

Points can be inserted into selected control-chain segments:

```toml
[refinement]
p_segments = [0]
q_segments = [0]
```

Repeating an index inserts several evenly spaced initial points:

```toml
p_segments = [0, 0, 0]
```

The inserted coordinates become independent variables, while all their copied
images remain exactly dependent through `SRN2Parameterization`.

Two weak objective terms remove underdetermined drift:

- relative equal-spacing energy;
- normalized bending energy.

The minimum edge length remains a hard solver constraint.

## 8. Replaceable objective

The solver does not contain any target-specific code. Objectives are selected in
`settings.toml` as a weighted list:

```toml
[objective]

[[objective.terms]]
name = "contact_length"
weight = 1.0

[[objective.terms]]
name = "equal_spacing"
weight = 0.002

[[objective.terms]]
name = "bending"
weight = 0.0002
```

Built-in names are registered in:

```text
src/voderberg_optimizer/objective_factory.py
```

To add the future minimum shell-thickness target:

1. construct the missing third/global contour in the parameterization;
2. expose it through `TileAssembly.additional_contours`;
3. implement a new `ObjectiveTerm` that receives `(state, assembly)`;
4. register its name in `TERM_FACTORIES`;
5. select it in `settings.toml`.

The solver, viewer, state encoding, continuation logic, and callback system do
not need to change. See `docs/custom_objective.md`.

## 9. Output files

Default outputs are:

```text
optimized_state.init
last_contour.svg
last_contour.stl
voderberg_optimisation_data\...
```

Every accepted iteration can be autosaved as a resumable `.init` file.

## 10. Recommended conservative first run

For initial testing on the historical state:

```toml
[solver]
clearance_increment = 0.00025
trust_radius = 0.005
maximum_trust_radius = 0.015
separator_activation_distance = 0.12
path_sample_spacing = 0.0005
maximum_path_samples = 512
```

For a geometry/load/export check without optimization:

```toml
[solver]
name = "noop"
```

## 11. Project structure

```text
src/voderberg_optimizer/
  acquisition.py
  app.py
  backend.py
  cli.py
  collision.py
  config.py
  constraints.py
  exporters.py
  geometry.py
  objective_factory.py
  objectives.py
  parameterization.py
  persistence.py
  problem.py
  refinement.py
  regularization.py
  state.py
  validation.py
  visualization.py
  solvers/
    base.py
    feasible_continuation.py
    noop.py
    scipy_solver.py
```

## 12. Validation status

The archive contains tests for:

- equivalence with the original SRN2 construction;
- dynamic state encoding and point refinement;
- proper crossing, touching, and collinear-overlap classification;
- self-contour and cross-contour collision validation;
- protection of initially intentional contacts against transverse crossing;
- additional/future contours;
- collision-safe backtracking from a bow-tie proposal;
- local separators;
- regularization;
- objective configuration from TOML;
- a complete constrained continuation solve on a synthetic problem.

The historical `.init` file was not available in this environment, so solver
weights and continuation scales still require calibration on the real tile.

# Voderberg SRN2 optimizer - shell-thickness edition

This project reconstructs the exact three-piece SRN2 assembly from the compact
free state `(theta, X, P, Q, Y, B)` and optimizes the minimum thickness between
the named inner and outer shell boundaries.

The solver remains a local, topology-conscious SciPy SLSQP process. It deforms a
known valid state continuously, rejects forbidden crossings, and never commits
an invalid endpoint.

## Required input

Copy the historical initialization file beside `settings.toml`:

```text
voderberg_srn2_angles45_contact_optimV4.init
```

The default state layout is:

```text
X = 7 points
P = 3 points
Q = 3 points
Y = 3 points
```

## Windows installation

From the project directory:

```bat
install_windows.bat
```

Equivalent PowerShell commands:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

No Docker image, external executable solver, or compiler toolchain is required.

## Launch optimization

With the interactive viewer:

```bat
run_optimize.bat
```

Equivalent command:

```powershell
.\.venv\Scripts\python.exe -m voderberg_optimizer.cli optimize --settings settings.toml
```

Without the viewer:

```bat
run_optimize_no_display.bat
```

Display only:

```bat
display_initial.bat
display_final.bat
```

Run tests:

```bat
run_tests.bat
```

## Active objective

The geometric target is:

```text
maximize minimum distance(inner shell boundary, outer shell boundary)
```

The active configuration is:

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

The historical contact-length and mean-angle objectives are not selected. The historical 45-degree angle term is neither configured nor active.

The solver uses a smooth nearest-distance estimate to obtain a useful gradient
when the active closest segment pair changes. The viewer, console output,
autosaves, and final state metadata also contain the exact segment-to-segment
minimum as `exact_shell_thickness`.

## Elastic regularization

The two weak regularizers prevent free control points from drifting along nearly
flat directions:

- `equal_spacing`: favors roughly uniform consecutive edge lengths;
- `bending`: suppresses unnecessary second-difference curvature and zigzags.

Their weights are intentionally tiny compared with the thickness term. The
minimum generated-edge length remains a hard solver constraint.

## Three-piece collision validation

The solver now validates all three complete pieces:

```text
reference tile
left surrounding copy
right half-turn copy
```

Exact interfaces and the two copy-to-copy seams present in the initial state are
registered as intentional contacts. All other self- and cross-contour segment
pairs remain protected.

Each proposed move is subjected to:

1. local SLSQP separator constraints;
2. exact endpoint intersection classification;
3. minimum-edge and coordinate checks;
4. conservative adaptive path validation;
5. geometric backtracking if the full step is invalid;
6. objective-improvement filtering.

## Continuation versus objective optimization

The small `target_clearance` in `settings.toml` is only a technical collision
safety margin. It is not the shell objective.

After clearance continuation finishes, the solver performs additional fixed-
clearance optimization stages:

```toml
objective_refinement_stages = 16
```

This avoids stopping simply because the safety-clearance schedule has completed.

## Viewer controls

```text
Mouse wheel       zoom around cursor
+ / -             zoom around center
Mouse drag        pan
F, R, Home        fit geometry
O                 show/hide outer boundary
H                 show/hide help
Q or Escape       close viewer
```

The three pieces are filled with different solid colors. Point circles are not
displayed. Closing the window does not stop a running headless optimization.

## Additional free points

Insert free points in control-chain segments with:

```toml
[refinement]
p_segments = [0]
q_segments = [0]
```

Repeating an index inserts several initially evenly spaced points. Their copied
images remain exact consequences of the SRN2 parameterization.

## Progressive saves and final outputs

Every accepted optimizer state is saved when `logging.autosave_iterations = true`:

```text
voderberg_optimisation_data\YYYYMMDD_HHMMSS_iterNNNNN.init
```

Each `.init` file contains the complete free-variable vector at 17-digit
round-trip precision, dynamic chain sizes, objective values, exact shell
thickness, clearance stage, and solver metadata.  The most recently accepted
state is also overwritten at a stable path:

```text
latest_accepted_state.init
```

After a normal completion, the final state and exports are:

```text
optimized_state.init             final free variables and metadata
three_piece_assembly.svg         centered colored aggregate of all three pieces
last_contour.stl                 extruded central piece, retained for compatibility
solution_definition.py           standalone executable mathematical definition
solution_definition.json         machine-readable numerical snapshot
```

The standalone Python report uses only the Python 3 standard library.  It
embeds `theta`, `X`, `P`, `Q`, `Y`, and `B` with 17 significant digits and
contains the explicit rotations, translations, central symmetry, reversals,
and concatenations that reconstruct every point in contour order. Running it
prints the three contours and shell topology as JSON:

```powershell
py -3 solution_definition.py > reconstructed_solution.json
```

The companion JSON file contains the same free variables, all evaluated ordered
contours, poles, primed poles, seams, and inner/outer shell boundaries.

Regenerate all final exports without running the optimizer:

```bat
export_final.bat
```

The final CLI summary prints both initial and final exact shell thicknesses and
the paths of these outputs.

## Relevant modules

```text
parameterization.py    exact three-piece and shell construction
topology.py            named seams, poles, and shell boundaries
shell_metrics.py       smooth and exact shell thickness
objectives.py          target and regularization terms
collision.py           exact crossing and distance validation
solvers/feasible_continuation.py
                       continuation, safe backtracking, objective refinement
visualization.py       responsive three-piece display
```

## Validation status

The project test suite covers the original SRN2 construction, shell topology,
independent P/Q refinement, exact and smooth shell metrics, Autograd direction,
collision predicates, safe backtracking, objective configuration, and
post-clearance objective refinement.

The historical `.init` file is not included, so final numerical calibration on
the real tile must be performed on your machine.

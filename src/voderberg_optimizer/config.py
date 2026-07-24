"""Typed loading of the user-facing TOML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class PathSettings:
    initial_state: Path
    optimized_state: Path
    output_directory: Path
    svg_output: Path
    stl_output: Path
    acquisition_image: Path


@dataclass(frozen=True)
class StateLayoutSettings:
    x_points: int = 7
    p_points: int = 3
    q_points: int = 3
    y_points: int = 3


@dataclass(frozen=True)
class InitializationSettings:
    mode: str = "file"


@dataclass(frozen=True)
class RefinementSettings:
    x_segments: tuple[int, ...] = ()
    p_segments: tuple[int, ...] = ()
    q_segments: tuple[int, ...] = ()
    y_segments: tuple[int, ...] = ()


@dataclass(frozen=True)
class GeometrySettings:
    minimum_distance: float = 0.01
    minimum_angle_degrees: float = 45.0
    coordinate_bound: float = 2.0
    include_segment_crossing_margin: bool = False


@dataclass(frozen=True)
class ObjectiveTermSettings:
    """One independently replaceable objective contribution.

    Built-in names are registered in objective_factory.py. Adding a future
    shell-thickness target therefore does not require changing the solver.
    """

    name: str
    weight: float = 1.0


_DEFAULT_OBJECTIVE_TERMS = (
    ObjectiveTermSettings("contact_length", 1.0),
    ObjectiveTermSettings("mean_angle", 0.01),
    ObjectiveTermSettings("equal_spacing", 0.002),
    ObjectiveTermSettings("bending", 0.0002),
)


@dataclass(frozen=True)
class ObjectiveSettings:
    terms: tuple[ObjectiveTermSettings, ...] = _DEFAULT_OBJECTIVE_TERMS
    use_legacy_barrier: bool = False
    barrier_weight: float = 1.0
    barrier_amplitude: float = 100.0


@dataclass(frozen=True)
class SolverSettings:
    name: str = "feasible_continuation"
    method: str = "SLSQP"
    maximum_iterations: int = 250
    function_tolerance: float = 1.0e-9
    finite_difference_step: float = 1.0e-7
    initial_clearance: float = 0.001
    target_clearance: float = 0.02
    clearance_increment: float = 0.001
    minimum_clearance_increment: float = 0.00005
    maximum_stages: int = 80
    maximum_local_passes: int = 8
    trust_radius: float = 0.02
    minimum_trust_radius: float = 0.0001
    maximum_trust_radius: float = 0.05
    theta_trust_radius: float = 0.01
    separator_activation_distance: float = 0.08
    minimum_edge_length: float = 0.001
    path_sample_spacing: float = 0.001
    maximum_path_samples: int = 256
    path_max_subdivision_depth: int = 14
    validation_tolerance: float = 1.0e-10
    enforce_cross_contour_clearance: bool = True
    initial_contact_exclusion_distance: float = 0.0001
    maximum_backtracking_steps: int = 18
    backtracking_factor: float = 0.5
    minimum_step_fraction: float = 1.0e-6
    stagnation_passes: int = 3
    escape_attempts: int = 6
    escape_radius: float = 0.002
    random_seed: int = 12345


@dataclass(frozen=True)
class DisplaySettings:
    enabled: bool = True
    screen_width: int = 1000
    screen_height: int = 800
    margin_pixels: int = 50
    point_radius_pixels: int = 3
    frames_per_second: int = 30
    show_clearance_circles: bool = True
    zoom_factor: float = 1.15
    minimum_zoom: float = 0.02
    maximum_zoom: float = 200.0
    show_help: bool = True


@dataclass(frozen=True)
class ExportSettings:
    write_svg: bool = True
    write_stl: bool = True
    stl_thickness: float = 0.1


@dataclass(frozen=True)
class LoggingSettings:
    autosave_iterations: bool = True
    print_contour_points: bool = False


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings
    state_layout: StateLayoutSettings = field(default_factory=StateLayoutSettings)
    initialization: InitializationSettings = field(default_factory=InitializationSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    objective: ObjectiveSettings = field(default_factory=ObjectiveSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


def _resolve(base_directory: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_directory / path


def _parse_objective(raw: dict[str, Any]) -> ObjectiveSettings:
    term_rows = raw.get("terms")
    if term_rows is None:
        # Compatibility with the previous revision's flat weight fields.
        terms = (
            ObjectiveTermSettings("contact_length", float(raw.get("contact_weight", 1.0))),
            ObjectiveTermSettings("mean_angle", float(raw.get("mean_angle_weight", 0.01))),
            ObjectiveTermSettings("equal_spacing", float(raw.get("equal_spacing_weight", 0.002))),
            ObjectiveTermSettings("bending", float(raw.get("bending_weight", 0.0002))),
        )
    else:
        if not isinstance(term_rows, list):
            raise TypeError("objective.terms must be an array of TOML tables.")
        terms = tuple(
            ObjectiveTermSettings(name=str(row["name"]), weight=float(row.get("weight", 1.0)))
            for row in term_rows
        )
        if not terms:
            raise ValueError("At least one [[objective.terms]] entry is required.")
    return ObjectiveSettings(
        terms=terms,
        use_legacy_barrier=bool(raw.get("use_legacy_barrier", False)),
        barrier_weight=float(raw.get("barrier_weight", 1.0)),
        barrier_amplitude=float(raw.get("barrier_amplitude", 100.0)),
    )


def load_settings(path: str | Path) -> AppSettings:
    settings_path = Path(path).resolve()
    with settings_path.open("rb") as stream:
        raw = tomllib.load(stream)

    base = settings_path.parent
    path_values = raw.get("paths", {})
    paths = PathSettings(
        initial_state=_resolve(base, path_values.get("initial_state", "state.init")),
        optimized_state=_resolve(base, path_values.get("optimized_state", "optimized_state.init")),
        output_directory=_resolve(base, path_values.get("output_directory", "output")),
        svg_output=_resolve(base, path_values.get("svg_output", "last_contour.svg")),
        stl_output=_resolve(base, path_values.get("stl_output", "last_contour.stl")),
        acquisition_image=_resolve(base, path_values.get("acquisition_image", "VoderbergSRN2Patron.png")),
    )

    refinement_raw = raw.get("refinement", {})
    refinement = RefinementSettings(
        x_segments=tuple(refinement_raw.get("x_segments", [])),
        p_segments=tuple(refinement_raw.get("p_segments", [])),
        q_segments=tuple(refinement_raw.get("q_segments", [])),
        y_segments=tuple(refinement_raw.get("y_segments", [])),
    )

    return AppSettings(
        paths=paths,
        state_layout=StateLayoutSettings(**raw.get("state_layout", {})),
        initialization=InitializationSettings(**raw.get("initialization", {})),
        refinement=refinement,
        geometry=GeometrySettings(**raw.get("geometry", {})),
        objective=_parse_objective(raw.get("objective", {})),
        solver=SolverSettings(**raw.get("solver", {})),
        display=DisplaySettings(**raw.get("display", {})),
        export=ExportSettings(**raw.get("export", {})),
        logging=LoggingSettings(**raw.get("logging", {})),
    )

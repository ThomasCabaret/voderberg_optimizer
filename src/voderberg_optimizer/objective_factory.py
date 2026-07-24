"""Registry-driven construction of solver-independent objectives.

To add a new target, implement an ObjectiveTerm in objectives.py (or another
module) and register one factory here. The solver only calls problem.value and
problem.gradient and therefore does not need to know what is being optimized.
"""

from __future__ import annotations

from collections.abc import Callable

from .config import AppSettings
from .constraints import ConstraintSettings
from .objectives import (
    BarrierTerm,
    BendingTerm,
    CompositeObjective,
    ContactLengthTerm,
    EqualSpacingTerm,
    MeanAngleTerm,
    WorstCornerAngleTerm,
    ShellThicknessTerm,
    ObjectiveTerm,
    WeightedTerm,
)

TermFactory = Callable[[AppSettings], ObjectiveTerm]


def _legacy_barrier(settings: AppSettings) -> ObjectiveTerm:
    return BarrierTerm(
        ConstraintSettings(
            minimum_distance=settings.geometry.minimum_distance,
            minimum_angle_degrees=settings.geometry.minimum_angle_degrees,
            coordinate_bound=settings.geometry.coordinate_bound,
            include_segment_crossing_margin=settings.geometry.include_segment_crossing_margin,
        ),
        settings.objective.barrier_amplitude,
    )


TERM_FACTORIES: dict[str, TermFactory] = {
    "shell_thickness": lambda settings: ShellThicknessTerm(
        settings.objective.shell_thickness_temperature
    ),
    "contact_length": lambda settings: ContactLengthTerm(),
    "mean_angle": lambda settings: MeanAngleTerm(),
    "equal_spacing": lambda settings: EqualSpacingTerm(),
    "bending": lambda settings: BendingTerm(),
    "worst_corner_angle": lambda settings: WorstCornerAngleTerm(),
    "legacy_barrier": _legacy_barrier,
}


def register_objective_term(name: str, factory: TermFactory) -> None:
    """Register a custom objective term before build_objective is called."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Objective term names cannot be empty.")
    TERM_FACTORIES[normalized] = factory


def build_objective(settings: AppSettings) -> CompositeObjective:
    weighted_terms: list[WeightedTerm] = []
    for configured in settings.objective.terms:
        name = configured.name.strip().lower()
        try:
            term = TERM_FACTORIES[name](settings)
        except KeyError as error:
            available = ", ".join(sorted(TERM_FACTORIES))
            raise ValueError(
                f"Unknown objective term '{configured.name}'. Available terms: {available}. "
                "Register a new term with register_objective_term()."
            ) from error
        weighted_terms.append(WeightedTerm(term, configured.weight))

    # Compatibility switch for old settings. New projects should instead add
    # [[objective.terms]] name = "legacy_barrier" explicitly.
    if settings.objective.use_legacy_barrier and not any(
        item.name.strip().lower() == "legacy_barrier" for item in settings.objective.terms
    ):
        weighted_terms.append(
            WeightedTerm(_legacy_barrier(settings), settings.objective.barrier_weight)
        )
    return CompositeObjective(tuple(weighted_terms))

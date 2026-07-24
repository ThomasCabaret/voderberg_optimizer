"""Solver construction."""

from __future__ import annotations

from ..config import SolverSettings
from .base import Solver
from .feasible_continuation import FeasibleContinuationSolver
from .noop import NoOpSolver
from .scipy_solver import ScipySolver


def create_solver(settings: SolverSettings, *, coordinate_bound: float = 2.0) -> Solver:
    normalized = settings.name.strip().lower()
    if normalized == "noop":
        return NoOpSolver()
    if normalized in {"scipy", "legacy_scipy"}:
        return ScipySolver(
            method=settings.method,
            maximum_iterations=settings.maximum_iterations,
            finite_difference_step=settings.finite_difference_step,
        )
    if normalized == "feasible_continuation":
        return FeasibleContinuationSolver(
            method=settings.method,
            maximum_iterations=settings.maximum_iterations,
            function_tolerance=settings.function_tolerance,
            finite_difference_step=settings.finite_difference_step,
            initial_clearance=settings.initial_clearance,
            target_clearance=settings.target_clearance,
            clearance_increment=settings.clearance_increment,
            minimum_clearance_increment=settings.minimum_clearance_increment,
            maximum_stages=settings.maximum_stages,
            maximum_local_passes=settings.maximum_local_passes,
            objective_refinement_stages=settings.objective_refinement_stages,
            trust_radius=settings.trust_radius,
            minimum_trust_radius=settings.minimum_trust_radius,
            maximum_trust_radius=settings.maximum_trust_radius,
            theta_trust_radius=settings.theta_trust_radius,
            separator_activation_distance=settings.separator_activation_distance,
            minimum_edge_length=settings.minimum_edge_length,
            path_sample_spacing=settings.path_sample_spacing,
            maximum_path_samples=settings.maximum_path_samples,
            path_max_subdivision_depth=settings.path_max_subdivision_depth,
            validation_tolerance=settings.validation_tolerance,
            enforce_cross_contour_clearance=settings.enforce_cross_contour_clearance,
            initial_contact_exclusion_distance=settings.initial_contact_exclusion_distance,
            maximum_backtracking_steps=settings.maximum_backtracking_steps,
            backtracking_factor=settings.backtracking_factor,
            minimum_step_fraction=settings.minimum_step_fraction,
            stagnation_passes=settings.stagnation_passes,
            escape_attempts=settings.escape_attempts,
            escape_radius=settings.escape_radius,
            random_seed=settings.random_seed,
            coordinate_bound=coordinate_bound,
        )
    raise ValueError(
        f"Unknown solver '{settings.name}'. Expected 'feasible_continuation', "
        "'legacy_scipy', or 'noop'."
    )


__all__ = ["create_solver"]

"""Topology-safe continuation solver based on sequential SLSQP subproblems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, minimize

from ..collision import (
    FeasibilityReport,
    build_active_separators,
    coordinate_margins,
    detect_initial_cross_contacts,
    edge_length_margins,
    separator_margins,
    validate_assembly,
)
from ..problem import OptimizationProblem
from .base import IterationCallback, SolverIteration, SolverResult


@dataclass
class FeasibleContinuationSolver:
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
    objective_refinement_stages: int = 12
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
    coordinate_bound: float = 2.0

    def _assembly(self, problem: OptimizationProblem, vector: Any):
        return problem.parameterization.build(problem.state_from_vector(vector))

    def _validate_state(
        self,
        problem: OptimizationProblem,
        vector: np.ndarray,
        clearance: float,
        excluded_cross_pairs,
    ) -> FeasibilityReport:
        assembly = self._assembly(problem, vector)
        points = np.vstack([np.asarray(contour, dtype=float) for contour in assembly.contours])
        if np.max(np.abs(points)) > self.coordinate_bound + self.validation_tolerance:
            return FeasibilityReport(
                False,
                np.inf,
                np.inf,
                message="Coordinate bound exceeded.",
            )
        return validate_assembly(
            assembly,
            clearance=clearance,
            minimum_edge_length=self.minimum_edge_length,
            enforce_cross_contour=self.enforce_cross_contour_clearance,
            excluded_cross_pairs=excluded_cross_pairs,
            tolerance=self.validation_tolerance,
        )

    @staticmethod
    def _maximum_vertex_displacement(first_assembly, second_assembly) -> float:
        maximum = 0.0
        for first, second in zip(first_assembly.contours, second_assembly.contours):
            first_points = np.asarray(first, dtype=float)
            second_points = np.asarray(second, dtype=float)
            maximum = max(
                maximum,
                float(np.max(np.linalg.norm(second_points - first_points, axis=1))),
            )
        return maximum

    def _validate_path(
        self,
        problem: OptimizationProblem,
        start: np.ndarray,
        end: np.ndarray,
        endpoint_clearance: float,
        excluded_cross_pairs,
    ) -> tuple[bool, str]:
        """Validate endpoint exactly and conservatively guard the complete path.

        Every midpoint is checked by exact segment-intersection predicates. An
        interval is certified without further subdivision only when its endpoint
        clearance is larger than a displacement-based Lipschitz bound. If the
        configured subdivision budget cannot certify an interval, the move is
        rejected rather than guessed safe.
        """

        start_report = self._validate_state(problem, start, 0.0, excluded_cross_pairs)
        if not start_report.feasible:
            return False, f"Current accepted state became invalid: {start_report.message}"
        end_report = self._validate_state(problem, end, endpoint_clearance, excluded_cross_pairs)
        if not end_report.feasible:
            return False, f"Endpoint rejected: {end_report.message}"

        start_assembly = self._assembly(problem, start)
        end_assembly = self._assembly(problem, end)
        checks = 2

        def recurse(
            left_fraction: float,
            left_vector: np.ndarray,
            left_assembly,
            left_report: FeasibilityReport,
            right_fraction: float,
            right_vector: np.ndarray,
            right_assembly,
            right_report: FeasibilityReport,
            depth: int,
        ) -> tuple[bool, str]:
            nonlocal checks
            displacement = self._maximum_vertex_displacement(left_assembly, right_assembly)
            endpoint_margin = min(left_report.minimum_clearance, right_report.minimum_clearance)
            # Segment distance is 2-Lipschitz with respect to a common maximum
            # endpoint displacement. This certifies intervals that are far from
            # contact without dense sampling.
            if np.isfinite(endpoint_margin) and endpoint_margin > 2.0 * displacement + self.validation_tolerance:
                return True, "Path interval certified by clearance bound."

            if checks >= self.maximum_path_samples:
                return False, "Path certification budget exhausted; move rejected conservatively."
            if depth >= self.path_max_subdivision_depth:
                return False, "Path subdivision depth exhausted; move rejected conservatively."

            middle_fraction = 0.5 * (left_fraction + right_fraction)
            middle_vector = 0.5 * (left_vector + right_vector)
            middle_report = self._validate_state(problem, middle_vector, 0.0, excluded_cross_pairs)
            checks += 1
            if not middle_report.feasible:
                return False, f"Path rejected at {middle_fraction:.8f}: {middle_report.message}"
            middle_assembly = self._assembly(problem, middle_vector)

            left_displacement = self._maximum_vertex_displacement(left_assembly, middle_assembly)
            right_displacement = self._maximum_vertex_displacement(middle_assembly, right_assembly)
            if max(left_displacement, right_displacement) <= self.path_sample_spacing:
                # All three states are exact-intersection free, and the movement
                # is now below the configured geometric resolution.
                return True, "Path certified at geometric sampling resolution."

            valid, message = recurse(
                left_fraction,
                left_vector,
                left_assembly,
                left_report,
                middle_fraction,
                middle_vector,
                middle_assembly,
                middle_report,
                depth + 1,
            )
            if not valid:
                return valid, message
            return recurse(
                middle_fraction,
                middle_vector,
                middle_assembly,
                middle_report,
                right_fraction,
                right_vector,
                right_assembly,
                right_report,
                depth + 1,
            )

        return recurse(
            0.0,
            start,
            start_assembly,
            start_report,
            1.0,
            end,
            end_assembly,
            end_report,
            0,
        )

    def _backtracked_candidate(
        self,
        problem: OptimizationProblem,
        current: np.ndarray,
        proposed: np.ndarray,
        endpoint_clearance: float,
        excluded_cross_pairs,
        require_improvement: bool,
        current_objective: float,
    ) -> tuple[np.ndarray | None, float, str]:
        direction = proposed - current
        if float(np.linalg.norm(direction)) <= 1.0e-15:
            return None, 0.0, "Proposed displacement is zero."

        fraction = 1.0
        last_message = "No feasible backtracking fraction found."
        for _ in range(self.maximum_backtracking_steps + 1):
            if fraction < self.minimum_step_fraction:
                break
            candidate = current + fraction * direction
            valid, message = self._validate_path(
                problem,
                current,
                candidate,
                endpoint_clearance,
                excluded_cross_pairs,
            )
            if valid:
                candidate_objective = float(problem.value(candidate))
                tolerance = max(self.function_tolerance, 1.0e-10)
                if not require_improvement or candidate_objective <= current_objective + tolerance:
                    return candidate, fraction, message
                last_message = "Backtracked state is feasible but does not improve the objective."
            else:
                last_message = message
            fraction *= self.backtracking_factor
        return None, fraction, last_message

    def _local_minimize(
        self,
        problem: OptimizationProblem,
        current: np.ndarray,
        clearance: float,
        trust_radius: float,
        excluded_cross_pairs,
    ):
        assembly = self._assembly(problem, current)
        separators = build_active_separators(
            assembly,
            activation_distance=max(self.separator_activation_distance, 2.0 * clearance),
            enforce_cross_contour=self.enforce_cross_contour_clearance,
            excluded_cross_pairs=excluded_cross_pairs,
        )

        def build(vector: Any):
            return self._assembly(problem, vector)

        constraints: list[dict[str, Any]] = [
            {
                "type": "ineq",
                "fun": lambda vector: edge_length_margins(build(vector), self.minimum_edge_length),
            },
            {
                "type": "ineq",
                "fun": lambda vector: coordinate_margins(build(vector), self.coordinate_bound),
            },
        ]
        if separators:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda vector: separator_margins(build(vector), separators, clearance),
                }
            )

        lower = current - trust_radius
        upper = current + trust_radius
        lower[0] = current[0] - self.theta_trust_radius
        upper[0] = current[0] + self.theta_trust_radius
        result = minimize(
            fun=problem.value,
            x0=current,
            jac=problem.gradient,
            method=self.method,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={
                "maxiter": self.maximum_iterations,
                "ftol": self.function_tolerance,
                "eps": self.finite_difference_step,
                "disp": False,
            },
        )
        return result, len(separators)

    def _escape_candidate(
        self,
        problem: OptimizationProblem,
        current: np.ndarray,
        clearance: float,
        excluded_cross_pairs,
        random: np.random.Generator,
        current_objective: float,
    ) -> np.ndarray | None:
        for _ in range(self.escape_attempts):
            direction = random.normal(size=current.shape)
            norm = float(np.linalg.norm(direction[1:]))
            if norm <= 1.0e-15:
                continue
            direction[1:] /= norm
            direction[0] = float(np.clip(direction[0], -0.5, 0.5))
            proposed = current + self.escape_radius * direction
            candidate, _, _ = self._backtracked_candidate(
                problem,
                current,
                proposed,
                clearance,
                excluded_cross_pairs,
                require_improvement=False,
                current_objective=current_objective,
            )
            if candidate is not None:
                return candidate
        return None

    def solve(
        self,
        problem: OptimizationProblem,
        initial_vector: Any,
        callback: IterationCallback | None = None,
    ) -> SolverResult:
        current = np.asarray(initial_vector, dtype=float).copy()
        initial_assembly = self._assembly(problem, current)
        excluded_cross_pairs = (
            detect_initial_cross_contacts(initial_assembly, self.initial_contact_exclusion_distance)
            if self.enforce_cross_contour_clearance
            else frozenset()
        )
        initial_report = self._validate_state(problem, current, 0.0, excluded_cross_pairs)
        if not initial_report.feasible:
            return SolverResult(
                vector=current,
                objective=float(problem.value(current)),
                iterations=0,
                success=False,
                message=f"Initial state is not feasible: {initial_report.message}",
            )

        measured_clearance = initial_report.minimum_clearance
        if not np.isfinite(measured_clearance):
            measured_clearance = self.target_clearance
        achieved_clearance = min(self.initial_clearance, 0.8 * measured_clearance)
        achieved_clearance = max(0.0, achieved_clearance)
        increment = max(self.clearance_increment, self.minimum_clearance_increment)
        next_target = min(self.target_clearance, achieved_clearance + increment)
        trust_radius = self.trust_radius
        current_objective = float(problem.value(current))
        best_objective = current_objective
        iteration_index = 0
        stages = 0
        random = np.random.default_rng(self.random_seed)
        messages: list[str] = []

        while stages < self.maximum_stages and achieved_clearance < self.target_clearance - 1.0e-15:
            stages += 1
            stage_start = current.copy()
            stage_start_objective = current_objective
            stage_success = False
            stagnation = 0

            for local_pass in range(self.maximum_local_passes):
                result, separator_count = self._local_minimize(
                    problem,
                    current,
                    next_target,
                    trust_radius,
                    excluded_cross_pairs,
                )
                proposed = np.asarray(result.x, dtype=float)
                candidate, step_fraction, path_message = self._backtracked_candidate(
                    problem,
                    current,
                    proposed,
                    next_target,
                    excluded_cross_pairs,
                    require_improvement=True,
                    current_objective=current_objective,
                )

                if candidate is not None:
                    candidate_objective = float(problem.value(candidate))
                    displacement = float(np.linalg.norm(candidate - current))
                    previous_objective = current_objective
                    current = candidate
                    current_objective = candidate_objective
                    stage_success = True
                    trust_radius = min(self.maximum_trust_radius, trust_radius * 1.15)
                    best_objective = min(best_objective, current_objective)
                    iteration_index += 1
                    if callback is not None:
                        callback(
                            SolverIteration(
                                index=iteration_index,
                                vector=current.copy(),
                                objective=current_objective,
                                message=(
                                    f"clearance={next_target:.6g}, pass={local_pass + 1}, "
                                    f"separators={separator_count}, step_fraction={step_fraction:.6g}, "
                                    f"displacement={displacement:.6g}"
                                ),
                                metadata={
                                    "clearance": next_target,
                                    "trust_radius": trust_radius,
                                    "step_fraction": step_fraction,
                                },
                            )
                        )
                    if abs(previous_objective - current_objective) <= self.function_tolerance:
                        stagnation += 1
                    else:
                        stagnation = 0
                else:
                    trust_radius *= 0.5
                    stagnation += 1
                    messages.append(
                        f"Stage {stages}, pass {local_pass + 1} rejected and backtracked: {path_message}"
                    )

                if stagnation >= self.stagnation_passes:
                    escaped = self._escape_candidate(
                        problem,
                        current,
                        next_target,
                        excluded_cross_pairs,
                        random,
                        current_objective,
                    )
                    if escaped is not None:
                        current = escaped
                        current_objective = float(problem.value(current))
                        stagnation = 0
                    else:
                        break
                if trust_radius < self.minimum_trust_radius:
                    break

            final_stage_report = self._validate_state(
                problem,
                current,
                next_target,
                excluded_cross_pairs,
            )
            if final_stage_report.feasible:
                achieved_clearance = next_target
                best_objective = min(best_objective, current_objective)
                if achieved_clearance >= self.target_clearance - 1.0e-15:
                    break
                next_target = min(self.target_clearance, achieved_clearance + increment)
            else:
                current = stage_start
                current_objective = stage_start_objective
                increment *= 0.5
                trust_radius = max(self.minimum_trust_radius, trust_radius * 0.5)
                if increment < self.minimum_clearance_increment:
                    messages.append("Continuation stopped because the clearance increment became too small.")
                    break
                next_target = min(self.target_clearance, achieved_clearance + increment)

        # Once the technical clearance continuation is complete (or can no
        # longer advance), keep optimizing the actual objective at fixed
        # clearance.  Earlier revisions stopped immediately at the clearance
        # target, which could leave the shell-thickness objective far from a
        # local optimum.
        refinement_stages_run = 0
        refinement_clearance = min(achieved_clearance, self.target_clearance)
        for refinement_stage in range(self.objective_refinement_stages):
            stage_start_objective = current_objective
            meaningful_improvement = False
            stagnation = 0

            for local_pass in range(self.maximum_local_passes):
                result, separator_count = self._local_minimize(
                    problem,
                    current,
                    refinement_clearance,
                    trust_radius,
                    excluded_cross_pairs,
                )
                proposed = np.asarray(result.x, dtype=float)
                candidate, step_fraction, path_message = self._backtracked_candidate(
                    problem,
                    current,
                    proposed,
                    refinement_clearance,
                    excluded_cross_pairs,
                    require_improvement=True,
                    current_objective=current_objective,
                )

                if candidate is not None:
                    candidate_objective = float(problem.value(candidate))
                    improvement = current_objective - candidate_objective
                    displacement = float(np.linalg.norm(candidate - current))
                    current = candidate
                    current_objective = candidate_objective
                    best_objective = min(best_objective, current_objective)
                    trust_radius = min(self.maximum_trust_radius, trust_radius * 1.1)
                    iteration_index += 1
                    if improvement > self.function_tolerance:
                        meaningful_improvement = True
                        stagnation = 0
                    else:
                        stagnation += 1
                    if callback is not None:
                        callback(
                            SolverIteration(
                                index=iteration_index,
                                vector=current.copy(),
                                objective=current_objective,
                                message=(
                                    f"objective-refinement={refinement_stage + 1}, "
                                    f"pass={local_pass + 1}, separators={separator_count}, "
                                    f"step_fraction={step_fraction:.6g}, "
                                    f"displacement={displacement:.6g}"
                                ),
                                metadata={
                                    "clearance": refinement_clearance,
                                    "trust_radius": trust_radius,
                                    "step_fraction": step_fraction,
                                    "objective_refinement_stage": float(refinement_stage + 1),
                                },
                            )
                        )
                else:
                    trust_radius *= 0.5
                    stagnation += 1
                    messages.append(
                        f"Objective refinement {refinement_stage + 1}, pass "
                        f"{local_pass + 1} rejected: {path_message}"
                    )

                if stagnation >= self.stagnation_passes:
                    escaped = self._escape_candidate(
                        problem,
                        current,
                        refinement_clearance,
                        excluded_cross_pairs,
                        random,
                        current_objective,
                    )
                    if escaped is not None:
                        current = escaped
                        current_objective = float(problem.value(current))
                        stagnation = 0
                    else:
                        break
                if trust_radius < self.minimum_trust_radius:
                    break

            refinement_stages_run += 1
            if not meaningful_improvement:
                # No useful descent remained at this clearance.
                if abs(stage_start_objective - current_objective) <= self.function_tolerance:
                    break

        final_report = self._validate_state(problem, current, 0.0, excluded_cross_pairs)
        if not final_report.feasible:
            # Defensive invariant: never return a topology-invalid state.
            current = np.asarray(initial_vector, dtype=float).copy()
            messages.append("Defensive fallback to initial state after final feasibility failure.")

        final_objective = float(problem.value(current))
        success = achieved_clearance >= self.target_clearance - 1.0e-12 and final_report.feasible
        message = (
            f"Feasible continuation finished after {stages} stages; "
            f"achieved clearance={achieved_clearance:.6g}, target={self.target_clearance:.6g}."
        )
        if messages:
            message += " " + messages[-1]
        return SolverResult(
            vector=current,
            objective=final_objective,
            iterations=iteration_index,
            success=success,
            message=message,
            metadata={
                "achieved_clearance": achieved_clearance,
                "target_clearance": self.target_clearance,
                "best_objective_seen": best_objective,
                "excluded_initial_cross_contacts": float(len(excluded_cross_pairs)),
                "objective_refinement_stages": float(refinement_stages_run),
            },
        )

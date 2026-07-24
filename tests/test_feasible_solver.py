import numpy as np

from voderberg_optimizer.parameterization import TileAssembly
from voderberg_optimizer.solvers.feasible_continuation import FeasibleContinuationSolver


class RectangleProblem:
    @property
    def parameterization(self):
        return self

    def state_from_vector(self, vector):
        return np.asarray(vector, dtype=float)

    def build(self, state):
        width = state[1]
        first = np.array([[0.0, 0.0], [width, 0.0], [width, 1.0], [0.0, 1.0]])
        second = first + np.array([10.0, 0.0])
        return TileAssembly(first, second)

    def value(self, vector):
        return (vector[1] - 2.0) ** 2

    def gradient(self, vector):
        return np.array([0.0, 2.0 * (vector[1] - 2.0)])


def test_feasible_continuation_solves_simple_constrained_problem() -> None:
    solver = FeasibleContinuationSolver(
        maximum_iterations=50,
        target_clearance=0.5,
        initial_clearance=0.1,
        clearance_increment=0.2,
        minimum_clearance_increment=0.01,
        maximum_stages=10,
        maximum_local_passes=4,
        trust_radius=0.5,
        maximum_trust_radius=0.5,
        theta_trust_radius=0.01,
        separator_activation_distance=2.0,
        minimum_edge_length=0.1,
        path_sample_spacing=0.1,
        maximum_path_samples=20,
        coordinate_bound=20.0,
        stagnation_passes=2,
        escape_attempts=0,
    )
    result = solver.solve(RectangleProblem(), np.array([0.0, 1.0]))
    assert result.success
    assert result.metadata["achieved_clearance"] >= 0.5
    assert abs(result.vector[1] - 2.0) < 1.0e-6


def test_objective_refinement_runs_after_clearance_target_is_already_met() -> None:
    solver = FeasibleContinuationSolver(
        maximum_iterations=50,
        initial_clearance=0.0,
        target_clearance=0.0,
        clearance_increment=0.1,
        minimum_clearance_increment=0.01,
        maximum_stages=1,
        maximum_local_passes=4,
        objective_refinement_stages=4,
        trust_radius=0.5,
        maximum_trust_radius=0.5,
        theta_trust_radius=0.01,
        separator_activation_distance=2.0,
        minimum_edge_length=0.1,
        path_sample_spacing=0.1,
        maximum_path_samples=20,
        coordinate_bound=20.0,
        stagnation_passes=2,
        escape_attempts=0,
    )
    result = solver.solve(RectangleProblem(), np.array([0.0, 1.0]))
    assert abs(result.vector[1] - 2.0) < 1.0e-6
    assert result.metadata["objective_refinement_stages"] >= 1

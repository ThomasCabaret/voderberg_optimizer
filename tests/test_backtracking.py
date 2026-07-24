import numpy as np

from voderberg_optimizer.parameterization import TileAssembly
from voderberg_optimizer.solvers.feasible_continuation import FeasibleContinuationSolver


class BowTiePathProblem:
    @property
    def parameterization(self):
        return self

    def state_from_vector(self, vector):
        return np.asarray(vector, dtype=float)

    def build(self, state):
        fraction = float(state[0])
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        bow_tie = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        first = (1.0 - fraction) * square + fraction * bow_tie
        second = first + np.array([10.0, 0.0])
        return TileAssembly(first, second)

    def value(self, vector):
        return -float(vector[0])


def test_backtracking_never_accepts_bow_tie_endpoint() -> None:
    problem = BowTiePathProblem()
    solver = FeasibleContinuationSolver(
        target_clearance=0.0,
        minimum_edge_length=1.0e-4,
        enforce_cross_contour_clearance=False,
        path_sample_spacing=0.01,
        maximum_path_samples=512,
        path_max_subdivision_depth=18,
        maximum_backtracking_steps=20,
        coordinate_bound=20.0,
    )
    current = np.array([0.0])
    candidate, fraction, _ = solver._backtracked_candidate(
        problem,
        current,
        np.array([1.0]),
        endpoint_clearance=0.0,
        excluded_cross_pairs=frozenset(),
        require_improvement=True,
        current_objective=problem.value(current),
    )
    assert candidate is not None
    assert 0.0 < fraction < 1.0
    report = solver._validate_state(problem, candidate, 0.0, frozenset())
    assert report.feasible

import numpy as np

from voderberg_optimizer.objectives import CompositeObjective
from voderberg_optimizer.parameterization import SRN2Parameterization
from voderberg_optimizer.problem import OptimizationProblem
from voderberg_optimizer.state import SRN2State


def test_problem_normalizes_theta_like_original_objective() -> None:
    state = SRN2State(
        theta=3.0 * np.pi,
        x=np.array([[0.0, 0.0], [0.1, 0.1]]),
        p=np.array([[0.0, -0.2], [0.0, -0.3]]),
        q=np.array([[0.0, -0.3], [0.0, -0.2]]),
        y=np.array([[0.0, 0.0], [-0.1, 0.1]]),
        b=np.array([0.0, -0.8]),
    )
    problem = OptimizationProblem(state.layout, SRN2Parameterization(), CompositeObjective(()))
    normalized = problem.state_from_vector(state.to_vector())
    assert np.isclose(normalized.theta, -np.pi)

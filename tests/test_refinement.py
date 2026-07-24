import numpy as np

from voderberg_optimizer.refinement import insert_midpoints, refine_state
from voderberg_optimizer.state import SRN2State


def test_insert_midpoint_is_initially_collinear() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    refined = insert_midpoints(points, [0])
    np.testing.assert_allclose(refined, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0]])


def test_refinement_changes_dynamic_layout() -> None:
    state = SRN2State(
        theta=0.1,
        x=np.array([[0.0, 0.0], [1.0, 0.0]]),
        p=np.array([[0.0, 0.0], [1.0, 0.0]]),
        q=np.array([[0.0, 0.0], [1.0, 0.0]]),
        y=np.array([[0.0, 0.0], [1.0, 0.0]]),
        b=np.array([0.0, -1.0]),
    )
    refined = refine_state(state, p_segments=[0], q_segments=[0])
    assert refined.layout.p_points == 3
    assert refined.layout.q_points == 3

import numpy as np

from voderberg_optimizer.parameterization import SRN2Parameterization
from voderberg_optimizer.state import SRN2State


def sample_state() -> SRN2State:
    return SRN2State(
        theta=0.2,
        x=np.array([[0.2, 0.4], [0.4, 0.0], [0.3, -0.3]]),
        p=np.array([[0.2, -0.5], [0.1, -0.7]]),
        q=np.array([[-0.1, -0.7], [-0.2, -0.5]]),
        y=np.array([[-0.3, -0.3], [-0.4, 0.0]]),
        b=np.array([0.1, -0.9]),
    )


def test_both_contours_have_dependency_predicted_size() -> None:
    state = sample_state()
    assembly = SRN2Parameterization().build(state)
    expected = 6 * len(state.p) + 6 * len(state.q) + 3 * len(state.y) + 4 * len(state.x) + 7
    assert len(assembly.main_contour) == expected
    assert len(assembly.left_contour) == expected
    assert assembly.right_contour is not None
    assert len(assembly.right_contour) == expected
    np.testing.assert_allclose(assembly.main_contour[0], [0.0, 1.0])
    np.testing.assert_allclose(assembly.left_contour[0], [0.0, 1.0])

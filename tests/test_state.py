import numpy as np

from voderberg_optimizer.state import SRN2State, StateLayout


def test_state_vector_round_trip() -> None:
    layout = StateLayout(x_points=2, p_points=3, q_points=4, y_points=2)
    vector = np.arange(layout.vector_size, dtype=float) / 10.0
    state = SRN2State.from_vector(vector, layout)
    np.testing.assert_allclose(state.to_vector(), vector)
    assert state.layout == layout

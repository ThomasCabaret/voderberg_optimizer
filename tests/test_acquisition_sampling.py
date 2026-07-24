import numpy as np

from voderberg_optimizer.acquisition import _sample_open_segment


def test_legacy_p_chain_sampling() -> None:
    result = _sample_open_segment(
        np.array([0.0, 0.0]), np.array([3.0, 0.0]), 3, include_start=True, include_end=False
    )
    np.testing.assert_allclose(result, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


def test_legacy_q_chain_sampling() -> None:
    result = _sample_open_segment(
        np.array([0.0, 0.0]), np.array([3.0, 0.0]), 3, include_start=False, include_end=True
    )
    np.testing.assert_allclose(result, [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

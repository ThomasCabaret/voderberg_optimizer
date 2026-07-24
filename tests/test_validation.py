import numpy as np

from voderberg_optimizer.validation import validate_polygon


def test_square_is_simple() -> None:
    report = validate_polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    assert report.is_simple


def test_bow_tie_has_strict_crossing() -> None:
    report = validate_polygon(np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float))
    assert not report.is_simple
    assert len(report.self_intersections) == 1

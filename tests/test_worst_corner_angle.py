import math

import numpy as np

from voderberg_optimizer.objectives import WorstCornerAngleTerm
from voderberg_optimizer.topology import TileAssembly


def assembly(points: np.ndarray) -> TileAssembly:
    translated = points + np.array([10.0, 0.0])
    return TileAssembly(main_contour=points, left_contour=translated)


def test_worst_corner_term_prefers_a_square_to_a_needle_spike() -> None:
    square = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    spike = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.01]],
        dtype=float,
    )
    term = WorstCornerAngleTerm()

    # The objective is minimized, hence a healthier larger worst angle must
    # produce a lower (more negative) contribution.
    assert term.value(None, assembly(square)) < term.value(None, assembly(spike))


def test_diagnostic_reports_unsigned_sharpness_angle() -> None:
    spike = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.01]],
        dtype=float,
    )
    diagnostic = WorstCornerAngleTerm().diagnostics(None, assembly(spike))

    expected = math.atan2(0.01, 1.0)
    assert abs(diagnostic["worst_corner_angle_radians"] - expected) < 1.0e-10
    assert diagnostic["worst_corner_angle_degrees"] < 1.0

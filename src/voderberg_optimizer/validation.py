"""Non-differentiable exact checks used to validate accepted geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Intersection:
    first_segment: int
    second_segment: int


@dataclass(frozen=True)
class PolygonValidation:
    self_intersections: tuple[Intersection, ...]
    repeated_consecutive_vertices: tuple[int, ...]

    @property
    def is_simple(self) -> bool:
        return not self.self_intersections and not self.repeated_consecutive_vertices


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _strict_opposite_sign(first: float, second: float, tolerance: float) -> bool:
    return (first > tolerance and second < -tolerance) or (first < -tolerance and second > tolerance)


def segments_cross_strictly(
    first_start: Any,
    first_end: Any,
    second_start: Any,
    second_end: Any,
    tolerance: float = 1.0e-12,
) -> bool:
    """Return True only for a proper interior crossing.

    Endpoint contact and collinear overlap are deliberately excluded because
    the SRN2 assembly contains intentional boundary identifications. Those cases
    need topology-aware handling rather than a generic intersection predicate.
    """

    a = np.asarray(first_start, dtype=float)
    b = np.asarray(first_end, dtype=float)
    c = np.asarray(second_start, dtype=float)
    d = np.asarray(second_end, dtype=float)
    return _strict_opposite_sign(_orientation(a, b, c), _orientation(a, b, d), tolerance) and _strict_opposite_sign(
        _orientation(c, d, a), _orientation(c, d, b), tolerance
    )


def validate_polygon(points: Any, tolerance: float = 1.0e-12) -> PolygonValidation:
    vertices = np.asarray(points, dtype=float)
    count = len(vertices)
    repeated = tuple(
        index
        for index in range(count)
        if np.linalg.norm(vertices[(index + 1) % count] - vertices[index]) <= tolerance
    )

    intersections: list[Intersection] = []
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            adjacent = (
                second == first_next
                or first == second_next
                or (first == 0 and second == count - 1)
            )
            if adjacent:
                continue
            if segments_cross_strictly(
                vertices[first],
                vertices[first_next],
                vertices[second],
                vertices[second_next],
                tolerance,
            ):
                intersections.append(Intersection(first, second))

    return PolygonValidation(tuple(intersections), repeated)

"""Differentiable margins and barrier terms for geometric validity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import np

from .constants import BARRIER_SLOPE_FACTOR, INFEASIBILITY_LINEAR_WEIGHT, MAX_EXPONENT
from .geometry import (
    interior_angles,
    point_to_segment_distance,
    polygon_segments,
    signed_crossing_margin,
)
from .parameterization import TileAssembly


@dataclass(frozen=True)
class ConstraintSettings:
    minimum_distance: float
    minimum_angle_degrees: float
    coordinate_bound: float
    include_segment_crossing_margin: bool = False


def point_segment_clearance_margin(points: Any, minimum_distance: float) -> Any:
    margins = []
    count = len(points)
    for segment_index, (start, end) in enumerate(polygon_segments(points, closed=True)):
        start_index = segment_index
        end_index = (segment_index + 1) % count
        for point_index, point in enumerate(points):
            if point_index in (start_index, end_index):
                continue
            margins.append(point_to_segment_distance(point, start, end) - minimum_distance)
    return np.min(np.array(margins)) if margins else np.inf


def segment_crossing_margin(points: Any, minimum_distance: float) -> Any:
    segments = polygon_segments(points, closed=True)
    margins = []
    segment_count = len(segments)
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            adjacent = second == first + 1 or (first == 0 and second == segment_count - 1)
            if adjacent:
                continue
            margins.append(signed_crossing_margin(segments[first], segments[second]) - minimum_distance)
    return np.min(np.array(margins)) if margins else np.inf


def angle_margin(points: Any, minimum_angle_degrees: float) -> Any:
    angles = interior_angles(points, closed=True)
    if not len(angles):
        return np.inf
    return np.min(angles - np.deg2rad(minimum_angle_degrees))


def coordinate_margin(points: Any, coordinate_bound: float) -> Any:
    return coordinate_bound - np.max(np.abs(points))


def contour_clearance_margin(points: Any, settings: ConstraintSettings) -> Any:
    margins = [
        point_segment_clearance_margin(points, settings.minimum_distance),
        coordinate_margin(points, settings.coordinate_bound),
    ]
    if settings.include_segment_crossing_margin:
        margins.append(segment_crossing_margin(points, settings.minimum_distance))
    return np.min(np.array(margins))


def assembly_margin(assembly: TileAssembly, settings: ConstraintSettings) -> Any:
    # Preserve the original formulation: clearance/bounds are checked on both
    # displayed contours, while the minimum-angle condition is checked only on
    # the main contour.
    clearance = np.min(
        np.array([contour_clearance_margin(contour, settings) for contour in assembly.contours])
    )
    main_angle = angle_margin(assembly.main_contour, settings.minimum_angle_degrees)
    return np.minimum(clearance, main_angle)


def soft_barrier(margin: Any, amplitude: float, distance_scale: float) -> Any:
    scale = max(distance_scale, 1.0e-9)
    alpha = BARRIER_SLOPE_FACTOR / scale
    sigmoid = amplitude / (1.0 + np.exp(np.clip(alpha * margin, -MAX_EXPONENT, MAX_EXPONENT)))
    infeasible_linear_term = INFEASIBILITY_LINEAR_WEIGHT * np.maximum(-margin, 0.0)
    return sigmoid + infeasible_linear_term

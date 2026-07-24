"""Differentiable and exact metrics for the reconstructed SRN2 shell.

The optimization target uses a smooth approximation of the minimum distance
between the inner and outer shell boundaries.  The exact polyline distance is
reported separately and is used by tests and diagnostics.

The differentiable segment distance is exact for disjoint planar segments: for
such a pair, at least one closest point is an endpoint, so the distance is the
minimum of the four endpoint-to-opposite-segment distances.  Topology and
collision guards keep accepted states in that disjoint regime.
"""

from __future__ import annotations

from typing import Any

import numpy as standard_numpy

from .backend import np
from .collision import closest_points_on_segments, contour_segments
from .constants import EPSILON
from .topology import TileAssembly


def _closed_segment_arrays(points: Any) -> tuple[Any, Any]:
    """Return start/end arrays for the implicitly closed polygonal contour."""

    return points, np.concatenate((points[1:], points[:1]), axis=0)


def _pairwise_point_to_segment_squared_distances(
    points: Any,
    segment_starts: Any,
    segment_ends: Any,
) -> Any:
    """Squared distances from every point to every closed segment."""

    point_grid = points[:, None, :]
    start_grid = segment_starts[None, :, :]
    vectors = (segment_ends - segment_starts)[None, :, :]
    offsets = point_grid - start_grid
    denominator = np.sum(vectors * vectors, axis=2) + EPSILON
    projection = np.sum(offsets * vectors, axis=2) / denominator
    projection = np.clip(projection, 0.0, 1.0)
    nearest = start_grid + projection[:, :, None] * vectors
    difference = point_grid - nearest
    return np.sum(difference * difference, axis=2)


def pairwise_segment_distances(first_boundary: Any, second_boundary: Any) -> Any:
    """Return all pairwise distances between two closed polygonal boundaries.

    The returned matrix has shape ``(len(first_boundary), len(second_boundary))``.
    It is designed for Autograd and therefore contains no Python branching based
    on the current numerical geometry.
    """

    first_start, first_end = _closed_segment_arrays(first_boundary)
    second_start, second_end = _closed_segment_arrays(second_boundary)

    first_start_to_second = _pairwise_point_to_segment_squared_distances(
        first_start, second_start, second_end
    )
    first_end_to_second = _pairwise_point_to_segment_squared_distances(
        first_end, second_start, second_end
    )
    second_start_to_first = _pairwise_point_to_segment_squared_distances(
        second_start, first_start, first_end
    ).T
    second_end_to_first = _pairwise_point_to_segment_squared_distances(
        second_end, first_start, first_end
    ).T

    squared = np.minimum(
        np.minimum(first_start_to_second, first_end_to_second),
        np.minimum(second_start_to_first, second_end_to_first),
    )
    return np.sqrt(np.maximum(squared, 0.0) + EPSILON)


def smooth_minimum(values: Any, temperature: float) -> Any:
    """Soft nearest-distance estimate based on a stable Boltzmann average.

    Unlike a raw log-sum-exp soft minimum, this weighted average has no term
    proportional to the number of segment pairs.  It approaches the exact
    minimum as ``temperature`` decreases while retaining contributions from
    nearby competing contact pairs.
    """

    if temperature <= 0.0:
        return np.min(values)
    flattened = np.ravel(values)
    minimum = np.min(flattened)
    weights = np.exp(-(flattened - minimum) / temperature)
    return np.sum(weights * flattened) / (np.sum(weights) + EPSILON)


def smooth_shell_thickness(assembly: TileAssembly, temperature: float) -> Any:
    """Differentiable shell thickness used by the optimizer."""

    if assembly.shell is None:
        raise ValueError("Shell topology is required to evaluate shell thickness.")
    distances = pairwise_segment_distances(
        assembly.shell.inner_boundary,
        assembly.shell.outer_boundary,
    )
    return smooth_minimum(distances, temperature)


def exact_boundary_distance(first_boundary: Any, second_boundary: Any) -> float:
    """Exact minimum Euclidean distance between two closed polygonal curves."""

    first_segments = contour_segments(standard_numpy.asarray(first_boundary, dtype=float))
    second_segments = contour_segments(standard_numpy.asarray(second_boundary, dtype=float))
    minimum = standard_numpy.inf
    for first in first_segments:
        for second in second_segments:
            distance, _, _ = closest_points_on_segments(*first, *second)
            minimum = min(minimum, distance)
    return float(minimum)


def exact_shell_thickness(assembly: TileAssembly) -> float:
    """Exact minimum inner-to-outer shell distance for diagnostics."""

    if assembly.shell is None:
        raise ValueError("Shell topology is required to evaluate shell thickness.")
    return exact_boundary_distance(
        assembly.shell.inner_boundary,
        assembly.shell.outer_boundary,
    )


__all__ = [
    "exact_boundary_distance",
    "exact_shell_thickness",
    "pairwise_segment_distances",
    "smooth_minimum",
    "smooth_shell_thickness",
]

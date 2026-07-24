"""Small differentiable geometry primitives."""

from __future__ import annotations

from typing import Any, Iterable

from .backend import np

from .constants import CROSSING_MARGIN_CAP, EPSILON


def rotate_point(point: Any, angle: Any, center: Any) -> Any:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return center + np.dot(rotation, point - center)


def rotate_points(points: Any, angle: Any, center: Any) -> Any:
    return np.array([rotate_point(point, angle, center) for point in points])


def similarity_transform(source_a: Any, source_b: Any, destination_a: Any, destination_b: Any) -> Any:
    source_vector = source_b - source_a
    destination_vector = destination_b - destination_a
    scale = np.linalg.norm(destination_vector) / (np.linalg.norm(source_vector) + EPSILON)
    angle = np.arctan2(destination_vector[1], destination_vector[0]) - np.arctan2(
        source_vector[1], source_vector[0]
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    translation = destination_a - scale * np.dot(
        np.array(((cosine, -sine), (sine, cosine))), source_a
    )
    return np.array(
        (
            (scale * cosine, -scale * sine, translation[0]),
            (scale * sine, scale * cosine, translation[1]),
            (0.0, 0.0, 1.0),
        )
    )


def apply_homogeneous_transform(transform: Any, points: Any) -> Any:
    points_2d = np.atleast_2d(points)
    homogeneous = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))
    return np.dot(transform, homogeneous.T).T[:, :2]


def point_to_segment_distance(point: Any, segment_start: Any, segment_end: Any) -> Any:
    segment = segment_end - segment_start
    projection = np.dot(point - segment_start, segment) / (np.dot(segment, segment) + EPSILON)
    nearest = segment_start + np.clip(projection, 0.0, 1.0) * segment
    return np.linalg.norm(point - nearest)


def signed_crossing_margin(segment_a: tuple[Any, Any], segment_b: tuple[Any, Any]) -> Any:
    """Legacy differentiable crossing margin retained for compatibility.

    Negative values indicate an interior crossing. This is not a robust exact
    segment-distance function for nearly parallel segments.
    """

    p, q = segment_a
    r, s = segment_b
    first = q - p
    second = s - r
    cross = first[0] * second[1] - first[1] * second[0]
    inverse_cross = 1.0 / (cross + EPSILON)
    difference = r - p
    t = (difference[0] * second[1] - difference[1] * second[0]) * inverse_cross
    u = (difference[0] * first[1] - difference[1] * first[0]) * inverse_cross
    penetration_first = np.minimum(t, 1.0 - t) * np.linalg.norm(first)
    penetration_second = np.minimum(u, 1.0 - u) * np.linalg.norm(second)
    return np.minimum(np.maximum(-penetration_first, -penetration_second), CROSSING_MARGIN_CAP)


def polygon_segments(points: Any, closed: bool = True) -> list[tuple[Any, Any]]:
    segments = [(points[index], points[index + 1]) for index in range(len(points) - 1)]
    if closed and len(points) > 1:
        segments.append((points[-1], points[0]))
    return segments


def interior_angles(points: Any, closed: bool = True) -> Any:
    count = len(points)
    if count < 3:
        return np.array([])

    indices: Iterable[int] = range(count) if closed else range(1, count - 1)
    angles = []
    for index in indices:
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]
        first = previous - current
        second = following - current
        denominator = np.linalg.norm(first) * np.linalg.norm(second) + EPSILON
        cosine = np.clip(np.dot(first, second) / denominator, -1.0 + EPSILON, 1.0 - EPSILON)
        angles.append(np.arccos(cosine))
    return np.array(angles)


def mean_interior_angle(points: Any, closed: bool = True) -> Any:
    angles = interior_angles(points, closed=closed)
    return np.mean(angles) if len(angles) else 0.0

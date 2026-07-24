"""Exact collision tests and local separating constraints.

The optimizer uses smooth-ish fixed separating planes locally, but acceptance
never relies on those approximations: every accepted state is checked with
explicit 2D segment-intersection predicates and Euclidean segment distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .parameterization import TileAssembly


@dataclass(frozen=True, order=True)
class SegmentRef:
    contour: int
    index: int


@dataclass(frozen=True, order=True)
class SegmentPair:
    first: SegmentRef
    second: SegmentRef


@dataclass(frozen=True)
class Separator:
    pair: SegmentPair
    normal: np.ndarray


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    minimum_clearance: float
    minimum_edge_length: float
    violating_pair: SegmentPair | None = None
    message: str = ""


def contour_segments(contour: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(contour, dtype=float)
    return [(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]


def segment_endpoints(assembly: TileAssembly, reference: SegmentRef) -> tuple[np.ndarray, np.ndarray]:
    contour = np.asarray(assembly.contours[reference.contour], dtype=float)
    return contour[reference.index], contour[(reference.index + 1) % len(contour)]


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return _cross(b - a, c - a)


def _orientation_tolerance(a: np.ndarray, b: np.ndarray, c: np.ndarray, tolerance: float) -> float:
    scale = max(
        1.0,
        float(np.linalg.norm(b - a)),
        float(np.linalg.norm(c - a)),
    )
    return tolerance * scale * scale


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, tolerance: float) -> bool:
    if abs(_orientation(a, b, p)) > _orientation_tolerance(a, b, p, tolerance):
        return False
    lower = np.minimum(a, b) - tolerance
    upper = np.maximum(a, b) + tolerance
    return bool(np.all(p >= lower) and np.all(p <= upper))


def segment_intersection_kind(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float = 1.0e-12,
) -> str:
    """Classify a 2D segment pair as none, touch, overlap, or proper."""

    a = np.asarray(first_start, dtype=float)
    b = np.asarray(first_end, dtype=float)
    c = np.asarray(second_start, dtype=float)
    d = np.asarray(second_end, dtype=float)
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    t1 = _orientation_tolerance(a, b, c, tolerance)
    t2 = _orientation_tolerance(a, b, d, tolerance)
    t3 = _orientation_tolerance(c, d, a, tolerance)
    t4 = _orientation_tolerance(c, d, b, tolerance)

    first_straddles = (o1 > t1 and o2 < -t2) or (o1 < -t1 and o2 > t2)
    second_straddles = (o3 > t3 and o4 < -t4) or (o3 < -t3 and o4 > t4)
    if first_straddles and second_straddles:
        return "proper"

    collinear = abs(o1) <= t1 and abs(o2) <= t2 and abs(o3) <= t3 and abs(o4) <= t4
    if collinear:
        direction = b - a
        axis = int(np.argmax(np.abs(direction))) if float(np.linalg.norm(direction)) > tolerance else 0
        first_low, first_high = sorted((float(a[axis]), float(b[axis])))
        second_low, second_high = sorted((float(c[axis]), float(d[axis])))
        overlap = min(first_high, second_high) - max(first_low, second_low)
        if overlap > tolerance:
            return "overlap"
        if overlap >= -tolerance:
            return "touch"
        return "none"

    if (
        (abs(o1) <= t1 and _on_segment(a, b, c, tolerance))
        or (abs(o2) <= t2 and _on_segment(a, b, d, tolerance))
        or (abs(o3) <= t3 and _on_segment(c, d, a, tolerance))
        or (abs(o4) <= t4 and _on_segment(c, d, b, tolerance))
    ):
        return "touch"
    return "none"


def segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float = 1.0e-12,
) -> bool:
    """Return True for proper crossings, tangencies, and collinear overlap."""

    return segment_intersection_kind(
        first_start, first_end, second_start, second_end, tolerance
    ) != "none"


def closest_points_on_segments(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    epsilon: float = 1.0e-15,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return Euclidean distance and closest points of two closed 2D segments."""

    p0 = np.asarray(first_start, dtype=float)
    p1 = np.asarray(first_end, dtype=float)
    q0 = np.asarray(second_start, dtype=float)
    q1 = np.asarray(second_end, dtype=float)

    if segments_intersect(p0, p1, q0, q1, tolerance=max(epsilon, 1.0e-14)):
        # The exact intersection location is not needed by validation. Returning
        # coincident representatives prevents a near-parallel distance routine
        # from masking a true crossing.
        midpoint = 0.25 * (p0 + p1 + q0 + q1)
        return 0.0, midpoint.copy(), midpoint.copy()

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b

    if a <= epsilon and c <= epsilon:
        return float(np.linalg.norm(p0 - q0)), p0.copy(), q0.copy()
    if a <= epsilon:
        t = float(np.clip(e / max(c, epsilon), 0.0, 1.0))
        point_q = q0 + t * v
        return float(np.linalg.norm(p0 - point_q)), p0.copy(), point_q
    if c <= epsilon:
        s = float(np.clip(-d / max(a, epsilon), 0.0, 1.0))
        point_p = p0 + s * u
        return float(np.linalg.norm(point_p - q0)), point_p, q0.copy()

    s_numerator = denominator
    s_denominator = denominator
    t_numerator = denominator
    t_denominator = denominator

    if denominator <= epsilon:
        s_numerator = 0.0
        s_denominator = 1.0
        t_numerator = e
        t_denominator = c
    else:
        s_numerator = b * e - c * d
        t_numerator = a * e - b * d
        if s_numerator < 0.0:
            s_numerator = 0.0
            t_numerator = e
            t_denominator = c
        elif s_numerator > s_denominator:
            s_numerator = s_denominator
            t_numerator = e + b
            t_denominator = c

    if t_numerator < 0.0:
        t_numerator = 0.0
        if -d < 0.0:
            s_numerator = 0.0
        elif -d > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d
            s_denominator = a
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if -d + b < 0.0:
            s_numerator = 0.0
        elif -d + b > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d + b
            s_denominator = a

    s = 0.0 if abs(s_numerator) <= epsilon else s_numerator / s_denominator
    t = 0.0 if abs(t_numerator) <= epsilon else t_numerator / t_denominator
    point_p = p0 + s * u
    point_q = q0 + t * v
    return float(np.linalg.norm(point_p - point_q)), point_p, point_q


def segment_distance(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> float:
    return closest_points_on_segments(first[0], first[1], second[0], second[1])[0]


def _self_pairs(contour_index: int, segment_count: int) -> Iterable[SegmentPair]:
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            adjacent = second == first + 1 or (first == 0 and second == segment_count - 1)
            if not adjacent:
                yield SegmentPair(SegmentRef(contour_index, first), SegmentRef(contour_index, second))


def forbidden_pairs(
    assembly: TileAssembly,
    enforce_cross_contour: bool,
    excluded_cross_pairs: frozenset[SegmentPair] = frozenset(),
) -> list[SegmentPair]:
    contours = assembly.contours
    pairs: list[SegmentPair] = []
    for contour_index, contour in enumerate(contours):
        pairs.extend(_self_pairs(contour_index, len(contour)))
    if enforce_cross_contour:
        for first_contour in range(len(contours)):
            for second_contour in range(first_contour + 1, len(contours)):
                for first in range(len(contours[first_contour])):
                    for second in range(len(contours[second_contour])):
                        pair = SegmentPair(
                            SegmentRef(first_contour, first),
                            SegmentRef(second_contour, second),
                        )
                        if pair not in excluded_cross_pairs:
                            pairs.append(pair)
    return pairs


def detect_initial_cross_contacts(assembly: TileAssembly, tolerance: float) -> frozenset[SegmentPair]:
    """Freeze cross-contour pairs already touching in the initial assembly."""

    excluded: set[SegmentPair] = set()
    contours = [contour_segments(np.asarray(contour, dtype=float)) for contour in assembly.contours]
    for first_contour in range(len(contours)):
        for second_contour in range(first_contour + 1, len(contours)):
            for first_index, first in enumerate(contours[first_contour]):
                for second_index, second in enumerate(contours[second_contour]):
                    if segment_distance(first, second) <= tolerance:
                        excluded.add(
                            SegmentPair(
                                SegmentRef(first_contour, first_index),
                                SegmentRef(second_contour, second_index),
                            )
                        )
    return frozenset(excluded)


def first_forbidden_intersection(
    assembly: TileAssembly,
    enforce_cross_contour: bool = False,
    excluded_cross_pairs: frozenset[SegmentPair] = frozenset(),
    tolerance: float = 1.0e-12,
) -> SegmentPair | None:
    for pair in forbidden_pairs(assembly, enforce_cross_contour, excluded_cross_pairs):
        if segments_intersect(
            *segment_endpoints(assembly, pair.first),
            *segment_endpoints(assembly, pair.second),
            tolerance=tolerance,
        ):
            return pair
    return None


def first_transverse_excluded_crossing(
    assembly: TileAssembly,
    excluded_cross_pairs: frozenset[SegmentPair],
    tolerance: float = 1.0e-12,
) -> SegmentPair | None:
    """Reject a formerly intentional contact if it becomes transverse."""

    for pair in excluded_cross_pairs:
        kind = segment_intersection_kind(
            *segment_endpoints(assembly, pair.first),
            *segment_endpoints(assembly, pair.second),
            tolerance=tolerance,
        )
        if kind == "proper":
            return pair
    return None


def minimum_forbidden_clearance(
    assembly: TileAssembly,
    enforce_cross_contour: bool = False,
    excluded_cross_pairs: frozenset[SegmentPair] = frozenset(),
    intersection_tolerance: float = 1.0e-12,
) -> tuple[float, SegmentPair | None]:
    intersection = first_forbidden_intersection(
        assembly,
        enforce_cross_contour=enforce_cross_contour,
        excluded_cross_pairs=excluded_cross_pairs,
        tolerance=intersection_tolerance,
    )
    if intersection is not None:
        return 0.0, intersection

    minimum = np.inf
    minimum_pair: SegmentPair | None = None
    for pair in forbidden_pairs(assembly, enforce_cross_contour, excluded_cross_pairs):
        distance, _, _ = closest_points_on_segments(
            *segment_endpoints(assembly, pair.first),
            *segment_endpoints(assembly, pair.second),
        )
        if distance < minimum:
            minimum = distance
            minimum_pair = pair
    return float(minimum), minimum_pair


def minimum_assembly_edge_length(assembly: TileAssembly) -> float:
    minimum = np.inf
    for contour in assembly.contours:
        points = np.asarray(contour, dtype=float)
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        minimum = min(minimum, float(np.min(lengths)))
    return float(minimum)


def build_active_separators(
    assembly: TileAssembly,
    activation_distance: float,
    enforce_cross_contour: bool = False,
    excluded_cross_pairs: frozenset[SegmentPair] = frozenset(),
    epsilon: float = 1.0e-12,
) -> tuple[Separator, ...]:
    separators: list[Separator] = []
    for pair in forbidden_pairs(assembly, enforce_cross_contour, excluded_cross_pairs):
        first_start, first_end = segment_endpoints(assembly, pair.first)
        second_start, second_end = segment_endpoints(assembly, pair.second)
        distance, first_closest, second_closest = closest_points_on_segments(
            first_start, first_end, second_start, second_end
        )
        if distance > activation_distance:
            continue
        direction = first_closest - second_closest
        norm = float(np.linalg.norm(direction))
        if norm <= epsilon:
            first_direction = first_end - first_start
            direction = np.array((-first_direction[1], first_direction[0]), dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm <= epsilon:
                continue
            first_midpoint = 0.5 * (first_start + first_end)
            second_midpoint = 0.5 * (second_start + second_end)
            if float(np.dot(direction, first_midpoint - second_midpoint)) < 0.0:
                direction = -direction
        separators.append(Separator(pair=pair, normal=direction / norm))
    return tuple(separators)


def separator_margins(assembly: TileAssembly, separators: tuple[Separator, ...], clearance: float) -> np.ndarray:
    margins: list[float] = []
    for separator in separators:
        a, b = segment_endpoints(assembly, separator.pair.first)
        c, d = segment_endpoints(assembly, separator.pair.second)
        normal = separator.normal
        margins.extend(
            (
                float(np.dot(normal, a - c) - clearance),
                float(np.dot(normal, a - d) - clearance),
                float(np.dot(normal, b - c) - clearance),
                float(np.dot(normal, b - d) - clearance),
            )
        )
    return np.asarray(margins, dtype=float)


def edge_length_margins(assembly: TileAssembly, minimum_edge_length: float) -> np.ndarray:
    margins: list[float] = []
    for contour in assembly.contours:
        points = np.asarray(contour, dtype=float)
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        margins.extend((lengths - minimum_edge_length).tolist())
    return np.asarray(margins, dtype=float)


def coordinate_margins(assembly: TileAssembly, coordinate_bound: float) -> np.ndarray:
    points = np.vstack([np.asarray(contour, dtype=float) for contour in assembly.contours])
    return np.concatenate((coordinate_bound - points.ravel(), coordinate_bound + points.ravel()))


def validate_assembly(
    assembly: TileAssembly,
    clearance: float,
    minimum_edge_length: float,
    enforce_cross_contour: bool = False,
    excluded_cross_pairs: frozenset[SegmentPair] = frozenset(),
    tolerance: float = 1.0e-10,
) -> FeasibilityReport:
    edge_length = minimum_assembly_edge_length(assembly)
    if edge_length < minimum_edge_length - tolerance:
        return FeasibilityReport(
            False,
            np.inf,
            edge_length,
            message=f"Minimum edge length {edge_length:.6g} is below {minimum_edge_length:.6g}.",
        )

    transverse_contact_crossing = first_transverse_excluded_crossing(
        assembly, excluded_cross_pairs, tolerance=tolerance
    )
    if transverse_contact_crossing is not None:
        return FeasibilityReport(
            False,
            0.0,
            edge_length,
            violating_pair=transverse_contact_crossing,
            message=(
                "An initially intentional cross-contour contact became a transverse crossing: "
                f"contour {transverse_contact_crossing.first.contour} edge "
                f"{transverse_contact_crossing.first.index} and contour "
                f"{transverse_contact_crossing.second.contour} edge "
                f"{transverse_contact_crossing.second.index}."
            ),
        )

    intersection = first_forbidden_intersection(
        assembly,
        enforce_cross_contour=enforce_cross_contour,
        excluded_cross_pairs=excluded_cross_pairs,
        tolerance=tolerance,
    )
    if intersection is not None:
        return FeasibilityReport(
            False,
            0.0,
            edge_length,
            violating_pair=intersection,
            message=(
                "Forbidden segments intersect or touch: "
                f"contour {intersection.first.contour} edge {intersection.first.index} and "
                f"contour {intersection.second.contour} edge {intersection.second.index}."
            ),
        )

    minimum, pair = minimum_forbidden_clearance(
        assembly,
        enforce_cross_contour=enforce_cross_contour,
        excluded_cross_pairs=excluded_cross_pairs,
        intersection_tolerance=tolerance,
    )
    if minimum < clearance - tolerance:
        return FeasibilityReport(
            False,
            minimum,
            edge_length,
            violating_pair=pair,
            message=f"Forbidden segment clearance {minimum:.6g} is below {clearance:.6g}.",
        )
    return FeasibilityReport(True, minimum, edge_length, pair, "Feasible.")

"""SVG and STL export of a polygonal contour."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

import numpy as np


ASSEMBLY_FILL_COLORS = (
    "#4A7EBB",  # central reference tile
    "#D68B48",  # left surrounding copy
    "#5BA470",  # right surrounding copy
)
ASSEMBLY_OUTLINE_COLOR = "#121212"
OUTER_BOUNDARY_COLOR = "#202020"


def _svg_number(value: float) -> str:
    return format(float(value), ".12g")


def save_piece_assembly_svg(
    contours: Any,
    path: str | Path,
    *,
    outer_boundary: Any | None = None,
    padding: float = 0.1,
) -> None:
    """Save the complete three-piece aggregate as centered filled polygons.

    The relative world coordinates are preserved.  A translation by the
    aggregate bounding-box center is applied only inside the SVG so that the
    assembled geometry is centered in the page/viewBox.
    """

    contour_arrays = tuple(np.asarray(contour, dtype=float) for contour in contours)
    if not contour_arrays or any(len(contour) < 3 for contour in contour_arrays):
        raise ValueError("Every SVG piece contour must contain at least three points.")

    all_points = np.vstack(contour_arrays)
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    center = (minimum + maximum) / 2.0
    size = np.maximum(maximum - minimum, 1.0e-12)
    width = float(size[0] + 2.0 * padding)
    height = float(size[1] + 2.0 * padding)
    view_box = f"{-width / 2.0:.12g} {-height / 2.0:.12g} {width:.12g} {height:.12g}"

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg"',
        f'     viewBox="{view_box}" preserveAspectRatio="xMidYMid meet">',
        (
            f'  <rect x="{_svg_number(-width / 2.0)}" y="{_svg_number(-height / 2.0)}" '
            f'width="{_svg_number(width)}" height="{_svg_number(height)}" fill="white"/>'
        ),
        '  <g stroke-linejoin="round" stroke-linecap="round">',
    ]
    for index, contour in enumerate(contour_arrays):
        shifted = contour - center
        point_text = " ".join(
            f"{_svg_number(point[0])},{_svg_number(-point[1])}" for point in shifted
        )
        fill = ASSEMBLY_FILL_COLORS[index % len(ASSEMBLY_FILL_COLORS)]
        lines.append(
            f'    <polygon id="piece-{index}" points="{point_text}" '
            f'fill="{fill}" stroke="{ASSEMBLY_OUTLINE_COLOR}" stroke-width="0.006"/>'
        )

    if outer_boundary is not None:
        boundary = np.asarray(outer_boundary, dtype=float) - center
        boundary_text = " ".join(
            f"{_svg_number(point[0])},{_svg_number(-point[1])}" for point in boundary
        )
        lines.append(
            f'    <polygon id="outer-shell-boundary" points="{boundary_text}" '
            f'fill="none" stroke="{OUTER_BOUNDARY_COLOR}" stroke-width="0.012"/>'
        )

    lines.extend(("  </g>", "</svg>", ""))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def save_svg(contour: Any, path: str | Path, padding: float = 0.1) -> None:
    points = np.asarray(contour, dtype=float)
    minimum_x, maximum_x = np.min(points[:, 0]), np.max(points[:, 0])
    minimum_y, maximum_y = np.min(points[:, 1]), np.max(points[:, 1])
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    view_box = (
        f"{minimum_x - padding} {-maximum_y - padding} "
        f"{width + 2.0 * padding} {height + 2.0 * padding}"
    )
    path_commands = [f"M {points[0, 0]},{-points[0, 1]}"]
    path_commands.extend(f"L {x},{-y}" for x, y in points[1:])
    path_commands.append("Z")
    document = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">\n'
        f'  <path d="{" ".join(path_commands)}" fill="black" stroke="none" />\n'
        "</svg>\n"
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
    )


def _point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    first = c - a
    second = b - a
    relative = point - a
    dot00 = np.dot(first, first)
    dot01 = np.dot(first, second)
    dot02 = np.dot(first, relative)
    dot11 = np.dot(second, second)
    dot12 = np.dot(second, relative)
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) < 1.0e-15:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denominator
    v = (dot00 * dot12 - dot01 * dot02) / denominator
    return bool(u >= 0.0 and v >= 0.0 and u + v <= 1.0)


def triangulate_polygon(contour: Any) -> list[tuple[int, int, int]]:
    points = np.asarray(contour, dtype=float)
    if len(points) < 3:
        return []
    indices = list(range(len(points)))
    if _signed_area(points) < 0.0:
        indices.reverse()

    triangles: list[tuple[int, int, int]] = []
    while len(indices) > 3:
        ear_found = False
        for local_index in range(len(indices)):
            previous_index = indices[local_index - 1]
            current_index = indices[local_index]
            next_index = indices[(local_index + 1) % len(indices)]
            a, b, c = points[previous_index], points[current_index], points[next_index]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0.0:
                continue
            if any(
                _point_in_triangle(points[candidate], a, b, c)
                for candidate in indices
                if candidate not in (previous_index, current_index, next_index)
            ):
                continue
            triangles.append((previous_index, current_index, next_index))
            indices.pop(local_index)
            ear_found = True
            break
        if not ear_found:
            raise ValueError("Ear clipping failed; the contour may be self-intersecting or degenerate.")

    triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def save_extruded_stl(contour: Any, path: str | Path, thickness: float = 0.1) -> None:
    points = np.asarray(contour, dtype=np.float32)
    if len(points) < 3:
        raise ValueError("A contour must contain at least three points.")

    face_indices = triangulate_polygon(points)
    top = np.hstack((points, np.full((len(points), 1), thickness, dtype=np.float32)))
    bottom = np.hstack((points, np.zeros((len(points), 1), dtype=np.float32)))
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for first, second, third in face_indices:
        triangles.append((np.array((0.0, 0.0, 1.0), dtype=np.float32), top[first], top[second], top[third]))
        triangles.append((np.array((0.0, 0.0, -1.0), dtype=np.float32), bottom[third], bottom[second], bottom[first]))

    for index in range(len(points)):
        following = (index + 1) % len(points)
        p0, p1, p2, p3 = bottom[index], bottom[following], top[following], top[index]
        normal = np.cross(p1 - p0, p3 - p0)
        norm = np.linalg.norm(normal)
        if norm > 0.0:
            normal = normal / norm
        triangles.append((normal, p0, p1, p2))
        triangles.append((normal, p0, p2, p3))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(b"Generated by voderberg-optimizer".ljust(80, b" "))
        stream.write(struct.pack("<I", len(triangles)))
        for normal, first, second, third in triangles:
            stream.write(struct.pack("<3f", *normal))
            stream.write(struct.pack("<3f", *first))
            stream.write(struct.pack("<3f", *second))
            stream.write(struct.pack("<3f", *third))
            stream.write(struct.pack("<H", 0))

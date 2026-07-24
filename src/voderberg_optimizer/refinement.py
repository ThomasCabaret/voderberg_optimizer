"""Insertion of additional free control points into parameter chains."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .backend import np

from .state import SRN2State


def insert_midpoints(points: Any, segment_indices: Iterable[int]) -> Any:
    """Insert one midpoint for every requested segment occurrence.

    The inserted point is initially collinear with the segment endpoints. It is
    subsequently an independent optimization variable. Because the complete
    assembly is rebuilt from the state, every transformed image of that point
    remains linked automatically.
    """

    point_array = np.asarray(points)
    counts = Counter(int(index) for index in segment_indices)
    if not counts:
        return point_array

    for index in counts:
        if index < 0 or index >= len(point_array) - 1:
            raise IndexError(
                f"Cannot refine segment {index}; a chain of {len(point_array)} points has "
                f"segments 0 through {len(point_array) - 2}."
            )

    refined = []
    for index in range(len(point_array) - 1):
        start = point_array[index]
        end = point_array[index + 1]
        refined.append(start)
        insertion_count = counts.get(index, 0)
        for insertion in range(1, insertion_count + 1):
            fraction = insertion / (insertion_count + 1)
            refined.append((1.0 - fraction) * start + fraction * end)
    refined.append(point_array[-1])
    return np.array(refined)


def refine_state(
    state: SRN2State,
    *,
    x_segments: Iterable[int] = (),
    p_segments: Iterable[int] = (),
    q_segments: Iterable[int] = (),
    y_segments: Iterable[int] = (),
) -> SRN2State:
    return state.with_chains(
        x=insert_midpoints(state.x, x_segments),
        p=insert_midpoints(state.p, p_segments),
        q=insert_midpoints(state.q, q_segments),
        y=insert_midpoints(state.y, y_segments),
    )

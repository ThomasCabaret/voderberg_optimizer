"""Exact SRN2 dependency construction from a compact free state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import np

from .constants import NORTH, SOUTH
from .geometry import rotate_point, rotate_points
from .state import SRN2State


@dataclass(frozen=True)
class TileAssembly:
    main_contour: Any
    left_contour: Any
    additional_contours: tuple[Any, ...] = ()

    @property
    def contours(self) -> tuple[Any, ...]:
        """All contours participating in display, validation, and objectives."""

        return (self.main_contour, self.left_contour, *self.additional_contours)


class SRN2Parameterization:
    """Build the three-tile SRN2 geometry by exact dependency propagation.

    The optimizer only sees theta and the X/P/Q/Y/B control data. All repeated
    or transformed vertices are generated here, so equality relations between
    images never need to be supplied as separate numerical constraints.
    """

    def build(self, state: SRN2State) -> TileAssembly:
        north = np.array(NORTH)
        south = np.array(SOUTH)
        theta = state.theta
        x, p, q, y, b = state.x, state.p, state.q, state.y, state.b

        a = rotate_point(south, theta, north) + (south - b)

        p_mapped = (p + (b - a))[::-1]
        q_mapped = (q + (b - a))[::-1]

        q_mapped_180 = -q_mapped[::-1]
        y_180 = -y[::-1]
        q_180 = -q[::-1]
        a_180 = -a
        p_180 = -p[::-1]
        x_180 = -x[::-1]
        p_mapped_180 = -p_mapped[::-1]
        b_180 = -b

        right_chain_used_by_left = np.vstack(
            (q_mapped_180, y_180, q_180, a_180, p_180, x_180, x, p)
        )
        right_chain_after_a = np.vstack((q, y, q_mapped, b, p_mapped))
        main_right = np.vstack(
            (p_mapped_180, b_180, right_chain_used_by_left, a, right_chain_after_a)
        )

        main_left = rotate_points(right_chain_used_by_left, -theta, b_180) + (north - b_180)
        main_contour = np.vstack((north, main_right, south, main_left[::-1]))

        left_lower_source = np.vstack((p_mapped_180[::-1], north, main_left, south))
        left_lower_partial = rotate_points(left_lower_source, -theta, b_180) + (north - b_180)
        transformed_after_a = rotate_points(right_chain_after_a, -theta, b_180) + (north - b_180)
        left_contour = np.vstack(
            (north, main_left, south, transformed_after_a, left_lower_partial[::-1])
        )

        return TileAssembly(main_contour=main_contour, left_contour=left_contour)

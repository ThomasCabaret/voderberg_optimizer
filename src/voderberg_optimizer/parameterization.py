"""Exact SRN2 dependency construction from a compact free state."""

from __future__ import annotations

from .backend import np
from .constants import NORTH, SOUTH
from .geometry import rotate_point, rotate_points
from .state import SRN2State
from .topology import SharedChain, ShellTopology, TileAssembly


class SRN2Parameterization:
    """Build the three-tile SRN2 geometry by exact dependency propagation.

    The optimizer only sees theta and the X/P/Q/Y/B control data. All repeated
    or transformed vertices are generated here, so equality relations between
    images never need to be supplied as separate numerical constraints.

    All three complete pieces and the shell topology are reconstructed at every
    evaluation.  The collision solver therefore validates the same geometry
    used by the shell-thickness objective and by the display.
    """

    def build(self, state: SRN2State) -> TileAssembly:
        north = np.array(NORTH)
        south = np.array(SOUTH)
        theta = state.theta
        x, p, q, y, b = state.x, state.p, state.q, state.y, state.b

        if len(p) < 1 or len(q) < 1:
            raise ValueError("The SRN2 construction requires non-empty P and Q chains.")

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
        left_tail = np.vstack((transformed_after_a, left_lower_partial[::-1]))
        left_contour = np.vstack((north, main_left, south, left_tail))

        # The omitted right copy in the historical code is the half-turn image
        # of the complete reference tile around the midpoint of the two poles.
        # NORTH and SOUTH are opposite, so that midpoint is the origin.
        right_contour = -main_contour

        shell = self._build_shell_topology(
            north=north,
            south=south,
            p_count=len(p),
            q_count=len(q),
            main_right_count=len(main_right),
            left_tail=left_tail,
            main_contour=main_contour,
            right_contour=right_contour,
        )

        return TileAssembly(
            main_contour=main_contour,
            left_contour=left_contour,
            right_contour=right_contour,
            shell=shell,
        )

    @staticmethod
    def _build_shell_topology(
        *,
        north,
        south,
        p_count: int,
        q_count: int,
        main_right_count: int,
        left_tail,
        main_contour,
        right_contour,
    ) -> ShellTopology:
        """Extract the two seams and the external shell boundary analytically.

        After the reference/left interface, the left contour consists of:

            lower seam, left external arc, upper seam.

        After the reference/right interface, the half-turned right contour
        consists of:

            upper seam, right external arc, lower seam.

        The lower seam contains ``q_count`` edges and the upper seam contains
        ``p_count`` edges. This is a property of the exact SRN2 dependency
        construction, not a geometric search performed on the current shape.
        """

        # The right contour starts at P0 and follows the reference/right
        # interface until P1.  Everything after that point is its exposed tail.
        right_inner_end = main_right_count + 1
        right_tail = right_contour[right_inner_end + 1 :]

        left_outer_start = q_count - 1
        left_outer_stop = len(left_tail) - p_count + 1
        right_outer_start = p_count - 1
        right_outer_stop = len(right_tail) - q_count + 1

        left_outer_arc = left_tail[left_outer_start:left_outer_stop]
        right_outer_arc = right_tail[right_outer_start:right_outer_stop]

        if len(left_outer_arc) < 2 or len(right_outer_arc) < 2:
            raise ValueError("The SRN2 shell decomposition produced a degenerate outer arc.")

        # Both seam traces are oriented pole -> primed pole.
        lower_left_trace = np.vstack((south, left_tail[:q_count]))
        lower_right_trace = np.vstack((south, right_tail[-q_count:][::-1]))
        upper_left_trace = np.vstack((north, left_tail[-p_count:][::-1]))
        upper_right_trace = np.vstack((north, right_tail[:p_count]))

        p0_prime = left_outer_arc[0]
        p1_prime = left_outer_arc[-1]

        # P0' -> P1' on the left copy, then P1' -> P0' on the right copy.
        # The shared endpoints are omitted from the second arc because polygon
        # closure supplies the final edge back to P0'.
        outer_boundary = np.vstack((left_outer_arc, right_outer_arc[1:-1]))

        return ShellTopology(
            p0=south,
            p1=north,
            p0_prime=p0_prime,
            p1_prime=p1_prime,
            lower_seam=SharedChain(
                start=south,
                end=p0_prime,
                first_trace=lower_left_trace,
                second_trace=lower_right_trace,
            ),
            upper_seam=SharedChain(
                start=north,
                end=p1_prime,
                first_trace=upper_left_trace,
                second_trace=upper_right_trace,
            ),
            left_outer_arc=left_outer_arc,
            right_outer_arc=right_outer_arc,
            inner_boundary=main_contour,
            outer_boundary=outer_boundary,
        )


__all__ = ["SRN2Parameterization", "TileAssembly"]

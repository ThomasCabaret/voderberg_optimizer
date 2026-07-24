"""Named topological objects reconstructed from the SRN2 free variables.

The numerical optimizer still consumes the historical two-contour view through
``TileAssembly.contours``.  The richer three-piece and shell objects are exposed
separately so display and future shell-thickness objectives can use them without
changing the current optimization problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SharedChain:
    """Two exactly dependent traces of the same geometrical seam.

    Both traces use the same orientation from ``start`` to ``end``.  Keeping the
    two traces is useful for validation and for assigning provenance to future
    collision constraints, even though they should coincide geometrically.
    """

    start: Any
    end: Any
    first_trace: Any
    second_trace: Any


@dataclass(frozen=True)
class ShellTopology:
    """Topological decomposition of the two surrounding pieces.

    Naming follows the problem statement:
      - ``p0`` is the lower three-piece pole;
      - ``p1`` is the upper three-piece pole;
      - ``p0_prime`` and ``p1_prime`` are the ends of the two seams shared by
        the surrounding pieces and therefore lie on the external boundary.

    The contour arrays do not repeat their first point at the end.  Closing is
    implicit, as for all other polygon contours in the project.
    """

    p0: Any
    p1: Any
    p0_prime: Any
    p1_prime: Any
    lower_seam: SharedChain
    upper_seam: SharedChain
    left_outer_arc: Any
    right_outer_arc: Any
    inner_boundary: Any
    outer_boundary: Any


@dataclass(frozen=True)
class TileAssembly:
    """Reconstructed tile assembly with a backward-compatible solver view.

    ``contours`` exposes every complete physical piece to collision validation.
    This is required once the shell-thickness objective depends on the right
    copy as well as the historical reference and left copy.

    ``piece_contours`` is the display-oriented alias of the three physical
    pieces.  Future objectives can read ``shell`` directly.
    """

    main_contour: Any
    left_contour: Any
    additional_contours: tuple[Any, ...] = ()
    right_contour: Any | None = None
    shell: ShellTopology | None = None

    @property
    def contours(self) -> tuple[Any, ...]:
        """Complete contours used by collision validation and constraints."""

        if self.right_contour is None:
            return (self.main_contour, self.left_contour, *self.additional_contours)
        return (
            self.main_contour,
            self.left_contour,
            self.right_contour,
            *self.additional_contours,
        )

    @property
    def piece_contours(self) -> tuple[Any, ...]:
        """All complete tile contours available for visualization."""

        if self.right_contour is None:
            return self.contours
        return (self.main_contour, self.left_contour, self.right_contour)

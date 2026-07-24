"""Composable objective terms independent of any solver backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

from .backend import np
from .constants import NORTH, SOUTH
from .constraints import ConstraintSettings, assembly_margin, soft_barrier
from .geometry import interior_angles, mean_interior_angle, rotate_point
from .shell_metrics import exact_shell_thickness, smooth_shell_thickness
from .parameterization import TileAssembly
from .regularization import state_bending_energy, state_equal_spacing_energy
from .state import SRN2State


class ObjectiveTerm(Protocol):
    name: str

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        ...


@dataclass(frozen=True)
class WeightedTerm:
    term: ObjectiveTerm
    weight: float


@dataclass(frozen=True)
class ContactLengthTerm:
    name: str = "negative_contact_length"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        del assembly
        north = np.array(NORTH)
        south = np.array(SOUTH)
        a = rotate_point(south, state.theta, north) + (south - state.b)
        length = np.linalg.norm(state.p[0] - a) + np.linalg.norm(a - state.q[-1])
        return -length


@dataclass(frozen=True)
class MeanAngleTerm:
    name: str = "negative_mean_angle"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        del state
        return -mean_interior_angle(assembly.main_contour, closed=True)




@dataclass(frozen=True)
class ShellThicknessTerm:
    """Minimize the negative smooth inner-to-outer shell thickness."""

    temperature: float
    name: str = "negative_shell_thickness"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        del state
        return -smooth_shell_thickness(assembly, self.temperature)

    def diagnostics(self, state: SRN2State, assembly: TileAssembly) -> dict[str, float]:
        del state
        return {"exact_shell_thickness": exact_shell_thickness(assembly)}


@dataclass(frozen=True)
class EqualSpacingTerm:
    name: str = "equal_spacing"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        del assembly
        return state_equal_spacing_energy(state)


@dataclass(frozen=True)
class BendingTerm:
    name: str = "bending"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        del assembly
        return state_bending_energy(state)


@dataclass(frozen=True)
class WorstCornerAngleTerm:
    """Maximize the sharpness angle of the worst central-tile corner.

    ``interior_angles`` returns the unsigned angle between the two rays from a
    vertex to its neighbours, in ``[0, pi]``.  It therefore represents
    ``min(alpha, 2*pi - alpha)`` for either a convex or a reflex polygon
    corner.  A low-temperature smooth minimum keeps this term differentiable
    while concentrating its gradient on the sharpest corner.
    """

    temperature_radians: float = math.radians(2.0)
    name: str = "negative_worst_corner_angle"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        del state
        angles = interior_angles(assembly.main_contour, closed=True)
        if len(angles) == 0:
            return 0.0
        temperature = max(float(self.temperature_radians), 1.0e-9)
        # The mean only adds a constant relative to log-sum-exp, so it leaves
        # the gradient weights unchanged while keeping the value interpretable.
        smooth_minimum = -temperature * np.log(
            np.mean(np.exp(-angles / temperature))
        )
        return -smooth_minimum

    def diagnostics(self, state: SRN2State, assembly: TileAssembly) -> dict[str, float]:
        del state
        angles = interior_angles(assembly.main_contour, closed=True)
        if len(angles) == 0:
            return {
                "worst_corner_angle_radians": 0.0,
                "worst_corner_angle_degrees": 0.0,
            }
        exact = float(np.min(angles))
        return {
            "worst_corner_angle_radians": exact,
            "worst_corner_angle_degrees": math.degrees(exact),
        }


@dataclass(frozen=True)
class BarrierTerm:
    settings: ConstraintSettings
    amplitude: float
    name: str = "geometric_barrier"

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        del state
        margin = assembly_margin(assembly, self.settings)
        return soft_barrier(margin, self.amplitude, self.settings.minimum_distance)


@dataclass(frozen=True)
class CompositeObjective:
    terms: tuple[WeightedTerm, ...]

    def value(self, state: SRN2State, assembly: TileAssembly) -> Any:
        total = 0.0
        for weighted in self.terms:
            total = total + weighted.weight * weighted.term.value(state, assembly)
        return total

    def breakdown(self, state: SRN2State, assembly: TileAssembly) -> dict[str, float]:
        result: dict[str, float] = {}
        for weighted in self.terms:
            raw = weighted.term.value(state, assembly)
            result[weighted.term.name] = float(raw)
            result[f"weighted_{weighted.term.name}"] = float(weighted.weight * raw)
            diagnostics = getattr(weighted.term, "diagnostics", None)
            if diagnostics is not None:
                result.update(diagnostics(state, assembly))
        result["total"] = sum(value for key, value in result.items() if key.startswith("weighted_"))
        return result

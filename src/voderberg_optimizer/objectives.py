"""Composable objective terms independent of any solver backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .backend import np
from .constants import NORTH, SOUTH
from .constraints import ConstraintSettings, assembly_margin, soft_barrier
from .geometry import mean_interior_angle, rotate_point
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
        result["total"] = sum(value for key, value in result.items() if key.startswith("weighted_"))
        return result

"""Weak mesh-quality energies for free points inserted in control chains."""

from __future__ import annotations

from typing import Any

from .backend import np
from .constants import EPSILON
from .state import SRN2State


def equal_spacing_energy(points: Any) -> Any:
    """Scale-independent variance of consecutive edge lengths."""

    if len(points) < 3:
        return 0.0
    lengths = np.sqrt(np.sum((points[1:] - points[:-1]) ** 2, axis=1) + EPSILON)
    mean_length = np.mean(lengths)
    normalized = lengths / (mean_length + EPSILON)
    return np.mean((normalized - 1.0) ** 2)


def bending_energy(points: Any) -> Any:
    """Normalized second-difference energy; zero on evenly spaced straight chains."""

    if len(points) < 3:
        return 0.0
    first_differences = points[1:] - points[:-1]
    scale = np.mean(np.sum(first_differences**2, axis=1)) + EPSILON
    second_differences = points[:-2] - 2.0 * points[1:-1] + points[2:]
    return np.mean(np.sum(second_differences**2, axis=1)) / scale


def state_equal_spacing_energy(state: SRN2State) -> Any:
    return sum(equal_spacing_energy(chain) for chain in (state.x, state.p, state.q, state.y))


def state_bending_energy(state: SRN2State) -> Any:
    return sum(bending_energy(chain) for chain in (state.x, state.p, state.q, state.y))

"""Solver interface used by the application layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..problem import OptimizationProblem


@dataclass(frozen=True)
class SolverIteration:
    index: int
    vector: Any
    objective: float
    message: str = ""
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverResult:
    vector: Any
    objective: float
    iterations: int
    success: bool
    message: str
    metadata: dict[str, float] = field(default_factory=dict)


IterationCallback = Callable[[SolverIteration], None]


class Solver(Protocol):
    def solve(
        self,
        problem: OptimizationProblem,
        initial_vector: Any,
        callback: IterationCallback | None = None,
    ) -> SolverResult:
        ...

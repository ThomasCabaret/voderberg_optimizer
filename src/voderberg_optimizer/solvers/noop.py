"""A placeholder solver that leaves the initial state unchanged."""

from __future__ import annotations

from typing import Any

from .base import IterationCallback, SolverResult
from ..problem import OptimizationProblem


class NoOpSolver:
    def solve(
        self,
        problem: OptimizationProblem,
        initial_vector: Any,
        callback: IterationCallback | None = None,
    ) -> SolverResult:
        del callback
        value = float(problem.value(initial_vector))
        return SolverResult(
            vector=initial_vector,
            objective=value,
            iterations=0,
            success=True,
            message="No-op solver selected; the initial state was evaluated without modification.",
        )

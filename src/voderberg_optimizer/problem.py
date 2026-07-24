"""Solver-neutral optimization problem wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import grad

from .objectives import CompositeObjective
from .parameterization import SRN2Parameterization, TileAssembly
from .state import SRN2State, StateLayout


@dataclass(frozen=True)
class ProblemEvaluation:
    state: SRN2State
    assembly: TileAssembly
    objective: float
    breakdown: dict[str, float]


class OptimizationProblem:
    def __init__(
        self,
        layout: StateLayout,
        parameterization: SRN2Parameterization,
        objective: CompositeObjective,
    ) -> None:
        self.layout = layout
        self.parameterization = parameterization
        self.objective = objective
        self._gradient = grad(self.value)

    def state_from_vector(self, vector: Any) -> SRN2State:
        return SRN2State.from_vector(vector, self.layout).with_normalized_theta()

    def value(self, vector: Any) -> Any:
        state = self.state_from_vector(vector)
        assembly = self.parameterization.build(state)
        return self.objective.value(state, assembly)

    def gradient(self, vector: Any) -> Any:
        return self._gradient(vector)

    def evaluate(self, vector: Any) -> ProblemEvaluation:
        state = self.state_from_vector(vector)
        assembly = self.parameterization.build(state)
        breakdown = self.objective.breakdown(state, assembly)
        return ProblemEvaluation(
            state=state,
            assembly=assembly,
            objective=breakdown["total"],
            breakdown=breakdown,
        )

"""Generic SciPy adapter; intentionally independent of objective composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.optimize import minimize

from .base import IterationCallback, SolverIteration, SolverResult
from ..problem import OptimizationProblem


@dataclass(frozen=True)
class ScipySolver:
    method: str = "L-BFGS-B"
    maximum_iterations: int = 500
    finite_difference_step: float = 1.0e-12

    def solve(
        self,
        problem: OptimizationProblem,
        initial_vector: Any,
        callback: IterationCallback | None = None,
    ) -> SolverResult:
        iteration = 0

        def scipy_callback(vector: Any) -> None:
            nonlocal iteration
            if callback is not None:
                callback(
                    SolverIteration(
                        index=iteration,
                        vector=vector.copy(),
                        objective=float(problem.value(vector)),
                    )
                )
            iteration += 1

        result = minimize(
            fun=problem.value,
            x0=initial_vector,
            jac=problem.gradient,
            method=self.method,
            callback=scipy_callback,
            options={
                "maxiter": self.maximum_iterations,
                "eps": self.finite_difference_step,
                "disp": True,
            },
        )
        return SolverResult(
            vector=result.x,
            objective=float(result.fun),
            iterations=int(getattr(result, "nit", iteration)),
            success=bool(result.success),
            message=str(result.message),
        )

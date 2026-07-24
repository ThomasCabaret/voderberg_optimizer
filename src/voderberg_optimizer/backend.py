"""Numerical backend with an Autograd-first and NumPy fallback policy."""

from __future__ import annotations

from typing import Any, Callable

try:
    from autograd import grad as _autograd_grad
    import autograd.numpy as np

    AUTOGRAD_AVAILABLE = True

    def grad(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        return _autograd_grad(function)

except ImportError:
    import numpy as np

    AUTOGRAD_AVAILABLE = False

    def grad(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Central finite-difference fallback used only when Autograd is absent."""

        def finite_difference(vector: Any) -> Any:
            values = np.asarray(vector, dtype=float)
            result = np.zeros_like(values)
            for index in range(values.size):
                step = 1.0e-7 * max(1.0, abs(values[index]))
                forward = values.copy()
                backward = values.copy()
                forward[index] += step
                backward[index] -= step
                result[index] = (function(forward) - function(backward)) / (2.0 * step)
            return result

        return finite_difference

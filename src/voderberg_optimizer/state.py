"""Optimization state and dynamic vector encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend import np


@dataclass(frozen=True)
class StateLayout:
    x_points: int
    p_points: int
    q_points: int
    y_points: int

    @property
    def vector_size(self) -> int:
        return 1 + 2 * (self.x_points + self.p_points + self.q_points + self.y_points + 1)

    def as_metadata(self) -> dict[str, int]:
        return {
            "x_points": self.x_points,
            "p_points": self.p_points,
            "q_points": self.q_points,
            "y_points": self.y_points,
        }


@dataclass(frozen=True)
class SRN2State:
    theta: Any
    x: Any
    p: Any
    q: Any
    y: Any
    b: Any

    @property
    def layout(self) -> StateLayout:
        return StateLayout(len(self.x), len(self.p), len(self.q), len(self.y))

    def to_vector(self) -> Any:
        return np.concatenate(
            (
                np.atleast_1d(self.theta),
                np.ravel(self.x),
                np.ravel(self.p),
                np.ravel(self.q),
                np.ravel(self.y),
                np.ravel(self.b),
            )
        )

    @classmethod
    def from_vector(cls, vector: Any, layout: StateLayout) -> "SRN2State":
        values = vector
        if len(values) != layout.vector_size:
            raise ValueError(
                f"Expected {layout.vector_size} values for layout {layout}, got {len(values)}."
            )

        cursor = 1

        def take_points(count: int) -> Any:
            nonlocal cursor
            result = values[cursor : cursor + 2 * count].reshape((count, 2))
            cursor += 2 * count
            return result

        theta = values[0]
        x = take_points(layout.x_points)
        p = take_points(layout.p_points)
        q = take_points(layout.q_points)
        y = take_points(layout.y_points)
        b = values[cursor : cursor + 2]
        return cls(theta=theta, x=x, p=p, q=q, y=y, b=b)

    def with_normalized_theta(self) -> "SRN2State":
        normalized = ((self.theta + np.pi) % (2.0 * np.pi)) - np.pi
        return SRN2State(
            theta=normalized,
            x=self.x,
            p=self.p,
            q=self.q,
            y=self.y,
            b=self.b,
        )

    def with_chains(self, *, x: Any | None = None, p: Any | None = None, q: Any | None = None, y: Any | None = None) -> "SRN2State":
        return SRN2State(
            theta=self.theta,
            x=self.x if x is None else x,
            p=self.p if p is None else p,
            q=self.q if q is None else q,
            y=self.y if y is None else y,
            b=self.b,
        )

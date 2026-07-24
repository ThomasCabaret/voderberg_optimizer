import numpy as np

from voderberg_optimizer.regularization import bending_energy, equal_spacing_energy


def test_equal_spacing_is_zero_for_uniform_chain() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert equal_spacing_energy(points) < 1.0e-20


def test_equal_spacing_penalizes_clustered_points() -> None:
    uniform = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    clustered = np.array([[0.0, 0.0], [0.01, 0.0], [2.0, 0.0]])
    assert equal_spacing_energy(clustered) > equal_spacing_energy(uniform)


def test_bending_is_zero_for_evenly_spaced_straight_chain() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert bending_energy(points) < 1.0e-20


def test_bending_penalizes_zigzag() -> None:
    straight = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    zigzag = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]])
    assert bending_energy(zigzag) > bending_energy(straight)

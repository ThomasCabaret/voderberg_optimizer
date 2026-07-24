import numpy as np

from voderberg_optimizer.backend import grad
from voderberg_optimizer.shell_metrics import (
    exact_shell_thickness,
    pairwise_segment_distances,
    smooth_shell_thickness,
)
from voderberg_optimizer.topology import SharedChain, ShellTopology, TileAssembly


def square_assembly(outer_half_extent: float = 2.0) -> TileAssembly:
    inner = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=float,
    )
    outer = outer_half_extent * np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=float,
    )
    seam = SharedChain(inner[0], outer[0], np.vstack((inner[0], outer[0])), np.vstack((inner[0], outer[0])))
    shell = ShellTopology(
        p0=inner[0],
        p1=inner[2],
        p0_prime=outer[0],
        p1_prime=outer[2],
        lower_seam=seam,
        upper_seam=seam,
        left_outer_arc=outer[:3],
        right_outer_arc=np.vstack((outer[2:], outer[:1])),
        inner_boundary=inner,
        outer_boundary=outer,
    )
    return TileAssembly(inner, outer, shell=shell)


def test_exact_and_smooth_shell_thickness_on_concentric_squares() -> None:
    assembly = square_assembly(2.0)
    distances = pairwise_segment_distances(
        assembly.shell.inner_boundary,
        assembly.shell.outer_boundary,
    )
    assert abs(float(np.min(distances)) - 1.0) < 1.0e-9
    assert abs(exact_shell_thickness(assembly) - 1.0) < 1.0e-12
    assert abs(float(smooth_shell_thickness(assembly, 0.001)) - 1.0) < 1.0e-6


def test_smooth_shell_thickness_has_outward_gradient() -> None:
    base_inner = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=float,
    )
    base_outer = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=float,
    )

    def value(vector):
        extent = vector[0]
        inner = base_inner
        outer = extent * base_outer
        seam = SharedChain(inner[0], outer[0], np.vstack((inner[0], outer[0])), np.vstack((inner[0], outer[0])))
        shell = ShellTopology(
            p0=inner[0],
            p1=inner[2],
            p0_prime=outer[0],
            p1_prime=outer[2],
            lower_seam=seam,
            upper_seam=seam,
            left_outer_arc=outer[:3],
            right_outer_arc=np.concatenate((outer[2:], outer[:1]), axis=0),
            inner_boundary=inner,
            outer_boundary=outer,
        )
        assembly = TileAssembly(inner, outer, shell=shell)
        return smooth_shell_thickness(assembly, 0.001)

    derivative = grad(value)(np.array([2.0]))
    assert float(derivative[0]) > 0.9

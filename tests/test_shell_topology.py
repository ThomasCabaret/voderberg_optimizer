import numpy as np

from voderberg_optimizer.parameterization import SRN2Parameterization
from voderberg_optimizer.state import SRN2State


def sample_state(p_points: int = 3, q_points: int = 3) -> SRN2State:
    p = np.column_stack(
        (
            np.linspace(0.24, 0.08, p_points),
            np.linspace(-0.45, -0.78, p_points),
        )
    )
    return SRN2State(
        theta=0.23,
        x=np.array([[0.16, 0.55], [0.31, 0.21], [0.38, -0.18], [0.28, -0.38]]),
        p=p,
        q=np.column_stack(
            (
                np.linspace(-0.08, -0.23, q_points),
                np.linspace(-0.76, -0.46, q_points),
            )
        ),
        y=np.array([[-0.31, -0.30], [-0.40, -0.02], [-0.34, 0.27]]),
        b=np.array([0.09, -0.92]),
    )


def test_right_piece_is_the_half_turn_of_the_reference_piece() -> None:
    assembly = SRN2Parameterization().build(sample_state())
    assert assembly.right_contour is not None
    np.testing.assert_allclose(assembly.right_contour, -assembly.main_contour, atol=1.0e-12)


def test_solver_and_display_expose_all_three_physical_pieces() -> None:
    assembly = SRN2Parameterization().build(sample_state())
    assert len(assembly.contours) == 3
    assert len(assembly.piece_contours) == 3
    assert assembly.piece_contours[0] is assembly.main_contour
    assert assembly.piece_contours[1] is assembly.left_contour
    assert assembly.piece_contours[2] is assembly.right_contour


def test_shared_seams_are_reconstructed_with_matching_orientations() -> None:
    assembly = SRN2Parameterization().build(sample_state())
    assert assembly.shell is not None
    shell = assembly.shell

    np.testing.assert_allclose(
        shell.lower_seam.first_trace,
        shell.lower_seam.second_trace,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        shell.upper_seam.first_trace,
        shell.upper_seam.second_trace,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(shell.lower_seam.start, shell.p0, atol=1.0e-12)
    np.testing.assert_allclose(shell.lower_seam.end, shell.p0_prime, atol=1.0e-12)
    np.testing.assert_allclose(shell.upper_seam.start, shell.p1, atol=1.0e-12)
    np.testing.assert_allclose(shell.upper_seam.end, shell.p1_prime, atol=1.0e-12)


def test_outer_boundary_joins_the_two_exposed_arcs() -> None:
    assembly = SRN2Parameterization().build(sample_state())
    assert assembly.shell is not None
    shell = assembly.shell

    np.testing.assert_allclose(shell.left_outer_arc[0], shell.p0_prime, atol=1.0e-12)
    np.testing.assert_allclose(shell.left_outer_arc[-1], shell.p1_prime, atol=1.0e-12)
    np.testing.assert_allclose(shell.right_outer_arc[0], shell.p1_prime, atol=1.0e-12)
    np.testing.assert_allclose(shell.right_outer_arc[-1], shell.p0_prime, atol=1.0e-12)
    np.testing.assert_allclose(shell.outer_boundary[0], shell.p0_prime, atol=1.0e-12)
    np.testing.assert_allclose(shell.inner_boundary, assembly.main_contour, atol=1.0e-12)

    expected = np.vstack((shell.left_outer_arc, shell.right_outer_arc[1:-1]))
    np.testing.assert_allclose(shell.outer_boundary, expected, atol=1.0e-12)


def test_shell_decomposition_tracks_p_and_q_chain_refinement() -> None:
    for p_points, q_points in ((1, 1), (2, 3), (3, 2), (6, 4)):
        assembly = SRN2Parameterization().build(
            sample_state(p_points=p_points, q_points=q_points)
        )
        assert assembly.shell is not None
        shell = assembly.shell
        assert len(shell.lower_seam.first_trace) == q_points + 1
        assert len(shell.upper_seam.first_trace) == p_points + 1
        np.testing.assert_allclose(
            shell.lower_seam.first_trace,
            shell.lower_seam.second_trace,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            shell.upper_seam.first_trace,
            shell.upper_seam.second_trace,
            atol=1.0e-12,
        )

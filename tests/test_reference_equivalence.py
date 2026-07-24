import numpy as np

from voderberg_optimizer.parameterization import SRN2Parameterization
from voderberg_optimizer.state import SRN2State


def rotate_point(point, angle, center):
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return center + rotation @ (point - center)


def original_create_contour_srn2(theta, x, p, q, y, b):
    north = np.array([0.0, 1.0])
    south = np.array([0.0, -1.0])
    a = rotate_point(south, theta, north) + (south - b)
    p_mapped = (p + (b - a))[::-1]
    q_mapped = (q + (b - a))[::-1]

    q_mapped_180 = -q_mapped[::-1]
    y_180 = -y[::-1]
    q_180 = -q[::-1]
    a_180 = -a
    p_180 = -p[::-1]
    x_180 = -x[::-1]
    p_mapped_180 = -p_mapped[::-1]
    b_180 = -b
    main_right_for_left = np.vstack(
        [q_mapped_180, y_180, q_180, a_180, p_180, x_180, x, p]
    )
    main_right_after_a = np.vstack([q, y, q_mapped, b, p_mapped])
    main_right = np.vstack(
        [p_mapped_180, b_180, main_right_for_left, a, main_right_after_a]
    )
    main_left = np.array(
        [rotate_point(point, -theta, b_180) + (north - b_180) for point in main_right_for_left]
    )
    main_contour = np.vstack([north, main_right, south, main_left[::-1]])

    lower_source = np.vstack([p_mapped_180[::-1], north, main_left, south])
    lower_partial = np.array(
        [rotate_point(point, -theta, b_180) + (north - b_180) for point in lower_source]
    )
    transformed_after_a = np.array(
        [rotate_point(point, -theta, b_180) + (north - b_180) for point in main_right_after_a]
    )
    left_contour = np.vstack(
        [north, main_left, south, transformed_after_a, lower_partial[::-1]]
    )
    return main_contour, left_contour


def test_refactored_parameterization_matches_original_formula() -> None:
    random = np.random.default_rng(1234)
    state = SRN2State(
        theta=0.37,
        x=random.normal(size=(7, 2)),
        p=random.normal(size=(3, 2)),
        q=random.normal(size=(3, 2)),
        y=random.normal(size=(3, 2)),
        b=random.normal(size=2),
    )
    expected = original_create_contour_srn2(
        state.theta, state.x, state.p, state.q, state.y, state.b
    )
    actual = SRN2Parameterization().build(state)
    np.testing.assert_allclose(actual.main_contour, expected[0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual.left_contour, expected[1], rtol=0.0, atol=1.0e-12)

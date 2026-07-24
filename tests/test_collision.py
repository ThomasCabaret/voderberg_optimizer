import numpy as np

from voderberg_optimizer.collision import (
    build_active_separators,
    closest_points_on_segments,
    detect_initial_cross_contacts,
    separator_margins,
    validate_assembly,
)
from voderberg_optimizer.parameterization import TileAssembly


def test_segment_distance_detects_crossing() -> None:
    distance, first, second = closest_points_on_segments(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
    )
    assert distance <= 1.0e-12
    np.testing.assert_allclose(first, second, atol=1.0e-12)


def test_bow_tie_fails_exact_clearance_validation() -> None:
    bow_tie = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    square = np.array([[3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0]])
    report = validate_assembly(
        TileAssembly(bow_tie, square),
        clearance=0.0,
        minimum_edge_length=1.0e-6,
    )
    assert not report.feasible
    assert report.minimum_clearance <= 1.0e-12


def test_local_separator_is_satisfied_by_reference_square() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    other = square + np.array([3.0, 0.0])
    assembly = TileAssembly(square, other)
    separators = build_active_separators(assembly, activation_distance=2.0)
    margins = separator_margins(assembly, separators, clearance=0.25)
    assert len(separators) == 4
    assert np.min(margins) >= 0.75 - 1.0e-12


def test_initial_cross_contour_contacts_can_be_excluded() -> None:
    first = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    second = first + np.array([1.0, 0.0])
    exclusions = detect_initial_cross_contacts(TileAssembly(first, second), tolerance=1.0e-12)
    assert exclusions


def test_intersection_classifier_distinguishes_proper_touch_and_overlap() -> None:
    from voderberg_optimizer.collision import segment_intersection_kind

    assert segment_intersection_kind(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, -1.0]),
        np.array([1.0, 1.0]),
    ) == "proper"
    assert segment_intersection_kind(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
    ) == "touch"
    assert segment_intersection_kind(
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([3.0, 0.0]),
    ) == "overlap"


def test_excluded_initial_contact_cannot_become_transverse() -> None:
    from voderberg_optimizer.collision import (
        SegmentPair,
        SegmentRef,
        first_transverse_excluded_crossing,
    )

    first = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, -2.0], [0.0, -2.0]])
    second = np.array([[1.0, -1.0], [1.0, 1.0], [3.0, 1.0], [3.0, -1.0]])
    pair = SegmentPair(SegmentRef(0, 0), SegmentRef(1, 0))
    found = first_transverse_excluded_crossing(TileAssembly(first, second), frozenset({pair}))
    assert found == pair


def test_cross_contour_validation_supports_additional_contours() -> None:
    first = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    second = first + np.array([3.0, 0.0])
    third = first + np.array([0.5, 0.5])
    assembly = TileAssembly(first, second, additional_contours=(third,))
    report = validate_assembly(
        assembly,
        clearance=0.0,
        minimum_edge_length=1.0e-6,
        enforce_cross_contour=True,
    )
    assert not report.feasible
    assert report.violating_pair is not None
    assert 2 in (report.violating_pair.first.contour, report.violating_pair.second.contour)

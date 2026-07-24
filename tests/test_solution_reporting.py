import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from voderberg_optimizer.exporters import save_piece_assembly_svg
from voderberg_optimizer.parameterization import SRN2Parameterization
from voderberg_optimizer.solution_report import (
    save_solution_json,
    save_standalone_solution_script,
)
from voderberg_optimizer.state import SRN2State


def sample_state() -> SRN2State:
    return SRN2State(
        theta=0.23,
        x=np.array([[0.16, 0.55], [0.31, 0.21], [0.38, -0.18], [0.28, -0.38]]),
        p=np.array([[0.24, -0.45], [0.16, -0.61], [0.08, -0.78]]),
        q=np.array([[-0.08, -0.76], [-0.15, -0.61], [-0.23, -0.46]]),
        y=np.array([[-0.31, -0.30], [-0.40, -0.02], [-0.34, 0.27]]),
        b=np.array([0.09, -0.92]),
    )


def test_assembly_svg_contains_three_centered_colored_polygons(tmp_path: Path) -> None:
    assembly = SRN2Parameterization().build(sample_state())
    assert assembly.shell is not None
    output = tmp_path / "assembly.svg"
    save_piece_assembly_svg(
        assembly.piece_contours,
        output,
        outer_boundary=assembly.shell.outer_boundary,
    )
    text = output.read_text(encoding="utf-8")
    assert 'id="piece-0"' in text
    assert 'id="piece-1"' in text
    assert 'id="piece-2"' in text
    assert 'id="outer-shell-boundary"' in text
    assert "#4A7EBB" in text
    assert "#D68B48" in text
    assert "#5BA470" in text

    view_box_text = text.split('viewBox="', 1)[1].split('"', 1)[0]
    x, y, width, height = (float(value) for value in view_box_text.split())
    assert np.isclose(x, -width / 2.0)
    assert np.isclose(y, -height / 2.0)


def test_standalone_script_reconstructs_the_same_geometry(tmp_path: Path) -> None:
    state = sample_state()
    assembly = SRN2Parameterization().build(state)
    script = tmp_path / "solution_definition.py"
    save_standalone_solution_script(state, script, metadata={"test": True})

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    np.testing.assert_allclose(payload["free_variables"]["theta"], state.theta, atol=0.0)
    np.testing.assert_allclose(payload["free_variables"]["X"], state.x, atol=0.0)
    central = payload["central_piece"]
    np.testing.assert_allclose(central["ordered_points"], assembly.main_contour, atol=1.0e-13)
    assert central["cyclic"] is True
    assert central["start_vertex"] == "P1"
    assert central["point_count"] == len(assembly.main_contour)
    assert len(central["point_labels"]) == len(assembly.main_contour)
    assert central["cyclic_edges"][-1] == [len(assembly.main_contour) - 1, 0]
    assert "pieces" not in payload
    assert "shell" not in payload


def test_json_report_contains_free_variables_and_ordered_contours(tmp_path: Path) -> None:
    state = sample_state()
    assembly = SRN2Parameterization().build(state)
    output = tmp_path / "solution_definition.json"
    save_solution_json(state, assembly, output, metadata={"objective": -0.1})
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["format"] == "voderberg-srn2-central-piece-v2"
    assert payload["construction"]["language"] == "Python 3"
    assert payload["construction"]["scope"] == "central piece only"
    assert payload["metadata"]["objective"] == -0.1
    np.testing.assert_allclose(payload["free_variables"]["P"], state.p)
    central = payload["central_piece"]
    np.testing.assert_allclose(central["ordered_points"], assembly.main_contour)
    assert central["cyclic"] is True
    assert central["point_count"] == len(assembly.main_contour)
    assert central["cyclic_edges"][-1] == [len(assembly.main_contour) - 1, 0]
    assert "pieces" not in payload
    assert "shell" not in payload

"""Compatibility I/O for the original indexed variable files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .state import SRN2State, StateLayout


def load_indexed_vector(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    values: list[float] = []
    metadata: dict[str, str] = {}
    reading_values = False

    with Path(path).open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line == "# VARS_START":
                reading_values = True
                continue
            if line == "# VARS_END":
                reading_values = False
                continue
            if reading_values:
                _, value = line.split(":", 1)
                values.append(float(value.strip()))
            elif line.startswith("# ") and ":" in line:
                key, value = line[2:].split(":", 1)
                metadata[key.strip()] = value.strip()

    if not values:
        raise ValueError(f"No values found in {path}.")
    return np.asarray(values, dtype=np.float64), metadata


def load_state(path: str | Path, layout: StateLayout) -> SRN2State:
    vector, _ = load_indexed_vector(path)
    return SRN2State.from_vector(vector, layout)


def save_state(
    path: str | Path,
    state: SRN2State,
    metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_metadata: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **state.layout.as_metadata(),
        **(metadata or {}),
    }
    with output.open("w", encoding="utf-8") as stream:
        for key, value in all_metadata.items():
            stream.write(f"# {key}: {value}\n")
        stream.write("# VARS_START\n")
        for index, value in enumerate(np.asarray(state.to_vector(), dtype=float)):
            stream.write(f"{index}: {value:.17g}\n")
        stream.write("# VARS_END\n")


def autosave_state(
    directory: str | Path,
    state: SRN2State,
    iteration: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_directory / f"{timestamp}_iter{iteration:05d}.init"
    save_state(output_path, state, metadata=metadata)
    return output_path


def load_state_auto(path: str | Path, fallback_layout: StateLayout | None = None) -> SRN2State:
    vector, metadata = load_indexed_vector(path)
    required = ("x_points", "p_points", "q_points", "y_points")
    if all(key in metadata for key in required):
        layout = StateLayout(*(int(metadata[key]) for key in required))
    elif fallback_layout is not None:
        layout = fallback_layout
    else:
        raise ValueError(
            f"State file {path} does not contain dynamic layout metadata and no fallback was supplied."
        )
    return SRN2State.from_vector(vector, layout)

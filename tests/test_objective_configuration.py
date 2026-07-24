from pathlib import Path

from voderberg_optimizer.config import load_settings


def test_objective_terms_are_loaded_from_toml(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.toml"
    settings_file.write_text(
        """
[paths]
initial_state = "state.init"

[objective]

[[objective.terms]]
name = "equal_spacing"
weight = 3.5

[[objective.terms]]
name = "bending"
weight = 0.25
""".strip(),
        encoding="utf-8",
    )
    settings = load_settings(settings_file)
    assert [(term.name, term.weight) for term in settings.objective.terms] == [
        ("equal_spacing", 3.5),
        ("bending", 0.25),
    ]

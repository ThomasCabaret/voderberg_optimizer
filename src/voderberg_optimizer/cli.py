"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .app import display_state, export_state, load_selected_state, optimize
from .config import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize and inspect an SRN2 Voderberg tile.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize_parser = subparsers.add_parser("optimize", help="Run feasible continuation optimization.")
    optimize_parser.add_argument("--settings", type=Path, default=Path("settings.toml"))
    optimize_parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening the Pygame window.",
    )

    display_parser = subparsers.add_parser("display", help="Display an initial, final, or explicit state.")
    display_parser.add_argument("--settings", type=Path, default=Path("settings.toml"))
    display_parser.add_argument("--final", action="store_true", help="Display paths.optimized_state.")
    display_parser.add_argument("--state", type=Path, default=None, help="Display this .init file.")

    export_parser = subparsers.add_parser("export", help="Export an initial, final, or explicit state.")
    export_parser.add_argument("--settings", type=Path, default=Path("settings.toml"))
    export_parser.add_argument("--initial", action="store_true", help="Export the initial state instead of the final state.")
    export_parser.add_argument("--state", type=Path, default=None, help="Export this .init file.")

    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    settings = load_settings(arguments.settings)

    if arguments.command == "optimize":
        summary = optimize(settings, display=not arguments.no_display)
        print(f"Initial objective: {summary.initial_objective:.12g}")
        print(f"Final objective:   {summary.final_objective:.12g}")
        print(f"Success:           {summary.success}")
        print(f"Message:           {summary.message}")
        print(f"Final state:       {settings.paths.optimized_state}")
        return

    if arguments.command == "display":
        state = load_selected_state(settings, state_path=arguments.state, final=arguments.final)
        display_state(settings, state)
        return

    if arguments.command == "export":
        state = load_selected_state(
            settings,
            state_path=arguments.state,
            final=not arguments.initial and arguments.state is None,
        )
        export_state(settings, state)
        print(f"SVG: {settings.paths.svg_output}")
        print(f"STL: {settings.paths.stl_output}")
        return

    raise RuntimeError(f"Unsupported command: {arguments.command}")


if __name__ == "__main__":
    main()

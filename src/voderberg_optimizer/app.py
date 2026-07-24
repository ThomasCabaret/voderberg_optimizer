"""Application orchestration with responsive visualization and no global state."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acquisition import acquire_normalized_keypoints, state_from_legacy_keypoints
from .config import AppSettings
from .exporters import save_extruded_stl, save_piece_assembly_svg
from .objective_factory import build_objective
from .parameterization import SRN2Parameterization
from .persistence import autosave_state, load_state, load_state_auto, save_state
from .problem import OptimizationProblem, ProblemEvaluation
from .refinement import refine_state
from .solvers import create_solver
from .solvers.base import SolverIteration, SolverResult
from .shell_metrics import exact_shell_thickness
from .solution_report import save_solution_json, save_standalone_solution_script
from .state import SRN2State, StateLayout
from .visualization import PygameViewer, ViewerSettings


@dataclass(frozen=True)
class RunSummary:
    initial_objective: float
    final_objective: float
    initial_shell_thickness: float | None
    final_shell_thickness: float | None
    success: bool
    message: str
    final_state: SRN2State


@dataclass(frozen=True)
class _ViewerUpdate:
    evaluation: ProblemEvaluation
    caption: str


def configured_layout(settings: AppSettings) -> StateLayout:
    layout = settings.state_layout
    return StateLayout(layout.x_points, layout.p_points, layout.q_points, layout.y_points)


def load_initial_state(settings: AppSettings) -> SRN2State:
    layout = configured_layout(settings)
    mode = settings.initialization.mode.strip().lower()
    if mode == "file":
        if not settings.paths.initial_state.is_file():
            raise FileNotFoundError(
                f"Initial state not found: {settings.paths.initial_state}. "
                "Copy the legacy .init file there or select initialization.mode = 'image'."
            )
        state = load_state(settings.paths.initial_state, layout)
    elif mode == "image":
        display = settings.display
        keypoints = acquire_normalized_keypoints(
            settings.paths.acquisition_image,
            (display.screen_width, display.screen_height),
            display.margin_pixels,
        )
        state = state_from_legacy_keypoints(keypoints, layout)
    else:
        raise ValueError("initialization.mode must be 'file' or 'image'.")

    refinement = settings.refinement
    return refine_state(
        state,
        x_segments=refinement.x_segments,
        p_segments=refinement.p_segments,
        q_segments=refinement.q_segments,
        y_segments=refinement.y_segments,
    )


def load_selected_state(settings: AppSettings, state_path: Path | None = None, final: bool = False) -> SRN2State:
    if state_path is not None:
        return load_state_auto(state_path, configured_layout(settings))
    if final:
        if not settings.paths.optimized_state.is_file():
            raise FileNotFoundError(f"Optimized state not found: {settings.paths.optimized_state}")
        return load_state_auto(settings.paths.optimized_state, configured_layout(settings))
    return load_initial_state(settings)


def build_problem(settings: AppSettings, layout: StateLayout) -> OptimizationProblem:
    return OptimizationProblem(layout, SRN2Parameterization(), build_objective(settings))


def _create_viewer(settings: AppSettings, enabled: bool | None = None) -> PygameViewer | None:
    display_enabled = settings.display.enabled if enabled is None else enabled
    if not display_enabled:
        return None
    display = settings.display
    return PygameViewer(
        ViewerSettings(
            screen_size=(display.screen_width, display.screen_height),
            margin_pixels=display.margin_pixels,
            point_radius_pixels=display.point_radius_pixels,
            frames_per_second=display.frames_per_second,
            show_clearance_circles=display.show_clearance_circles,
            minimum_distance=settings.solver.target_clearance,
            zoom_factor=display.zoom_factor,
            minimum_zoom=display.minimum_zoom,
            maximum_zoom=display.maximum_zoom,
            show_help=display.show_help,
            show_outer_boundary=display.show_outer_boundary,
        )
    )


def export_state(
    settings: AppSettings,
    state: SRN2State,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Export visual and standalone descriptions of one numerical state.

    SVG and reports are deliberately peripheral to the optimizer: they rebuild
    the exact assembly from the supplied free variables and never mutate the
    state or participate in acceptance decisions.
    """

    normalized_state = state.with_normalized_theta()
    assembly = SRN2Parameterization().build(normalized_state)
    report_metadata = {
        "exact_shell_thickness": exact_shell_thickness(assembly),
        **(metadata or {}),
    }
    if settings.export.write_svg:
        outer_boundary = assembly.shell.outer_boundary if assembly.shell is not None else None
        save_piece_assembly_svg(
            assembly.piece_contours,
            settings.paths.svg_output,
            outer_boundary=outer_boundary,
        )
    if settings.export.write_stl:
        # STL remains the extruded central tile, as in previous revisions.
        save_extruded_stl(
            assembly.main_contour,
            settings.paths.stl_output,
            settings.export.stl_thickness,
        )
    if settings.export.write_solution_report:
        save_standalone_solution_script(
            normalized_state,
            settings.paths.solution_script_output,
            metadata=report_metadata,
        )
        save_solution_json(
            normalized_state,
            assembly,
            settings.paths.solution_json_output,
            metadata=report_metadata,
        )


def _shell_thickness_caption(evaluation: ProblemEvaluation) -> str:
    thickness = evaluation.breakdown.get("exact_shell_thickness")
    return "" if thickness is None else f" thickness={thickness:.6f}"


def display_state(settings: AppSettings, state: SRN2State) -> None:
    viewer = _create_viewer(settings, enabled=True)
    assert viewer is not None
    problem = build_problem(settings, state.layout)
    evaluation = problem.evaluate(state.to_vector())
    caption = (
        f"Objective: {evaluation.objective:.6f}"
        f"{_shell_thickness_caption(evaluation)}"
    )
    assembly = evaluation.assembly
    outer_boundary = assembly.shell.outer_boundary if assembly.shell is not None else None
    viewer.draw(
        assembly.piece_contours,
        caption=caption,
        force=True,
        outer_boundary=outer_boundary,
    )
    viewer.wait_until_closed(
        assembly.piece_contours,
        caption,
        outer_boundary=outer_boundary,
    )


def _put_latest(target: queue.Queue[_ViewerUpdate], update: _ViewerUpdate) -> None:
    try:
        target.put_nowait(update)
        return
    except queue.Full:
        pass
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    target.put_nowait(update)


def optimize(settings: AppSettings, display: bool | None = None) -> RunSummary:
    initial_state = load_initial_state(settings)
    problem = build_problem(settings, initial_state.layout)
    initial_vector = initial_state.to_vector()
    initial_evaluation = problem.evaluate(initial_vector)
    viewer = _create_viewer(settings, enabled=display)
    solver = create_solver(settings.solver, coordinate_bound=settings.geometry.coordinate_bound)
    viewer_updates: queue.Queue[_ViewerUpdate] = queue.Queue(maxsize=2)

    def iteration_callback(iteration: SolverIteration) -> None:
        evaluation = problem.evaluate(iteration.vector)
        if settings.logging.print_contour_points:
            print(evaluation.assembly.main_contour)
        if settings.logging.autosave_iterations:
            iteration_metadata = {
                "objective": evaluation.objective,
                **evaluation.breakdown,
                **iteration.metadata,
            }
            autosave_state(
                settings.paths.output_directory,
                evaluation.state,
                iteration.index,
                metadata=iteration_metadata,
            )
            # Stable path for external inspection while a long run is active.
            save_state(
                settings.paths.latest_accepted_state,
                evaluation.state,
                metadata={"accepted_iteration": iteration.index, **iteration_metadata},
            )
        if viewer is not None:
            _put_latest(
                viewer_updates,
                _ViewerUpdate(
                    evaluation=evaluation,
                    caption=(
                        f"Accepted {iteration.index}: {evaluation.objective:.6f}"
                        f"{_shell_thickness_caption(evaluation)} {iteration.message}"
                    ),
                ),
            )
        print(
            f"Iteration {iteration.index:04d} objective={evaluation.objective:.12g}"
            f"{_shell_thickness_caption(evaluation)} {iteration.message}"
        )

    if viewer is None:
        result = solver.solve(problem, initial_vector, callback=iteration_callback)
    else:
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def solver_worker() -> None:
            try:
                result_queue.put(("result", solver.solve(problem, initial_vector, callback=iteration_callback)))
            except BaseException as error:  # Propagate worker failures to the main thread.
                result_queue.put(("error", error))

        worker = threading.Thread(target=solver_worker, name="voderberg-solver", daemon=True)
        worker.start()
        current_evaluation = initial_evaluation
        current_caption = (
            f"Searching... initial objective {initial_evaluation.objective:.6f}"
            f"{_shell_thickness_caption(initial_evaluation)}"
        )
        current_assembly = current_evaluation.assembly
        current_outer_boundary = (
            current_assembly.shell.outer_boundary if current_assembly.shell is not None else None
        )
        viewer.draw(
            current_assembly.piece_contours,
            caption=current_caption,
            force=True,
            outer_boundary=current_outer_boundary,
        )

        while worker.is_alive():
            new_geometry = False
            while True:
                try:
                    update = viewer_updates.get_nowait()
                except queue.Empty:
                    break
                current_evaluation = update.evaluation
                current_caption = update.caption
                new_geometry = True
            if not viewer.closed:
                current_assembly = current_evaluation.assembly
                current_outer_boundary = (
                    current_assembly.shell.outer_boundary
                    if current_assembly.shell is not None
                    else None
                )
                viewer.frame(
                    current_assembly.piece_contours,
                    caption=current_caption,
                    new_geometry=new_geometry,
                    outer_boundary=current_outer_boundary,
                )
            else:
                # Closing the viewer does not discard a long optimization run.
                time.sleep(0.03)

        worker.join()
        kind, payload = result_queue.get()
        if kind == "error":
            raise payload
        result = payload

    assert isinstance(result, SolverResult)
    final_evaluation = problem.evaluate(result.vector)
    save_state(
        settings.paths.optimized_state,
        final_evaluation.state,
        metadata={
            "objective": final_evaluation.objective,
            **final_evaluation.breakdown,
            **result.metadata,
        },
    )
    export_state(
        settings,
        final_evaluation.state,
        metadata={
            "objective": final_evaluation.objective,
            **final_evaluation.breakdown,
            **result.metadata,
        },
    )

    if viewer is not None and not viewer.closed:
        final_caption = (
            f"Finished: objective {final_evaluation.objective:.6f}"
            f"{_shell_thickness_caption(final_evaluation)} - {result.message}"
        )
        final_assembly = final_evaluation.assembly
        final_outer_boundary = (
            final_assembly.shell.outer_boundary if final_assembly.shell is not None else None
        )
        viewer.draw(
            final_assembly.piece_contours,
            caption=final_caption,
            force=True,
            outer_boundary=final_outer_boundary,
        )
        viewer.wait_until_closed(
            final_assembly.piece_contours,
            final_caption,
            outer_boundary=final_outer_boundary,
        )

    return RunSummary(
        initial_objective=initial_evaluation.objective,
        final_objective=final_evaluation.objective,
        initial_shell_thickness=initial_evaluation.breakdown.get("exact_shell_thickness"),
        final_shell_thickness=final_evaluation.breakdown.get("exact_shell_thickness"),
        success=result.success,
        message=result.message,
        final_state=final_evaluation.state,
    )

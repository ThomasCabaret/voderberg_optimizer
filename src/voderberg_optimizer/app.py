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
from .exporters import save_extruded_stl, save_svg
from .objective_factory import build_objective
from .parameterization import SRN2Parameterization
from .persistence import autosave_state, load_state, load_state_auto, save_state
from .problem import OptimizationProblem, ProblemEvaluation
from .refinement import refine_state
from .solvers import create_solver
from .solvers.base import SolverIteration, SolverResult
from .state import SRN2State, StateLayout
from .visualization import PygameViewer, ViewerSettings


@dataclass(frozen=True)
class RunSummary:
    initial_objective: float
    final_objective: float
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
        )
    )


def export_state(settings: AppSettings, state: SRN2State) -> None:
    assembly = SRN2Parameterization().build(state.with_normalized_theta())
    if settings.export.write_svg:
        save_svg(assembly.main_contour, settings.paths.svg_output)
    if settings.export.write_stl:
        save_extruded_stl(assembly.main_contour, settings.paths.stl_output, settings.export.stl_thickness)


def display_state(settings: AppSettings, state: SRN2State) -> None:
    viewer = _create_viewer(settings, enabled=True)
    assert viewer is not None
    problem = build_problem(settings, state.layout)
    evaluation = problem.evaluate(state.to_vector())
    caption = f"Objective: {evaluation.objective:.6f}"
    viewer.draw(evaluation.assembly.contours, caption=caption, force=True)
    viewer.wait_until_closed(evaluation.assembly.contours, caption)


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
            autosave_state(
                settings.paths.output_directory,
                evaluation.state,
                iteration.index,
                metadata={"objective": evaluation.objective, **iteration.metadata},
            )
        if viewer is not None:
            _put_latest(
                viewer_updates,
                _ViewerUpdate(
                    evaluation=evaluation,
                    caption=(
                        f"Accepted {iteration.index}: {evaluation.objective:.6f} "
                        f"{iteration.message}"
                    ),
                ),
            )
        print(
            f"Iteration {iteration.index:04d} objective={evaluation.objective:.12g} "
            f"{iteration.message}"
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
        current_caption = f"Searching... initial objective {initial_evaluation.objective:.6f}"
        viewer.draw(current_evaluation.assembly.contours, caption=current_caption, force=True)

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
                viewer.frame(
                    current_evaluation.assembly.contours,
                    caption=current_caption,
                    new_geometry=new_geometry,
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
        metadata={"objective": final_evaluation.objective, **result.metadata},
    )
    export_state(settings, final_evaluation.state)

    if viewer is not None and not viewer.closed:
        final_caption = f"Finished: objective {final_evaluation.objective:.6f} - {result.message}"
        viewer.draw(final_evaluation.assembly.contours, caption=final_caption, force=True)
        viewer.wait_until_closed(final_evaluation.assembly.contours, final_caption)

    return RunSummary(
        initial_objective=initial_evaluation.objective,
        final_objective=final_evaluation.objective,
        success=result.success,
        message=result.message,
        final_state=final_evaluation.state,
    )

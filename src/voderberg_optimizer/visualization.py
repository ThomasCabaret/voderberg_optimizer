"""Responsive Pygame visualization independent from the numerical model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BACKGROUND_COLOR = (30, 30, 30)
PIECE_FILL_COLORS = (
    (74, 126, 187),   # reference tile
    (214, 139, 72),   # left surrounding copy
    (91, 164, 112),   # right surrounding copy
)
PIECE_OUTLINE_COLOR = (18, 18, 18)
OUTER_BOUNDARY_COLOR = (245, 245, 245)
HELP_TEXT_COLOR = (235, 235, 235)


@dataclass(frozen=True)
class ViewerSettings:
    screen_size: tuple[int, int]
    margin_pixels: int
    point_radius_pixels: int
    frames_per_second: int
    show_clearance_circles: bool
    minimum_distance: float
    zoom_factor: float = 1.15
    minimum_zoom: float = 0.02
    maximum_zoom: float = 200.0
    show_help: bool = True
    show_outer_boundary: bool = True


class PygameViewer:
    """Interactive read-only viewer for the reconstructed three-piece assembly.

    Controls:
      - mouse wheel or +/-: zoom around cursor / window center
      - left, middle, or right drag: pan
      - F or R: fit all displayed pieces
      - O: toggle the reconstructed external shell boundary
      - H: toggle help
      - Escape, Q, or window close: close only the viewer

    The legacy point and clearance-circle settings remain in ``ViewerSettings``
    for configuration compatibility, but vertices are deliberately not drawn.
    """

    def __init__(self, settings: ViewerSettings) -> None:
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("Visualization requires pygame.") from error
        self.pygame = pygame
        self.settings = settings
        self.closed = False
        self._dragging = False
        self._view_initialized = False
        self._view_dirty = True
        self._show_help = settings.show_help
        self._show_outer_boundary = settings.show_outer_boundary
        self._scale = 1.0
        self._fit_scale = 1.0
        self._offset_x = settings.screen_size[0] / 2.0
        self._offset_y = settings.screen_size[1] / 2.0
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(settings.screen_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)

    def _fit_view(self, contours: tuple[Any, ...]) -> None:
        all_points = np.vstack([np.asarray(contour, dtype=float) for contour in contours])
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        width, height = np.maximum(maximum - minimum, 1.0e-12)
        screen_width, screen_height = self.screen.get_size()
        available_width = max(1.0, screen_width - 2.0 * self.settings.margin_pixels)
        available_height = max(1.0, screen_height - 2.0 * self.settings.margin_pixels)
        self._scale = min(available_width / width, available_height / height)
        self._fit_scale = self._scale
        center = (minimum + maximum) / 2.0
        self._offset_x = screen_width / 2.0 - center[0] * self._scale
        self._offset_y = screen_height / 2.0 + center[1] * self._scale
        self._view_initialized = True
        self._view_dirty = True

    def _screen_point(self, point: Any) -> tuple[int, int]:
        return (
            int(point[0] * self._scale + self._offset_x),
            int(-point[1] * self._scale + self._offset_y),
        )

    def _zoom_at(self, pixel: tuple[int, int], factor: float) -> None:
        if factor <= 0.0:
            return
        previous_scale = self._scale
        new_scale = float(
            np.clip(
                previous_scale * factor,
                self._fit_scale * self.settings.minimum_zoom,
                self._fit_scale * self.settings.maximum_zoom,
            )
        )
        if abs(new_scale - previous_scale) <= 1.0e-15:
            return
        px, py = pixel
        world_x = (px - self._offset_x) / previous_scale
        world_y = -(py - self._offset_y) / previous_scale
        self._scale = new_scale
        self._offset_x = px - world_x * new_scale
        self._offset_y = py + world_y * new_scale
        self._view_dirty = True

    def process_events(self, contours: tuple[Any, ...]) -> bool:
        """Process all pending events and return True when a redraw is useful."""

        if self.closed:
            return False
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                break
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self._view_dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
                self._dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 2, 3):
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                self._offset_x += event.rel[0]
                self._offset_y += event.rel[1]
                self._view_dirty = True
            elif event.type == pygame.MOUSEWHEEL:
                factor = self.settings.zoom_factor ** event.y
                self._zoom_at(pygame.mouse.get_pos(), factor)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.close()
                    break
                if event.key in (pygame.K_f, pygame.K_r, pygame.K_HOME):
                    self._fit_view(contours)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._zoom_at(self._window_center(), self.settings.zoom_factor)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._zoom_at(self._window_center(), 1.0 / self.settings.zoom_factor)
                elif event.key == pygame.K_o:
                    self._show_outer_boundary = not self._show_outer_boundary
                    self._view_dirty = True
                elif event.key == pygame.K_h:
                    self._show_help = not self._show_help
                    self._view_dirty = True
        return self._view_dirty

    def _window_center(self) -> tuple[int, int]:
        width, height = self.screen.get_size()
        return width // 2, height // 2

    def draw(
        self,
        contours: tuple[Any, ...],
        caption: str = "Voderberg SRN2",
        force: bool = False,
        outer_boundary: Any | None = None,
    ) -> None:
        if self.closed:
            return
        if not self._view_initialized:
            self._fit_view(contours)
        if not force and not self._view_dirty:
            return

        pygame = self.pygame
        self.screen.fill(BACKGROUND_COLOR)
        for index, contour in enumerate(contours):
            screen_points = [self._screen_point(point) for point in contour]
            if len(screen_points) >= 3:
                fill = PIECE_FILL_COLORS[index % len(PIECE_FILL_COLORS)]
                pygame.draw.polygon(self.screen, fill, screen_points, width=0)
                pygame.draw.polygon(self.screen, PIECE_OUTLINE_COLOR, screen_points, width=2)
            elif len(screen_points) >= 2:
                pygame.draw.lines(self.screen, PIECE_OUTLINE_COLOR, False, screen_points, width=2)

        if self._show_outer_boundary and outer_boundary is not None:
            boundary_points = [self._screen_point(point) for point in outer_boundary]
            if len(boundary_points) >= 3:
                pygame.draw.polygon(
                    self.screen,
                    OUTER_BOUNDARY_COLOR,
                    boundary_points,
                    width=3,
                )

        if self._show_help:
            lines = (
                "Wheel / +/-: zoom   Drag: pan   F/R: fit",
                "O: outer boundary   H: help   Q/Esc: close viewer",
            )
            y = 8
            for line in lines:
                surface = self.font.render(line, True, HELP_TEXT_COLOR)
                self.screen.blit(surface, (8, y))
                y += surface.get_height() + 2

        pygame.display.set_caption(caption)
        pygame.display.flip()
        self._view_dirty = False

    def frame(
        self,
        contours: tuple[Any, ...],
        caption: str,
        new_geometry: bool = False,
        outer_boundary: Any | None = None,
    ) -> None:
        if self.closed:
            return
        redraw_requested = self.process_events(contours)
        if new_geometry:
            self._view_dirty = True
        self.draw(
            contours,
            caption=caption,
            force=redraw_requested or new_geometry,
            outer_boundary=outer_boundary,
        )
        self.clock.tick(self.settings.frames_per_second)

    def wait_until_closed(
        self,
        contours: tuple[Any, ...],
        caption: str,
        outer_boundary: Any | None = None,
    ) -> None:
        while not self.closed:
            self.frame(contours, caption, outer_boundary=outer_boundary)

    def wait_until_closed_or_keypress(self) -> None:
        """Compatibility helper; use wait_until_closed for interactive viewing."""

        while not self.closed:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.close()
                    break
                if event.type == self.pygame.KEYDOWN:
                    return
            self.clock.tick(self.settings.frames_per_second)

    def close(self) -> None:
        if not self.closed:
            self.pygame.quit()
        self.closed = True

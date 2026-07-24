"""Optional legacy image-based point acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import apply_homogeneous_transform, similarity_transform
from .state import SRN2State, StateLayout


def _sample_open_segment(
    start: np.ndarray,
    end: np.ndarray,
    count: int,
    include_start: bool,
    include_end: bool,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=float)
    if include_start and include_end:
        fractions = np.array([0.5]) if count == 1 else np.linspace(0.0, 1.0, count)
    elif include_start:
        fractions = np.arange(count, dtype=float) / count
    elif include_end:
        fractions = np.arange(1, count + 1, dtype=float) / count
    else:
        fractions = np.arange(1, count + 1, dtype=float) / (count + 1)
    return np.array([(1.0 - fraction) * start + fraction * end for fraction in fractions])


def state_from_legacy_keypoints(keypoints: Any, layout: StateLayout) -> SRN2State:
    """Convert the old N-X-P-A-Q-Y-B-S acquisition convention to a state."""

    points = np.asarray(keypoints, dtype=float)
    expected = layout.x_points + layout.y_points + 6
    if len(points) != expected:
        raise ValueError(f"Expected {expected} normalized keypoints, got {len(points)}.")

    cursor = 1
    x = points[cursor : cursor + layout.x_points]
    cursor += layout.x_points
    p_anchor = points[cursor]
    a = points[cursor + 1]
    q_anchor = points[cursor + 2]
    cursor += 3
    y = points[cursor : cursor + layout.y_points]
    cursor += layout.y_points
    b = points[cursor]

    p = _sample_open_segment(p_anchor, a, layout.p_points, include_start=True, include_end=False)
    q = _sample_open_segment(a, q_anchor, layout.q_points, include_start=False, include_end=True)
    b_mirror = -b
    reference = np.array((0.0, -2.0))
    vector = a - b_mirror
    theta = np.arctan2(vector[1], vector[0]) - np.arctan2(reference[1], reference[0])
    return SRN2State(theta=theta, x=x, p=p, q=q, y=y, b=b)


def acquire_normalized_keypoints(image_path: str | Path, screen_size: tuple[int, int], margin: int) -> np.ndarray:
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("Image acquisition requires the optional 'viewer' dependencies.") from error

    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    image = pygame.image.load(str(image_path)).convert_alpha()

    aspect = screen_size[0] / screen_size[1]
    true_height = 3.0
    true_width = true_height * aspect
    scale = min(
        (screen_size[0] - 2 * margin) / true_width,
        (screen_size[1] - 2 * margin) / true_height,
    )
    offset_x = screen_size[0] / 2.0
    offset_y = screen_size[1] / 2.0

    target_width = int(true_width * scale)
    target_height = int(true_height * scale)
    image_scale = min(target_width / image.get_width(), target_height / image.get_height())
    image = pygame.transform.smoothscale(
        image,
        (int(image.get_width() * image_scale), int(image.get_height() * image_scale)),
    )
    image.set_alpha(128)

    def screen_to_model(position: tuple[int, int]) -> np.ndarray:
        return np.array(((position[0] - offset_x) / scale, -(position[1] - offset_y) / scale))

    def model_to_screen(point: np.ndarray) -> tuple[int, int]:
        return int(point[0] * scale + offset_x), int(-point[1] * scale + offset_y)

    selected: list[np.ndarray] = []
    acquiring = True
    while acquiring:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise RuntimeError("Acquisition cancelled.")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                acquiring = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL:
                if selected:
                    selected.pop()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                selected.append(screen_to_model(event.pos))

        screen.fill((30, 30, 30))
        image_position = (
            int(offset_x - image.get_width() / 2),
            int(offset_y - image.get_height() / 2),
        )
        screen.blit(image, image_position)
        if len(selected) > 1:
            pygame.draw.lines(screen, (255, 255, 255), False, [model_to_screen(point) for point in selected], 2)
        for point in selected:
            pygame.draw.circle(screen, (255, 0, 0), model_to_screen(point), 3)
        pygame.display.flip()

    pygame.quit()
    if len(selected) < 2:
        raise ValueError("At least the north and south points are required.")
    transform = similarity_transform(
        selected[0], selected[-1], np.array((0.0, 1.0)), np.array((0.0, -1.0))
    )
    return apply_homogeneous_transform(transform, np.asarray(selected))

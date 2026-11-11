"""Interactive pygame viewers. Requires `pygame-ce` (or `pygame`)."""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .lensing2d import Lensing2D
from .render import compose, trace
from .scene import Camera, Scene

TIME_SCALE = 2000.0  # simulated seconds of Newtonian gravity per real second


def _pygame():
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The interactive viewer needs pygame: pip install pygame-ce") from exc
    return pygame


def _blit(pygame, screen, image: np.ndarray) -> None:
    surface = pygame.surfarray.make_surface(np.ascontiguousarray(np.transpose(image, (1, 0, 2))))
    screen.blit(surface, (0, 0))


def run_viewer(
    size=(800, 600),
    move_size=(160, 120),
    still_size=(320, 240),
    *,
    orbit: bool = False,
    max_frames: int | None = None,
    screenshot: str | None = None,
) -> None:
    """Orbit around the black hole. Left-drag: orbit, wheel: zoom, hold right button or G: gravity.

    While you drag or gravity runs the scene is traced at `move_size`; once things are
    still it is re-traced at `still_size`. Both are upscaled to the window.
    """
    pg = _pygame()
    pg.init()
    screen = pg.display.set_mode(size)
    font = pg.font.Font(None, 22)
    clock = pg.time.Clock()

    scene = Scene.default(orbit=orbit)
    camera = Camera()
    grid, lensing = True, True
    dragging = False
    dirty = True
    last_quality = None
    last_ms = 0.0
    frames = 0
    last_tick = time.perf_counter()

    running = True
    while running:
        now = time.perf_counter()
        dt, last_tick = now - last_tick, now
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                elif event.button == 3:
                    scene.gravity = True
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    dirty = True
                elif event.button == 3:
                    scene.gravity = False
                    dirty = True
            elif event.type == pg.MOUSEMOTION and dragging:
                camera.orbit(*event.rel)
                dirty = True
            elif event.type == pg.MOUSEWHEEL:
                camera.zoom(event.y)
                dirty = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_g:
                    scene.gravity = not scene.gravity
                elif event.key == pg.K_l:
                    lensing = not lensing
                elif event.key == pg.K_t:
                    grid = not grid
                elif event.key == pg.K_r:
                    scene, camera = Scene.default(orbit=orbit), Camera()
                elif event.key == pg.K_s:
                    Image.fromarray(np.transpose(pg.surfarray.array3d(screen), (1, 0, 2))).save("screenshot.png")
                dirty = True

        if scene.gravity:
            scene.advance(min(dt, 1.0) * TIME_SCALE)
            dirty = True

        moving = dragging or scene.gravity
        quality = move_size if moving else still_size
        if dirty or quality != last_quality:
            t0 = time.perf_counter()
            rgba = trace(scene, camera, *quality, lensing=lensing)
            _blit(pg, screen, compose(rgba, scene, camera, size, grid=grid))
            last_ms = (time.perf_counter() - t0) * 1000.0
            last_quality = quality
            dirty = False
            lines = (
                f"{quality[0]}x{quality[1]} trace, {last_ms:.0f} ms    drag: orbit    wheel: zoom    right button: gravity",
                f"G: gravity [{'on' if scene.gravity else 'off'}]   L: lensing [{'on' if lensing else 'off'}]   "
                f"T: grid [{'on' if grid else 'off'}]   R: reset   S: screenshot   Esc: quit",
            )
            for row, text in enumerate(lines):
                screen.blit(font.render(text, True, (200, 200, 200)), (8, 8 + 20 * row))
            pg.display.flip()
        pg.display.set_caption("Python Black Hole Simulator")
        clock.tick(60)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    if screenshot:
        Image.fromarray(np.transpose(pg.surfarray.array3d(screen), (1, 0, 2))).save(screenshot)
    pg.quit()


def run_lensing2d(size=(800, 600), *, max_frames: int | None = None, screenshot: str | None = None) -> None:
    """2D light bending. Click to fire a ray from the left edge at that height; Space: beam; C: clear."""
    pg = _pygame()
    pg.init()
    screen = pg.display.set_mode(size)
    pg.display.set_caption("Python Black Hole Simulator - 2D lensing")
    clock = pg.time.Clock()
    sim = Lensing2D()
    sim.launch_beam()
    frames = 0
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                sim.launch_beam()
            elif event.type == pg.KEYDOWN and event.key == pg.K_c:
                sim.clear()
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                y = (size[1] / 2.0 - event.pos[1]) / size[1] * 2.0 * sim.half_height
                sim.launch((-sim.half_width, y), (1.0, 0.0))
        sim.advance(6)
        _blit(pg, screen, sim.render(size))
        pg.display.flip()
        clock.tick(60)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False
    if screenshot:
        Image.fromarray(np.transpose(pg.surfarray.array3d(screen), (1, 0, 2))).save(screenshot)
    pg.quit()

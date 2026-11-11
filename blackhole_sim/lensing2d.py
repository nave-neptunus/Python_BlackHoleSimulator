"""2D light bending around a black hole (port of 2D_lensing.cpp).

Rays are launched in the plane and bent by the Schwarzschild geodesic equations.
Distances are in units of the Schwarzschild radius, matching the original view
(+-1e11 m by +-7.5e10 m is roughly +-7.9 r_s by +-5.9 r_s for Sgr A*).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .constants import SAGA_MASS, schwarzschild_radius
from .geodesic import rk4_step

HALF_WIDTH = 1e11 / schwarzschild_radius(SAGA_MASS)
HALF_HEIGHT = 7.5e10 / schwarzschild_radius(SAGA_MASS)
PHOTON_SPHERE = 1.5
SHADOW_RADIUS = 3.0 * math.sqrt(3.0) / 2.0  # critical impact parameter


class Lensing2D:
    def __init__(
        self,
        half_width: float = HALF_WIDTH,
        half_height: float = HALF_HEIGHT,
        step: float = 0.02,
        max_trail: int = 3000,
    ):
        self.half_width = half_width
        self.half_height = half_height
        self.step = step
        self.max_trail = max_trail
        self.state = np.empty((4, 0))
        self.alive = np.zeros(0, dtype=bool)
        self.trails: list[list[tuple[float, float]]] = []

    def launch(self, pos, direction) -> None:
        """Start a ray at `pos` (r_s units) travelling along `direction` (any length)."""
        x, y = float(pos[0]), float(pos[1])
        dx, dy = float(direction[0]), float(direction[1])
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm, dy / norm
        r = math.hypot(x, y)
        if r <= 1.0:
            raise ValueError("rays must start outside the event horizon")
        phi = math.atan2(y, x)
        cos_a = dx * math.cos(phi) + dy * math.sin(phi)  # along r-hat
        sin_a = -dx * math.sin(phi) + dy * math.cos(phi)  # along phi-hat (signed)
        f = 1.0 - 1.0 / r
        new = np.array([[r], [phi], [math.sqrt(f) * cos_a], [sin_a / r]])
        self.state = np.concatenate([self.state, new], axis=1)
        self.alive = np.append(self.alive, True)
        self.trails.append([(x, y)])

    def launch_beam(self, count: int = 15, spread: float = 4.5) -> None:
        """A parallel beam coming from the left edge, like starlight passing the hole."""
        for y in np.linspace(-spread, spread, count):
            self.launch((-self.half_width, float(y)), (1.0, 0.0))

    def clear(self) -> None:
        self.__init__(self.half_width, self.half_height, self.step, self.max_trail)

    def advance(self, steps: int = 1) -> None:
        far = 3.0 * max(self.half_width, self.half_height)
        for _ in range(steps):
            idx = np.flatnonzero(self.alive)
            if idx.size == 0:
                return
            y = self.state[:, idx]
            y = rk4_step(y, self.step * np.maximum(y[0], 1.0))
            self.state[:, idx] = y
            xs, ys = y[0] * np.cos(y[1]), y[0] * np.sin(y[1])
            for k, i in enumerate(idx):
                trail = self.trails[i]
                trail.append((float(xs[k]), float(ys[k])))
                if len(trail) > self.max_trail:
                    del trail[0]
            dead = (y[0] <= 1.0) | (y[0] > far)
            self.alive[idx[dead]] = False

    def render(self, size: tuple[int, int] = (800, 600)) -> np.ndarray:
        w, h = size
        img = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        sx = w / (2.0 * self.half_width)
        sy = h / (2.0 * self.half_height)

        def px(x, y):
            return (x * sx + w / 2.0, h / 2.0 - y * sy)

        def circle(radius, **kw):
            c = px(0, 0)
            draw.ellipse([c[0] - radius * sx, c[1] - radius * sy, c[0] + radius * sx, c[1] + radius * sy], **kw)

        circle(PHOTON_SPHERE, outline=(70, 20, 20))
        circle(1.0, fill=(255, 0, 0))  # the original draws the hole as a red disk
        for trail in self.trails:
            n = len(trail)
            if n < 2:
                continue
            for i in range(n - 1):
                level = int(255 * max(i / (n - 1), 0.05))
                draw.line([px(*trail[i]), px(*trail[i + 1])], fill=(level, level, level), width=1)
        for i, trail in enumerate(self.trails):
            if self.alive[i]:
                x, y = px(*trail[-1])
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 60, 60))
        return np.asarray(img)

    def deflection(self, index: int) -> float:
        """Total turning angle (radians) of a finished ray: angle between its start and end direction."""
        trail = self.trails[index]
        (x0, y0), (x1, y1) = trail[0], trail[1]
        (x2, y2), (x3, y3) = trail[-2], trail[-1]
        a0 = math.atan2(y1 - y0, x1 - x0)
        a1 = math.atan2(y3 - y2, x3 - x2)
        return (a1 - a0 + math.pi) % (2 * math.pi) - math.pi

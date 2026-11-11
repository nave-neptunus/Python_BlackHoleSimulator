"""Scene description: black hole, accretion disk, bodies, camera and spacetime grid.

Positions in a Scene are in metres (like the original C++ project). The
renderer converts to units of the Schwarzschild radius internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import G, SAGA_MASS, SOLAR_MASS, schwarzschild_radius


@dataclass
class Body:
    position: np.ndarray  # metres
    radius: float  # metres
    color: tuple[float, float, float]
    mass: float  # kg
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)


@dataclass
class Scene:
    bh_mass: float = SAGA_MASS
    disk_inner: float = 2.2  # in Schwarzschild radii; 0 disables the disk
    disk_outer: float = 5.2
    bodies: list[Body] = field(default_factory=list)
    gravity: bool = False

    @property
    def rs(self) -> float:
        return schwarzschild_radius(self.bh_mass)

    @classmethod
    def default(cls, orbit: bool = False) -> "Scene":
        """The original demo: Sgr A*, a flat accretion disk, a yellow and a red star."""
        scene = cls(
            bodies=[
                Body([4e11, 0, 0], 4e10, (1.0, 1.0, 0.0), SOLAR_MASS),
                Body([0, 0, 4e11], 4e10, (1.0, 0.0, 0.0), SOLAR_MASS),
            ]
        )
        if orbit:
            scene.give_circular_orbits()
        return scene

    def give_circular_orbits(self) -> None:
        for b in self.bodies:
            r = float(np.linalg.norm(b.position))
            speed = math.sqrt(G * self.bh_mass / r)
            tangent = np.cross([0.0, 1.0, 0.0], b.position / r)
            b.velocity = tangent * speed

    def outer_radius(self) -> float:
        """Radius (in r_s) beyond which an outgoing ray can no longer hit anything."""
        rs = self.rs
        r = self.disk_outer
        for b in self.bodies:
            r = max(r, (np.linalg.norm(b.position) + b.radius) / rs)
        return r

    # ---- Newtonian dynamics -------------------------------------------------
    def _accelerations(self) -> np.ndarray:
        n = len(self.bodies)
        acc = np.zeros((n, 3))
        for i, bi in enumerate(self.bodies):
            d = -bi.position  # towards the (fixed) black hole at the origin
            acc[i] += G * self.bh_mass * d / np.linalg.norm(d) ** 3
            for j, bj in enumerate(self.bodies):
                if i != j:
                    d = bj.position - bi.position
                    acc[i] += G * bj.mass * d / np.linalg.norm(d) ** 3
        return acc

    def advance(self, dt: float, accuracy: float = 0.01, max_substeps: int = 200_000) -> None:
        """Leapfrog-integrate `dt` seconds of Newtonian gravity.

        The black hole is held fixed at the origin; bodies that fall inside the
        Schwarzschild radius are swallowed (removed). The substep is `accuracy`
        times the shortest local dynamical time sqrt(r^3 / GM), so a body plunging
        towards the hole is resolved instead of being flung away by the 1/r^2 force.
        """
        if not self.gravity:
            return
        rs = self.rs
        remaining = dt
        for _ in range(max_substeps):
            self.bodies = [b for b in self.bodies if np.linalg.norm(b.position) > rs]
            if not self.bodies or remaining <= 0.0:
                return
            r_min = min(float(np.linalg.norm(b.position)) for b in self.bodies)
            h = min(remaining, accuracy * math.sqrt(r_min**3 / (G * self.bh_mass)))
            acc = self._accelerations()
            for b, a in zip(self.bodies, acc):
                b.velocity += 0.5 * h * a
                b.position += h * b.velocity
            acc = self._accelerations()
            for b, a in zip(self.bodies, acc):
                b.velocity += 0.5 * h * a
            remaining -= h

    # ---- Spacetime grid -----------------------------------------------------
    def grid_vertices(self, n: int = 25, spacing: float = 1e10, offset: float = 3e10) -> np.ndarray:
        """(n+1, n+1, 3) vertices of a grid in the x-z plane bent by Flamm's paraboloid.

        Each mass adds 2*sqrt(r_s_i * (d - r_s_i)) of depth at horizontal distance d,
        which is the embedding of the Schwarzschild spatial geometry.
        """
        idx = (np.arange(n + 1) - n // 2) * spacing
        X, Z = np.meshgrid(idx, idx)
        Y = np.full_like(X, -offset)
        masses = [(np.zeros(3), self.bh_mass)] + [(b.position, b.mass) for b in self.bodies]
        for pos, mass in masses:
            rs_i = schwarzschild_radius(mass)
            d = np.hypot(X - pos[0], Z - pos[2])
            Y += 2.0 * np.sqrt(rs_i * np.maximum(d - rs_i, 0.0))
        return np.stack([X, Y, Z], axis=-1)


@dataclass
class Camera:
    """Orbit camera looking at the black hole (the origin)."""

    azimuth: float = 0.0
    elevation: float = math.radians(80.0)  # polar angle from +y; 90 deg = edge-on to the disk
    radius: float = 1.6e11  # metres (about 12.6 r_s). The original starts at 6.34194e10, inside the disk's footprint.
    fov_deg: float = 60.0
    min_radius: float = 1e10
    max_radius: float = 1e12

    def position(self) -> np.ndarray:
        e = min(max(self.elevation, 0.01), math.pi - 0.01)
        return np.array(
            [
                self.radius * math.sin(e) * math.cos(self.azimuth),
                self.radius * math.cos(e),
                self.radius * math.sin(e) * math.sin(self.azimuth),
            ]
        )

    def basis(self):
        """(position, right, up, forward) with y as world-up."""
        pos = self.position()
        fwd = -pos / np.linalg.norm(pos)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return pos, right, up, fwd

    def orbit(self, dx: float, dy: float, speed: float = 0.01) -> None:
        self.azimuth += dx * speed
        self.elevation = min(max(self.elevation - dy * speed, 0.01), math.pi - 0.01)

    def zoom(self, scroll: float, speed: float = 25e9) -> None:
        self.radius = min(max(self.radius - scroll * speed, self.min_radius), self.max_radius)

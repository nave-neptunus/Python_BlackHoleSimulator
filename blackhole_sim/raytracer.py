"""Classic sphere ray tracer with diffuse light and hard shadows (port of ray_tracing.cpp)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sphere:
    centre: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]


def default_spheres() -> list[Sphere]:
    return [
        Sphere((0.0, 0.0, -5.0), 2.0, (1.0, 0.2, 0.2)),
        Sphere((3.0, 0.0, -7.0), 1.5, (0.2, 1.0, 0.2)),
    ]


def _intersect(origin: np.ndarray, dirs: np.ndarray, sphere: Sphere) -> np.ndarray:
    """Nearest positive ray parameter for unit-length `dirs` (inf where there is no hit)."""
    oc = origin - np.asarray(sphere.centre)
    b = 2.0 * np.einsum("ij,ij->i", oc, dirs)
    c = np.einsum("ij,ij->i", oc, oc) - sphere.radius**2
    disc = b * b - 4.0 * c
    ok = disc >= 0.0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    t = np.where(t1 >= 0.0, t1, t2)
    return np.where(ok & (t >= 0.0), t, np.inf)


def render_spheres(
    width: int = 800,
    height: int = 600,
    spheres: list[Sphere] | None = None,
    light: tuple[float, float, float] = (5.0, 5.0, 5.0),
) -> np.ndarray:
    spheres = default_spheres() if spheres is None else spheres
    aspect = width / height
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    dirs = np.stack([(2 * uu - 1) * aspect, -(2 * vv - 1), -np.ones_like(uu)], axis=-1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origin = np.zeros_like(dirs)

    closest = np.full(dirs.shape[0], np.inf)
    which = np.full(dirs.shape[0], -1)
    for k, s in enumerate(spheres):
        t = _intersect(origin, dirs, s)
        better = t < closest
        closest = np.where(better, t, closest)
        which = np.where(better, k, which)

    image = np.tile(np.array([0.0, 0.0, 0.1]), (dirs.shape[0], 1))
    light = np.asarray(light, dtype=float)
    for k, s in enumerate(spheres):
        sel = np.flatnonzero(which == k)
        if sel.size == 0:
            continue
        hit = dirs[sel] * closest[sel, None]
        normal = hit - np.asarray(s.centre)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        to_light = light - hit
        to_light /= np.linalg.norm(to_light, axis=1, keepdims=True)
        diffuse = np.maximum(np.einsum("ij,ij->i", normal, to_light), 0.0)

        shadow_origin = hit + normal * 0.001
        shadowed = np.zeros(sel.size, dtype=bool)
        for other in spheres:
            shadowed |= np.isfinite(_intersect(shadow_origin, to_light, other))
        brightness = np.where(shadowed, 0.1, 0.1 + 0.9 * diffuse)
        image[sel] = np.asarray(s.color)[None, :] * brightness[:, None]

    return (np.clip(image, 0.0, 1.0).reshape(height, width, 3) * 255).astype(np.uint8)

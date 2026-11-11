"""CPU ray tracer for a Schwarzschild black hole with an accretion disk and bodies.

All rays are integrated together with NumPy. Rays that hit something (or escape)
are dropped from the working set, so the cost falls quickly as the march goes on.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .geodesic import initial_state, rk4_step, to_cartesian
from .scene import Camera, Scene


def camera_rays(camera: Camera, width: int, height: int):
    """Observer position (metres) and (H*W, 3) unit ray directions."""
    pos, right, up, fwd = camera.basis()
    aspect = width / height
    t = math.tan(math.radians(camera.fov_deg) / 2.0)
    xs = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * aspect * t
    ys = (1.0 - 2.0 * (np.arange(height) + 0.5) / height) * t
    u, v = np.meshgrid(xs, ys)  # row 0 is the top of the image
    d = u[..., None] * right + v[..., None] * up + fwd
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return pos, d.reshape(-1, 3)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _segment_sphere(p0: np.ndarray, d: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Parameter t in [0, 1] where the segment p0 + t*d first enters the sphere (inf if it doesn't)."""
    f = p0 - center
    a = _dot(d, d)
    b = 2.0 * _dot(f, d)
    c = _dot(f, f) - radius * radius
    disc = b * b - 4.0 * a * c
    good = (disc >= 0.0) & (a > 1e-30)
    sq = np.sqrt(np.where(good, disc, 0.0))
    t1 = np.where(good, (-b - sq) / (2.0 * np.where(good, a, 1.0)), np.inf)
    inside = c <= 0.0
    hit = good & (((t1 >= 0.0) & (t1 <= 1.0)) | inside)
    return np.where(hit, np.where(inside, 0.0, t1), np.inf)


def trace(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    *,
    lensing: bool = True,
    step: float = 0.04,
    max_steps: int = 4000,
) -> np.ndarray:
    """Trace the scene and return a float32 (H, W, 4) RGBA image in [0, 1].

    Alpha follows the original shader: the background is transparent (alpha 0), the
    black-hole shadow and bodies are opaque, and the disk's alpha grows with radius.

    lensing: curve the rays with Schwarzschild geodesics (True) or use straight
             lines (False) to see what the same scene looks like without gravity.
    step:    RK4 step as a fraction of the current radius (smaller = more accurate).
    """
    rs = scene.rs
    cam_m, dirs = camera_rays(camera, width, height)
    cam = cam_m / rs
    n_total = dirs.shape[0]
    out = np.zeros((n_total, 4), dtype=np.float32)

    r1, r2 = scene.disk_inner, scene.disk_outer
    has_disk = r2 > r1 > 0.0
    centers = [b.position / rs for b in scene.bodies]
    radii = [b.radius / rs for b in scene.bodies]
    colors = [np.asarray(b.color, dtype=float) for b in scene.bodies]
    r_out = scene.outer_radius() * 1.02

    idx = np.arange(n_total)
    if np.linalg.norm(cam) <= 1.0:
        raise ValueError("camera must be outside the event horizon")
    if lensing:
        y, e1, e2 = initial_state(cam, dirs)
        prev = to_cartesian(y, e1, e2)
    else:
        y = e1 = e2 = None
        prev = np.broadcast_to(cam, dirs.shape).copy()

    for _ in range(max_steps):
        if idx.size == 0:
            break
        n = idx.size
        r = y[0] if lensing else np.linalg.norm(prev, axis=1)
        h = step * np.maximum(r, 1.0)
        if lensing:
            y = rk4_step(y, h)
            new = to_cartesian(y, e1, e2)
        else:
            new = prev + dirs * h[:, None]
        d = new - prev

        best_t = np.full(n, np.inf)
        kind = np.zeros(n, dtype=np.int8)  # 0 none, 1 black hole, 2 disk, 3+k body k
        disk_rr = np.zeros(n)

        if has_disk:
            y0, y1 = prev[:, 1], new[:, 1]
            crossed = (y0 * y1 < 0.0) & (np.abs(y0) > 1e-9)  # a ray starting in the plane is not a crossing
            t = np.where(crossed, y0 / np.where(crossed, y0 - y1, 1.0), np.inf)
            rr = np.hypot(prev[:, 0] + t * d[:, 0], prev[:, 2] + t * d[:, 2])
            ok = crossed & (rr >= r1) & (rr <= r2)
            best_t = np.where(ok, t, best_t)
            kind[ok] = 2
            disk_rr = rr

        for k, (c, rad) in enumerate(zip(centers, radii)):
            t = _segment_sphere(prev, d, c, rad)
            better = t < best_t
            best_t = np.where(better, t, best_t)
            kind[better] = 3 + k

        if lensing:
            kind[(y[0] <= 1.0) & (kind == 0)] = 1
        else:
            t = _segment_sphere(prev, d, np.zeros(3), 1.0)
            kind[(t < best_t) & np.isfinite(t)] = 1

        escaped = (kind == 0) & (np.linalg.norm(new, axis=1) > r_out) & (_dot(new, d) > 0.0)

        done = (kind != 0) | escaped
        if done.any():
            sel = np.flatnonzero(done & (kind != 0))
            if sel.size:
                px = idx[sel]
                kd = kind[sel]
                bh = kd == 1
                out[px[bh]] = (0.0, 0.0, 0.0, 1.0)
                dk = kd == 2
                if dk.any():
                    s = np.clip(disk_rr[sel][dk] / r2, 0.0, 1.0).astype(np.float32)
                    out[px[dk], 0] = 1.0
                    out[px[dk], 1] = s
                    out[px[dk], 2] = 0.2
                    out[px[dk], 3] = s
                for k in range(len(centers)):
                    ok = kd == 3 + k
                    if not ok.any():
                        continue
                    s_i = sel[ok]
                    hit = prev[s_i] + best_t[s_i, None] * d[s_i]
                    normal = hit - centers[k]
                    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
                    view = cam - hit
                    view /= np.linalg.norm(view, axis=1, keepdims=True)
                    shade = 0.1 + 0.9 * np.maximum(_dot(normal, view), 0.0)
                    out[px[ok], :3] = (colors[k][None, :] * shade[:, None]).astype(np.float32)
                    out[px[ok], 3] = 1.0
            keep = ~done
            idx = idx[keep]
            new = new[keep]
            if lensing:
                y = y[:, keep]
                e1 = e1[keep]
                e2 = e2[keep]
            else:
                dirs = dirs[keep]
        prev = new
    return out.reshape(height, width, 4)


def project(points: np.ndarray, camera: Camera, width: int, height: int):
    """Project world points (metres) to pixel coordinates; `valid` marks points in front of the camera."""
    pos, right, up, fwd = camera.basis()
    rel = points - pos
    xc, yc, zc = rel @ right, rel @ up, rel @ fwd
    t = math.tan(math.radians(camera.fov_deg) / 2.0)
    valid = zc > 1e6
    z = np.where(valid, zc, 1.0)
    sx = (xc / (z * (width / height) * t) * 0.5 + 0.5) * width
    sy = (0.5 - yc / (z * t) * 0.5) * height
    return sx, sy, valid


def grid_layer(scene: Scene, camera: Camera, width: int, height: int) -> np.ndarray:
    """Spacetime grid drawn on black, as float RGB in [0, 1] with shape (H, W, 3)."""
    verts = scene.grid_vertices()
    n1 = verts.shape[0]
    sx, sy, valid = project(verts.reshape(-1, 3), camera, width, height)
    sx, sy, valid = sx.reshape(n1, n1), sy.reshape(n1, n1), valid.reshape(n1, n1)

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    colour = (128, 128, 128, 179)  # the original grid fragment shader: (0.5, 0.5, 0.5, 0.7)

    def polyline(xs, ys, ok):
        for i in range(len(xs) - 1):
            if ok[i] and ok[i + 1]:
                draw.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill=colour, width=1)

    for i in range(n1):
        polyline(sx[i], sy[i], valid[i])  # one grid row
        polyline(sx[:, i], sy[:, i], valid[:, i])  # one grid column
    black = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    flat = Image.alpha_composite(black, layer).convert("RGB")
    return np.asarray(flat, dtype=np.float32) / 255.0


def _resize(channel: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(channel, mode="F").resize(size, Image.BILINEAR))


def compose(
    rgba: np.ndarray,
    scene: Scene,
    camera: Camera,
    size: tuple[int, int] | None = None,
    *,
    grid: bool = True,
) -> np.ndarray:
    """Upscale the traced RGBA to `size` and composite it over the spacetime grid -> uint8 RGB."""
    h, w = rgba.shape[:2]
    width, height = size if size else (w, h)
    alpha = np.ascontiguousarray(rgba[..., 3], dtype=np.float32)
    premult = [np.ascontiguousarray(rgba[..., i] * rgba[..., 3], dtype=np.float32) for i in range(3)]
    if (width, height) != (w, h):
        alpha = _resize(alpha, (width, height))
        premult = [_resize(p, (width, height)) for p in premult]
    bg = grid_layer(scene, camera, width, height) if grid else np.zeros((height, width, 3), np.float32)
    rgb = np.stack(premult, axis=-1) + bg * (1.0 - alpha)[..., None]
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def render_image(
    scene: Scene,
    camera: Camera,
    size: tuple[int, int] = (800, 600),
    render_size: tuple[int, int] | None = None,
    *,
    grid: bool = True,
    lensing: bool = True,
    step: float = 0.04,
    max_steps: int = 4000,
) -> np.ndarray:
    """Trace at `render_size` (default: `size`), then upscale/compose to `size` as uint8 RGB."""
    rw, rh = render_size if render_size else size
    rgba = trace(scene, camera, rw, rh, lensing=lensing, step=step, max_steps=max_steps)
    return compose(rgba, scene, camera, size, grid=grid)

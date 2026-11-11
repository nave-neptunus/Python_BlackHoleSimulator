"""Null geodesics in Schwarzschild spacetime, vectorised over many rays.

Everything here uses units where the Schwarzschild radius r_s = 1 and G = c = 1.

Schwarzschild is spherically symmetric, so every light ray stays in the plane
spanned by its starting position and its starting direction. Inside that plane
a ray is described by polar coordinates (r, phi) and the geodesic equations
reduce to

    r''   = (r - 3/2) * phi'^2
    phi'' = -2 r' phi' / r

(primes are derivatives with respect to the affine parameter). This is exact,
needs no tracking of the polar angle theta (so there is no pole singularity),
and is regular through the event horizon.
"""

from __future__ import annotations

import numpy as np

# State layout: y has shape (4, N) = (r, phi, dr, dphi) for N rays.


def rhs(y: np.ndarray) -> np.ndarray:
    r, _, dr, dphi = y
    out = np.empty_like(y)
    out[0] = dr
    out[1] = dphi
    out[2] = (r - 1.5) * dphi * dphi
    out[3] = -2.0 * dr * dphi / r
    return out


def rk4_step(y: np.ndarray, h: np.ndarray | float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step. `h` may be per-ray."""
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * h * k1)
    k3 = rhs(y + 0.5 * h * k2)
    k4 = rhs(y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def initial_state(pos: np.ndarray, dirs: np.ndarray):
    """Build the in-plane state for rays leaving a static observer.

    pos:  (3,) observer position (units of r_s), outside the horizon.
    dirs: (N, 3) unit direction vectors in the observer's local frame.

    Returns (y, e1, e2): state (4, N) and the two in-plane unit vectors so that
    the 3D position is  r * (cos(phi) * e1 + sin(phi) * e2).
    """
    pos = np.asarray(pos, dtype=float)
    r0 = float(np.linalg.norm(pos))
    if r0 <= 1.0:
        raise ValueError("observer must be outside the event horizon (r > r_s)")
    e1 = pos / r0
    dr = dirs @ e1  # cos(alpha), alpha = angle to the outward radial direction
    perp = dirs - dr[:, None] * e1
    s = np.linalg.norm(perp, axis=1)  # sin(alpha)

    # Radial rays have no preferred plane: pick any vector perpendicular to e1.
    helper = np.array([0.0, 1.0, 0.0]) if abs(e1[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    fallback = np.cross(e1, helper)
    fallback /= np.linalg.norm(fallback)
    safe = s > 1e-12
    e2 = np.where(safe[:, None], perp / np.where(safe, s, 1.0)[:, None], fallback)

    # A static observer measures local direction (cos a, sin a). In coordinate
    # terms:  dr/dlam = sqrt(f) cos a,   r dphi/dlam = sin a   (f = 1 - 1/r).
    f = 1.0 - 1.0 / r0
    y = np.empty((4, dirs.shape[0]))
    y[0] = r0
    y[1] = 0.0
    y[2] = np.sqrt(f) * dr
    y[3] = s / r0
    return y, np.broadcast_to(e1, e2.shape), e2


def to_cartesian(y: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    r, phi = y[0], y[1]
    return r[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)


def conserved(y: np.ndarray):
    """(E^2, L): E^2 = dr^2 + f L^2 / r^2 and L = r^2 dphi are constants of motion."""
    r, _, dr, dphi = y
    L = r * r * dphi
    E2 = dr * dr + (1.0 - 1.0 / r) * L * L / (r * r)
    return E2, L

"""The geodesic integrator against known results of Schwarzschild optics (r_s = 1 units)."""

import math

import numpy as np
import pytest

from blackhole_sim.geodesic import conserved, initial_state, rk4_step, to_cartesian

B_CRIT = 3.0 * math.sqrt(3.0) / 2.0  # critical impact parameter, in r_s


def start_ray(b: float, r0: float):
    """A ray from a static observer at radius r0, aimed inward with impact parameter b = L/E."""
    f = 1.0 - 1.0 / r0
    sin_a = b * math.sqrt(f) / r0
    direction = np.array([[-math.sqrt(1.0 - sin_a**2), sin_a, 0.0]])
    return initial_state(np.array([r0, 0.0, 0.0]), direction)


def integrate(y, step, max_steps, stop):
    for i in range(max_steps):
        y_new = rk4_step(y, step * np.maximum(y[0], 1.0))
        if stop(y_new):
            return y_new, y, i
        y = y_new
    return y, y, max_steps


def test_initial_state_is_null_and_matches_impact_parameter():
    y, _, _ = start_ray(b=7.0, r0=40.0)
    E2, L = conserved(y)
    assert math.isclose(L[0] / math.sqrt(E2[0]), 7.0, rel_tol=1e-12)


def test_conserved_quantities_along_a_ray():
    y, _, _ = start_ray(b=4.0, r0=30.0)
    E2_0, L_0 = conserved(y)
    for _ in range(600):
        y = rk4_step(y, 0.02 * np.maximum(y[0], 1.0))
    E2, L = conserved(y)
    assert abs(E2[0] / E2_0[0] - 1.0) < 1e-6
    assert abs(L[0] / L_0[0] - 1.0) < 1e-6


def deflection(b: float, r0: float = 1e5, step: float = 0.02) -> float:
    y, _, _ = start_ray(b, r0)
    passed_periapsis = {"yes": False}

    def stop(state):
        if state[2, 0] > 0:
            passed_periapsis["yes"] = True
        return passed_periapsis["yes"] and state[0, 0] >= r0

    prev, after, _ = integrate(y, step, 200_000, stop)
    # `integrate` returns (state just past r0, state before it, steps); interpolate to r = r0.
    past, before = prev, after
    w = (r0 - before[0, 0]) / (past[0, 0] - before[0, 0])
    phi = before[1, 0] + w * (past[1, 0] - before[1, 0])
    return phi - (math.pi - 2.0 * math.asin(b / r0))


@pytest.mark.parametrize("b", [50.0, 100.0])
def test_weak_field_deflection(b):
    expected = 2.0 / b + (15.0 * math.pi / 16.0) / b**2  # 4GM/(c^2 b) + (15 pi/4)(GM/(c^2 b))^2
    assert deflection(b) == pytest.approx(expected, rel=0.01)


def test_photon_sphere_capture_threshold():
    captured_y, _, _ = start_ray(B_CRIT - 0.05, 100.0)
    for _ in range(20_000):
        captured_y = rk4_step(captured_y, 0.02 * np.maximum(captured_y[0], 1.0))
        if captured_y[0, 0] <= 1.0 or captured_y[0, 0] > 100.0:
            break
    assert captured_y[0, 0] <= 1.0, "ray just inside the critical impact parameter should fall in"

    escaping_y, _, _ = start_ray(B_CRIT + 0.05, 100.0)
    for _ in range(20_000):
        escaping_y = rk4_step(escaping_y, 0.02 * np.maximum(escaping_y[0], 1.0))
        if escaping_y[0, 0] <= 1.0 or escaping_y[0, 0] > 100.0:
            break
    assert escaping_y[0, 0] > 100.0, "ray just outside the critical impact parameter should escape"


def test_radial_ray_stays_radial():
    y, e1, e2 = initial_state(np.array([10.0, 0.0, 0.0]), np.array([[-1.0, 0.0, 0.0]]))
    for _ in range(50):
        y = rk4_step(y, 0.02 * np.maximum(y[0], 1.0))
    pos = to_cartesian(y, e1, e2)[0]
    assert abs(pos[1]) < 1e-12 and abs(pos[2]) < 1e-12 and pos[0] > 0


def test_observer_inside_horizon_is_rejected():
    with pytest.raises(ValueError):
        initial_state(np.array([0.5, 0.0, 0.0]), np.array([[-1.0, 0.0, 0.0]]))

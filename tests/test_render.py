import math

import numpy as np
import pytest

from blackhole_sim import Camera, Scene, compose, render_image, schwarzschild_radius, trace
from blackhole_sim.constants import SAGA_MASS
from blackhole_sim.lensing2d import SHADOW_RADIUS, Lensing2D
from blackhole_sim.raytracer import Sphere, render_spheres

B_CRIT = 3.0 * math.sqrt(3.0) / 2.0


def test_sgr_a_schwarzschild_radius():
    assert schwarzschild_radius(SAGA_MASS) == pytest.approx(1.2684e10, rel=1e-3)


def empty_scene():
    return Scene(disk_inner=0.0, disk_outer=0.0)


def shadow_pixels(lensing: bool, distance: float = 50.0) -> tuple[int, float]:
    scene = empty_scene()
    camera = Camera(elevation=math.pi / 2, radius=distance * scene.rs, fov_deg=10.0)
    w, h = 120, 90
    rgba = trace(scene, camera, w, h, lensing=lensing)
    # expected pixel radius from the shadow's angular radius seen by a static observer
    f = 1.0 - 1.0 / distance
    theta = math.asin((B_CRIT if lensing else 1.0) * (math.sqrt(f) if lensing else 1.0) / distance)
    pixel_radius = math.tan(theta) / (math.tan(math.radians(10.0) / 2.0) / (h / 2.0))
    return int((rgba[..., 3] == 1.0).sum()), math.pi * pixel_radius**2


def test_shadow_size_matches_critical_impact_parameter():
    count, expected = shadow_pixels(lensing=True)
    assert count == pytest.approx(expected, rel=0.06)


def test_flat_space_shadow_is_the_geometric_horizon():
    count, expected = shadow_pixels(lensing=False)
    assert count == pytest.approx(expected, rel=0.08)


def test_lensing_enlarges_the_shadow():
    lensed, _ = shadow_pixels(lensing=True)
    flat, _ = shadow_pixels(lensing=False)
    assert lensed > 5 * flat


def test_default_scene_shows_disk_and_shadow():
    scene = Scene.default()
    camera = Camera()
    rgba = trace(scene, camera, 80, 60)
    assert rgba.shape == (60, 80, 4)
    alpha = rgba[..., 3]
    assert (alpha == 1.0).any(), "black hole shadow"
    disk = (rgba[..., 0] == 1.0) & (np.abs(rgba[..., 2] - 0.2) < 1e-6)
    assert disk.any(), "accretion disk"
    assert (alpha == 0.0).any(), "transparent background"


def test_camera_inside_disk_plane_does_not_fill_half_the_screen():
    scene = Scene.default()
    camera = Camera(elevation=math.pi / 2, radius=6.34194e10)  # original default, exactly in the disk plane
    rgba = trace(scene, camera, 40, 30)
    disk = (rgba[..., 0] == 1.0) & (np.abs(rgba[..., 2] - 0.2) < 1e-6)
    assert disk.mean() < 0.5


def test_body_is_visible_and_shaded():
    scene = Scene.default()
    scene.disk_inner = scene.disk_outer = 0.0
    scene.bodies = scene.bodies[:1]  # yellow star on +x
    camera = Camera(azimuth=math.radians(180), elevation=math.pi / 2, radius=1.5e11, fov_deg=40.0)
    rgba = trace(scene, camera, 80, 60)
    yellow = (rgba[..., 0] > 0.05) & (rgba[..., 1] > 0.05) & (rgba[..., 2] == 0.0) & (rgba[..., 3] == 1.0)
    assert yellow.any()
    assert np.allclose(rgba[yellow][:, 0], rgba[yellow][:, 1])
    assert rgba[yellow][:, 0].max() <= 1.0


def test_compose_and_render_image_shapes_and_grid():
    scene, camera = Scene.default(), Camera()
    rgba = trace(scene, camera, 40, 30)
    with_grid = compose(rgba, scene, camera, (160, 120), grid=True)
    without = compose(rgba, scene, camera, (160, 120), grid=False)
    assert with_grid.shape == without.shape == (120, 160, 3) and with_grid.dtype == np.uint8
    assert with_grid.sum() > without.sum()
    assert render_image(scene, camera, (64, 48), (32, 24)).shape == (48, 64, 3)


def test_flat_and_curved_agree_far_from_the_hole_looking_away():
    scene = empty_scene()
    camera = Camera(elevation=math.pi / 2, radius=200 * scene.rs, fov_deg=10.0)
    # pointing at the hole from far away the outer pixels see nothing in both modes
    a = trace(scene, camera, 20, 15, lensing=True)
    b = trace(scene, camera, 20, 15, lensing=False)
    assert np.array_equal(a[0], b[0])


def test_gravity_circular_orbit_closes():
    scene = Scene.default(orbit=True)
    scene.gravity = True
    start = scene.bodies[0].position.copy()
    r = float(np.linalg.norm(start))
    period = 2.0 * math.pi * math.sqrt(r**3 / (6.67430e-11 * scene.bh_mass))
    scene.advance(period)
    assert np.linalg.norm(scene.bodies[0].position - start) / r < 0.01


def test_falling_body_is_swallowed():
    scene = Scene.default()
    scene.gravity = True
    scene.advance(5e4)
    assert scene.bodies == []


def test_grid_dips_towards_the_black_hole():
    scene = Scene.default()
    v = scene.grid_vertices()
    centre = v[v.shape[0] // 2, v.shape[1] // 2, 1]
    edge = v[0, 0, 1]
    assert centre < edge


def test_2d_beam_bends_towards_the_hole_and_far_rays_barely_deflect():
    sim = Lensing2D()
    sim.launch_beam(count=2, spread=7.0)  # y = -7 and +7: far from the hole
    sim.advance(1500)
    for i in range(2):
        assert abs(sim.deflection(i)) == pytest.approx(2.0 / 7.0, rel=0.25)
    assert sim.deflection(0) > 0 > sim.deflection(1)  # lower ray bends up, upper ray bends down


def test_2d_ray_inside_critical_parameter_is_captured_outside_is_not():
    sim = Lensing2D()
    sim.launch((-7.0, SHADOW_RADIUS - 0.2), (1.0, 0.0))
    sim.launch((-7.0, SHADOW_RADIUS + 0.2), (1.0, 0.0))
    sim.advance(4000)
    assert not sim.alive[0] and math.hypot(*sim.trails[0][-1]) <= 1.0 + 1e-6
    assert math.hypot(*sim.trails[1][-1]) > 3.0


def test_raytracer_hits_spheres_and_background():
    img = render_spheres(80, 60)
    assert img.shape == (60, 80, 3)
    assert tuple(img[0, 0]) == (0, 0, 25)  # background (0, 0, 0.1)
    assert img[30, 40, 0] > img[30, 40, 1]  # centre is the red sphere
    shadowed = render_spheres(80, 60, [Sphere((0, 0, -5), 2.0, (1, 1, 1))], light=(0, 0, 5))
    assert shadowed[30, 40].min() > 200  # facing the light head-on

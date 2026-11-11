"""Command line interface: python -m blackhole_sim <command> --help"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
from PIL import Image

from .lensing2d import Lensing2D
from .raytracer import render_spheres
from .render import compose, render_image, trace
from .scene import Camera, Scene


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from exc


def _camera(scene: Scene, azimuth: float, elevation: float, distance: float) -> Camera:
    """Elevation is degrees above the disk plane (0 = edge-on, 90 = top-down); distance is in r_s."""
    return Camera(
        azimuth=math.radians(azimuth),
        elevation=math.radians(90.0 - elevation),
        radius=distance * scene.rs,
        max_radius=max(1e12, distance * scene.rs),
    )


def _add_camera_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--azimuth", type=float, default=0.0, help="camera azimuth in degrees (default 0)")
    p.add_argument("--elevation", type=float, default=10.0, help="degrees above the disk plane (default 10)")
    p.add_argument("--distance", type=float, default=12.6, help="camera distance in Schwarzschild radii (default 12.6)")
    p.add_argument("--no-grid", action="store_true", help="hide the spacetime grid")
    p.add_argument("--flat", action="store_true", help="straight rays, i.e. the same scene without gravitational lensing")
    p.add_argument("--no-disk", action="store_true", help="remove the accretion disk")
    p.add_argument("--no-bodies", action="store_true", help="remove the two stars")


def _scene(args) -> Scene:
    scene = Scene.default(orbit=getattr(args, "orbit", False))
    if args.no_disk:
        scene.disk_inner = scene.disk_outer = 0.0
    if args.no_bodies:
        scene.bodies = []
    return scene


def cmd_render(args) -> None:
    scene = _scene(args)
    camera = _camera(scene, args.azimuth, args.elevation, args.distance)
    t0 = time.perf_counter()
    image = render_image(
        scene,
        camera,
        args.size,
        args.render_size or args.size,
        grid=not args.no_grid,
        lensing=not args.flat,
        step=args.step,
    )
    Image.fromarray(image).save(args.out)
    print(f"wrote {args.out} ({time.perf_counter() - t0:.1f} s)")


def cmd_animate(args) -> None:
    scene = _scene(args)
    scene.gravity = args.mode == "gravity"
    frames = []
    t0 = time.perf_counter()
    for i in range(args.frames):
        if args.mode == "camera":
            azimuth = args.azimuth + 360.0 * i / args.frames
            camera = _camera(scene, azimuth, args.elevation, args.distance)
        else:
            camera = _camera(scene, args.azimuth, args.elevation, args.distance)
            if i:
                scene.advance(args.dt)
        rgba = trace(scene, camera, *(args.render_size or args.size), lensing=not args.flat, step=args.step)
        frames.append(Image.fromarray(compose(rgba, scene, camera, args.size, grid=not args.no_grid)))
        print(f"\rframe {i + 1}/{args.frames}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out} ({time.perf_counter() - t0:.1f} s)")


def cmd_view(args) -> None:
    from .viewer import run_viewer

    run_viewer(args.size, orbit=args.orbit)


def cmd_lensing2d(args) -> None:
    if args.interactive:
        from .viewer import run_lensing2d

        run_lensing2d(args.size)
        return
    sim = Lensing2D()
    sim.launch_beam(args.rays)
    sim.advance(args.steps)
    Image.fromarray(sim.render(args.size)).save(args.out)
    print(f"wrote {args.out}")


def cmd_raytrace(args) -> None:
    Image.fromarray(render_spheres(*args.size)).save(args.out)
    print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="blackhole_sim", description="Python port of kavan010/black_hole.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("render", help="render a still image of the 3D scene")
    _add_camera_args(p)
    p.add_argument("-o", "--out", default="blackhole.png")
    p.add_argument("--size", type=_size, default=(800, 600), help="output size, e.g. 800x600")
    p.add_argument("--render-size", type=_size, default=None, help="trace resolution; upscaled to --size (default: same as --size)")
    p.add_argument("--step", type=float, default=0.04, help="integrator step as a fraction of radius (smaller = slower, more accurate)")
    p.add_argument("--orbit", action="store_true", help="give the stars circular-orbit velocities")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("animate", help="render an animated GIF")
    _add_camera_args(p)
    p.add_argument("-o", "--out", default="blackhole.gif")
    p.add_argument("--mode", choices=["camera", "gravity"], default="camera", help="orbit the camera, or let the stars fall")
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--dt", type=float, default=1500.0, help="simulated seconds per frame in gravity mode")
    p.add_argument("--size", type=_size, default=(480, 360))
    p.add_argument("--render-size", type=_size, default=(240, 180))
    p.add_argument("--step", type=float, default=0.04)
    p.add_argument("--orbit", action="store_true")
    p.set_defaults(func=cmd_animate)

    p = sub.add_parser("view", help="interactive 3D viewer (needs pygame-ce)")
    p.add_argument("--size", type=_size, default=(800, 600))
    p.add_argument("--orbit", action="store_true", help="give the stars circular-orbit velocities")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("lensing2d", help="2D light bending demo")
    p.add_argument("-o", "--out", default="lensing2d.png")
    p.add_argument("--size", type=_size, default=(800, 600))
    p.add_argument("--rays", type=int, default=15)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--interactive", action="store_true", help="open a window (needs pygame-ce)")
    p.set_defaults(func=cmd_lensing2d)

    p = sub.add_parser("raytrace", help="plain sphere ray tracer (port of ray_tracing.cpp)")
    p.add_argument("-o", "--out", default="raytrace.png")
    p.add_argument("--size", type=_size, default=(800, 600))
    p.set_defaults(func=cmd_raytrace)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

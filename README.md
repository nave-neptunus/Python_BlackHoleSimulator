# Python Black Hole Simulator

A ray-traced Schwarzschild black hole in pure Python and NumPy: light bends around the hole, the accretion
disk wraps over and under the shadow, and stars orbit and fall in. It runs on any CPU with no GPU, OpenGL or
compiler.

![A black hole with accretion disk, warped spacetime grid and two stars](docs/hero.png)

Based on Kavan's C++/OpenGL project [kavan010/black_hole](https://github.com/kavan010/black_hole), rewritten
from scratch in Python with corrected light-bending physics.

## Try it

```bash
git clone https://github.com/nave-neptunus/Python_BlackHoleSimulator.git
cd Python_BlackHoleSimulator
pip install -e ".[viewer]"
python -m blackhole_sim view
```

Drag to orbit, scroll to zoom, press `L` to switch gravity's effect on light off and on.

## What you're looking at

![Same scene with straight light rays (left) and with lensing (right)](docs/lensing_vs_flat.png)

Same disk, same camera. On the left, light travels in straight lines. On the right, it follows the curved
paths general relativity predicts: the far side of the disk is bent up over the hole and down under it, and
the shadow grows to about 2.6 times the horizon radius.

| Camera orbit | Stars under gravity |
|:---:|:---:|
| ![Camera orbiting the black hole](docs/orbit.gif) | ![Stars orbiting and falling in](docs/gravity.gif) |

![Light rays bending around a black hole in 2D](docs/lensing2d.png)

*2D view: a beam of light rays passing the hole. Rays aimed within 2.6 r_s of the center are captured.*

## How it works

Light travels along null geodesics of the Schwarzschild metric. Spacetime around a non-rotating hole is
spherically symmetric, so each ray stays in a single plane through the hole. The ray only needs two
coordinates, radius `r` and angle `φ`. With the Schwarzschild radius set to 1:

```
r'' = (r − 3/2) φ'²
φ'' = −2 r' φ' / r
```

Every pixel's ray is advanced at once with vectorised fourth-order Runge–Kutta (RK4) steps. The step size
grows with distance (`h = 0.04 r`). A ray leaves the working set as soon as it falls in, hits the disk or a
star, or escapes past the scene. Each step's segment is tested against the disk and spheres, so a large step
can't jump over an object.

## Fixes over the original

Porting meant reading the original GLSL shader line by line. These are the problems that turned up and what
this version does instead:

- **Light bent too little.** The shader's radial equation uses `r` where the Schwarzschild geodesic needs
  `r − r_s`, and the photon energy is normalised inconsistently. Integrated numerically, rays bend about a
  third as much as general relativity predicts. This version matches the weak-field deflection to within
  0.1% at 60 r_s (0.03% at 100 r_s) and gives the expected shadow radius of 3√3/2 r_s.
- **"RK4" was Euler.** The shader's `rk4Step` function takes a single first-order Euler step. This version
  uses real RK4.
- **No pole singularity.** The original tracks the polar angle θ and divides by sin θ, which blows up near
  the poles. Integrating in each ray's own plane removes θ entirely.
- **Speed.** An adaptive step and early exit replace the original 60,000 fixed steps per ray. The viewer
  traces at low resolution while you move and sharpens when you stop.
- **Gravity.** Stars move with leapfrog integration on a timestep that adapts to how close they are, and
  anything that crosses the horizon is removed. The original adds acceleration once per frame with no
  timestep.
- **Camera.** Rays are converted to what a stationary observer at the camera's radius would see. The default
  camera starts at 12.6 r_s, 10° above the disk; the original starts inside the disk.

New here: a `--flat` mode and `L` key to compare with and without lensing, GIF export, and click-to-fire rays
in the 2D demo.

## Tested against known physics

```bash
pip install -e ".[dev]"
pytest
```

22 tests check the integrator and renderer against textbook results:

- energy and angular momentum stay constant along a ray (to 1 part in a million)
- weak-field deflection `2 r_s/b + (15π/16)(r_s/b)²`
- rays inside the photon-sphere impact parameter `b = 3√3/2 r_s` are captured and rays outside escape
- the rendered shadow is the right size, with and without lensing
- a circular orbit closes and a falling star is swallowed

## All commands

```bash
python -m blackhole_sim view                       # interactive 3D viewer
python -m blackhole_sim render -o hero.png --azimuth 225 --elevation 25 --distance 16
python -m blackhole_sim render --flat -o flat.png  # same scene, straight rays
python -m blackhole_sim animate -o orbit.gif       # camera orbit
python -m blackhole_sim animate --mode gravity --orbit -o gravity.gif --distance 70 --elevation 60
python -m blackhole_sim lensing2d --interactive    # 2D light bending, click to fire rays
python -m blackhole_sim raytrace -o spheres.png    # plain sphere ray tracer
```

`--elevation` is degrees above the disk (0 is edge-on, 90 is top-down). `--distance` is in Schwarzschild
radii. Add `--help` to any command for every option.

| Key | Action |
|---|---|
| Left-drag | Orbit the camera |
| Mouse wheel | Zoom |
| Right mouse (hold) or `G` | Turn on gravity for the stars |
| `L` | Lensing on/off |
| `T` | Spacetime grid on/off |
| `R` / `S` / `Esc` | Reset / save screenshot / quit |

It renders on the CPU, so each sharp frame takes a moment rather than running at 60 fps.

Use it as a library:

```python
from PIL import Image
from blackhole_sim import Scene, Camera, render_image

image = render_image(Scene.default(), Camera(), size=(800, 600), render_size=(400, 300))
Image.fromarray(image).save("blackhole.png")
```

Needs Python 3.10 or newer (tested on 3.13). `.[viewer]` adds `pygame-ce` for the windows. Still images and
GIFs only need `pip install -e .` (NumPy and Pillow).

## Limits

This is a visualisation, not a research tool. The hole doesn't spin, the disk is a flat ring coloured by
radius with no redshift or beaming, there is no background sky, and the stars follow Newtonian gravity.

## Code layout

```
blackhole_sim/
  geodesic.py   Schwarzschild light paths (vectorised RK4)
  scene.py      scene, camera, Newtonian gravity, spacetime grid
  render.py     3D ray tracer and compositing
  lensing2d.py  2D light bending
  raytracer.py  sphere ray tracer
  viewer.py     pygame windows
  cli.py        command line
tests/          physics and rendering tests
docs/           images in this README
```

## Credit and license

The scene, controls and visual design come from [kavan010/black_hole](https://github.com/kavan010/black_hole).
That project has no license file, so this one doesn't declare one either, and rights in the original design
stay with its author.

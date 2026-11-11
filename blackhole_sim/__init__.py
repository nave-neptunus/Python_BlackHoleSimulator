"""Python Black Hole Simulator: a NumPy port of kavan010/black_hole."""

from .constants import C, G, SAGA_MASS, SOLAR_MASS, schwarzschild_radius
from .render import compose, render_image, trace
from .scene import Body, Camera, Scene

__all__ = [
    "Body",
    "C",
    "Camera",
    "G",
    "SAGA_MASS",
    "SOLAR_MASS",
    "Scene",
    "compose",
    "render_image",
    "schwarzschild_radius",
    "trace",
]
__version__ = "0.1.0"

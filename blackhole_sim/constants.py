"""Physical constants (SI) and helpers."""

C = 299_792_458.0  # speed of light, m/s
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
SAGA_MASS = 8.54e36  # Sagittarius A*, kg (value used by the original project)
SOLAR_MASS = 1.98892e30  # kg


def schwarzschild_radius(mass: float) -> float:
    """r_s = 2GM/c^2, in metres."""
    return 2.0 * G * mass / (C * C)

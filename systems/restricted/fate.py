"""Fate of each launch (pixel) in the restricted three-body problem.

The pixel functions take the integrator as `integrate`: fate_fast (C++, fast.cpp; the default) or
reference.fate_scipy (the scipy reference). Both take a state [x, y, vx, vy] and t_max, and return
(0 bound at t_max / 1 escaped / 2, 3 crashed into primary 1, 2;  time the run stopped;  |Jacobi drift|).
"""
import math
import numpy as np
from systems.restricted import _fast   # compiled from fast.cpp: run `make` after editing it
from systems.restricted.model import MU, R_COLL, launch

FORBIDDEN = 4   # fate index for launch points that can't exist at the chosen C


def fate_fast(y0, t_max):
    """Integrate one launch in C++ (fast.cpp: Boost.Odeint fehlberg78, rtol 1e-10, atol 1e-12)."""
    return _fast.fate(*y0, t_max)


def _inside_primary(x, y):
    """Index of the primary (x, y) is inside, else None. Checked before anything evaluates the potential there."""
    return next((p for p, px in enumerate((-MU, 1 - MU)) if math.hypot(x - px, y) <= R_COLL), None)


def pixel_fixed_c(x, y, C, t_max, integrate=fate_fast):
    """Launch at (x, y) with Jacobi constant C, moving perpendicular to the position vector."""
    p = _inside_primary(x, y)
    if p is not None:
        return 2 + p, 0.0, 0.0
    y0 = launch(x, y, C)
    if y0 is None:
        return FORBIDDEN, np.nan, 0.0
    return integrate(y0, t_max)


def pixel_rest(x, y, t_max, integrate=fate_fast):
    """Release at rest in the rotating frame (co-rotating with the primaries); each pixel has its own C = 2*omega(x, y)."""
    p = _inside_primary(x, y)
    if p is not None:
        return 2 + p, 0.0, 0.0
    return integrate(np.array([x, y, 0.0, 0.0]), t_max)

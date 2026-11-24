"""Fate of each launch (pixel) in the restricted three-body problem.

The pixel functions take the integrator as `integrate`: fate_fast (C++, fast.cpp; the default) or
reference.fate_scipy (the scipy reference). Both take a state [x, y, vx, vy] and t_max, and return
(0 bound at t_max / 1 escaped / 2, 3 crashed into primary 1, 2;  time the run stopped;  |Jacobi drift|).
"""
import math
import numpy as np
try:
    import systems.restricted._fast as _fast   # compiled from fast.cpp: run `make` after editing it
except ModuleNotFoundError:   # not built (e.g. no compiler); a broken build still raises ImportError
    _fast = None
from lib.sweep import grid_sweep
from systems.restricted.model import MU, R_COLL, launch

T_MAX = 100.0
FORBIDDEN = 4   # fate index for launch points that can't exist at the chosen C
FATES = [f"bound at t={T_MAX:g}", "escapes", "crashes into primary 1", "crashes into primary 2", "forbidden at this C"]
COLORS = ["#d9d8d3", "#2a78d6", "#eb6834", "#1baf7a", "#383835"]
SWAP = np.array([0, 1, 3, 2, 4])   # fate under the 180-degree rotation: crash into primary 1 <-> primary 2


def fate_fast(y0, t_max):
    """Integrate one launch in C++ (fast.cpp: Boost.Odeint fehlberg78, rtol 1e-10, atol 1e-12)."""
    if _fast is None:
        raise ImportError("C++ integrator not built: run `make`, or pass --scipy to use the scipy reference")
    return _fast.fate(*y0, t_max)


def _inside_primary(x, y):
    """Index of the primary (x, y) is inside, else None. Checked before anything evaluates the potential there."""
    return next((p for p, px in enumerate((-MU, 1 - MU)) if math.hypot(x - px, y) <= R_COLL), None)


def pixel_fixed_c(x, y, C, t_max, integrate=fate_fast, alpha=math.pi / 2):
    """Launch at (x, y) with Jacobi constant C, heading at angle alpha from the position vector."""
    p = _inside_primary(x, y)
    if p is not None:
        return 2 + p, 0.0, 0.0
    y0 = launch(x, y, C, alpha)
    if y0 is None:
        return FORBIDDEN, np.nan, 0.0
    return integrate(y0, t_max)


def pixel_rest(x, y, t_max, integrate=fate_fast):
    """Release at rest in the rotating frame (co-rotating with the primaries); each pixel has its own C = 2*omega(x, y)."""
    p = _inside_primary(x, y)
    if p is not None:
        return 2 + p, 0.0, 0.0
    return integrate(np.array([x, y, 0.0, 0.0]), t_max)


def mirror(half, n, swap=None):
    """Full n-row grid from rows 0..n//2 via the 180-degree symmetry: row r, col c of the top is row n-1-r, col n-1-c."""
    top = half[: n // 2][::-1, ::-1]
    return np.vstack([half, top if swap is None else swap[top]])


def symmetric_sweep(fn, xs, ys, cache, fields=("fates", "times", "drifts")):
    """Sweep the bottom half (plus the y = 0 row) of a grid symmetric about the origin; return the mirrored full maps."""
    assert MU == 0.5 and len(ys) % 2 == 1, "the half-grid symmetry needs equal primaries and an odd row count (y = 0 row)"
    half = grid_sweep(fn, xs, ys[: len(ys) // 2 + 1], cache, fields)
    return tuple(mirror(half[k], len(ys), SWAP if k == fields[0] else None) for k in fields)

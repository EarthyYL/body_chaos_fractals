"""Circular restricted three-body problem, in the frame rotating with the two primaries.

Units: G = 1, m1 + m2 = 1, primary separation 1, so the frame rotates at angular velocity 1.
Primary 1 (mass 1 - MU) sits at (-MU, 0), primary 2 (mass MU) at (1 - MU, 0); neither moves in this frame.
The test particle is massless: it feels the primaries but doesn't pull on them.
Rotating -> inertial at time t: rotate position and velocity by angle t, after V_inertial = (vx - y, vy + x).
The equations of motion and the fate events are implemented twice: fast.cpp (used) and reference.py (scipy, for checking).
"""
import math
import numpy as np

MU = 0.5          # m2 / (m1 + m2); 0.5 = the Copenhagen problem (equal primaries)
R_COLL = 1e-3     # physical radius of each primary: closer than this is a crash
R_ESC = 10.0      # escaped: this far from the centre of mass AND unbound (see escape_gap in fast.cpp / reference.py)


def omega(x: float, y: float) -> float:
    """Effective potential: centrifugal + gravity. Jacobi constant is C = 2*omega - v^2."""
    return (x * x + y * y) / 2 + (1 - MU) / math.hypot(x + MU, y) + MU / math.hypot(x - 1 + MU, y)


def launch(x: float, y: float, C: float) -> np.ndarray | None:
    """State [x, y, vx, vy] of a particle at (x, y) with Jacobi constant C, moving perpendicular to its position vector
    (counter-clockwise). None if (x, y) is forbidden at this C (it would need negative kinetic energy)."""
    v2 = 2 * omega(x, y) - C
    if v2 < 0:
        return None
    v, th = math.sqrt(v2), math.atan2(y, x)
    return np.array([x, y, -v * math.sin(th), v * math.cos(th)])

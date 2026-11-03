"""Circular restricted three-body problem, in the frame rotating with the two primaries.

Units: G = 1, m1 + m2 = 1, primary separation 1, so the frame rotates at angular velocity 1.
Primary 1 (mass 1 - MU) sits at (-MU, 0), primary 2 (mass MU) at (1 - MU, 0); neither moves in this frame.
The test particle is massless: it feels the primaries but doesn't pull on them.
Rotating -> inertial at time t: rotate position and velocity by angle t, after V_inertial = (vx - y, vy + x).
"""
import math
import numpy as np
from dataclasses import dataclass

MU = 0.5          # m2 / (m1 + m2); 0.5 = the Copenhagen problem (equal primaries)
R_COLL = 1e-3     # physical radius of each primary: closer than this is a crash
R_ESC = 10.0      # escaped: this far from the centre of mass AND unbound (see escape_gap)


def omega(x: float, y: float) -> float:
    """Effective potential: centrifugal + gravity. Jacobi constant is C = 2*omega - v^2."""
    return (x * x + y * y) / 2 + (1 - MU) / math.hypot(x + MU, y) + MU / math.hypot(x - 1 + MU, y)


@dataclass
class RotState:
    y: np.ndarray   # [x, y, vx, vy] in the rotating frame

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.float64)

    def derivative(self) -> np.ndarray:
        x, y, vx, vy = self.y.tolist()
        r1, r2 = math.hypot(x + MU, y), math.hypot(x - 1 + MU, y)
        g1, g2 = (1 - MU) / r1**3, MU / r2**3
        ax = 2 * vy + x - g1 * (x + MU) - g2 * (x - 1 + MU)   # Coriolis + centrifugal + gravity
        ay = -2 * vx + y - g1 * y - g2 * y
        return np.array([vx, vy, ax, ay])

    def jacobi(self) -> float:
        x, y, vx, vy = self.y.tolist()
        return 2 * omega(x, y) - (vx * vx + vy * vy)

    def crash_gap(self, primary: int) -> float:
        """Crosses 0 downward when the particle hits primary 0 or 1."""
        x, y = self.y[0], self.y[1]
        px = -MU if primary == 0 else 1 - MU
        return math.hypot(x - px, y) - R_COLL

    def escape_gap(self) -> float:
        """Crosses 0 upward when the particle is past R_ESC and unbound from the primaries (seen as one mass at the origin)."""
        x, y, vx, vy = self.y.tolist()
        r = math.hypot(x, y)
        if r < R_ESC:   # can't have escaped yet; also avoids 1/r at the origin
            return r - R_ESC
        energy = 0.5 * ((vx - y) ** 2 + (vy + x) ** 2) - 1 / r   # inertial-frame energy per unit mass
        return min(r - R_ESC, energy)


def launch(x: float, y: float, C: float) -> RotState | None:
    """Particle at (x, y) with Jacobi constant C, moving perpendicular to its position vector (counter-clockwise).
    None if (x, y) is forbidden at this C (it would need negative kinetic energy)."""
    v2 = 2 * omega(x, y) - C
    if v2 < 0:
        return None
    v, th = math.sqrt(v2), math.atan2(y, x)
    return RotState(np.array([x, y, -v * math.sin(th), v * math.cos(th)]))

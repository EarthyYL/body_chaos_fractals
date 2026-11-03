import math
import numpy as np
from dataclasses import dataclass
from config import G, V, M

defaultPrecision = np.float64
PAIRS = ((0, 1), (0, 2), (1, 2))
R_COLL = 1e-5
R_ESC = 20

@dataclass
class State:
    y: np.ndarray   # [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3]

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=defaultPrecision)   # no copy if already a float64 array

    @property
    def pos(self) -> np.ndarray:
        return self.y.reshape(3, 4)[:, :2]   # [body, (x, y)], a view into y

    @property
    def vel(self) -> np.ndarray:
        return self.y.reshape(3, 4)[:, 2:]   # [body, (vx, vy)], a view into y

    def derivative(self) -> np.ndarray:
        # plain Python floats instead of numpy: ~10x faster on arrays this small (3 vs 33 us/call)
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = self.y.tolist()
        m1, m2, m3 = M
        dx12, dy12 = x1 - x2, y1 - y2
        dx13, dy13 = x1 - x3, y1 - y3
        dx23, dy23 = x2 - x3, y2 - y3
        r12, r13, r23 = math.hypot(dx12, dy12), math.hypot(dx13, dy13), math.hypot(dx23, dy23)
        g12, g13, g23 = G(r12) / r12, G(r13) / r13, G(r23) / r23   # force on i from j is M[i]*M[j]*g*(r_i - r_j)
        # same layout as y: [vx, vy, ax, ay] per body
        return np.array([
            vx1, vy1, m2 * g12 * dx12 + m3 * g13 * dx13, m2 * g12 * dy12 + m3 * g13 * dy13,
            vx2, vy2, m3 * g23 * dx23 - m1 * g12 * dx12, m3 * g23 * dy23 - m1 * g12 * dy12,
            vx3, vy3, -m1 * g13 * dx13 - m2 * g23 * dx23, -m1 * g13 * dy13 - m2 * g23 * dy23,
        ])

    def smallest_gap(self) -> float:
        """Goes to 0 when collision happens"""
        pos = self.pos
        return min(np.linalg.norm(pos[i, :] - pos[j, :]) for i, j in PAIRS) - R_COLL

    def body_escaping(self, body_index) -> float:
        """Crosses 0 upward when the body is far from the other two AND unbound from them."""
        i = body_index
        j, k = (n for n in range(3) if n != i)
        mjk = M[j] + M[k]
        pos, vel = self.pos, self.vel
        d = pos[i] - (M[j] * pos[j] + M[k] * pos[k]) / mjk   # relative to the pair's centre of mass
        v = vel[i] - (M[j] * vel[j] + M[k] * vel[k]) / mjk
        r = np.linalg.norm(d)
        mu = M[i] * mjk / (M[i] + mjk)                       # reduced mass
        energy = 0.5 * mu * (v @ v) + M[i] * mjk * V(r)
        return min(r - R_ESC, energy)       

"""Python reference for fast.cpp: the same equations of motion, events and Jacobi constant, integrated with scipy's
solve_ivp (DOP853). About 200x slower than the C++ path; kept to test it (tests/test_restricted.py) and to reproduce the
caches made before it (the ones without an integrator tag in their names).
"""
import math
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from systems.restricted.model import MU, R_COLL, R_ESC, omega


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


def rhs(t, y):
    return RotState(y).derivative()


def escape(t, y):
    return RotState(y).escape_gap()
escape.terminal, escape.direction = True, 1  # type: ignore[reportFunctionMemberAccess]


def make_crash(p):
    def event(t, y):
        return RotState(y).crash_gap(p)
    event.terminal, event.direction = True, -1  # type: ignore[reportFunctionMemberAccess]
    return event


EVENTS = [escape, make_crash(0), make_crash(1)]


def fate_scipy(y0, t_max):
    """Same contract as fate.fate_fast: (0 bound at t_max / 1 escaped / 2, 3 crashed into primary 1, 2;
    time the run stopped;  |Jacobi drift|)."""
    s = RotState(y0)
    c0 = s.jacobi()
    sol = solve_ivp(rhs, (0, t_max), s.y, method="DOP853", rtol=1e-10, atol=1e-12, events=EVENTS)
    hit = [n for n, te in enumerate(sol.t_events) if te.size]
    return (hit[0] + 1 if hit else 0), sol.t[-1], abs(RotState(sol.y[:, -1]).jacobi() - c0)

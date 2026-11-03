import numpy as np
from scipy.integrate import solve_ivp
from systems.restricted.model import RotState, launch

FORBIDDEN = 4   # fate index for launch points that can't exist at the chosen C


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


def _inside_primary(x, y):
    """Index of the primary (x, y) is inside, else None. Checked before anything evaluates the potential there."""
    s = RotState(np.array([x, y, 0.0, 0.0]))
    return next((p for p in (0, 1) if s.crash_gap(p) <= 0), None)


def fate(s, t_max):
    """Returns (0 bound at t_max / 1 escaped / 2, 3 crashed into primary 1, 2;  time the run stopped;  Jacobi drift)."""
    c0 = s.jacobi()
    sol = solve_ivp(rhs, (0, t_max), s.y, method="DOP853", rtol=1e-10, atol=1e-12, events=EVENTS)
    hit = [n for n, te in enumerate(sol.t_events) if te.size]
    return (hit[0] + 1 if hit else 0), sol.t[-1], abs(RotState(sol.y[:, -1]).jacobi() - c0)


def pixel_fixed_c(x, y, C, t_max):
    """Launch at (x, y) with Jacobi constant C, moving perpendicular to the position vector."""
    p = _inside_primary(x, y)
    if p is not None:
        return 2 + p, 0.0, 0.0
    s = launch(x, y, C)
    if s is None:
        return FORBIDDEN, np.nan, 0.0
    return fate(s, t_max)


def pixel_rest(x, y, t_max):
    """Release at rest in the rotating frame (co-rotating with the primaries); each pixel has its own C = 2*omega(x, y)."""
    p = _inside_primary(x, y)
    if p is not None:
        return 2 + p, 0.0, 0.0
    return fate(RotState(np.array([x, y, 0.0, 0.0])), t_max)

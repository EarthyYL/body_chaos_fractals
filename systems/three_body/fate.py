from scipy.integrate import solve_ivp
from systems.three_body.state import State


def rhs(t, y):
    return State(y).derivative()


def make_escape(i):
    def event(t, y):
        return State(y).body_escaping(i)
    event.terminal, event.direction = True, 1  # type: ignore[reportFunctionMemberAccess]
    return event


# no collision event: softening (config.EPS) lets bodies pass through close encounters
EVENTS = [make_escape(0), make_escape(1), make_escape(2)]


def fate(y0, t_max):
    """Returns (0 = still bound at t_max / 1-3 = that body escaped, time the run stopped)."""
    sol = solve_ivp(rhs, (0, t_max), y0, method="DOP853", rtol=1e-10, atol=1e-12, events=EVENTS)
    hit = [n for n, te in enumerate(sol.t_events) if te.size]
    return (hit[0] + 1 if hit else 0), sol.t[-1]


def body3_offset(dx, dy, y_base, t_max):
    """Fate of y_base with body 3 (id 2) shifted by (dx, dy)."""
    y0 = y_base.copy()
    y0[8] += dx
    y0[9] += dy
    return fate(y0, t_max)

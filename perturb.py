import os
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from state import State, R_ESC
from states import figure_eight
from config import EPS

T_MAX = 100.0
N = 51       # grid is N x N
SPAN = 0.5   # body 3 offsets range over [-SPAN, SPAN] in x and y
FATES = [f"bound at t={T_MAX:g}", "body 1 escapes", "body 2 escapes", "body 3 escapes"]
COLORS = ["#d9d8d3", "#2a78d6", "#eb6834", "#1baf7a"]
# parameters in the filename, so changing any of them starts a fresh cache instead of reusing stale results
CACHE = f"fates_N{N}_span{SPAN:g}_T{T_MAX:g}_soft{EPS:g}_esc{R_ESC:g}.npz"


def rhs(t, y):
    return State(y).derivative()


def make_escape(i):
    def event(t, y):
        return State(y).body_escaping(i)
    event.terminal, event.direction = True, 1 # type: ignore[reportFunctionMemberAccess]
    return event


# no collision event: softening (config.EPS) lets bodies pass through close encounters
EVENTS = [make_escape(0), make_escape(1), make_escape(2)]


def fate(y0):
    """Returns (index into FATES, time the run stopped)."""
    sol = solve_ivp(rhs, (0, T_MAX), y0, method="DOP853", rtol=1e-10, atol=1e-12, events=EVENTS)
    hit = [n for n, te in enumerate(sol.t_events) if te.size]
    return (hit[0] + 1 if hit else 0), sol.t[-1]


def run_pixel(job):
    r, c, dx, dy = job
    y0 = figure_eight.y.copy()
    y0[8] += dx
    y0[9] += dy
    return r, c, *fate(y0)


def sweep():
    """Fate of every body-3 offset; fates[row, col] is offset (dx=offsets[col], dy=offsets[row]).
    Runs pixels on all cores, saves every N pixels and skips finished ones, so an interrupted run resumes."""
    offsets = np.linspace(-SPAN, SPAN, N)
    if os.path.exists(CACHE):
        data = np.load(CACHE)
        fates, times = data["fates"], data["times"]
    else:
        fates, times = np.full((N, N), -1), np.full((N, N), np.nan)   # -1 = not computed yet
    todo = [(r, c, dx, dy) for r, dy in enumerate(offsets) for c, dx in enumerate(offsets) if fates[r, c] < 0]
    with Pool() as pool:
        for n, (r, c, f, t) in enumerate(pool.imap_unordered(run_pixel, todo), 1):
            fates[r, c], times[r, c] = f, t
            if n % N == 0 or n == len(todo):
                np.savez(CACHE, fates=fates, times=times, offsets=offsets)
                print(f"{n}/{len(todo)} pixels saved", flush=True)
    return offsets, fates, times


def plot(offsets, fates, times, path):
    """Fate map and escape-time map side by side."""
    from matplotlib.colors import ListedColormap, LogNorm
    from matplotlib.patches import Patch

    h = offsets[1] - offsets[0]   # pixel width, so pixel centres land on the offsets
    extent = (offsets[0] - h / 2, offsets[-1] + h / 2, offsets[0] - h / 2, offsets[-1] + h / 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    ax1.imshow(fates, origin="lower", extent=extent, cmap=ListedColormap(COLORS),
               vmin=-0.5, vmax=len(COLORS) - 0.5, interpolation="nearest")
    ax1.legend(handles=[Patch(color=c, label=l) for c, l in zip(COLORS, FATES)],
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax1.set_title("Fate")

    # one-hue ramp (light = early escape, dark = late), starting off-white so it stays distinct from the gray
    blues = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.3, 1, 256))).with_extremes(bad=COLORS[0])
    im = ax2.imshow(np.ma.masked_where(fates == 0, times), origin="lower", extent=extent, cmap=blues,
                    norm=LogNorm(vmin=times[fates > 0].min(), vmax=T_MAX), interpolation="nearest")
    fig.colorbar(im, ax=ax2, label=f"escape time (log scale); gray = bound at t={T_MAX:g}")
    ax2.set_title("Escape time")

    for ax in (ax1, ax2):
        ax.set_xlabel("body 3 offset dx")
        ax.set_ylabel("body 3 offset dy")
    fig.suptitle(f"Body 3 offset from the figure-eight start (softening {EPS:g}, {len(offsets)}x{len(offsets)} grid)")
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    plot(*sweep(), CACHE.replace(".npz", ".png"))

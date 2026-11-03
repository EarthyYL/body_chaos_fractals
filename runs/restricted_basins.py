"""Restricted three-body problem (rotating frame): basin maps over launch position.

Run from the repo root:
    python -m runs.restricted_basins fixed_c   # every launch has Jacobi constant C, moving perpendicular to its position
    python -m runs.restricted_basins rest      # every launch starts at rest in the rotating frame (its own C)
Resumes from its cache in data/restricted/ if one exists for these parameters.

Only the bottom half of the grid (plus the y = 0 row) is integrated: with equal primaries, rotating a launch by
180 degrees gives the same fate with the two primaries swapped, so the top half is filled in from the bottom.
"""
import sys
from functools import partial
import numpy as np
from lib.sweep import grid_sweep
from lib.plotting import fate_and_time
from systems.restricted.model import MU, R_COLL, R_ESC
from systems.restricted.fate import FORBIDDEN, pixel_fixed_c, pixel_rest

C = 3.2          # fixed_c only; below 3.4568 (L2/L3) the exits to infinity are open
T_MAX = 100.0
SPAN = 1.5       # launch positions cover [-SPAN, SPAN] in x and y
FATES = [f"bound at t={T_MAX:g}", "escapes", "crashes into primary 1", "crashes into primary 2", "forbidden at this C"]
COLORS = ["#d9d8d3", "#2a78d6", "#eb6834", "#1baf7a", "#383835"]
SWAP = np.array([0, 1, 3, 2, 4])   # fate under the 180-degree rotation: crash into primary 1 <-> primary 2
MODES = {   # mode: (pixel function, grid size N (odd), cache prefix, number of legend entries, title text)
    "fixed_c": (partial(pixel_fixed_c, C=C, t_max=T_MAX), 601, f"basins_C{C:g}", 5, f"C = {C:g}"),
    "rest": (partial(pixel_rest, t_max=T_MAX), 201, "basins_rest", 4, "released at rest"),
}


def mirror(half, n, swap=None):
    """Full n-row grid from rows 0..n//2 via the 180-degree symmetry: row r, col c of the top is row n-1-r, col n-1-c."""
    top = half[: n // 2][::-1, ::-1]
    return np.vstack([half, top if swap is None else swap[top]])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fixed_c"
    if mode not in MODES:
        sys.exit(f"unknown mode {mode!r}: use {' or '.join(MODES)}")
    fn, N, prefix, n_fates, desc = MODES[mode]
    assert MU == 0.5 and N % 2 == 1, "the half-grid symmetry needs equal primaries and an odd grid (middle row at y = 0)"
    cache = f"data/restricted/{prefix}_mu{MU:g}_N{N}_span{SPAN:g}_T{T_MAX:g}_coll{R_COLL:g}_esc{R_ESC:g}_half.npz"

    xs = np.linspace(-SPAN, SPAN, N)
    half = grid_sweep(fn, xs, xs[: N // 2 + 1], cache, ("fates", "times", "drifts"))
    fates, times, drifts = mirror(half["fates"], N, SWAP), mirror(half["times"], N), mirror(half["drifts"], N)
    moved = fates != FORBIDDEN
    print(f"max Jacobi drift {np.nanmax(drifts[moved]):.1e}; pixels above 1e-8: {np.mean(drifts[moved] > 1e-8):.1%}")
    fate_and_time(xs, xs, fates, times, FATES[:n_fates], COLORS[:n_fates], timed=(fates >= 1) & (fates <= 3),
                  t_max=T_MAX, path=cache.replace(".npz", ".png"),
                  title=f"Restricted three-body basins (mu = {MU:g}, {desc}, {N}x{N} grid)",
                  xlabel="launch x (rotating frame)", ylabel="launch y (rotating frame)",
                  marks=[(-MU, 0.0), (1 - MU, 0.0)])

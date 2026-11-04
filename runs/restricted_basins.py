"""Restricted three-body problem (rotating frame): basin maps over launch position.

Run from the repo root:
    python -m runs.restricted_basins fixed_c         # every launch has Jacobi constant C, moving perpendicular to its position
    python -m runs.restricted_basins rest            # every launch starts at rest in the rotating frame (its own C)
    python -m runs.restricted_basins fixed_c_inset   # fixed_c map with the detailed centre re-run K times finer
Resumes from its cache in data/restricted/ if one exists for these parameters.

Only the bottom half of each grid (plus the y = 0 row) is integrated: with equal primaries, rotating a launch by
180 degrees gives the same fate with the two primaries swapped, so the top half is filled in from the bottom.
"""
import sys
from functools import partial
import numpy as np
from lib.sweep import grid_sweep
from lib.plotting import fate_and_time, fate_map
from systems.restricted.model import MU, R_COLL, R_ESC
from systems.restricted.fate import FORBIDDEN, pixel_fixed_c, pixel_rest

C = 3.2          # fixed_c only; below 3.4568 (L2/L3) the exits to infinity are open
T_MAX = 100.0
SPAN = 1.5       # launch positions cover [-SPAN, SPAN] in x and y
K = 3            # fixed_c_inset: inset pixels are K times finer than the fixed_c grid (odd, so y = 0 stays a grid row)
INSET = (1.1, 1.15)   # fixed_c_inset: |x|, |y| bounds (at fixed_c pixel centres); holds all bound/crash detail at C = 3.2
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


def symmetric_sweep(fn, xs, ys, cache):
    """Sweep the bottom half (plus the y = 0 row) of a grid symmetric about the origin; return the mirrored full maps."""
    assert MU == 0.5 and len(ys) % 2 == 1, "the half-grid symmetry needs equal primaries and an odd row count (y = 0 row)"
    half = grid_sweep(fn, xs, ys[: len(ys) // 2 + 1], cache, ("fates", "times", "drifts"))
    return mirror(half["fates"], len(ys), SWAP), mirror(half["times"], len(ys)), mirror(half["drifts"], len(ys))


def refine(fn, xs, coarse, k, bounds, cache):
    """Re-run the window |x| <= bounds[0], |y| <= bounds[1] of the square grid xs with k x k pixels per coarse pixel,
    and paste it into the coarse maps upsampled k times. Returns the fine axis and the combined maps."""
    d = xs[1] - xs[0]
    offsets = (np.arange(k) - (k - 1) / 2) * d / k                     # fine pixel centres inside one coarse pixel
    split = lambda centres: (centres[:, None] + offsets).ravel()
    cols, rows = (np.flatnonzero(np.abs(xs) <= b + d / 2) for b in bounds)
    inset = symmetric_sweep(fn, split(xs[cols]), split(xs[rows]), cache)
    combined = []
    for c, i in zip(coarse, inset):
        a = np.repeat(np.repeat(c, k, 0), k, 1)
        a[rows[0] * k: (rows[-1] + 1) * k, cols[0] * k: (cols[-1] + 1) * k] = i
        combined.append(a)
    # alignment check: with k odd, the middle pixel of each k x k block is exactly the coarse launch point
    m = k // 2
    agree = np.mean(inset[0][m::k, m::k] == coarse[0][np.ix_(rows, cols)])
    print(f"inset centre pixels reproduce the coarse map on {agree:.2%} of pixels (should be 100%)")
    return split(xs), combined


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fixed_c"
    if mode not in (*MODES, "fixed_c_inset"):
        sys.exit(f"unknown mode {mode!r}: use {', '.join(MODES)} or fixed_c_inset")
    fn, N, prefix, n_fates, desc = MODES["fixed_c" if mode == "fixed_c_inset" else mode]
    cache = f"data/restricted/{prefix}_mu{MU:g}_N{N}_span{SPAN:g}_T{T_MAX:g}_coll{R_COLL:g}_esc{R_ESC:g}_half.npz"

    xs = np.linspace(-SPAN, SPAN, N)
    fates, times, drifts = symmetric_sweep(fn, xs, xs, cache)   # for the inset mode this just loads the coarse map
    grid = f"{N}x{N} grid"
    if mode == "fixed_c_inset":
        assert K % 2 == 1, "K must be odd so the inset keeps a y = 0 row for the symmetry"
        cache = cache.replace("_half.npz", f"_inset{INSET[0]:g}x{INSET[1]:g}_k{K}_half.npz")
        xs, (fates, times, drifts) = refine(fn, xs, (fates, times, drifts), K, INSET, cache)
        grid = f"{N}x{N} grid, centre {K}x finer"

    moved = fates != FORBIDDEN
    print(f"max Jacobi drift {np.nanmax(drifts[moved]):.1e}; pixels above 1e-8: {np.mean(drifts[moved] > 1e-8):.1%}")
    labels = dict(xlabel="launch x (rotating frame)", ylabel="launch y (rotating frame)", marks=[(-MU, 0.0), (1 - MU, 0.0)])
    fate_and_time(xs, xs, fates, times, FATES[:n_fates], COLORS[:n_fates], timed=(fates >= 1) & (fates <= 3),
                  t_max=T_MAX, path=cache.replace(".npz", ".png"),
                  title=f"Restricted three-body basins (mu = {MU:g}, {desc}, {grid})", **labels)
    fate_map(xs, xs, fates, FATES[:n_fates], COLORS[:n_fates], path=cache.replace(".npz", "_fate_large.png"),
             title=f"Restricted three-body basins: fate (mu = {MU:g}, {desc}, {grid})", **labels)

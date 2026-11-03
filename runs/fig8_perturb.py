"""Full three-body problem: fate of the figure-eight with body 3's start shifted by (dx, dy).

Run from the repo root:  python -m runs.fig8_perturb
Resumes from its cache in data/three_body/ if one exists for these parameters.
"""
from functools import partial
import numpy as np
from lib.sweep import grid_sweep
from lib.plotting import fate_and_time
from systems.three_body.config import EPS
from systems.three_body.state import R_ESC
from systems.three_body.states import figure_eight
from systems.three_body.fate import body3_offset

T_MAX = 100.0
N = 171      # grid is N x N
SPAN = 0.5   # body 3 offsets range over [-SPAN, SPAN] in x and y
FATES = [f"bound at t={T_MAX:g}", "body 1 escapes", "body 2 escapes", "body 3 escapes"]
COLORS = ["#d9d8d3", "#2a78d6", "#eb6834", "#1baf7a"]
# parameters in the filename, so changing any of them starts a fresh cache instead of reusing stale results
CACHE = f"data/three_body/fates_N{N}_span{SPAN:g}_T{T_MAX:g}_soft{EPS:g}_esc{R_ESC:g}.npz"

if __name__ == "__main__":
    xs = np.linspace(-SPAN, SPAN, N)
    out = grid_sweep(partial(body3_offset, y_base=figure_eight.y, t_max=T_MAX), xs, xs, CACHE, ("fates", "times"))
    fates, times = out["fates"], out["times"]
    fate_and_time(xs, xs, fates, times, FATES, COLORS, timed=fates > 0, t_max=T_MAX,
                  path=CACHE.replace(".npz", ".png"),
                  title=f"Body 3 offset from the figure-eight start (softening {EPS:g}, {N}x{N} grid)",
                  xlabel="body 3 offset dx", ylabel="body 3 offset dy")

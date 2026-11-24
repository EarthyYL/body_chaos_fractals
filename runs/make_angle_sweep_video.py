"""Looping GIF of the fixed-C basin map as the launch angle alpha turns through 360 degrees.

    python -m runs.make_angle_sweep_video [N] [STEP]   # defaults: 401, 5 degrees
"""
import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from lib.plotting import fate_legend, fate_panel
from systems.restricted.model import MU, R_COLL, R_ESC
from systems.restricted.fate import COLORS, FATES, T_MAX, pixel_fixed_c, symmetric_sweep

C = 3.2
SPAN = 1.5
FPS = 12

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 401
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    n = round(360 / step)
    assert abs(n * step - 360) < 1e-9, "step must divide 360"
    angles = (90 + step * np.arange(n)) % 360   # no 360 frame: the GIF wraps back to the first
    folder = f"data/restricted/sweep_C{C:g}_mu{MU:g}_N{N}_span{SPAN:g}_T{T_MAX:g}_coll{R_COLL:g}_esc{R_ESC:g}_f78"

    xs = np.linspace(-SPAN, SPAN, N)
    frames = []
    for k, a in enumerate(angles):
        print(f"alpha = {a:g} deg ({k + 1}/{n})", flush=True)
        fn = partial(pixel_fixed_c, C=C, t_max=T_MAX, alpha=np.radians(a))
        frames.append(symmetric_sweep(fn, xs, xs, f"{folder}/a{a:g}_half.npz")[0].astype(np.int8))

    fig, ax = plt.subplots(figsize=(7, 7.8), constrained_layout=True)
    im = fate_panel(ax, xs, xs, frames[0], COLORS, marks=[(-MU, 0.0), (1 - MU, 0.0)])
    fate_legend(ax, FATES, COLORS, bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.set_xlabel("launch x (rotating frame)")
    ax.set_ylabel("launch y (rotating frame)")
    fig.suptitle(f"Restricted three-body basins (mu = {MU:g}, C = {C:g}, {N}x{N} grid)")

    def show(k):
        im.set_data(frames[k])
        ax.set_title(f"launch angle α = {angles[k]:g}° from the position vector")

    show(0)
    fig.canvas.draw()
    fig.set_layout_engine("none")   # freeze the layout so the axes don't jitter as the title changes
    path = f"{folder}_step{step:g}.gif"
    FuncAnimation(fig, show, frames=n).save(path, writer=PillowWriter(fps=FPS), dpi=100)
    print("saved", path)

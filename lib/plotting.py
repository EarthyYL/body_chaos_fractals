import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch


def fate_panel(ax, xs, ys, fates, colors, marks=(), ms=5):
    """Fate map on ax, marks as white dots; returns the image."""
    hx, hy = xs[1] - xs[0], ys[1] - ys[0]   # pixel size, so pixel centres land on the grid points
    extent = (xs[0] - hx / 2, xs[-1] + hx / 2, ys[0] - hy / 2, ys[-1] + hy / 2)
    im = ax.imshow(fates, origin="lower", extent=extent, cmap=ListedColormap(colors),
                   vmin=-0.5, vmax=len(colors) - 0.5, interpolation="nearest")
    if marks:
        ax.plot(*zip(*marks), "o", ms=ms, mfc="white", mec="black")
    return im


def fate_legend(ax, labels, colors, **kw):
    """Fate legend below ax."""
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)], loc="upper center", frameon=False, **kw)


def fate_and_time(xs, ys, fates, times, labels, colors, timed, t_max, path, title, xlabel, ylabel, marks=()):
    """Two panels: the fate map, and the time of the outcome (log scale) for pixels in the `timed` mask.
    Pixels outside `timed` keep their fate color in the time panel. marks: (x, y) points drawn as white dots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    fate_panel(ax1, xs, ys, fates, colors, marks)
    fate_legend(ax1, labels, colors, bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax1.set_title("Fate")

    timed = timed & (times > 0)
    # one-hue ramp (light = early, dark = late), starting off-white so it stays distinct from the gray
    blues = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.3, 1, 256)))
    under = fate_panel(ax2, xs, ys, fates, colors, marks)   # underneath: untimed pixels show their fate color
    im = ax2.imshow(np.ma.masked_where(~timed, times), origin="lower", extent=under.get_extent(), cmap=blues,
                    norm=LogNorm(vmin=times[timed].min(), vmax=t_max), interpolation="nearest")
    fig.colorbar(im, ax=ax2, label="time of outcome (log scale); other pixels colored as in Fate")
    ax2.set_title("Time until outcome")

    for ax in (ax1, ax2):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fate_map(xs, ys, fates, labels, colors, path, title, xlabel, ylabel, marks=(), dpi=300):
    """One large fate panel, for looking at fine structure (about 3600 px wide at the default dpi)."""
    fig, ax = plt.subplots(figsize=(12, 13), constrained_layout=True)
    fate_panel(ax, xs, ys, fates, colors, marks, ms=7)
    fate_legend(ax, labels, colors, bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

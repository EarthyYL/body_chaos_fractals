import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch


def fate_and_time(xs, ys, fates, times, labels, colors, timed, t_max, path, title, xlabel, ylabel, marks=()):
    """Two panels: the fate map, and the time of the outcome (log scale) for pixels in the `timed` mask.
    Pixels outside `timed` keep their fate color in the time panel. marks: (x, y) points drawn as white dots."""
    hx, hy = xs[1] - xs[0], ys[1] - ys[0]   # pixel size, so pixel centres land on the grid points
    extent = (xs[0] - hx / 2, xs[-1] + hx / 2, ys[0] - hy / 2, ys[-1] + hy / 2)
    fate_img = dict(origin="lower", extent=extent, cmap=ListedColormap(colors),
                    vmin=-0.5, vmax=len(colors) - 0.5, interpolation="nearest")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    ax1.imshow(fates, **fate_img)
    ax1.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)],
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax1.set_title("Fate")

    timed = timed & (times > 0)
    # one-hue ramp (light = early, dark = late), starting off-white so it stays distinct from the gray
    blues = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.3, 1, 256)))
    ax2.imshow(fates, **fate_img)   # underneath: untimed pixels show their fate color
    im = ax2.imshow(np.ma.masked_where(~timed, times), origin="lower", extent=extent, cmap=blues,
                    norm=LogNorm(vmin=times[timed].min(), vmax=t_max), interpolation="nearest")
    fig.colorbar(im, ax=ax2, label="time of outcome (log scale); other pixels colored as in Fate")
    ax2.set_title("Time until outcome")

    for ax in (ax1, ax2):
        if marks:
            ax.plot(*zip(*marks), "o", ms=5, mfc="white", mec="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fate_map(xs, ys, fates, labels, colors, path, title, xlabel, ylabel, marks=(), dpi=300):
    """One large fate panel, for looking at fine structure (about 3600 px wide at the default dpi)."""
    hx, hy = xs[1] - xs[0], ys[1] - ys[0]
    extent = (xs[0] - hx / 2, xs[-1] + hx / 2, ys[0] - hy / 2, ys[-1] + hy / 2)
    fig, ax = plt.subplots(figsize=(12, 13), constrained_layout=True)
    ax.imshow(fates, origin="lower", extent=extent, cmap=ListedColormap(colors),
              vmin=-0.5, vmax=len(colors) - 0.5, interpolation="nearest")
    if marks:
        ax.plot(*zip(*marks), "o", ms=7, mfc="white", mec="black")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)],
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

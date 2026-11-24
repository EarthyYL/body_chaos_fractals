"""Pixel accuracy and cost of integrators, tolerances and float types on the C = 3.2 basin map, against a reference.

    python -m runs.compare_integrator_accuracy [N]   # default 401
"""
import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import systems.restricted._fast as _fast
from systems.restricted.model import MU, R_COLL, R_ESC
from systems.restricted.fate import FATES, FORBIDDEN, T_MAX, pixel_fixed_c, symmetric_sweep

C = 3.2
SPAN = 1.5
FIELDS = ("fates", "times", "drifts", "rhs")
REF = ("f78", "long double", 1e-17, "double")       # (method, float type, rtol or RK4 step, start rounded to)
REF_CHECK = ("f78", "double", 1e-14, "double")      # pixels where it disagrees with REF are left out of the scores
decades = lambda a, b: [10.0 ** -k for k in range(a, b + 1)]
SERIES = {
    "Fehlberg 7(8)": [("f78", "double", t, "double") for t in decades(4, 13)],   # 1e-14 is REF_CHECK
    "Bulirsch–Stoer": [("bs", "double", t, "double") for t in decades(4, 14)],
    "Dormand–Prince 5(4)": [("dopri5", "double", t, "double") for t in decades(4, 12)],
    "Cash–Karp 5(4)": [("ck54", "double", t, "double") for t in decades(4, 12)],
    "RK4, fixed step": [("rk4", "double", h, "double") for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)],
    "Fehlberg 7(8), long double": [("f78", "long double", t, "double") for t in decades(10, 16)],
    "Fehlberg 7(8), float": [("f78", "float", t, "double") for t in decades(3, 6)],
    "Fehlberg 7(8), float-rounded start": [("f78", "double", 1e-14, "float")],
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def integrate(y0, t_max, method, precision, knob, start):
    if start == "float":
        y0 = y0.astype(np.float32).astype(np.float64)
    rtol, atol, dt = (0.0, 0.0, knob) if method == "rk4" else (knob, knob / 100, 1e-3)
    return _fast.fate_with(*y0, t_max, method, precision, rtol, atol, dt)


def sweep(cfg, xs, folder):
    method, precision, knob, start = cfg
    fn = partial(pixel_fixed_c, C=C, t_max=T_MAX,
                 integrate=partial(integrate, method=method, precision=precision, knob=knob, start=start))
    name = f"{method}_{precision.replace(' ', '')}_{knob:g}" + ("_f32start" if start == "float" else "")
    return dict(zip(FIELDS, symmetric_sweep(fn, xs, xs, f"{folder}/{name}_half.npz", FIELDS)))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 401
    folder = f"data/restricted/integrators_C{C:g}_mu{MU:g}_N{N}_span{SPAN:g}_T{T_MAX:g}_coll{R_COLL:g}_esc{R_ESC:g}"
    xs = np.linspace(-SPAN, SPAN, N)

    ref, check = sweep(REF, xs, folder), sweep(REF_CHECK, xs, folder)
    allowed = ref["fates"] != FORBIDDEN
    settled = allowed & (ref["fates"] == check["fates"])
    ref_frac = np.bincount(ref["fates"][allowed], minlength=4)[:4] / allowed.sum()
    print(f"references agree on {settled.sum() / allowed.sum():.3%} of {allowed.sum()} allowed pixels; scoring on those")
    print("reference fractions:", "  ".join(f"{l}: {f:.4f}" for l, f in zip(FATES, ref_frac)))
    print(f"{'series':36s} {'tol/step':>8s} {'wrong':>9s} {'rhs/px':>8s} {'drift':>8s}  fraction - reference, per fate")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)
    floor = 0.5 / settled.sum()   # zero wrong pixels is drawn here
    for (name, cfgs), color in zip(SERIES.items(), COLORS):
        pts = []
        for cfg in cfgs:
            r = sweep(cfg, xs, folder)
            wrong = np.mean(r["fates"][settled] != ref["fates"][settled])
            cost = np.nanmean(r["rhs"][allowed])
            drift = np.nanmedian(r["drifts"][allowed])
            dfrac = np.bincount(r["fates"][allowed], minlength=4)[:4] / allowed.sum() - ref_frac
            print(f"{name:36s} {cfg[2]:8.0e} {wrong:9.2e} {cost:8.0f} {drift:8.1e}  " + " ".join(f"{d:+.4f}" for d in dfrac))
            pts.append((cost, drift, max(wrong, floor)))
        cost, drift, wrong = np.array(pts).T
        style = dict(color=color, marker="o", ms=5, lw=2, ls="-" if cfgs[0][1] == cfgs[0][3] == "double" else "--", label=name)
        ax1.plot(cost, wrong, **style)
        ax2.plot(drift, wrong, **style)

    ax1.set_xlabel("cost: right-hand-side evaluations per pixel (mean)")
    ax2.set_xlabel("measured error: |Jacobi constant drift| at the end of the run (median over pixels)")
    ax1.set_ylabel("fraction of settled pixels with the wrong fate")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Integrator accuracy on the C = {C:g} basin map ({N}x{N})\nscored against Fehlberg 7(8) in long double at "
                 f"rtol 1e-17, on the {settled.sum() / allowed.sum():.2%} of pixels where it agrees with double at rtol 1e-14")
    fig.savefig(f"{folder}.png", dpi=150)
    print("saved", f"{folder}.png")

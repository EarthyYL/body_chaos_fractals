"""The C++ integrator (fast.cpp) against the scipy reference. Run from the repo root: python -m pytest"""
import numpy as np
from systems.restricted.fate import pixel_fixed_c, fate_fast
from systems.restricted.reference import fate_scipy


def test_fast_matches_reference():
    pts = np.random.default_rng(0).uniform(-1.5, 1.5, (100, 2))
    fast = [pixel_fixed_c(x, y, 3.2, 100.0, fate_fast) for x, y in pts]
    ref = [pixel_fixed_c(x, y, 3.2, 100.0, fate_scipy) for x, y in pts]
    assert {f for f, *_ in fast} == {0, 1, 2, 3, 4}   # the sample covers every fate, so swapped labels would show
    # chaotic boundary pixels can differ between any two integrators (README, Accuracy); a real bug breaks many more
    assert sum(a[0] == b[0] for a, b in zip(fast, ref)) >= 97
    assert max(drift for *_, drift in fast) < 1e-5     # Jacobi constant conserved

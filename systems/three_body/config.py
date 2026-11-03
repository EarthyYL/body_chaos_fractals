import numpy as np

EPS = 1e-3   # softening length: gravity is smoothed below r ~ EPS, which removes the 1/r singularity

# Pair energy between bodies i, j is M[i] * M[j] * V(r).
def V(r: np.ndarray) -> np.ndarray:
    return -1 / np.sqrt(r**2 + EPS**2)

# G = -dV/dr: force on i from j is M[i] * M[j] * G(r) * (r_i - r_j) / r  (positive = repulsive).
# Keep it consistent with V; V is only used for the energy check and escape test.
def G(r: np.ndarray) -> np.ndarray:
    return -r / (r**2 + EPS**2) ** 1.5

M = (1.0, 1.0, 1.0)  # masses of bodies 1, 2, 3

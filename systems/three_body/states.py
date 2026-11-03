import numpy as np
from systems.three_body.state import State

# Both assume G = 1 and M = (1, 1, 1).

# Figure-eight (Moore 1993; Chenciner & Montgomery 2000). Linearly stable, period T = 6.32591398.
figure_eight = State(np.array([
    0.97000436, -0.24308753, 0.46620368, 0.43236573,
    -0.97000436, 0.24308753, 0.46620368, 0.43236573,
    0.0, 0.0, -0.93240737, -0.86473146,
]))

# Hierarchical triple: bodies 1, 2 in a circular binary (separation 1), body 3 on a circular
# orbit at distance 10 around the binary's centre of mass. Stable since 10 >> ~3.3 (Mardling-Aarseth).
_vo, _vi = np.sqrt(3 / 10), np.sqrt(2) / 2   # outer relative speed, inner per-body speed
hierarchical = State(np.array([
    -23 / 6, 0.0, 0.0, -_vo / 3 - _vi,
    -17 / 6, 0.0, 0.0, -_vo / 3 + _vi,
    20 / 3, 0.0, 0.0, 2 * _vo / 3,
]))

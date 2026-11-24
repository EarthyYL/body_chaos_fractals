import math
import numpy as np
from systems.restricted.model import launch, omega


def test_launch_angle():
    x, y, C = 0.3, -0.2, 3.2
    th, v = math.atan2(y, x), math.sqrt(2 * omega(x, y) - C)
    assert np.allclose(launch(x, y, C), [x, y, -v * math.sin(th), v * math.cos(th)], rtol=0, atol=1e-15)   # old default
    for alpha in (0.0, 1.0, math.pi, 5.0):
        _, _, vx, vy = launch(x, y, C, alpha)
        assert math.isclose(math.hypot(vx, vy), v)
        assert abs(math.remainder(math.atan2(vy, vx) - th - alpha, 2 * math.pi)) < 1e-12
    assert launch(0.0, 0.9, C, 0.0) is None   # forbidden

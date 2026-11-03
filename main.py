import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from state import State
from states import figure_eight


def rhs(t, y):
    return State(y).derivative()


system_path = solve_ivp(rhs, (0,5), y0 = figure_eight.y, method = "DOP853", rtol=1e-10, atol=1e-12)


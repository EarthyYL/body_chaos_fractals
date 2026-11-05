/* Compiled fate() for the restricted three-body problem: same physics as reference.py (its scipy twin), integrator from Boost.Odeint.
   Build from the repo root (rebuild after every edit to this file):
     c++ -O3 -march=native -Wall -std=c++17 -shared -fPIC $(python -m pybind11 --includes) \
         systems/restricted/fast.cpp -o systems/restricted/_fast$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // converts std::tuple to a Python tuple
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"   // false positive inside Odeint with GCC 13; still on for our code
#include <boost/numeric/odeint.hpp>
#pragma GCC diagnostic pop
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace py = pybind11;
namespace ode = boost::numeric::odeint;
using state = std::array<double, 4>;   // [x, y, vx, vy] in the rotating frame (reused for derivative)

constexpr double MU = 0.5, R_COLL = 1e-3, R_ESC = 10.0;


/**
 * @brief Calculates the derivative [vx, vy, ax, ay] of the system state.
 * Odeint needs exactly this signature: (state in, derivative out, time).
 */
void rhs(const state& s, state& d, double /* t */) {
    double x = s[0], y = s[1], vx = s[2], vy = s[3];
    double r1 = std::hypot(x + MU, y);
    double r2 = std::hypot(x - 1 + MU, y);
    double g1 = (1 - MU) / (r1 * r1 * r1);
    double g2 = (MU) / (r2 * r2 * r2);
    double ax = 2*vy + x - g1*(x + MU) - g2*(x - 1 + MU);  // Coriolis + centrifugal + gravity
    double ay = -2*vx + y - g1*y - g2*y;
    d = {vx, vy, ax, ay};
    
}

/// Return C, the Jacobi number of the state
double jacobi(const state& s) {
    // port of RotState.jacobi()
    double x = s[0], y = s[1], vx = s[2], vy = s[3];
    double omega = (x*x + y*y)/2 + (1 - MU)/std::hypot(x + MU, y) + MU/std::hypot(x - 1 + MU, y);
    return 2*omega - (vx*vx + vy*vy);
}

/// Crosses 0 upward when the particle is past R_ESC and unbound from the primaries (seen as one mass at the origin)
double escape_gap(const state& s){
    // port of RotState.escape_gap()
    double x = s[0], y = s[1], vx = s[2], vy = s[3];
    double r = std::hypot(x,y);
    if (r < R_ESC) {
        return r - R_ESC;
    }
    double energy = 0.5*((vx - y)*(vx - y) + (vy + x)*(vy + x)) - 1/r;   // inertial-frame energy per unit mass
    return std::min(r-R_ESC, energy);
}

/// Crosses 0 downward when the particle hits primary 0 or 1.
double crash_gap(const state& s, int primary){
    // port of RotState.crash_gap()
    double x = s[0], y = s[1];
    double px = (primary == 0) ? -MU : 1 - MU;
    return std::hypot(x - px, y) - R_COLL;
}


std::array<double, 3> events(const state& s) {
    return {escape_gap(s), crash_gap(s,0), crash_gap(s,1)};
}

// Same contract as reference.py's fate_scipy(): (0 bound / 1 escaped / 2, 3 crashed;  time the run stopped;  |Jacobi drift|)
std::tuple<int, double, double> fate(double x, double y, double vx, double vy, double t_max, double rtol, double atol) {
    state s{x, y, vx, vy};
    double c0 = jacobi(s);
    auto g = events(s);
    int f = 0;
    auto hit = [&](const auto& step) {   // a lambda; [&] lets it read and change g and f. step = (state, time)
        auto gn = events(step.first);
        if (g[0] <= 0 && gn[0] >= 0) f = 1;        // escape: upward crossing
        else if (g[1] >= 0 && gn[1] <= 0) f = 2;   // crashes: downward crossings
        else if (g[2] >= 0 && gn[2] <= 0) f = 3;
        g = gn;
        return f != 0;   // true stops the integration
    };
    auto stepper = ode::make_controlled(atol, rtol, ode::runge_kutta_fehlberg78<state>());   // can swap integrator
    auto steps = ode::make_adaptive_time_range(stepper, rhs, s, 0.0, t_max, 1e-3);   // steps s in place
    auto it = std::find_if(steps.first, steps.second, hit); // uses the iterator and looks for hit
    double t = it == steps.second ? t_max : (*it).second;   // end of the step the event fell in, not the exact crossing
    return {f, t, std::abs(jacobi(s) - c0)};
}

// The module name must match the file name you build (_fast.cpython-315-...so).
// PYBIND11_MODULE runs once, when Python imports _fast; m is the module object, and each m.def adds one function to it.
PYBIND11_MODULE(_fast, m) {
    m.def("fate",      // _fast.fate(...)
          &fate,       // pybind11 converts float <-> double and std::tuple -> tuple
          "Integrate one launch; returns (fate, stop time, |Jacobi drift|)",   
          py::arg("x"), py::arg("y"), py::arg("vx"), py::arg("vy"), py::arg("t_max"),   
          py::arg("rtol") = 1e-10, py::arg("atol") = 1e-12);                         // defaults, as in reference.py
}

/* Compiled fate() for the restricted three-body problem: same physics as reference.py (its scipy twin), integrator from Boost.Odeint.
   Build from the repo root (rebuild after every edit to this file):
     c++ -O3 -march=native -Wall -std=c++17 -shared -fPIC $(python -m pybind11 --includes) \
         systems/restricted/fast.cpp -o systems/restricted/_fast$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
   Everything is a template on Real, the float type (float, double or long double): the compiler writes one copy per type used.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // converts std::tuple to a Python tuple
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"   // false positives inside Odeint with GCC 13; still on for our code
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/numeric/odeint.hpp>
#pragma GCC diagnostic pop
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace py = pybind11;
namespace ode = boost::numeric::odeint;
template <class Real> using state = std::array<Real, 4>;   // [x, y, vx, vy] in the rotating frame (reused for derivative)
using result = std::tuple<int, double, double, std::int64_t>;     // (fate, stop time, |Jacobi drift|, RHS evaluations)

constexpr double MU = 0.5, R_COLL = 1e-3, R_ESC = 10.0;


/**
 * @brief Calculates the derivative [vx, vy, ax, ay] of the system state.
 * Odeint needs exactly this signature: (state in, derivative out, time).
 */
template <class Real>
void rhs(const state<Real>& s, state<Real>& d, Real /* t */) {
    const Real mu = MU;   // so float runs don't silently compute in double
    Real x = s[0], y = s[1], vx = s[2], vy = s[3];
    Real r1 = std::hypot(x + mu, y);
    Real r2 = std::hypot(x - 1 + mu, y);
    Real g1 = (1 - mu) / (r1 * r1 * r1);
    Real g2 = (mu) / (r2 * r2 * r2);
    Real ax = 2*vy + x - g1*(x + mu) - g2*(x - 1 + mu);  // Coriolis + centrifugal + gravity
    Real ay = -2*vx + y - g1*y - g2*y;
    d = {vx, vy, ax, ay};
}

/// Return C, the Jacobi number of the state
template <class Real>
Real jacobi(const state<Real>& s) {
    // port of RotState.jacobi()
    const Real mu = MU;
    Real x = s[0], y = s[1], vx = s[2], vy = s[3];
    Real omega = (x*x + y*y)/2 + (1 - mu)/std::hypot(x + mu, y) + mu/std::hypot(x - 1 + mu, y);
    return 2*omega - (vx*vx + vy*vy);
}

/// Crosses 0 upward when the particle is past R_ESC and unbound from the primaries (seen as one mass at the origin)
template <class Real>
Real escape_gap(const state<Real>& s){
    // port of RotState.escape_gap()
    const Real r_esc = R_ESC;
    Real x = s[0], y = s[1], vx = s[2], vy = s[3];
    Real r = std::hypot(x,y);
    if (r < r_esc) {
        return r - r_esc;
    }
    Real energy = Real(0.5)*((vx - y)*(vx - y) + (vy + x)*(vy + x)) - 1/r;   // inertial-frame energy per unit mass
    return std::min(r - r_esc, energy);
}

/// Crosses 0 downward when the particle hits primary 0 or 1.
template <class Real>
Real crash_gap(const state<Real>& s, int primary){
    // port of RotState.crash_gap()
    Real x = s[0], y = s[1];
    Real px = (primary == 0) ? -MU : 1 - MU;
    return std::hypot(x - px, y) - Real(R_COLL);
}


template <class Real>
std::array<Real, 3> events(const state<Real>& s) {
    return {escape_gap(s), crash_gap(s,0), crash_gap(s,1)};
}

// Same contract as reference.py's fate_scipy(), plus the RHS evaluation count.
// method: rk4 (fixed step dt), ck54, dopri5, f78 or bs (adaptive, first step dt)
template <class Real>
result fate(state<Real> s, double t_max, const std::string& method, double rtol, double atol, double dt) {
    using S = state<Real>;
    std::int64_t n = 0;
    auto sys = [&n](const S& s, S& d, Real t) { ++n; rhs(s, d, t); };
    Real c0 = jacobi(s);
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
    Real t = t_max;
    auto run = [&](auto steps) {   // steps s in place, stopping at the first hit
        auto it = std::find_if(steps.first, steps.second, hit);
        if (it != steps.second) t = (*it).second;   // end of the step the event fell in, not the exact crossing
    };
    Real T = t_max, h = dt, rt = rtol, at = atol;
    if (method == "rk4")         run(ode::make_const_step_time_range(ode::runge_kutta4<S, Real>(), sys, s, Real(0), T, h));
    else if (method == "ck54")   run(ode::make_adaptive_time_range(ode::make_controlled(at, rt, ode::runge_kutta_cash_karp54<S, Real>()), sys, s, Real(0), T, h));
    else if (method == "dopri5") run(ode::make_adaptive_time_range(ode::make_controlled(at, rt, ode::runge_kutta_dopri5<S, Real>()), sys, s, Real(0), T, h));
    else if (method == "f78")    run(ode::make_adaptive_time_range(ode::make_controlled(at, rt, ode::runge_kutta_fehlberg78<S, Real>()), sys, s, Real(0), T, h));
    else if (method == "bs")     run(ode::make_adaptive_time_range(ode::bulirsch_stoer<S, Real>(at, rt), sys, s, Real(0), T, h));
    else throw std::invalid_argument("unknown method: " + method);
    return {f, double(t), double(std::abs(jacobi(s) - c0)), n};
}

template <class Real>
result fate_as(double x, double y, double vx, double vy, double t_max, const std::string& method, double rtol, double atol, double dt) {
    return fate<Real>({Real(x), Real(y), Real(vx), Real(vy)}, t_max, method, rtol, atol, dt);
}

// pybind exports
PYBIND11_MODULE(_fast, m) {
    m.def("fate",      // _fast.fate(...): the maps' integrator, f78 in double
          [](double x, double y, double vx, double vy, double t_max, double rtol, double atol) {
              auto [f, t, drift, n] = fate_as<double>(x, y, vx, vy, t_max, "f78", rtol, atol, 1e-3);
              return std::make_tuple(f, t, drift);
          },
          "Integrate one launch; returns (fate, stop time, |Jacobi drift|)",
          py::arg("x"), py::arg("y"), py::arg("vx"), py::arg("vy"), py::arg("t_max"),
          py::arg("rtol") = 1e-10, py::arg("atol") = 1e-12);                         // defaults, as in reference.py
    m.def("fate_with",
          [](double x, double y, double vx, double vy, double t_max, const std::string& method, const std::string& precision,
             double rtol, double atol, double dt) {
              if (precision == "float")       return fate_as<float>(x, y, vx, vy, t_max, method, rtol, atol, dt);
              if (precision == "double")      return fate_as<double>(x, y, vx, vy, t_max, method, rtol, atol, dt);
              if (precision == "long double") {
                  if (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)   // e.g. MSVC, Apple Silicon
                      throw std::runtime_error("long double is no wider than double on this platform");
                  return fate_as<long double>(x, y, vx, vy, t_max, method, rtol, atol, dt);
              }
              throw std::invalid_argument("unknown precision: " + precision);
          },
          "Integrate one launch with a chosen method and float type; returns (fate, stop time, |Jacobi drift|, RHS evaluations)",
          py::arg("x"), py::arg("y"), py::arg("vx"), py::arg("vy"), py::arg("t_max"), py::arg("method"), py::arg("precision"),
          py::arg("rtol"), py::arg("atol"), py::arg("dt") = 1e-3);
}

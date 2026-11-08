"""
Open-channel flow calculations for rectangular and trapezoidal channels.

All units are SI: metres, seconds, m^3/s.
A rectangular channel is simply a trapezoidal channel with side slope m = 0.
"""

from dataclasses import dataclass
import math

G = 9.81  # gravitational acceleration [m/s^2]


@dataclass
class Channel:
    """Prismatic trapezoidal channel.

    Parameters
    ----------
    b : bottom width [m]
    m : side slope, horizontal : vertical (m = 0 -> rectangular)
    n : Manning roughness coefficient [s/m^(1/3)]
    S : longitudinal bed slope [m/m]
    """

    b: float
    m: float = 0.0
    n: float = 0.015
    S: float = 0.001

    def __post_init__(self):
        if self.b < 0 or self.m < 0:
            raise ValueError("Bottom width and side slope must be non-negative.")
        if self.b == 0 and self.m == 0:
            raise ValueError("Channel needs a bottom width or a side slope.")
        if self.n <= 0 or self.S <= 0:
            raise ValueError("Manning n and bed slope S must be positive.")

    # --- geometry -----------------------------------------------------------
    def area(self, y: float) -> float:
        """Flow area A [m^2] at depth y."""
        return (self.b + self.m * y) * y

    def wetted_perimeter(self, y: float) -> float:
        """Wetted perimeter P [m] at depth y."""
        return self.b + 2.0 * y * math.sqrt(1.0 + self.m ** 2)

    def top_width(self, y: float) -> float:
        """Free-surface width T [m] at depth y."""
        return self.b + 2.0 * self.m * y

    def hydraulic_radius(self, y: float) -> float:
        """Hydraulic radius R = A / P [m]."""
        return self.area(y) / self.wetted_perimeter(y)

    def hydraulic_depth(self, y: float) -> float:
        """Hydraulic depth D = A / T [m]."""
        return self.area(y) / self.top_width(y)

    # --- flow ---------------------------------------------------------------
    def discharge(self, y: float) -> float:
        """Uniform-flow discharge Q [m^3/s] from Manning's equation."""
        A = self.area(y)
        R = self.hydraulic_radius(y)
        return (1.0 / self.n) * A * R ** (2.0 / 3.0) * math.sqrt(self.S)

    def velocity(self, y: float, Q: float) -> float:
        """Mean velocity V = Q / A [m/s]."""
        return Q / self.area(y)

    def froude(self, y: float, Q: float) -> float:
        """Froude number Fr = V / sqrt(g D)."""
        return self.velocity(y, Q) / math.sqrt(G * self.hydraulic_depth(y))

    def specific_energy(self, y: float, Q: float) -> float:
        """Specific energy E = y + V^2 / 2g [m]."""
        V = self.velocity(y, Q)
        return y + V ** 2 / (2.0 * G)

    def normal_depth(self, Q: float) -> float:
        """Normal (uniform-flow) depth y_n [m] for discharge Q."""
        return _solve(lambda y: self.discharge(y) - Q, Q)

    def critical_depth(self, Q: float) -> float:
        """Critical depth y_c [m], where Q^2 T / (g A^3) = 1."""
        return _solve(lambda y: Q ** 2 * self.top_width(y) / (G * self.area(y) ** 3) - 1.0, Q,
                      increasing=False)

    def flow_regime(self, y: float, Q: float) -> str:
        """Classify flow as subcritical, critical or supercritical."""
        fr = self.froude(y, Q)
        if abs(fr - 1.0) < 0.01:
            return "critical"
        return "subcritical" if fr < 1.0 else "supercritical"

    def summary(self, Q: float) -> dict:
        """Key results for a design discharge Q."""
        yn = self.normal_depth(Q)
        yc = self.critical_depth(Q)
        return {
            "discharge_m3s": Q,
            "normal_depth_m": yn,
            "critical_depth_m": yc,
            "velocity_ms": self.velocity(yn, Q),
            "froude": self.froude(yn, Q),
            "regime": self.flow_regime(yn, Q),
            "slope_type": "mild" if yn > yc else "steep",
            "specific_energy_m": self.specific_energy(yn, Q),
        }


def _solve(f, Q, increasing=True, rel_tol=1e-10, max_iter=300):
    """Bisection root finder for a depth y > 0.

    f         : function of depth whose root is sought
    increasing: True if f grows with depth, False if it decreases
    The search interval is expanded upwards until it contains the root,
    then halved until its width is below rel_tol * y.
    """
    if Q <= 0:
        raise ValueError("Discharge Q must be positive.")
    lo, hi = 1e-9, 1.0
    # expand the bracket until the sign changes
    while (f(hi) < 0) == increasing:
        lo = hi
        hi *= 2.0
        if hi > 1e5:
            raise RuntimeError("Could not bracket the solution.")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if (f(mid) < 0) == increasing:
            lo = mid
        else:
            hi = mid
        if hi - lo < rel_tol * hi:
            break
    return 0.5 * (lo + hi)

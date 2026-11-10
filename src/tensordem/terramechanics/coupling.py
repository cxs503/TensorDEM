"""Slip–sinkage coupling for tracked vehicles on soft sediment.

Physics
-------
When a tracked vehicle moves on soft sediment, the slip ratio ``s`` and the
sinkage ``z`` are coupled through the Bekker pressure–sinkage relation and the
Janosi shear–traction curve.

The slip ratio is defined as:

    s = 1 - V / (omega * r)    (driving, s > 0)

where ``V`` is the vehicle forward speed, ``omega`` is the sprocket angular
velocity, and ``r`` is the effective track radius.

The shear displacement along the track contact length ``L`` is:

    j(x) = s * x    for driving (s > 0)

The shear stress distribution is:

    tau(x) = tau_max * (1 - exp(-j(x) / K))
           = tau_max * (1 - exp(-s * x / K))

The total tractive force (thrust) is:

    F_t = b * integral_0^L tau(x) dx
        = b * tau_max * [L - K/s * (1 - exp(-s * L / K))]

The motion resistance (compaction resistance) is:

    R_c = b * integral_0^z p(z') dz'
        = b * (k_c/b + k_phi) * z^{n+1} / (n+1)

The drawbar pull is:

    DP = F_t - R_c

The sinkage is determined by the average ground pressure:

    p_avg = W / (b * L) = (k_c/b + k_phi) * z^n

where ``W`` is the vehicle weight.

References
----------
- Wong, J.Y. (2001) *Theory of Ground Vehicles*, 3rd ed., Wiley, Ch. 2.
- Bekker, M.G. (1956) *Theory of Land Locomotion*, Univ. of Michigan Press.

Units are SI (m, Pa, N, -).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SlipSinkageState",
    "track_slip_sinkage",
    "track_drawbar_pull",
    "track_motion_resistance",
]


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class SlipSinkageState:
    """Complete state of a tracked vehicle on soft sediment.

    Parameters
    ----------
    weight : float
        Vehicle weight (N).
    track_width : float
        Track width ``b`` (m).
    track_length : float
        Contact length ``L`` (m).
    slip_ratio : float
        Slip ratio ``s`` (dimensionless, 0–1 for driving).
    k_c : float
        Bekker cohesive modulus (N/m^{n+1}).
    k_phi : float
        Bekker frictional modulus (N/m^{n+2}).
    n : float
        Bekker deformation exponent.
    cohesion : float
        Soil cohesion ``c`` (Pa).
    friction_angle_rad : float
        Soil internal friction angle (rad).
    K : float
        Janosi shear deformation modulus (m).
    """

    weight: float
    track_width: float
    track_length: float
    slip_ratio: float
    k_c: float
    k_phi: float
    n: float
    cohesion: float
    friction_angle_rad: float
    K: float

    def __post_init__(self) -> None:
        for name in ("weight", "track_width", "track_length", "k_phi", "K"):
            _finite(name, getattr(self, name), positive=True)
        _finite("k_c", self.k_c, nonnegative=True)
        _finite("n", self.n, positive=True)
        _finite("cohesion", self.cohesion, nonnegative=True)
        _finite("friction_angle_rad", self.friction_angle_rad)
        _finite("slip_ratio", self.slip_ratio, nonnegative=True)
        if self.friction_angle_rad < 0.0 or self.friction_angle_rad >= math.pi / 2.0:
            raise ValueError("friction_angle_rad must be in [0, pi/2)")
        if self.slip_ratio > 1.0:
            raise ValueError("slip_ratio must be <= 1.0")

    @property
    def contact_area(self) -> float:
        return self.track_width * self.track_length

    @property
    def avg_pressure(self) -> float:
        return self.weight / self.contact_area

    @property
    def sinkage(self) -> float:
        """Sinkage from Bekker relation."""
        stiffness = self.k_c / self.track_width + self.k_phi
        return (self.avg_pressure / stiffness) ** (1.0 / self.n)

    @property
    def tau_max(self) -> float:
        """Maximum shear stress from Mohr–Coulomb."""
        return self.cohesion + self.avg_pressure * math.tan(self.friction_angle_rad)

    @property
    def thrust(self) -> float:
        """Total tractive force (N).

        F_t = b * tau_max * [L - K/s * (1 - exp(-s * L / K))]
        """
        s = self.slip_ratio
        if s < 1e-12:
            # Limit as s -> 0: thrust -> 0
            return 0.0
        b = self.track_width
        tmax = self.tau_max
        L = self.track_length
        K = self.K
        return b * tmax * (L - K / s * (1.0 - math.exp(-s * L / K)))

    @property
    def motion_resistance(self) -> float:
        """Compaction resistance (N).

        R_c = b * (k_c/b + k_phi) * z^{n+1} / (n+1)
        """
        z = self.sinkage
        stiffness = self.k_c / self.track_width + self.k_phi
        return self.track_width * stiffness * z ** (self.n + 1.0) / (self.n + 1.0)

    @property
    def drawbar_pull(self) -> float:
        """Net drawbar pull = thrust - motion resistance (N)."""
        return self.thrust - self.motion_resistance

    @property
    def tractive_efficiency(self) -> float:
        """Tractive efficiency = DP / thrust (dimensionless)."""
        if self.thrust < 1e-12:
            return 0.0
        return max(0.0, self.drawbar_pull / self.thrust)


def track_slip_sinkage(
    weight: float, track_width: float, track_length: float,
    slip_ratio: float, k_c: float, k_phi: float, n: float,
    cohesion: float, friction_angle_rad: float, K: float,
) -> dict[str, float]:
    """Compute full slip–sinkage state in one call.

    Returns dict with keys: sinkage, avg_pressure, tau_max, thrust,
    motion_resistance, drawbar_pull, tractive_efficiency.
    """
    state = SlipSinkageState(
        weight=weight, track_width=track_width, track_length=track_length,
        slip_ratio=slip_ratio, k_c=k_c, k_phi=k_phi, n=n,
        cohesion=cohesion, friction_angle_rad=friction_angle_rad, K=K,
    )
    return {
        "sinkage": state.sinkage,
        "avg_pressure": state.avg_pressure,
        "tau_max": state.tau_max,
        "thrust": state.thrust,
        "motion_resistance": state.motion_resistance,
        "drawbar_pull": state.drawbar_pull,
        "tractive_efficiency": state.tractive_efficiency,
    }


def track_drawbar_pull(
    weight: float, track_width: float, track_length: float,
    slip_ratio: float, k_c: float, k_phi: float, n: float,
    cohesion: float, friction_angle_rad: float, K: float,
) -> float:
    """Drawbar pull as a function of slip (N)."""
    return track_slip_sinkage(
        weight, track_width, track_length, slip_ratio,
        k_c, k_phi, n, cohesion, friction_angle_rad, K,
    )["drawbar_pull"]


def track_motion_resistance(
    weight: float, track_width: float, track_length: float,
    k_c: float, k_phi: float, n: float,
) -> float:
    """Compaction resistance (N), independent of slip."""
    state = SlipSinkageState(
        weight=weight, track_width=track_width, track_length=track_length,
        slip_ratio=0.01, k_c=k_c, k_phi=k_phi, n=n,
        cohesion=0.0, friction_angle_rad=0.0, K=0.01,
    )
    return state.motion_resistance

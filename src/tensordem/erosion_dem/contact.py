"""Contact mechanics of one erodent particle on a ductile wall (DEM-lite).

Two ingredients, both from published contact mechanics with a verifiable
internal structure (no fitted constants):

1. **Normal plastic coefficient of restitution** -- Johnson (1985) /
   Thornton & Ning (1998): a sphere impacting a wall faster than the yield
   velocity ``v_y`` leaves plastically deformed material behind, so the
   rebound speed scales as the fourth root of the impact speed,

       e_n = (v_y / v_n) ** 0.25   for v_n > v_y,   e_n = 1 otherwise.

   The yield velocity follows from the Hertz solution at the yield point,
   ``v_y = (pi/(2m))**2 (2R/(3E*))**2 (1.6 p)**5`` with ``p`` the target flow
   stress.  For sand-on-metal speeds of a few tens of m/s this puts every
   impact deep into the plastic branch -- exactly the regime in which an
   erosion model has anything to remove.

2. **Oblique (tangential) restitution with Coulomb friction** -- the
   rigid-body sliding-impact model: the normal impulse changes the normal
   velocity by ``(1 + e_n) v_n``, so a friction coefficient ``mu`` removes a
   tangential impulse ``mu (1 + e_n) v_n`` from the particle's momentum.  If
   friction can consume the whole tangential slip the contact *sticks* and the
   particle leaves with zero tangential velocity (``beta_t = 0``).

Both laws are used by :mod:`tensordem.erosion_dem.stream`, which marches the
trajectory instead of sampling a distribution of impact angles.  Units: SI.
"""
from __future__ import annotations

import math

__all__ = [
    "HERTZ_PLASTIC_PREFACTOR",
    "yield_velocity_m_s",
    "plastic_normal_restitution",
    "oblique_restitution",
    "JointImpactKinematics",
    "flight_time_to_wall_s",
]

HERTZ_PLASTIC_PREFACTOR = 1.6
"""Mean contact pressure at yield in units of the flow stress (Johnson 1985)."""

HERTZ_YIELD_PREFACTOR = (16.0 / 15.0) * math.pi**2 / 4.0
"""``C = (16/15) (pi**2/4)**(5/2)`` of the derived yield velocity (see below)."""


def yield_velocity_m_s(
    particle_mass_kg: float,
    particle_radius_m: float,
    target_flow_stress_pa: float,
    reduced_modulus_pa: float,
) -> float:
    """Impact speed above which the normal contact stays plastic.

    Derived (not tabulated) from the Hertz solution at the yield point.  With
    ``p_y = 1.6 p`` the mean pressure at yield, the contact radius is
    ``a = pi R p_y / (2 E*)``, the approach ``delta = a**2 / R``, and equating
    the incoming normal kinetic energy to the Hertzian elastic work
    ``(8/15) E* sqrt(R) delta**(5/2)`` gives the closed form ::

        v_y = sqrt( C R**3 p_y**5 / (E*star**4 m) ),
        C = (16/15) (pi**2/4)**(5/2) = 10.20 .

    Sanity: a 150 um sand grain on a 0.5 GPa-hardness aluminium wall yields at
    ``v_y ~ 1 mm/s``, so every sand-jet impact is far inside the plastic branch
    -- which is exactly the regime in which Johnson's quarter-power restitution
    applies and in which erosion happens at all.
    """
    if particle_mass_kg <= 0.0:
        raise ValueError("particle_mass_kg must be > 0")
    if particle_radius_m <= 0.0:
        raise ValueError("particle_radius_m must be > 0")
    if target_flow_stress_pa <= 0.0:
        raise ValueError("target_flow_stress_pa must be > 0")
    if reduced_modulus_pa <= 0.0:
        raise ValueError("reduced_modulus_pa must be > 0")
    p_y = HERTZ_PLASTIC_PREFACTOR * target_flow_stress_pa
    c = (16.0 / 15.0) * (math.pi**2 / 4.0) ** 2.5
    return math.sqrt(
        c * particle_radius_m**3 * p_y**5 / (reduced_modulus_pa**4 * particle_mass_kg)
    )


def plastic_normal_restitution(
    normal_speed_m_s: float,
    yield_velocity_m_s: float,
) -> float:
    """Johnson's plastic normal coefficient of restitution, clipped to [0, 1].

    ``e_n = (v_y / v_n) ** (1/4)`` above the yield speed and ``1`` below it.
    The quarter power is the signature of the plastic branch: the *fraction* of
    normal energy dissipated grows with impact speed, ``1 - e_n**2 =
    1 - sqrt(v_y/v_n)``, which is what makes the aggregate erosion exponent of
    a stream differ from the exponent of the erosion law it is built from
    (measured in criterion C2).
    """
    v_n = float(normal_speed_m_s)
    v_y = float(yield_velocity_m_s)
    if v_n < 0.0 or not math.isfinite(v_n):
        raise ValueError("normal_speed_m_s must be finite and >= 0")
    if not v_y > 0.0 or not math.isfinite(v_y):
        raise ValueError("yield_velocity_m_s must be finite and > 0")
    if v_n <= v_y:
        return 1.0
    return min(1.0, (v_y / v_n) ** 0.25)


def oblique_restitution(
    speed_m_s: float,
    angle_deg: float,
    normal_restitution: float,
    friction_coefficient: float,
) -> dict:
    """Reflected velocity components of a sliding/sticking oblique impact.

    ``angle_deg`` is the *measured* impact angle (0 = grazing, 90 = normal).
    Returns reflected normal/tangential speeds, the tangential restitution
    ``beta_t = v_t_out / v_t_in``, the sticking flag and the per-unit-mass
    kinetic-energy split.  Energy never increases (``e_n <= 1``,
    ``beta_t <= 1``), which the tests assert as a negative control.
    """
    v = float(speed_m_s)
    a = float(angle_deg)
    e_n = float(normal_restitution)
    mu = float(friction_coefficient)
    if v < 0.0 or not math.isfinite(v):
        raise ValueError("speed_m_s must be finite and >= 0")
    if not 0.0 <= a <= 90.0:
        raise ValueError("angle_deg must be within [0, 90]")
    if not 0.0 <= e_n <= 1.0:
        raise ValueError("normal_restitution must be within [0, 1]")
    if mu < 0.0 or not math.isfinite(mu):
        raise ValueError("friction_coefficient must be finite and >= 0")
    s, c = math.sin(math.radians(a)), math.cos(math.radians(a))
    v_n, v_t = v * s, v * c
    v_n_out = e_n * v_n
    v_t_out = max(0.0, v_t - mu * (1.0 + e_n) * v_n)
    ke_in = 0.5 * v * v
    ke_out = 0.5 * (v_n_out * v_n_out + v_t_out * v_t_out)
    beta_t = 1.0 if v_t <= 0.0 else v_t_out / v_t
    return {
        "speed_in_m_s": v,
        "angle_in_deg": a,
        "normal_speed_in_m_s": v_n,
        "tangential_speed_in_m_s": v_t,
        "normal_speed_out_m_s": v_n_out,
        "tangential_speed_out_m_s": v_t_out,
        "normal_restitution": e_n,
        "tangential_restitution": beta_t,
        "sticking": bool(v_t_out <= 0.0),
        "kinetic_energy_in_per_kg": ke_in,
        "kinetic_energy_out_per_kg": ke_out,
        "dissipated_fraction": 0.0 if ke_in <= 0.0 else 1.0 - ke_out / ke_in,
        "elastic_energy_conserved": bool(ke_out <= ke_in * (1.0 + 1e-15)),
    }


class JointImpactKinematics:
    """Algebraically exact ballistics of a particle fired at a flat wall.

    The particle leaves a nozzle at ``z = height`` with speed ``v`` at an angle
    ``theta`` *below* the horizontal, flies under gravity and strikes the plane
    ``z = 0``.  The impact angle is **not** imposed: flight time and the
    measured ``arctan(v_n / v_t)`` come out of the parabola, so the stream in
    :mod:`tensordem.erosion_dem.stream` reads contact kinematics out of the
    trajectory instead of sampling ``alpha`` from a distribution.
    """

    __slots__ = ("height_m", "gravity_m_s2")

    def __init__(self, height_m: float, gravity_m_s2: float = 9.81) -> None:
        if not height_m > 0.0:
            raise ValueError("height_m must be > 0")
        if not gravity_m_s2 > 0.0:
            raise ValueError("gravity_m_s2 must be > 0")
        self.height_m = float(height_m)
        self.gravity_m_s2 = float(gravity_m_s2)

    def flight_time_s(self, speed_m_s: float, launch_angle_deg: float) -> float:
        """Positive root of ``h - v_z t - g t^2/2 = 0`` (closed form)."""
        v = float(speed_m_s)
        a = math.radians(float(launch_angle_deg))
        vz = -v * math.sin(a)          # downward component is negative
        g = self.gravity_m_s2
        disc = vz * vz + 2.0 * g * self.height_m
        # root of (g/2) t^2 - vz t - h = 0 with vz < 0 for a downward launch
        return (vz + math.sqrt(disc)) / g

    def impact(self, speed_m_s: float, launch_angle_deg: float) -> dict:
        """Measured kinematics of the first touch-down."""
        v = float(speed_m_s)
        a = math.radians(float(launch_angle_deg))
        t = self.flight_time_s(v, launch_angle_deg)
        vx = v * math.cos(a)
        vz = -v * math.sin(a) - self.gravity_m_s2 * t
        speed = math.hypot(vx, vz)
        angle = math.degrees(math.atan2(-vz, vx)) if speed > 0.0 else 0.0
        return {
            "flight_time_s": t,
            "impact_speed_m_s": speed,
            "impact_angle_deg": max(0.0, min(90.0, angle)),
            "horizontal_speed_m_s": vx,
        }


def flight_time_to_wall_s(
    speed_m_s: float,
    launch_angle_deg: float,
    height_m: float,
    gravity_m_s2: float = 9.81,
) -> float:
    """Functional wrapper of :meth:`JointImpactKinematics.flight_time_s`."""
    return JointImpactKinematics(height_m, gravity_m_s2).flight_time_s(
        speed_m_s, launch_angle_deg
    )
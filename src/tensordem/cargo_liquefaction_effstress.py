"""Minimal closed chain for bulk-cargo liquefaction (campaign IN60-N09).

The failed predecessor (IN39-N08) tried to close a full 6-DOF hold/ship
coupling around the liquefied cargo and could not be judged by machine.  This
module deliberately rebuilds only the *judgeable* link of the chain, one
physically independent factor per stage:

``rouault_pendular_force`` / ``rouault_pendular_energy``
    Pendular liquid-bridge force between two equal spheres with the smooth
    approximation

        F(h) = 2 pi gamma R cos(theta) / (1 + a S + b S^2),
        S    = h / (2 a_eq),   a_eq = (3 V / 4 pi)^(1/3),

    ``a = 1.05``, ``b = 2.5`` (Willett et al. 2000 smooth-fit constants, the
    form adopted as the "Rouault" smooth approximation in wet-granular DEM).
    ``S`` is dimensionless and is a function of ``(V/R^3, h/R)`` only, as
    required by the campaign specification.  The potential ``U(h)`` that
    generates ``F = -dU/dh`` is integrated in closed form so the DEM kernel is
    provably conservative (energy audit G4).

``CargoMatrix``
    Microstructural bookkeeping of the cargo: porosity, particle radius,
    saturation ``S_r = MR / FMP``, liquid volume per bridge from the water
    balance, Rumpf's tensile-strength formula for the capillary (apparent)
    cohesion, and the undrained contractancy that produces the excess pore
    pressure ratio ``r_u`` near saturation.

``effective_friction_angle``
    Mohr-Coulomb reduction with Terzaghi effective stress,
    ``lambda = 1 - r_u`` applied to the *normal* stress, and the same factor
    applied to the matric-suction cohesion (a continuous, positively
    pressurised pore fluid cannot hold menisci):

        tan phi_eff = (1 - r_u) [ c(S_r) / sigma_n + tan phi_0 ].

``GranularBed``
    Quasi-static slope-limited bed relaxation (a sandpile automaton with
    Coulomb yield, depth dependent cohesion and hysteretic residual angle).
    It replaces the removed 6-DOF coupling by the one thing the coupling was
    supposed to deliver: the transverse cargo shift ``delta`` under a
    prescribed roll history.  It conserves cargo volume exactly and is
    timestep convergent by construction (G5).

``gz_area_loss`` / ``assess_stability``
    Transverse shift -> heeling moment -> residual list and GZ area loss.

Everything is SI.  Sign convention: attraction is positive, ``h`` is the
surface-to-surface gap, ``theta`` is the contact angle measured inside the
liquid.  ``x`` runs to starboard (positive) in the hold frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from tensordem.cargo_liquefaction import GRAVITY

WATER_SURFACE_TENSION = 0.072   # N/m at 20 degC
WATER_DENSITY = 1000.0          # kg/m^3

#: Willett et al. (2000) smooth-fit constants of the Rouault approximation.
ROUAULT_A = 1.05
ROUAULT_B = 2.5

#: (3 / 4 pi)^(1/3): equivalent liquid-sphere radius per unit V^(1/3).
_EQUIV_COEF = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# 1. Pendular liquid bridge: force, potential, rupture
# ---------------------------------------------------------------------------

def equivalent_liquid_radius(volume: float) -> float:
    """Radius of the sphere with the same volume (m)."""
    if volume <= 0.0:
        raise ValueError("liquid volume must be positive")
    return _EQUIV_COEF * volume ** (1.0 / 3.0)


def rouault_separation_number(radius: float, volume: float, gap: float) -> float:
    """Dimensionless separation ``S = h / (2 a_eq)`` of the Rouault form."""
    if radius <= 0.0:
        raise ValueError("particle radius must be positive")
    if gap < 0.0:
        raise ValueError("gap must be non-negative")
    return gap / (2.0 * equivalent_liquid_radius(volume))


def rouault_force_factor(radius: float, volume: float, gap: float) -> float:
    """``f(V/R^3, h/R) = 1 / (1 + a S + b S^2)`` (dimensionless, in (0, 1])."""
    s = rouault_separation_number(radius, volume, gap)
    return 1.0 / (1.0 + ROUAULT_A * s + ROUAULT_B * s * s)


def rouault_pendular_force(
    radius: float,
    gamma: float,
    volume: float,
    gap: float,
    contact_angle_deg: float = 0.0,
) -> float:
    """Pendular-bridge attraction ``F = 2 pi gamma R cos(theta) f(S)`` (N).

    Limits (asserted in the tests): ``h -> 0`` gives the exact Derjaguin value
    ``2 pi gamma R cos(theta)``; ``theta >= 90 deg`` gives no pendular bridge
    (non-wetting, ``F = 0``); ``F`` decreases monotonically in ``h`` and
    increases monotonically in ``V``.
    """
    if radius <= 0.0 or gamma < 0.0 or volume <= 0.0:
        raise ValueError("R > 0, gamma >= 0 and V > 0 are required")
    cos_t = math.cos(math.radians(contact_angle_deg))
    if cos_t <= 0.0:
        return 0.0
    return (
        2.0
        * math.pi
        * gamma
        * radius
        * cos_t
        * rouault_force_factor(radius, volume, gap)
    )


def delenne_pendular_force(
    radius: float,
    gamma: float,
    volume: float,
    gap: float,
    contact_angle_deg: float = 0.0,
) -> float:
    """Rabinovich/Delenne closed form, kept only as a comparison kernel.

    ``F = 2 pi gamma R cos(theta) / (1 + h / (2 V^(1/3)))``.  The only
    difference from :func:`rouault_pendular_force` is the normalisation of the
    separation (``V^(1/3)`` vs the equivalent-sphere radius) and the quadratic
    term; both share the exact Derjaguin limit.
    """
    if radius <= 0.0 or gamma < 0.0 or volume <= 0.0:
        raise ValueError("R > 0, gamma >= 0 and V > 0 are required")
    cos_t = math.cos(math.radians(contact_angle_deg))
    if cos_t <= 0.0:
        return 0.0
    return 2.0 * math.pi * gamma * radius * cos_t / (
        1.0 + gap / (2.0 * volume ** (1.0 / 3.0))
    )


def _energy_quadrature(s: float) -> float:
    """``int_s^inf dS / (1 + a S + b S^2)`` in closed form."""
    disc = 4.0 * ROUAULT_B - ROUAULT_A * ROUAULT_A
    root = math.sqrt(disc)
    return (2.0 / root) * (
        0.5 * math.pi - math.atan((2.0 * ROUAULT_B * s + ROUAULT_A) / root)
    )


def rouault_pendular_energy(
    radius: float,
    gamma: float,
    volume: float,
    gap: float,
    contact_angle_deg: float = 0.0,
) -> float:
    """Capillary potential ``U(h) = int_h^inf F(h') dh'`` (J), closed form.

    ``U`` is the reversible work needed to separate the two spheres from the
    gap ``h`` to infinity, so ``F(h) = -dU/dh`` exactly.  ``U(inf) = 0`` by
    construction and ``U(0)`` is the total capillary binding energy.
    """
    f0 = rouault_pendular_force(radius, gamma, volume, 0.0, contact_angle_deg)
    if f0 == 0.0:
        return 0.0
    s = rouault_separation_number(radius, volume, gap)
    return f0 * 2.0 * equivalent_liquid_radius(volume) * _energy_quadrature(s)


def bridge_rupture_gap(volume: float, contact_angle_deg: float = 0.0) -> float:
    """Rupture separation estimate ``h_r = V^(1/3) (0.5 theta^2 + 0.1)`` (Lian)."""
    theta = math.radians(contact_angle_deg)
    return volume ** (1.0 / 3.0) * (0.5 * theta * theta + 0.1)


def force_energy_derivative_residual(
    radius: float,
    gamma: float,
    volume: float,
    gap: float,
    contact_angle_deg: float = 0.0,
    step: float = 1.0e-6,
) -> float:
    """Relative residual of ``-dU/dh`` against ``F`` by central differences."""
    d = step * max(gap, equivalent_liquid_radius(volume))
    u_plus = rouault_pendular_energy(radius, gamma, volume, gap + d, contact_angle_deg)
    u_minus = rouault_pendular_energy(radius, gamma, volume, max(gap - d, 0.0), contact_angle_deg)
    h_span = (gap + d) - max(gap - d, 0.0)
    force_from_energy = -(u_plus - u_minus) / h_span
    force = rouault_pendular_force(radius, gamma, volume, gap, contact_angle_deg)
    if force == 0.0:
        return 0.0
    return abs(force_from_energy - force) / abs(force)


def separate_pair_energy_budget(
    mass: float,
    radius: float,
    gamma: float,
    volume: float,
    gap0: float,
    velocity0: float,
    dt: float,
    n_steps: int,
    contact_angle_deg: float = 0.0,
) -> dict:
    """Two-particle separation with velocity Verlet: force/energy ledger.

    One particle is fixed, the other (mass ``mass``) starts at ``gap0`` with
    the outward velocity ``velocity0`` and feels only the conservative
    capillary force ``F = -dU/dh`` (no damping, no gravity): the ledger should
    close to integrator accuracy.  Returns the initial/final mechanical energy
    and the accumulated work ``int F dh`` so the residual can be judged (G4).
    """
    reduced_mass = mass
    h = gap0
    v = velocity0
    u0 = rouault_pendular_energy(radius, gamma, volume, h, contact_angle_deg)
    e0 = 0.5 * reduced_mass * v * v + u0
    work = 0.0
    for _ in range(n_steps):
        f_now = rouault_pendular_force(radius, gamma, volume, h, contact_angle_deg)
        a_now = f_now / reduced_mass
        h_half = h + 0.5 * dt * v
        v_half = v + 0.5 * dt * a_now
        h_new = max(h_half + 0.5 * dt * v_half, 0.0)
        f_new = rouault_pendular_force(radius, gamma, volume, h_new, contact_angle_deg)
        v_new = v_half + 0.5 * dt * (f_new / reduced_mass)
        work += 0.5 * (f_now + f_new) * (h_new - h)
        h, v = h_new, v_new
    u1 = rouault_pendular_energy(radius, gamma, volume, h, contact_angle_deg)
    e1 = 0.5 * reduced_mass * v * v + u1
    return {
        "gap_initial": gap0,
        "gap_final": h,
        "energy_initial": e0,
        "energy_final": e1,
        "work_by_capillary": work,
        "potential_change": u0 - u1,
        "energy_drift": e1 - e0,
        "residual_energy": abs(e1 - e0) / max(abs(e0), 1e-30),
        "residual_work_vs_potential": abs(work - (u0 - u1)) / max(abs(u0 - u1), 1e-30),
    }


# ---------------------------------------------------------------------------
# 2. Cargo microstructure: saturation, bridge volume, capillary cohesion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CargoMatrix:
    """Microstructural state of a granular bulk cargo (defaults: quartz sand).

    ``saturation_pendular`` is the saturation at which every coordination site
    carries a bridge (end of the *pendular* regime); ``saturation_onset`` is
    the saturation above which the loose matrix is contractive enough to build
    excess pore pressure under undrained shear.  Both are declared literature
    anchors (see ``docs/IN60_N09_cargo_liquefaction_effstress.md`` §5) and are
    the only two saturation anchors of the model.
    """

    rho_s: float = 2650.0
    rho_w: float = WATER_DENSITY
    porosity: float = 0.40
    particle_radius: float = 1.0e-4
    surface_tension: float = WATER_SURFACE_TENSION
    contact_angle_deg: float = 0.0
    coordination: float = 6.0
    saturation_pendular: float = 0.88
    saturation_onset: float = 0.82
    phi_dry_deg: float = 32.0
    sigma_n_ref: float = 10.0e3
    #: exponent of the contractancy ramp between onset and full saturation
    pore_pressure_exponent: float = 1.5

    # -- derived ---------------------------------------------------------
    @property
    def bulk_density(self) -> float:
        return (1.0 - self.porosity) * self.rho_s

    @property
    def full_moisture_point(self) -> float:
        """Mass moisture ratio at full saturation (``MR / S_r``)."""
        return self.porosity * self.rho_w / ((1.0 - self.porosity) * self.rho_s)

    @property
    def pore_volume_per_particle(self) -> float:
        vp = 4.0 / 3.0 * math.pi * self.particle_radius ** 3
        return self.porosity / (1.0 - self.porosity) * vp

    @property
    def water_volume_per_particle(self) -> float:
        return self.pore_volume_per_particle

    def saturation(self, moisture_ratio: float) -> float:
        """``S_r = MR / FMP`` clipped to [0, 1]."""
        if moisture_ratio < 0.0:
            raise ValueError("moisture ratio must be non-negative")
        return min(moisture_ratio / self.full_moisture_point, 1.0)

    def bridge_count_fraction(self, moisture_ratio: float) -> float:
        """Fraction of the ``z`` coordination sites carrying a liquid bridge."""
        return min(self.saturation(moisture_ratio) / self.saturation_pendular, 1.0)

    def liquid_volume_per_bridge(self, moisture_ratio: float) -> float:
        """Liquid volume of one pendular bridge from the water balance (m^3).

        ``z f`` bridges per particle at saturation ``S_r`` with ``f`` the
        bridged coordination fraction give ``V_b = 2 S_r V_pore / (z f)``,
        i.e. a *step-independent* bridge volume that grows only once the
        coordination is fully bridged (``f = 1``).
        """
        s_r = self.saturation(moisture_ratio)
        f = self.bridge_count_fraction(moisture_ratio)
        if s_r <= 0.0 or f <= 0.0:
            return 0.0
        return 2.0 * s_r * self.pore_volume_per_particle / (self.coordination * f)

    def bridge_volume_ratio(self, moisture_ratio: float) -> float:
        """``V_b / R^3`` — the dimensionless bridge volume (input of f)."""
        return self.liquid_volume_per_bridge(moisture_ratio) / self.particle_radius ** 3

    def capillary_force_per_bridge(self, moisture_ratio: float, gap_ratio: float = 0.0) -> float:
        """Bridge force at the mean contact separation ``h = gap_ratio R`` (N)."""
        vb = self.liquid_volume_per_bridge(moisture_ratio)
        if vb <= 0.0:
            return 0.0
        return rouault_pendular_force(
            self.particle_radius,
            self.surface_tension,
            vb,
            gap_ratio * self.particle_radius,
            self.contact_angle_deg,
        )

    def capillary_cohesion(self, moisture_ratio: float, gap_ratio: float = 0.0) -> float:
        """Apparent cohesion from Rumpf's tensile-strength formula (Pa).

        ``c = (9/8) ((1-eps)/eps) f_b F_cap / d^2`` with ``d = 2R`` and ``f_b``
        the bridged coordination fraction.  The suspended fraction is treated
        as cohesionless, which is the standard wet-granular identification.
        """
        f_b = self.bridge_count_fraction(moisture_ratio)
        if f_b <= 0.0:
            return 0.0
        f_cap = self.capillary_force_per_bridge(moisture_ratio, gap_ratio)
        d = 2.0 * self.particle_radius
        return (9.0 / 8.0) * ((1.0 - self.porosity) / self.porosity) * f_b * f_cap / (d * d)

    def undrained_pore_pressure_ratio(self, moisture_ratio: float) -> float:
        """Shear-induced ``r_u`` of the loose matrix (0 below onset, -> 1 at FMP).

        The excess pore pressure of an undrained contractive matrix grows with
        the initial saturation; the declared anchors are ``r_u = 0`` for
        ``S_r <= saturation_onset`` (partially saturated sand dilates and the
        menisci buffer the pore pressure) and ``r_u -> 1`` at full saturation
        (flow liquefaction of the loose matrix).
        """
        return contractancy_fraction(
            self.saturation(moisture_ratio),
            self.saturation_onset,
            self.pore_pressure_exponent,
        )

    def effective_friction_angle(
        self,
        moisture_ratio: float,
        pore_pressure_ratio: Optional[float] = None,
        sigma_n: Optional[float] = None,
    ) -> dict:
        """Apparent (mobilised) friction angle at moisture ratio ``MR``.

        With ``sigma'_n = (1 - r_u) sigma_n`` and the suction cohesion scaled by
        the same factor (a continuous pressurised pore fluid cannot hold
        menisci)::

            tan phi_eff = (1 - r_u) [ c(MR) / sigma_n + tan phi_0 ]

        ``pore_pressure_ratio`` overrides the monotonic-test value (this is how
        the cyclic history and the ``r_u -> 1`` negative control enter).
        """
        sigma_n = self.sigma_n_ref if sigma_n is None else sigma_n
        if sigma_n <= 0.0:
            raise ValueError("sigma_n must be positive")
        r_u = (
            self.undrained_pore_pressure_ratio(moisture_ratio)
            if pore_pressure_ratio is None
            else float(min(max(pore_pressure_ratio, 0.0), 1.0))
        )
        c = self.capillary_cohesion(moisture_ratio)
        tan_phi0 = math.tan(math.radians(self.phi_dry_deg))
        tan_eff = (1.0 - r_u) * (c / sigma_n + tan_phi0)
        return {
            "moisture_ratio": moisture_ratio,
            "saturation": self.saturation(moisture_ratio),
            "saturation_over_fmp": self.saturation(moisture_ratio),
            "pore_pressure_ratio": r_u,
            "cohesion_pa": c,
            "bridge_volume_ratio": self.bridge_volume_ratio(moisture_ratio),
            "tan_phi_eff": tan_eff,
            "phi_eff_deg": math.degrees(math.atan(tan_eff)),
        }


# ---------------------------------------------------------------------------
# 3. Cyclic pore pressure: exposure -> r_u -> strength
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollExcitation:
    """Harmonic roll of the hold frame plus a sustained heel bias."""

    amplitude_deg: float = 10.0
    period_s: float = 8.0
    cog_height_m: float = 5.0
    heel_deg: float = 2.0

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.period_s

    def roll_angle_deg(self, t: float) -> float:
        return self.heel_deg + self.amplitude_deg * math.sin(self.omega * t)

    @property
    def max_horizontal_acceleration(self) -> float:
        phi0 = math.radians(self.amplitude_deg)
        return self.omega ** 2 * self.cog_height_m * phi0

    @property
    def csr(self) -> float:
        """Cyclic stress ratio at the cargo CoG ``omega^2 z phi0 / g``."""
        return self.max_horizontal_acceleration / GRAVITY

    def free_surface_angle_deg(self, t: float) -> float:
        """Quasi-static effective-gravity tilt of a free surface (deg).

        The hold rolls about an axis ``cog_height_m`` below the cargo CoG.  In
        the earth frame the CoG traces ``z_c (sin phi, cos phi)``, so its
        acceleration is ``z_c (phiddot cos phi - phidot^2 sin phi,
        -phiddot sin phi - phidot^2 cos phi)``; the effective gravity
        ``g_eff = g - rddot`` is rotated back into the hold frame and the free
        surface is normal to it.  Rigid-hold (no sloshing) approximation.
        """
        w = self.omega
        phi = math.radians(self.roll_angle_deg(t))
        phi0 = math.radians(self.amplitude_deg)
        phid = w * phi0 * math.cos(w * t)
        phidd = -w * w * phi0 * math.sin(w * t)
        ax = -self.cog_height_m * (phidd * math.cos(phi) - phid ** 2 * math.sin(phi))
        az = -self.cog_height_m * (-phidd * math.sin(phi) - phid ** 2 * math.cos(phi))
        gx = -ax
        gz = -GRAVITY - az
        c, s = math.cos(phi), math.sin(phi)
        g_hx = c * gx + s * gz          # rotate by -phi
        g_hz = -s * gx + c * gz
        return math.degrees(math.atan2(g_hx, -g_hz))


def cycles_to_liquefaction(
    csr: float,
    crr: float = 0.15,
    n_ref: float = 10.0,
    exponent: float = 0.25,
) -> float:
    """Seed-type resistance curve ``CSR = CRR (N_L / N_ref)^(-k)``."""
    if csr <= 0.0 or crr <= 0.0:
        raise ValueError("csr and crr must be positive")
    return n_ref * (csr / crr) ** (-1.0 / exponent)


def contractancy_fraction(saturation: float, saturation_onset: float = 0.84,
                          exponent: float = 1.5) -> float:
    """Contractancy limit of the excess pore pressure ratio, ``r_u^max(S_r)``.

    ``0`` below the onset saturation (partially saturated matrix dilates and the
    menisci buffer the pore pressure), ramping to ``1`` at full saturation.
    """
    s_r = min(max(saturation, 0.0), 1.0)
    if s_r <= saturation_onset:
        return 0.0
    ramp = (s_r - saturation_onset) / (1.0 - saturation_onset)
    return float(min(ramp ** exponent, 1.0))


def cyclic_resistance_ratio(
    saturation: float,
    crr_saturated: float = 0.15,
    desaturation_hardening: float = 1.0,
) -> float:
    """CRR grows as the pore space desaturates (air buffers the pore pressure).

    Linear hardening anchor: ``CRR(S_r) = CRR_sat [1 + kappa (1 - S_r)]``;
    ``kappa = 1`` doubles the resistance at zero saturation.
    """
    s_r = min(max(saturation, 0.0), 1.0)
    return crr_saturated * (1.0 + desaturation_hardening * (1.0 - s_r))


def cyclic_pore_pressure_ratio(
    cycles: float,
    csr: float,
    saturation: float,
    *,
    crr_saturated: float = 0.15,
    desaturation_hardening: float = 1.0,
    theta: float = 0.7,
    n_ref: float = 10.0,
    exponent: float = 0.25,
    saturation_onset: float = 0.84,
) -> float:
    """``r_u(N) = (2/pi) arcsin[(N/N_L)^(1/(2 theta))]``, clipped at 1."""
    if cycles <= 0.0:
        return 0.0
    crr = cyclic_resistance_ratio(saturation, crr_saturated, desaturation_hardening)
    n_l = cycles_to_liquefaction(csr, crr, n_ref=n_ref, exponent=exponent)
    ratio = min((cycles / n_l) ** (1.0 / (2.0 * theta)), 1.0)
    seed = float(2.0 / math.pi * math.asin(ratio))
    # the matrix can only carry as much excess pore pressure as its
    # contractancy allows: a partially saturated matrix (below the onset) has
    # no excess pore pressure at all, a saturated loose matrix liquefies fully.
    return seed * contractancy_fraction(saturation, saturation_onset)


# ---------------------------------------------------------------------------
# 4. Bed relaxation: transverse cargo shift under roll
# ---------------------------------------------------------------------------

@dataclass
class GranularBed:
    """Column-height bed in the hold frame with Coulomb-yield relaxation.

    ``h[i]`` is the cargo surface height in column ``i``; cargo volume
    ``sum(h) dx`` is conserved exactly by every relaxation step.  Stability is
    judged in the earth frame: the local slope angle minus the free-surface
    angle ``psi`` of the effective gravity field must not exceed the friction
    angle.  On failure the material flows downslope until the local slope
    returns to the *residual* angle ``phi_res < phi_fail``; this hysteresis is
    what makes the transport irreversible across a roll reversal.
    """

    heights: np.ndarray
    beam: float
    centre_of_gravity_x: float = 0.0
    initial_centre_of_gravity_x: float = 0.0
    volume_moved: float = 0.0

    @classmethod
    def rectangular(cls, beam: float, fill_height: float, nx: int = 96) -> "GranularBed":
        if beam <= 0.0 or fill_height <= 0.0:
            raise ValueError("beam and fill height must be positive")
        if nx < 8:
            raise ValueError("need at least 8 columns")
        bed = cls(heights=np.full(nx, fill_height, dtype=float), beam=beam)
        bed.centre_of_gravity_x = bed.current_centre_of_gravity_x()
        bed.initial_centre_of_gravity_x = bed.centre_of_gravity_x
        return bed

    @property
    def nx(self) -> int:
        return self.heights.size

    @property
    def dx(self) -> float:
        return self.beam / self.nx

    @property
    def volume(self) -> float:
        return float(self.heights.sum() * self.dx)

    def centre_x(self) -> np.ndarray:
        """Column centres (m), x = 0 at the hold centreline."""
        return (np.arange(self.nx) + 0.5) * self.dx - 0.5 * self.beam

    def current_centre_of_gravity_x(self) -> float:
        v = self.heights * self.dx
        total = v.sum()
        if total <= 0.0:
            return 0.0
        return float((v * self.centre_x()).sum() / total)

    def shift(self) -> float:
        """Transverse CoG shift relative to the initial (level) bed (m)."""
        return self.current_centre_of_gravity_x() - self.initial_centre_of_gravity_x

    def surface_angles_deg(self) -> np.ndarray:
        """Earth-frame slope angle of each segment, hold frame included."""
        slope = (self.heights[1:] - self.heights[:-1]) / self.dx
        return np.degrees(np.arctan(slope))

    def admissible_angles_deg(self, cohesion_pa: float, rho_bulk: float,
                              friction_deg: float) -> np.ndarray:
        """Depth dependent Coulomb limit ``atan(tan phi + c / (rho g z))``.

        The overburden of column ``i`` is ``rho_bulk g h_i``; cohesion therefore
        steepens the surface where the cargo is shallow (heap crest) and is
        negligible at depth.
        """
        z = np.maximum(self.heights, 1e-6)
        tan_f = math.tan(math.radians(friction_deg)) + cohesion_pa / (rho_bulk * GRAVITY * z)
        return np.degrees(np.arctan(tan_f))

    def relax(
        self,
        free_surface_angle_deg: float,
        friction_deg: float,
        *,
        cohesion_pa: float = 0.0,
        rho_bulk: float = 1590.0,
        residual_drop_deg: float = 4.0,
        max_sweeps: int = 400,
        gain: float = 0.35,
        tol_deg: float = 5.0e-2,
    ) -> float:
        """Quasi-static projection of the surface onto the Coulomb-admissible set.

        The surface is judged in the *earth* frame: the segment angle is
        ``A = atan(dh/dx) - psi`` where ``psi`` is the effective-gravity tilt of
        the hold frame.  Segments with ``|A|`` above the friction angle are
        unstable; material flows downslope until ``|A|`` reaches the residual
        angle ``phi_adm - residual_drop`` (``phi_adm = atan(tan phi +
        c/(rho g h))``, depth dependent because cohesion is an added shear
        strength against the local overburden).  A parallel (Jacobi) update is
        iterated to equilibrium — the bed response is quasi-static, so the
        residual shift depends on the *strength history* (``dt``) and not on an
        extra flow-rate coefficient.  Mass is conserved to machine precision
        (paired +/- increments); returns the volume moved (m^3 per m width).
        """
        col_lim = self.admissible_angles_deg(cohesion_pa, rho_bulk, friction_deg)
        allowed = 0.5 * (col_lim[:-1] + col_lim[1:])
        target = np.maximum(allowed - residual_drop_deg, 0.0)
        dx = self.dx
        h = self.heights
        moved = 0.0
        for _ in range(max_sweeps):
            slope = (h[1:] - h[:-1]) / dx
            ang = np.degrees(np.arctan(slope)) - free_surface_angle_deg
            pos = np.maximum(ang - target, 0.0)     # too steep rising to +x
            neg = np.maximum(-ang - target, 0.0)    # too steep falling to +x
            if max(float(pos.max()), float(neg.max())) <= tol_deg:
                break
            dh_pos = 0.5 * gain * np.radians(pos) * dx
            dh_neg = 0.5 * gain * np.radians(neg) * dx
            dh_pos = np.minimum(dh_pos, 0.15 * np.maximum(h[1:], 0.0))
            dh_neg = np.minimum(dh_neg, 0.15 * np.maximum(h[:-1], 0.0))
            h[:-1] += dh_pos - dh_neg
            h[1:] += -dh_pos + dh_neg
            moved += float((dh_pos + dh_neg).sum() * dx)
        self.volume_moved += moved
        self.centre_of_gravity_x = self.current_centre_of_gravity_x()
        return moved


@dataclass(frozen=True)
class ShiftResult:
    """Outcome of a roll history applied to a bed."""

    moisture_ratio: float
    saturation: float
    shift_final_m: float
    shift_max_abs_m: float
    shift_cycle_mean_m: float
    shift_cycle_amplitude_m: float
    phi_eff_deg_initial: float
    phi_eff_deg_final: float
    pore_pressure_ratio_final: float
    time_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shift_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi_eff_history: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self) -> dict:
        return {
            "moisture_ratio": self.moisture_ratio,
            "saturation": self.saturation,
            "shift_final_m": self.shift_final_m,
            "shift_max_abs_m": self.shift_max_abs_m,
            "shift_cycle_mean_m": self.shift_cycle_mean_m,
            "shift_cycle_amplitude_m": self.shift_cycle_amplitude_m,
            "phi_eff_deg_initial": self.phi_eff_deg_initial,
            "phi_eff_deg_final": self.phi_eff_deg_final,
            "pore_pressure_ratio_final": self.pore_pressure_ratio_final,
        }


def simulate_roll_shift(
    matrix: CargoMatrix,
    loading: RollExcitation,
    moisture_ratio: float,
    *,
    fill_height: float = 3.0,
    beam: float = 15.0,
    cycles: int = 200,
    dt: float = 0.25,
    nx: int = 96,
    sigma_n: Optional[float] = None,
    crr_saturated: float = 0.15,
    desaturation_hardening: float = 1.0,
    residual_drop_deg: float = 4.0,
    bed: Optional[GranularBed] = None,
    record: bool = True,
) -> ShiftResult:
    """Roll the hold for ``cycles`` periods and track the cargo CoG shift.

    The strength at time ``t`` is the *degraded* strength: the cyclic pore
    pressure ratio ``r_u(N(t))`` is fed into the effective-stress reduction, so
    the bed weakens as the storm proceeds (undrained, no dissipation).
    """
    sigma_n = matrix.sigma_n_ref if sigma_n is None else sigma_n
    bed = GranularBed.rectangular(beam, fill_height, nx) if bed is None else bed
    state0 = matrix.effective_friction_angle(moisture_ratio, 0.0, sigma_n)
    duration = cycles * loading.period_s
    n_steps = max(int(round(duration / dt)), 1)
    times = np.empty(n_steps + 1)
    shifts = np.empty(n_steps + 1)
    phis = np.empty(n_steps + 1)
    r_u_final = 0.0
    for k in range(n_steps + 1):
        t = k * dt
        n_cycles = t / loading.period_s
        r_u = cyclic_pore_pressure_ratio(
            n_cycles,
            loading.csr,
            matrix.saturation(moisture_ratio),
            crr_saturated=crr_saturated,
            desaturation_hardening=desaturation_hardening,
            saturation_onset=matrix.saturation_onset,
        )
        r_u_final = r_u
        state = matrix.effective_friction_angle(moisture_ratio, r_u, sigma_n)
        psi = loading.free_surface_angle_deg(t)
        bed.relax(
            psi,
            state["phi_eff_deg"],
            # phi_eff already embeds c/(1-r_u)sigma_n at the reference normal
            # stress, so no extra depth-dependent cohesion is added here
            cohesion_pa=0.0,
            rho_bulk=matrix.bulk_density,
            residual_drop_deg=residual_drop_deg,
        )
        if record:
            times[k] = t
            shifts[k] = bed.shift()
            phis[k] = state["phi_eff_deg"]
    shift_hist = shifts if record else np.zeros(0)
    # last full period statistics: the cargo CoG offset is only meaningful as a
    # cycle statistic (a liquefied bed returns to level whenever the hold is
    # upright), so the residual offset is the period mean and the dynamic
    # excursion is the half peak-to-peak of the same period.
    if record and n_steps > 0:
        per_step = duration / n_steps
        n_last = max(int(round(loading.period_s / per_step)), 1)
        tail = shifts[-n_last - 1:]
        cycle_mean = float(np.mean(tail))
        cycle_amp = float(0.5 * (np.max(tail) - np.min(tail)))
    else:
        cycle_mean = bed.shift()
        cycle_amp = 0.0
    return ShiftResult(
        moisture_ratio=moisture_ratio,
        saturation=matrix.saturation(moisture_ratio),
        shift_final_m=bed.shift(),
        shift_max_abs_m=float(np.max(np.abs(shift_hist))) if shift_hist.size else abs(bed.shift()),
        shift_cycle_mean_m=cycle_mean,
        shift_cycle_amplitude_m=cycle_amp,
        phi_eff_deg_initial=state0["phi_eff_deg"],
        phi_eff_deg_final=phis[-1] if record else float("nan"),
        pore_pressure_ratio_final=r_u_final,
        time_s=times if record else np.zeros(0),
        shift_history=shift_hist,
        phi_eff_history=phis if record else np.zeros(0),
    )


def critical_roll_amplitude(
    matrix: CargoMatrix,
    moisture_ratio: float,
    *,
    shift_limit_m: float = 0.25,
    cycles: int = 40,
    period_s: float = 8.0,
    cog_height_m: float = 5.0,
    heel_deg: float = 2.0,
    amplitude_bounds_deg: tuple[float, float] = (1.0, 45.0),
    tol_deg: float = 0.25,
    **kwargs,
) -> dict:
    """Smallest roll amplitude whose residual shift exceeds ``shift_limit_m``.

    Bisection on ``amplitude_deg``; monotonicity of ``shift(amplitude)`` is a
    physical assumption of the criterion (a bigger roll cannot transport less),
    so the bracket is checked and reported instead of assumed.
    """
    lo, hi = amplitude_bounds_deg

    def shift_at(amp: float) -> float:
        res = simulate_roll_shift(
            matrix,
            RollExcitation(amplitude_deg=amp, period_s=period_s,
                           cog_height_m=cog_height_m, heel_deg=heel_deg),
            moisture_ratio,
            cycles=cycles,
            record=False,
            **kwargs,
        )
        return abs(res.shift_cycle_mean_m)

    s_lo, s_hi = shift_at(lo), shift_at(hi)
    if s_hi <= shift_limit_m:
        return {"critical_amplitude_deg": float("nan"), "status": "not_exceeded",
                "shift_at_bounds": (s_lo, s_hi)}
    if s_lo >= shift_limit_m:
        return {"critical_amplitude_deg": lo, "status": "exceeded_at_lower_bound",
                "shift_at_bounds": (s_lo, s_hi)}
    a_lo, a_hi = lo, hi
    while a_hi - a_lo > tol_deg:
        mid = 0.5 * (a_lo + a_hi)
        if shift_at(mid) > shift_limit_m:
            a_hi = mid
        else:
            a_lo = mid
    return {
        "critical_amplitude_deg": 0.5 * (a_lo + a_hi),
        "status": "bracketed",
        "shift_at_bounds": (s_lo, s_hi),
        "shift_at_critical_m": shift_at(0.5 * (a_lo + a_hi)),
    }


# ---------------------------------------------------------------------------
# 5. Stability: residual heel, GZ area loss, margins
# ---------------------------------------------------------------------------

def gz_area_loss(
    gm_m: float,
    shift_m: float,
    cargo_mass_t: float,
    displacement_t: float,
    *,
    theta_max_deg: float = 40.0,
    n_theta: int = 2001,
) -> dict:
    """GZ area loss and residual heel produced by a transverse CoG shift.

    The shift moves the ship CoG transversely by ``dy = m_cargo delta / Delta``
    (metres, independent of the vertical position at small angles), so the
    righting arm loses a cosine term::

        GZ_eff(theta) = GM sin(theta) - dy cos(theta)

    (wall-sided hull, small-angle form).  The ``0``-to-``theta_max`` area ratio
    is reported together with the residual list angle ``atan(dy / GM)`` and the
    IMO 2008 IS Code 30 deg criterion ``GZ(30) >= 0.20 m``.
    """
    if displacement_t <= 0.0 or gm_m <= 0.0:
        raise ValueError("displacement and GM must be positive")
    dy = cargo_mass_t * shift_m / displacement_t
    theta = np.linspace(0.0, math.radians(theta_max_deg), n_theta)
    gz_intact = gm_m * np.sin(theta)
    gz_shifted = gm_m * np.sin(theta) - dy * np.cos(theta)
    area_intact = float(np.trapezoid(gz_intact, theta))
    area_shifted = float(np.trapezoid(gz_shifted, theta))
    loss = 0.0 if area_intact <= 0.0 else max(0.0, 1.0 - area_shifted / area_intact)
    return {
        "transverse_cog_shift_m": dy,
        "area_intact_m_rad": area_intact,
        "area_shifted_m_rad": area_shifted,
        "gz_area_loss_fraction": loss,
        "gz_area_loss_percent": 100.0 * loss,
        "residual_list_deg": math.degrees(math.atan2(dy, gm_m)),
        "gz_at_30deg_m": float(
            gm_m * math.sin(math.radians(30.0)) - dy * math.cos(math.radians(30.0))
        ),
        "imo_30deg_margin_m": float(
            gm_m * math.sin(math.radians(30.0)) - dy * math.cos(math.radians(30.0)) - 0.20
        ),
    }


def assess_stability(
    matrix: CargoMatrix,
    loading: RollExcitation,
    moisture_ratio: float,
    *,
    gm_m: float = 1.5,
    cargo_mass_t: float = 25_000.0,
    displacement_t: float = 50_000.0,
    **kwargs,
) -> dict:
    """Shift + stability verdict for one moisture ratio and roll history."""
    shift = simulate_roll_shift(matrix, loading, moisture_ratio, **kwargs)
    gz = gz_area_loss(gm_m, shift.shift_final_m, cargo_mass_t, displacement_t)
    out = shift.to_dict()
    out.update(gz)
    out["roll_amplitude_deg"] = loading.amplitude_deg
    out["csr"] = loading.csr
    return out


def effective_stress_reduction(pore_pressure_ratio: float) -> float:
    """Terzaghi factor ``lambda = 1 - r_u`` applied to the normal contact force."""
    return 1.0 - min(max(float(pore_pressure_ratio), 0.0), 1.0)


def apply_effective_stress(
    normal_force: float,
    pore_pressure_ratio: float,
) -> float:
    """Reduce a DEM normal contact force by the excess pore pressure.

    ``F_eff = (1 - r_u) F``; at ``r_u -> 1`` the inter-particle normal force
    (and with it the friction limit ``mu F_eff``) vanishes, which is the
    force-level statement of liquefaction.
    """
    return effective_stress_reduction(pore_pressure_ratio) * normal_force


def moisture_scan(
    matrix: CargoMatrix,
    saturations: Sequence[float],
    *,
    pore_pressure_ratio: Optional[float] = None,
) -> list[dict]:
    """``phi_eff`` versus ``S_r = MR / FMP`` (this is the G3 curve)."""
    rows = []
    for s in saturations:
        mr = s * matrix.full_moisture_point
        row = matrix.effective_friction_angle(mr, pore_pressure_ratio)
        row["saturation"] = s
        rows.append(row)
    return rows
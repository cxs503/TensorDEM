"""Closed-form closures for the melt-water lubricated ice/steel contact.

Physical picture (Bowden & Hughes, *Proc. R. Soc. A* 1939; Bowden & Tabor 1950;
Blok 1937 flash temperature; Greenwood & Williamson 1966; Reynolds squeeze film):

    q_in = mu(V,sigma,T) * sigma * V          friction work -> heat at the interface
    q_in = q_solid + q_latent + q_conv        interface energy balance
    q_solid = (T_i - T_m)/(1/G_hs + h/k_w)    transient half-space + water film
    rho_w L_f dh/dt = q_latent - rho_w L_f * u_drain(h)   film mass balance
    mu = f(Lambda) mu_asp + (1 - f(Lambda)) mu_film,  Lambda = h/sigma_rms
    mu_film = eta(T_f) V/(h sigma)            viscous shear of the water film

Nothing in this module is fitted to the acceptance criteria: every constant is a
material or surface-texture input (see :class:`MaterialPair`), and the
acceptance sweep in :mod:`tensordem.ice_friction_meltfilm.sweep` is run once on
the literature-valued nominal set, with a sensitivity table as the honest
boundary of where the conclusions hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "MaterialPair",
    "effusivity",
    "halfspace_conductance",
    "water_viscosity",
    "gw_area_fraction",
    "greenwood_williamson",
    "reynolds_drainage_velocity",
    "film_shear_friction",
]


# ---------------------------------------------------------------------------
# material data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialPair:
    """Thermophysical data of an ice/steel sliding pair plus the water film.

    Every entry is a handbook value at the melting point unless stated; the
    surface-texture entries (``sigma_rms``, ``r_asp``, ``l_c``) are geometry
    inputs and are swept in the sensitivity study.

    ``mu_asp`` is the *input* asperity (dry / creep-controlled) friction of the
    pair.  It is never fitted: criteria C1/C5 compare against it.
    """

    # ice Ih near 0 degC
    k_ice: float = 2.2          # W/(m K)
    rho_ice: float = 917.0      # kg/m^3
    cp_ice: float = 2050.0      # J/(kg K)
    # hull / rig steel
    k_steel: float = 45.0       # W/(m K)
    rho_steel: float = 7850.0   # kg/m^3
    cp_steel: float = 470.0     # J/(kg K)
    # melt water
    k_water: float = 0.57       # W/(m K)
    rho_water: float = 1000.0   # kg/m^3
    latent_heat: float = 3.34e5  # J/kg
    # contact mechanics / tribology
    mu_asp: float = 0.40        # dry asperity friction of the pair (input)
    sigma_rms: float = 6.0e-8   # combined RMS roughness [m] (run-in, polished pair)
    r_asp: float = 3.0e-5       # asperity contact-spot radius [m]
    l_c: float = 4.0e-3         # sliding contact length (slider size) [m]
    t_melt: float = 0.0         # interface melting temperature [degC]
    h_conv: float = 50.0        # gap convective loss coefficient [W/(m^2 K)]
    eta_scale: float = 1.0      # effective/bulk water viscosity ratio (swept)

    # -- derived -----------------------------------------------------------
    @property
    def effusivity_ice(self) -> float:
        return effusivity(self.k_ice, self.rho_ice, self.cp_ice)

    @property
    def effusivity_steel(self) -> float:
        return effusivity(self.k_steel, self.rho_steel, self.cp_steel)

    @property
    def effusivity_eff(self) -> float:
        """Blok series effusivity of the pair, 1/e = 1/e_ice + 1/e_steel."""
        ei, es = self.effusivity_ice, self.effusivity_steel
        return ei * es / (ei + es)

    @property
    def latent_rho(self) -> float:
        """rho_w * L_f  [J/m^3]  (latent heat per unit melt volume)."""
        return self.rho_water * self.latent_heat

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(
            effusivity_ice=self.effusivity_ice,
            effusivity_steel=self.effusivity_steel,
            effusivity_eff=self.effusivity_eff,
            latent_rho=self.latent_rho,
        )
        return d


def effusivity(k: float, rho: float, cp: float) -> float:
    """Thermal effusivity e = sqrt(k rho cp)  [J/(m^2 K sqrt(s))]."""
    return math.sqrt(k * rho * cp)


def halfspace_conductance(material: MaterialPair, speed: float) -> float:
    """Transient half-space conductance of the sliding pair [W/(m^2 K)].

    A slider that has been in contact for the dwell time ``t_c = l_c/V`` has
    pushed the heat front ``sqrt(alpha t_c)`` into the bodies, so the interface
    conductance into the pair is ``e_eff/sqrt(pi t_c)``.  This is the Blok
    transient solution of the sliding heat source with the two effusivities in
    series; it grows as ``sqrt(V)``, which is what makes slow sliding thermally
    benign (and is the reason V -> 0 has no melt film).
    """
    if speed <= 0.0:
        # No sliding: the contact sits isothermally in the bodies, i.e. an
        # infinite contact time.  Guarded so V -> 0 never divides by zero.
        return 0.0
    t_c = material.l_c / speed
    return material.effusivity_eff / math.sqrt(math.pi * t_c)


def water_viscosity(t_celsius: float) -> float:
    """Dynamic viscosity of (supercooled) water, Vogel-Fulcher-Tammann fit.

    ``log10(eta/Pa s) = -4.5318 - 220.57/(T - 149.39)`` with T in kelvin is the
    standard VFT form for water; it reproduces 1.75e-3 Pa s at 0 degC and rises
    steeply on cooling, which is the physical reason a cold film is *not* a good
    lubricant.
    """
    t_k = max(t_celsius + 273.15, 150.0)
    return 2.414e-5 * 10.0 ** (247.8 / (t_k - 140.0))


def gw_area_fraction(lambda_ratio: float) -> float:
    """Greenwood-Williamson real-area fraction, normalised so it is 1 at 0.

    For a Gaussian asperity-height distribution the load-independent contact
    area fraction is

        A_r/A = 2 * [phi(Lambda) - Lambda (1 - Phi(Lambda))],

    which is ``0.798`` at zero separation.  Since the *dry* limit of this model
    must reproduce the input ``mu_asp`` exactly (criterion C1), the geometric
    fraction is normalised by its zero-separation value.  This is a deliberate,
    documented change of the prefactor only: the shape of A(Lambda) is GW's.
    """
    lam = max(lambda_ratio, 0.0)
    tail = 0.5 * math.erfc(lam / math.sqrt(2.0))
    phi = math.exp(-0.5 * lam * lam) / math.sqrt(2.0 * math.pi)
    raw = 2.0 * (phi - lam * tail)
    raw0 = 2.0 / math.sqrt(2.0 * math.pi)      # = 0.7979
    return max(0.0, min(1.0, raw / raw0))


greenwood_williamson = gw_area_fraction


def reynolds_drainage_velocity(h: float, pressure_real: float,
                               material: MaterialPair,
                               t_interface: float) -> float:
    """Squeeze-film drainage velocity of the melt film [m/s].

    Reynolds' solution for a circular contact patch of radius ``r_asp`` squeezed
    at the real contact pressure:  ``dh/dt = -2 h^3 p /(3 eta r^2)``.  Reported
    positive as a *thinning* rate (water leaving the interface).
    """
    if h <= 0.0:
        return 0.0
    eta = water_viscosity(t_interface) * material.eta_scale
    return 2.0 * h ** 3 * max(pressure_real, 0.0) / (3.0 * eta * material.r_asp ** 2)


def film_shear_friction(h: float, speed: float, sigma: float,
                        t_interface: float, eta_scale: float = 1.0) -> float:
    """Viscous friction coefficient of a film of thickness ``h``.

    ``mu_film = tau/sigma`` with ``tau = eta V/h`` the Couette shear stress of
    the water film.  Diverges as ``h -> 0``, so it is only ever evaluated for
    h > 0 and is blended out by the asperity fraction as the film closes.
    """
    if h <= 0.0 or sigma <= 0.0:
        return 0.0
    return water_viscosity(t_interface) * eta_scale * speed / (h * sigma)
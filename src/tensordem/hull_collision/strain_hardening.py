"""Strain-hardening plastic hinges for ice-region side-shell collision energy.

Where this sits
---------------
:mod:`tensordem.hull_collision.plastic_hinge` idealises the side-shell hinge as
*rigid, perfectly plastic*: the moment stays at ``M_p = sigma_y*Z`` for any
rotation, so a hinge line dissipates ``E = M_p*theta``.  Real hull structural
steel carries a substantial hardening reserve between first yield and rupture
(bilinear tangent modulus ``E_h = 0.005...0.02 * E``), which this module adds.
The extension is built so that the perfectly-plastic (N01) surface is an exact
special case: the hardening contribution is *additive* and vanishes identically
for ``E_h = 0`` (:class:`BilinearHinge` with ``hardening_modulus = 0.0`` returns
0.0 from :meth:`BilinearHinge.moment_increment` and
:meth:`BilinearHinge.energy_increment`, so a layer reduces bit-for-bit to
:class:`tensordem.hull_collision.resistance.SideShellLayer`).

Mechanics
---------
Section: plating of thickness ``t`` (half-depth ``c = t/2``), width ``b = 1`` per
unit plate width.  Bilinear material::

    sigma(eps) = E*eps                            |eps| <= eps_y
    sigma(eps) = sgn(eps)*(sigma_y + E_h*(|eps|-eps_y))     otherwise

with ``eps_y = sigma_y/E``.  Under curvature ``kappa`` the strain is
``kappa*y``, so the yielded zone is ``|y| > a`` with the elastic-core half depth
``a = eps_y/kappa``.  The section integral gives (exact, no series expansion)::

    M(kappa) = sigma_y*b*c^2 - sigma_y*b*a^2/3
               + 2*b*E_h*kappa*[c^3/3 - a*c^2/2 + a^3/6]          (a <= c)

The first line is the perfectly-plastic section moment *with the same elastic
core*; the second line is precisely the part carried by the hardening branch of
the constitutive law.  This module therefore defines the hardening increment as

    dM_h(kappa) = M_bilinear(kappa) - M_perfectly_plastic(kappa)
                = b*E_h*[(2*c^3/3)*kappa - eps_y*c^2 + eps_y^3/(3*kappa^2)]

which is *identically zero* for ``E_h = 0`` and asymptotes to ``E_h*I*kappa``
(``I = b*t^3/12``) once the elastic core has shrunk, i.e. the classical
rigid-hardening moment increment.  Its integral over the mechanism rotation is
closed-form::

    dE_h(theta) = L_h * b*E_h * [ (c^3/3)kappa^2 - eps_y*c^2*kappa
                                  - eps_y^3/(3*kappa) + eps_y^2*c ]

with ``kappa = theta/L_h`` the hinge curvature and ``L_h`` the plastic hinge
length.

Hinge length (rotation capacity)
--------------------------------
A hinge does not localise into zero length: the plate spreads the plastic
curvature until the extreme-fibre strain reaches the rotation capacity
``eps_u`` (rupture strain).  The declared law is

    L_h(theta) = max(k_h*t, (t/2)*theta/eps_u)

so below ``theta_c = 2*eps_u*k_h`` the hinge is one plate thickness thick
(``k_h = 1`` default) and above it the hinge spreads, holding the peak fibre
strain at exactly ``eps_u``.  ``theta`` beyond that point therefore increases the
dissipation at a *constant* hardening moment instead of a rising one, which is
the physically correct statement that the material cannot take more strain
(fracture is not modelled - see the limitation list of the benchmark report).

Verification routes (independent of the closed form)
---------------------------------------------------
* :func:`section_moment_per_width` - fibre quadrature of the same constitutive
  law over the section, with the perfectly-plastic moment subtracted
  (:func:`moment_increment_fibre`); reproduces :func:`
  bilinear_moment_increment_closed_form` to quadrature accuracy.
* :func:`energy_increment_numeric` - Simpson integration of the fibre moment
  increment over the rotation; reproduces the closed-form energy increment.
* :func:`energy_increment_upper_bound` - the rigid-hardening (asymptotic)
  bound ``int E_h*I*kappa dtheta``; the closed form must stay below it.

Power-law hardening (``sigma = sigma_y + K*eps_p^(1/n)``) is provided as
:class:`PowerLawHinge`, evaluated in the same fibre quadrature (no closed form);
it is a sensitivity option, the design default is bilinear.

All quantities SI (m, N, J, Pa, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .resistance import DoubleHullSide, SideShellLayer

__all__ = [
    "ELASTIC_MODULUS_STEEL_PA",
    "DEFAULT_ROTATION_CAPACITY",
    "CLAMPED_HINGE_ROTATION_FACTOR",
    "section_moment_per_width",
    "moment_increment_fibre",
    "bilinear_moment_increment_closed_form",
    "BilinearHinge",
    "PowerLawHinge",
    "energy_increment_numeric",
    "HardenedLayer",
    "HardenedSide",
    "build_hardened_side",
]

#: Young's modulus of hull structural steel [Pa] (steel-grade independent to
#: within 2 %; the hardening *ratio* E_h/E is the quantity that matters).
ELASTIC_MODULUS_STEEL_PA = 210.0e9

#: Rupture (total) strain of AH32/AH36 hull plate used as the hinge rotation
#: capacity; 0.15-0.25 is the usual range for these grades in collision work.
DEFAULT_ROTATION_CAPACITY = 0.20

#: ``theta = 8*u/span`` of the clamped-clamped three-hinge mechanism of
#: :mod:`tensordem.hull_collision.resistance`.
CLAMPED_HINGE_ROTATION_FACTOR = 8.0


def _real(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


# ---------------------------------------------------------------------------
# fibre quadrature of the section (independent of the closed form)
# ---------------------------------------------------------------------------
def section_moment_per_width(
    kappa: float,
    *,
    sigma_y: float,
    thickness: float,
    elastic_modulus: float = ELASTIC_MODULUS_STEEL_PA,
    hardening_modulus: float = 0.0,
    power_exponent: Optional[float] = None,
    hardening_coefficient: float = 0.0,
    n_fibres: int = 8001,
) -> float:
    """Section moment per unit width [N*m/m] from a fibre quadrature.

    Strains are ``eps = kappa*y`` with ``y`` measured from the section mid-plane
    (``-t/2 .. +t/2``).  The constitutive law is bilinear (``hardening_modulus``,
    ``power_exponent = None``) or power-law
    ``sigma = sigma_y + hardening_coefficient*eps_p**(1/power_exponent)`` for the
    yielded part.  Only ``b = 1`` m of plate width is integrated; the result is
    therefore a moment per unit width, dividing out ``b`` of
    ``Z = b*t^2/4``.
    """
    ky = _real("kappa", kappa, nonnegative=True)
    sy = _real("sigma_y", sigma_y, positive=True)
    t = _real("thickness", thickness, positive=True)
    e = _real("elastic_modulus", elastic_modulus, positive=True)
    eh = _real("hardening_modulus", hardening_modulus, nonnegative=True)
    n = _positive_int("n_fibres", n_fibres, 3)

    eps_y = sy / e
    c = 0.5 * t
    dy = c / n
    moment = 0.0
    for i in range(n):
        y = (i + 0.5) * dy
        eps = ky * y
        if eps <= eps_y:
            sigma = e * eps
        else:
            eps_p = eps - eps_y
            if power_exponent is None:
                sigma = sy + eh * eps_p
            else:
                k = _real("hardening_coefficient", hardening_coefficient, nonnegative=True)
                p = _real("power_exponent", power_exponent, positive=True)
                sigma = sy + k * eps_p ** (1.0 / p)
        moment += sigma * y * dy
    return 2.0 * moment


def moment_increment_fibre(
    kappa: float,
    *,
    sigma_y: float,
    thickness: float,
    hardening_modulus: float,
    elastic_modulus: float = ELASTIC_MODULUS_STEEL_PA,
    n_fibres: int = 8001,
) -> float:
    """``M_bilinear(kappa) - M_plastic(kappa)`` [N*m/m] from two fibre runs.

    Subtracting two fibre quadratures with the *same* elastic core is what makes
    this an independent check of the closed form: no ``b*t^2/4`` and no analytic
    section algebra enters.
    """
    common = dict(
        sigma_y=sigma_y,
        thickness=thickness,
        elastic_modulus=elastic_modulus,
        n_fibres=n_fibres,
    )
    return section_moment_per_width(
        kappa, hardening_modulus=hardening_modulus, **common
    ) - section_moment_per_width(kappa, hardening_modulus=0.0, **common)


def bilinear_moment_increment_closed_form(
    kappa: float,
    *,
    sigma_y: float,
    thickness: float,
    hardening_modulus: float,
    elastic_modulus: float = ELASTIC_MODULUS_STEEL_PA,
) -> float:
    """Closed form of :func:`moment_increment_fibre` (per unit plate width).

    ``dM_h = E_h*[(2*c^3/3)*kappa - eps_y*c^2 + eps_y^3/(3*kappa^2)]`` for
    ``kappa > eps_y/c`` and exactly ``0.0`` below first yield / for ``E_h = 0``.
    """
    sy = _real("sigma_y", sigma_y, positive=True)
    t = _real("thickness", thickness, positive=True)
    eh = _real("hardening_modulus", hardening_modulus, nonnegative=True)
    e = _real("elastic_modulus", elastic_modulus, positive=True)
    ky = _real("kappa", kappa, nonnegative=True)
    if eh == 0.0:
        return 0.0
    c = 0.5 * t
    eps_y = sy / e
    if ky <= eps_y / c:
        return 0.0
    return eh * (
        (2.0 * c ** 3 / 3.0) * ky - eps_y * c * c + eps_y ** 3 / (3.0 * ky * ky)
    )


# ---------------------------------------------------------------------------
# hardening models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BilinearHinge:
    """Bilinear strain-hardening hinge of a plating section (per unit width).

    ``hardening_modulus = 0`` reproduces the perfectly-plastic N01 hinge exactly
    (every increment is returned as a literal ``0.0``).
    """

    sigma_y: float
    thickness: float
    hardening_modulus: float
    elastic_modulus: float = ELASTIC_MODULUS_STEEL_PA
    rotation_capacity: float = DEFAULT_ROTATION_CAPACITY
    hinge_length_factor: float = 1.0

    def __post_init__(self) -> None:
        _real("sigma_y", self.sigma_y, positive=True)
        _real("thickness", self.thickness, positive=True)
        _real("hardening_modulus", self.hardening_modulus, nonnegative=True)
        _real("elastic_modulus", self.elastic_modulus, positive=True)
        _real("rotation_capacity", self.rotation_capacity, positive=True)
        _real("hinge_length_factor", self.hinge_length_factor, positive=True)

    # -- geometry --------------------------------------------------------
    @property
    def half_thickness(self) -> float:
        return 0.5 * self.thickness

    @property
    def yield_strain(self) -> float:
        return self.sigma_y / self.elastic_modulus

    @property
    def yield_curvature(self) -> float:
        """Curvature at first yield of the extreme fibre [1/m]."""
        return self.yield_strain / self.half_thickness

    @property
    def second_moment_per_width(self) -> float:
        return self.thickness ** 3 / 12.0

    @property
    def plastic_moment_per_width(self) -> float:
        """``M_p = sigma_y*t^2/4`` [N*m/m] - the N01 perfect-plasticity moment."""
        return self.sigma_y * self.thickness ** 2 / 4.0

    @property
    def hinge_length_floor(self) -> float:
        return self.hinge_length_factor * self.thickness

    @property
    def rotation_capacity_angle(self) -> float:
        """Rotation at which the hinge starts spreading [rad]."""
        return 2.0 * self.rotation_capacity * self.hinge_length_factor

    def hinge_length(self, theta: float) -> float:
        """Plastic hinge length [m] for a mechanism rotation ``theta`` [rad]."""
        th = _real("theta", theta, nonnegative=True)
        spread = self.half_thickness * th / self.rotation_capacity
        return max(self.hinge_length_floor, spread)

    def curvature(self, theta: float) -> float:
        th = _real("theta", theta, nonnegative=True)
        return th / self.hinge_length(th)

    def peak_fibre_strain(self, theta: float) -> float:
        return self.curvature(theta) * self.half_thickness

    # -- mechanics -------------------------------------------------------
    def moment_increment(self, theta: float) -> float:
        """Hardening part of the hinge moment per unit width [N*m/m]."""
        if self.hardening_modulus == 0.0:
            return 0.0
        th = _real("theta", theta, nonnegative=True)
        if th == 0.0:
            return 0.0
        return bilinear_moment_increment_closed_form(
            self.curvature(th),
            sigma_y=self.sigma_y,
            thickness=self.thickness,
            hardening_modulus=self.hardening_modulus,
            elastic_modulus=self.elastic_modulus,
        )

    def energy_increment(self, theta: float) -> float:
        """``int_0^theta dM_h dtheta'`` per unit width [J/m] (closed form)."""
        if self.hardening_modulus == 0.0:
            return 0.0
        th = _real("theta", theta, nonnegative=True)
        if th == 0.0:
            return 0.0
        c = self.half_thickness
        eps_y = self.yield_strain
        eh = self.hardening_modulus
        theta_c = self.rotation_capacity_angle
        floor = self.hinge_length_floor

        def _integral(length: float, kappa: float) -> float:
            return length * eh * (
                c ** 3 * kappa * kappa / 3.0
                - eps_y * c * c * kappa
                - eps_y ** 3 / (3.0 * kappa)
                + eps_y * eps_y * c
            )

        if th <= theta_c:
            kappa = th / floor
            if kappa <= self.yield_curvature:
                return 0.0
            return _integral(floor, kappa)
        # capped branch: hinge spreads, curvature (and dM_h) is constant
        kappa_u = 2.0 * self.rotation_capacity / self.thickness
        energy_c = _integral(floor, theta_c / floor)
        dm_u = bilinear_moment_increment_closed_form(
            kappa_u,
            sigma_y=self.sigma_y,
            thickness=self.thickness,
            hardening_modulus=eh,
            elastic_modulus=self.elastic_modulus,
        )
        return energy_c + dm_u * (th - theta_c)

    def energy_increment_upper_bound(self, theta: float) -> float:
        """Rigid-hardening bound ``int_0^theta E_h*I*kappa dtheta`` [J/m].

        Equal to ``0.5*E_h*I*theta^2/L_h`` while the hinge has not started to
        spread, and exact (piecewise) afterwards.  The closed-form hardening
        increment must stay below it (the elastic core carries less than the
        asymptotic ``E_h*I*kappa``).
        """
        if self.hardening_modulus == 0.0:
            return 0.0
        th = _real("theta", theta, nonnegative=True)
        if th == 0.0:
            return 0.0
        iw = self.second_moment_per_width
        eh = self.hardening_modulus
        theta_c = self.rotation_capacity_angle
        floor = self.hinge_length_floor
        if th <= theta_c:
            return 0.5 * eh * iw * th * th / floor
        kappa_u = 2.0 * self.rotation_capacity / self.thickness
        return (
            0.5 * eh * iw * theta_c * theta_c / floor
            + eh * iw * kappa_u * (th - theta_c)
        )

    def as_dict(self) -> dict:
        return {
            "sigma_y_MPa": self.sigma_y / 1e6,
            "thickness_m": self.thickness,
            "hardening_modulus_GPa": self.hardening_modulus / 1e9,
            "hardening_ratio_Eh_over_E": self.hardening_modulus / self.elastic_modulus,
            "elastic_modulus_GPa": self.elastic_modulus / 1e9,
            "rotation_capacity": self.rotation_capacity,
            "hinge_length_factor": self.hinge_length_factor,
            "yield_curvature_1_per_m": self.yield_curvature,
            "plastic_moment_per_width_Nm_per_m": self.plastic_moment_per_width,
            "hinge_length_floor_m": self.hinge_length_floor,
            "rotation_capacity_angle_rad": self.rotation_capacity_angle,
        }


@dataclass(frozen=True)
class PowerLawHinge:
    """Power-law strain-hardening hinge (fibre-quadrature only).

    ``sigma = sigma_y + K*eps_p**(1/n)`` for the yielded part; ``n = 1`` reduces
    to the bilinear law with ``E_h = K``.  Provided as a sensitivity/robustness
    option (no closed form is claimed), and checked against
    :class:`BilinearHinge` in the ``n = 1`` limit by the acceptance tests.
    """

    sigma_y: float
    thickness: float
    hardening_coefficient: float
    exponent: float
    elastic_modulus: float = ELASTIC_MODULUS_STEEL_PA
    rotation_capacity: float = DEFAULT_ROTATION_CAPACITY
    hinge_length_factor: float = 1.0

    def __post_init__(self) -> None:
        _real("sigma_y", self.sigma_y, positive=True)
        _real("thickness", self.thickness, positive=True)
        _real("hardening_coefficient", self.hardening_coefficient, nonnegative=True)
        _real("exponent", self.exponent, positive=True)
        _real("elastic_modulus", self.elastic_modulus, positive=True)
        _real("rotation_capacity", self.rotation_capacity, positive=True)
        _real("hinge_length_factor", self.hinge_length_factor, positive=True)

    @property
    def half_thickness(self) -> float:
        return 0.5 * self.thickness

    @property
    def yield_strain(self) -> float:
        return self.sigma_y / self.elastic_modulus

    @property
    def hinge_length_floor(self) -> float:
        return self.hinge_length_factor * self.thickness

    @property
    def rotation_capacity_angle(self) -> float:
        return 2.0 * self.rotation_capacity * self.hinge_length_factor

    def hinge_length(self, theta: float) -> float:
        th = _real("theta", theta, nonnegative=True)
        return max(self.hinge_length_floor, self.half_thickness * th / self.rotation_capacity)

    def curvature(self, theta: float) -> float:
        th = _real("theta", theta, nonnegative=True)
        return th / self.hinge_length(th)

    def peak_fibre_strain(self, theta: float) -> float:
        return self.curvature(theta) * self.half_thickness

    def moment_increment(self, theta: float, *, n_fibres: int = 4001) -> float:
        if self.hardening_coefficient == 0.0:
            return 0.0
        kappa = self.curvature(theta)
        if kappa <= 0.0:
            return 0.0
        common = dict(
            sigma_y=self.sigma_y,
            thickness=self.thickness,
            elastic_modulus=self.elastic_modulus,
            n_fibres=n_fibres,
            power_exponent=self.exponent,
            hardening_coefficient=self.hardening_coefficient,
        )
        return section_moment_per_width(kappa, **common) - section_moment_per_width(
            kappa,
            sigma_y=self.sigma_y,
            thickness=self.thickness,
            elastic_modulus=self.elastic_modulus,
            hardening_modulus=0.0,
            n_fibres=n_fibres,
        )

    def energy_increment(self, theta: float, *, n_steps: int = 128, n_fibres: int = 2001) -> float:
        """``int_0^theta dM_h dtheta'`` by Simpson integration of the fibre moment."""
        if self.hardening_coefficient == 0.0:
            return 0.0
        th = _real("theta", theta, nonnegative=True)
        n_steps = _positive_int("n_steps", n_steps, 2)
        if n_steps % 2 == 1:
            n_steps += 1
        if th == 0.0:
            return 0.0
        h = th / n_steps
        total = self.moment_increment(0.0, n_fibres=n_fibres) + self.moment_increment(
            th, n_fibres=n_fibres
        )
        for i in range(1, n_steps):
            total += (4.0 if i % 2 == 1 else 2.0) * self.moment_increment(
                i * h, n_fibres=n_fibres
            )
        return total * h / 3.0


def energy_increment_numeric(
    hinge: BilinearHinge, theta: float, *, n_steps: int = 128, n_fibres: int = 2001
) -> float:
    """Simpson integration of the *fibre* moment increment over the rotation [J/m].

    Independent of :meth:`BilinearHinge.energy_increment` (which is an analytic
    integral of the closed-form moment) and of ``b*t^2/4``.
    """
    th = _real("theta", theta, nonnegative=True)
    n_steps = _positive_int("n_steps", n_steps, 2)
    if n_steps % 2 == 1:
        n_steps += 1
    if hinge.hardening_modulus == 0.0 or th == 0.0:
        return 0.0

    def f(angle: float) -> float:
        kappa = hinge.curvature(angle)
        if kappa <= 0.0:
            return 0.0
        return moment_increment_fibre(
            kappa,
            sigma_y=hinge.sigma_y,
            thickness=hinge.thickness,
            hardening_modulus=hinge.hardening_modulus,
            elastic_modulus=hinge.elastic_modulus,
            n_fibres=n_fibres,
        )

    h = th / n_steps
    total = f(0.0) + f(th)
    for i in range(1, n_steps):
        total += (4.0 if i % 2 == 1 else 2.0) * f(i * h)
    return total * h / 3.0


# ---------------------------------------------------------------------------
# layer / side wrappers (series stack, same ledger as N01)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HardenedLayer:
    """A :class:`SideShellLayer` with a strain-hardening hinge law.

    The hardening ratio is computed on the *plating* section of the layer and
    applied to the layer's total hinge energy.  This is the documented closure
    assumption: plate and longitudinals of one hinge line rotate by the same
    amount and are of the same steel, so the section-level hardening ratio
    ``Phi(theta) = dE_h/(M_p*theta)`` multiplies the whole hinge term, i.e.
    ``E = k1*u + k1*(span/8)*(dE_h(θ)/M_p) + k2*u^2`` with ``θ = 8*u/span``.  It
    keeps the membrane term (a *stretch* mechanism, not a hinge) untouched and
    makes ``E_h -> 0`` an exact identity with N01.
    """

    base: SideShellLayer
    hinge: Optional[BilinearHinge] = None
    rotation_factor: float = CLAMPED_HINGE_ROTATION_FACTOR

    def __post_init__(self) -> None:
        if not isinstance(self.base, SideShellLayer):
            raise TypeError("base must be a SideShellLayer")
        if self.hinge is not None:
            if not isinstance(self.hinge, BilinearHinge):
                raise TypeError("hinge must be a BilinearHinge or None")
            if abs(self.hinge.thickness - self.base.thickness) > 1e-12:
                raise ValueError("hinge thickness must equal the layer thickness")
            if abs(self.hinge.sigma_y - self.base.sigma_y) > 1e-6 * self.base.sigma_y:
                raise ValueError("hinge sigma_y must equal the layer yield stress")
        _real("rotation_factor", self.rotation_factor, positive=True)

    @property
    def name(self) -> str:
        return self.base.name

    def hinge_rotation(self, local_penetration: float) -> float:
        u = _real("local_penetration", local_penetration, nonnegative=True)
        return self.rotation_factor * u / self.base.frame_span

    def moment_ratio(self, local_penetration: float) -> float:
        """``dM_h/M_p`` of the hinge at this local penetration (dimensionless)."""
        if self.hinge is None:
            return 0.0
        dm = self.hinge.moment_increment(self.hinge_rotation(local_penetration))
        if dm == 0.0:
            return 0.0
        return dm / self.hinge.plastic_moment_per_width

    def energy(self, local_penetration: float) -> float:
        u = min(
            max(_real("local_penetration", local_penetration, nonnegative=True), 0.0),
            self.base.crush_depth,
        )
        total = self.base.energy(u)
        if self.hinge is not None:
            # u/theta = span/rotation_factor, so the plate hardening increment
            # (per unit plate width) is scaled to the layer hinge by
            # k1*(span/rotation_factor)/M_p; its derivative w.r.t. u is exactly
            # k1*dM_h/M_p, which is what :meth:`resistance` returns.
            total += (
                self.base.hinge_coefficient
                * self.base.frame_span
                / self.rotation_factor
                / self.hinge.plastic_moment_per_width
                * self.hinge.energy_increment(self.hinge_rotation(u))
            )
        return total

    def resistance(self, local_penetration: float) -> float:
        u = _real("local_penetration", local_penetration, nonnegative=True)
        if u >= self.base.crush_depth:
            return 0.0
        force = self.base.resistance(u)
        if self.hinge is not None:
            force += self.base.hinge_coefficient * self.moment_ratio(u)
        return force

    @property
    def capacity(self) -> float:
        return self.energy(self.base.crush_depth)

    def peak_fibre_strain(self, local_penetration: float) -> float:
        if self.hinge is None:
            return 0.0
        return self.hinge.peak_fibre_strain(self.hinge_rotation(local_penetration))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "thickness_m": self.base.thickness,
            "sigma_y_MPa": self.base.sigma_y / 1e6,
            "hardening_moment_ratio_at_capacity": self.moment_ratio(
                self.base.crush_depth
            ),
            "hinge_coefficient_N_per_m": self.base.hinge_coefficient,
            "membrane_coefficient_N_per_m2": self.base.membrane_coefficient,
            "capacity_MJ": self.capacity / 1e6,
        }


@dataclass(frozen=True)
class HardenedSide:
    """Series stack of :class:`HardenedLayer` with the N01 energy ledger."""

    layers: tuple
    friction_coefficient: float = 0.0

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("at least one layer is required")
        for layer in layers:
            if not isinstance(layer, HardenedLayer):
                raise TypeError("layers must be HardenedLayer instances")
        _real("friction_coefficient", self.friction_coefficient, nonnegative=True)
        object.__setattr__(self, "layers", layers)

    @property
    def window_starts(self) -> tuple:
        starts, acc = [], 0.0
        for layer in self.layers:
            starts.append(acc)
            acc += layer.base.crush_depth
        return tuple(starts)

    @property
    def max_depth(self) -> float:
        return sum(layer.base.crush_depth for layer in self.layers)

    def local_penetrations(self, depth: float) -> tuple:
        d = _real("depth", depth, nonnegative=True)
        return tuple(
            min(max(d - start, 0.0), layer.base.crush_depth)
            for layer, start in zip(self.layers, self.window_starts)
        )

    def layer_energies(self, depth: float) -> tuple:
        return tuple(
            layer.energy(u) for layer, u in zip(self.layers, self.local_penetrations(depth))
        )

    def structural_energy(self, depth: float) -> float:
        return math.fsum(self.layer_energies(depth))

    def friction_energy(self, depth: float) -> float:
        return self.friction_coefficient * self.structural_energy(depth)

    def absorbed(self, depth: float) -> float:
        return self.structural_energy(depth) + self.friction_energy(depth)

    def resistance(self, depth: float) -> float:
        d = _real("depth", depth, nonnegative=True)
        forces = []
        for layer, start in zip(self.layers, self.window_starts):
            if start > 0.0 and d <= start:
                continue
            forces.append(layer.resistance(d - start))
        return math.fsum(forces)

    @property
    def total_capacity(self) -> float:
        return self.absorbed(self.max_depth)

    def inner_shell_intact(self, depth: float) -> bool:
        locals_ = self.local_penetrations(depth)
        for layer, u in zip(self.layers[1:], locals_[1:]):
            if u >= layer.base.thickness:
                return False
        return True

    def breached_layers(self, depth: float) -> tuple:
        names, locals_ = [], self.local_penetrations(depth)
        for layer, u in zip(self.layers, locals_):
            if u >= layer.base.thickness:
                names.append(layer.name)
        return tuple(names)

    def peak_fibre_strains(self, depth: float) -> tuple:
        return tuple(
            layer.peak_fibre_strain(u)
            for layer, u in zip(self.layers, self.local_penetrations(depth))
        )

    def to_perfectly_plastic(self) -> DoubleHullSide:
        """The N01 structure (used as the ``E_h -> 0`` negative control)."""
        return DoubleHullSide(
            layers=tuple(layer.base for layer in self.layers),
            friction_coefficient=self.friction_coefficient,
        )


def build_hardened_side(
    structure: DoubleHullSide,
    *,
    hardening_ratio: float = 0.0,
    elastic_modulus: float = ELASTIC_MODULUS_STEEL_PA,
    rotation_capacity: float = DEFAULT_ROTATION_CAPACITY,
    hinge_length_factor: float = 1.0,
) -> HardenedSide:
    """Attach a bilinear hardening law to every layer of an N01 structure.

    ``hardening_ratio = E_h/E``; ``0.0`` gives back the perfectly-plastic N01
    surface exactly (all hardening increments are literal zeros).
    """
    if not isinstance(structure, DoubleHullSide):
        raise TypeError("structure must be a DoubleHullSide")
    ratio = _real("hardening_ratio", hardening_ratio, nonnegative=True)
    layers = []
    for layer in structure.layers:
        hinge = BilinearHinge(
            sigma_y=layer.sigma_y,
            thickness=layer.thickness,
            hardening_modulus=ratio * elastic_modulus if ratio > 0.0 else 0.0,
            elastic_modulus=elastic_modulus,
            rotation_capacity=rotation_capacity,
            hinge_length_factor=hinge_length_factor,
        )
        layers.append(HardenedLayer(base=layer, hinge=hinge))
    return HardenedSide(
        layers=tuple(layers), friction_coefficient=structure.friction_coefficient
    )
"""ISO 19906 pressure-area ice load, pressure-driven indentation and damage geometry.

Why this module exists
----------------------
The N01 surface (:mod:`tensordem.hull_collision.resistance`) treats the ice as a
*rigid* indenter: all of the collision energy must be absorbed by the steel, and
the damage footprint is the mechanism box ``L_c + 2*span`` (an equivalent
rectangle standing in for the real, lens-shaped damage extent).  Both are
ideal-isations.  This module replaces them with the ice-side physics:

* **ice load** - the ISO 19906 global pressure-area relation

      p(A) = p0 * (A/A_ref)**(-alpha),   alpha = 0.4 ... 0.6

  applied to the contact area that the indentation creates,
  ``A(d) = H*d`` (``H`` = contact height, ``d`` = penetration).  The ice load is
  ``N_ice(d) = p(A)*A = p0*A_ref**alpha*H**(1-alpha)*d**(1-alpha)`` and the
  crushing energy has the closed form
  ``E_ice(d) = p0*A_ref**alpha*H**(1-alpha)*d**(2-alpha)/(2-alpha)``.

* **damage length from the pressure integral** - instead of the box
  ``L_c + 2*span``, the length of the yielding strip that carries the ice load is
  obtained by integrating the ice pressure over the hull and equating it to the
  panel's resistance per unit length,

      L_y(d) = N_ice(d) / (kappa1*(1+dM_h/M_p) + 2*kappa2*d)

  with ``kappa1 = k1/W_eff`` and ``kappa2 = k2/W_eff`` the hinge and membrane
  coefficients of the struck layer per unit damaged length, i.e. the derivative
  of a yielding strip's energy ``(L/W_eff)*(k1*d + k2*d^2)``.  The reported damage
  length is ``max(L_c, L_y)``: the panel cannot yield over less than the geometric
  contact, and beyond it the load spreads over the yielding strip.

* **coupled energy ledger** - ``E_k = (1+mu)*E_struct(d) + E_ice(d) + residual``,
  solved for ``d`` by bracketed bisection exactly like N01, so the residual is
  reported and never forced to zero.  Because the ice side now absorbs energy,
  the penetration depth is *smaller* than the N01 rigid-indenter bound for the
  same kinetic energy - that difference is the physical content of the node.

* **force-time history** - the quasi-static history parameterised by energy,
  ``F(d) = R_struct(d) + N_ice(d)``, ``v(d) = sqrt(2*(E_k - E_abs(d))/m_eff)``,
  ``t(d) = int_0^d dd'/v(d')``.  This is *not* a dynamic simulation: it is the
  energy-parameterised loading path, explicitly labelled as such (the
  deceleration history of a real impact needs added-mass coupling and is out of
  scope here).

Limitations (stated, not hidden)
--------------------------------
* Monotone, quasi-static indentation; no strain-rate effect on either ice or
  steel, no ice fracture, no rubble build-up, no elastic contact compliance.
* The pressure-area law is applied to the *nominal* contact area of the growing
  indentation; it is a global (not local/line-load) relation, so it cannot
  reproduce local pressure spikes at ice features.
* ``p0`` is the pressure at ``A_ref`` (1 m^2); the value used for a case is
  scenario data (see ``scenario.py`` and the benchmark report), never a library
  constant fitted to the verification cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .resistance import SideShellLayer
from .strain_hardening import HardenedSide

__all__ = [
    "ISO_PRESSURE_AREA_EXPONENTS",
    "PressureAreaLaw",
    "IndentationSolution",
    "structural_coefficients_per_length",
    "solve_indentation",
    "damage_geometry_pressure_driven",
    "force_history",
    "iso19906_scaling_exponents",
]

#: ISO 19906 / Masterson pressure-area exponent family for the global ice
#: pressure on a growing contact area.
ISO_PRESSURE_AREA_EXPONENTS = (0.40, 0.45, 0.50, 0.55, 0.60)


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


@dataclass(frozen=True)
class PressureAreaLaw:
    """ISO 19906 global pressure-area relation ``p = p0*(A/A_ref)**(-alpha)``."""

    reference_pressure: float
    exponent: float
    reference_area: float = 1.0

    def __post_init__(self) -> None:
        _real("reference_pressure", self.reference_pressure, positive=True)
        _real("exponent", self.exponent, nonnegative=True)
        _real("reference_area", self.reference_area, positive=True)
        if self.exponent >= 1.0:
            raise ValueError("exponent must be < 1 (a growing contact must not lose load)")

    def pressure(self, area: float) -> float:
        """Nominal ice pressure over an area ``area`` [Pa]."""
        a = _real("area", area, positive=True)
        return self.reference_pressure * (a / self.reference_area) ** (-self.exponent)

    def load(self, area: float) -> float:
        """Ice load on a contact area [N] - ``p(A)*A``."""
        a = _real("area", area, positive=True)
        return self.pressure(a) * a

    def contact_area(self, contact_height: float, penetration: float) -> float:
        h = _real("contact_height", contact_height, positive=True)
        d = _real("penetration", penetration, nonnegative=True)
        return h * d

    def ice_load(self, contact_height: float, penetration: float) -> float:
        """``N_ice(d)`` [N]; zero at zero penetration."""
        d = _real("penetration", penetration, nonnegative=True)
        if d == 0.0:
            return 0.0
        return self.load(self.contact_area(contact_height, d))

    def ice_energy(self, contact_height: float, penetration: float) -> float:
        """Closed form of ``int_0^d N_ice dd'`` [J]."""
        h = _real("contact_height", contact_height, positive=True)
        d = _real("penetration", penetration, nonnegative=True)
        if d == 0.0:
            return 0.0
        alpha = self.exponent
        return (
            self.reference_pressure
            * self.reference_area ** alpha
            * h ** (1.0 - alpha)
            * d ** (2.0 - alpha)
            / (2.0 - alpha)
        )

    def ice_energy_numeric(
        self, contact_height: float, penetration: float, *, n_steps: int = 4000
    ) -> float:
        """Simpson integration of :meth:`ice_load` (independent route)."""
        h = _real("contact_height", contact_height, positive=True)
        d = _real("penetration", penetration, nonnegative=True)
        if d == 0.0:
            return 0.0
        if n_steps % 2 == 1:
            n_steps += 1
        step = d / n_steps
        total = 0.0  # load is 0 at d = 0
        for i in range(1, n_steps):
            total += (4.0 if i % 2 == 1 else 2.0) * self.ice_load(h, i * step)
        total += self.ice_load(h, d)
        return total * step / 3.0

    @property
    def penetration_exponent(self) -> float:
        """Exponent ``beta`` of ``d ~ E_k**beta`` for ice-dominated crushing."""
        return 1.0 / (2.0 - self.exponent)

    @property
    def yield_length_exponent(self) -> float:
        """Exponent ``gamma`` of ``L_y ~ E_k**gamma`` (membrane-dominated panel)."""
        return -self.exponent / (2.0 - self.exponent)

    def as_dict(self) -> dict:
        return {
            "reference_pressure_MPa": self.reference_pressure / 1e6,
            "exponent": self.exponent,
            "reference_area_m2": self.reference_area,
            "penetration_exponent": self.penetration_exponent,
            "yield_length_exponent": self.yield_length_exponent,
        }


def structural_coefficients_per_length(layer: SideShellLayer) -> tuple:
    """``(kappa1, kappa2)`` of one layer per unit damaged length.

    ``kappa1 = k1/W_eff`` [N/m] and ``kappa2 = k2/W_eff`` [N/m^2] are the hinge
    and membrane resistances *per unit damaged length*, so that a yielding strip
    of length ``L`` has ``R = L*(kappa1 + 2*kappa2*d)``, which is identically
    N01's ``k1 + 2*k2*d`` at ``L = W_eff`` and the derivative of its energy.
    """
    if not isinstance(layer, SideShellLayer):
        raise TypeError("layer must be a SideShellLayer")
    width = layer.effective_width
    return layer.hinge_coefficient / width, layer.membrane_coefficient / width


@dataclass(frozen=True)
class IndentationSolution:
    penetration_depth: float
    kinetic_energy: float
    structural_energy: float
    friction_energy: float
    ice_energy: float
    residual: float
    residual_relative: float
    layer_energies: tuple
    layer_local_penetrations: tuple
    layer_names: tuple
    ice_load_N: float
    structural_resistance_N: float
    yield_length_m: float
    damage_length_m: float
    inner_shell_intact: bool
    breached_layers: tuple
    peak_fibre_strains: tuple
    iterations: int
    capacity_J: float

    def as_dict(self) -> dict:
        return {
            "penetration_depth_m": self.penetration_depth,
            "kinetic_energy_MJ": self.kinetic_energy / 1e6,
            "structural_energy_MJ": self.structural_energy / 1e6,
            "friction_energy_MJ": self.friction_energy / 1e6,
            "ice_energy_MJ": self.ice_energy / 1e6,
            "residual_J": self.residual,
            "residual_relative": self.residual_relative,
            "layer_energies_MJ": [e / 1e6 for e in self.layer_energies],
            "layer_local_penetrations_m": list(self.layer_local_penetrations),
            "layer_names": list(self.layer_names),
            "ice_load_MN": self.ice_load_N / 1e6,
            "structural_resistance_MN": self.structural_resistance_N / 1e6,
            "yield_length_m": self.yield_length_m,
            "damage_length_m": self.damage_length_m,
            "inner_shell_intact": self.inner_shell_intact,
            "breached_layers": list(self.breached_layers),
            "peak_fibre_strains": list(self.peak_fibre_strains),
            "iterations": self.iterations,
            "capacity_MJ": self.capacity_J / 1e6,
        }


def yield_length(side: HardenedSide, depth: float, ice_load: float) -> float:
    """Yielding strip length that carries ``ice_load`` at ``depth`` [m].

    This is the *pressure integral* replacement for the box ``L_c + 2*span``:
    the ice load is equilibrated by the struck layer's resistance per unit
    damaged length, which itself carries the hardening ratio of the hinge.
    """
    if not isinstance(side, HardenedSide):
        raise TypeError("side must be a HardenedSide")
    if ice_load <= 0.0:
        return 0.0
    layer = side.layers[0]
    u = side.local_penetrations(depth)[0]
    kappa1, kappa2 = structural_coefficients_per_length(layer.base)
    hardening = 1.0 + layer.moment_ratio(u)
    # force per unit damaged length: d/du of the strip energy, i.e. the hinge
    # coefficient (times the hinge hardening ratio) plus 2*kappa2*u
    per_length = kappa1 * hardening + 2.0 * kappa2 * u
    if per_length <= 0.0:
        return math.inf
    return ice_load / per_length


def solve_indentation(
    kinetic_energy: float,
    side: HardenedSide,
    *,
    law: Optional[PressureAreaLaw] = None,
    contact_height: float = 1.0,
    contact_length: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> IndentationSolution:
    """Solve ``E_k = (1+mu)*E_struct(d) + E_ice(d)`` for the penetration depth.

    ``law = None`` switches the ice side off entirely (rigid indenter), which is
    the N01 model: that limit is the negative control of this node.
    """
    if not isinstance(side, HardenedSide):
        raise TypeError("side must be a HardenedSide")
    e_k = _real("kinetic_energy", kinetic_energy, nonnegative=True)
    h = _real("contact_height", contact_height, positive=True)
    l_c = _real("contact_length", contact_length, nonnegative=True)
    tol = _real("tol", tol, positive=True)
    if law is not None and not isinstance(law, PressureAreaLaw):
        raise TypeError("law must be a PressureAreaLaw or None")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    def ice_energy(depth: float) -> float:
        return 0.0 if law is None else law.ice_energy(h, depth)

    def absorbed(depth: float) -> float:
        return side.absorbed(depth) + ice_energy(depth)

    capacity = absorbed(side.max_depth)
    if e_k > capacity:
        raise ValueError(
            f"kinetic energy {e_k:.6e} J exceeds structural + ice capacity {capacity:.6e} J"
        )

    def build(depth: float, iterations: int) -> IndentationSolution:
        structural = side.structural_energy(depth)
        friction = side.friction_energy(depth)
        ice = ice_energy(depth)
        residual = e_k - (structural + friction + ice)
        ice_load = 0.0 if law is None else law.ice_load(h, depth)
        l_y = yield_length(side, depth, ice_load) if ice_load > 0.0 else 0.0
        damage_length = max(l_c, l_y)
        return IndentationSolution(
            penetration_depth=depth,
            kinetic_energy=e_k,
            structural_energy=structural,
            friction_energy=friction,
            ice_energy=ice,
            residual=residual,
            residual_relative=abs(residual) / e_k if e_k > 0.0 else 0.0,
            layer_energies=side.layer_energies(depth),
            layer_local_penetrations=side.local_penetrations(depth),
            layer_names=tuple(layer.name for layer in side.layers),
            ice_load_N=ice_load,
            structural_resistance_N=side.resistance(depth),
            yield_length_m=l_y,
            damage_length_m=damage_length,
            inner_shell_intact=side.inner_shell_intact(depth),
            breached_layers=side.breached_layers(depth),
            peak_fibre_strains=side.peak_fibre_strains(depth),
            iterations=iterations,
            capacity_J=capacity,
        )

    if e_k == 0.0:
        return build(0.0, 0)

    lo, hi = 0.0, side.max_depth
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        mid = 0.5 * (lo + hi)
        if absorbed(mid) < e_k:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return build(0.5 * (lo + hi), iterations)


def damage_geometry_pressure_driven(
    side: HardenedSide,
    solution: IndentationSolution,
    *,
    contact_length: float,
    length: Optional[float] = None,
):
    """Damage extents from the pressure-integrated length instead of the box.

    Reuses the N01 accounting rule (plating volume ``L*W_k*t_k`` plus
    longitudinals ``n_k*A_k*L`` of every engaged layer) but with
    ``L = solution.damage_length_m`` (the pressure integral) rather than
    ``L_c + 2*span``; ``length`` overrides the length for the negative control.
    """
    if not isinstance(side, HardenedSide):
        raise TypeError("side must be a HardenedSide")
    if not isinstance(solution, IndentationSolution):
        raise TypeError("solution must be an IndentationSolution")
    l_c = _real("contact_length", contact_length, positive=True)
    length = solution.damage_length_m if length is None else _real("length", length, positive=True)
    volumes, width = [], 0.0
    for layer, u in zip(side.layers, solution.layer_local_penetrations):
        if u <= 0.0:
            volumes.append(0.0)
            continue
        width = max(width, layer.base.effective_width)
        volumes.append(
            length * layer.base.effective_width * layer.base.thickness
            + layer.base.n_longitudinals * layer.base.longitudinal_area * length
        )
    if width == 0.0:
        width = side.layers[0].base.effective_width
    from .resistance import DamageGeometry

    return DamageGeometry(
        length_m=length,
        width_m=width,
        depth_m=solution.penetration_depth,
        volume_m3=math.fsum(volumes),
        inner_shell_intact=solution.inner_shell_intact,
        breached_layers=solution.breached_layers,
        layer_volumes_m3=tuple(volumes),
    ), l_c


def force_history(
    side: HardenedSide,
    *,
    kinetic_energy: float,
    effective_mass: float,
    law: Optional[PressureAreaLaw] = None,
    contact_height: float = 1.0,
    contact_length: float = 0.0,
    n_points: int = 65,
    depth_fraction: float = 0.98,
) -> dict:
    """Quasi-static force/velocity/time history along the indentation path.

    ``F(d) = R_struct(d) + N_ice(d)`` and the velocity follows from the remaining
    kinetic energy, ``v(d) = sqrt(2*(E_k - E_abs(d))/m_eff)``; the time is
    ``t(d) = int_0^d dd'/v(d')`` (trapezoid).  The history stops at
    ``depth_fraction`` of the solved depth because ``v -> 0`` there and the
    contact time diverges logarithmically - reported, not clipped silently.
    """
    if not isinstance(side, HardenedSide):
        raise TypeError("side must be a HardenedSide")
    e_k = _real("kinetic_energy", kinetic_energy, positive=True)
    m = _real("effective_mass", effective_mass, positive=True)
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 3:
        raise ValueError("n_points must be an integer >= 3")
    fraction = _real("depth_fraction", depth_fraction, positive=True)
    if fraction > 1.0:
        raise ValueError("depth_fraction must be <= 1")

    solution = solve_indentation(
        e_k, side, law=law, contact_height=contact_height, contact_length=contact_length
    )
    depth_max = solution.penetration_depth * fraction
    depths, loads, velocities, times = [], [], [], []
    t = 0.0
    previous = None
    for i in range(n_points):
        d = depth_max * i / (n_points - 1)
        absorbed = side.absorbed(d) + (0.0 if law is None else law.ice_energy(contact_height, d))
        remaining = max(e_k - absorbed, 0.0)
        v = math.sqrt(2.0 * remaining / m)
        load = side.resistance(d) + (
            0.0 if law is None else law.ice_load(contact_height, d)
        )
        if previous is not None:
            step = d - previous["d"]
            if v > 0.0 and previous["v"] > 0.0:
                t += step / (0.5 * (v + previous["v"]))
        depths.append(d)
        loads.append(load)
        velocities.append(v)
        times.append(t)
        previous = {"d": d, "v": v}
    return {
        "depth_m": depths,
        "load_N": loads,
        "velocity_m_s": velocities,
        "time_s": times,
        "solved_depth_m": solution.penetration_depth,
        "history_depth_m": depth_max,
        "depth_fraction": fraction,
        "effective_mass_kg": m,
        "solution": solution,
    }


def iso19906_scaling_exponents(exponent: float) -> dict:
    """Closed-form scaling exponents implied by the pressure-area exponent.

    For an ice-crushing-dominated indentation::

        N_ice ~ d**(1-alpha)            -> E_ice ~ d**(2-alpha)
        d ~ E_k**(1/(2-alpha))          -> L_y ~ d**(-alpha) ~ E_k**(-alpha/(2-alpha))

    The last line holds in the membrane-dominated regime (``kappa2*d^2``
    dominates the yielding strip), which is the regime of the design case.
    """
    alpha = _real("exponent", exponent, nonnegative=True)
    if alpha >= 1.0:
        raise ValueError("exponent must be < 1")
    return {
        "exponent": alpha,
        "load_vs_penetration": 1.0 - alpha,
        "energy_vs_penetration": 2.0 - alpha,
        "penetration_vs_energy": 1.0 / (2.0 - alpha),
        "yield_length_vs_energy": -alpha / (2.0 - alpha),
    }
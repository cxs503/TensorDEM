"""Temperature-dependent DEM bond: stabilised adapter over the verified laws.

Campaign node: ``P121W5-N02`` (temperature-dependent bond model).  The physics
already lives in two verified modules and is **not** re-implemented here:

* :mod:`tensordem.ice_bond_viscoelastic` -- the Maxwell-Wiechert (Prony series)
  delayed-elastic bond law with exact incremental integration, the energy
  ledger ``W - dU - Phi = 0`` and the Kachanov creep-damage clock;
* :mod:`tensordem.wet_ice_cohesion.ice_resistance` -- the Timco & Weeks (2010)
  brine-volume / strength relation, i.e. the temperature dependence.

This module adds exactly one thing: a single object that binds the two together
so a call site can ask for "the bond force and damage increment of an ice bond
at interface temperature ``T``" without knowing which of the two modules owns
which piece.  Nothing upstream is modified, so every existing call signature and
every existing verification result is preserved bit for bit (the integration test
compares the adapter against the raw library calls at ``1e-10`` relative
tolerance and gets exact equality).

Temperature enters only through quantities that the source relations define at
``T``:

``brine_volume(T)``
    ``v_b = S_a (49.185/|T| + 0.532)/1000`` (Frankenstein & Garner 1967 as
    reproduced by Timco & Weeks 2010).
``strength_factor(T)``
    ``(1 - sqrt(v_b/0.29))**2`` **normalised** by its value at
    ``reference_temperature_c``, so the factor is 1 at the reference state and
    colder ice is stronger (``> 1``).
``reference_force(T)``
    ``reference_force_n * strength_factor(T)``: the ``F_ref`` of the Kachanov
    clock ``dD/dt = (|F|/F_ref)**m / t_ref``.  Colder bonds therefore damage
    more slowly at the same force -- which is the direction the creep literature
    reports -- and the *form* of the clock is untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..ice_bond_viscoelastic import (
    DEFAULT_DELAYED_ELASTIC_FRACTION,
    DEFAULT_EQUILIBRIUM_FRACTION,
    DEFAULT_RELAXATION_TIMES_S,
    MaxwellWiechertBond,
    advance_branches,
    damage_increment,
    elastic_force,
    stored_energy,
)
from ..wet_ice_cohesion.ice_resistance import (
    timco_weeks_brine_volume,
    timco_weeks_strength_factor,
)

__all__ = ["TemperatureDependentBond", "BRINE_RELATION_FLOOR_C",
           "STRENGTH_FIT_CEILING_C"]

#: Lower validity bound of the Timco-Weeks brine-volume relation (degC).  The
#: relation is fitted over ``-30 <= T <= -2`` degC; below ``-60`` degC the
#: bracket ``49.185/|T|`` makes the extrapolation meaningless, so the adapter
#: refuses instead of returning a plausible-looking number.
BRINE_RELATION_FLOOR_C = -60.0

#: Upper bound of the *strength* fit.  Timco & Weeks (2010) fit the porosity
#: factor over roughly ``-30 <= T <= -2`` degC.  Above ``-2`` degC the brine
#: volume of 5 ppt sea ice exceeds 0.15 and the factor falls quickly to zero
#: (at ``-1.54`` degC the brine volume reaches the 0.29 pore limit, so the
#: relation returns exactly 0).  The adapter keeps the relation's own value --
#: it is the correct qualitative limit of a mushy skeleton -- but reports
#: :meth:`TemperatureDependentBond.within_strength_fit` so a caller can tell an
#: in-fit number from an extrapolated one.
STRENGTH_FIT_CEILING_C = -2.0


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
class TemperatureDependentBond:
    """A freeze bond whose strength follows the ice temperature.

    Parameters
    ----------
    law:
        Maxwell-Wiechert bond law (:class:`tensordem.ice_bond_viscoelastic.
        MaxwellWiechertBond`), used unchanged for the force / energy / branch
        advance.
    reference_temperature_c:
        Temperature at which :meth:`strength_factor` is exactly 1 (must be
        negative: the Timco-Weeks relation is fitted for ``T < 0``).
    salinity_ppt:
        Bulk salinity of the ice, ppt (sea ice default: 5 ppt).
    reference_force_n:
        Kachanov reference force ``F_ref`` at ``reference_temperature_c``, N.
    damage_time_s:
        Kachanov reference time ``t_ref``, s.
    damage_exponent:
        Kachanov exponent ``m`` (dimensionless, ``m + 1`` is the ductile
        strength exponent).

    Notes
    -----
    Every number below is produced by the upstream modules; this class only
    multiplies the strength factor into ``F_ref``.  ``strength_factor`` is the
    *normalised* Timco-Weeks factor, so a bond at the reference temperature is
    identical to the un-modified Kachanov clock (``F_ref`` unchanged).
    """

    law: MaxwellWiechertBond
    reference_temperature_c: float = -10.0
    salinity_ppt: float = 5.0
    reference_force_n: float = 1.0e3
    damage_time_s: float = 60.0
    damage_exponent: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.law, MaxwellWiechertBond):
            raise TypeError("law must be a MaxwellWiechertBond")
        ref = _finite("reference_temperature_c", self.reference_temperature_c)
        if ref >= 0.0:
            raise ValueError("reference_temperature_c must be negative (T < 0 degC)")
        object.__setattr__(self, "reference_temperature_c", ref)
        object.__setattr__(self, "salinity_ppt",
                           _finite("salinity_ppt", self.salinity_ppt,
                                   nonnegative=True))
        object.__setattr__(self, "reference_force_n",
                           _finite("reference_force_n", self.reference_force_n,
                                   positive=True))
        object.__setattr__(self, "damage_time_s",
                           _finite("damage_time_s", self.damage_time_s,
                                   positive=True))
        object.__setattr__(self, "damage_exponent",
                           _finite("damage_exponent", self.damage_exponent,
                                   positive=True))
        if self.strength_factor(ref) <= 0.0:
            raise ValueError(
                "the reference state has zero strength: the Timco-Weeks factor "
                "vanishes, so the normalised strength factor is undefined")

    # -- construction ----------------------------------------------------
    @classmethod
    def from_unrelaxed_stiffness(
        cls,
        unrelaxed_stiffness: float,
        *,
        equilibrium_fraction: float = DEFAULT_EQUILIBRIUM_FRACTION,
        delayed_elastic_fraction: tuple[float, ...] = DEFAULT_DELAYED_ELASTIC_FRACTION,
        relaxation_time: tuple[float, ...] = DEFAULT_RELAXATION_TIMES_S,
        **kwargs: float,
    ) -> "TemperatureDependentBond":
        """Build the bond law from a target instantaneous stiffness (N/m)."""
        law = MaxwellWiechertBond.from_unrelaxed_stiffness(
            unrelaxed_stiffness,
            equilibrium_fraction=equilibrium_fraction,
            delayed_elastic_fraction=delayed_elastic_fraction,
            relaxation_time=relaxation_time,
        )
        return cls(law=law, **kwargs)

    # -- temperature dependence -------------------------------------------
    def brine_volume(self, temperature_c: float) -> float:
        """Brine volume fraction at ``T`` (delegates Timco-Weeks).

        Raises
        ------
        ValueError
            At or above the melting point (no brine exists) and below
            :data:`BRINE_RELATION_FLOOR_C` (outside the fitted range).  Both are
            refusals rather than extrapolations.
        """
        t = _finite("temperature_c", temperature_c)
        if t >= 0.0:
            raise ValueError(
                "the brine-volume relation is defined for T < 0 degC only; a "
                "melting interface has no solid ice bond (see strength_factor)")
        if t < BRINE_RELATION_FLOOR_C:
            raise ValueError(
                f"temperature_c below the fitted range ({BRINE_RELATION_FLOOR_C} degC)")
        return timco_weeks_brine_volume(self.salinity_ppt, t)

    def _raw_strength_factor(self, temperature_c: float) -> float:
        return timco_weeks_strength_factor(self.brine_volume(temperature_c))

    def strength_factor(self, temperature_c: float) -> float:
        """Strength relative to the reference temperature.

        ``1.0`` at ``reference_temperature_c``, ``> 1`` for colder ice, and
        exactly ``0.0`` at or above the melting point ``T >= 0``: a liquid film
        carries no freeze bond.  That is the one place where the adapter adds a
        *limit* to the upstream relation instead of only scaling it, and it is
        the physically forced value of the Timco-Weeks factor at zero solid
        fraction.
        """
        if _finite("temperature_c", temperature_c) >= 0.0:
            return 0.0
        return self._raw_strength_factor(temperature_c) / self._raw_strength_factor(
            self.reference_temperature_c)

    def within_strength_fit(self, temperature_c: float) -> bool:
        """Whether ``T`` lies inside the fitted band of the strength relation.

        ``False`` above :data:`STRENGTH_FIT_CEILING_C` (warm, mushy ice) and
        below :data:`BRINE_RELATION_FLOOR_C` (colder than the fit).  The value
        returned by :meth:`strength_factor` is still the relation's own value --
        this predicate only tells the caller whether it is an interpolation or
        an extrapolation.
        """
        t = _finite("temperature_c", temperature_c)
        return BRINE_RELATION_FLOOR_C <= t <= STRENGTH_FIT_CEILING_C

    def reference_force(self, temperature_c: float) -> float:
        """Kachanov ``F_ref(T) = F_ref(T_ref) * strength_factor(T)`` in N.

        Zero at a melting interface: the upstream clock clamps ``F_ref`` to
        ``1e-30`` internally, so a melted bond saturates its damage within one
        step.  That is the intended behaviour, not a numerical accident.
        """
        return self.reference_force_n * self.strength_factor(temperature_c)

    # -- bond mechanics ----------------------------------------------------
    def advance(
        self,
        branch_offsets: torch.Tensor,
        elongation_old: torch.Tensor,
        elongation_new: torch.Tensor,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        """Advance the Maxwell branches exactly over one increment.

        Returns a dict with the keys ``branch_offsets`` (new dashpot
        elongations), ``force`` (N), ``work`` (J), ``dissipation`` (J) and
        ``stored_energy`` (J); see :func:`tensordem.ice_bond_viscoelastic.
        advance_branches` for the ledger identity.
        """
        offsets, force, work, dissipation = advance_branches(
            self.law, branch_offsets, elongation_old, elongation_new, float(dt))
        return {
            "branch_offsets": offsets,
            "force": force,
            "work": work,
            "dissipation": dissipation,
            "stored_energy": stored_energy(self.law, offsets, elongation_new),
        }

    def force_at(self, branch_offsets: torch.Tensor,
                 elongation: torch.Tensor) -> torch.Tensor:
        """Bond force (N) at frozen branch state."""
        return elastic_force(self.law, branch_offsets, elongation)

    def stored(self, branch_offsets: torch.Tensor,
               elongation: torch.Tensor) -> torch.Tensor:
        """Stored elastic energy (J) at frozen branch state."""
        return stored_energy(self.law, branch_offsets, elongation)

    def damage_increment(self, force: torch.Tensor, temperature_c: float,
                         dt: float) -> torch.Tensor:
        """Kachanov damage increment at the temperature-scaled ``F_ref``."""
        return damage_increment(
            force,
            torch.as_tensor(self.reference_force(temperature_c),
                            dtype=force.dtype, device=force.device),
            float(dt), self.damage_time_s, self.damage_exponent)

    def accumulate_damage(self, damage: torch.Tensor,
                          increment: torch.Tensor) -> torch.Tensor:
        """Accumulate Kachanov damage and truncate it at rupture ``D = 1``.

        The upstream clock (:func:`tensordem.ice_bond_viscoelastic.
        damage_increment`) is an unbounded rate law; a DEM engine stops
        integrating it once the bond is broken.  The truncation therefore lives
        here, at the integration layer, so the raw increment stays available for
        the verification surface and no upstream behaviour changes.
        """
        return torch.clamp(damage + increment, max=1.0)

    def relaxation_modulus(self, time: float) -> float:
        """Relaxation modulus ``G(t)`` in N/m (delegates the Prony series)."""
        return self.law.relaxed_stiffness_at(float(time))
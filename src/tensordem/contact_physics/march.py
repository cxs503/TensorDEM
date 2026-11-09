"""Integrated DEM contact-physics march: bond + lubrication + statistics.

Campaign node: ``P121W5-N11`` (maturation: integrated verification of the
contact-physics nodes ``N02`` bond / ``N08`` lubrication / ``N05`` statistics).

One :class:`ContactPhysicsMarch` advances a *single* operating point through all
three matured modules in one loop and returns one record per step:

1. :class:`~tensordem.contact_physics.lubrication.LubricationContactCoupling`
   advances the melt-lubricated sliding contact (delegating N08's coupled
   Newton + Stefan solver) and yields the interface temperature ``T_int``;
2. :class:`~tensordem.contact_physics.bond.TemperatureDependentBond` advances the
   Maxwell-Wiechert branch state over the prescribed strain increment and
   accumulates Kachanov damage at ``F_ref(T_int)`` (delegating N02);
3. :class:`~tensordem.contact_physics.statistics.RidgeContactStatistics` samples
   the ridge morphology and the contact-force network of the accumulated window
   (delegating N05).

The linkage is deliberately **directed and non-invasive**: the temperature comes
from the lubrication module and is consumed by the bond module; the bond forces
feed the statistics sampler.  No module's internal numerics are modified, which
is why an integrated run reproduces the three standalone runs exactly (the
integration test asserts ``<= 1e-10`` relative and measures 0.0).  Making the
loop genuinely *two-way* (bond force -> contact pressure -> friction heat) is a
physics change and is therefore out of scope for a maturation node; it is listed
as a known limitation in ``docs/contact_physics_module.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..interface_thermal.coupling import IceSteelContact
from ..ridge_dem import RidgeKeel, RidgeMaterial
from .bond import TemperatureDependentBond
from .lubrication import LubricationContactCoupling
from .statistics import RidgeContactStatistics, RidgeMorphologyCase

__all__ = ["OperatingPoint", "ContactPhysicsMarch"]


def _default_bond() -> TemperatureDependentBond:
    return TemperatureDependentBond.from_unrelaxed_stiffness(
        5.0e6, reference_force_n=1.0e3)


def _default_ridge() -> RidgeMorphologyCase:
    return RidgeMorphologyCase(
        keel=RidgeKeel(keel_depth=8.0, keel_width=6.0, porosity=0.30),
        material=RidgeMaterial(particle_radius=0.30))


@dataclass(frozen=True)
class OperatingPoint:
    """The工况 shared by the three contact-physics modules.

    Parameters
    ----------
    contact:
        Sliding ice-steel contact (N08 input).
    bond:
        Temperature-dependent bond law (N02 input).
    ridge:
        Ridge realisation whose morphology is sampled (N05 input).
    strain_rate_1_s:
        Prescribed bond strain rate ``edot``, 1/s.
    n_bonds:
        Number of bonds in the bundle; their gauge lengths are drawn once from a
        seeded uniform distribution, so the force scatter of the network is
        reproducible.
    gauge_length_range_m:
        ``(min, max)`` gauge length of the bond bundle, m.
    dt_s:
        Default step, s.
    statistics_stride:
        Sample a force window every ``statistics_stride`` steps.
    seed:
        Seed of the gauge-length draw (never of the physics).
    """

    contact: IceSteelContact = field(default_factory=IceSteelContact)
    bond: TemperatureDependentBond = field(default_factory=_default_bond)
    ridge: RidgeMorphologyCase = field(default_factory=_default_ridge)
    strain_rate_1_s: float = 1.0e-3
    n_bonds: int = 64
    gauge_length_range_m: tuple[float, float] = (0.5, 1.5)
    dt_s: float = 2.0e-4
    statistics_stride: int = 10
    seed: int = 20261108

    def __post_init__(self) -> None:
        if not isinstance(self.contact, IceSteelContact):
            raise TypeError("contact must be an IceSteelContact")
        if not isinstance(self.bond, TemperatureDependentBond):
            raise TypeError("bond must be a TemperatureDependentBond")
        if not isinstance(self.ridge, RidgeMorphologyCase):
            raise TypeError("ridge must be a RidgeMorphologyCase")
        for name in ("strain_rate_1_s", "dt_s"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if int(self.n_bonds) < 4:
            raise ValueError("n_bonds must be at least 4 (the force PDF needs four contacts)")
        low, high = (float(x) for x in self.gauge_length_range_m)
        if not (0.0 < low <= high and math.isfinite(high)):
            raise ValueError("gauge_length_range_m must be 0 < low <= high")
        if int(self.statistics_stride) < 1:
            raise ValueError("statistics_stride must be at least 1")

    def gauge_lengths(self) -> np.ndarray:
        """Deterministic per-bond gauge lengths, m."""
        rng = np.random.default_rng(int(self.seed))
        low, high = self.gauge_length_range_m
        return rng.uniform(low, high, int(self.n_bonds))


class ContactPhysicsMarch:
    """March one operating point through the three modules in one loop.

    Parameters
    ----------
    point:
        The shared operating point (:class:`OperatingPoint`).
    max_iterations, tolerance, allow_melting:
        Forwarded to the N08 coupled solver unchanged.
    statistics:
        Optional pre-assembled :class:`RidgeContactStatistics`; the ridge
        lattice build is state independent, so reusing one sampler across many
        marches is legitimate (and is what the overhead benchmark does).

    Attributes
    ----------
    bond, lubrication, statistics:
        The three module adapters, all reading the same operating point.
    """

    def __init__(self, point: OperatingPoint | None = None, *,
                 max_iterations: int = 40, tolerance: float = 1e-6,
                 allow_melting: bool = True,
                 statistics: RidgeContactStatistics | None = None) -> None:
        self.point = point if point is not None else OperatingPoint()
        if not isinstance(self.point, OperatingPoint):
            raise TypeError("point must be an OperatingPoint")
        self.bond = self.point.bond
        self.lubrication = LubricationContactCoupling(
            self.point.contact, max_iterations=max_iterations,
            tolerance=tolerance, allow_melting=allow_melting)
        if statistics is not None and not isinstance(statistics,
                                                     RidgeContactStatistics):
            raise TypeError("statistics must be a RidgeContactStatistics")
        # the ridge realisation is expensive (bonded lattice build) and does not
        # depend on the march state, so a caller may assemble it once and reuse it
        self.statistics = (statistics if statistics is not None
                           else RidgeContactStatistics(self.point.ridge))
        self._gauge = torch.as_tensor(self.point.gauge_lengths(), dtype=torch.float64)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> "ContactPhysicsMarch":
        """Return to the initial state (cold interface, virgin bonds)."""
        self.lubrication.reset()
        self._strain = 0.0
        self._offsets = torch.zeros((int(self.point.n_bonds),
                                     self.bond.law.n_branches), dtype=torch.float64)
        self._damage = torch.zeros(int(self.point.n_bonds), dtype=torch.float64)
        self._broken = torch.zeros(int(self.point.n_bonds), dtype=torch.bool)
        self._elongation = torch.zeros(int(self.point.n_bonds), dtype=torch.float64)
        self._cumulative = {"work_j": 0.0, "dissipation_j": 0.0,
                            "stored_energy_j": 0.0}
        self._history: dict[str, list] = {}
        self._force_samples: list[float] = []
        self._steps = 0
        return self

    @property
    def time(self) -> float:
        """Elapsed physical time, s (owned by the lubrication module)."""
        return self.lubrication.time

    @property
    def history(self) -> dict[str, list]:
        """Per-step records of the current run."""
        return self._history

    # -- one integrated step -------------------------------------------------
    def step(self, dt: float | None = None) -> dict[str, Any]:
        """Advance all three modules by one increment and return the record.

        The order is fixed: lubrication (N08) -> bond (N02) using that
        temperature -> statistics sampling (N05) every ``statistics_stride``
        steps.  Any other order would change the numbers, so it is not exposed.
        """
        dt = float(self.point.dt_s if dt is None else dt)
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be finite and positive")

        # 1. N08: melt-lubricated contact (two-way coupled inside the module)
        lubricated = self.lubrication.step(dt)
        temperature = float(lubricated["interface_temperature_c"])

        # 2. N02: bond advance at that interface temperature
        strain_old = self._strain
        strain_new = strain_old + float(self.point.strain_rate_1_s) * dt
        elongation_old = self._elongation
        elongation_new = torch.as_tensor(strain_new, dtype=torch.float64) * self._gauge
        advanced = self.bond.advance(self._offsets, elongation_old, elongation_new, dt)
        stored_old = self.bond.stored(self._offsets, elongation_old)
        # force before rupture masking: a broken bond still accumulates its last
        # increment, then stops (truncated at D = 1 by the adapter)
        force_raw = advanced["force"]
        damage_increment = self.bond.damage_increment(force_raw, temperature, dt)
        self._damage = self.bond.accumulate_damage(self._damage, damage_increment)
        self._broken = self._damage >= 1.0
        self._offsets = advanced["branch_offsets"]
        self._elongation = elongation_new
        self._strain = strain_new

        work = float(advanced["work"].sum())
        dissipation = float(advanced["dissipation"].sum())
        stored_new = float(advanced["stored_energy"].sum())
        self._cumulative["work_j"] += work
        self._cumulative["dissipation_j"] += dissipation
        self._cumulative["stored_energy_j"] += stored_new - float(stored_old.sum())

        force = force_raw.masked_fill(self._broken, 0.0)
        record = {
            "step": self._steps,
            "time": self.lubrication.time,
            "interface_temperature_c": temperature,
            "mu": float(lubricated["mu"]),
            "melt_fraction": float(lubricated.get(
                "melt_fraction", self.lubrication.melt_fraction(temperature))),
            "load_bearing_fraction": float(lubricated["load_bearing_fraction"]),
            "friction_flux_w_m2": float(lubricated["friction_flux_w_m2"]),
            "film_thickness_m": float(lubricated["film_thickness_m"]),
            "film_normal_stiffness_n_m": float(lubricated["film_normal_stiffness_n_m"]),
            "bond_force_mean_n": float(force.mean()),
            "broken_bonds": int(self._broken.sum()),
            "bond_force_max_n": float(force.max()),
            "bond_stiffness_ratio": (float(force.mean() / elongation_new.mean())
                                     if float(elongation_new.mean()) > 0.0 else 0.0),
            "bond_strain": float(strain_new),
            "damage_increment_max": float(damage_increment.max()),
            "damage_max": float(self._damage.max()),
            "work_j": work,
            "dissipation_j": dissipation,
            "stored_energy_j": stored_new,
            "coupled_step_mode": str(lubricated["mode"]),
            "coupled_iterations": int(lubricated["iterations"]),
        }
        for key, value in record.items():
            self._history.setdefault(key, []).append(value)
        if self._steps % int(self.point.statistics_stride) == 0:
            self._force_samples.extend(float(x) for x in force.abs().tolist())
        self._steps += 1
        return record

    # -- a whole run ---------------------------------------------------------
    def run(self, n_steps: int) -> dict[str, Any]:
        """March ``n_steps`` increments and return the run summary.

        Returns
        -------
        dict with ``steps``, ``time_s``, ``final`` (last record),
        ``histories`` (per-step lists), ``energy`` (cumulative work /
        dissipation / stored energy and the ledger error
        ``(W - dU - Phi)/max(|W|, 1)``), ``statistics`` (N05 sampling) and
        ``invariants`` (the self-consistency checks the integration test pins).
        """
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        for _ in range(n_steps):
            self.step()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        """Summary of the current run state without advancing it."""
        history = self._history
        final = {key: values[-1] for key, values in history.items()} if history else {}
        work = self._cumulative["work_j"]
        dissipation = self._cumulative["dissipation_j"]
        stored = self._cumulative["stored_energy_j"]
        ledger_error = (work - stored - dissipation) / max(abs(work), 1.0)
        temperatures = [float(x) for x in history.get("interface_temperature_c", [])]
        films = [float(x) for x in history.get("film_thickness_m", [])]
        damages = [float(x) for x in history.get("damage_max", [])]
        positive = [value for value in self._force_samples if value > 0.0]
        statistics = (self.statistics.sample(positive)
                      if len(positive) >= 4 else None)
        return {
            "steps": int(self._steps),
            "time_s": float(self.lubrication.time),
            "final": final,
            "histories": {key: list(values) for key, values in history.items()},
            "energy": {
                "work_j": work,
                "dissipation_j": dissipation,
                "stored_energy_j": stored,
                "ledger_error": ledger_error,
            },
            "statistics": statistics,
            "force_samples": {"total": len(self._force_samples),
                              "positive": len(positive)},
            "invariants": {
                "interface_temperature_min_c": min(temperatures) if temperatures else
                float(self.point.contact.initial_temperature_c),
                "interface_temperature_max_c": max(temperatures) if temperatures else
                float(self.point.contact.initial_temperature_c),
                "interface_temperature_within_bounds": all(
                    self.point.contact.initial_temperature_c - 1e-12 <= t
                    <= self.point.contact.t_melt + 1e-12 for t in temperatures),
                "film_monotone": all(b >= a - 1e-18 for a, b in zip(films, films[1:])),
                "damage_monotone": all(b >= a - 1e-18 for a, b in zip(damages, damages[1:])),
                "mu_within_bounds": all(
                    0.0 <= float(m) <= self.point.contact.mu_cold
                    for m in history.get("mu", [])),
            },
        }
"""Vectorised mixed-mode cohesive bond element with a reversible history variable.

One call to :meth:`CohesiveBondSet.trial` maps the *separation vector* of every
bond ``(delta_n, delta_t)`` to the traction vector ``(t_n, t_t)`` [Pa] through the
freeze-bond law of :mod:`tensordem.freeze_bond.constitutive`.  The constitutive
branch is a damage model with linear softening,

.. math::

    \\delta_e = \\sqrt{\\delta_n^2 + \\delta_t^2},\\qquad
    t_e = (1 - d)\\, k_0 \\, \\delta_e,\\qquad
    d = d(\\delta_{max};\\, \\delta_0, \\delta_f)

with ``k0`` the bond stiffness per unit area [Pa/m], ``delta_0 = sig_p / k0`` the
elastic limit, ``delta_f = 2 G_f / sig_p`` the softening length and ``d`` chosen so
that the softening branch is *linear in the separation*:

.. math::

    d = 1 - \\frac{\\delta_0 (\\delta_f - \\delta_{max})}
                 {\\delta_{max} (\\delta_f - \\delta_0)},\\qquad
    t_e(\\delta_e > \\delta_0) = \\sigma_p \\frac{\\delta_f - \\delta_e}{\\delta_f - \\delta_0}.

Two mechanical properties follow and are verified in
``tests/test_freeze_bond_constitutive.py`` / ``tests/test_freeze_bond_energy.py``:

1. the **peak traction is exactly the closed-form strength** ``sig_p`` (the
   element is self-consistent with :class:`FreezeBondLaw` to machine precision,
   which is a self-consistency check, *not* a physical validation);
2. the work spent per unit bond area when ``delta_max`` reaches ``delta_f`` is
   exactly ``G_f``: ``0.5 sig_p delta_0 + 0.5 sig_p (delta_f - delta_0)
   = 0.5 sig_p delta_f = G_f``.  Unloading follows the secant to the origin, so
   the recoverable energy is ``0.5 t_e delta_e`` and everything else is
   dissipated.

History variables
-----------------
* ``delta_max`` -- irreversible peak effective separation (damage memory),
* ``dissipated_J_m2`` / ``work_J_m2`` -- committed work per unit area,
* ``healed_events`` -- number of *re-freezing* events.  A fully broken bond that
  is held in contact for ``t_hold`` under a supercooling ``Delta T`` regrows a
  neck whose strength is again given by the same kinetic law (a fresh neck, so
  its consolidation age restarts).  :meth:`refreeze` therefore resets the damage
  memory of the selected bonds while **keeping the dissipated energy** -- the
  bond is reversible in strength but not in energy, which is the honest
  bookkeeping for a fracture-and-heal cycle.

Mode mixing
-----------
``sig_p = sig_s + (sig_t - sig_s) * delta_n / delta_e`` for opening bonds
(``sig_t`` tension, ``sig_s = kappa_s sig_t`` shear) and ``sig_p = sig_c`` (bulk
crush strength) for closing bonds.  The energy bookkeeping is incremental
(trapezoidal along the committed path) so it stays exact for pure modes and
conservative for mixed paths; the residual is reported, never hidden.

Everything is SI and dimensionless-free: separations in m, tractions in Pa,
energies per *unit bond area* in J/m^2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constitutive import FreezeBondLaw

__all__ = ["CohesiveBondSet", "BondEnvironment"]


@dataclass(frozen=True)
class BondEnvironment:
    """Per-bond freezing state driving the strength law."""

    delta_T_K: np.ndarray
    salinity_ppt: np.ndarray
    consolidation_time_s: np.ndarray

    @staticmethod
    def uniform(n: int, *, delta_T_K: float = 5.0, salinity_ppt: float = 0.0,
                consolidation_time_s: float = 3600.0) -> "BondEnvironment":
        return BondEnvironment(
            delta_T_K=np.full(n, float(delta_T_K)),
            salinity_ppt=np.full(n, float(salinity_ppt)),
            consolidation_time_s=np.full(n, float(consolidation_time_s)),
        )


class CohesiveBondSet:
    """Cohesive law state for ``n`` independent bonds (vectorised)."""

    def __init__(
        self,
        law: FreezeBondLaw,
        environment: BondEnvironment,
        *,
        stiffness_pa_per_m: float,
        areas_m2=None,
    ) -> None:
        if stiffness_pa_per_m <= 0.0:
            raise ValueError("stiffness_pa_per_m must be positive")
        self.law = law
        self.environment = environment
        self.stiffness = float(stiffness_pa_per_m)
        n = self.environment.delta_T_K.size
        if not (self.environment.salinity_ppt.size == n
                and self.environment.consolidation_time_s.size == n):
            raise ValueError("environment arrays must share one length")
        self.n = n
        #: Nominal bond cross-section [m^2]; converts the per-unit-area energy
        #: ledger (J/m^2) into joules.  Defaults to unit area so the scalar
        #: element tests stay dimensionless-friendly.
        self.areas_m2 = (np.ones(n) if areas_m2 is None
                         else np.broadcast_to(np.asarray(areas_m2, dtype=np.float64),
                                              (n,)).copy())
        # Irreversible history variables.
        self.delta_max = np.zeros(n)
        self.delta_committed = np.zeros(n)
        self.traction_committed = np.zeros(n)
        self.work_J_m2 = np.zeros(n)
        self.dissipated_J_m2 = np.zeros(n)
        self.healed_events = np.zeros(n, dtype=np.int64)
        self.fracture_energy_J_m2 = np.zeros(n)
        self.was_broken = np.zeros(n, dtype=bool)

    # ------------------------------------------------------------ strengths
    def strengths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(sigma_tension, sigma_shear, sigma_crush)`` per bond [Pa]."""
        env = self.environment
        sig_t = self.law.tension_strength(
            env.delta_T_K, env.salinity_ppt, env.consolidation_time_s)
        sig_t = np.broadcast_to(np.asarray(sig_t, dtype=np.float64), (self.n,)).copy()
        sig_s = self.law.shear_to_tension_ratio * sig_t
        sig_c = np.full(self.n, self.law.crush_strength_pa)
        return sig_t, sig_s, sig_c

    def set_environment(
        self,
        *,
        delta_T_K=None,
        salinity_ppt=None,
        consolidation_time_s=None,
    ) -> None:
        """Overwrite the freezing state (no mechanical state is touched)."""
        env = self.environment
        self.environment = BondEnvironment(
            delta_T_K=env.delta_T_K if delta_T_K is None
            else np.broadcast_to(np.asarray(delta_T_K, dtype=np.float64), (self.n,)).copy(),
            salinity_ppt=env.salinity_ppt if salinity_ppt is None
            else np.broadcast_to(np.asarray(salinity_ppt, dtype=np.float64), (self.n,)).copy(),
            consolidation_time_s=env.consolidation_time_s if consolidation_time_s is None
            else np.broadcast_to(np.asarray(consolidation_time_s, dtype=np.float64),
                                 (self.n,)).copy(),
        )

    # -------------------------------------------------------------- damage
    def _damage(self, delta_max: np.ndarray, sig_p: np.ndarray,
                fracture_energy: np.ndarray) -> np.ndarray:
        k0 = self.stiffness
        # A bond with zero strength (no supercooling: no freeze bond at all) has
        # no cohesive capacity by definition.
        zero = sig_p <= 0.0
        delta_0 = sig_p / k0
        delta_f = np.divide(2.0 * fracture_energy, sig_p,
                            out=np.zeros_like(sig_p), where=sig_p > 0.0)
        # Brittle limit: G_f smaller than the elastic energy at peak means the
        # bond snaps at delta_0 (delta_f <= delta_0); treat as d = 1 beyond peak.
        brittle = delta_f <= delta_0
        span = np.where(brittle, 1.0, delta_f - delta_0)
        frac = (delta_max - delta_0) / span
        d = np.where(delta_max <= delta_0, 0.0,
                     np.where(brittle, 1.0,
                              1.0 - delta_0 * (delta_f - delta_max)
                              / (np.maximum(delta_max, 1e-300) * span)))
        return np.clip(np.where(zero, 1.0, d), 0.0, 1.0)

    # --------------------------------------------------------------- trial
    def trial(self, delta_n, delta_t) -> dict[str, np.ndarray]:
        """Traction for a trial separation; **mutates nothing**.

        Returns a dict with ``t_n``, ``t_t`` [Pa], ``traction`` (magnitude),
        ``delta_e``, ``damage``, ``delta_max`` (trial history variable),
        ``sigma_p``, ``fracture_energy`` and ``broken``.
        """
        dn = np.asarray(delta_n, dtype=np.float64)
        dt_ = np.asarray(delta_t, dtype=np.float64)
        dn, dt_ = np.broadcast_arrays(dn, dt_)
        if dn.shape != (self.n,):
            raise ValueError(f"expected separation arrays of length {self.n}, got {dn.shape}")
        # One-sided cohesive element: the bond resists *tension* and *shear*.
        # The compressive normal branch is carried by the ice-ice disk contacts of
        # the assembly (a thin freeze-bond neck between two floe faces carries
        # little compression once the faces touch), so the opening that drives
        # damage is ``max(delta_n, 0)`` and a closing bond does no normal work.
        dn_pos = np.maximum(dn, 0.0)
        delta_e = np.sqrt(dn_pos * dn_pos + dt_ * dt_)
        sig_t, sig_s, _sig_c = self.strengths()
        mix = np.clip(np.divide(dn_pos, delta_e,
                                out=np.zeros_like(delta_e), where=delta_e > 0.0),
                      0.0, 1.0)
        # mode mix: 0 = pure shear, 1 = pure opening
        sig_p = np.maximum(sig_s + (sig_t - sig_s) * mix, 0.0)
        if self.law.softening_ratio is not None:
            # energetically consistent for every strength: G_f = 0.5 ratio sigma^2/k0
            gf = self.law.fracture_energy_at(sig_p, self.stiffness)
        else:
            gf_normal = self.law.fracture_energy("tension")
            gf = self.law.fracture_energy("shear") + (
                gf_normal - self.law.fracture_energy("shear")) * mix
        delta_max = np.maximum(self.delta_max, delta_e)
        d = self._damage(delta_max, sig_p, gf)
        t_e = np.where(sig_p > 0.0, (1.0 - d) * self.stiffness * delta_e, 0.0)
        inv = np.divide(1.0, delta_e, out=np.zeros_like(delta_e), where=delta_e > 0.0)
        return {
            "t_n": t_e * dn_pos * inv,
            "t_t": t_e * dt_ * inv,
            "traction": t_e,
            "delta_e": delta_e,
            "damage": d,
            "delta_max": delta_max,
            "sigma_p": sig_p,
            "fracture_energy": gf,
            "broken": d >= 1.0 - 1e-12,
        }

    # ---------------------------------------------------------- dissipation
    def envelope_work(self, delta: np.ndarray, sig_p: np.ndarray,
                      fracture_energy: np.ndarray) -> np.ndarray:
        """Monotone-envelope work per unit area ``W(delta) = int_0^delta t ddelta``.

        For the piecewise-linear law (elastic up to ``delta_0``, linear softening
        to ``delta_f``) this integral is closed form:

        * ``delta <= delta_0``: ``0.5 k0 delta^2``
        * ``delta_0 < delta <= delta_f``:
          ``0.5 sig_p delta_0 + sig_p/(delta_f - delta_0) *
          [delta_f (delta - delta_0) - (delta^2 - delta_0^2)/2]``
        * ``delta > delta_f``: ``G_f`` (the bond is fully broken).

        At ``delta_f`` the expression equals ``0.5 sig_p delta_f = G_f`` exactly,
        which is the identity the element test checks.
        """
        k0 = self.stiffness
        delta = np.asarray(delta, dtype=np.float64)
        sig_p = np.maximum(np.asarray(sig_p, dtype=np.float64), 0.0)
        gf = np.asarray(fracture_energy, dtype=np.float64)
        out = np.zeros_like(delta)
        live = sig_p > 0.0
        d0 = np.divide(sig_p, k0, out=np.zeros_like(sig_p), where=live)
        df = np.divide(2.0 * gf, sig_p, out=np.zeros_like(sig_p), where=live)
        elastic = live & (delta <= d0)
        out = np.where(elastic, 0.5 * k0 * delta ** 2, out)
        soften = live & (delta > d0) & (df > d0)
        span = np.where(soften, df - d0, 1.0)
        soft_curve = sig_p / span * (df * (delta - d0)
                                     - 0.5 * (delta ** 2 - d0 ** 2))
        out = np.where(soften,
                       np.minimum(0.5 * sig_p * d0 + soft_curve, gf),
                       out)
        brittle = live & (delta > d0) & (df <= d0)
        out = np.where(brittle, 0.5 * k0 * d0 ** 2, out)
        return out

    def dissipated_for(self, delta_max: np.ndarray, sig_p: np.ndarray,
                       fracture_energy: np.ndarray) -> np.ndarray:
        """Dissipated energy per unit area at the current damage state.

        ``W_diss(delta_max) = W_env(delta_max) - 0.5 t_e(delta_max) delta_max``:
        the work done minus the recoverable part.  It is a *state function of the
        history variable* ``delta_max`` (monotone, irreversible), which is what
        makes the multi-bond energy ledger a well-posed check instead of an
        accumulation of path artefacts.  At complete failure it returns ``G_f``.
        """
        keff = self.stiffness * (1.0 - self._damage(delta_max, sig_p, fracture_energy))
        t_e = np.maximum(keff * delta_max, 0.0)
        work = self.envelope_work(delta_max, sig_p, fracture_energy)
        return np.maximum(work - 0.5 * t_e * delta_max, 0.0)

    # -------------------------------------------------------------- commit
    def commit(self, state: dict[str, np.ndarray]) -> dict[str, float]:
        """Commit a converged trial state and update the energy ledger.

        The dissipated energy is *recomputed* from the history variable
        (:meth:`dissipated_for`) rather than accumulated from finite increments:
        the damage model is irreversible, so ``W_diss`` is a monotone function of
        ``delta_max`` and the increment is its change.  Work is reported
        incrementally (trapezoid along the committed path) for diagnostics only.
        """
        d_e = state["delta_e"]
        t_e = state["traction"]
        d_work = 0.5 * (self.traction_committed + t_e) * (d_e - self.delta_committed)
        self.work_J_m2 += d_work
        diss_now = self.dissipated_for(state["delta_max"], state["sigma_p"],
                                       state["fracture_energy"])
        diss_new = np.maximum(self.dissipated_J_m2, diss_now)
        d_diss = diss_new - self.dissipated_J_m2
        self.dissipated_J_m2 = diss_new
        newly_broken = state["broken"] & ~self.was_broken
        self.fracture_energy_J_m2 += np.where(newly_broken,
                                              state["fracture_energy"], 0.0)
        self.was_broken = state["broken"].copy()
        self.delta_max = state["delta_max"]
        self.delta_committed = d_e.copy()
        self.traction_committed = t_e.copy()
        area = self.areas_m2
        return {
            "work_increment_J_m2": float(np.sum(d_work)),
            "dissipated_increment_J_m2": float(np.sum(d_diss)),
            "total_work_J_m2": float(np.sum(self.work_J_m2)),
            "total_dissipated_J_m2": float(np.sum(self.dissipated_J_m2)),
            "total_work_J": float(np.sum(self.work_J_m2 * area)),
            "total_dissipated_J": float(np.sum(self.dissipated_J_m2 * area)),
            "fracture_energy_J": float(np.sum(self.fracture_energy_J_m2 * area)),
            "fracture_energy_J_m2": float(np.sum(self.fracture_energy_J_m2)),
            "broken_bonds": int(np.count_nonzero(state["broken"])),
        }

    # -------------------------------------------------------- reference
    def reset_reference(self, delta_n, delta_t) -> None:
        """Re-zero the history and the ledger at the as-built reference state.

        Used once, after the specimen has been equilibrated under its wall
        preload: the committed separations and tractions become the reference,
        ``work``/``dissipated`` are set to zero and the damage memory is set to
        the reference separation (so the preload does not pre-damage the bonds).
        Documented as a *ledger origin*, not as a physical process.
        """
        state = self.trial(delta_n, delta_t)
        self.delta_max = state["delta_e"].copy()
        self.delta_committed = state["delta_e"].copy()
        self.traction_committed = state["traction"].copy()
        self.work_J_m2 = np.zeros(self.n)
        self.dissipated_J_m2 = np.zeros(self.n)
        self.healed_events = np.zeros(self.n, dtype=np.int64)
        self.fracture_energy_J_m2 = np.zeros(self.n)
        self.was_broken = np.zeros(self.n, dtype=bool)

    # ------------------------------------------------------------ refreeze
    def refreeze(self, mask, hold_time_s: float) -> int:
        """Re-freezing of broken bonds: restore strength, keep dissipated energy.

        The selected bonds get a *fresh* neck: the damage memory
        (``delta_max``) and committed path are reset, and the consolidation age
        restarts at ``hold_time_s`` so the regrown strength is
        ``sigma_fb(Delta T, S, hold_time_s)`` from the same kinetic law.
        ``hold_time_s`` must be positive; the dissipated energy of the previous
        fracture is **not** recovered.
        """
        if hold_time_s <= 0.0:
            raise ValueError("hold_time_s must be positive")
        sel = np.broadcast_to(np.asarray(mask, dtype=bool), (self.n,)).copy()
        count = int(np.count_nonzero(sel))
        if count == 0:
            return 0
        self.delta_max[sel] = 0.0
        self.delta_committed[sel] = 0.0
        self.traction_committed[sel] = 0.0
        env = self.environment
        env.consolidation_time_s[sel] = float(hold_time_s)
        self.healed_events[sel] += 1
        self.was_broken[sel] = False
        return count

    # --------------------------------------------------------------- report
    def summary(self) -> dict[str, Any]:
        sig_t, sig_s, sig_c = self.strengths()
        return {
            "n_bonds": int(self.n),
            "stiffness_pa_per_m": self.stiffness,
            "sigma_tension_pa_mean": float(np.mean(sig_t)),
            "sigma_tension_pa_min": float(np.min(sig_t)),
            "sigma_tension_pa_max": float(np.max(sig_t)),
            "sigma_shear_pa_mean": float(np.mean(sig_s)),
            "sigma_crush_pa": float(sig_c[0]) if self.n else float(self.law.crush_strength_pa),
            "delta_max_mean_m": float(np.mean(self.delta_max)) if self.n else 0.0,
            "damage_mean": float(np.mean(self._damage(
                self.delta_max, sig_t, np.full(self.n, self.law.fracture_energy("tension"))))
            ) if self.n else 0.0,
            "broken_bonds": int(np.count_nonzero(self.delta_max > 0.0)) and int(
                np.count_nonzero(self._damage(self.delta_max, sig_t,
                                              np.full(self.n, self.law.fracture_energy("tension")))
                                 >= 1.0 - 1e-12)),
            "dissipated_J_m2_total": float(np.sum(self.dissipated_J_m2)),
            "healed_events": int(np.sum(self.healed_events)),
        }
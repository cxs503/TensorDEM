"""Multi-pass ice resistance decay: rounding vs consolidation, closed to 100 %.

Scenario
--------
A ship transits a brash-ice channel ``N`` times.  Two mechanisms act on the
channel ice at the same time and with *different* clocks:

* **corner attrition** (this module's main content): each transit slides the
  channel ice along the hull for ``s_t`` metres at a corner contact pressure
  ``p``, blunting the floe corners (:mod:`tensordem.floe_attrition.wear`).  The
  effective friction of the channel drops with roundness
  (:mod:`tensordem.floe_attrition.friction`), so the *friction* component of the
  resistance decays -- the "多次通航阻力衰减" of the node.
* **consolidation** (freeze bonding during the rest time between passages): the
  brash layer regains cohesion and carries more load, which *raises* the
  resistance.  It is deliberately modelled as a separate channel with its own
  time constant, so that the two share can be separated instead of being blended.

Resistance model (per metre of waterline width)
-----------------------------------------------
.. math::

    R(N) = R_{crush} + \\mu_{eff}(R(N), C)\\; F_n(b_N) + \\sigma_c\\, b_N\\, A_b

    F_n(b) = \\rho_i g h C L_c (1 + \\beta_c b),   b_N = 1 - e^{-N \\Delta t / t_c}

``R_crush`` is the unchanged breakup term, ``F_n`` the buoyant normal load of the
brash layer (which the consolidated bonds amplify by ``beta_c``), ``sigma_c b A_b``
the freeze-bond cohesion.  ``b`` is the consolidated (bonded) fraction.

Closure and decoupling
----------------------
The decay is split *exactly*:

.. math::

    \\Delta R_{tot}(N) = \\underbrace{[\\mu_0 - \\mu_N] F_n(b_N)}_{\\Delta R_{round}}
                       + \\underbrace{\\{[F_n(0) - F_n(b_N)] \\mu_0 + \\sigma_c b_N A_b\\}}_{\\Delta R_{cons}}

(the two terms are the two terms of the total difference, so they sum to 100 %
by construction -- verified numerically to ``<= 1e-12``).  Because the bonding
amplifies the *same* normal load that the friction multiplies, the two mechanisms
are **not** additive in general: single-mechanism runs (rounding only,
consolidation only) are computed as well and the superposition residual is
reported as a percentage of ``Delta R_tot``.  A non-zero residual means the
mechanisms are coupled and must not be reported as independent contributions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .friction import (InterlockingFriction, SphereBaseline, load_sphere_baseline)
from .geometry import RoundedCube, RoundedPolygon
from .wear import ArchardWear, AttritionHistory, integrate_cube_attrition, integrate_polygon_attrition

__all__ = [
    "TransitScenario",
    "simulate_multi_pass",
    "decay_shares",
]


@dataclass(frozen=True)
class TransitScenario:
    """Channel transit scenario (SI, per metre of waterline width)."""

    concentration: float = 0.5
    n_transits: int = 400
    slip_per_transit: float = 2.5          # m of sliding at a floe corner per transit
    contact_pressure: float = 2.0e5        # Pa on the corner patch
    rest_time_per_transit: float = 300.0   # s between passages (5 min headway)
    brash_thickness: float = 0.5           # m
    density_ice: float = 900.0             # kg/m^3
    gravity: float = 9.81
    contact_length: float = 5.0            # m of hull rubbing the brash layer
    crush_term: float = 4.0e3              # N/m of width, unchanged breakup term
    bond_time_constant: float = 3.6e5      # s (100 h), freeze-bond growth time
    bond_strength_max: float = 2.0e4       # Pa, cohesion added by consolidation
    bond_area: float = 0.15                # m^2 per metre of width actually bonded
    load_coupling: float = 0.30            # beta_c: bonded layer amplifies F_n

    def __post_init__(self) -> None:
        if int(self.n_transits) != self.n_transits or self.n_transits <= 0:
            raise ValueError("n_transits must be a positive integer")
        for name in ("slip_per_transit", "contact_pressure", "rest_time_per_transit",
                     "brash_thickness", "density_ice", "gravity", "contact_length",
                     "crush_term", "bond_time_constant", "bond_strength_max",
                     "bond_area"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("concentration", "load_coupling"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def sliding_total(self) -> float:
        return self.n_transits * self.slip_per_transit

    def bond_fraction(self, n_transits: float) -> float:
        """Consolidated fraction after ``n`` passages, ``1 - exp(-t/t_c)``."""
        if n_transits < 0:
            raise ValueError("n_transits must be non-negative")
        t = n_transits * self.rest_time_per_transit
        return 1.0 - math.exp(-t / self.bond_time_constant)

    def load(self, bond_fraction: float) -> float:
        """Buoyant normal load per metre of width [N/m], amplified by bonding."""
        base = (self.density_ice * self.gravity * self.brash_thickness
                * self.concentration * self.contact_length)
        return base * (1.0 + self.load_coupling * bond_fraction)

    def cohesion(self, bond_fraction: float) -> float:
        return self.bond_strength_max * bond_fraction * self.bond_area

    def as_dict(self) -> dict:
        return {
            "concentration": self.concentration,
            "n_transits": int(self.n_transits),
            "slip_per_transit": self.slip_per_transit,
            "sliding_total": self.sliding_total,
            "contact_pressure": self.contact_pressure,
            "rest_time_per_transit": self.rest_time_per_transit,
            "brash_thickness": self.brash_thickness,
            "crush_term": self.crush_term,
            "bond_time_constant": self.bond_time_constant,
            "bond_strength_max": self.bond_strength_max,
            "load_coupling": self.load_coupling,
        }


def simulate_multi_pass(scenario: TransitScenario, shape, wear: ArchardWear, *,
                        baseline: Optional[SphereBaseline] = None,
                        model: Optional[InterlockingFriction] = None,
                        allow_rounding: bool = True,
                        allow_consolidation: bool = True) -> dict:
    """Run the multi-pass decay for one floe shape and scenario.

    ``shape`` is a :class:`RoundedPolygon` (2D) or :class:`RoundedCube` (3D); the
    returned dict carries the per-transit curves, the exact two-term decay
    decomposition and the contact-energy ledger.
    """
    baseline = baseline or load_sphere_baseline()
    model = model or InterlockingFriction()
    is_2d = isinstance(shape, RoundedPolygon)
    if not is_2d and not isinstance(shape, RoundedCube):
        raise ValueError("shape must be a RoundedPolygon or a RoundedCube")

    if allow_rounding:
        if is_2d:
            history: AttritionHistory = integrate_polygon_attrition(
                shape, wear, pressure=scenario.contact_pressure,
                slip_per_step=scenario.slip_per_transit,
                n_steps=scenario.n_transits)
        else:
            history = integrate_cube_attrition(
                shape, wear, pressure=scenario.contact_pressure,
                slip_per_step=scenario.slip_per_transit,
                n_steps=scenario.n_transits)
        states = np.asarray(history.state)
        indices = np.asarray(history.interlocking)
        worn = np.asarray(history.worn_archard)
    else:
        r = shape.fillet_radius if is_2d else shape.radius
        states = np.full(scenario.n_transits + 1, float(r))
        indices = np.full(scenario.n_transits + 1,
                          float(shape.interlocking_index if is_2d
                                else shape.interlocking_index))
        worn = np.zeros(scenario.n_transits + 1)

    n = np.arange(scenario.n_transits + 1, dtype=float)
    if allow_consolidation:
        bonds = np.array([scenario.bond_fraction(k) for k in n])
    else:
        bonds = np.zeros_like(n)

    mu = np.array([model.mu(idx, scenario.concentration, baseline) for idx in indices])
    load = np.array([scenario.load(b) for b in bonds])
    cohesion = np.array([scenario.cohesion(b) for b in bonds])
    friction = mu * load
    resistance = scenario.crush_term + friction + cohesion
    energy = np.concatenate([[0.0], np.cumsum(mu[1:] * load[1:] * scenario.slip_per_transit)])

    mu0 = model.mu(float(shape.interlocking_index), scenario.concentration, baseline)
    shares = decay_shares(mu[0], mu[-1], load[0], load[-1], cohesion[0], cohesion[-1], mu0)

    return {
        "scenario": scenario.as_dict(),
        "family": "rounded_polygon" if is_2d else "rounded_cube",
        "transits": n.tolist(),
        "state": states.tolist(),
        "interlocking": indices.tolist(),
        "bond_fraction": bonds.tolist(),
        "mu_effective": mu.tolist(),
        "normal_load": load.tolist(),
        "cohesion": cohesion.tolist(),
        "friction_resistance": friction.tolist(),
        "resistance": resistance.tolist(),
        "contact_energy": energy.tolist(),
        "worn_volume": worn.tolist(),
        "shares": shares,
        "initial": {
            "resistance": float(resistance[0]),
            "mu_effective": float(mu[0]),
            "interlocking": float(indices[0]),
        },
        "final": {
            "resistance": float(resistance[-1]),
            "mu_effective": float(mu[-1]),
            "interlocking": float(indices[-1]),
            "decay_fraction": float((resistance[0] - resistance[-1]) / resistance[0]),
            "contact_energy": float(energy[-1]),
            "worn_volume": float(worn[-1]),
        },
    }


def decay_shares(mu_start: float, mu_end: float, load_start: float, load_end: float,
                 cohesion_start: float, cohesion_end: float, mu_reference: float) -> dict:
    """Exact two-term split of the resistance change, closed to 100 %.

    ``mu_reference`` is the *initial* effective friction; it multiplies the load
    change so that the two terms are exactly the two halves of the total
    difference (see the module docstring).
    """
    friction_start = mu_start * load_start
    friction_end = mu_end * load_end
    delta_total = (friction_start - friction_end) + (cohesion_start - cohesion_end)
    delta_round = (mu_start - mu_end) * load_end
    delta_cons = (load_start - load_end) * mu_reference + (cohesion_start - cohesion_end)
    closure = delta_round + delta_cons - delta_total
    scale = abs(delta_total) if abs(delta_total) > 0 else 1.0
    return {
        "delta_total": delta_total,
        "delta_rounding": delta_round,
        "delta_consolidation": delta_cons,
        "closure_residual": closure,
        "closure_relative": abs(closure) / scale,
        "share_rounding_pct": 100.0 * delta_round / delta_total if delta_total != 0 else 0.0,
        "share_consolidation_pct": 100.0 * delta_cons / delta_total if delta_total != 0 else 0.0,
        "share_sum_pct": (100.0 * (delta_round + delta_cons) / delta_total
                          if delta_total != 0 else 0.0),
    }
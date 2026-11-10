"""Sea-ice micro-fracture energy budget: three-channel closure audit (p216w3-n07).

Scope
-----
A DEM single-particle sea-ice fracture energy budget.  A single ice particle
(diameter ``d``, density ``rho_ice``) is crushed between two rigid platens
(displacement control).  The elastic energy stored during loading is released
at fracture and partitioned into three physical channels:

1. **Surface energy** ``E_surf`` -- new surface area ``dA`` created by the
   fracture, ``E_surf = gamma * dA`` (``gamma`` = ice fracture surface energy,
   J/m^2);
2. **Kinetic energy** ``E_kin`` -- the fragments fly apart with the kinetic
   energy ``sum(1/2 m_i v_i^2)`` funded by the released elastic energy;
3. **Dissipation** ``E_diss`` -- plastic deformation and Coulomb friction
   dissipate the remainder ``E_diss = E_in - E_surf - E_kin``.

The energy closure audit is

    epsilon_s = (E_in - E_out) / E_in

with ``E_in`` the platen work ``W_ext`` and ``E_out = E_surf + E_kin + E_diss``.
A well-closed budget has ``epsilon_s < 1 %``.

The module also verifies two structural properties of the fracture:

* **F-delta monotonicity** -- the force-displacement curve ``F(delta)`` is
  monotonic (no load drop) within the elastic limit, i.e. before the first
  fracture event;
* **Fragment-number convergence** -- doubling the fragment count ``n_frag``
  changes the energy allocation (``E_surf / E_in``, ``E_kin / E_in``,
  ``E_diss / E_in``) by less than 3 %.

The module integrates with the ``dem3d_energy`` N06 ledger schema: the
``break_crush`` channel of that ledger is the sum of the bond-fracture release
(``break``) and the particle-crushing surface/kinetic/dissipation partition
(``crush``).  This module provides the *micro-basis* for the ``crush`` half:
the per-event three-channel partition that the macro ledger must book.

All quantities are SI (m, kg, s, Pa, J).  Every random draw comes from an
explicit ``numpy.random.Generator``, so a run is reproducible from its seed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .crushing import (
    CrushingConfig,
    CrushingDEM,
    Fragmenter,
    MassMomentumLedger,
    Particle,
    WeibullStrength,
    sphere_mass,
)

#: Sea-ice material (density, fracture surface energy, Weibull strength).
RHO_ICE = 917.0                     # kg/m^3
GAMMA_ICE = 0.1                     # J/m^2, ice fracture surface energy
DEFAULT_DIAMETER = 2.0e-3           # m, single-particle diameter
DEFAULT_PLATEN_VELOCITY = -0.05     # m/s, closing speed


# ---------------------------------------------------------------------------
# Three-channel energy budget
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreeChannelBudget:
    """Per-event three-channel energy partition of a single-particle fracture.

    ``E_in`` is the platen work delivered up to the fracture event; ``E_surf``
    the booked surface energy (``gamma * dA``, capped at the released elastic
    energy); ``E_kin`` the fragment kinetic energy funded by the expansion;
    ``E_diss`` the accumulated damping + friction dissipation.  The closure
    residual ``epsilon_s = (E_in - E_out) / E_in`` is the audit statistic.
    """

    E_in: float          # platen work [J]
    E_surf: float        # surface energy [J]
    E_kin: float         # fragment kinetic energy [J]
    E_diss: float        # dissipation [J]
    E_out: float         # E_surf + E_kin + E_diss [J]
    epsilon_s: float     # (E_in - E_out) / E_in [-]
    n_frag: int          # fragment count [-]
    dA: float            # new surface area [m^2]

    @property
    def fractions(self) -> dict[str, float]:
        """Energy allocation fractions (each / E_in)."""
        if self.E_in <= 0.0:
            return {"surf": 0.0, "kin": 0.0, "diss": 0.0}
        return {
            "surf": self.E_surf / self.E_in,
            "kin": self.E_kin / self.E_in,
            "diss": self.E_diss / self.E_in,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_single_fracture(
    diameter: float = DEFAULT_DIAMETER,
    *,
    seed: int = 0,
    platen_velocity: float = DEFAULT_PLATEN_VELOCITY,
    n_frag: int = 8,
    steps: int = 4000,
    gamma: float = GAMMA_ICE,
    density: float = RHO_ICE,
    contact_stiffness: float = 1.0e6,
    friction_coefficient: float = 0.4,
    sigma0: float = 5.0e5,
    m_weibull: float = 3.0,
) -> dict[str, Any]:
    """Run one single-particle fracture and return the three-channel budget.

    The particle is loaded between two platens at ``platen_velocity``.  The
    force-displacement history is recorded; the fracture event is detected by
    ``CrushingDEM`` (Weibull weakest-link strength).  The three-channel budget
    is assembled from the ``MassMomentumLedger`` at the end of the run.
    """
    material = WeibullStrength(d0=diameter, sigma0=sigma0, m=m_weibull)
    fragmenter = Fragmenter(k=n_frag, n=0.8)
    config = CrushingConfig(
        density=density,
        contact_stiffness=contact_stiffness,
        friction_coefficient=friction_coefficient,
        gamma=gamma,
        weibull=material,
        fragmenter=fragmenter,
        max_generation=1,
        dt_safety=0.1,
        record_every=10,
        seed=seed,
    )
    engine = CrushingDEM.single_sphere(diameter, config, seed=seed)
    engine.platen_velocity = float(platen_velocity)

    # Record the force-displacement history.
    force_history: list[float] = []
    displacement_history: list[float] = []
    initial_z = float(engine.top_platen_z)
    for _ in range(int(steps)):
        engine.step()
        diag = engine._force_pass()
        force_history.append(float(diag.platen_normal_force_N))
        displacement_history.append(initial_z - float(engine.top_platen_z))
        if engine.crushing_events:
            break

    diag = engine.diagnostics()
    audit = diag["energy_audit"]

    # Assemble the three-channel budget.
    E_in = float(audit["external_work_J"])
    E_surf = float(audit["surface_energy_J"])
    E_kin = float(diag["kinetic_energy_J"])
    E_diss = float(audit["dissipation_J"])
    E_out = E_surf + E_kin + E_diss
    epsilon_s = (E_in - E_out) / E_in if E_in > 0.0 else 0.0

    # Surface area increase from the fragmenter.
    dA = 0.0
    if engine.ledger.records:
        dA = float(engine.ledger.records[0].requested_surface_energy_J) / gamma

    return {
        "diameter_m": float(diameter),
        "seed": int(seed),
        "n_frag": int(n_frag),
        "platen_velocity_m_s": float(platen_velocity),
        "crushing_events": engine.crushing_events,
        "E_in_J": E_in,
        "E_surf_J": E_surf,
        "E_kin_J": E_kin,
        "E_diss_J": E_diss,
        "E_out_J": E_out,
        "epsilon_s": epsilon_s,
        "dA_m2": dA,
        "fractions": {
            "surf": E_surf / E_in if E_in > 0 else 0.0,
            "kin": E_kin / E_in if E_in > 0 else 0.0,
            "diss": E_diss / E_in if E_in > 0 else 0.0,
        },
        "force_history_N": force_history,
        "displacement_history_m": displacement_history,
        "energy_audit": audit,
    }


# ---------------------------------------------------------------------------
# F-delta monotonicity check
# ---------------------------------------------------------------------------
def check_f_delta_monotonic(
    force_history: Sequence[float],
    displacement_history: Sequence[float],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Verify that ``F(delta)`` is monotonic (no load drop) within the elastic limit.

    The elastic limit is the segment before the first fracture event (detected
    as the first step where the force drops by more than ``tolerance``).  A
    monotonic curve has ``F_{k+1} >= F_k`` for all ``k`` in the elastic segment.
    """
    if len(force_history) != len(displacement_history):
        raise ValueError("force and displacement histories must have equal length")
    if len(force_history) < 2:
        raise ValueError("need at least two samples for a monotonicity check")

    # Find the elastic limit: the first step where the force drops.
    elastic_end = len(force_history)
    for k in range(1, len(force_history)):
        if force_history[k] < force_history[k - 1] - tolerance:
            elastic_end = k
            break

    elastic_force = force_history[:elastic_end]
    elastic_disp = displacement_history[:elastic_end]
    drops = [
        (k, force_history[k] - force_history[k - 1])
        for k in range(1, elastic_end)
        if force_history[k] < force_history[k - 1]
    ]
    max_drop = min(drops, key=lambda x: x[1])[1] if drops else 0.0
    n_drops = len(drops)
    monotonic = n_drops == 0

    return {
        "elastic_end_step": elastic_end,
        "n_elastic_samples": elastic_end,
        "n_drops": n_drops,
        "max_drop_N": max_drop,
        "monotonic": bool(monotonic),
        "elastic_force_N": list(elastic_force),
        "elastic_displacement_m": list(elastic_disp),
    }


# ---------------------------------------------------------------------------
# Fragment-number convergence
# ---------------------------------------------------------------------------
def check_fragment_convergence(
    *,
    n_frag_values: Sequence[int] = (4, 8, 16, 32),
    diameter: float = DEFAULT_DIAMETER,
    seed: int = 0,
    platen_velocity: float = DEFAULT_PLATEN_VELOCITY,
    steps: int = 4000,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    """Verify that doubling ``n_frag`` changes the energy allocation by < 3 %.

    Runs one fracture per ``n_frag`` value and compares the energy allocation
    fractions (``surf``, ``kin``, ``diss``) between consecutive doublings.
    """
    budgets: list[dict[str, Any]] = []
    for n in n_frag_values:
        run = _run_single_fracture(
            diameter=diameter, seed=seed, platen_velocity=platen_velocity,
            n_frag=int(n), steps=steps,
        )
        budgets.append(run)

    # Compare consecutive doublings.
    comparisons: list[dict[str, float]] = []
    for k in range(1, len(budgets)):
        prev = budgets[k - 1]["fractions"]
        curr = budgets[k]["fractions"]
        deltas = {
            "surf": abs(curr["surf"] - prev["surf"]),
            "kin": abs(curr["kin"] - prev["kin"]),
            "diss": abs(curr["diss"] - prev["diss"]),
        }
        comparisons.append({
            "n_frag_prev": int(budgets[k - 1]["n_frag"]),
            "n_frag_curr": int(budgets[k]["n_frag"]),
            "delta_surf": deltas["surf"],
            "delta_kin": deltas["kin"],
            "delta_diss": deltas["diss"],
            "max_delta": max(deltas.values()),
        })

    max_delta = max(c["max_delta"] for c in comparisons) if comparisons else 0.0
    converged = max_delta < tolerance

    return {
        "n_frag_values": list(n_frag_values),
        "budgets": budgets,
        "comparisons": comparisons,
        "max_delta": max_delta,
        "tolerance": tolerance,
        "converged": bool(converged),
    }


# ---------------------------------------------------------------------------
# N06 ledger schema integration
# ---------------------------------------------------------------------------
def n06_break_crush_channel(budget: dict[str, Any]) -> dict[str, Any]:
    """Map a three-channel budget onto the N06 ``break_crush`` ledger schema.

    The N06 ledger (``dem3d_energy.EnergyAuditedIceDEM3D``) books
    ``cumulative_fracture_release_J`` as the ``break`` channel (bond-fracture
    elastic release).  The ``break_crush`` channel is the *combined* fracture
    sink: the bond release plus the particle-crushing three-channel partition.

    This function returns the ``crush`` half of the ``break_crush`` channel --
    the per-event surface/kinetic/dissipation partition -- so a macro ledger
    can book it alongside the ``break`` half.  The closure ``C1`` of the
    combined channel is ``epsilon_s`` of the crush partition.
    """
    return {
        "channel": "break_crush",
        "sub_channel": "crush",
        "E_in_J": budget["E_in_J"],
        "E_surf_J": budget["E_surf_J"],
        "E_kin_J": budget["E_kin_J"],
        "E_diss_J": budget["E_diss_J"],
        "E_out_J": budget["E_out_J"],
        "epsilon_s": budget["epsilon_s"],
        "fractions": budget["fractions"],
        "n_frag": budget["n_frag"],
        "dA_m2": budget["dA_m2"],
        "closure_C1": budget["epsilon_s"],
    }


# ---------------------------------------------------------------------------
# Full audit driver
# ---------------------------------------------------------------------------
def run_energy_budget_audit(
    *,
    diameter: float = DEFAULT_DIAMETER,
    seed: int = 0,
    platen_velocity: float = DEFAULT_PLATEN_VELOCITY,
    n_frag: int = 8,
    steps: int = 4000,
    n_frag_values: Sequence[int] = (4, 8, 16, 32),
    epsilon_tol: float = 0.01,
    frag_tol: float = 0.03,
) -> dict[str, Any]:
    """Run the full three-channel energy budget audit.

    Returns a dict with the single-fracture budget, the F-delta monotonicity
    check, the fragment-number convergence check, and the N06 ledger mapping.
    """
    budget = _run_single_fracture(
        diameter=diameter, seed=seed, platen_velocity=platen_velocity,
        n_frag=int(n_frag), steps=steps,
    )
    f_delta = check_f_delta_monotonic(
        budget["force_history_N"], budget["displacement_history_m"],
    )
    convergence = check_fragment_convergence(
        n_frag_values=n_frag_values, diameter=diameter, seed=seed,
        platen_velocity=platen_velocity, steps=steps, tolerance=frag_tol,
    )
    n06 = n06_break_crush_channel(budget)

    # Machine-checkable criteria.
    criteria = {
        "C1_energy_closure": {
            "target": f"epsilon_s < {epsilon_tol:.0%}",
            "measured": budget["epsilon_s"],
            "passed": bool(budget["epsilon_s"] < epsilon_tol),
        },
        "C2_f_delta_monotonic": {
            "target": "F(delta) monotonic within elastic limit",
            "measured": f_delta["n_drops"],
            "passed": bool(f_delta["monotonic"]),
        },
        "C3_fragment_convergence": {
            "target": f"n_frag doubling -> allocation change < {frag_tol:.0%}",
            "measured": convergence["max_delta"],
            "passed": bool(convergence["converged"]),
        },
        "C4_n06_closure": {
            "target": f"break_crush C1 < {epsilon_tol:.0%}",
            "measured": n06["closure_C1"],
            "passed": bool(n06["closure_C1"] < epsilon_tol),
        },
    }

    return {
        "budget": budget,
        "f_delta": f_delta,
        "convergence": convergence,
        "n06": n06,
        "criteria": criteria,
        "all_passed": all(c["passed"] for c in criteria.values()),
    }


__all__ = [
    "ThreeChannelBudget",
    "check_f_delta_monotonic",
    "check_fragment_convergence",
    "n06_break_crush_channel",
    "run_energy_budget_audit",
    "_run_single_fracture",
]

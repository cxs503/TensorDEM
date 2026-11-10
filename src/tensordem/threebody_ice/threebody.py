"""Hull-floe-floe (three-body) collision scenarios and their analytic references.

What this module answers
------------------------
The node question is *how the hull load is distributed when the ship is in
contact with several floes at the same time*, and *how much of the collision
kinetic energy the contacts dissipate*.  Two reference cases make the DEM
answer machine-checkable:

``elastic_reference``
    A free, head-on, equal-mass floe collision with zero damping (``e = 1``).
    Newton's third law and the elastic limit are exact: total momentum must be
    conserved to round-off and ``E_kin + E_spring`` must not drift (criterion
    F1/F2).  With ``e < 1`` the same case has the closed-form post-collision
    velocities ``v1' = (1-e)/2 v``, ``v2' = (1+e)/2 v``, which is how the
    *measured* restitution is defined.

``run_three_body``
    A kinematic vertical stem (block outline, infinite mass) advancing at
    ``V`` into **two** floes placed in tandem with clear spacing ``d``.  The
    near floe is hit first, transmits the load to the far floe through an
    ice-ice contact, is re-hit as the hull catches up, and so on.  Outputs are
    the per-floe hull contact force time series ``F_11(t)``, ``F_12(t)``, the
    total ``F_total(t)``, the impulse partition, the fraction of the contact
    time with *both* floes loaded simultaneously, and the energy ledger of the
    event.

Energy-dissipation definitions used here (both reported, never cherry-picked)
----------------------------------------------------------------------------
``dissipation_fraction_of_input``   ``E_diss / W_hull``: of the work the hull
    puts into the ice, the share destroyed by the contacts.
``dissipation_over_ke_gain``        ``E_diss / dE_kin``: the same energy
    measured against the kinetic energy the ice actually gained.

For a single hull-floe collision with restitution ``e`` at constant stem speed
the two are ``(1-e)/2`` and ``(1-e)/(1+e)`` exactly (derivation in
``docs/P262W1_N04_THREEBODY_ICE.md`` §2.3), which is what lets the node's
declared band ``E_diss / dE_k in [0.3, 0.8]`` be checked against a closed form
*before* the DEM is run.

Sign convention: the hull stem face normal is ``+x``; ``F_total`` and the
per-floe hull forces are the forces the hull exerts **on the floes**, so the
ice resistance is ``R = -F_total . e_x``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..broken_ice_resistance.hull import ShipForm
from .model import (ContactLaw, FloeState, Ledger, MultiBodyModel,
                    damping_ratio_from_restitution)

DTYPE = torch.float64

__all__ = [
    "ScenarioSpec",
    "build_three_body",
    "run_three_body",
    "sweep_spacing",
    "elastic_reference",
    "measure_restitution",
    "calibrate_law",
    "single_collision_fractions",
    "single_impact_energies",
    "reference_law",
    "REFERENCE_TIME_STEP_SAFETY",
]


@dataclass(frozen=True)
class ScenarioSpec:
    """Geometry and kinematics of one three-body case (all SI)."""

    hull_length_m: float = 24.0
    hull_beam_m: float = 12.0
    speed_m_s: float = 1.0
    floe_radius_m: float = 1.5
    spacing_ratio: float = 1.0        #: clear gap between the floes, in floe diameters
    lateral_fraction: float = 0.6     #: |y| of each floe, in floe radii
    initial_gap_m: float = 0.05       #: stem-to-near-floe gap at t = 0
    tail_time_s: float = 0.8          #: run on after the hull has passed the floes

    def __post_init__(self) -> None:
        for name in ("hull_length_m", "hull_beam_m", "speed_m_s", "floe_radius_m"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.spacing_ratio) < 0.0:
            raise ValueError("spacing_ratio must be non-negative")
        if not 0.0 <= float(self.lateral_fraction) < 1.0:
            raise ValueError("lateral_fraction must lie in [0, 1)")

    @property
    def spacing_m(self) -> float:
        """Clear gap between the two floe surfaces [m]."""
        return float(self.spacing_ratio) * 2.0 * float(self.floe_radius_m)

    @property
    def lateral_offset_m(self) -> float:
        return float(self.lateral_fraction) * float(self.floe_radius_m)

    def floe_positions(self) -> List[Tuple[float, float]]:
        """Initial positions: near floe on +y, far floe on -y, tandem in x."""
        a = float(self.floe_radius_m)
        x_stem = 0.0
        x_near = x_stem + float(self.initial_gap_m) + a
        x_far = x_near + 2.0 * a + self.spacing_m
        y = self.lateral_offset_m
        return [(x_near, +y), (x_far, -y)]

    def hull(self) -> ShipForm:
        return ShipForm(length=self.hull_length_m, beam=self.hull_beam_m,
                        speed=self.speed_m_s, entry_angle_deg=45.0,
                        draught=1.0, form="block", x_start=0.0)

    def as_dict(self) -> Dict[str, float]:
        return {"hull_length_m": self.hull_length_m, "hull_beam_m": self.hull_beam_m,
                "speed_m_s": self.speed_m_s, "floe_radius_m": self.floe_radius_m,
                "spacing_ratio": self.spacing_ratio,
                "spacing_m": self.spacing_m,
                "lateral_fraction": self.lateral_fraction,
                "initial_gap_m": self.initial_gap_m,
                "tail_time_s": self.tail_time_s}


def reference_law(law: ContactLaw) -> ContactLaw:
    """Drag-free copy of ``law`` for the isolated collision references.

    The momentum and elastic-energy checks must not be polluted by the water
    drag, which is an external force on the ice system by construction.
    """
    return replace(law, water_drag_coefficient=0.0)


def single_collision_fractions(restitution: float) -> Dict[str, float]:
    """Closed-form dissipation fractions of one constant-speed-stem collision.

    A floe at rest is hit by a stem moving at ``V``; the floe leaves at
    ``(1+e) V``.  The work the hull does is ``W = V * m (1+e) V``, the kinetic
    energy gained is ``m (1+e)^2 V^2 / 2`` and the dissipated part is the
    difference, so the two fractions are ``(1-e)/2`` and ``(1-e)/(1+e)``.
    """
    e = float(restitution)
    if not 0.0 < e <= 1.0:
        raise ValueError("restitution must lie in (0, 1]")
    return {"dissipation_fraction_of_input": (1.0 - e) / 2.0,
            "dissipation_over_ke_gain": (1.0 - e) / (1.0 + e)}


# ---------------------------------------------------------------------------
# Timestep
# ---------------------------------------------------------------------------

#: Time-step safety factor for the collision references and the three-body
#: event: ``dt = safety * 2 pi sqrt(m_eff / k)``.  The impact cases are short and
#: violent (``omega dt`` near the stability limit at safety = 0.08), so they use a
#: finer step than the field transit; the convergence of the energy ledger with
#: this parameter is reported in the doc.
REFERENCE_TIME_STEP_SAFETY = 0.03


def _time_step(law: ContactLaw, radius: float,
               safety: float = REFERENCE_TIME_STEP_SAFETY) -> float:
    mass = law.ice_density * math.pi * radius ** 2 * law.thickness
    return law.stable_time_step(0.5 * mass, safety=safety)


# ---------------------------------------------------------------------------
# Reference cases
# ---------------------------------------------------------------------------

def elastic_reference(law: ContactLaw, *, speed_m_s: float = 1.0,
                      radius_m: float = 2.0, seed_offset: float = 0.6
                      ) -> Dict[str, object]:
    """Free head-on equal-mass floe collision: momentum and energy checks.

    Returns the measured restitution, the relative momentum error and the
    relative drift of ``E_kin + E_spring`` (criteria F1/F2), plus the analytic
    post-collision velocities.
    """
    law = reference_law(law)
    dt = _time_step(law, radius_m)
    gap = seed_offset
    half = radius_m + 0.5 * gap
    state = FloeState.create([radius_m, radius_m],
                             [[-half, 0.0], [half, 0.0]], law=law)
    state.vel[0] = torch.tensor([speed_m_s, 0.0], dtype=DTYPE)
    model = MultiBodyModel(state, law)
    energy_0 = model.mechanical_energy()
    momentum_0 = state.momentum().clone()
    v_rel_0 = float((state.vel[1] - state.vel[0]).norm())

    # run until the pair has separated and is at least one radius apart again
    n_steps = 0
    max_spring = 0.0
    while n_steps < 200000:
        model.step(dt)
        n_steps += 1
        max_spring = max(max_spring, model.spring_energy())
        separated = float((state.pos[1] - state.pos[0]).norm()) > 2.0 * radius_m + gap
        if separated and model.spring_energy() < 1.0e-6 * energy_0:
            break
    v_rel_1 = float((state.vel[1] - state.vel[0]).norm())
    restitution = v_rel_1 / v_rel_0
    momentum_1 = state.momentum()
    energy_1 = model.mechanical_energy()
    e_analytic = law.floe_floe_restitution
    reduced_mass = 0.5 * float(state.mass[0])      # equal pair
    analytic_impulse = (1.0 + e_analytic) * reduced_mass * v_rel_0
    measured_impulse = float(state.mass[1] * state.vel[1, 0])
    return {
        "reduced_mass_kg": reduced_mass,
        "analytic_impulse_N_s": analytic_impulse,
        "measured_impulse_N_s": measured_impulse,
        "impulse_relative_error": abs(measured_impulse - analytic_impulse)
        / max(abs(analytic_impulse), 1e-30),
        "case": "elastic_reference" if e_analytic >= 1.0 else "restitution_reference",
        "time_step_s": dt, "n_steps": n_steps,
        "speed_m_s": speed_m_s, "radius_m": radius_m,
        "target_restitution": e_analytic,
        "measured_restitution": restitution,
        "restitution_relative_error": abs(restitution - e_analytic) / e_analytic,
        "relative_momentum_error": float((momentum_1 - momentum_0).abs().max()) / max(
            abs(momentum_0.abs().max().item()), 1.0e-30),
        "energy_0_J": energy_0, "energy_end_J": energy_1,
        "relative_energy_drift": abs(energy_1 - energy_0) / energy_0,
        "peak_spring_energy_J": max_spring,
        "measured_velocities_m_s": state.vel.tolist(),
        "analytic_velocities_m_s": [
            [float((1.0 - e_analytic) / 2.0 * speed_m_s), 0.0],
            [float((1.0 + e_analytic) / 2.0 * speed_m_s), 0.0],
        ],
        "dissipation_J": model.ledger.e_diss_normal + model.ledger.e_diss_friction,
        "ledger": model.ledger.as_dict(),
    }


def single_impact_energies(law: ContactLaw, *, speed_m_s: float = 1.0,
                           radius_m: float = 1.5) -> Dict[str, object]:
    """Energy split of **one isolated hull-floe impact** (the collision grain).

    A floe at rest is hit once by the constant-speed stem and separates.  The
    work the hull does is ``W = int F_hull . V dt``, the ice gains ``dE_k`` and
    the contact dissipates the difference, so both of the node's ratios

    ``E_diss / W_hull``   fraction of the input work destroyed in the contact
    ``E_diss / dE_k``     dissipation per unit kinetic energy gained

    are measured directly on a single collision -- which is the granularity the
    band ``[0.3, 0.8]`` is about (the analytic values are ``(1-e)/2`` and
    ``(1-e)/(1+e)``).
    """
    law_ref = reference_law(law)
    spec = ScenarioSpec(speed_m_s=speed_m_s, floe_radius_m=radius_m)
    state = FloeState.create([radius_m], [[spec.initial_gap_m + radius_m, 0.0]],
                             law=law_ref)
    model = MultiBodyModel(state, law_ref, hull=spec.hull(),
                           hull_speed=spec.speed_m_s)
    dt = _time_step(law_ref, radius_m)
    ke_0 = state.kinetic_energy()
    contacted = False
    steps = 0
    while steps < 500000:
        model.step(dt)
        steps += 1
        if model.last_hull_contacts > 0:
            contacted = True
        elif contacted:
            break
    if not contacted:
        raise RuntimeError("the single impact never made contact")
    for _ in range(max(1, int(0.02 / dt))):
        model.step(dt)
    ke_1 = state.kinetic_energy()
    w_hull = model.ledger.w_hull
    e_diss = model.ledger.e_diss_normal + model.ledger.e_diss_friction
    d_ke = ke_1 - ke_0
    closure = w_hull - (d_ke + e_diss)
    return {
        "case": "single_impact",
        "restitution_target": law_ref.hull_floe_restitution,
        "speed_m_s": speed_m_s, "radius_m": radius_m, "time_step_s": dt,
        "n_steps": steps,
        "w_hull_J": w_hull, "e_dissipated_J": e_diss, "ke_gain_J": d_ke,
        "closure_J": closure,
        "closure_relative": abs(closure) / max(abs(w_hull), 1e-30),
        "dissipation_fraction_of_input": e_diss / max(abs(w_hull), 1e-30),
        "dissipation_over_ke_gain": (e_diss / d_ke if abs(d_ke) > 0
                                     else float("nan")),
        "analytic_single_collision": single_collision_fractions(
            law_ref.hull_floe_restitution),
        "measured_restitution": measure_restitution(law_ref, kind="hull_floe",
                                                    speed_m_s=speed_m_s,
                                                    radius_m=radius_m),
    }


def measure_restitution(law: ContactLaw, *, kind: str = "floe_floe",
                        speed_m_s: float = 1.0, radius_m: float = 2.0
                        ) -> float:
    """Measured coefficient of restitution of one isolated collision."""
    law = reference_law(law)
    dt = _time_step(law, radius_m)
    if kind == "floe_floe":
        return float(elastic_reference(law, speed_m_s=speed_m_s,
                                       radius_m=radius_m)["measured_restitution"])
    if kind != "hull_floe":
        raise ValueError("kind must be 'floe_floe' or 'hull_floe'")
    spec = ScenarioSpec(speed_m_s=speed_m_s, floe_radius_m=radius_m)
    state = FloeState.create([radius_m], [[spec.initial_gap_m + radius_m, 0.0]],
                             law=law)
    model = MultiBodyModel(state, law, hull=spec.hull(),
                           hull_speed=spec.speed_m_s)
    v_0 = float(state.vel[0, 0]) - spec.speed_m_s          # approach speed, < 0
    contacted = False
    n_steps = 0
    while n_steps < 500000:
        model.step(dt)
        n_steps += 1
        if model.last_hull_contacts > 0:
            contacted = True
        elif contacted:
            break
    if not contacted:
        raise RuntimeError("hull-floe reference case never made contact")
    for _ in range(max(1, int(0.02 / dt))):        # let it separate cleanly
        model.step(dt)
    v_1 = float(state.vel[0, 0]) - spec.speed_m_s
    return abs(v_1 / v_0)


def calibrate_law(restitution: float, *, law: Optional[ContactLaw] = None,
                  tolerance: float = 1.0e-3, max_iterations: int = 30
                  ) -> Dict[str, object]:
    """Numerically calibrate the damping ratios to hit ``restitution``.

    The analytic ``beta = f(e)`` of a linear spring-dashpot is only approximate
    (the dashpot force is clipped and the contact law is not exactly Maxwell),
    so the damping ratio is solved by bisection on the *measured* restitution of
    an isolated free-pair and an isolated hull-floe collision.  The result is a
    :class:`ContactLaw` carrying explicit ``beta`` overrides, plus the measured
    restitutions so the calibration is auditable.
    """
    e_target = float(restitution)
    if not 0.0 < e_target <= 1.0:
        raise ValueError("restitution must lie in (0, 1]")
    base = law if law is not None else ContactLaw(floe_floe_restitution=e_target,
                                                  hull_floe_restitution=e_target)

    def solve(kind: str) -> Tuple[float, float]:
        lo, hi = 0.0, 0.85
        best_beta, best_e = 0.0, 1.0
        for _ in range(max_iterations):
            beta = 0.5 * (lo + hi)
            if kind == "floe_floe":
                trial = replace(base, beta_floe_floe_override=beta)
            else:
                trial = replace(base, beta_hull_floe_override=beta)
            e_meas = measure_restitution(trial, kind=kind)
            if abs(e_meas - e_target) < abs(best_e - e_target):
                best_beta, best_e = beta, e_meas
            if abs(e_meas - e_target) <= tolerance:
                return beta, e_meas
            # restitution decreases monotonically with the damping ratio
            if e_meas > e_target:
                lo = beta
            else:
                hi = beta
        return best_beta, best_e

    beta_ff, e_ff = solve("floe_floe")
    beta_hf, e_hf = solve("hull_floe")
    calibrated = replace(base, floe_floe_restitution=e_target,
                         hull_floe_restitution=e_target,
                         beta_floe_floe_override=beta_ff,
                         beta_hull_floe_override=beta_hf)
    return {
        "law": calibrated,
        "target_restitution": e_target,
        "beta_floe_floe_analytic": damping_ratio_from_restitution(e_target),
        "beta_floe_floe_calibrated": beta_ff,
        "beta_hull_floe_calibrated": beta_hf,
        "measured_restitution_floe_floe": e_ff,
        "measured_restitution_hull_floe": e_hf,
        "restitution_error_floe_floe": abs(e_ff - e_target),
        "restitution_error_hull_floe": abs(e_hf - e_target),
        "converged": bool(abs(e_ff - e_target) <= tolerance
                          and abs(e_hf - e_target) <= tolerance),
    }


# ---------------------------------------------------------------------------
# Three-body scenario
# ---------------------------------------------------------------------------

def build_three_body(spec: ScenarioSpec, law: ContactLaw
                     ) -> Tuple[MultiBodyModel, float]:
    """Build the tandem two-floe model in front of a block stem."""
    positions = spec.floe_positions()
    state = FloeState.create([spec.floe_radius_m] * 2, positions, law=law)
    model = MultiBodyModel(state, law, hull=spec.hull(),
                           hull_speed=spec.speed_m_s)
    return model, _time_step(law, spec.floe_radius_m)


def run_three_body(spec: ScenarioSpec, law: ContactLaw, *,
                   record: bool = True, drag: bool = False) -> Dict[str, object]:
    """Run one tandem two-floe impact and return the force/energy bookkeeping.

    ``drag=False`` isolates the impact event (no water drag): the hull impulse is
    the only external impulse, so total momentum and the energy ledger close
    exactly and the contact-dissipation fraction is unambiguous.  ``drag=True``
    is the physical sustained case -- the hull carries the near floe against the
    water drag and keeps re-contacting it, so both floes end up loaded at the
    same time and the *force allocation* between them becomes measurable.
    """
    if not drag:
        law = reference_law(law)
    model, dt = build_three_body(spec, law)
    state = model.state
    a = float(spec.floe_radius_m)
    # stop once the stem is a floe diameter past the far floe's initial station
    x_far = spec.floe_positions()[1][0]
    t_stop = (x_far + 2.0 * a + spec.tail_time_s * spec.speed_m_s) / spec.speed_m_s
    n_steps = int(math.ceil(t_stop / dt))

    energy_0 = model.mechanical_energy()
    ke_0 = state.kinetic_energy()
    momentum_0 = state.momentum().clone()
    series: Dict[str, List[float]] = {k: [] for k in
                                      ("t", "f_total_x", "f_1_x", "f_2_x",
                                       "v_1_x", "v_2_x", "ke")}
    both_loaded_steps = 0
    any_loaded_steps = 0
    impulse_1 = 0.0
    impulse_2 = 0.0
    peak_1 = 0.0
    peak_2 = 0.0
    peak_total = 0.0
    max_share_error = 0.0
    max_dynamics_error = 0.0
    contact_started = False

    for _ in range(n_steps):
        per_floe = model.step(dt)["hull_force_per_floe"]
        f1 = float(per_floe[0, 0])
        f2 = float(per_floe[1, 0])
        f_total = float(model.last_hull_force[0])
        loaded_1 = abs(f1) > 0.0
        loaded_2 = abs(f2) > 0.0
        if loaded_1 or loaded_2:
            contact_started = True
            any_loaded_steps += 1
        if loaded_1 and loaded_2:
            both_loaded_steps += 1
        impulse_1 += f1 * dt
        impulse_2 += f2 * dt
        peak_1 = max(peak_1, abs(f1))
        peak_2 = max(peak_2, abs(f2))
        peak_total = max(peak_total, abs(f_total))
        # F3: the two independent routes to F_total must agree
        denom = max(abs(f1) + abs(f2), 1.0e-30)
        max_share_error = max(max_share_error, abs((f1 + f2) - f_total) / denom)
        if record:
            series["t"].append(model.time)
            series["f_total_x"].append(f_total)
            series["f_1_x"].append(f1)
            series["f_2_x"].append(f2)
            series["v_1_x"].append(float(state.vel[0, 0]))
            series["v_2_x"].append(float(state.vel[1, 0]))
            series["ke"].append(state.kinetic_energy())

    # momentum-rate cross-check over the whole event (no water drag in this case)
    momentum_end = state.momentum()
    delta_p = float((momentum_end - momentum_0)[0])
    total_impulse = float(model.hull_impulse[0])
    external_impulse = float(model.external_impulse()[0])
    impulse_error = abs(external_impulse - delta_p) / max(abs(external_impulse), 1.0e-30)

    energy_end = model.mechanical_energy()
    w_hull = model.ledger.w_hull
    e_diss = model.ledger.e_diss_normal + model.ledger.e_diss_friction
    d_ke = state.kinetic_energy() - ke_0
    closure = abs(w_hull + model.ledger.w_drag - (energy_end - energy_0) - e_diss)
    fractions = single_collision_fractions(law.hull_floe_restitution)
    return {
        "case": "three_body",
        "spec": spec.as_dict(),
        "law": law.as_dict(),
        "time_step_s": dt, "n_steps": n_steps, "duration_s": model.time,
        "floe_positions_initial": spec.floe_positions(),
        "impulse_1": impulse_1, "impulse_2": impulse_2,
        "impulse_total": impulse_1 + impulse_2,
        "impulse_share_1": impulse_1 / max(abs(impulse_1) + abs(impulse_2), 1e-30),
        "impulse_share_2": impulse_2 / max(abs(impulse_1) + abs(impulse_2), 1e-30),
        "peak_force_1_N": peak_1, "peak_force_2_N": peak_2,
        "peak_force_total_N": peak_total,
        "simultaneous_contact_fraction": (both_loaded_steps / any_loaded_steps
                                         if any_loaded_steps else 0.0),
        "contact_steps": any_loaded_steps,
        "both_loaded_steps": both_loaded_steps,
        "force_share_algebraic_error": max_share_error,
        "impulse_vs_momentum_error": impulse_error,
        "drag_impulse_N_s": float(model.drag_impulse[0]),
        "drag_enabled": bool(drag),
        "total_hull_impulse_N_s": total_impulse,
        "momentum_change_N_s": delta_p,
        "w_hull_J": w_hull,
        "e_dissipated_J": e_diss,
        "ke_gain_J": d_ke,
        "energy_closure_error_J": closure,
        "energy_closure_relative": closure / max(abs(w_hull), 1.0e-30),
        "dissipation_fraction_of_input": e_diss / max(abs(w_hull), 1.0e-30),
        "dissipation_over_ke_gain": (e_diss / max(abs(d_ke), 1.0e-30)
                                     if abs(d_ke) > 1.0e-9 * max(abs(e_diss), 1.0) else float("nan")),
        "analytic_single_collision_fractions": fractions,
        "max_penetration_fraction": model.ledger.max_penetration_fraction,
        "series": series if record else {},
    }


def sweep_spacing(law: ContactLaw, ratios: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0),
                  *, speed_m_s: float = 1.0, floe_radius_m: float = 1.5
                  ) -> List[Dict[str, object]]:
    """Run the tandem impact over a sweep of the clear spacing ``d / D``."""
    rows: List[Dict[str, object]] = []
    for ratio in ratios:
        spec = ScenarioSpec(speed_m_s=speed_m_s, floe_radius_m=floe_radius_m,
                            spacing_ratio=float(ratio))
        rows.append(run_three_body(spec, law))
    return rows
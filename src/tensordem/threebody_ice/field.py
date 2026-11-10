"""Broken-ice field transit and the resistance decomposition.

Scenario
--------
A ship-formed outline (:class:`tensordem.broken_ice_resistance.hull.ShipForm`,
wedge bow of declared entry angle) advances at constant speed ``V`` through a
plan-view field of circular floes of areal concentration ``c_ice``.  Floes are
buoyant disks with water drag, so they are deflected by the bow, pushed along the
corridor, jostled against each other and eventually leave the domain.

Resistance decomposition (measured, not fitted)
----------------------------------------------
Every term below is read from the model's own energy ledger over the *steady
window* (stem between two declared fractions of the domain), divided by the
distance travelled ``V T``:

``R_collision``         contact dissipation (dashpot + Coulomb friction, hull-ice
                        **and** ice-ice) per unit distance
``R_submerged_drag``    water-drag work on the floes per unit distance
``R_channel_clearing``  kinetic energy carried out of the swept corridor by the
                        floes that leave it (lateral and downstream separately)
``R_residual``          change of stored (kinetic + spring) energy over the
                        window, divided by the distance -- the closure residual

and the identity ``R_ice = R_collision + R_submerged_drag + R_channel_clearing
+ R_residual`` holds because it *is* the energy balance of the integrator:

    W_hull + W_drag = dE_kin + dE_spring + E_diss_normal + E_diss_friction + E_out

``R_ice`` itself is measured twice, independently: from the time-average of the
hull reaction ``-<F_hull . e_x>`` and from ``W_hull / (V T)``.  The two must
agree (they do to integrator order); the small difference is reported.

Velocity dependence is measured, not assumed: :func:`velocity_sweep` fits the
log-log slope of each component over the declared speed range and compares it
with the node's predeclared expectation (collision ``V^2``, drag ``V^2``,
clearing ``V^1``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..broken_ice_resistance.hull import ShipForm
from .model import ContactLaw, FloeState, Ledger, MultiBodyModel

DTYPE = torch.float64

__all__ = ["FieldSpec", "build_field", "run_field", "velocity_sweep",
           "concentration_sweep", "power_law_exponent", "seed_region_m"]


@dataclass(frozen=True)
class FieldSpec:
    """Geometry, ice state and kinematics of one broken-ice transit case."""

    domain_length_m: float = 120.0
    domain_width_m: float = 48.0
    hull_length_m: float = 40.0
    hull_beam_m: float = 14.0
    hull_draught_m: float = 6.0
    entry_angle_deg: float = 20.0
    concentration: float = 0.5
    speed_m_s: float = 1.0
    floe_radius_min_m: float = 1.2
    floe_radius_max_m: float = 2.4
    seed: int = 20261110
    time_step_safety: float = 0.05
    window_fraction: Tuple[float, float] = (0.35, 0.85)
    seed_margin_m: float = 4.0

    def __post_init__(self) -> None:
        for name in ("domain_length_m", "domain_width_m", "hull_length_m",
                     "hull_beam_m", "speed_m_s", "floe_radius_min_m",
                     "floe_radius_max_m"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.concentration) <= 0.9:
            raise ValueError("concentration must lie in (0, 0.9]")
        if not float(self.floe_radius_max_m) >= float(self.floe_radius_min_m):
            raise ValueError("floe_radius_max_m must be >= floe_radius_min_m")
        lo, hi = self.window_fraction
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("window_fraction must be an increasing pair in [0, 1]")

    # -- derived ------------------------------------------------------------
    @property
    def half_width_m(self) -> float:
        return 0.5 * self.domain_width_m

    @property
    def corridor_half_width(self) -> float:
        """Half-width of the swept corridor [m]: hull half-beam + one mean radius.

        A floe further out than this cannot be reached by the hull again, so it
        has been cleared out of the channel; its kinetic energy is banked as the
        channel-clearing flux (see :meth:`MultiBodyModel._retire_escaped`).
        """
        return 0.5 * float(self.hull_beam_m) + 1.5 * float(self.floe_radius_max_m)

    @property
    def ice_area_target_m2(self) -> float:
        return self.concentration * self.domain_length_m * self.domain_width_m

    def hull(self) -> ShipForm:
        return ShipForm(length=self.hull_length_m, beam=self.hull_beam_m,
                        speed=self.speed_m_s,
                        entry_angle_deg=self.entry_angle_deg,
                        draught=self.hull_draught_m, form="ship", x_start=0.0)

    @property
    def floe_radius_mean_m(self) -> float:
        return 0.5 * (float(self.floe_radius_min_m) + float(self.floe_radius_max_m))

    def as_dict(self) -> Dict[str, object]:
        return {"domain_length_m": self.domain_length_m,
                "domain_width_m": self.domain_width_m,
                "hull_length_m": self.hull_length_m,
                "hull_beam_m": self.hull_beam_m,
                "hull_draught_m": self.hull_draught_m,
                "entry_angle_deg": self.entry_angle_deg,
                "concentration": self.concentration,
                "speed_m_s": self.speed_m_s,
                "floe_radius_min_m": self.floe_radius_min_m,
                "floe_radius_max_m": self.floe_radius_max_m,
                "seed": self.seed,
                "time_step_safety": self.time_step_safety,
                "window_fraction": list(self.window_fraction)}


def seed_region_m(spec: "FieldSpec") -> Dict[str, float]:
    """The ice-covered part of the domain: where floes are seeded.

    The areal concentration is defined over *this* area (the ice field), not over
    the full tank: the tank is longer/wider than the seeded field so that the ice
    has room to be deflected into open water.
    """
    x_lo = max(float(spec.seed_margin_m), 0.5 * float(spec.hull_beam_m))
    x_hi = float(spec.domain_length_m) - float(spec.seed_margin_m)
    y_hi = spec.half_width_m - float(spec.seed_margin_m)
    return {"x_lo": x_lo, "x_hi": x_hi, "y_hi": y_hi,
            "area_m2": (x_hi - x_lo) * 2.0 * y_hi}


def _floe_layout(spec: FieldSpec, *, iterations: int = 5
                 ) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Jittered-triangular floe pack with radii tied to the nearest-neighbour gap.

    Every radius is ``r_i = alpha d_i`` with ``d_i`` the distance from floe ``i``
    to its nearest neighbour, so ``r_i + r_j <= 2 alpha d_ij`` for *any* pair and
    the pack is overlap-free for ``alpha <= 0.47``.  The areal concentration is
    then ``alpha^2 sum(pi d_i^2) / A``, which fixes ``alpha`` for the requested
    concentration; the lattice spacing fixes the mean radius.  The two knobs are
    decoupled in that way, so the layout is built by (i) iterating the spacing to
    the requested mean radius and (ii) lowering that radius target if the
    concentration cannot be reached below the overlap cap -- a real limit: a
    non-overlapping pack of disks jams near ``c = 0.47^2 pi = 0.69`` (random
    sequential adsorption jams already near 0.55).  The realised concentration
    and radius range are written into the artifact, never assumed.
    """
    region = seed_region_m(spec)
    area = region["area_m2"]
    x_lo, x_hi, y_hi = region["x_lo"], region["x_hi"], region["y_hi"]
    alpha_cap = 0.47
    c_target = float(spec.concentration)
    r_target = spec.floe_radius_mean_m

    radii: List[float] = []
    points: List[Tuple[float, float]] = []
    for _outer in range(6):
        spacing = r_target / 0.40
        alpha = alpha_cap
        for _inner in range(iterations):
            generator = torch.Generator().manual_seed(int(spec.seed))
            jitter = 0.08 * spacing
            points = []
            rows = int(math.ceil((2.0 * y_hi) / (spacing * math.sqrt(3.0) / 2.0))) + 2
            cols = int(math.ceil((x_hi - x_lo) / spacing)) + 2
            for row in range(rows):
                for col in range(cols):
                    x = x_lo + (col + 0.5 * (row % 2)) * spacing
                    y = -y_hi + row * spacing * math.sqrt(3.0) / 2.0
                    x += (float(torch.rand(1, generator=generator)) * 2.0 - 1.0) * jitter
                    y += (float(torch.rand(1, generator=generator)) * 2.0 - 1.0) * jitter
                    if x_lo <= x <= x_hi and -y_hi <= y <= y_hi:
                        points.append((x, y))
            if len(points) < 4:
                raise RuntimeError("floe lattice produced too few sites; enlarge the domain")
            coords = torch.tensor(points, dtype=DTYPE)
            dmat = torch.cdist(coords, coords) + torch.eye(coords.shape[0], dtype=DTYPE) * 1.0e9
            d_min = dmat.min(dim=1).values
            alpha = min(alpha_cap, math.sqrt(
                c_target * area / (math.pi * float((d_min ** 2).sum()))))
            radii = (alpha * d_min).tolist()
            r_mean = alpha * float(d_min.mean())
            spacing *= r_target / max(r_mean, 1.0e-9)
        c_real = math.pi * sum(r * r for r in radii) / area
        if c_real >= 0.98 * c_target or alpha < alpha_cap - 1.0e-9:
            break
        r_target *= math.sqrt(c_real / c_target)      # more floes, smaller radii
    return radii, points


def build_field(spec: FieldSpec, law: ContactLaw) -> Tuple[MultiBodyModel, float]:
    """Build the floe field + kinematic hull model."""
    radii, positions = _floe_layout(spec)
    state = FloeState.create(radii, positions, law=law)
    model = MultiBodyModel(state, law, hull=spec.hull(),
                           hull_speed=spec.speed_m_s,
                           domain_half_width=spec.half_width_m,
                           domain_length=spec.domain_length_m,
                           corridor_half_width=spec.corridor_half_width)
    mass = float(state.mass.min())
    dt = law.stable_time_step(0.5 * mass, safety=spec.time_step_safety)
    return model, dt


@dataclass
class WindowStatistics:
    """Accumulated quantities over the steady window."""

    duration: float = 0.0
    force_x_integral: float = 0.0        # integral of F_hull . e_x  (on the ice)
    samples: int = 0
    max_hull_contacts: int = 0
    floe_speed_sum: float = 0.0
    n_floes: int = 1

    def as_dict(self) -> Dict[str, float]:
        return {"duration_s": self.duration,
                "force_x_integral_N_s": self.force_x_integral,
                "samples": self.samples,
                "max_hull_contacts": self.max_hull_contacts,
                "mean_floe_speed_m_s": (self.floe_speed_sum / self.samples
                                        / max(self.n_floes, 1)
                                        if self.samples else float("nan"))}


def run_field(spec: FieldSpec, law: ContactLaw, *, record_every: int = 20,
              max_steps: int = 400000) -> Dict[str, object]:
    """Transit one broken-ice field and decompose the ice resistance."""
    model, dt = build_field(spec, law)
    state = model.state
    hull = spec.hull()
    x_start = 0.0
    x_stop = float(spec.domain_length_m) + 0.25 * float(spec.hull_length_m)
    n_steps = int(math.ceil((x_stop - x_start) / (spec.speed_m_s * dt)))
    n_steps = min(n_steps, max_steps)

    lo_frac, hi_frac = spec.window_fraction
    x_window_lo = lo_frac * float(spec.domain_length_m)
    x_window_hi = hi_frac * float(spec.domain_length_m)

    stats = WindowStatistics(n_floes=state.n)
    ledger_at_start: Optional[Ledger] = None
    energy_at_start = 0.0
    ledger_at_end: Optional[Ledger] = None
    energy_at_end = 0.0
    series: Dict[str, List[float]] = {k: [] for k in
                                      ("t", "x_stem", "f_hull_x", "r_running",
                                       "e_hull_cum", "e_diss_cum", "e_drag_cum",
                                       "e_out_cum")}
    in_window = False

    for step_index in range(n_steps):
        out = model.step(dt)
        x_stem = float(hull.x_centre(model.time))
        inside = x_window_lo <= x_stem <= x_window_hi
        if inside and not in_window:
            in_window = True
            ledger_at_start = model.ledger.copy()
            energy_at_start = model.mechanical_energy()
        if in_window:
            stats.duration += dt
            stats.force_x_integral += float(out["hull_force"][0]) * dt
            stats.samples += 1
            stats.max_hull_contacts = max(stats.max_hull_contacts,
                                          int(out["n_hull_contacts"]))
            stats.floe_speed_sum += float(
                (state.vel * state.active.to(DTYPE).unsqueeze(1)).norm(dim=1).sum())
        if in_window and step_index % max(1, record_every) == 0 and ledger_at_start is not None:
            led = model.ledger
            series["t"].append(float(model.time))
            series["x_stem"].append(x_stem)
            series["f_hull_x"].append(float(out["hull_force"][0]))
            series["e_hull_cum"].append(led.w_hull - ledger_at_start.w_hull)
            series["e_diss_cum"].append(
                led.e_diss_normal + led.e_diss_friction
                - ledger_at_start.e_diss_normal - ledger_at_start.e_diss_friction)
            series["e_drag_cum"].append(led.w_drag - ledger_at_start.w_drag)
            series["e_out_cum"].append(led.e_out - ledger_at_start.e_out)
            travelled = max(model.time * spec.speed_m_s - x_window_lo, 1.0e-9)
            series["r_running"].append(
                -(led.w_hull - ledger_at_start.w_hull) / travelled)
        if inside:
            ledger_at_end = model.ledger.copy()
            energy_at_end = model.mechanical_energy()
        if x_stem >= x_window_hi and in_window:
            break

    if ledger_at_start is None or ledger_at_end is None or stats.duration <= 0.0:
        raise RuntimeError("the transit never entered the steady window")
    window = ledger_at_end.difference(ledger_at_start)
    distance = spec.speed_m_s * stats.duration
    # The hull pushes the ice along +x, so the ice resists with -x: the ice
    # resistance (positive) is the *time mean* of the hull force on the ice along
    # e_x.  Dividing by the distance instead of the duration would scale it by V.
    r_force = stats.force_x_integral / stats.duration
    r_power = window.w_hull / distance
    r_collision = (window.e_diss_normal + window.e_diss_friction) / distance
    r_drag = abs(window.w_drag) / distance
    r_clearing = window.e_out / distance      # all ice left the swept corridor

    r_residual = (energy_at_end - energy_at_start) / distance
    r_sum = r_collision + r_drag + r_clearing
    return {
        "case": "field_transit",
        "spec": spec.as_dict(),
        "law": law.as_dict(),
        "time_step_s": dt, "n_steps": n_steps, "duration_s": model.time,
        "n_floes": state.n,
        "seed_region_m": seed_region_m(spec),
        "realised_concentration": float(
            (math.pi * state.radius ** 2).sum()) / seed_region_m(spec)["area_m2"],
        "realised_radius_m": {"min": float(state.radius.min()),
                              "mean": float(state.radius.mean()),
                              "max": float(state.radius.max())},
        "window": stats.as_dict(),
        "distance_m": distance,
        "R_ice_force_N": r_force,
        "R_ice_power_N": r_power,
        "R_force_vs_power_relative": abs(r_power - r_force) / max(abs(r_force), 1e-30),
        "R_collision_N": r_collision,
        "R_submerged_drag_N": r_drag,
        "R_channel_clearing_N": r_clearing,
        "R_channel_clearing_wake_N": window.e_out_wake / distance,
        "R_channel_clearing_lateral_N": window.e_out_clearing / distance,
        "R_residual_N": r_residual,
        "R_component_sum_N": r_sum,
        "decomposition_closure_relative": abs(r_sum - r_force) / max(abs(r_force), 1e-30),
        "decomposition_closure_with_residual_relative":
            abs(r_sum + r_residual - r_force) / max(abs(r_force), 1e-30),
        "residual_relative": abs(r_residual) / max(abs(r_force), 1e-30),
        "clearing_retired_N": (window.e_out_clearing + window.e_out_downstream) / distance,
        "n_removed": window.n_removed,
        "max_penetration_fraction": model.ledger.max_penetration_fraction,
        "ledger_window": window.as_dict(),
        "series": series,
    }


def power_law_exponent(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """Least-squares log-log slope with its standard error and ``R^2``."""
    px = [math.log(float(x)) for x in xs]
    py = [math.log(float(y)) for y in ys]
    n = len(px)
    if n < 3:
        raise ValueError("need at least three points for a power-law fit")
    mx = sum(px) / n
    my = sum(py) / n
    sxx = sum((p - mx) ** 2 for p in px)
    sxy = sum((p - mx) * (q - my) for p, q in zip(px, py))
    slope = sxy / sxx
    intercept = my - slope * mx
    residual = [q - (intercept + slope * p) for p, q in zip(px, py)]
    ss_res = sum(r * r for r in residual)
    ss_tot = sum((q - my) ** 2 for q in py)
    dof = n - 2
    sigma2 = ss_res / dof if dof > 0 else float("nan")
    stderr = math.sqrt(sigma2 / sxx) if dof > 0 else float("nan")
    return {"exponent": slope, "stderr": stderr,
            "intercept_log": intercept, "K": math.exp(intercept),
            "r2_log": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n": n}


def velocity_sweep(law: ContactLaw, *, speeds: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 4.0),
                   concentration: float = 0.5, base: Optional[FieldSpec] = None,
                   **kwargs) -> List[Dict[str, object]]:
    """Run the transit over a speed sweep at fixed concentration."""
    rows: List[Dict[str, object]] = []
    for speed in speeds:
        spec = base if base is not None else FieldSpec()
        from dataclasses import replace
        rows.append(run_field(replace(spec, speed_m_s=float(speed),
                                      concentration=float(concentration)), law, **kwargs))
    return rows


def concentration_sweep(law: ContactLaw,
                        concentrations: Sequence[float] = (0.3, 0.5, 0.7),
                        *, speed_m_s: float = 1.0,
                        base: Optional[FieldSpec] = None,
                        **kwargs) -> List[Dict[str, object]]:
    """Run the transit over the declared concentration range."""
    rows: List[Dict[str, object]] = []
    for concentration in concentrations:
        spec = base if base is not None else FieldSpec()
        from dataclasses import replace
        rows.append(run_field(replace(spec, speed_m_s=float(speed_m_s),
                                      concentration=float(concentration)), law, **kwargs))
    return rows
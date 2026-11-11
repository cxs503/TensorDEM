"""Multi-particle erosion stream: trajectory-driven DEM impacts -> E [kg/kg].

The stream is the aggregation half of the verification face, and it is
deliberately *not* a sampler of imposed impact angles:

* every particle is launched from a nozzle, flies ballistically to the wall, and
  its impact kinematics (speed, angle) are **measured from the trajectory**;
* contact restitution (plastic normal, Coulomb tangential) decides how much
  energy stays in the particle, so a grazing particle can skip along the wall
  and strike it several times;
* each impact converts an erosion ratio into a crater volume that is deposited
  into a volume-conserving wall depth field, so a single impact
  (:mod:`tensordem.erosion_dem.crater`) and the cumulative stream share one code
  path and one bookkeeping convention;
* the cumulative ratio ``E = removed wall mass / delivered erodent mass`` is the
  quantity the criteria are written on, and the mass ledger closes against the
  depth-field integral.

Comminution is deliberately outside this module.  :mod:`tensordem.erosion_dem.
breakage_bridge` injects particle fragmentation *between* bounces through the
``on_impact`` callback, which is how criterion C7 shows that the erosion ledger
residual is invariant to the breakage switch (and that the erosion core has no
import of the breakage machinery).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from tensordem.erosion.materials import Material
from tensordem.erosion.models import fit_velocity_exponent

from .contact import (
    JointImpactKinematics,
    oblique_restitution,
    plastic_normal_restitution,
    yield_velocity_m_s,
)
from .crater import SPHERE_FACTOR, OkaAnchoredLaw, WallSurface, single_impact_crater

__all__ = [
    "CASE_MATERIAL",
    "CASE_LAW",
    "CASE_ANCHOR_PROVENANCE",
    "StreamConfig",
    "run_impact_stream",
    "angle_sweep",
    "velocity_sweep",
    "aggregate_exponent",
    "peak_angle_from_samples",
    "monotonicity",
]

CASE_MATERIAL = Material(
    name="aluminium_soft_liner",
    material_class="ductile",
    hardness_hv_gpa=0.50,
    density_kg_m3=2700.0,
    source=(
        "soft aluminium liner / AlMg3 sheet, 51 HV = 0.50 GPa, 2700 kg/m3 "
        "(handbook hardness of annealed Al-Mg); the erodible target of the "
        "sand-jet case of node p176w2-n08, kept identical so the two "
        "verification faces are comparable"
    ),
)
"""Target material of the sand-jet case (ductile, low hardness).

The hardness is what places Oka's angle peak at 21.6 deg, i.e. inside the
ductile window [15, 25] deg, so the *stream* angle curve has its maximum where
the plastic-erosion model says it should.
"""

CASE_ANCHOR_PROVENANCE = (
    "one calibrated number: Oka's leading constant has a unit convention that "
    "cannot be re-derived offline, so the ratio at (v_ref=30 m/s, "
    "alpha_ref=90 deg, d_ref=150 um) is fixed once by least squares in log space "
    "against the DNV-RP-O501 ductile rate of the aluminium pair (n=2.4; the "
    "tabulated K is marked ASSUMED in tensordem.erosion.materials and cancels "
    "in the anchor). Per-point residuals after that fit are shape errors, "
    "criterion C3."
)


def _default_anchor() -> float:
    """Anchor placing the law on the DNV-RP-O501 ductile scale for the case."""
    k_dnv, n_dnv = 1.3e-9, 2.4          # DNV aluminium pair, ASSUMED tabulation
    return float(k_dnv * 30.0**n_dnv)   # F(90 deg) = 1


CASE_LAW = OkaAnchoredLaw.for_material(
    CASE_MATERIAL,
    _default_anchor(),
    v_ref_m_s=30.0,
    alpha_ref_deg=90.0,
    d_ref_um=150.0,
    anchor_provenance=CASE_ANCHOR_PROVENANCE,
)


@dataclass(frozen=True)
class StreamConfig:
    """Configuration of one impact stream (SI units unless a name says otherwise)."""

    n_particles: int = 4000
    d_p_m: float = 150e-6
    rho_p_kg_m3: float = 2650.0
    speed_mean_m_s: float = 30.0
    speed_cv: float = 0.10
    angle_mean_deg: float = 90.0
    angle_std_deg: float = 6.0
    height_m: float = 0.05
    gravity_m_s2: float = 9.81
    friction_coefficient: float = 0.35
    target_yield_strength_pa: float = 1.0e9
    reduced_modulus_pa: float = 5.0e10
    spot_sigma_m: float = 3.0e-3
    patch_extent_m: float = 0.012
    grid_cells: int = 96
    max_bounces: int = 4
    min_rebound_speed_m_s: float = 0.05
    seed: int = 20261111

    def __post_init__(self) -> None:
        if self.n_particles < 0:
            raise ValueError("n_particles must be >= 0")
        if self.d_p_m <= 0.0 or self.rho_p_kg_m3 <= 0.0:
            raise ValueError("d_p_m and rho_p_kg_m3 must be > 0")
        if self.speed_mean_m_s < 0.0:
            raise ValueError("speed_mean_m_s must be >= 0")
        if self.speed_cv < 0.0 or self.angle_std_deg < 0.0:
            raise ValueError("spreads must be >= 0")
        if not 0.0 <= self.angle_mean_deg <= 90.0:
            raise ValueError("angle_mean_deg must be within [0, 90]")
        if self.height_m <= 0.0:
            raise ValueError("height_m must be > 0")
        if self.gravity_m_s2 < 0.0:
            raise ValueError("gravity_m_s2 must be >= 0")
        if self.friction_coefficient < 0.0:
            raise ValueError("friction_coefficient must be >= 0")
        if self.patch_extent_m <= 0.0 or self.grid_cells < 2:
            raise ValueError("patch_extent_m must be > 0 and grid_cells >= 2")
        if self.max_bounces < 0:
            raise ValueError("max_bounces must be >= 0")


@dataclass
class _Ledger:
    impacts: int = 0
    delivered_mass_kg: float = 0.0
    removed_mass_kg: float = 0.0
    removed_volume_m3: float = 0.0
    ke_in_j: float = 0.0
    ke_out_j: float = 0.0
    contact_mass_kg: float = 0.0
    sibling_mass_kg: float = 0.0
    carried_mass_end_kg: float = 0.0
    inadmissible: int = 0
    crater_depths: list = field(default_factory=list)
    crater_ratios: list = field(default_factory=list)
    bounces: dict = field(default_factory=dict)


def run_impact_stream(
    cfg: StreamConfig | None = None,
    *,
    law: OkaAnchoredLaw | None = None,
    on_impact=None,
) -> dict:
    """March ``cfg.n_particles`` particles onto the wall and close the ledger.

    ``on_impact(record) -> dict | None`` is the comminution hook: it may return
    ``{"particle_scale": s, "sibling_mass_kg": m}`` to change the *carried*
    particle's size after the impact has been accounted, mass conservatively
    (``s`` is a mass ratio, the diameter follows as ``s**(1/3)``).  ``None``
    means no coupling, so the erosion path is unaffected.

    Returns scalars plus the :class:`WallSurface` and, for streams of at most
    20000 impacts, the per-impact records.
    """
    cfg = StreamConfig() if cfg is None else cfg
    law = CASE_LAW if law is None else law
    material = law.material
    rng = np.random.default_rng(cfg.seed)
    gravity = max(cfg.gravity_m_s2, 1e-9)
    flight = JointImpactKinematics(cfg.height_m, gravity)

    wall = WallSurface(
        cfg.grid_cells, cfg.grid_cells, (cfg.patch_extent_m / cfg.grid_cells) ** 2
    )
    ledger = _Ledger()
    records: list[dict] = []
    sigma_profile = max(cfg.spot_sigma_m, 1e-9)

    for _ in range(cfg.n_particles):
        speed = float(rng.normal(cfg.speed_mean_m_s, cfg.speed_cv * cfg.speed_mean_m_s))
        speed = max(0.0, speed)
        angle = float(rng.normal(cfg.angle_mean_deg, cfg.angle_std_deg))
        angle = min(90.0, max(0.5, angle))
        radius = float(np.hypot(rng.normal(0.0, sigma_profile), rng.normal(0.0, sigma_profile)))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        x, y = radius * math.cos(phi), radius * math.sin(phi)
        if speed <= 0.0:
            continue                       # nothing leaves the nozzle: no impact
        # The erodent flux that crosses the control surface is the *particle*
        # mass, counted once, however many times the particle touches the wall
        # (a bouncing grain is still the same kilogram of sand).  Dividing by
        # the contact-mass instead would dilute E with the sub-impacts and
        # destroy the angle curve -- documented in the report.
        delivered_per_particle = cfg.rho_p_kg_m3 * SPHERE_FACTOR * cfg.d_p_m**3
        ledger.delivered_mass_kg += delivered_per_particle
        d = float(cfg.d_p_m)
        kin = flight.impact(speed, angle)
        for bounce in range(cfg.max_bounces + 1):
            v_imp = float(kin["impact_speed_m_s"])
            a_imp = float(kin["impact_angle_deg"])
            m_p = cfg.rho_p_kg_m3 * SPHERE_FACTOR * d**3
            crater = single_impact_crater(
                v_imp, a_imp, d, cfg.rho_p_kg_m3, material, law=law
            )
            rx = float(crater.radius_m)
            wall.deposit(
                x + float(rng.uniform(-rx, rx)),
                y + float(rng.uniform(-rx, rx)),
                crater.removed_volume_m3,
                cfg.patch_extent_m,
            )
            v_n_imp = v_imp * math.sin(math.radians(a_imp))
            v_yield = yield_velocity_m_s(
                m_p, 0.5 * d, material.flow_stress_pa, cfg.reduced_modulus_pa
            )
            e_n = plastic_normal_restitution(v_n_imp, v_yield)
            reb = oblique_restitution(v_imp, a_imp, e_n, cfg.friction_coefficient)
            ke_out = 0.5 * m_p * (
                reb["normal_speed_out_m_s"] ** 2 + reb["tangential_speed_out_m_s"] ** 2
            )
            record = {
                "bounce_index": bounce,
                "particle_diameter_m": d,
                "particle_mass_kg": m_p,
                "impact_speed_m_s": v_imp,
                "impact_angle_deg": a_imp,
                "normal_restitution": e_n,
                "tangential_restitution": reb["tangential_restitution"],
                "erosion_ratio_kg_kg": (crater.removed_mass_kg / m_p) if m_p > 0 else 0.0,
                "removed_mass_kg": crater.removed_mass_kg,
                "removed_volume_m3": crater.removed_volume_m3,
                "crater_depth_m": crater.depth_m,
                "crater_radius_m": rx,
                "crater_admissible": crater.physically_admissible,
                "kinetic_energy_in_j": 0.5 * m_p * v_imp**2,
                "kinetic_energy_out_j": ke_out,
                "x_m": x,
                "y_m": y,
            }
            ledger.impacts += 1
            ledger.contact_mass_kg += m_p
            ledger.removed_mass_kg += crater.removed_mass_kg
            ledger.removed_volume_m3 += crater.removed_volume_m3
            ledger.ke_in_j += record["kinetic_energy_in_j"]
            ledger.ke_out_j += ke_out
            ledger.crater_depths.append(crater.depth_m)
            ledger.crater_ratios.append(record["erosion_ratio_kg_kg"])
            ledger.bounces[bounce] = ledger.bounces.get(bounce, 0) + 1
            if not crater.physically_admissible:
                ledger.inadmissible += 1
            if len(records) < 20000:
                records.append(record)

            if on_impact is not None:
                delta = on_impact(record)
                if delta:
                    scale = float(delta.get("particle_scale", 1.0))
                    if not scale > 0.0:
                        raise ValueError("particle_scale must be > 0")
                    if scale != 1.0:
                        d = d * scale ** (1.0 / 3.0)
                    ledger.sibling_mass_kg += float(delta.get("sibling_mass_kg", 0.0))

            v_n_out = reb["normal_speed_out_m_s"]
            v_t_out = reb["tangential_speed_out_m_s"]
            if bounce >= cfg.max_bounces or v_n_out < cfg.min_rebound_speed_m_s:
                break
            if v_t_out <= 0.0:
                break
            x += v_t_out * (2.0 * v_n_out / gravity)
            kin = {
                "impact_speed_m_s": math.hypot(v_n_out, v_t_out),
                "impact_angle_deg": math.degrees(
                    math.atan2(v_n_out, max(v_t_out, 1e-300))
                ),
            }
        # mass still carried by the particle when it leaves the wall zone: the
        # erodent bookkeeping invariant is delivered = carried_end + siblings
        ledger.carried_mass_end_kg += cfg.rho_p_kg_m3 * SPHERE_FACTOR * d**3

    delivered = ledger.delivered_mass_kg
    ratio = (ledger.removed_mass_kg / delivered) if delivered > 0.0 else 0.0
    depths = np.asarray(ledger.crater_depths, float) if ledger.crater_depths else np.zeros(1)
    ratios = np.asarray(ledger.crater_ratios, float) if ledger.crater_ratios else np.zeros(1)
    return {
        "particles_launched": int(cfg.n_particles),
        "impacts": ledger.impacts,
        "bounce_histogram": dict(sorted(ledger.bounces.items())),
        "delivered_mass_kg": delivered,
        "contact_mass_kg": ledger.contact_mass_kg,
        "removed_mass_kg": ledger.removed_mass_kg,
        "removed_volume_m3": ledger.removed_volume_m3,
        "erosion_ratio_kg_kg": ratio,
        "closure_residual": wall.closure_residual,
        "wall_integrated_volume_m3": wall.integrated_volume_m3,
        "wall_max_depth_m": wall.max_depth_m,
        "wall_mean_depth_m": wall.mean_depth_m,
        "crater_depth_mean_m": float(depths.mean()),
        "crater_depth_max_m": float(depths.max()),
        "crater_ratio_mean_kg_kg": float(ratios.mean()),
        "inadmissible_craters": ledger.inadmissible,
        "kinetic_energy_in_j": ledger.ke_in_j,
        "kinetic_energy_out_j": ledger.ke_out_j,
        "sibling_mass_kg": ledger.sibling_mass_kg,
        "carried_mass_end_kg": ledger.carried_mass_end_kg,
        "wall": wall,
        "records": records,
    }


def angle_sweep(
    angles_deg,
    *,
    cfg: StreamConfig | None = None,
    law: OkaAnchoredLaw | None = None,
    **overrides,
) -> dict:
    """Stream erosion ratio versus launch angle (monodisperse, deterministic)."""
    base = cfg or StreamConfig()
    ratios, impacts, mean_angles = [], [], []
    for a in angles_deg:
        run_cfg = replace(
            base,
            speed_cv=0.0,
            angle_std_deg=0.0,
            angle_mean_deg=float(a),
            **overrides,
        )
        out = run_impact_stream(run_cfg, law=law)
        ratios.append(out["erosion_ratio_kg_kg"])
        impacts.append(out["impacts"])
        mean_angles.append(
            float(np.mean([r["impact_angle_deg"] for r in out["records"]]))
            if out["records"]
            else float(a)
        )
    return {
        "launch_angles_deg": [float(a) for a in angles_deg],
        "measured_impact_angles_deg": mean_angles,
        "erosion_ratios_kg_kg": ratios,
        "impacts": impacts,
    }


def velocity_sweep(
    speeds_m_s,
    *,
    cfg: StreamConfig | None = None,
    law: OkaAnchoredLaw | None = None,
    **overrides,
) -> dict:
    """Stream erosion ratio versus launch speed at a fixed angle."""
    base = cfg or StreamConfig()
    ratios, impacts = [], []
    for v in speeds_m_s:
        run_cfg = replace(
            base,
            speed_mean_m_s=float(v),
            speed_cv=0.0,
            angle_std_deg=0.0,
            **overrides,
        )
        out = run_impact_stream(run_cfg, law=law)
        ratios.append(out["erosion_ratio_kg_kg"])
        impacts.append(out["impacts"])
    return {
        "launch_speeds_m_s": [float(v) for v in speeds_m_s],
        "erosion_ratios_kg_kg": ratios,
        "impacts": impacts,
    }


def aggregate_exponent(speeds_m_s, ratios) -> dict:
    """Fit ``n`` of ``E ~ v**n`` from a stream sweep."""
    n, rms = fit_velocity_exponent(speeds_m_s, ratios)
    return {"exponent": float(n), "log_residual_rms": float(rms)}


def peak_angle_from_samples(angles_deg, ratios) -> dict:
    """Peak of a sampled angle curve, refined by a parabola through 3 points."""
    a = np.asarray(angles_deg, dtype=float)
    r = np.asarray(ratios, dtype=float)
    if a.size < 3:
        raise ValueError("need at least 3 angle samples")
    i = int(np.argmax(r))
    if 0 < i < a.size - 1:
        x1, x2, x3 = a[i - 1], a[i], a[i + 1]
        y1, y2, y3 = r[i - 1], r[i], r[i + 1]
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if denom != 0.0:
            A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            B = (
                x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)
            ) / denom
            if A < 0.0:
                return {
                    "peak_angle_deg": float(-B / (2.0 * A)),
                    "peak_ratio_kg_kg": float(max(r)),
                    "grid_argmax_deg": float(a[i]),
                    "refined": True,
                }
    return {
        "peak_angle_deg": float(a[i]),
        "peak_ratio_kg_kg": float(r[i]),
        "grid_argmax_deg": float(a[i]),
        "refined": False,
    }


def monotonicity(xs, ys, *, increasing: bool, tol: float = 0.0) -> dict:
    """Check that a sampled curve is monotone (with an absolute tolerance)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("xs and ys must be equally long and have >= 2 samples")
    d = np.diff(y)
    ok = (d >= -tol) if increasing else (d <= tol)
    worst = float(d.min()) if increasing else float(d.max())
    return {
        "monotone": bool(np.all(ok)),
        "samples": int(x.size),
        "worst_step": worst,
        "violations": int(np.count_nonzero(~ok)),
    }
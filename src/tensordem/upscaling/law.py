"""Coarse-graining (upscaling) scaling laws for monodisperse sphere DEM.

Physical background
-------------------
Coarse-graining replaces a fine granular medium (grain radius ``R``, grain
mass ``m``) by an f-times coarser one (``R* = f R``, ``m* = f^3 m`` for
constant solid density) and rescales the contact parameters so that the
*macroscopic* response of the two models coincides.  There is no unique
rescaling: two independent dimensionless requirements can be imposed, and
they are incompatible.

口径 A — contact-time invariance (``t_c* = t_c``)
    ``t_c = pi sqrt(m_eff / k_n)`` with ``m_eff = m/2``.
    ``m_eff -> f^3 m_eff`` therefore requires ``k_n -> f^3 k_n``.
    Consequences: ``k_t -> f^3 k_t``, ``k_r = k_t R^2 -> f^5 k_r``,
    damping ``gamma = 2 zeta sqrt(m_eff k_n) -> f^3 gamma``,
    contact force ``F = k_n delta`` with ``delta -> f delta`` gives
    ``F -> f^4 F`` (stress ``-> f^2 sigma``, i.e. ``E_eff -> f^2 E_eff``),
    particle wave speed ``c = sqrt(k_n/m_eff) R -> f c``.
    Dimensionless stiffness ``k_n/(m g)`` is *invariant* -> gravity-driven
    quasi-static similarity is preserved exactly.

口径 B — stiffness (sound-speed) invariance (``c* = c``, ``E_eff* = E_eff``)
    ``c = R sqrt(k_n/m_eff)`` and ``E ~ k_n / R`` (linear spring packing)
    require ``k_n -> f k_n``; ``k_t -> f k_t``, ``k_r -> f^3 k_r``,
    ``gamma = 2 zeta sqrt(m_eff k_n) -> f^2 gamma``.
    Consequences: ``t_c -> f t_c`` (contact time grows), ``F -> f^2 F``
    (stress and ``E_eff`` invariant), ``k_n/(m g) -> f^{-2}`` (the medium
    becomes *softer relative to gravity*; the contact overlap ratio grows
    proportional to f for self-weight driven problems).

Both口径 keep the contact *restitution* ``e``, the Coulomb friction ``mu``,
the rolling friction ``mu_r`` and gravity ``g`` invariant (dimensionless
similarity).  The dashpot coefficient is obtained by *inverting* the
restitution relation ``zeta = |ln e| / sqrt(pi^2 + ln^2 e)`` and then
``gamma = 2 zeta sqrt(m_eff k_n)``; :func:`restitution_from_damping_ratio`
is the exact inverse and is used as a round-trip verification gate.

Incorrect (negative-control) rescalings implemented here so that machine
criteria can reject them:

* ``mass_only``: ``m -> f^3 m`` but ``k_n`` unchanged.  Then
  ``t_c -> f^{3/2} t_c`` (> 180 % deviation at f = 2): dynamic similarity
  is destroyed.
* ``kn_f3_for_stiffness``: 口径 A applied to a problem whose acceptance
  criterion is macroscopic stiffness invariance.  ``E_eff -> f^2 E_eff``
  (300 % at f = 2, 6300 % at f = 8): rejected by the ``E_eff`` criterion.

All quantities are SI (m, kg, s, N, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

# ---------------------------------------------------------------------------
# Restitution <-> damping (exact round trip)
# ---------------------------------------------------------------------------


def damping_ratio_from_restitution(restitution: float) -> float:
    """zeta = |ln e| / sqrt(pi^2 + ln^2 e) (linear spring-dashpot)."""
    e = max(min(float(restitution), 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def restitution_from_damping_ratio(zeta: float) -> float:
    """Exact inverse of :func:`damping_ratio_from_restitution`.

    ``e = exp(-pi zeta / sqrt(1 - zeta^2))`` (underdamped solution of the
    linear spring-dashpot impact problem).  Used to *reverse-solve* the
    dashpot coefficient from a target restitution and to verify the
    round trip.
    """
    z = float(zeta)
    if z <= 0.0:
        return 1.0
    if z >= 1.0:
        return 0.0
    return math.exp(-math.pi * z / math.sqrt(1.0 - z * z))


def pair_damping(mass_eff: float, kn: float, restitution: float) -> float:
    """gamma = 2 zeta sqrt(m_eff k_n)  [N s / m]."""
    return 2.0 * damping_ratio_from_restitution(restitution) * math.sqrt(mass_eff * kn)


# ---------------------------------------------------------------------------
# Dashpot calibration: reverse-solve c_n so that the *simulated* rebound
# matches the target restitution
# ---------------------------------------------------------------------------
#
# The closed-form gamma = 2 zeta sqrt(m_eff k_n) reproduces e only for the
# ideal linear spring-dashpot without the f_n >= 0 (no-tension) clip.  With
# the clip the contact detaches slightly early during restitution and the
# realised restitution is higher than requested (e.g. 0.46 instead of 0.30
# at e=0.30, dt = t_c/20).  We therefore *reverse-solve* a damping multiplier
# alpha such that the simulated head-on rebound equals the target e:
#
#     gamma = alpha(e, dt/t_c) * 2 zeta(e) sqrt(m_eff k_n)
#
# alpha depends only on the dimensionless step ratio dt/t_c, which both
# scaling口径 keep invariant, so the law exponents (gamma: f^3 / f^2) and the
# restitution invariance are unaffected.

_CALIBRATION_CACHE: Dict[tuple, float] = {}

# Explicit integration of the damping term is stable only while
# (gamma/m_eff) dt < 2; at dt = t_c/20 that bounds alpha <= ~6.
ALPHA_MAX = 6.0


def _sim_restitution(gamma_scale: float, restitution: float,
                     steps_per_contact: float, v0: float = 0.01) -> float:
    """Nondimensional head-on collision (k=1, m=1, m_eff=0.5) -> e_measured.

    Same contact law and semi-implicit update as :mod:`tensordem.upscaling.engine`:
    ``f_n = max(k delta - gamma * ydot, 0)`` (no-tension clip).
    """
    kn, m_eff, radius = 1.0, 0.5, 0.5
    zeta = damping_ratio_from_restitution(restitution)
    gamma = gamma_scale * 2.0 * zeta * math.sqrt(m_eff * kn)
    dt = math.pi * math.sqrt(m_eff / kn) / steps_per_contact
    y, ydot = 2.0 * radius, -v0            # relative position / velocity
    for _ in range(int(200.0 * steps_per_contact)):
        delta = 2.0 * radius - y
        f = max(kn * delta - gamma * ydot, 0.0) if delta > 0.0 else 0.0
        ydot += (f / m_eff) * dt
        y += ydot * dt
        if ydot > 0.0 and (2.0 * radius - y) < 0.0:
            return abs(ydot) / v0
    return 1.0


def calibrated_damping_scale(restitution: float, steps_per_contact: float = 20.0,
                             tol: float = 1.0e-3, alpha_max: float = ALPHA_MAX) -> float:
    """Reverse-solve the dashpot multiplier alpha so the rebound equals ``e``.

    ``gamma = alpha * 2 zeta sqrt(m_eff k_n)``.  The no-tension clip makes the
    realised restitution differ from the closed-form value, so alpha is found
    by bisection on a single-pair collision (nondimensional, cached).
    alpha > 1 means "more damping than the closed form", alpha < 1 "less";
    if the target is out of reach (e below the attainable minimum) the largest
    stable value is returned and reported by :func:`simulated_restitution`.
    """
    key = (round(float(restitution), 8), round(float(steps_per_contact), 6))
    if key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[key]
    target = float(restitution)
    grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, alpha_max]
    es = [_sim_restitution(a, restitution, steps_per_contact) for a in grid]
    lo = hi = None
    for i in range(len(grid) - 1):
        if es[i] >= target >= es[i + 1]:
            lo, hi = grid[i], grid[i + 1]
            break
    if lo is None:
        alpha = alpha_max if target < es[-1] else grid[0]
        _CALIBRATION_CACHE[key] = alpha
        return alpha
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        e_mid = _sim_restitution(mid, restitution, steps_per_contact)
        if abs(e_mid - target) <= tol:
            lo = hi = mid
            break
        if e_mid > target:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    _CALIBRATION_CACHE[key] = alpha
    return alpha


def simulated_restitution(restitution: float, steps_per_contact: float = 20.0,
                          alpha: float | None = None) -> float:
    """Realised restitution of the calibrated contact law (for verification)."""
    if alpha is None:
        alpha = calibrated_damping_scale(restitution, steps_per_contact)
    return _sim_restitution(alpha, restitution, steps_per_contact)


# ---------------------------------------------------------------------------
# Material models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseMaterial:
    """Fine (reference) material: monodisperse spheres, linear spring-dashpot."""

    radius: float = 1.0e-3          # m
    density: float = 2500.0         # kg/m^3
    kn: float = 100.0               # N/m
    kt_ratio: float = 0.5           # k_t / k_n
    restitution: float = 0.3        # -
    mu: float = 0.5                 # - (particle-particle and wall)
    mu_rolling: float = 0.05        # -
    gravity: float = 9.81           # m/s^2
    steps_per_contact: float = 20.0  # dt = t_c / steps_per_contact

    # -- derived ---------------------------------------------------------
    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def kr(self) -> float:
        """Rolling stiffness; k_r = k_t R^2 (torque per unit relative spin)."""
        return self.kt * self.radius ** 2

    @property
    def zeta(self) -> float:
        return damping_ratio_from_restitution(self.restitution)

    @property
    def damping_scale(self) -> float:
        """Dashpot multiplier reverse-solved from the target restitution."""
        return calibrated_damping_scale(self.restitution, self.steps_per_contact)

    @property
    def gamma_pair(self) -> float:
        return self.damping_scale * pair_damping(0.5 * self.mass, self.kn, self.restitution)

    @property
    def gamma_wall(self) -> float:
        # wall (infinite mass) contact: m_eff = m
        return self.damping_scale * pair_damping(self.mass, self.kn, self.restitution)

    @property
    def contact_time(self) -> float:
        return math.pi * math.sqrt((0.5 * self.mass) / self.kn)

    @property
    def timestep(self) -> float:
        return self.contact_time / self.steps_per_contact

    @property
    def stiffness_ratio(self) -> float:
        """Dimensionless stiffness k_n / (m g) - controls gravitational similarity."""
        return self.kn / (self.mass * self.gravity)


APPROACHES = ("contact_time", "stiffness_invariant")
NEGATIVE_CONTROLS = ("mass_only", "kn_f3_for_stiffness")
ALL_VARIANTS = APPROACHES + NEGATIVE_CONTROLS


@dataclass(frozen=True)
class ScaledMaterial:
    """f-times coarse-grained material produced by a given scaling口径."""

    f: float
    approach: str
    radius: float
    density: float
    mass: float
    inertia: float
    kn: float
    kt: float
    kr: float
    gamma_pair: float
    gamma_wall: float
    restitution: float
    mu: float
    mu_rolling: float
    gravity: float
    contact_time: float
    timestep: float
    zeta: float
    damping_scale: float
    restitution_realised: float
    force_exponent: float
    stress_exponent: float
    notes: str = ""

    # -- convenience -----------------------------------------------------
    @property
    def steps_per_contact(self) -> float:
        return self.contact_time / self.timestep

    @property
    def stiffness_ratio(self) -> float:
        return self.kn / (self.mass * self.gravity)

    @property
    def wave_speed(self) -> float:
        """Particle-level wave speed c = sqrt(k_n/m_eff) R."""
        return math.sqrt(self.kn / (0.5 * self.mass)) * self.radius

    def overlap_ratio(self, confining_stress: float) -> float:
        """Contact overlap / radius for a particle carrying ``sigma (2R)^2``.

        Linear spring model: ``delta = F / k_n`` with ``F = sigma (2R)^2``;
        the ratio ``delta/R = 4 sigma R / k_n`` quantifies deformability
        (linear-contact validity is conventionally asserted for < 2 %).
        """
        return 4.0 * confining_stress * self.radius / self.kn

    def to_dict(self) -> Dict[str, float]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["steps_per_contact"] = self.steps_per_contact
        d["stiffness_ratio"] = self.stiffness_ratio
        d["wave_speed"] = self.wave_speed
        return d


def scale_material(base: BaseMaterial, f: float, approach: str = "contact_time",
                   calibrate_damping: bool = True) -> ScaledMaterial:
    """Apply the f-fold coarse-graining law ``approach`` to ``base``.

    Parameters
    ----------
    f : float
        Coarse-graining factor (> 0).  ``f = 1`` returns the base material.
    approach : str
        ``contact_time`` (口径 A), ``stiffness_invariant`` (口径 B),
        ``mass_only`` or ``kn_f3_for_stiffness`` (negative controls).
    """
    if f <= 0.0:
        raise ValueError("f must be positive")
    if approach not in ALL_VARIANTS:
        raise ValueError(f"unknown approach {approach!r}; choose from {ALL_VARIANTS}")

    radius = f * base.radius
    mass = base.density * 4.0 / 3.0 * math.pi * radius ** 3
    inertia = 0.4 * mass * radius ** 2

    if approach in ("contact_time", "kn_f3_for_stiffness"):
        kn = f ** 3 * base.kn
        kt = f ** 3 * base.kt
        kr = f ** 5 * base.kr
        approach_note = "k_n ∝ f³ (contact-time invariant)"
    elif approach == "stiffness_invariant":
        kn = f * base.kn
        kt = f * base.kt
        kr = f ** 3 * base.kr
        approach_note = "k_n ∝ f (stiffness / sound-speed invariant)"
    else:  # mass_only
        kn = base.kn
        kt = base.kt
        kr = base.kr
        approach_note = "k_n unchanged after mass scaling (negative control)"

    zeta = base.zeta  # restitution is a dimensionless invariant
    alpha = (calibrated_damping_scale(base.restitution, base.steps_per_contact)
             if calibrate_damping else 1.0)
    gamma_pair = alpha * 2.0 * zeta * math.sqrt(0.5 * mass * kn)
    gamma_wall = alpha * 2.0 * zeta * math.sqrt(mass * kn)
    contact_time = math.pi * math.sqrt((0.5 * mass) / kn)
    timestep = contact_time / base.steps_per_contact

    # Force / stress / modulus scaling exponents (against the base model).
    if approach in ("contact_time", "kn_f3_for_stiffness"):
        force_exp, stress_exp = 4.0, 2.0
    elif approach == "stiffness_invariant":
        force_exp, stress_exp = 2.0, 0.0
    else:
        force_exp, stress_exp = 1.0, 0.0  # m->f^3 m, k unchanged: delta invariant

    return ScaledMaterial(
        f=float(f),
        approach=approach,
        radius=radius,
        density=base.density,
        mass=mass,
        inertia=inertia,
        kn=kn,
        kt=kt,
        kr=kr,
        gamma_pair=gamma_pair,
        gamma_wall=gamma_wall,
        restitution=base.restitution,
        mu=base.mu,
        mu_rolling=base.mu_rolling,
        gravity=base.gravity,
        contact_time=contact_time,
        timestep=timestep,
        zeta=zeta,
        damping_scale=alpha,
        restitution_realised=_sim_restitution(alpha, base.restitution,
                                              base.steps_per_contact),
        force_exponent=force_exp,
        stress_exponent=stress_exp,
        notes=approach_note,
    )


# ---------------------------------------------------------------------------
# Rule table / applicability domain (machine readable, rendered in docs)
# ---------------------------------------------------------------------------

SCALE_TABLE: Dict[str, Dict[str, str]] = {
    "radius":        {"contact_time": "f·R",      "stiffness_invariant": "f·R"},
    "mass":          {"contact_time": "f³·m",     "stiffness_invariant": "f³·m"},
    "inertia":       {"contact_time": "f⁵·I",     "stiffness_invariant": "f⁵·I"},
    "k_n":           {"contact_time": "f³·k_n",   "stiffness_invariant": "f·k_n"},
    "k_t":           {"contact_time": "f³·k_t",   "stiffness_invariant": "f·k_t"},
    "k_r":           {"contact_time": "f⁵·k_r",   "stiffness_invariant": "f³·k_r"},
    "gamma_p":       {"contact_time": "f³·γ",     "stiffness_invariant": "f²·γ"},
    "gamma_w":       {"contact_time": "f³·γ_w",   "stiffness_invariant": "f²·γ_w"},
    "restitution":   {"contact_time": "e",        "stiffness_invariant": "e"},
    "mu":            {"contact_time": "μ",        "stiffness_invariant": "μ"},
    "mu_rolling":    {"contact_time": "μ_r",      "stiffness_invariant": "μ_r"},
    "gravity":       {"contact_time": "g",        "stiffness_invariant": "g"},
    "contact_time":  {"contact_time": "t_c",      "stiffness_invariant": "f·t_c"},
    "timestep":      {"contact_time": "Δt",       "stiffness_invariant": "f·Δt"},
    "wave_speed":    {"contact_time": "f·c",      "stiffness_invariant": "c"},
    "force":         {"contact_time": "f⁴·F",     "stiffness_invariant": "f²·F"},
    "stress/E_eff":  {"contact_time": "f²·σ",     "stiffness_invariant": "σ"},
    "k_n/(m g)":     {"contact_time": "invariant", "stiffness_invariant": "f⁻²"},
}

APPLICABILITY: Dict[str, Dict[str, object]] = {
    "contact_time": {
        "preserves": [
            "contact time t_c", "dimensionless stiffness k_n/(m g)",
            "restitution & friction", "gravity-driven quasi-static similarity",
        ],
        "violates": ["E_eff / stress (∝ f²)", "particle wave speed (∝ f)",
                     "force (∝ f⁴)"],
        "valid_use": "quasi-static gravity-driven flow, angle of repose, shear strength, "
                     "impact/contact-time-sensitive dynamics",
        "failure_boundary": "any criterion phrased on absolute stiffness/stress/wave speed: "
                            "E_eff deviation is f² (300 % at f=2, 6300 % at f=8)",
        "overlap_ratio": "∝ 1/f² under imposed stress, ∝ 1/f under self weight (gets stiffer)",
    },
    "stiffness_invariant": {
        "preserves": ["E_eff / stress", "particle wave speed c", "contact force per area"],
        "violates": ["contact time (∝ f)", "k_n/(m g) (∝ f⁻²)"],
        "valid_use": "wave propagation, elastic loading of stiff packings, uniaxial/triaxial "
                     "modulus, dynamic compaction",
        "failure_boundary": "self-weight driven problems: contact overlap ratio ∝ f, "
                            "linear-contact validity (δ/R ≤ 2 %) is lost for large f",
        "overlap_ratio": "∝ f under self weight, invariant under imposed stress",
    },
    "mass_only": {
        "preserves": ["nothing relevant (negative control)"],
        "violates": ["t_c (∝ f^{3/2})", "everything dynamic"],
        "valid_use": "negative control only",
        "failure_boundary": "rejected by every dynamic similarity criterion",
        "overlap_ratio": "∝ f³ under self weight",
    },
    "kn_f3_for_stiffness": {
        "preserves": ["contact-time similarity (as 口径 A)"],
        "violates": ["E_eff / stress (∝ f²) - the criterion this control claims to satisfy"],
        "valid_use": "negative control: 口径 A misapplied to a stiffness-invariance problem",
        "failure_boundary": "rejected by the E_eff ≤ 10 % criterion for any f > 1",
        "overlap_ratio": "∝ 1/f² with imposed stress",
    },
}


def scaling_rule_table() -> Dict[str, Dict[str, str]]:
    """Return the rule table (quantity -> {口径: expression})."""
    return {k: dict(v) for k, v in SCALE_TABLE.items()}


def applicability_domain() -> Dict[str, Dict[str, object]]:
    """Return declared applicability domains and failure boundaries."""
    return {k: dict(v) for k, v in APPLICABILITY.items()}


# ---------------------------------------------------------------------------
# Time-step / stability gate
# ---------------------------------------------------------------------------

DEFAULT_LIMITS = {
    "min_steps_per_contact": 20.0,      # dt <= t_c/20
    "max_overlap_ratio": 0.02,          # linear contact model validity
    "max_advective_fraction": 0.05,     # v_max dt / R <= 5 %
    "min_particles_per_diameter": 15.0, # specimen resolution
}


def stability_gate(model: ScaledMaterial, confining_stress: float,
                   max_velocity: float,
                   limits: Dict[str, float] | None = None) -> Dict[str, object]:
    """Check the numerical-stability gates for one scaled model.

    Gates (all must pass):
      * ``t_c / dt >= 20`` - contact period resolved;
      * overlap ratio ``δ/R ≤ 2 %`` - linear contact law still meaningful;
      * advective fraction ``v dt / R ≤ 5 %`` - no particle tunnels a grain
        diameter in one step;
      * ``e < 1`` and ``gamma > 0`` - dissipative dashpot present.
    """
    lim = dict(DEFAULT_LIMITS)
    if limits:
        lim.update(limits)
    spc = model.contact_time / model.timestep
    overlap = model.overlap_ratio(confining_stress)
    adv = max_velocity * model.timestep / model.radius if model.radius > 0 else float("inf")
    checks = {
        "steps_per_contact": spc,
        "steps_per_contact_ok": spc >= lim["min_steps_per_contact"] - 1e-9,
        "overlap_ratio": overlap,
        "overlap_ok": overlap <= lim["max_overlap_ratio"],
        "advective_fraction": adv,
        "advective_ok": adv <= lim["max_advective_fraction"],
        "dashpot_ok": (model.gamma_pair > 0.0) and (model.restitution < 1.0),
    }
    checks["pass"] = all(bool(checks[k]) for k in
                         ("steps_per_contact_ok", "overlap_ok", "advective_ok", "dashpot_ok"))
    return checks
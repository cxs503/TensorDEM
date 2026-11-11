"""Ice model-test scale cascade: Cauchy collapse and mismatch deviation law.

Why this module exists
----------------------
A model test of a ship in broken ice is only useful if the measured resistance
can be carried back to full scale.  The classical statement of that transfer is
a collapse in dimensionless space: with the ice crushing strength ``sigma_f``,
the ice thickness ``t`` and the hull length ``L`` as the repeating set,

    R* = R / (sigma_f t^2) = f(Fn, Sigma, Mod, t/L, phi, mu)
    Fn    = v / sqrt(g L)                (Froude number)
    Sigma = sigma_f / (rho_i g t)        (strength / buoyancy)
    Mod   = E / sigma_f                  (elastic modulus / strength)
    phi   = ice concentration, mu = friction

A "Cauchy" model at geometric scale ``1:lam`` matches *all* of them and the
measured ``R*`` must be independent of ``lam``.  Engineering practice
(Froude + strength scaling of the model ice) does **not** match ``Mod``, because
the modulus of model ice cannot be chosen freely; the residual is exactly the
model-test scale effect this module quantifies:

    delta(lam) = (R*_model(lam) - R*_ref) / R*_ref

Why the DEM can answer it
-------------------------
:class:`tensordem.ice_channel.IceChannelSimulation` is a plan-view penalty
contact DEM with mass ``m = rho_i pi r^2 t`` and normal stiffness

    k_n = 2 sigma_c t / sqrt(frac),     frac = reference_penetration_fraction

so under ``length -> s * length``, ``sigma -> s * sigma``, ``v -> s^(1/2) v``
(``rho`` and ``g`` fixed) every term of the force balance scales as ``s^3``:
``k_n pen ~ (s^2)(s)``, the normal dashpot ``c_n v ~ sqrt(k_n m) v ~ s^(2.5) s^(0.5)``,
the quadratic water drag ``rho * side * v^2 ~ s^2 s`` and the heave stiffness
``rho g A_wp ~ s^2`` (heave period ``~ s^(1/2)``).  The model is therefore
*analytically* Cauchy covariant, which is what makes it a usable reference: a
measured departure from collapse is either (i) the intended ``Mod`` mismatch or
(ii) an absolute numerical scale leaking in (fixed time step, fixed contact
tolerance, seed-induced field differences) -- and (ii) is controlled separately
by the resolution and contacts-per-floe tests.

Identifying ``Mod`` in the DEM
------------------------------
The only compliance in the model is the penalty fraction ``frac``: ``k_n`` is
the numerical surrogate of the elastic contact stiffness, which for a disc-like
plane-stress contact is ``k_n ~ E t`` up to an order-one geometric factor.  From
``k_n = 2 sigma_c t / sqrt(frac)``,

    sqrt(frac) = 2 sigma_f / (c_E E)  =>  frac ~ (sigma_f / E)^2 ~ Mod^-2.

Only the *ratio* ``q = Mod / Mod_ref`` carries physics here, so the mapping is
anchored at :data:`REFERENCE_FRACTION` / :data:`REFERENCE_MODULUS_RATIO`; this is
a declared modelling assumption, not a derived law -- see
``docs/ice_scale_cascade.md`` for the limitation.

Protocols
---------
``P-Cauchy``           all lengths ``/lam``, ``v/sqrt(lam)``, ``sigma/lam``,
                       ``E/lam``  ->  ``Mod`` matched (``q = 1``).
``P-Froude+strength``  engineering protocol: ``Fn`` and ``Sigma`` matched,
                       modulus left to the model-ice material,
                       ``E/lam^p_E`` (``p_E = 0.5`` default) -> ``q = lam^(1-p_E)``.
``P-mismatch``         Cauchy plus a deliberate ``Mod`` perturbation
                       ``q = 1 + dMod`` with ``dMod`` in ``[-0.3, +0.3]``.

All quantities SI (m, Pa, kg/m3, m/s, N); forces are the total planar force of a
body of thickness ``t`` (thickness-aware 2-D model, not per unit width).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ice_channel import (
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
)

ICE_DENSITY = 917.0
WATER_DENSITY = 1025.0
GRAVITY = 9.80665

#: Reference (full-scale) scenario the cascade is anchored on: a 20 m beam hull
#: in 1.0 m broken ice at 2 m/s (``Fn = 0.071``).
PROTOTYPE: Dict[str, float] = dict(
    beam_m=20.0,
    thickness_m=1.0,
    speed_m_s=2.0,
    concentration=0.7,
    strength_pa=1.0e6,
    friction=0.3,
    modulus_pa=5.0e9,
)

#: Penalty fraction anchored at the reference ``Mod``.  0.05 (5x softer than the
#: channel module default 0.01) keeps ``Mod`` observable: in the stiff limit
#: ``frac -> 0`` the contact becomes rigid and ``R*`` saturates, which would make
#: the deviation law a no-op.
REFERENCE_FRACTION = 0.05
REFERENCE_MODULUS_RATIO = PROTOTYPE["modulus_pa"] / PROTOTYPE["strength_pa"]  # 5000

PROTOCOL_CAUCHY = "P-Cauchy"
PROTOCOL_FROUDE = "P-Froude+strength"
PROTOCOL_MISMATCH = "P-mismatch"
PROTOCOLS: Tuple[str, ...] = (PROTOCOL_CAUCHY, PROTOCOL_FROUDE, PROTOCOL_MISMATCH)

#: Scale ladder of the cascade (1 = full-scale anchor, the rest are models).
DEFAULT_LAMBDAS: Tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 40.0)


# ---------------------------------------------------------------------------
# scaling rules
# ---------------------------------------------------------------------------

def modulus_fraction(modulus_ratio: float) -> float:
    """Penalty fraction that represents ``Mod = E/sigma_f`` (``frac ~ Mod^-2``)."""
    if not math.isfinite(modulus_ratio) or modulus_ratio <= 0.0:
        raise ValueError(f"modulus_ratio must be positive and finite, got {modulus_ratio!r}")
    frac = REFERENCE_FRACTION * (REFERENCE_MODULUS_RATIO / modulus_ratio) ** 2
    if not (0.0 < frac < 0.5):
        raise ValueError(
            f"modulus_ratio={modulus_ratio:g} maps to frac={frac:g}, outside the "
            "contact model's admissible (0, 0.5); widen the anchor or narrow the sweep")
    return frac


def nondimensional_groups(*, speed_m_s: float, hull_length_m: float,
                          strength_pa: float, thickness_m: float,
                          modulus_pa: float, concentration: float,
                          friction: float, beam_m: float) -> Dict[str, float]:
    """The six groups of the collapse statement plus two shape ratios."""
    return {
        "Fn": speed_m_s / math.sqrt(GRAVITY * hull_length_m),
        "Sigma": strength_pa / (ICE_DENSITY * GRAVITY * thickness_m),
        "Mod": modulus_pa / strength_pa,
        "t_over_L": thickness_m / hull_length_m,
        "phi": concentration,
        "mu": friction,
        "B_over_t": beam_m / thickness_m,
        "L_over_t": hull_length_m / thickness_m,
    }


@dataclass(frozen=True)
class ScaleCase:
    """One member of the cascade: a scale ratio, a protocol and the DEM knobs."""

    lam: float = 10.0
    protocol: str = PROTOCOL_CAUCHY
    #: ``Mod = E/sigma_f`` mismatch factor of this case relative to the Cauchy
    #: model at the same ``lam``: ``q = Mod_case / (E_p sigma_p^-1 / lam)``.
    modulus_factor: float = 1.0
    #: exponent of the engineering protocol's modulus rule ``E_m = E_p lam^-p_E``
    modulus_exponent: float = 0.5
    beam_m: float = PROTOTYPE["beam_m"]
    thickness_m: float = PROTOTYPE["thickness_m"]
    speed_m_s: float = PROTOTYPE["speed_m_s"]
    concentration: float = PROTOTYPE["concentration"]
    strength_pa: float = PROTOTYPE["strength_pa"]
    friction: float = PROTOTYPE["friction"]
    modulus_pa: float = PROTOTYPE["modulus_pa"]
    seed: int = 20261110
    dt_scale: float = 0.25
    #: geometric-similarity ratios (identical at every scale -> same floe count)
    floe_radius_ratio: float = 0.35
    hull_length_ratio: float = 4.0
    domain_width_ratio: float = 6.0
    domain_length_ratio: float = 14.0
    radius_jitter: float = 0.08
    contact_sample_every: int = 20
    label: str = ""

    # -- validation ------------------------------------------------------
    def __post_init__(self) -> None:
        if not (math.isfinite(self.lam) and self.lam >= 1.0):
            raise ValueError(f"lam must be >= 1 (got {self.lam!r})")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if not (0.0 <= self.concentration <= 1.0):
            raise ValueError("concentration must lie in [0, 1]")
        for name in ("thickness_m", "beam_m", "speed_m_s", "strength_pa",
                     "modulus_pa", "dt_scale", "floe_radius_ratio"):
            if not (getattr(self, name) > 0.0):
                raise ValueError(f"{name} must be positive (got {getattr(self, name)!r})")
        if self.contact_sample_every < 1:
            raise ValueError("contact_sample_every must be >= 1")
        if not (0.0 < self.reference_penetration_fraction < 0.5):
            raise ValueError(
                f"case modulus ratio {self.modulus_ratio:g} is outside the contact "
                "model's admissible penetration fraction")

    # -- geometry --------------------------------------------------------
    @property
    def floe_radius_m(self) -> float:
        return self.floe_radius_ratio * self.beam_m

    @property
    def hull_length_m(self) -> float:
        return self.hull_length_ratio * self.beam_m

    @property
    def domain_m(self) -> Tuple[float, float]:
        return (self.domain_length_ratio * self.beam_m,
                self.domain_width_ratio * self.beam_m)

    @property
    def hull_entry_offset_m(self) -> float:
        """Start one hull length + 3 floe radii upstream of the field edge."""
        return self.hull_length_m + 3.0 * self.floe_radius_m

    @property
    def x_start_m(self) -> float:
        return -self.hull_entry_offset_m

    @property
    def transit_time_s(self) -> float:
        length, _ = self.domain_m
        travel = (length + 2.0 * self.floe_radius_m + 0.5 * self.hull_length_m
                  - self.x_start_m)
        return travel / self.speed_m_s

    @property
    def steady_window_s(self) -> Tuple[float, float]:
        """Window where the bow is >= 3R inside the field and the stern <= L-3R."""
        length, _ = self.domain_m
        radius = self.floe_radius_m
        half_length = 0.5 * self.hull_length_m
        t_a = (3.0 * radius - half_length - self.x_start_m) / self.speed_m_s
        t_b = (length - 3.0 * radius + half_length - self.x_start_m) / self.speed_m_s
        return (t_a, t_b)

    # -- contact / time step ---------------------------------------------
    @property
    def modulus_ratio(self) -> float:
        return self.modulus_pa / self.strength_pa

    @property
    def reference_penetration_fraction(self) -> float:
        return modulus_fraction(self.modulus_ratio)

    @property
    def contact_stiffness_n_m(self) -> float:
        return 2.0 * self.strength_pa * self.thickness_m / math.sqrt(
            self.reference_penetration_fraction)

    @property
    def dt(self) -> float:
        """Contact-CFL step ``dt_scale * 0.5 sqrt(m_min / k_n)`` [s].

        Under Cauchy scaling ``m ~ s^3`` and ``k_n ~ s^2`` so ``dt ~ s^(1/2)``:
        the step follows the Froude time scale automatically, which is the point
        -- a fixed absolute ``dt`` is exactly the confound the resolution test
        looks for.
        """
        radius = 0.85 * self.floe_radius_m
        mass = ICE_DENSITY * math.pi * radius * radius * self.thickness_m
        return self.dt_scale * 0.5 * math.sqrt(mass / self.contact_stiffness_n_m)

    # -- dimensionless space ---------------------------------------------
    def groups(self) -> Dict[str, float]:
        return nondimensional_groups(
            speed_m_s=self.speed_m_s, hull_length_m=self.hull_length_m,
            strength_pa=self.strength_pa, thickness_m=self.thickness_m,
            modulus_pa=self.modulus_pa, concentration=self.concentration,
            friction=self.friction, beam_m=self.beam_m)

    def id(self) -> str:
        if self.label:
            return self.label
        return (f"{self.protocol}_lam{self.lam:g}_q{self.modulus_factor:.4g}"
                f"_dt{self.dt_scale:g}_j{self.radius_jitter:g}_s{self.seed}")

    def as_dict(self) -> Dict[str, Any]:
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload.update({
            "floe_radius_m": self.floe_radius_m,
            "hull_length_m": self.hull_length_m,
            "domain_m": list(self.domain_m),
            "transit_time_s": self.transit_time_s,
            "dt_s": self.dt,
            "modulus_ratio": self.modulus_ratio,
            "reference_penetration_fraction": self.reference_penetration_fraction,
            "contact_stiffness_n_m": self.contact_stiffness_n_m,
            "groups": self.groups(),
            "id": self.id(),
        })
        return payload


def build_case(protocol: str = PROTOCOL_CAUCHY, lam: float = 10.0, *,
               modulus_factor: Optional[float] = None,
               modulus_exponent: float = 0.5,
               prototype: Optional[Dict[str, float]] = None,
               **overrides: Any) -> ScaleCase:
    """Map the prototype scenario onto the model at scale ``1:lam``.

    ``lengths -> /lam``, ``speed -> /sqrt(lam)``, ``strength -> /lam`` (the
    ``Sigma`` group), modulus by protocol:

    ``P-Cauchy``            ``E/lam``              (``q = 1``)
    ``P-Froude+strength``   ``E/lam^p_E``          (``q = lam^(1-p_E)``)
    ``P-mismatch``          ``(E/lam) * q``        (``q = 1 + dMod``)
    """
    proto = dict(PROTOTYPE if prototype is None else prototype)
    s = 1.0 / float(lam)
    strength = proto["strength_pa"] * s
    cauchy_modulus = proto["modulus_pa"] * s
    if protocol in (PROTOCOL_CAUCHY, PROTOCOL_MISMATCH):
        modulus = cauchy_modulus * (1.0 if modulus_factor is None else float(modulus_factor))
    elif protocol == PROTOCOL_FROUDE:
        modulus = proto["modulus_pa"] * s ** float(modulus_exponent)
        if modulus_factor is not None:
            modulus *= float(modulus_factor)
    else:  # pragma: no cover - guarded by ScaleCase
        raise ValueError(f"unknown protocol {protocol!r}")

    kwargs: Dict[str, Any] = dict(
        lam=float(lam), protocol=protocol,
        modulus_factor=float(modulus / cauchy_modulus),
        modulus_exponent=float(modulus_exponent),
        beam_m=proto["beam_m"] * s,
        thickness_m=proto["thickness_m"] * s,
        speed_m_s=proto["speed_m_s"] * math.sqrt(s),
        concentration=proto["concentration"],
        strength_pa=strength,
        friction=proto["friction"],
        modulus_pa=modulus,
    )
    kwargs.update(overrides)
    return ScaleCase(**kwargs)


# ---------------------------------------------------------------------------
# DEM record
# ---------------------------------------------------------------------------

def build_simulation(case: ScaleCase) -> Tuple[IceChannelSimulation, HullGeometry]:
    """Floe field + prescribed hull for ``case`` (no stepping)."""
    length, width = case.domain_m
    radius = case.floe_radius_m
    hull = HullGeometry(length=case.hull_length_m, beam=case.beam_m,
                        speed=case.speed_m_s, y_position=0.5 * width,
                        x_start=case.x_start_m)
    water = WaterEnvironment(ice_density=ICE_DENSITY, water_density=WATER_DENSITY,
                             gravity=GRAVITY, wind_velocity=(0.0, 0.0),
                             current_velocity=(0.0, 0.0))
    contact = ContactParameters(
        crushing_strength=case.strength_pa, friction=case.friction,
        reference_penetration_fraction=case.reference_penetration_fraction)
    sim = IceChannelSimulation.from_field(
        case.concentration, domain_length=length, domain_width=width,
        mean_radius=radius,
        radius_range=(0.85 * radius, 1.15 * radius),
        thickness_range=(case.thickness_m, case.thickness_m),
        radius_jitter=case.radius_jitter, seed=case.seed, dt=case.dt, hull=hull,
        water=water, contact=contact, integrator="euler")
    return sim, hull


def simulate_scale_case(case: ScaleCase) -> Dict[str, Any]:
    """Run one cascade member and return its dimensionless resistance record.

    Measured force: ``R = mean(-F_hull,x)`` over the steady transit window;
    dimensionless result ``R* = R / (sigma_f t^2)``.  Also returned: the
    floe-level contact census (floe-floe contacts per floe + hull contacts per
    floe), the realised field statistics and the wall clock.
    """
    started = time.time()
    sim, hull = build_simulation(case)
    dt = sim.dt
    n_steps = max(1, int(round(case.transit_time_s / dt)))
    t0, t_end = case.steady_window_s
    n = sim.n

    series_r: List[float] = []
    floe_cpf: List[float] = []
    hull_cpf: List[float] = []
    for step in range(n_steps):
        force, _torque = sim._hull_force(sim.time, sim.pos, hull=hull)
        if t0 <= sim.time <= t_end:
            series_r.append(float(force[:, 0].sum()))
            if step % case.contact_sample_every == 0:
                idx, *_ = sim._floe_floe_contacts()
                floe_cpf.append(2.0 * float(idx.shape[0]) / n)
                hull_cpf.append(float((force.norm(dim=1) > 0.0).sum()) / n)
        sim.step()

    if not series_r:
        raise RuntimeError("steady window contained no samples")
    series = np.asarray(series_r, dtype=float)
    r_mean = float(series.mean())
    r_sem = (float(series.std(ddof=1) / math.sqrt(series.size))
             if series.size > 1 else float("nan"))
    field_stats = sim.field_statistics()
    sigma_t2 = case.strength_pa * case.thickness_m ** 2
    fcp = float(np.mean(floe_cpf)) if floe_cpf else float("nan")
    hcp = float(np.mean(hull_cpf)) if hull_cpf else float("nan")
    return {
        "case": case.as_dict(),
        "id": case.id(),
        "protocol": case.protocol,
        "lam": case.lam,
        "modulus_factor": case.modulus_factor,
        "groups": case.groups(),
        "R_mean_N": r_mean,
        "R_sem_N": r_sem,
        "R_rel_sem": r_sem / abs(r_mean) if r_mean else float("nan"),
        "R_star": r_mean / sigma_t2,
        "R_star_buoyancy": r_mean / (ICE_DENSITY * GRAVITY * case.thickness_m ** 2
                                     * case.beam_m),
        "contacts_per_floe": fcp + hcp,
        "floe_contacts_per_floe": fcp,
        "hull_contacts_per_floe": hcp,
        "contact_samples": len(floe_cpf),
        "floe_count": int(n),
        "realised_concentration": float(field_stats["realised_concentration"]),
        "max_initial_overlap_m": float(field_stats.get("max_initial_overlap", 0.0)),
        "cleared_width_over_beam": float(sim.cleared_width()) / case.beam_m,
        "dt_s": dt,
        "steps": n_steps,
        "samples": int(series.size),
        "simulated_time_s": float(sim.time),
        "wall_time_s": time.time() - started,
        "state_digest": sim.state_digest(),
    }


# ---------------------------------------------------------------------------
# collapse / deviation analysis
# ---------------------------------------------------------------------------

def collapse_deviation(r_star: float, reference: float) -> Dict[str, float]:
    """Relative departure of one cascade member from the reference ``R*``."""
    if reference == 0.0:
        raise ValueError("reference R* is zero; collapse deviation undefined")
    delta = (float(r_star) - float(reference)) / float(reference)
    return {"R_star": float(r_star), "R_star_ref": float(reference),
            "delta": delta, "delta_pct": 100.0 * delta}


def loglog_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Least squares ``ln y = p ln x + ln A`` with ``R^2`` and slope s.e."""
    xa = np.asarray([float(v) for v in x], dtype=float)
    ya = np.asarray([float(v) for v in y], dtype=float)
    if xa.size != ya.size or xa.size < 3:
        raise ValueError("loglog_fit needs >= 3 paired samples")
    if np.any(xa <= 0.0) or np.any(ya <= 0.0):
        raise ValueError("loglog_fit needs strictly positive samples")
    lx, ly = np.log(xa), np.log(ya)
    p, c = np.polyfit(lx, ly, 1)
    pred = p * lx + c
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    dof = max(1, xa.size - 2)
    sxx = float(((lx - lx.mean()) ** 2).sum())
    slope_se = math.sqrt((ss_res / dof) / sxx) if sxx > 0.0 else float("inf")
    return {"exponent": float(p), "prefactor": float(math.exp(c)), "r2": float(r2),
            "slope_se": float(slope_se), "n": int(xa.size),
            "ss_res": ss_res, "ss_tot": ss_tot}


def linear_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Least squares ``y = a x + b`` with ``R^2``."""
    xa = np.asarray([float(v) for v in x], dtype=float)
    ya = np.asarray([float(v) for v in y], dtype=float)
    if xa.size != ya.size or xa.size < 2:
        raise ValueError("linear_fit needs >= 2 paired samples")
    a, b = np.polyfit(xa, ya, 1)
    pred = a * xa + b
    ss_res = float(((ya - pred) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"slope": float(a), "intercept": float(b), "r2": float(r2),
            "n": int(xa.size), "ss_res": ss_res, "ss_tot": ss_tot}


#: Deviation-law form family (predeclared; selected by residual, no free choice
#: after the data are seen).
DEVIATION_FORMS: Tuple[str, ...] = ("linear", "quadratic", "power")


def _r2(y: Sequence[float], pred: Sequence[float]) -> float:
    ya = np.asarray(y, dtype=float)
    pa = np.asarray(pred, dtype=float)
    ss_res = float(((ya - pa) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def fit_deviation_law(points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit ``delta`` against the ``Mod`` mismatch ``dMod = q - 1``.

    ``points``: dicts with keys ``dMod`` and ``delta_pct``.  Three predeclared
    forms are fitted; the one with the smallest residual sum of squares is
    returned together with all three, so the model selection is auditable.
    All forms pass through ``delta(0) = 0`` (no mismatch -> no deviation).
    """
    pts = [p for p in points if math.isfinite(float(p["dMod"]))
           and math.isfinite(float(p["delta_pct"]))]
    if len(pts) < 3:
        raise ValueError("fit_deviation_law needs >= 3 finite points")
    X = np.asarray([float(p["dMod"]) for p in pts], dtype=float)
    y = np.asarray([float(p["delta_pct"]) for p in pts], dtype=float)
    fits: Dict[str, Dict[str, Any]] = {}

    a = float((X * y).sum() / (X ** 2).sum()) if float((X ** 2).sum()) > 0.0 else 0.0
    pred = a * X
    fits["linear"] = {"a": a, "ss_res": float(((y - pred) ** 2).sum()),
                      "r2": _r2(y, pred)}

    design = np.vstack([X, X ** 2]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    fits["quadratic"] = {"a": float(coef[0]), "b": float(coef[1]),
                         "ss_res": float(((y - pred) ** 2).sum()),
                         "r2": _r2(y, pred)}

    if np.all(X != 0.0) and float(np.abs(y).min()) > 0.0:
        pl = loglog_fit(np.abs(X).tolist(), np.abs(y).tolist())
        pred = pl["prefactor"] * np.abs(X) ** pl["exponent"]
        fits["power"] = {"A": pl["prefactor"], "p": pl["exponent"],
                         "ss_res": float(((np.abs(y) - pred) ** 2).sum()),
                         "r2": _r2(np.abs(y), pred)}

    order = [k for k in DEVIATION_FORMS if k in fits]
    best = min(order, key=lambda k: (fits[k]["ss_res"], order.index(k)))
    return {"form": best, "fits": fits, "n_points": len(pts),
            "points": [{"dMod": float(d), "delta_pct": float(v)}
                       for d, v in zip(X, y)]}


def exact_covariance_pairs(records: Sequence[Dict[str, Any]], *,
                           rtol: float = 1e-12) -> List[Dict[str, Any]]:
    """Pairs of cases that must agree *bit for bit* if the model is covariant.

    Two cascade members at ``lambda_2 / lambda_1 = 4**k`` with the same seed and
    the same remaining knobs differ by a power of four in every length *and* by
    a power of two in the Froude velocity ``v ~ length^(1/2)``; both are exact in
    binary floating point, so the DEM state -- and therefore ``R*`` -- has to be
    identical to the last bit.  A ratio of only ``2**k`` with odd ``k`` is *not*
    exact (the square root of the velocity then introduces a rounding), so those
    pairs are excluded.  Any remaining difference in an included pair is a
    rounding leak (an absolute tolerance, a fixed step, a clamped index) -- the
    confound a two-scale experiment has to rule out.
    """
    out: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(records, 2):
        ca, cb = a["case"], b["case"]
        if ca["protocol"] != cb["protocol"]:
            continue
        if int(ca["seed"]) != int(cb["seed"]):
            continue
        if (abs(float(ca["dt_scale"]) - float(cb["dt_scale"])) > 0.0
                or abs(float(ca["radius_jitter"]) - float(cb["radius_jitter"])) > 0.0):
            continue
        if abs(float(ca["modulus_factor"]) - float(cb["modulus_factor"])) > 1e-12:
            continue
        ratio = float(cb["lam"]) / float(ca["lam"])
        if ratio <= 1.0:
            continue
        exponent = math.log2(ratio)
        if abs(exponent - round(exponent)) > 1e-12:
            continue
        if int(round(exponent)) % 2 != 0:      # only 4**k keeps sqrt exact
            continue
        rel = abs(float(b["R_star"]) - float(a["R_star"])) / abs(float(a["R_star"]))
        out.append({"a": a["id"], "b": b["id"], "lam_ratio": ratio,
                    "power_of_two_exponent": int(round(exponent)),
                    "R_star_a": float(a["R_star"]), "R_star_b": float(b["R_star"]),
                    "rel_diff": rel, "exact": bool(rel <= rtol)})
    out.sort(key=lambda d: (d["lam_ratio"], d["a"]))
    return out


def collapse_statistics(values: Sequence[float]) -> Dict[str, float]:
    """Spread of a Cauchy group in dimensionless space."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        raise ValueError("empty sample")
    mean = float(arr.mean())
    spread = float((arr.max() - arr.min()) / abs(mean)) if mean else float("nan")
    return {"n": int(arr.size), "mean": mean,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()),
            "max_rel_departure": spread}


def convergence_check(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolution convergence of ``R*`` over ``dt_scale`` levels.

    ``rel_change_finest`` is the change between the two finest levels -- the
    quantity compared with the ``|dR|/R < 3%`` criterion -- and
    ``rel_change_coarsest`` the change between the two coarsest ones.
    """
    if len(records) < 3:
        raise ValueError("convergence_check needs >= 3 refinement levels")
    ordered = sorted(records, key=lambda r: float(r["case"]["dt_scale"]))
    r_star = [float(r["R_star"]) for r in ordered]
    levels = [{"dt_scale": float(r["case"]["dt_scale"]), "dt_s": float(r["dt_s"]),
               "steps": int(r["steps"]), "R_star": r_star[i],
               "wall_time_s": float(r["wall_time_s"])}
              for i, r in enumerate(ordered)]
    rel = [abs(r_star[i + 1] - r_star[i]) / abs(r_star[i + 1])
           for i in range(len(r_star) - 1)]
    # levels ascend in dt_scale, so rel[0] is the change between the two
    # *finest* steps (the one the convergence criterion is stated on) and
    # rel[-1] the change between the two coarsest.
    return {"levels": levels, "rel_change_finest": rel[0],
            "rel_change_coarsest": rel[-1], "rel_changes": rel}


def contact_sensitivity_pairs(base: Dict[str, Any],
                              variants: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Deviation response to a *numerical* contact-count perturbation.

    Each variant is the same physical case with a numerical knob (initial-field
    jitter, seed, time step) moved, so the change in ``contacts_per_floe`` is a
    discretisation artefact.  The pair records both the contact-count change and
    the resulting change of ``delta`` (percentage points).
    """
    pairs = []
    d_base = float(base["delta_pct"])
    for v in variants:
        c_change = ((float(v["contacts_per_floe"]) - float(base["contacts_per_floe"]))
                    / float(base["contacts_per_floe"]) * 100.0)
        pairs.append({
            "variant": v["id"],
            "knob": v.get("knob", ""),
            "contacts_per_floe": float(v["contacts_per_floe"]),
            "contact_change_pct": c_change,
            "delta_pct": float(v["delta_pct"]),
            "delta_change_pct": float(v["delta_pct"]) - d_base,
        })
    return {"base": {"id": base["id"],
                     "contacts_per_floe": float(base["contacts_per_floe"]),
                     "delta_pct": d_base},
            "pairs": pairs}


def evaluate_criteria(*, cauchy: Sequence[Dict[str, Any]],
                      froude: Sequence[Dict[str, Any]],
                      mismatch: Sequence[Dict[str, Any]],
                      convergence: Dict[str, Any],
                      contact_sensitivity: Dict[str, Any],
                      criterion_a_pct: float = 15.0,
                      criterion_b_r2: float = 0.90,
                      criterion_c_contact_pct: float = 10.0,
                      criterion_c_delta_pct: float = 5.0,
                      criterion_d_pct: float = 3.0) -> Dict[str, Any]:
    """Machine-checkable verdicts (a)-(d) of the campaign node."""
    a_vals = [abs(float(r["delta_pct"])) for r in cauchy]
    a_max = max(a_vals) if a_vals else float("nan")
    a_pass = bool(a_vals and a_max < criterion_a_pct)

    lam = [float(r["lam"]) for r in froude]
    dlt = [abs(float(r["delta_pct"])) for r in froude]
    b_fit = (loglog_fit(lam, dlt)
             if (len(lam) >= 3 and all(v > 0.0 for v in dlt)) else None)
    b_pass = bool(b_fit is not None and b_fit["r2"] > criterion_b_r2)

    pairs = list(contact_sensitivity.get("pairs", []))
    viol = [p for p in pairs
            if abs(float(p["contact_change_pct"])) < criterion_c_contact_pct
            and abs(float(p["delta_change_pct"])) >= criterion_c_delta_pct]
    c_pass = bool(pairs and not viol)

    d_val = float(convergence.get("rel_change_finest", float("nan")))
    d_pass = bool(math.isfinite(d_val) and d_val * 100.0 < criterion_d_pct)

    law_points = [
        p if "dMod" in p else {"dMod": float(p["case"]["modulus_factor"]) - 1.0,
                               "delta_pct": float(p["delta_pct"])}
        for p in mismatch]
    law = fit_deviation_law(law_points) if len(law_points) >= 3 else None
    return {
        "a": {"criterion": f"P-Cauchy max|delta| < {criterion_a_pct:g}%",
              "pass": a_pass, "max_abs_delta_pct": a_max,
              "per_case": [{"id": r["id"], "lam": r["lam"],
                            "delta_pct": float(r["delta_pct"])} for r in cauchy]},
        "b": {"criterion": (f"P-Froude+strength |delta| ~ lam power fit "
                            f"R^2 > {criterion_b_r2:g}"),
              "pass": b_pass, "fit": b_fit,
              "points": [{"lam": l, "abs_delta_pct": d} for l, d in zip(lam, dlt)]},
        "c": {"criterion": (f"contacts-per-floe change < {criterion_c_contact_pct:g}% "
                            f"-> |delta| change < {criterion_c_delta_pct:g}%"),
              "pass": c_pass, "violations": viol, "pairs": pairs,
              "max_abs_delta_change_pct": max(
                  [abs(float(p["delta_change_pct"])) for p in pairs],
                  default=float("nan"))},
        "d": {"criterion": f"resolution convergence |dR|/R < {criterion_d_pct:g}%",
              "pass": d_pass, "rel_change_finest": d_val,
              "levels": convergence.get("levels", [])},
        "deviation_law": law,
        "all_pass": bool(a_pass and b_pass and c_pass and d_pass),
    }


def group_table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flat ``(id, lam, protocol, groups...)`` table for reports."""
    keys = ("Fn", "Sigma", "Mod", "t_over_L", "phi", "mu")
    rows = []
    for r in records:
        g = r["groups"]
        rows.append({"id": r["id"], "lam": r["lam"], "protocol": r["protocol"],
                     "q": r["modulus_factor"],
                     **{k: float(g[k]) for k in keys},
                     "R_star": float(r["R_star"]),
                     "contacts_per_floe": float(r["contacts_per_floe"])})
    return rows
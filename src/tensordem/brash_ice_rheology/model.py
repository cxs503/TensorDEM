"""Three-component brash-ice channel resistance with a mu(I) shear closure.

Components
----------
1. Granular shear of the brash against the hull (*the mu(I) part*).  The hull
   shears the brash in a band of thickness ``delta = c_delta d``, so the shear
   rate is ``gamma_dot = v/delta`` and

   .. math:: F_{shear} = \\mu(I)\\,P\\,A_{shear}\\,(1-\\phi)\\,f(v),\\quad
             A_{shear} = L_{wl}\\,h_{sub}

   (``(1-phi)`` = solid contact fraction: pure water carries no shear).  The
   confinement pressure is the sum of a static buoyant part and the *dynamic
   compaction* of the brash flowing into the hull:

   .. math:: P = (\\rho_w-\\rho_b) g h_{sub} + \\tfrac12 C_c \\rho_b v^2 .

   The dynamic term is what makes ``P`` - and therefore ``I`` - a function of
   the transit speed; at the standard case it contributes ~72 % of ``P``.

2. Buoyancy displacement, ``F_buoy = rho_w g V_displaced`` with
   ``V_displaced = (1-phi) B h_sub``: only the *solid* ice volume swept per
   metre of advance displaces water (the voids are already water), so the
   Lindqvist level-ice convention ``0.9 B H^2`` is replaced by the brash
   version that carries the correct ``phi -> 1`` and ``h_sub -> 0`` limits.

3. Underwater keel drag, ``F_drag = 0.5 C_D rho_w v^2 A_keel`` with
   ``A_keel = (1-phi) B h_sub``: the submerged brash mass under the hull is a
   bluff, porous, interlocked body and drags a large wake.

All three carry the processing closure ``f(v) = v/(v+v_0)``: the resistance
reported here is the *processing* resistance of a ship under way, which
vanishes with the advance (``R(0) = 0``); the static break-out force of a stuck
ship is deliberately outside this model (see ``docs``).

Porosity limits are structural, not tuned: ``phi -> 1`` gives
``rho_b -> rho_w`` (no buoyancy deficit, ``F_buoy -> 0``), ``1-phi -> 0``
(no ice-ice contacts, ``F_shear -> 0``) and ``A_keel -> 0`` (no obstruction,
``F_drag -> 0``), so ``R -> 0`` in pure water.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

__all__ = [
    "BrashHull",
    "Closure",
    "ResistanceComponents",
    "STANDARD_CASE",
    "STANDARD_BEAMS_M",
    "STANDARD_HULL_TABLE",
    "standard_hull",
    "shear_component",
    "buoyancy_component",
    "keel_drag_component",
    "brash_resistance",
    "speed_sweep",
    "porosity_sweep",
    "power_law_exponent",
]

from .coefficients import CONFINEMENT_COEFFICIENT, check_coefficient, ledger_sha256
from .material import STANDARD_BRASH, BrashIceState
from .mu_i import JopParameters, inertial_number


@dataclass(frozen=True)
class BrashHull:
    """Ship form seen by the brash layer (SI units).

    ``waterline_length_m`` defaults to the overall length (bow-to-stern contact
    with the brash at the waterline); ``bow_length_m = (B/2)/tan(entry)`` is the
    length over which the brash is engaged.
    """

    length_m: float = 180.0
    beam_m: float = 25.0
    draught_m: float = 7.0
    entry_angle_deg: float = 25.0

    def __post_init__(self) -> None:
        if not self.length_m > 0.0:
            raise ValueError("length_m must be positive")
        if not self.beam_m > 0.0:
            raise ValueError("beam_m must be positive")
        if not 0.0 < self.entry_angle_deg < 90.0:
            raise ValueError("entry_angle_deg must be in (0, 90)")

    @property
    def bow_length_m(self) -> float:
        return 0.5 * self.beam_m / math.tan(math.radians(self.entry_angle_deg))

    @property
    def waterline_length_m(self) -> float:
        return self.length_m

    @property
    def length_to_beam(self) -> float:
        return self.length_m / self.beam_m

    def as_dict(self) -> Dict[str, float]:
        return {
            "length_m": self.length_m, "beam_m": self.beam_m,
            "draught_m": self.draught_m, "entry_angle_deg": self.entry_angle_deg,
            "bow_length_m": self.bow_length_m,
            "waterline_length_m": self.waterline_length_m,
            "length_to_beam": self.length_to_beam,
        }


#: Standard campaign beam series: 15 / 25 / 35 m with realistic
#: length-to-beam (7.3 / 7.2 / 7.1) and draught.
STANDARD_HULL_TABLE: Dict[float, Tuple[float, float]] = {
    15.0: (110.0, 5.0),
    25.0: (180.0, 7.0),
    35.0: (250.0, 9.0),
}
STANDARD_BEAMS_M: Tuple[float, ...] = (15.0, 25.0, 35.0)


def standard_hull(beam_m: float) -> BrashHull:
    """Hull of the standard beam series (linear ``L = 7.2 B`` outside the table)."""
    if beam_m in STANDARD_HULL_TABLE:
        length, draught = STANDARD_HULL_TABLE[beam_m]
    else:
        length, draught = 7.2 * beam_m, 0.32 * beam_m
    return BrashHull(length_m=length, beam_m=beam_m, draught_m=draught)


@dataclass(frozen=True)
class Closure:
    """The model's closure constants (defaults = graded ledger values)."""

    mu_s: float = 0.36
    mu_2: float = 0.84
    i_0: float = 0.30
    shear_band_coefficient: float = 2.0
    confinement_coefficient: float = CONFINEMENT_COEFFICIENT
    displaced_ice_volume_factor: float = 1.0
    keel_drag_coefficient: float = 2.2
    mobilization_speed_m_s: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.mu_s < self.mu_2:
            raise ValueError("need 0 < mu_s < mu_2")
        if not self.i_0 > 0.0:
            raise ValueError("i_0 must be positive")
        if not self.shear_band_coefficient > 0.0:
            raise ValueError("shear_band_coefficient must be positive")
        if self.confinement_coefficient < 0.0:
            raise ValueError("confinement_coefficient must be non-negative")
        if not 0.0 < self.displaced_ice_volume_factor:
            raise ValueError("displaced_ice_volume_factor must be positive")
        if self.keel_drag_coefficient < 0.0:
            raise ValueError("keel_drag_coefficient must be non-negative")
        if self.mobilization_speed_m_s < 0.0:
            raise ValueError("mobilization_speed_m_s must be non-negative")

    @classmethod
    def from_ledger(cls) -> "Closure":
        """Closure taken verbatim from the graded coefficient ledger."""
        return cls(
            mu_s=check_coefficient("mu_s")[0],
            mu_2=check_coefficient("mu_2")[0],
            i_0=check_coefficient("i_0")[0],
            shear_band_coefficient=check_coefficient("shear_band_coefficient")[0],
            confinement_coefficient=check_coefficient("confinement_coefficient")[0],
            displaced_ice_volume_factor=check_coefficient(
                "displaced_ice_volume_factor")[0],
            keel_drag_coefficient=check_coefficient("keel_drag_coefficient")[0],
            mobilization_speed_m_s=check_coefficient("mobilization_speed_m_s")[0],
        )

    def with_confinement(self, value: float) -> "Closure":
        """Copy with a different confinement coefficient (calibration/sweeps)."""
        return Closure(mu_s=self.mu_s, mu_2=self.mu_2, i_0=self.i_0,
                       shear_band_coefficient=self.shear_band_coefficient,
                       confinement_coefficient=float(value),
                       displaced_ice_volume_factor=self.displaced_ice_volume_factor,
                       keel_drag_coefficient=self.keel_drag_coefficient,
                       mobilization_speed_m_s=self.mobilization_speed_m_s)

    @property
    def jop(self) -> JopParameters:
        return JopParameters(mu_s=self.mu_s, mu_2=self.mu_2, i_0=self.i_0)

    def as_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ResistanceComponents:
    """Decomposed brash-channel resistance (N) plus the state that produced it."""

    F_shear_N: float
    F_buoy_N: float
    F_drag_N: float
    total_N: float
    inertial_number: float
    mu_effective: float
    confinement_pa: float
    static_confinement_pa: float
    dynamic_confinement_pa: float
    shear_rate_1_s: float
    shear_stress_pa: float
    shear_area_m2: float
    displacement_volume_m3: float
    keel_area_m2: float
    mobilization: float
    speed_m_s: float
    state: BrashIceState
    hull: BrashHull
    ledger_sha256: str = field(default_factory=ledger_sha256)

    # -- convenience --------------------------------------------------------
    @property
    def total_kN(self) -> float:
        return self.total_N / 1.0e3

    @property
    def share_shear(self) -> float:
        return self.F_shear_N / self.total_N if self.total_N else 0.0

    @property
    def share_buoyancy(self) -> float:
        return self.F_buoy_N / self.total_N if self.total_N else 0.0

    @property
    def share_drag(self) -> float:
        return self.F_drag_N / self.total_N if self.total_N else 0.0

    @property
    def shares(self) -> Dict[str, float]:
        return {"shear": self.share_shear, "buoyancy": self.share_buoyancy,
                "drag": self.share_drag}

    def as_dict(self) -> Dict[str, object]:
        return {
            "F_shear_N": self.F_shear_N,
            "F_buoy_N": self.F_buoy_N,
            "F_drag_N": self.F_drag_N,
            "total_N": self.total_N,
            "total_kN": self.total_kN,
            "shares": self.shares,
            "inertial_number": self.inertial_number,
            "mu_effective": self.mu_effective,
            "confinement_pa": self.confinement_pa,
            "static_confinement_pa": self.static_confinement_pa,
            "dynamic_confinement_pa": self.dynamic_confinement_pa,
            "shear_rate_1_s": self.shear_rate_1_s,
            "shear_stress_pa": self.shear_stress_pa,
            "shear_area_m2": self.shear_area_m2,
            "displacement_volume_m3": self.displacement_volume_m3,
            "keel_area_m2": self.keel_area_m2,
            "mobilization": self.mobilization,
            "speed_m_s": self.speed_m_s,
            "state": self.state.as_dict(),
            "hull": self.hull.as_dict(),
            "ledger_sha256": self.ledger_sha256,
        }


#: Campaign standard case (H = 1 m, d = 0.5 m, phi = 0.3, v = 3 m/s).
STANDARD_CASE: Dict[str, float] = {"layer_thickness_m": 1.0, "floe_diameter_m": 0.5,
                                    "porosity": 0.3, "speed_m_s": 3.0}


def _check_speed(speed: float) -> float:
    speed = float(speed)
    if speed < 0.0:
        raise ValueError("speed must be non-negative")
    return speed


def _mobilization(speed: float, closure: Closure) -> float:
    """Processing closure ``f = v/(v+v_0)`` (``f(0) = 0`` exactly)."""
    if speed <= 0.0:
        return 0.0
    v0 = closure.mobilization_speed_m_s
    if v0 == 0.0:
        return 1.0
    return speed / (speed + v0)


def _confinement(state: BrashIceState, speed: float, closure: Closure
                 ) -> Tuple[float, float, float]:
    """``(P, p_static, p_dyn)`` in Pa."""
    p_static = state.static_confinement_pa
    p_dyn = 0.5 * closure.confinement_coefficient * state.bulk_density_kg_m3 * speed ** 2
    return p_static + p_dyn, p_static, p_dyn


def shear_component(state: BrashIceState, hull: BrashHull, speed: float,
                    closure: Closure = None) -> Dict[str, float]:
    """Granular (mu(I)) shear of the brash against the hull, in N."""
    closure = closure or Closure.from_ledger()
    speed = _check_speed(speed)
    P, p_static, p_dyn = _confinement(state, speed, closure)
    delta = closure.shear_band_coefficient * state.floe_diameter_m
    shear_rate = speed / delta
    inertial = inertial_number(shear_rate, state.floe_diameter_m, P,
                               state.bulk_density_kg_m3)
    mu = closure.jop.mu(inertial)
    tau = mu * P * state.solid_fraction
    area = hull.waterline_length_m * state.submerged_thickness_m
    f = _mobilization(speed, closure)
    return {
        "F_shear_N": tau * area * f,
        "inertial_number": inertial,
        "mu_effective": mu,
        "confinement_pa": P,
        "static_confinement_pa": p_static,
        "dynamic_confinement_pa": p_dyn,
        "shear_rate_1_s": shear_rate,
        "shear_stress_pa": tau,
        "shear_area_m2": area,
        "mobilization": f,
    }


def buoyancy_component(state: BrashIceState, hull: BrashHull, speed: float,
                       closure: Closure = None) -> Dict[str, float]:
    """Buoyancy displacement resistance ``rho_w g V_displaced``, in N."""
    closure = closure or Closure.from_ledger()
    speed = _check_speed(speed)
    B = hull.beam_m
    volume = (closure.displaced_ice_volume_factor * state.solid_fraction
              * B * state.submerged_thickness_m)
    f = _mobilization(speed, closure)
    return {
        "F_buoy_N": state.water_density_kg_m3 * state.gravity_m_s2 * volume * f,
        "displacement_volume_m3": volume,
        "mobilization": f,
    }


def keel_drag_component(state: BrashIceState, hull: BrashHull, speed: float,
                        closure: Closure = None) -> Dict[str, float]:
    """Underwater keel form drag ``0.5 C_D rho_w v^2 A_keel``, in N."""
    closure = closure or Closure.from_ledger()
    speed = _check_speed(speed)
    area = state.solid_fraction * hull.beam_m * state.submerged_thickness_m
    f = _mobilization(speed, closure)
    F = 0.5 * closure.keel_drag_coefficient * state.water_density_kg_m3 * speed ** 2 * area
    return {"F_drag_N": F * f, "keel_area_m2": area, "mobilization": f}


def brash_resistance(state: BrashIceState = STANDARD_BRASH,
                     hull: BrashHull = None,
                     speed: float = STANDARD_CASE["speed_m_s"],
                     closure: Closure = None) -> ResistanceComponents:
    """Total brash-channel resistance and its three-component decomposition."""
    closure = closure or Closure.from_ledger()
    hull = hull or standard_hull(25.0)
    speed = _check_speed(speed)
    shear = shear_component(state, hull, speed, closure)
    buoy = buoyancy_component(state, hull, speed, closure)
    drag = keel_drag_component(state, hull, speed, closure)
    total = shear["F_shear_N"] + buoy["F_buoy_N"] + drag["F_drag_N"]
    return ResistanceComponents(
        F_shear_N=shear["F_shear_N"], F_buoy_N=buoy["F_buoy_N"],
        F_drag_N=drag["F_drag_N"], total_N=total,
        inertial_number=shear["inertial_number"],
        mu_effective=shear["mu_effective"],
        confinement_pa=shear["confinement_pa"],
        static_confinement_pa=shear["static_confinement_pa"],
        dynamic_confinement_pa=shear["dynamic_confinement_pa"],
        shear_rate_1_s=shear["shear_rate_1_s"],
        shear_stress_pa=shear["shear_stress_pa"],
        shear_area_m2=shear["shear_area_m2"],
        displacement_volume_m3=buoy["displacement_volume_m3"],
        keel_area_m2=drag["keel_area_m2"],
        mobilization=shear["mobilization"],
        speed_m_s=speed, state=state, hull=hull,
    )


def speed_sweep(state: BrashIceState = STANDARD_BRASH, hull: BrashHull = None,
                speeds: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0),
                closure: Closure = None) -> List[Dict[str, float]]:
    """Resistance (and components) over a speed sweep."""
    hull = hull or standard_hull(25.0)
    return [{**brash_resistance(state, hull, v, closure).as_dict(),
             "speed_m_s": float(v)} for v in speeds]


def porosity_sweep(state: BrashIceState = STANDARD_BRASH, hull: BrashHull = None,
                   speed: float = STANDARD_CASE["speed_m_s"],
                   porosities: Sequence[float] = (0.2, 0.25, 0.3, 0.35, 0.4),
                   closure: Closure = None) -> List[Dict[str, float]]:
    """Resistance over a porosity sweep (state bulk density follows phi)."""
    hull = hull or standard_hull(25.0)
    return [brash_resistance(state.with_porosity(p), hull, speed,
                             closure).as_dict() for p in porosities]


def power_law_exponent(speeds: Sequence[float], totals: Sequence[float]) -> float:
    """Least-squares exponent ``n`` of ``R ~ v^n`` in log-log space."""
    pairs = [(v, r) for v, r in zip(speeds, totals) if v > 0.0 and r > 0.0]
    if len(pairs) < 2:
        raise ValueError("need at least 2 positive (speed, resistance) pairs")
    xs = [math.log(v) for v, _ in pairs]
    ys = [math.log(r) for _, r in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx if sxx else float("nan")

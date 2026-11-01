"""DEM-FSM coupled ice loads on rudder blades and steering gear.

Scope of this module (increment W5-N01)
--------------------------------------
1. ``IceStrengthTemperatureModel`` -- temperature dependent ice compressive
   strength ``sigma_ice(T) = sigma_0 * exp(-alpha * (T - T_ref))`` with
   least-squares calibration against literature anchors.
2. ``IACSF12DesignLoad`` -- a rule-style design ice load card for rudder
   blades and steering gear (parametric sweep over ice thickness / speed /
   design temperature).
3. ``RudderIceLoadDEM`` -- a bonded-sphere ice field (inherits the existing
   :class:`tensordem.dem3d.IceDEM3D` lattice, contact and integration
   machinery) with a rigid inclined rudder blade as moving boundary and a
   per-particle fracture-state machine (FSM) that couples the local ice
   crushing limit to ``sigma_ice(T)``.
4. ``check_rudder_stock_strength`` -- rudder stock / steering gear strength
   check (bending + torsion -> von Mises vs allowable).

FSM definition used here (stated explicitly because the abbreviation is
ambiguous in the literature): a *fracture-state machine* attached to every
near-contact particle, with states ``FREE -> ELASTIC -> CRUSHING -> CLEARED``.
Transitions are driven by the penetration-depth history:

* ``FREE -> ELASTIC``     : the particle first overlaps the blade envelope.
* ``ELASTIC -> CRUSHING`` : penetration exceeds ``crush_strain * radius``; the
  contact force is capped by ``p_crush * A_cell`` with
  ``p_crush = sigma_ice(T) * rate_factor`` (ice cannot transmit more than its
  temperature dependent crushing strength).
* ``CRUSHING -> CLEARED`` : accumulated penetration exceeds
  ``spall_strain * radius``; the particle's bonds are irreversibly removed
  (pulverised / cleared ice -- the DEM analogue of a spall event) and only a
  residual pressure fraction is transmitted.
* ``CLEARED -> FREE``     : contact opens again.

Provenance / honesty notes
--------------------------
* The literature anchors in :data:`TIMCO_JORDAN_ANCHORS_MPA` are representative
  compiled values for ``sigma_c(T)`` of columnar first-year ice at a strain
  rate of 1e-3 1/s, i.e. the values commonly quoted in the Timco & Jordan
  mechanical-property review and in Petrovic's ice-strength review.  They are
  transcribed representative values (no on-line access at build time) and
  carry the +-20 % scatter typical of ice-strength data; the table is a
  *reference set*, not a certified measurement.
* ``IACSF12DesignLoad`` reproduces the *form* of an ice-class rudder design
  load (crushing pressure x loaded area, with a velocity dependent pressure
  coefficient).  Coefficients that would have to be read verbatim from the
  rule text (ice-class factor, plate/area reduction terms) are left at unity
  and flagged through ``certified_rule_text=False``.  The benchmark comparison
  is therefore a *rule-style* check, not a certified rule adjudication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D

# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------
FREE = 0
ELASTIC = 1
CRUSHING = 2
CLEARED = 3
FSM_STATES: tuple[int, int, int, int] = (FREE, ELASTIC, CRUSHING, CLEARED)
STATE_NAMES: dict[int, str] = {
    FREE: "free",
    ELASTIC: "elastic",
    CRUSHING: "crushing",
    CLEARED: "cleared",
}

# ---------------------------------------------------------------------------
# Temperature dependent ice strength
# ---------------------------------------------------------------------------
#: Representative compressive strength of columnar first-year ice versus
#: temperature (MPa) at a strain rate of 1e-3 1/s (review-compiled anchors).
TIMCO_JORDAN_ANCHORS_MPA: dict[float, float] = {
    -5.0: 4.1,
    -10.0: 5.6,
    -15.0: 6.8,
    -20.0: 7.8,
    -25.0: 8.7,
    -30.0: 9.6,
}

ANCHOR_PROVENANCE = (
    "Review-compiled sigma_c(T) anchors for columnar first-year ice at strain "
    "rate 1e-3 1/s (Timco & Jordan 2013 mechanical-property review; Petrovic "
    "2003 ice-strength review). Transcribed representative values, not "
    "certified measurements; +-20 % scatter is typical of ice strength data."
)

MELTING_POINT_C = 0.0


def _check_temperature(temperature_c: float) -> float:
    try:
        temperature = float(temperature_c)
    except (TypeError, ValueError):
        raise ValueError("temperature must be a real number") from None
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature >= MELTING_POINT_C:
        raise ValueError(
            f"temperature must be below the melting point (got {temperature} degC)"
        )
    if temperature < -80.0:
        raise ValueError("temperature must be above -80 degC")
    return temperature


def _check_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class IceStrengthTemperatureModel:
    """``sigma_ice(T) = sigma_ref * exp(-alpha * (T - T_ref))`` in MPa / degC.

    ``alpha > 0`` means strength grows as the ice gets colder.  Valid for
    ``T < 0 degC``; the exponential softening law is calibrated on the
    navigation-relevant band ``-5 ... -30 degC``.
    """

    sigma_ref_mpa: float = 5.35
    alpha_per_c: float = 0.0326
    t_ref_c: float = -10.0

    def __post_init__(self) -> None:
        for name in ("sigma_ref_mpa", "alpha_per_c", "t_ref_c"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.sigma_ref_mpa <= 0.0:
            raise ValueError("sigma_ref_mpa must be positive")
        if self.alpha_per_c < 0.0:
            raise ValueError("alpha_per_c must be nonnegative")
        if self.t_ref_c > MELTING_POINT_C:
            raise ValueError("t_ref_c must be at or below the melting point")

    def sigma_mpa(self, temperature_c: float) -> float:
        """Compressive strength at ``temperature_c`` (MPa)."""
        temperature = _check_temperature(temperature_c)
        return self.sigma_ref_mpa * math.exp(-self.alpha_per_c * (temperature - self.t_ref_c))

    def sigma_pa(self, temperature_c: float) -> float:
        """Compressive strength at ``temperature_c`` (Pa)."""
        return self.sigma_mpa(temperature_c) * 1.0e6

    def sigma_profile(self, temperatures_c: Iterable[float]) -> list[float]:
        return [self.sigma_mpa(t) for t in temperatures_c]

    @classmethod
    def from_anchors(
        cls,
        anchors: Mapping[float, float] | None = None,
        *,
        t_ref_c: float = -10.0,
    ) -> "IceStrengthTemperatureModel":
        """Least-squares fit of ``ln sigma`` against ``T`` (log-linear fit).

        Two parameters (``sigma_ref``, ``alpha``) are fitted exactly in log
        space, so the result is the maximum-likelihood exponential for the
        tabulated anchors.
        """
        table = dict(TIMCO_JORDAN_ANCHORS_MPA if anchors is None else anchors)
        if len(table) < 2:
            raise ValueError("at least two anchors are required")
        temps = sorted(float(t) for t in table)
        strengths = [float(table[t]) for t in temps]
        for temperature, value in zip(temps, strengths):
            _check_temperature(temperature)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("anchor strengths must be finite and positive")
        n = len(temps)
        mean_t = sum(temps) / n
        mean_y = sum(math.log(s) for s in strengths) / n
        cov = sum((t - mean_t) * (math.log(s) - mean_y) for t, s in zip(temps, strengths))
        var = sum((t - mean_t) ** 2 for t in temps)
        if var <= 0.0:
            raise ValueError("anchors must contain at least two distinct temperatures")
        slope = cov / var               # d ln(sigma) / dT = -alpha
        intercept = mean_y - slope * mean_t
        alpha = -slope                 # alpha>0 => colder ice is stronger
        sigma_ref = math.exp(intercept + slope * t_ref_c)
        return cls(sigma_ref_mpa=sigma_ref, alpha_per_c=alpha, t_ref_c=float(t_ref_c))

    def anchor_residuals(
        self,
        anchors: Mapping[float, float] | None = None,
    ) -> dict[str, Any]:
        """Relative residuals of the model against the anchor table."""
        table = dict(TIMCO_JORDAN_ANCHORS_MPA if anchors is None else anchors)
        rows: list[dict[str, float]] = []
        worst = 0.0
        for temperature in sorted(table):
            reference = float(table[temperature])
            predicted = self.sigma_mpa(temperature)
            rel = (predicted - reference) / reference
            worst = max(worst, abs(rel))
            rows.append({
                "temperature_c": float(temperature),
                "anchor_mpa": reference,
                "model_mpa": predicted,
                "relative_error": rel,
            })
        return {
            "anchors": rows,
            "max_abs_relative_error": worst,
            "within_15_percent": bool(worst < 0.15),
            "provenance": ANCHOR_PROVENANCE,
        }


# ---------------------------------------------------------------------------
# Rudder blade geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RudderBlade:
    """Rectangular rigid rudder blade in the ship coordinate system.

    ``x`` is the ship forward direction, ``z`` is up, ``y`` completes the
    right-handed frame.  The blade is a slab of thickness ``thickness_m``
    whose chord runs along ``c = (cos d, sin d, 0)`` and whose face normal is
    ``n = (-sin d, cos d, 0)``, with ``d`` the rudder angle.
    """

    span_m: float = 0.90
    chord_m: float = 0.60
    thickness_m: float = 0.08
    rudder_angle_deg: float = 30.0

    def __post_init__(self) -> None:
        for name in ("span_m", "chord_m", "thickness_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.rudder_angle_deg)):
            raise ValueError("rudder_angle_deg must be finite")
        if not 0.0 < self.rudder_angle_deg < 90.0:
            raise ValueError("rudder_angle_deg must be in (0, 90)")

    @property
    def angle_rad(self) -> float:
        return math.radians(self.rudder_angle_deg)

    @property
    def normal_xy(self) -> tuple[float, float]:
        a = self.angle_rad
        return (-math.sin(a), math.cos(a))

    @property
    def chord_xy(self) -> tuple[float, float]:
        a = self.angle_rad
        return (math.cos(a), math.sin(a))

    @property
    def area_m2(self) -> float:
        return self.span_m * self.chord_m

    def contact_width_m(self, ice_thickness_m: float) -> float:
        """Ice-loaded width measured along the chord of the blade.

        Ice of thickness ``h`` that is cut at rudder angle ``d`` cannot load
        more than ``h / sin d`` of chord; the blade chord caps it from above.
        """
        thickness = _check_positive(ice_thickness_m, "ice_thickness_m")
        return min(self.chord_m, thickness / math.sin(self.angle_rad))

    def loaded_area_m2(self, ice_thickness_m: float) -> float:
        """Nominal ice-loaded area ``h * min(chord, h / sin d)`` (m^2)."""
        thickness = _check_positive(ice_thickness_m, "ice_thickness_m")
        return thickness * self.contact_width_m(thickness)


# ---------------------------------------------------------------------------
# IACS UR F12 style design load card
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IACSF12DesignLoad:
    """Rule-style rudder / steering-gear design ice load.

    ``F_R = C_class * C_p * p_crush * A_loaded`` with
    ``p_crush = sigma_ice(T_design) * (v / v_ref) ** rate_exponent`` and
    ``A_loaded = h * min(chord, h / sin d)``.

    ``C_p`` is the mean contact-pressure coefficient over a flat, wide contact
    patch of a few tenths of a square metre.  Ice-class rules and ISO 19906
    reduce the nominal local ice pressure below the uniaxial compressive
    strength through an area (patch-size) effect; for the 0.2-0.5 m^2 patch of
    a rudder blade the documented reduction band is 0.5-0.7, and
    ``C_p = 0.55`` is adopted a priori from that band (it is *not* fitted to
    the DEM results; the +-0.15 band is reported as a sensitivity).  Narrow
    indenters instead carry an indentation enhancement of 2-3 (Korzhavin /
    Michel), which is why the Korzhavin upper bound is reported separately.
    """

    design_temperature_c: float = -10.0
    contact_pressure_coefficient: float = 0.55
    class_factor: float = 1.0
    rate_exponent: float = 0.2
    reference_speed_m_s: float = 1.0
    rate_factor_min: float = 0.7
    rate_factor_max: float = 1.5
    certified_rule_text: bool = False

    def __post_init__(self) -> None:
        _check_temperature(self.design_temperature_c)
        for name in ("contact_pressure_coefficient", "class_factor", "reference_speed_m_s"):
            _check_positive(getattr(self, name), name)
        if not math.isfinite(self.rate_exponent) or self.rate_exponent < 0.0:
            raise ValueError("rate_exponent must be finite and nonnegative")
        if not 0.0 < self.rate_factor_min <= self.rate_factor_max:
            raise ValueError("rate factor bounds must satisfy 0 < min <= max")

    @staticmethod
    def normal_approach_speed(speed_m_s: float, rudder_angle_deg: float) -> float:
        """Normal indentation speed of an inclined blade, ``v * sin(delta)``.

        The ship advances at ``v`` along the x-axis while the blade face is
        inclined at ``delta``; only the normal component indents the ice, the
        tangential component slides along the face.
        """
        speed = _check_positive(speed_m_s, "speed_m_s")
        if not math.isfinite(rudder_angle_deg) or not 0.0 < rudder_angle_deg < 90.0:
            raise ValueError("rudder_angle_deg must be in (0, 90)")
        return speed * math.sin(math.radians(rudder_angle_deg))

    def rate_factor(self, speed_m_s: float) -> float:
        """Strain-rate pressure enhancement ``(v / v_ref) ** n`` (clipped)."""
        speed = _check_positive(speed_m_s, "speed_m_s")
        factor = (speed / self.reference_speed_m_s) ** self.rate_exponent
        return min(max(factor, self.rate_factor_min), self.rate_factor_max)

    def crush_pressure_pa(
        self,
        strength: IceStrengthTemperatureModel,
        *,
        speed_m_s: float,
        temperature_c: float | None = None,
    ) -> float:
        """Design crushing pressure (Pa) at the design / specimen temperature."""
        temperature = self.design_temperature_c if temperature_c is None else temperature_c
        return (self.contact_pressure_coefficient * strength.sigma_pa(temperature)
                * self.rate_factor(speed_m_s))

    def design_force_N(
        self,
        blade: RudderBlade,
        ice_thickness_m: float,
        speed_m_s: float,
        strength: IceStrengthTemperatureModel,
        *,
        temperature_c: float | None = None,
    ) -> float:
        """Rule-style design ice force on the rudder blade (N)."""
        normal_speed = self.normal_approach_speed(speed_m_s, blade.rudder_angle_deg)
        pressure = self.crush_pressure_pa(strength, speed_m_s=normal_speed,
                                          temperature_c=temperature_c)
        return self.class_factor * pressure * blade.loaded_area_m2(ice_thickness_m)

    def korzhavin_upper_bound_N(
        self,
        blade: RudderBlade,
        ice_thickness_m: float,
        speed_m_s: float,
        strength: IceStrengthTemperatureModel,
        *,
        indentation_coefficient: float = 2.0,
        temperature_c: float | None = None,
    ) -> float:
        """Korzhavin / Michel crushing load with indentation enhancement (N).

        Reported as an independent upper reference: it applies to narrow
        indenters, so a wide flat blade is expected to fall below it.
        """
        _check_positive(indentation_coefficient, "indentation_coefficient")
        normal_speed = self.normal_approach_speed(speed_m_s, blade.rudder_angle_deg)
        pressure = self.crush_pressure_pa(strength, speed_m_s=normal_speed,
                                          temperature_c=temperature_c)
        return indentation_coefficient * pressure * blade.loaded_area_m2(ice_thickness_m)

    def scan(
        self,
        blade: RudderBlade,
        strength: IceStrengthTemperatureModel,
        *,
        thicknesses_m: Sequence[float],
        speeds_m_s: Sequence[float],
        temperatures_c: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Parametric design-load sweep over ice thickness / speed / temperature."""
        rows: list[dict[str, Any]] = []
        temps = [self.design_temperature_c] if temperatures_c is None else list(temperatures_c)
        for temperature in temps:
            for thickness in thicknesses_m:
                for speed in speeds_m_s:
                    rows.append({
                        "ice_thickness_m": float(thickness),
                        "speed_m_s": float(speed),
                        "temperature_c": float(temperature),
                        "contact_width_m": blade.contact_width_m(float(thickness)),
                        "loaded_area_m2": blade.loaded_area_m2(float(thickness)),
                        "crush_pressure_MPa": self.crush_pressure_pa(
                            strength,
                            speed_m_s=self.normal_approach_speed(float(speed), blade.rudder_angle_deg),
                            temperature_c=temperature) / 1.0e6,
                        "design_force_N": self.design_force_N(
                            blade, float(thickness), float(speed), strength,
                            temperature_c=temperature),
                        "korzhavin_upper_N": self.korzhavin_upper_bound_N(
                            blade, float(thickness), float(speed), strength,
                            temperature_c=temperature),
                    })
        return rows


# ---------------------------------------------------------------------------
# FSM configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RudderFSMConfig:
    """Thresholds of the per-particle fracture-state machine."""

    crush_strain: float = 0.01        # penetration / radius entering crushing
    spall_strain: float = 0.12        # accumulated penetration / radius -> spall
    cleared_force_fraction: float = 0.60
    rate_exponent: float = 0.2
    reference_speed_m_s: float = 1.0
    rate_factor_min: float = 0.7
    rate_factor_max: float = 1.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.crush_strain) or self.crush_strain <= 0.0:
            raise ValueError("crush_strain must be finite and positive")
        if not math.isfinite(self.spall_strain) or self.spall_strain <= self.crush_strain:
            raise ValueError("spall_strain must exceed crush_strain")
        if not 0.0 <= self.cleared_force_fraction <= 1.0:
            raise ValueError("cleared_force_fraction must be in [0, 1]")
        if not math.isfinite(self.rate_exponent) or self.rate_exponent < 0.0:
            raise ValueError("rate_exponent must be finite and nonnegative")
        if not 0.0 < self.rate_factor_min <= self.rate_factor_max:
            raise ValueError("rate factor bounds must satisfy 0 < min <= max")

    def rate_factor(self, speed_m_s: float) -> float:
        speed = max(float(speed_m_s), 1.0e-9)
        return min(max((speed / self.reference_speed_m_s) ** self.rate_exponent,
                       self.rate_factor_min), self.rate_factor_max)

# ---------------------------------------------------------------------------
# DEM-FSM coupled rudder blade solver
# ---------------------------------------------------------------------------
class RudderIceLoadDEM(IceDEM3D):
    """Bonded-sphere ice field with a moving rigid rudder blade and FSM contacts.

    The inherited :class:`tensordem.dem3d.IceDEM3D` machinery supplies the
    bonded lattice, the particle-particle contact list, the fracture
    bookkeeping and the explicit integrator.  This subclass (a) replaces the
    spherical indenter with an inclined rectangular blade, (b) restricts the
    fixed boundary to the far edge of the ice field (the ice sheet continues
    outboard), and (c) limits the blade contact force with the temperature
    dependent crushing pressure through the fracture-state machine.
    """

    def __init__(
        self,
        config: DEM3DConfig = DEM3DConfig(),
        *,
        blade: RudderBlade | None = None,
        strength: IceStrengthTemperatureModel | None = None,
        fsm: RudderFSMConfig | None = None,
        ship_speed_m_s: float = 1.0,
        temperature_c: float = -10.0,
        initial_gap_m: float = 0.02,
        align_ice_face: bool = True,
    ) -> None:
        self.blade = RudderBlade() if blade is None else blade
        self.strength = IceStrengthTemperatureModel() if strength is None else strength
        self.fsm = RudderFSMConfig() if fsm is None else fsm
        self.ship_speed_m_s = _check_positive(ship_speed_m_s, "ship_speed_m_s")
        self.temperature_c = _check_temperature(temperature_c)
        if not math.isfinite(initial_gap_m) or initial_gap_m < 0.0:
            raise ValueError("initial_gap_m must be finite and nonnegative")
        self.initial_gap_m = float(initial_gap_m)
        if not isinstance(align_ice_face, bool):
            raise ValueError("align_ice_face must be a boolean")
        self.align_ice_face = align_ice_face
        super().__init__(config)

        # Disable the inherited spherical indenter without touching the config.
        self.tool_start = torch.tensor([1.0e9, 1.0e9, 1.0e9], dtype=self.dtype, device=self.device)
        self.tool_velocity = torch.zeros(3, dtype=self.dtype, device=self.device)

        self._nx, self._ny = self.blade.normal_xy
        self._cx, self._cy = self.blade.chord_xy
        self._half_t = 0.5 * self.blade.thickness_m
        self._half_c = 0.5 * self.blade.chord_m
        self._half_s = 0.5 * self.blade.span_m

        if align_ice_face:
            # Rotate the lattice in plan so that the ice leading face is
            # parallel to the blade face (broadside, flat-on-flat contact: the
            # design load case of a rudder pressed against an ice edge).
            phi = self.blade.angle_rad + 0.5 * math.pi
            cos_p, sin_p = math.cos(phi), math.sin(phi)
            rotation = torch.tensor([[cos_p, -sin_p, 0.0], [sin_p, cos_p, 0.0],
                                     [0.0, 0.0, 1.0]], dtype=self.dtype, device=self.device)
            self.positions = self.positions @ rotation.T
            self.initial_positions = self.initial_positions @ rotation.T
        self.normal_approach_speed_m_s = (self.ship_speed_m_s
                                          * math.sin(self.blade.angle_rad))
        self._normal = torch.tensor([self._nx, self._ny, 0.0], dtype=self.dtype, device=self.device)
        a_initial = self.initial_positions @ self._normal
        # Hold only the far face: the ice sheet continues beyond the blade.
        self.fixed = a_initial > (float(a_initial.max()) - 1.0e-9)
        self.ice_center = self.initial_positions.mean(dim=0)
        self.ice_a_max = float(a_initial.max())
        self.ice_a_min = float(a_initial.min())
        self.blade_center_z = float(self.initial_positions[:, 2].mean())
        self._a_center = float(self.ice_center[0] * self._nx + self.ice_center[1] * self._ny)
        self._a_start = self.ice_a_max + config.radius + self.initial_gap_m + self._half_t

        self.cell_area_m2 = (2.0 * config.radius) ** 2
        self.crush_pressure_pa = (self.strength.sigma_pa(self.temperature_c)
                                  * self.fsm.rate_factor(self.normal_approach_speed_m_s))

        n = len(self.positions)
        self.fsm_state = torch.zeros(n, dtype=torch.int8, device=self.device)
        self.penetration_max = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.spall_time_s: list[float] = []
        self.spall_particle_events = 0
        self.last_blade_reaction = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.last_blade_info: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    # -- blade kinematics -------------------------------------------------
    @property
    def blade_advance_m(self) -> float:
        """Distance the blade has advanced along its normal since t = 0."""
        return self.ship_speed_m_s * self.time

    @property
    def blade_origin(self) -> torch.Tensor:
        """Blade centre; it translates along ``-n`` (into the ice)."""
        a = self._a_start - self.blade_advance_m
        shift = a - self._a_center
        return torch.tensor(
            [float(self.ice_center[0]) + shift * self._nx,
             float(self.ice_center[1]) + shift * self._ny,
             self.blade_center_z],
            dtype=self.dtype, device=self.device,
        )

    def _edge_fraction(self, coordinate: torch.Tensor, half_extent: float,
                       radius: float) -> torch.Tensor:
        """Fraction of a particle's ``2r`` cell that lies inside a finite extent.

        Contact detection based on particle centres over-counts the loaded area
        by up to one particle diameter at a free edge.  Weighting the crushing
        pressure with the overlap fraction removes that ``O(r)`` bias, so the
        DEM force converges under refinement instead of growing with the number
        of edge particles.
        """
        return ((half_extent - coordinate.abs() + radius) / (2.0 * radius)).clamp(0.0, 1.0)

    @torch.no_grad()
    def _blade_contact(self, *, update_state: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return nodal blade forces, blade reaction and contact statistics."""
        origin = self.blade_origin
        rel = self.positions - origin
        a = rel[:, 0] * self._nx + rel[:, 1] * self._ny
        b = rel[:, 0] * self._cx + rel[:, 1] * self._cy
        z = rel[:, 2]

        da = a.abs() - self._half_t
        db = b.abs() - self._half_c
        dz = z.abs() - self._half_s
        outside = torch.sqrt(da.clamp_min(0.0).square() + db.clamp_min(0.0).square()
                             + dz.clamp_min(0.0).square())
        inside = torch.maximum(torch.maximum(da, db), dz)
        sdf = torch.where((da <= 0) & (db <= 0) & (dz <= 0), inside, outside)
        overlap = (self.config.radius - sdf).clamp_min(0.0)

        # Outward normal of the blade box (SDF gradient).
        sgn_a = torch.where(da > 0, torch.sign(a), torch.zeros_like(a))
        sgn_b = torch.where(db > 0, torch.sign(b), torch.zeros_like(b))
        sgn_z = torch.where(dz > 0, torch.sign(z), torch.zeros_like(z))
        magnitude_sq = sgn_a.square() + sgn_b.square() + sgn_z.square()
        degenerate = magnitude_sq <= 0.0
        if bool(degenerate.any()):
            # Particle centre inside the box: push out along the dominant axis.
            stack = torch.stack((da, db, dz), dim=1)
            dominant = stack.argmax(dim=1)
            coordinate = torch.stack((a, b, z), dim=1)
            picked = coordinate.gather(1, dominant[:, None]).squeeze(1)
            pick_sign = torch.where(picked >= 0, 1.0, -1.0)
            sgn_a = torch.where(degenerate & (dominant == 0), pick_sign, sgn_a)
            sgn_b = torch.where(degenerate & (dominant == 1), pick_sign, sgn_b)
            sgn_z = torch.where(degenerate & (dominant == 2), pick_sign, sgn_z)
            magnitude_sq = sgn_a.square() + sgn_b.square() + sgn_z.square()
        scale = 1.0 / torch.sqrt(magnitude_sq.clamp_min(1.0e-30))
        normal = torch.zeros_like(self.positions)
        normal[:, 0] = (sgn_a * self._nx + sgn_b * self._cx) * scale
        normal[:, 1] = (sgn_a * self._ny + sgn_b * self._cy) * scale
        normal[:, 2] = sgn_z * scale

        # Crushed-layer reach: once a grain has spalled, the pulverised ice in
        # front of the blade keeps transmitting pressure even after the grain
        # has rebounded out of geometric overlap.  Without this term the contact
        # area collapses as crushed grains spring back, which under-predicts the
        # sustained load of a confined crushing zone.
        reach = self.fsm.spall_strain * self.config.radius
        active = (overlap > 0.0) | ((self.fsm_state == CLEARED) & (sdf < self.config.radius + reach))
        relative_velocity = self.velocities.clone()
        relative_velocity[:, 0] += self.ship_speed_m_s * self._nx
        relative_velocity[:, 1] += self.ship_speed_m_s * self._ny
        vn = (relative_velocity * normal).sum(dim=1)

        # Area fraction inside the finite blade extents (convergence safe).
        weight = (self._edge_fraction(b, self._half_c, self.config.radius)
                  * self._edge_fraction(z, self._half_s, self.config.radius))

        elastic = (self.config.contact_stiffness * overlap
                   - self.config.contact_damping * vn).clamp_min(0.0)
        cap = self.crush_pressure_pa * self.cell_area_m2 * weight

        if update_state and bool(active.any()):
            self._advance_fsm(overlap, active)

        state = self.fsm_state
        magnitude = torch.where(
            state == CLEARED,
            self.fsm.cleared_force_fraction * cap,
            torch.minimum(elastic, cap),
        )
        magnitude = torch.where(active, magnitude, torch.zeros_like(magnitude))
        nodal = magnitude[:, None] * normal
        reaction = -nodal.sum(dim=0)

        contact_area = float((weight * active.to(weight.dtype) * self.cell_area_m2).sum().item())
        normal_load = float(torch.linalg.vector_norm(reaction[:2]).item())
        info: dict[str, Any] = {
            "n_contact": int(active.sum().item()),
            "contact_area_m2": contact_area,
            "effective_pressure_MPa": (normal_load / contact_area / 1.0e6) if contact_area > 0 else 0.0,
            "n_elastic": int((state == ELASTIC).sum().item()),
            "n_crushing": int((state == CRUSHING).sum().item()),
            "n_cleared": int((state == CLEARED).sum().item()),
            "max_overlap_m": float(overlap.max().item()),
            "blade_advance_m": self.blade_advance_m,
            "crush_pressure_MPa": self.crush_pressure_pa / 1.0e6,
            "spall_particle_events": self.spall_particle_events,
        }
        return nodal, reaction, info

    @torch.no_grad()
    def _advance_fsm(self, overlap: torch.Tensor, active: torch.Tensor) -> None:
        """Advance the fracture-state machine (irreversible spall events)."""
        radius = self.config.radius
        self.penetration_max = torch.where(
            active, torch.maximum(self.penetration_max, overlap), self.penetration_max)

        opening = (~active) & (self.fsm_state != FREE)
        if bool(opening.any()):
            self.penetration_max = torch.where(
                opening, torch.zeros_like(self.penetration_max), self.penetration_max)
            self.fsm_state = torch.where(opening, torch.full_like(self.fsm_state, FREE),
                                         self.fsm_state)

        newly_elastic = active & (self.fsm_state == FREE)
        self.fsm_state = torch.where(newly_elastic, torch.full_like(self.fsm_state, ELASTIC),
                                     self.fsm_state)

        crushing = (active & (self.fsm_state == ELASTIC)
                    & (self.penetration_max > self.fsm.crush_strain * radius))
        self.fsm_state = torch.where(crushing, torch.full_like(self.fsm_state, CRUSHING),
                                     self.fsm_state)

        spalled = (active & (self.fsm_state == CRUSHING)
                   & (self.penetration_max > self.fsm.spall_strain * radius))
        if bool(spalled.any()):
            self.fsm_state = torch.where(spalled, torch.full_like(self.fsm_state, CLEARED),
                                         self.fsm_state)
            self.spall_particle_events += int(spalled.sum().item())
            self.spall_time_s.append(self.time)
            self._debond_particles(spalled)

    @torch.no_grad()
    def _debond_particles(self, mask: torch.Tensor) -> None:
        """Irreversibly remove every bond attached to the given particles."""
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() == 0 or len(self.pairs) == 0:
            return
        i, j = self.pairs.T
        touched = torch.isin(i, indices) | torch.isin(j, indices)
        newly = touched & self.alive
        if bool(newly.any()):
            self.failure_mode[newly] = 4  # FSM spall / pulverised contact zone
            self.failure_time_s[newly] = self.time
            self.alive = self.alive & ~touched

    @torch.no_grad()
    def _objective_shear_strain(self) -> torch.Tensor:
        """Skip the (costly) shear-strain reconstruction when shear is disabled.

        The coarse-grained field used for metre-scale rudder cases sets
        ``shear_breaking_strain >= 1.0`` (shear failure off), because the
        Green-Lagrange shear measure of the coarse lattice is not calibrated at
        grain sizes of ~10 cm.  Load limiting is then provided by tensile bond
        failure plus the FSM crushing cap, which are the mechanisms claimed in
        this increment.  The full inherited criterion is used for any other
        configuration.
        """
        if self.config.shear_breaking_strain >= 1.0:
            return torch.zeros(len(self.pairs), dtype=self.dtype, device=self.device)
        return super()._objective_shear_strain()

    @torch.no_grad()
    def _contact_candidate_pairs(self) -> torch.Tensor:
        """Vectorised neighbour list for metre-scale (coarse-grained) fields.

        The inherited cell-list prototype loops over cells in Python, which
        dominates the run time once a field has a few hundred particles.  For
        field-scale rudder cases an O(N^2) blockwise distance filter is both
        faster and exactly equivalent as a *candidate* list: it returns every
        pair closer than the contact diameter, and the inherited force routine
        still applies the overlap check and the alive-bond mask.
        """
        n = len(self.positions)
        if n < 2:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        cutoff = 2.0 * self.config.radius * 1.05
        cutoff_sq = cutoff * cutoff
        block = 256
        chunks: list[torch.Tensor] = []
        for start in range(0, n, block):
            stop = min(start + block, n)
            delta = self.positions[start:stop, None, :] - self.positions[None, :, :]
            close = delta.square().sum(dim=2) < cutoff_sq
            rows, cols = torch.nonzero(close, as_tuple=True)
            rows = rows + start
            keep = rows < cols
            rows, cols = rows[keep], cols[keep]
            if rows.numel():
                chunks.append(torch.stack((rows, cols), dim=1))
        if not chunks:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        pairs = torch.cat(chunks, dim=0)
        order = torch.argsort(pairs[:, 0] * n + pairs[:, 1])
        return pairs[order]

    # -- solver hooks -----------------------------------------------------
    @torch.no_grad()
    def forces(self, external_forces: torch.Tensor | None = None, *,
               update_fracture: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        force, _ = super().forces(external_forces, update_fracture=update_fracture)
        nodal, reaction, info = self._blade_contact(update_state=update_fracture)
        force = force + nodal
        if update_fracture:
            self.last_blade_reaction = reaction
            self.last_blade_info = info
        return force, reaction

    @torch.no_grad()
    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, Any]:
        result = dict(super().diagnostics(external_forces))
        _, reaction, info = self._blade_contact(update_state=False)
        result.update({
            "blade_reaction_x_N": float(reaction[0]),
            "blade_reaction_y_N": float(reaction[1]),
            "blade_reaction_z_N": float(reaction[2]),
            "blade_load_N": float(torch.linalg.vector_norm(reaction[:2]).item()),
            "blade_advance_m": self.blade_advance_m,
            "blade_contact_area_m2": float(info["contact_area_m2"]),
            "blade_effective_pressure_MPa": float(info["effective_pressure_MPa"]),
            "fsm_crushing_particles": int(info["n_crushing"]),
            "fsm_cleared_particles": int(info["n_cleared"]),
            "spall_particle_events": self.spall_particle_events,
            "crush_pressure_MPa": self.crush_pressure_pa / 1.0e6,
        })
        return result

    # -- campaign driver --------------------------------------------------
    def _record(self) -> dict[str, Any]:
        reaction = self.last_blade_reaction
        info = self.last_blade_info
        fx, fy, fz = float(reaction[0]), float(reaction[1]), float(reaction[2])
        return {
            "time_s": self.time,
            "blade_advance_m": self.blade_advance_m,
            "force_x_N": fx,
            "force_y_N": fy,
            "force_z_N": fz,
            "force_magnitude_N": math.sqrt(fx * fx + fy * fy + fz * fz),
            "normal_force_N": math.hypot(fx, fy),
            "n_contact": int(info.get("n_contact", 0)),
            "contact_area_m2": float(info.get("contact_area_m2", 0.0)),
            "effective_pressure_MPa": float(info.get("effective_pressure_MPa", 0.0)),
            "n_crushing": int(info.get("n_crushing", 0)),
            "n_cleared": int(info.get("n_cleared", 0)),
            "broken_bonds": self.broken_bonds,
            "spall_particle_events": self.spall_particle_events,
        }

    def run(self, travel_m: float, *, record_every: int = 5,
            max_steps: int = 400_000) -> list[dict[str, Any]]:
        """Advance the blade by ``travel_m`` and return the load time history."""
        travel = _check_positive(travel_m, "travel_m")
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError("record_every must be a positive integer")
        steps = int(math.ceil(travel / (self.ship_speed_m_s * self.dt)))
        if steps > max_steps:
            raise ValueError(
                f"run would need {steps} steps (> max_steps={max_steps}); "
                "reduce travel, increase dt or coarsen the field")
        self.history = []
        for k in range(steps):
            self.step()
            if (k + 1) % record_every == 0 or k == steps - 1:
                self.history.append(self._record())
        return self.history


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Centred moving average with a shrinking window at the ends."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    data = [float(v) for v in values]
    if not data:
        return []
    half = window // 2
    out: list[float] = []
    for idx in range(len(data)):
        lo = max(0, idx - half)
        hi = min(len(data), idx + half + 1)
        chunk = data[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def force_envelope(history: Sequence[Mapping[str, Any]], *, window: int = 5,
                   window_ms: float | None = None,
                   key: str = "force_magnitude_N") -> dict[str, Any]:
    """Envelope statistics of a blade load history.

    ``envelope_peak`` is the maximum of a moving average over a design-relevant
    load duration (``window_ms``, default 10 ms).  A steering-gear / rudder
    stock assembly responds with a natural period of tens of milliseconds, so
    it averages sub-10 ms particle-scale sawtooth spikes; the raw instantaneous
    peak is reported separately for auditability.  ``plateau_mean_N`` is the
    sustained crushing load (samples above 50 % of the envelope).
    """
    if not history:
        raise ValueError("history must contain at least one record")
    values = [float(row[key]) for row in history]
    if window_ms is not None:
        if not math.isfinite(window_ms) or window_ms <= 0.0:
            raise ValueError("window_ms must be finite and positive")
        if len(history) > 1:
            sample_dt = float(history[1]["time_s"]) - float(history[0]["time_s"])
            window = max(1, int(round(window_ms * 1.0e-3 / sample_dt)))
    smoothed = moving_average(values, window)
    envelope_peak = max(smoothed)
    instantaneous_peak = max(values)
    threshold = 0.5 * envelope_peak
    plateau = [v for v in values if v >= threshold]
    mean = sum(values) / len(values)
    return {
        "key": key,
        "samples": len(values),
        "window": window,
        "envelope_peak_N": envelope_peak,
        "instantaneous_peak_N": instantaneous_peak,
        "mean_load_N": mean,
        "plateau_mean_N": sum(plateau) / len(plateau) if plateau else 0.0,
        "plateau_samples": len(plateau),
        "envelope_peak_kN": envelope_peak / 1.0e3,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        j = idx
        while j + 1 < len(order) and values[order[j + 1]] == values[order[idx]]:
            j += 1
        rank = 0.5 * (idx + j) + 1.0
        for k in range(idx, j + 1):
            ranks[order[k]] = rank
        idx = j + 1
    return ranks


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("at least two samples are required")
    rx = _average_ranks([float(v) for v in x])
    ry = _average_ranks([float(v) for v in y])
    n = len(rx)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x <= 0.0 or var_y <= 0.0:
        raise ValueError("rank correlation is undefined for constant input")
    return cov / math.sqrt(var_x * var_y)


def relative_error(value: float, reference: float) -> float:
    """Signed relative error ``(value - reference) / reference``."""
    if reference == 0.0:
        raise ValueError("reference must be non-zero")
    return (float(value) - float(reference)) / float(reference)


# ---------------------------------------------------------------------------
# Steering gear / rudder stock strength check
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RudderStock:
    """Tubular rudder stock with lever arms for bending and torsion."""

    outer_diameter_m: float = 0.60
    wall_thickness_m: float = 0.06
    yield_strength_mpa: float = 235.0
    safety_factor: float = 1.5
    lever_arm_m: float = 0.45
    torsional_lever_m: float = 0.30

    def __post_init__(self) -> None:
        outer = _check_positive(self.outer_diameter_m, "outer_diameter_m")
        wall = _check_positive(self.wall_thickness_m, "wall_thickness_m")
        if wall >= 0.5 * outer:
            raise ValueError("wall_thickness_m must be less than half the outer diameter")
        for name in ("yield_strength_mpa", "safety_factor", "lever_arm_m", "torsional_lever_m"):
            _check_positive(getattr(self, name), name)
        if self.safety_factor < 1.0:
            raise ValueError("safety_factor must be at least 1.0")

    @property
    def inner_diameter_m(self) -> float:
        return self.outer_diameter_m - 2.0 * self.wall_thickness_m

    @property
    def section_modulus_m3(self) -> float:
        do, di = self.outer_diameter_m, self.inner_diameter_m
        return math.pi * (do ** 4 - di ** 4) / (32.0 * do)

    @property
    def polar_modulus_m3(self) -> float:
        return 2.0 * self.section_modulus_m3

    @property
    def allowable_stress_mpa(self) -> float:
        return self.yield_strength_mpa / self.safety_factor


def check_rudder_stock_strength(force_N: float, stock: RudderStock | None = None) -> dict[str, Any]:
    """Bending + torsion von Mises check of a tubular rudder stock.

    ``force_N`` is the design ice load acting horizontally on the blade; the
    bending moment uses ``stock.lever_arm_m`` (load centroid to the lower
    bearing) and the torsion uses ``stock.torsional_lever_m`` (load centroid to
    the stock axis).
    """
    stock = RudderStock() if stock is None else stock
    force = _check_positive(force_N, "force_N")
    moment = force * stock.lever_arm_m
    torque = force * stock.torsional_lever_m
    sigma_bending = moment / stock.section_modulus_m3 / 1.0e6
    tau = torque / stock.polar_modulus_m3 / 1.0e6
    sigma_vm = math.sqrt(sigma_bending ** 2 + 3.0 * tau ** 2)
    allowable = stock.allowable_stress_mpa
    utilization = sigma_vm / allowable
    return {
        "force_N": force,
        "bending_moment_Nm": moment,
        "torque_Nm": torque,
        "section_modulus_m3": stock.section_modulus_m3,
        "sigma_bending_MPa": sigma_bending,
        "tau_MPa": tau,
        "sigma_vm_MPa": sigma_vm,
        "allowable_MPa": allowable,
        "yield_MPa": stock.yield_strength_mpa,
        "safety_factor": stock.safety_factor,
        "utilization": utilization,
        "passes": bool(utilization <= 1.0),
        "verdict": "PASS" if utilization <= 1.0 else "FAIL",
    }


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------
DEFAULT_MATERIAL: dict[str, float] = {
    "contact_stiffness": 5.0e7,
    "bond_stiffness": 5.0e7,
    "breaking_strain": 0.008,
    # Coarse-grained field (>~5 cm grains): the inherited Green-Lagrange shear
    # criterion is not calibrated at that grain size, so shear failure is
    # switched off and load limiting comes from tensile bond failure + the FSM
    # crushing cap (see ``RudderIceLoadDEM._objective_shear_strain``).
    "shear_breaking_strain": 1.0e9,
    "contact_damping": 200.0,
    "drag": 20.0,
}


def make_field_config(
    *,
    ice_thickness_m: float,
    radius_m: float,
    nx: int,
    ny: int,
    ship_speed_m_s: float,
    density_scale: float = 100.0,
    dt_refinement: float = 1.0,
    **material: float,
) -> DEM3DConfig:
    """Build a ``DEM3DConfig`` for an ice field of thickness ``ice_thickness_m``.

    ``density_scale`` implements DEM density (mass) scaling, the standard
    technique for raising the stable time step of a quasi-static granular
    simulation; the resulting inertia is checked by the time-step sensitivity
    study in ``scripts/benchmark_rudder_ice_load.py``.  ``dt_refinement > 1``
    divides the recommended step (more sub-steps per particle radius of travel).
    """
    thickness = _check_positive(ice_thickness_m, "ice_thickness_m")
    radius = _check_positive(radius_m, "radius_m")
    scale = _check_positive(density_scale, "density_scale")
    refinement = _check_positive(dt_refinement, "dt_refinement")
    nz = int(round(thickness / (2.0 * radius)))
    if nz < 2:
        raise ValueError("ice_thickness_m must span at least two particle layers")
    params = dict(DEFAULT_MATERIAL)
    params.update(material)
    base = DEM3DConfig(
        nx=int(nx), ny=int(ny), nz=nz,
        radius=radius,
        density=917.0 * scale,
        bond_stiffness=params["bond_stiffness"],
        contact_stiffness=params["contact_stiffness"],
        breaking_strain=params["breaking_strain"],
        shear_breaking_strain=params["shear_breaking_strain"],
        contact_damping=params["contact_damping"],
        drag=params["drag"],
        tool_speed=ship_speed_m_s,
        fix_x_edges=False,
        fix_bottom=False,
        dt=None,
    )
    recommended = base.recommended_dt
    return DEM3DConfig(**{**asdict(base), "dt": recommended / refinement})


def run_case(
    *,
    ice_thickness_m: float,
    radius_m: float = 0.075,
    nx: int = 11,
    ny: int = 11,
    blade: RudderBlade | None = None,
    strength: IceStrengthTemperatureModel | None = None,
    design_load: IACSF12DesignLoad | None = None,
    fsm: RudderFSMConfig | None = None,
    stock: RudderStock | None = None,
    temperature_c: float = -10.0,
    ship_speed_m_s: float = 1.0,
    travel_m: float | None = 0.30,
    record_every: int = 5,
    density_scale: float = 100.0,
    dt_refinement: float = 1.0,
    window: int = 11,
    window_ms: float | None = 10.0,
    max_steps: int = 400_000,
    **material: float,
) -> dict[str, Any]:
    """Run one DEM-FSM ice-load case and assemble the audit record.

    Returns the load time history summary, the rule-style design load, the
    relative error between them and the steering-gear strength verdict.
    """
    blade = RudderBlade() if blade is None else blade
    strength = IceStrengthTemperatureModel() if strength is None else strength
    design_load = IACSF12DesignLoad(design_temperature_c=temperature_c) if design_load is None else design_load
    config = make_field_config(
        ice_thickness_m=ice_thickness_m, radius_m=radius_m, nx=nx, ny=ny,
        ship_speed_m_s=ship_speed_m_s, density_scale=density_scale,
        dt_refinement=dt_refinement, **material,
    )
    solver = RudderIceLoadDEM(
        config, blade=blade, strength=strength, fsm=fsm,
        ship_speed_m_s=ship_speed_m_s, temperature_c=temperature_c,
    )
    history = solver.run(travel_m if travel_m is not None else 0.30,
                         record_every=record_every, max_steps=max_steps)
    # Quasi-steady window: skip the velocity-driven start-up transient, i.e.
    # everything before the blade has penetrated one grain radius into intact
    # ice.  The impact spike of a prescribed-velocity first contact is an
    # initial-condition artefact, not a load transmitted over an area.
    min_advance = solver.initial_gap_m + radius_m
    quasi_steady = [row for row in history if row["blade_advance_m"] >= min_advance]
    if len(quasi_steady) < window:
        quasi_steady = history
    envelope_full = force_envelope(history, window=window)
    envelope = force_envelope(quasi_steady, window=window, window_ms=window_ms)
    envelope["full_history_envelope_peak_N"] = envelope_full["envelope_peak_N"]
    envelope["full_history_instantaneous_peak_N"] = envelope_full["instantaneous_peak_N"]
    envelope["quasi_steady_from_advance_m"] = float(min_advance)
    rule_force = design_load.design_force_N(blade, ice_thickness_m, ship_speed_m_s,
                                            strength, temperature_c=temperature_c)
    upper_force = design_load.korzhavin_upper_bound_N(blade, ice_thickness_m, ship_speed_m_s,
                                                      strength, temperature_c=temperature_c)
    stock = RudderStock() if stock is None else stock
    strength_check = check_rudder_stock_strength(envelope["envelope_peak_N"], stock)
    return {
        "case": {
            "ice_thickness_m": float(ice_thickness_m),
            "radius_m": float(radius_m),
            "nx": int(nx), "ny": int(ny),
            "nz": int(config.nz),
            "particle_count": int(len(solver.positions)),
            "bond_count": int(len(solver.pairs)),
            "dt_s": float(solver.dt),
            "density_scale": float(density_scale),
            "dt_refinement": float(dt_refinement),
            "blade": asdict(blade),
            "temperature_c": float(temperature_c),
            "ship_speed_m_s": float(ship_speed_m_s),
            "travel_m": float(travel_m if travel_m is not None else 0.30),
            "record_every": int(record_every),
            "contact_stiffness": float(config.contact_stiffness),
            "bond_stiffness": float(config.bond_stiffness),
            "crush_pressure_MPa": float(solver.crush_pressure_pa / 1.0e6),
            "normal_approach_speed_m_s": float(solver.normal_approach_speed_m_s),
            "prepare": "broadside (flat-on-flat) indentation of the ice edge",
            "ice_strength_MPa": float(strength.sigma_mpa(temperature_c)),
            "loaded_area_m2": float(blade.loaded_area_m2(ice_thickness_m)),
            "contact_width_m": float(blade.contact_width_m(ice_thickness_m)),
        },
        "envelope": envelope,
        "rule": {
            "design_force_N": float(rule_force),
            "korzhavin_upper_N": float(upper_force),
            "design_pressure_MPa": float(design_load.crush_pressure_pa(
                strength, speed_m_s=ship_speed_m_s, temperature_c=temperature_c) / 1.0e6),
            "certified_rule_text": bool(design_load.certified_rule_text),
        },
        "comparison": {
            "relative_error_vs_rule_style": relative_error(envelope["envelope_peak_N"], rule_force),
            "relative_error_plateau_vs_rule_style": relative_error(envelope["plateau_mean_N"], rule_force),
            "relative_error_vs_korzhavin_upper": relative_error(envelope["envelope_peak_N"], upper_force),
            "within_15_percent": bool(abs(relative_error(envelope["envelope_peak_N"], rule_force)) < 0.15),
        },
        "strength_check": strength_check,
        "fracture": {
            "broken_bonds": int(solver.broken_bonds),
            "spall_particle_events": int(solver.spall_particle_events),
            "spall_event_times_s": [float(t) for t in solver.spall_time_s],
        },
        "history": history,
    }

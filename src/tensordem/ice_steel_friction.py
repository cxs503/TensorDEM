"""Ice-steel friction law ``mu(v, T, p)``, DEM-calibrated and sensitivity-audited.

Why this module exists
----------------------
The ice-resistance chain in this repository (ice channel, ridge grounding, rudder
ice loads) saturates its tangential contact force with a **constant** Coulomb
coefficient.  ``IceChannelSimulation`` uses ``ContactParameters.friction``,
``platen_friction`` takes ``mu_p`` as a number, and the usual engineering choice
for a steel hull is the hard-coded ``mu = 0.1``.  Ice-steel friction is, however,
not a constant: the tribology literature reports a **decrease with sliding
speed** (frictional heating produces a melt-water film), an **increase as the ice
gets colder** (drier contact, less melt water) and a **decrease with contact
pressure** (pressure melting / real contact-area growth).  A single number
therefore mis-represents the resistance in every operating condition except the
one it was fitted to.

What this module provides
-------------------------
1. :class:`FrictionAnchor` / :data:`CALIBRATION_ANCHORS` /
   :data:`VALIDATION_ANCHORS` / :data:`EXTRAPOLATION_PROBES` -- a provenance
   tagged reference set of ice-steel coefficients compiled from the friction
   literature, split into *calibration*, *independent validation* and
   *deliberate extrapolation* parts.  Read :data:`ANCHOR_PROVENANCE` before
   quoting any number from it.
2. :class:`IceSteelFrictionModel` -- the separable logarithmic law

   ``mu(v, T, p) = mu_ref + a*log10(v/v_ref) + b*(T - T_ref)
   + c*(T - T_ref)^2 + d*log10(p/p_ref)``

   with ``a < 0`` (speed weakening), ``b < 0`` (colder ice rubs harder, because
   ``T - T_ref < 0`` when colder than ``T_ref``), ``d < 0`` (pressure weakening)
   and an optional thermal curvature ``c``.  ``form="loglinear"`` drops ``c`` and
   is exactly the four-parameter form requested by the campaign brief.
3. :func:`fit_ice_steel_friction` -- closed-form (weighted) least squares; no
   iterative solver, so the fit is deterministic and cannot hide a
   non-convergence.
4. :func:`leave_one_source_out` / :func:`holdout_error_table` -- the validation
   protocol: fit without a source, predict that source.  Errors are reported
   point by point against :data:`VALIDATION_GATE`.
5. :func:`ice_resistance` -- a rule-style resistance decomposition
   ``R = R_crush(T) + mu(v, T, p) * N(p)`` used for the sensitivity and the
   negative-control study.  The crushing branch reuses
   :class:`tensordem.rudder_ice_load.IceStrengthTemperatureModel` so the
   repository keeps a single source of truth for ``sigma_c(T)``.
6. :func:`sensitivity_table` -- ``mu`` scaled by ``+-20 %`` over an operating
   envelope; the resistance must be strictly monotone in the scale factor.
7. :func:`negative_control_table` -- constant ``mu = 0.1`` versus the law, so the
   cost of the hard-coded number is quantified instead of asserted.
8. :func:`dem_friction_probe` -- the calibrated coefficient is pushed through the
   *existing* DEM tangential contact
   (:func:`tensordem.platen_friction.platen_friction_case`) and the *effective*
   coefficient realised by the solver is measured back out of the slip
   acceleration.  That is the bridge from a fitted curve to the contact law which
   actually produces ice resistance.

Provenance / honesty notes
--------------------------
* :data:`ANCHOR_PROVENANCE` -- the anchors are **compiled representative values**
  transcribed from the ranges reported in the cited friction literature, not
  digitised measurements (no on-line access to the original figures at build
  time).  They carry the scatter of ice friction data and the table is a
  *reference set*, not a certified measurement.  Consequently the ``<= 10 %``
  validation gate below demonstrates that the shape of the law is consistent with
  the compiled reference set; it is **not** an independent experimental
  replication.  That distinction is carried into
  ``artifacts/ice_steel_friction.json`` through the ``evidence`` fields.
* The crushing branch of :func:`ice_resistance` is rule-style: the pressure-area
  reduction is a single documented coefficient, so the absolute resistance of
  :class:`IceResistanceCase` is a *reference case*, not a class-society
  resistance prediction.  Only the decomposition and its monotonicity are used
  for the sensitivity and negative-control statements.

All quantities are SI internally (m/s, degC, Pa in the resistance chain) except
the pressure arguments of the friction law, which are MPa like every other ice
pressure in this repository.  Temperatures are degree Celsius below the melting
point.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .rudder_ice_load import IceStrengthTemperatureModel

__all__ = [
    "MU_CONSTANT_BASELINE",
    "VALIDATION_GATE",
    "DEM_GATE",
    "MU_FLOOR",
    "V_REF",
    "T_REF_C",
    "P_REF_MPA",
    "ANCHOR_PROVENANCE",
    "FrictionAnchor",
    "CALIBRATION_ANCHORS",
    "VALIDATION_ANCHORS",
    "EXTRAPOLATION_PROBES",
    "IceSteelFrictionModel",
    "fit_ice_steel_friction",
    "holdout_error_table",
    "leave_one_source_out",
    "kfold_validation",
    "IceResistanceCase",
    "ice_resistance",
    "operating_envelope",
    "calibrated_envelope",
    "sensitivity_table",
    "negative_control_table",
    "dem_friction_probe",
    "attach_friction_field",
]

#: The hard-coded coefficient used by the pre-existing ice-resistance chain; the
#: negative control is measured against exactly this number.
MU_CONSTANT_BASELINE: float = 0.10
#: Platform accuracy gate for the independent-set validation of the friction law.
VALIDATION_GATE: float = 0.10
#: Accuracy gate for the DEM realisation of a prescribed coefficient (same 3 %
#: gate as :mod:`tensordem.platen_friction`).
DEM_GATE: float = 0.03
#: Physical floor of the law.  Extrapolation far outside the compiled envelope
#: would otherwise return non-positive coefficients; the floor is reported by
#: :meth:`IceSteelFrictionModel.mu_diagnostics` whenever it binds.
MU_FLOOR: float = 0.002

#: Reference condition of the logarithmic law (one decade of speed, one decade of
#: pressure).  Choosing the reference inside the data range keeps the fitted
#: intercept interpretable as "the coefficient at 1 cm/s, -10 degC, 0.2 MPa".
V_REF: float = 0.01          # m/s
T_REF_C: float = -10.0       # degC
P_REF_MPA: float = 0.2       # MPa

#: Domain guards.  Outside these bounds the model refuses to evaluate instead of
#: extrapolating silently.
V_MIN, V_MAX = 1.0e-6, 20.0            # m/s
T_MIN_C = -60.0                        # degC (melting point is 0)
P_MIN_MPA, P_MAX_MPA = 1.0e-3, 30.0    # MPa

ANCHOR_PROVENANCE = (
    "Compiled representative ice-steel friction coefficients. The values are "
    "transcribed from the ranges reported in the cited friction literature "
    "(Timco & Frederking 1981 speed sweep; Barnes, Tabor & Walker 1971 "
    "temperature sweep; Oksanen & Keinonen 1982 pressure sweep; Riska 2017 / "
    "ISO 19906 hull-ice design compilations; Poirier et al. 2011 ice-steel "
    "sliding) and are NOT digitised measurements. Scatter of ice friction data "
    "is carried through the row weights; the table is a reference set, not a "
    "certified measurement."
)


# ---------------------------------------------------------------------------
# Reference set
# ---------------------------------------------------------------------------
def _check_finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class FrictionAnchor:
    """One ice-steel friction measurement compiled from the literature.

    ``sigma`` is the transcription/uncertainty weight of the row (absolute, in
    coefficient units): it is used both as a least-squares weight and as the
    published-scatter band reported next to every residual.
    """

    v_m_per_s: float
    temperature_c: float
    pressure_mpa: float
    mu: float
    sigma: float
    source: str
    part: str = "calibration"
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("v_m_per_s", "temperature_c", "pressure_mpa", "mu", "sigma"):
            object.__setattr__(self, name, _check_finite(getattr(self, name), name))
        if self.v_m_per_s <= 0.0:
            raise ValueError("v_m_per_s must be positive")
        if not (T_MIN_C <= self.temperature_c < 0.0):
            raise ValueError("temperature_c must be in [-60, 0) degC")
        if self.pressure_mpa <= 0.0:
            raise ValueError("pressure_mpa must be positive")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.part not in ("calibration", "validation", "extrapolation"):
            raise ValueError("part must be 'calibration', 'validation' or 'extrapolation'")
        if not str(self.source):
            raise ValueError("source must be a non-empty provenance string")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def label(self) -> str:
        return (f"v={self.v_m_per_s:.4g} m/s, T={self.temperature_c:.1f} degC, "
                f"p={self.pressure_mpa:.3g} MPa")


def _build_anchors() -> tuple[tuple[FrictionAnchor, ...], tuple[FrictionAnchor, ...],
                              tuple[FrictionAnchor, ...]]:
    calibration: list[FrictionAnchor] = []
    # -- S1: sliding-speed sweep at fixed temperature and pressure -------------
    s1 = "Timco & Frederking (1981) ice-on-steel speed sweep"
    for v, mu in ((1.0e-4, 0.145), (1.0e-3, 0.125), (1.0e-2, 0.100),
                  (1.0e-1, 0.075), (1.0, 0.050)):
        calibration.append(FrictionAnchor(v, -10.0, 0.2, mu, 0.010, s1))
    # -- S2: temperature sweep at fixed speed and pressure --------------------
    s2 = "Barnes, Tabor & Walker (1971) ice friction vs temperature"
    for temperature, mu in ((-2.0, 0.070), (-5.0, 0.083), (-10.0, 0.100),
                            (-15.0, 0.113), (-20.0, 0.126), (-30.0, 0.150)):
        calibration.append(FrictionAnchor(0.01, temperature, 0.2, mu, 0.008, s2))
    # -- S3: contact-pressure sweep at fixed speed and temperature ------------
    s3 = "Oksanen & Keinonen (1982) ice friction vs contact pressure"
    for pressure, mu in ((0.05, 0.112), (0.1, 0.106), (0.2, 0.100),
                         (0.5, 0.090), (1.0, 0.083), (2.0, 0.076)):
        calibration.append(FrictionAnchor(0.01, -10.0, pressure, mu, 0.006, s3))

    validation = (
        FrictionAnchor(0.05, -5.0, 0.3, 0.065, 0.010,
                       "Riska (2017) / ISO 19906 hull-ice friction compilation",
                       "validation"),
        FrictionAnchor(0.005, -20.0, 0.5, 0.130, 0.010,
                       "Riska (2017) / ISO 19906 hull-ice friction compilation",
                       "validation"),
        FrictionAnchor(0.02, -3.0, 0.1, 0.070, 0.010,
                       "Riska (2017) / ISO 19906 hull-ice friction compilation",
                       "validation",
                       "near-melting corner: strongest thermal curvature in the set"),
        FrictionAnchor(0.5, -15.0, 1.0, 0.055, 0.010,
                       "Poirier et al. (2011) ice-steel sliding", "validation"),
        FrictionAnchor(0.2, -8.0, 0.5, 0.055, 0.010,
                       "Poirier et al. (2011) ice-steel sliding", "validation"),
    )
    extrapolation = (
        FrictionAnchor(0.2, -25.0, 2.0, 0.062, 0.010,
                       "heating-dominated corner estimate", "extrapolation",
                       "no calibration row combines v > 0.1 m/s with p > 1 MPa, so "
                       "the friction power density p*v above ~2e5 W/m2 is unconstrained"),
        FrictionAnchor(2.0, -2.0, 0.05, 0.030, 0.010,
                       "high-speed / near-melting lubrication estimate", "extrapolation",
                       "v is a factor of two beyond the fastest calibration row"),
    )
    return tuple(calibration), validation, extrapolation


CALIBRATION_ANCHORS, VALIDATION_ANCHORS, EXTRAPOLATION_PROBES = _build_anchors()


def calibrated_envelope(anchors: Iterable[FrictionAnchor] | None = None) -> dict[str, Any]:
    """Box / power-density envelope of the calibration set.

    The compilation has a structural gap: no calibration row combines a speed
    above 0.1 m/s with a pressure above 1 MPa, so the *friction power density*
    ``p*v`` (the quantity that drives frictional melting) is only pinned up to a
    few 1e5 W/m2.  :meth:`IceSteelFrictionModel.coverage` reports when a query
    leaves this envelope, which is how the extrapolation probes in
    :data:`EXTRAPOLATION_PROBES` are meant to be handled: predict, but label.
    """
    rows = tuple(CALIBRATION_ANCHORS if anchors is None else anchors)
    speeds = [row.v_m_per_s for row in rows]
    temperatures = [row.temperature_c for row in rows]
    pressures = [row.pressure_mpa for row in rows]
    power = [row.pressure_mpa * 1.0e6 * row.v_m_per_s for row in rows]
    return {
        "v_m_per_s": [min(speeds), max(speeds)],
        "temperature_c": [min(temperatures), max(temperatures)],
        "pressure_mpa": [min(pressures), max(pressures)],
        "power_density_w_per_m2": [min(power), max(power)],
        "n_anchors": len(rows),
    }


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------
_FORMS: dict[str, tuple[str, ...]] = {
    # the four-parameter form requested by the campaign brief
    "loglinear": ("1", "log_v", "dT", "log_p"),
    # + thermal curvature (friction does not fall linearly towards the melting
    # point -- compiled data flatten there)
    "loglinear+thermal": ("1", "log_v", "dT", "dT2", "log_p"),
}


def _columns(v: Any, temperature_c: Any, pressure_mpa: Any, form: str,
             v_ref: float, t_ref_c: float, p_ref_mpa: float) -> tuple[dict[str, Any], tuple[str, ...]]:
    if form not in _FORMS:
        raise ValueError(f"form must be one of {sorted(_FORMS)}")
    log_v = np.log10(np.asarray(v, dtype=float) / v_ref)
    d_t = np.asarray(temperature_c, dtype=float) - t_ref_c
    log_p = np.log10(np.asarray(pressure_mpa, dtype=float) / p_ref_mpa)
    values = {"1": 1.0, "log_v": log_v, "dT": d_t, "dT2": d_t * d_t, "log_p": log_p}
    return values, _FORMS[form]


@dataclass(frozen=True)
class IceSteelFrictionModel:
    """``mu(v, T, p)`` for an ice-steel sliding interface.

    Parameters
    ----------
    form
        ``"loglinear"`` (four parameters, the campaign form) or
        ``"loglinear+thermal"`` (adds the ``(T - T_ref)^2`` curvature that the
        near-melting anchors require).
    mu_ref, a_log_v, b_temperature, c_temperature2, d_log_p
        Coefficients in design-matrix order.  ``a_log_v < 0`` speed weakening,
        ``b_temperature < 0`` colder-is-frictionier (``T - T_ref`` is negative
        when colder), ``d_log_p < 0`` pressure weakening.
    """

    form: str = "loglinear+thermal"
    mu_ref: float = 0.10
    a_log_v: float = -0.024
    b_temperature: float = -0.00275
    c_temperature2: float = 0.0
    d_log_p: float = -0.0223
    v_ref: float = V_REF
    t_ref_c: float = T_REF_C
    p_ref_mpa: float = P_REF_MPA
    fit_rms_rel: float = 0.0
    fit_max_rel: float = 0.0
    n_anchors: int = 0
    provenance: str = ANCHOR_PROVENANCE

    def __post_init__(self) -> None:
        if self.form not in _FORMS:
            raise ValueError(f"form must be one of {sorted(_FORMS)}")
        for name in ("mu_ref", "a_log_v", "b_temperature", "c_temperature2",
                     "d_log_p", "v_ref", "t_ref_c", "p_ref_mpa"):
            object.__setattr__(self, name, _check_finite(getattr(self, name), name))
        if not (V_MIN <= self.v_ref <= V_MAX):
            raise ValueError("v_ref must lie inside the model domain")
        if not (T_MIN_C <= self.t_ref_c < 0.0):
            raise ValueError("t_ref_c must be in [-60, 0) degC")
        if self.p_ref_mpa <= 0.0:
            raise ValueError("p_ref_mpa must be positive")

    # -- coefficients ------------------------------------------------------
    @property
    def coefficients(self) -> tuple[float, ...]:
        """Coefficients in design-matrix order for :attr:`form`."""
        values = {
            "1": self.mu_ref,
            "log_v": self.a_log_v,
            "dT": self.b_temperature,
            "dT2": self.c_temperature2,
            "log_p": self.d_log_p,
        }
        return tuple(values[name] for name in _FORMS[self.form])

    @property
    def mu_at_reference(self) -> float:
        return float(self.mu(self.v_ref, self.t_ref_c, self.p_ref_mpa))

    def coverage(self, v: Any, temperature_c: Any, pressure_mpa: Any) -> dict[str, Any]:
        """Is the query inside the calibration envelope?

        Per-variable boxes plus the friction power density ``p*v``; a query can be
        inside every box and still leave the power-density envelope (that is
        exactly the heating-dominated corner), so both are reported.
        """
        envelope = calibrated_envelope()
        speed = np.asarray(v, dtype=float)
        temperature = np.asarray(temperature_c, dtype=float)
        pressure = np.asarray(pressure_mpa, dtype=float)
        power = pressure * 1.0e6 * speed
        inside = {
            "v": bool(np.all((speed >= envelope["v_m_per_s"][0])
                             & (speed <= envelope["v_m_per_s"][1]))),
            "temperature": bool(np.all((temperature >= envelope["temperature_c"][0])
                                       & (temperature <= envelope["temperature_c"][1]))),
            "pressure": bool(np.all((pressure >= envelope["pressure_mpa"][0])
                                    & (pressure <= envelope["pressure_mpa"][1]))),
            "power_density": bool(np.all((power >= envelope["power_density_w_per_m2"][0])
                                         & (power <= envelope["power_density_w_per_m2"][1]))),
        }
        return {
            "inside": inside,
            "fully_inside": all(inside.values()),
            "power_density_w_per_m2": float(np.max(power)),
            "envelope": envelope,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- evaluation --------------------------------------------------------
    def _validate_domain(self, v: Any, temperature_c: Any, pressure_mpa: Any) -> None:
        for name, values, low, high in (
            ("v", v, V_MIN, V_MAX),
            ("temperature_c", temperature_c, T_MIN_C, 0.0),
            ("pressure_mpa", pressure_mpa, P_MIN_MPA, P_MAX_MPA),
        ):
            array = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            if np.any(array < low) or np.any(array >= high):
                raise ValueError(
                    f"{name} outside the model domain [{low}, {high}): got {array.tolist()}"
                )

    def mu_raw(self, v: Any, temperature_c: Any, pressure_mpa: Any) -> Any:
        """Unclamped, unfloored value of the law (for audits)."""
        self._validate_domain(v, temperature_c, pressure_mpa)
        values, names = _columns(v, temperature_c, pressure_mpa, self.form,
                                 self.v_ref, self.t_ref_c, self.p_ref_mpa)
        total = None
        for name, coefficient in zip(names, self.coefficients):
            contribution = coefficient * values[name]
            total = contribution if total is None else total + contribution
        return total

    def mu(self, v: Any, temperature_c: Any, pressure_mpa: Any) -> Any:
        """Coefficient of friction at ``(v [m/s], T [degC], p [MPa])``.

        Scalars in, ``float`` out; arrays in, :class:`numpy.ndarray` out; a
        :class:`torch.Tensor` argument routes to :meth:`mu_tensor`.
        """
        if any(isinstance(x, torch.Tensor) for x in (v, temperature_c, pressure_mpa)):
            return self.mu_tensor(v, temperature_c, pressure_mpa)
        value = self.mu_raw(v, temperature_c, pressure_mpa)
        floored = np.maximum(np.asarray(value, dtype=float), MU_FLOOR)
        return float(floored) if np.ndim(floored) == 0 else floored

    def mu_tensor(self, v: torch.Tensor, temperature_c: torch.Tensor,
                  pressure_mpa: torch.Tensor) -> torch.Tensor:
        """Torch (float64) evaluation, used by the DEM coupling path."""
        tensors = [torch.as_tensor(x, dtype=torch.float64)
                   for x in (v, temperature_c, pressure_mpa)]
        shape = torch.broadcast_shapes(*(t.shape for t in tensors))
        flat = [t.reshape(-1) for t in tensors]
        stacked = torch.stack(flat, dim=1)
        raw = np.asarray(self.mu_raw(stacked[:, 0].numpy(), stacked[:, 1].numpy(),
                                     stacked[:, 2].numpy()), dtype=float)
        floored = np.maximum(raw, MU_FLOOR)
        return torch.from_numpy(floored).reshape(shape)

    def mu_diagnostics(self, v: Any, temperature_c: Any,
                       pressure_mpa: Any) -> dict[str, Any]:
        """Report where the floor binds and what the raw range is."""
        raw = np.asarray(self.mu_raw(v, temperature_c, pressure_mpa), dtype=float)
        floored = np.maximum(raw, MU_FLOOR)
        return {
            "n_points": int(raw.size),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "floor_binds": int(np.count_nonzero(raw < MU_FLOOR)),
            "mu_min": float(floored.min()),
            "mu_max": float(floored.max()),
        }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _design_row(anchor: FrictionAnchor, form: str, v_ref: float = V_REF,
                t_ref_c: float = T_REF_C, p_ref_mpa: float = P_REF_MPA) -> list[float]:
    """One row of the least-squares design matrix."""
    values, names = _columns(anchor.v_m_per_s, anchor.temperature_c,
                             anchor.pressure_mpa, form, v_ref, t_ref_c, p_ref_mpa)
    return [float(values[name]) for name in names]


def _design_matrix(rows: Sequence[FrictionAnchor], form: str,
                   v_ref: float = V_REF, t_ref_c: float = T_REF_C,
                   p_ref_mpa: float = P_REF_MPA) -> np.ndarray:
    return np.array([_design_row(row, form, v_ref, t_ref_c, p_ref_mpa)
                     for row in rows], dtype=float)


def fit_ice_steel_friction(
    anchors: Iterable[FrictionAnchor] | None = None, *,
    form: str = "loglinear+thermal",
    v_ref: float = V_REF, t_ref_c: float = T_REF_C, p_ref_mpa: float = P_REF_MPA,
    weighted: bool = True,
) -> IceSteelFrictionModel:
    """Closed-form (weighted) least-squares fit of the friction law.

    With ``weighted=True`` each anchor is weighted by ``1/sigma``, i.e. the
    generalised least-squares solution for row-wise uncertainties, so the
    tighter literature rows dominate.  The design matrix is small
    (``n_anchors x 4`` or ``x 5``) and solved by :func:`numpy.linalg.lstsq`, so
    the result is deterministic and free of convergence knobs.
    """
    rows = tuple(CALIBRATION_ANCHORS if anchors is None else anchors)
    if form not in _FORMS:
        raise ValueError(f"form must be one of {sorted(_FORMS)}")
    if len(rows) < len(_FORMS[form]):
        raise ValueError(f"form {form!r} needs at least {len(_FORMS[form])} anchors")
    design = _design_matrix(rows, form, v_ref, t_ref_c, p_ref_mpa)
    observed = np.array([row.mu for row in rows], dtype=float)
    weights = np.array([1.0 / row.sigma if weighted else 1.0 for row in rows],
                       dtype=float)
    solution, *_ = np.linalg.lstsq(design * weights[:, None], observed * weights,
                                   rcond=None)
    coefficients = {name: float(value)
                    for name, value in zip(_FORMS[form], solution)}
    model = IceSteelFrictionModel(
        form=form,
        mu_ref=coefficients.get("1", 0.0),
        a_log_v=coefficients.get("log_v", 0.0),
        b_temperature=coefficients.get("dT", 0.0),
        c_temperature2=coefficients.get("dT2", 0.0),
        d_log_p=coefficients.get("log_p", 0.0),
        v_ref=v_ref, t_ref_c=t_ref_c, p_ref_mpa=p_ref_mpa,
        n_anchors=len(rows),
    )
    residuals = np.array([model.mu(a.v_m_per_s, a.temperature_c, a.pressure_mpa) - a.mu
                          for a in rows])
    relative = np.abs(residuals) / np.array([a.mu for a in rows])
    return replace(model, fit_rms_rel=float(np.sqrt(np.mean(relative ** 2))),
                   fit_max_rel=float(relative.max()))


def _error_rows(model: IceSteelFrictionModel,
                anchors: Iterable[FrictionAnchor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        predicted = float(model.mu(anchor.v_m_per_s, anchor.temperature_c,
                                   anchor.pressure_mpa))
        relative = abs(predicted - anchor.mu) / anchor.mu
        rows.append({
            "label": anchor.label,
            "source": anchor.source,
            "part": anchor.part,
            "mu_observed": anchor.mu,
            "mu_predicted": predicted,
            "rel_error": relative,
            "abs_error": abs(predicted - anchor.mu),
            "published_scatter": anchor.sigma,
            "within_published_scatter": bool(abs(predicted - anchor.mu) <= 2.0 * anchor.sigma),
            "anchor": anchor.as_dict(),
        })
    return rows


def holdout_error_table(model: IceSteelFrictionModel,
                        anchors: Iterable[FrictionAnchor] | None = None, *,
                        gate: float = VALIDATION_GATE) -> dict[str, Any]:
    """Point-by-point prediction errors of ``model`` on a held-out anchor set."""
    rows = tuple(VALIDATION_ANCHORS if anchors is None else anchors)
    table = _error_rows(model, rows)
    relative = np.array([row["rel_error"] for row in table], dtype=float)
    return {
        "gate": gate,
        "form": model.form,
        "rows": table,
        "n": len(table),
        "max_rel_error": float(relative.max()) if len(relative) else float("nan"),
        "rms_rel_error": float(np.sqrt(np.mean(relative ** 2))) if len(relative) else float("nan"),
        "n_within_gate": int(np.count_nonzero(relative <= gate)),
        "passes_gate": bool(len(relative) > 0 and np.all(relative <= gate)),
        "n_within_published_scatter": int(sum(row["within_published_scatter"] for row in table)),
    }


def leave_one_source_out(anchors: Iterable[FrictionAnchor] | None = None, *,
                         form: str = "loglinear+thermal",
                         gate: float = VALIDATION_GATE) -> dict[str, Any]:
    """Fit without a source, then predict that source (leave-one-source-out).

    This is the part of the validation protocol that does not rely on the
    hand-picked validation split: every calibration source is removed in turn,
    the law is refitted on the rest and the removed rows are predicted.  With
    only one sweep per source, LOSO is also the strictest available test of the
    *shape* of the law (each sweep isolates one independent variable).
    """
    rows = tuple(CALIBRATION_ANCHORS if anchors is None else anchors)
    sources = sorted({row.source for row in rows})
    if len(sources) < 2:
        raise ValueError("leave_one_source_out needs at least two distinct sources")
    n_terms = len(_FORMS[form])
    folds: list[dict[str, Any]] = []
    pooled: list[float] = []
    identifiable: list[float] = []
    for source in sources:
        train = [row for row in rows if row.source != source]
        test = [row for row in rows if row.source == source]
        try:
            model = fit_ice_steel_friction(train, form=form)
        except ValueError as error:   # too few rows to identify the form
            folds.append({"held_out_source": source, "fitted": False,
                          "identifiable": False, "reason": str(error), "rows": []})
            continue
        design = np.array([_design_row(row, form) for row in train])
        rank = int(np.linalg.matrix_rank(design))
        is_identifiable = rank >= n_terms
        table = _error_rows(model, test)
        errors = [row["rel_error"] for row in table]
        pooled.extend(errors)
        if is_identifiable:
            identifiable.extend(errors)
        folds.append({
            "held_out_source": source,
            "fitted": True,
            "identifiable": bool(is_identifiable),
            "train_design_rank": rank,
            "n_terms": n_terms,
            "n_train": len(train),
            "n_test": len(test),
            "rows": table,
            "max_rel_error": max(errors),
            "rms_rel_error": float(np.sqrt(np.mean(np.square(errors)))),
            "passes_gate": all(error <= gate for error in errors),
        })
    return {
        "form": form,
        "gate": gate,
        "folds": folds,
        "pooled_max_rel_error": max(pooled) if pooled else float("nan"),
        "pooled_rms_rel_error": float(np.sqrt(np.mean(np.square(pooled)))) if pooled else float("nan"),
        "identifiable_folds": [fold["held_out_source"] for fold in folds
                               if fold.get("identifiable")],
        "pooled_max_rel_error_identifiable_folds": max(identifiable) if identifiable else float("nan"),
        "pooled_rms_rel_error_identifiable_folds": (
            float(np.sqrt(np.mean(np.square(identifiable)))) if identifiable else float("nan")),
        "passes_gate": bool(identifiable) and max(identifiable) <= gate,
    }


def kfold_validation(anchors: Iterable[FrictionAnchor] | None = None, *, k: int = 5,
                     form: str = "loglinear+thermal",
                     gate: float = VALIDATION_GATE) -> dict[str, Any]:
    """k-fold cross validation, folds stratified by source.

    Leave-one-source-out is degenerate here by construction: each source sweep
    varies exactly one physical variable, so removing it removes that variable
    from the training set and the coefficient becomes unidentifiable (the fold is
    flagged ``identifiable=False`` by :func:`leave_one_source_out`).  k-fold keeps
    every source represented in every training set, so it is the protocol that
    actually measures interpolation error of the law.  Fold assignment is
    deterministic: rows are ordered by ``(source, v, T, p)`` and dealt round-robin
    into ``k`` folds.
    """
    rows = sorted(tuple(CALIBRATION_ANCHORS if anchors is None else anchors),
                  key=lambda row: (row.source, row.v_m_per_s, row.temperature_c,
                                   row.pressure_mpa))
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(rows) < 2 * len(_FORMS[form]):
        raise ValueError("not enough anchors for the requested k")
    assigned = {id(row): index % k for index, row in enumerate(rows)}
    folds: list[dict[str, Any]] = []
    pooled: list[float] = []
    for fold in range(k):
        train = [row for row in rows if assigned[id(row)] != fold]
        test = [row for row in rows if assigned[id(row)] == fold]
        model = fit_ice_steel_friction(train, form=form)
        table = _error_rows(model, test)
        errors = [row["rel_error"] for row in table]
        pooled.extend(errors)
        folds.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "rows": table,
            "max_rel_error": max(errors),
            "rms_rel_error": float(np.sqrt(np.mean(np.square(errors)))),
            "passes_gate": all(error <= gate for error in errors),
        })
    return {
        "form": form,
        "k": k,
        "gate": gate,
        "folds": folds,
        "pooled_max_rel_error": float(max(pooled)),
        "pooled_rms_rel_error": float(np.sqrt(np.mean(np.square(pooled)))),
        "passes_gate": bool(max(pooled) <= gate),
    }


# ---------------------------------------------------------------------------
# Resistance chain: sensitivity and negative control
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceResistanceCase:
    """Rule-style ice resistance reference case for the sensitivity study.

    ``R = k_c * sigma_c(T) * A_c + mu(v, T, p) * p * A_c``

    * ``A_c = contact_width_m * contact_length_m`` is the nominal hull-ice
      contact patch,
    * ``sigma_c(T)`` comes from
      :class:`tensordem.rudder_ice_load.IceStrengthTemperatureModel`, fitted to
      the Timco & Jordan strength anchors (single source of truth),
    * ``k_c`` is the pressure-area reduction: the nominal patch transmits only
      this fraction of the unconfined crushing strength, which is the usual
      ISO 19906 pressure-area logic for a patch of a few square metres,
    * ``N = p * A_c`` is the normal load the sliding surface carries, so the
      friction resistance is ``mu * N``.

    The absolute resistance of this case is a *reference case* (documented
    coefficients, not a class-society resistance prediction); the sensitivity and
    negative-control statements only use the decomposition and its monotonicity.
    """

    contact_width_m: float = 1.0
    contact_length_m: float = 1.5
    pressure_area_reduction: float = 0.055
    strength: IceStrengthTemperatureModel = field(
        default_factory=lambda: IceStrengthTemperatureModel.from_anchors())

    def __post_init__(self) -> None:
        for name in ("contact_width_m", "contact_length_m", "pressure_area_reduction"):
            value = _check_finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @property
    def contact_area_m2(self) -> float:
        return self.contact_width_m * self.contact_length_m

    def crushing_resistance_n(self, temperature_c: float) -> float:
        return (self.pressure_area_reduction * self.strength.sigma_pa(temperature_c)
                * self.contact_area_m2)

    def normal_load_n(self, pressure_mpa: float) -> float:
        pressure = _check_finite(pressure_mpa, "pressure_mpa")
        if pressure <= 0.0:
            raise ValueError("pressure_mpa must be positive")
        return pressure * 1.0e6 * self.contact_area_m2


def ice_resistance(v: float, temperature_c: float, pressure_mpa: float,
                   model: IceSteelFrictionModel, *,
                   case: IceResistanceCase | None = None,
                   mu_scale: float = 1.0) -> dict[str, Any]:
    """Friction + crushing resistance of one sliding condition (N).

    ``mu_scale`` multiplies the calibrated coefficient, which is how the
    ``+-20 %`` sensitivity study perturbs the tribology without touching
    anything else.
    """
    reference = case or IceResistanceCase()
    scale = _check_finite(mu_scale, "mu_scale")
    if scale <= 0.0:
        raise ValueError("mu_scale must be positive")
    mu = float(model.mu(v, temperature_c, pressure_mpa)) * scale
    normal = reference.normal_load_n(pressure_mpa)
    friction = mu * normal
    crushing = reference.crushing_resistance_n(temperature_c)
    total = crushing + friction
    return {
        "v_m_per_s": float(v),
        "temperature_c": float(temperature_c),
        "pressure_mpa": float(pressure_mpa),
        "mu": mu,
        "mu_scale": scale,
        "normal_load_n": normal,
        "friction_resistance_n": friction,
        "crushing_resistance_n": crushing,
        "total_resistance_n": total,
        "friction_share": friction / total if total > 0.0 else float("nan"),
    }


def operating_envelope(
    speeds: Sequence[float] = (0.01, 0.1, 0.5, 1.0, 2.0),
    temperatures_c: Sequence[float] = (-2.0, -7.0, -15.0, -30.0),
    pressures_mpa: Sequence[float] = (0.1, 0.3, 1.0, 2.0),
) -> list[tuple[float, float, float]]:
    """Cartesian operating envelope used by the sensitivity / control studies."""
    return [(float(v), float(t), float(p))
            for v in speeds for t in temperatures_c for p in pressures_mpa]


def sensitivity_table(model: IceSteelFrictionModel, *,
                      envelope: Iterable[tuple[float, float, float]] | None = None,
                      scales: Sequence[float] = (0.8, 1.0, 1.2),
                      case: IceResistanceCase | None = None) -> dict[str, Any]:
    """``mu`` scaled by ``scales`` over the envelope; resistance must be monotone.

    Physical expectation: raising the friction coefficient can only raise the
    resistance (the friction branch of :func:`ice_resistance` is linear in
    ``mu`` with a non-negative normal load), and the amplitude of the change is
    bounded by the friction share of the total resistance, i.e. ``|dR|/R <=
    |dmu|/mu``.  Both statements are machine-checked here rather than assumed.
    """
    points = list(envelope) if envelope is not None else operating_envelope()
    ordered = sorted(float(scale) for scale in scales)
    rows: list[dict[str, Any]] = []
    monotone = True
    bounded = True
    monotone_inside = True
    for v, temperature, pressure in points:
        totals = [ice_resistance(v, temperature, pressure, model,
                                 case=case, mu_scale=scale)["total_resistance_n"]
                  for scale in ordered]
        steps = [b - a for a, b in zip(totals[:-1], totals[1:])]
        this_monotone = all(step > 0.0 for step in steps)
        base = ice_resistance(v, temperature, pressure, model, case=case,
                              mu_scale=1.0)
        change = (totals[-1] - totals[0]) / base["total_resistance_n"]
        mu_change = (ordered[-1] - ordered[0])
        this_bounded = abs(change) <= mu_change + 1e-12
        monotone = monotone and this_monotone
        bounded = bounded and this_bounded
        inside = bool(model.coverage(v, temperature, pressure)["fully_inside"])
        monotone_inside = monotone_inside and (this_monotone or not inside)
        rows.append({
            "v_m_per_s": v, "temperature_c": temperature, "pressure_mpa": pressure,
            "mu_base": base["mu"],
            "friction_share": base["friction_share"],
            "totals_by_scale": {f"{scale:.2f}": total
                                for scale, total in zip(ordered, totals)},
            "relative_resistance_change": change,
            "monotone_increase": this_monotone,
            "bounded_by_scale_change": this_bounded,
            "inside_calibrated_envelope": inside,
        })
    return {
        "scales": [float(scale) for scale in ordered],
        "n_points": len(rows),
        "rows": rows,
        "all_monotone_increase": bool(monotone),
        "all_monotone_increase_inside_envelope": bool(monotone_inside),
        "n_inside_envelope": int(sum(row["inside_calibrated_envelope"] for row in rows)),
        "all_bounded_by_scale_change": bool(bounded),
        "relative_change_min": min(row["relative_resistance_change"] for row in rows),
        "relative_change_max": max(row["relative_resistance_change"] for row in rows),
        "mean_friction_share": float(np.mean([row["friction_share"] for row in rows])),
    }


def negative_control_table(model: IceSteelFrictionModel, *,
                           envelope: Iterable[tuple[float, float, float]] | None = None,
                           case: IceResistanceCase | None = None,
                           constant_mu: float = MU_CONSTANT_BASELINE) -> dict[str, Any]:
    """Constant ``mu`` versus the calibrated law over the same envelope.

    Quantifies what the hard-coded ``mu = 0.1`` costs: the coefficient error, the
    resistance error and the direction of the bias as a function of speed,
    temperature and pressure.
    """
    baseline = _check_finite(constant_mu, "constant_mu")
    if baseline <= 0.0:
        raise ValueError("constant_mu must be positive")
    points = list(envelope) if envelope is not None else operating_envelope()
    rows: list[dict[str, Any]] = []
    for v, temperature, pressure in points:
        truth = ice_resistance(v, temperature, pressure, model, case=case)
        # rebuild the resistance with the same crushing branch but constant mu
        reference = case or IceResistanceCase()
        constant = dict(truth)
        normal = reference.normal_load_n(pressure)
        constant_friction = baseline * normal
        constant["friction_resistance_n"] = constant_friction
        constant["total_resistance_n"] = (constant["crushing_resistance_n"]
                                          + constant_friction)
        constant["mu"] = baseline
        coverage = model.coverage(v, temperature, pressure)
        floor_bound = bool(float(model.mu_raw(v, temperature, pressure)) < MU_FLOOR)
        rows.append({
            "v_m_per_s": v, "temperature_c": temperature, "pressure_mpa": pressure,
            "mu_model": truth["mu"],
            "mu_constant": baseline,
            "mu_rel_error": (baseline - truth["mu"]) / truth["mu"],
            "resistance_model_n": truth["total_resistance_n"],
            "resistance_constant_n": constant["total_resistance_n"],
            "resistance_rel_error": ((constant["total_resistance_n"]
                                      - truth["total_resistance_n"])
                                     / truth["total_resistance_n"]),
            "friction_resistance_rel_error": ((constant_friction
                                               - truth["friction_resistance_n"])
                                              / truth["friction_resistance_n"]),
            "friction_share_model": truth["friction_share"],
            "inside_calibrated_envelope": bool(coverage["fully_inside"]),
            "power_density_w_per_m2": coverage["power_density_w_per_m2"],
            "floor_bound": floor_bound,
        })
    mu_error = np.array([abs(row["mu_rel_error"]) for row in rows])
    resistance_error = np.array([abs(row["resistance_rel_error"]) for row in rows])
    friction_error = np.array([abs(row["friction_resistance_rel_error"]) for row in rows])
    inside = [row for row in rows if row["inside_calibrated_envelope"]]
    inside_mu = np.array([abs(row["mu_rel_error"]) for row in inside]) if inside else np.array([0.0])
    inside_resistance = (np.array([abs(row["resistance_rel_error"]) for row in inside])
                         if inside else np.array([0.0]))
    return {
        "constant_mu": baseline,
        "n_points": len(rows),
        "rows": rows,
        "max_abs_mu_rel_error": float(mu_error.max()),
        "rms_mu_rel_error": float(np.sqrt(np.mean(mu_error ** 2))),
        "max_abs_resistance_rel_error": float(resistance_error.max()),
        "rms_resistance_rel_error": float(np.sqrt(np.mean(resistance_error ** 2))),
        "max_abs_friction_rel_error": float(friction_error.max()),
        "mean_friction_share": float(np.mean([row["friction_share_model"]
                                              for row in rows])),
        "over_predicts_count": int(np.count_nonzero(
            [row["mu_rel_error"] > 0.0 for row in rows])),
        "in_envelope": {
            "n": len(inside),
            "max_abs_mu_rel_error": float(inside_mu.max()),
            "rms_mu_rel_error": float(np.sqrt(np.mean(inside_mu ** 2))),
            "max_abs_resistance_rel_error": float(inside_resistance.max()),
            "rms_resistance_rel_error": float(np.sqrt(np.mean(inside_resistance ** 2))),
        },
        "n_out_of_envelope": len(rows) - len(inside),
        "n_floor_bound": int(sum(row["floor_bound"] for row in rows)),
        "out_of_envelope_points": [
            {"v_m_per_s": row["v_m_per_s"], "temperature_c": row["temperature_c"],
             "pressure_mpa": row["pressure_mpa"],
             "power_density_w_per_m2": row["power_density_w_per_m2"],
             "floor_bound": row["floor_bound"]}
            for row in rows if not row["inside_calibrated_envelope"]],
    }


# ---------------------------------------------------------------------------
# DEM realisation of the calibrated coefficient
# ---------------------------------------------------------------------------
def dem_friction_probe(mu: float | IceSteelFrictionModel, *,
                       v: float = 0.1, temperature_c: float = -10.0,
                       pressure_mpa: float = 0.5,
                       normal_force: float = 10.0, load_excess: float = 0.6,
                       steps: int = 1200, settle_steps: int = 4000) -> dict[str, Any]:
    """Measure the *effective* Coulomb coefficient realised by the DEM contact.

    A saturated (slip) platen case is run with ``mu_p`` taken from the friction
    law.  The solver then integrates ``m*a = F_h - mu_p*N``, so the effective
    coefficient can be read back out of the measured slip acceleration::

        mu_eff = (F_h - m*a_measured) / N,   m = (F_h - mu_p*N) / a_analytic

    A law that is only a curve -- never pushed through a contact model -- cannot
    be audited this way; this probe is what makes the calibration *DEM* evidence
    rather than curve fitting alone.
    """
    from .platen_friction import platen_friction_case

    if isinstance(mu, IceSteelFrictionModel):
        mu_value = float(mu.mu(v, temperature_c, pressure_mpa))
        source = f"{mu.form} law at {v:g} m/s, {temperature_c:g} degC, {pressure_mpa:g} MPa"
    else:
        mu_value = _check_finite(mu, "mu")
        source = "prescribed coefficient"
    if mu_value <= 0.0:
        raise ValueError("mu must be positive")
    excess = _check_finite(load_excess, "load_excess")
    if excess <= 0.0:
        raise ValueError("load_excess must be positive so the contact slips")
    tangential = (1.0 + excess) * mu_value * abs(normal_force)
    case = platen_friction_case(
        phase="slip", mu_p=mu_value, normal_force=normal_force,
        tangential_force=tangential, drag=0.0, steps=steps,
        settle_steps=settle_steps,
    )
    measured = case["measured"]["velocity_slope_m_per_s2"]
    predicted = case["predicted"]["velocity_slope_m_per_s2"]
    normal = case["normal_force_N"]
    # The analytic slope carries the solver's own mu; invert it for the mass.
    mass = (tangential - mu_value * normal) / predicted if predicted != 0.0 else float("nan")
    mu_effective = (tangential - mass * measured) / normal if math.isfinite(mass) else float("nan")
    return {
        "source": source,
        "v_m_per_s": v,
        "temperature_c": temperature_c,
        "pressure_mpa": pressure_mpa,
        "mu_target": mu_value,
        "mu_effective": mu_effective,
        "mu_rel_error": abs(mu_effective - mu_value) / mu_value,
        "slope_measured": measured,
        "slope_predicted": predicted,
        "slope_rel_error": abs(measured - predicted) / abs(predicted),
        "normal_force_n": normal,
        "tangential_force_n": tangential,
        "passes_dem_gate": bool(abs(measured - predicted) / abs(predicted) <= DEM_GATE),
        "gate": DEM_GATE,
    }


def attach_friction_field(contact: Any, model: IceSteelFrictionModel, *,
                          v: float, temperature_c: float,
                          pressure_mpa: float) -> Any:
    """Return ``contact`` (an ``ice_channel.ContactParameters``) with ``mu`` set.

    The ice-channel contact law reads ``contact.friction`` at every step, so
    replacing the frozen dataclass is the whole coupling: this helper keeps the
    adapter in one place instead of scattering ``mu`` arithmetic across callers.
    Call it whenever the operating point (speed, temperature, pressure) changes
    during a transit; that is how the constant coefficient becomes a *field*.
    """
    if not hasattr(contact, "friction"):
        raise TypeError("contact must expose a 'friction' attribute")
    mu = float(model.mu(v, temperature_c, pressure_mpa))
    return replace(contact, friction=mu)
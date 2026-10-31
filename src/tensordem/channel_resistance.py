"""Riska (1989) brash-ice channel resistance: calibration, convoy energy optimum
and the platform API contract.

Three layers are provided on top of the DEM engine in
:mod:`tensordem.brash_channel`.

``riska_channel_resistance``
    Channel resistance of a hull advancing in a brash-ice channel,

        ``R = C_h * (B h sigma_f) * (C/C_ref)^p * (h/h_ref)^a * (1 + k_v v)``

    ``C_h`` is the dimensionless channel-resistance coefficient; ``p`` and
    ``a`` are the concentration and thickness exponents.  The DEM sweeps decide
    ``C_h``, ``p`` and ``a``.

``calibrate_riska_coefficients``
    Log-linear least squares identification of ``(C_h, p, a)`` from DEM
    measurements, with in-sample and leave-one-out relative errors.

``convoy_energy_metre`` / ``optimal_escort_spacing``
    Energy per metre of an icebreaker + ``n_escort`` convoy,

        ``E(s) = R_lead/eta_lead + n * R_escort(s)/eta_escort``

    ``R_escort(s) = R_ch(B_M) * (1 + alpha (1 - exp(-s/L_c)) + beta exp(-(s-s0)/lam))``

    The first bracket term is *channel consolidation* (a follower arriving
    later meets brash ice that has had time to close up and refreeze), the
    second is the *close-spacing interference penalty* (crowded ice between
    hulls, wake spray).  The two length scales are different, so ``E(s)`` has an
    interior minimum; the optimiser reports the bracket, the curvature and a
    flag that rules out a silent boundary optimum.

``handle_channel_resistance``
    Framework-agnostic handler for ``POST /calculations/channel-resistance``
    (see ``docs/channel-resistance/API_CONTRACT.md``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "RiskaReference",
    "RiskaCalibration",
    "riska_channel_resistance",
    "calibrate_riska_coefficients",
    "spearman_rank_correlation",
    "convoy_energy_metre",
    "optimal_escort_spacing",
    "handle_channel_resistance",
    "DEFAULT_REFERENCE",
]


@dataclass(frozen=True)
class RiskaReference:
    """Reference state of the channel-resistance shape factor."""

    concentration: float = 0.6
    thickness_m: float = 1.0
    sigma_f_pa: float = 0.5e6
    velocity_ms: float = 1.0

    def __post_init__(self) -> None:
        for name in ("concentration", "thickness_m", "sigma_f_pa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.velocity_ms) or self.velocity_ms < 0:
            raise ValueError("velocity_ms must be finite and nonnegative")


DEFAULT_REFERENCE = RiskaReference()


def riska_channel_resistance(
    *,
    beam_m: float,
    thickness_m: float,
    concentration: float,
    velocity_ms: float,
    channel_coefficient: float,
    concentration_exponent: float = 1.0,
    thickness_exponent: float = 1.0,
    velocity_coefficient: float = 0.0,
    reference: RiskaReference = DEFAULT_REFERENCE,
) -> float:
    """Riska-style brash-ice channel resistance ``C_h (B h sigma_f) f`` in N."""
    for name, value in (
        ("beam_m", beam_m), ("thickness_m", thickness_m), ("concentration", concentration),
        ("velocity_ms", velocity_ms), ("channel_coefficient", channel_coefficient),
        ("concentration_exponent", concentration_exponent),
        ("thickness_exponent", thickness_exponent),
        ("velocity_coefficient", velocity_coefficient),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if beam_m <= 0 or thickness_m <= 0 or concentration <= 0:
        raise ValueError("beam, thickness and concentration must be positive")
    if velocity_ms < 0:
        raise ValueError("velocity_ms must be nonnegative")
    if channel_coefficient <= 0:
        raise ValueError("channel_coefficient must be positive")
    shape = (concentration / reference.concentration) ** concentration_exponent
    shape *= (thickness_m / reference.thickness_m) ** thickness_exponent
    shape *= 1.0 + velocity_coefficient * velocity_ms
    return channel_coefficient * beam_m * thickness_m * reference.sigma_f_pa * shape


def _midranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda k: values[k])
    ranks = [0.0] * len(values)
    k = 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and values[order[j + 1]] == values[order[k]]:
            j += 1
        midrank = 0.5 * (k + j) + 1.0
        for m in range(k, j + 1):
            ranks[order[m]] = midrank
        k = j + 1
    return ranks


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho as the Pearson correlation of mid-ranks (no SciPy)."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 2:
        raise ValueError("at least two pairs are required")
    rx, ry = _midranks(x), _midranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0.0 or syy <= 0.0:
        raise ValueError("rank variance is zero; correlation is undefined")
    return sxy / math.sqrt(sxx * syy)


@dataclass
class RiskaCalibration:
    """Result of a log-linear channel-resistance calibration."""

    channel_coefficient: float
    concentration_exponent: float
    thickness_exponent: float
    velocity_coefficient: float = 0.0
    reference: RiskaReference = field(default_factory=RiskaReference)
    cases: list[dict[str, Any]] = field(default_factory=list)
    residual_rms: float = 0.0
    max_relative_error: float = 0.0
    leave_one_out_max_relative_error: float = 0.0
    method: str = "log-linear least squares"

    def predict(
        self,
        *,
        beam_m: float,
        thickness_m: float,
        concentration: float,
        velocity_ms: float,
    ) -> float:
        return riska_channel_resistance(
            beam_m=beam_m,
            thickness_m=thickness_m,
            concentration=concentration,
            velocity_ms=velocity_ms,
            channel_coefficient=self.channel_coefficient,
            concentration_exponent=self.concentration_exponent,
            thickness_exponent=self.thickness_exponent,
            velocity_coefficient=self.velocity_coefficient,
            reference=self.reference,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel_coefficient": self.channel_coefficient,
            "concentration_exponent": self.concentration_exponent,
            "thickness_exponent": self.thickness_exponent,
            "velocity_coefficient": self.velocity_coefficient,
            "reference": {
                "concentration": self.reference.concentration,
                "thickness_m": self.reference.thickness_m,
                "sigma_f_pa": self.reference.sigma_f_pa,
                "velocity_ms": self.reference.velocity_ms,
            },
            "residual_rms": self.residual_rms,
            "max_relative_error": self.max_relative_error,
            "leave_one_out_max_relative_error": self.leave_one_out_max_relative_error,
            "method": self.method,
            "cases": self.cases,
        }


def _solve3(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a 3x3 linear system by Gaussian elimination with pivoting."""
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ValueError("singular normal-equation matrix")
        m[col], m[pivot] = m[pivot], m[col]
        for row in range(col + 1, 3):
            factor = m[row][col] / m[col][col]
            for k in range(col, 4):
                m[row][k] -= factor * m[col][k]
    x = [0.0, 0.0, 0.0]
    for row in (2, 1, 0):
        acc = m[row][3] - sum(m[row][k] * x[k] for k in range(row + 1, 3))
        x[row] = acc / m[row][row]
    return x


def _fit_log_linear(
    rows: Sequence[tuple[float, float, float, float, float]],
    reference: RiskaReference,
    velocity_coefficient: float,
) -> tuple[float, float, float]:
    """Fit log C_h, p, a to ``(ln R, ln(C/Cref), ln(h/href), ln scale)`` rows."""
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0, 0.0, 0.0]
    for ln_r, ln_c, ln_h, ln_scale, _velocity in rows:
        lhs = 1.0 + velocity_coefficient * _velocity
        if lhs <= 0:
            raise ValueError("velocity factor must stay positive")
        target = ln_r - ln_scale - math.log(lhs)
        x = [1.0, ln_c, ln_h]
        for i in range(3):
            atb[i] += x[i] * target
            for j in range(3):
                ata[i][j] += x[i] * x[j]
    ln_ch, p, a = _solve3(ata, atb)
    return math.exp(ln_ch), p, a


def calibrate_riska_coefficients(
    measurements: Iterable[dict[str, Any]],
    *,
    reference: RiskaReference = DEFAULT_REFERENCE,
    velocity_coefficient: float = 0.0,
    free_exponents: bool = True,
) -> RiskaCalibration:
    """Least-squares calibration of ``(C_h, p, a)`` to DEM channel resistance.

    Each measurement is a mapping with keys ``beam_m``, ``thickness_m``,
    ``concentration``, ``velocity_ms`` and ``resistance_n``.  The objective is
    the log-residual ``ln(R_meas) - ln(R_model)``, i.e. relative error is
    minimised uniformly over the sweep.
    """
    cases = [dict(m) for m in measurements]
    if len(cases) < (4 if free_exponents else 1):
        raise ValueError("at least four measurements are required for a three-parameter fit")
    rows: list[tuple[float, float, float, float, float]] = []
    for case in cases:
        resistance = float(case["resistance_n"])
        if not math.isfinite(resistance) or resistance <= 0.0:
            raise ValueError("resistance_n must be finite and positive")
        beam = float(case["beam_m"])
        thickness = float(case["thickness_m"])
        concentration = float(case["concentration"])
        velocity = float(case.get("velocity_ms", reference.velocity_ms))
        if beam <= 0 or thickness <= 0 or concentration <= 0:
            raise ValueError("beam, thickness and concentration must be positive")
        ln_scale = math.log(beam * thickness * reference.sigma_f_pa)
        rows.append(
            (
                math.log(resistance),
                math.log(concentration / reference.concentration),
                math.log(thickness / reference.thickness_m),
                ln_scale,
                velocity,
            )
        )
    if free_exponents:
        coefficient, p, a = _fit_log_linear(rows, reference, velocity_coefficient)
    else:
        coefficient, p, a = _fit_log_linear(rows, reference, velocity_coefficient)
        p, a = 1.0, 1.0
        # Refit the coefficient only, with the exponents fixed at unity.
        num = 0.0
        den = 0.0
        for ln_r, ln_c, ln_h, ln_scale, velocity in rows:
            target = ln_r - ln_scale - ln_c - ln_h - math.log(1.0 + velocity_coefficient * velocity)
            num += target
            den += 1.0
        coefficient = math.exp(num / den)
    calibration = RiskaCalibration(
        channel_coefficient=coefficient,
        concentration_exponent=p,
        thickness_exponent=a,
        velocity_coefficient=velocity_coefficient,
        reference=reference,
    )
    squared: list[float] = []
    worst = 0.0
    fitted_cases: list[dict[str, Any]] = []
    for case in cases:
        velocity = float(case.get("velocity_ms", reference.velocity_ms))
        predicted = calibration.predict(
            beam_m=float(case["beam_m"]),
            thickness_m=float(case["thickness_m"]),
            concentration=float(case["concentration"]),
            velocity_ms=velocity,
        )
        measured = float(case["resistance_n"])
        rel = abs(predicted - measured) / measured
        squared.append(rel**2)
        worst = max(worst, rel)
        fitted_cases.append(
            {
                **{k: case[k] for k in case if k != "resistance_n"},
                "resistance_n": measured,
                "predicted_n": predicted,
                "relative_error": rel,
            }
        )
    calibration.cases = fitted_cases
    calibration.residual_rms = math.sqrt(sum(squared) / len(squared))
    calibration.max_relative_error = worst

    # Leave-one-out stability of the three-parameter fit.
    loo_worst = 0.0
    if len(cases) > 4:
        for drop in range(len(cases)):
            subset = [c for k, c in enumerate(cases) if k != drop]
            try:
                sub = calibrate_riska_coefficients(
                    subset,
                    reference=reference,
                    velocity_coefficient=velocity_coefficient,
                    free_exponents=free_exponents,
                )
            except ValueError:
                loo_worst = float("inf")
                break
            held = cases[drop]
            predicted = sub.predict(
                beam_m=float(held["beam_m"]),
                thickness_m=float(held["thickness_m"]),
                concentration=float(held["concentration"]),
                velocity_ms=float(held.get("velocity_ms", reference.velocity_ms)),
            )
            loo_worst = max(loo_worst, abs(predicted - float(held["resistance_n"])) / float(held["resistance_n"]))
    calibration.leave_one_out_max_relative_error = loo_worst
    return calibration


# --------------------------------------------------------------------- convoy


def convoy_energy_metre(
    *,
    spacing_m: float,
    lead_resistance_n: float,
    escort_channel_resistance_n: float,
    n_escort: int = 2,
    consolidation_gain: float = 0.35,
    consolidation_length_m: float = 6.0,
    interference_penalty: float = 0.6,
    interference_length_m: float = 2.0,
    reference_spacing_m: float = 0.0,
    lead_efficiency: float = 0.30,
    escort_efficiency: float = 0.32,
) -> float:
    """Convoy energy per metre (J/m) at escort spacing ``spacing_m``.

    ``R_escort(s) = R_ch [1 + alpha (1 - exp(-s/L_c)) + beta exp(-(s-s0)/lam)]``
    where ``alpha``/``L_c`` describe channel consolidation with channel age
    ``s/V`` and ``beta``/``lam`` the close-spacing interference penalty.
    ``E(s)`` is the sum of the propulsive energy of the icebreaker and of the
    escorts per unit distance travelled.
    """
    for name, value in (
        ("spacing_m", spacing_m), ("lead_resistance_n", lead_resistance_n),
        ("escort_channel_resistance_n", escort_channel_resistance_n),
        ("consolidation_gain", consolidation_gain),
        ("consolidation_length_m", consolidation_length_m),
        ("interference_penalty", interference_penalty),
        ("interference_length_m", interference_length_m),
        ("lead_efficiency", lead_efficiency), ("escort_efficiency", escort_efficiency),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    if n_escort < 0:
        raise ValueError("n_escort must be nonnegative")
    if consolidation_length_m <= 0 or interference_length_m <= 0:
        raise ValueError("length scales must be positive")
    if not 0.0 < lead_efficiency <= 1.0 or not 0.0 < escort_efficiency <= 1.0:
        raise ValueError("efficiencies must be in (0, 1]")
    if consolidation_gain < 0 or interference_penalty < 0:
        raise ValueError("gains and penalties must be nonnegative")
    consolidation = consolidation_gain * (1.0 - math.exp(-spacing_m / consolidation_length_m))
    interference = interference_penalty * math.exp(
        -(spacing_m - reference_spacing_m) / interference_length_m
    )
    escort_resistance = escort_channel_resistance_n * (1.0 + consolidation + interference)
    energy = lead_resistance_n / lead_efficiency
    energy += n_escort * escort_resistance / escort_efficiency
    return energy


def optimal_escort_spacing(
    *,
    ship_length_m: float,
    lead_resistance_n: float,
    escort_channel_resistance_n: float,
    n_escort: int = 2,
    consolidation_gain: float = 0.35,
    consolidation_length_m: float = 6.0,
    interference_penalty: float = 0.6,
    interference_length_m: float = 2.0,
    lead_efficiency: float = 0.30,
    escort_efficiency: float = 0.32,
    spacing_min_factor: float = 0.5,
    spacing_max_factor: float = 3.0,
    samples: int = 801,
) -> dict[str, Any]:
    """Minimise the convoy energy per metre over ``s in [0.5 L, 3 L]``.

    Returns the interior optimum when it exists, together with the bracket, the
    second derivative and the boundary values so a boundary optimum can never be
    reported as an interior one.
    """
    if ship_length_m <= 0:
        raise ValueError("ship_length_m must be positive")
    s_lo = spacing_min_factor * ship_length_m
    s_hi = spacing_max_factor * ship_length_m
    if s_hi <= s_lo:
        raise ValueError("spacing bracket is empty")

    def energy(s: float) -> float:
        return convoy_energy_metre(
            spacing_m=s,
            lead_resistance_n=lead_resistance_n,
            escort_channel_resistance_n=escort_channel_resistance_n,
            n_escort=n_escort,
            consolidation_gain=consolidation_gain,
            consolidation_length_m=consolidation_length_m,
            interference_penalty=interference_penalty,
            interference_length_m=interference_length_m,
            reference_spacing_m=s_lo,
            lead_efficiency=lead_efficiency,
            escort_efficiency=escort_efficiency,
        )

    grid = [s_lo + (s_hi - s_lo) * k / (samples - 1) for k in range(samples)]
    values = [energy(s) for s in grid]
    best = min(range(samples), key=lambda k: values[k])
    interior = 0 < best < samples - 1
    spacing = grid[best]
    if interior:
        h = grid[1] - grid[0]
        second = (values[best + 1] - 2.0 * values[best] + values[best - 1]) / (h * h)
        # Refine with a parabolic vertex through the three bracketing samples.
        denom = values[best + 1] - 2.0 * values[best] + values[best - 1]
        if abs(denom) > 1e-30:
            spacing = grid[best] + 0.5 * h * (values[best - 1] - values[best + 1]) / denom
        else:
            second = second
    else:
        second = float("nan")
    return {
        "spacing_opt_m": spacing,
        "spacing_bracket_m": [s_lo, s_hi],
        "energy_opt_j_per_m": energy(spacing),
        "energy_at_lower_bound_j_per_m": values[0],
        "energy_at_upper_bound_j_per_m": values[-1],
        "interior": interior,
        "curvature_j_per_m3": second,
        "spacing_opt_over_length": spacing / ship_length_m,
        "grid": {"spacing_m": grid, "energy_j_per_m": values},
        "parameters": {
            "consolidation_gain": consolidation_gain,
            "consolidation_length_m": consolidation_length_m,
            "interference_penalty": interference_penalty,
            "interference_length_m": interference_length_m,
            "n_escort": n_escort,
            "lead_efficiency": lead_efficiency,
            "escort_efficiency": escort_efficiency,
        },
    }


# ------------------------------------------------------------------------ API


def handle_channel_resistance(payload: dict[str, Any]) -> dict[str, Any]:
    """Handler for ``POST /calculations/channel-resistance``.

    Request (all lengths in metres, forces in newtons)::

        {"beam_m": 4.0, "thickness_m": 1.0, "concentration": 0.6,
         "velocity_ms": 0.5, "sigma_f_pa": 5.0e5,
         "channel_coefficient": 0.012, "concentration_exponent": 1.0,
         "thickness_exponent": 1.5,
         "convoy": {"n_escort": 2, "ship_length_m": 2.35,
                    "consolidation_gain": 0.35, "consolidation_length_m": 40.0,
                    "interference_penalty": 0.6, "interference_length_m": 3.0}}

    The response always echoes the model form and the reference state so that a
    downstream operator can reproduce the number.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a mapping")
    beam = float(payload["beam_m"])
    thickness = float(payload["thickness_m"])
    concentration = float(payload["concentration"])
    velocity = float(payload.get("velocity_ms", 0.5))
    sigma_f = float(payload.get("sigma_f_pa", DEFAULT_REFERENCE.sigma_f_pa))
    coefficient = float(payload["channel_coefficient"])
    p = float(payload.get("concentration_exponent", 1.0))
    a = float(payload.get("thickness_exponent", 1.0))
    k_v = float(payload.get("velocity_coefficient", 0.0))
    reference = RiskaReference(
        concentration=DEFAULT_REFERENCE.concentration,
        thickness_m=DEFAULT_REFERENCE.thickness_m,
        sigma_f_pa=sigma_f,
        velocity_ms=DEFAULT_REFERENCE.velocity_ms,
    )
    resistance = riska_channel_resistance(
        beam_m=beam,
        thickness_m=thickness,
        concentration=concentration,
        velocity_ms=velocity,
        channel_coefficient=coefficient,
        concentration_exponent=p,
        thickness_exponent=a,
        velocity_coefficient=k_v,
        reference=reference,
    )
    response: dict[str, Any] = {
        "status": "ok",
        "resistance_n": resistance,
        "resistance_kn": resistance / 1000.0,
        "model": "Riska(1989) brash-ice channel resistance",
        "form": "R = C_h * (B * h * sigma_f) * (C/C_ref)^p * (h/h_ref)^a * (1 + k_v v)",
        "inputs": {
            "beam_m": beam, "thickness_m": thickness, "concentration": concentration,
            "velocity_ms": velocity, "sigma_f_pa": sigma_f,
            "channel_coefficient": coefficient, "concentration_exponent": p,
            "thickness_exponent": a, "velocity_coefficient": k_v,
        },
        "reference": {
            "concentration": reference.concentration,
            "thickness_m": reference.thickness_m,
            "sigma_f_pa": reference.sigma_f_pa,
        },
        "limitations": [
            "calibrated against the 2-D plan-view brash-channel DEM in this repository",
            "valid for 0.25 <= C <= 0.7 and 0.3 m <= h <= 2.0 m",
            "velocity dependence is linear and weak (k_v assumed, not fitted)",
        ],
    }
    convoy = payload.get("convoy")
    if convoy:
        if not isinstance(convoy, dict):
            raise ValueError("convoy must be a mapping")
        spacing = float(convoy.get("spacing_m", 0.0))
        ship_length = float(convoy["ship_length_m"])
        params = dict(
            ship_length_m=ship_length,
            lead_resistance_n=float(convoy.get("lead_resistance_n", resistance)),
            escort_channel_resistance_n=float(convoy.get("escort_channel_resistance_n", resistance)),
            n_escort=int(convoy.get("n_escort", 2)),
            consolidation_gain=float(convoy.get("consolidation_gain", 0.35)),
            consolidation_length_m=float(convoy.get("consolidation_length_m", 6.0)),
            interference_penalty=float(convoy.get("interference_penalty", 0.6)),
            interference_length_m=float(convoy.get("interference_length_m", 2.0)),
        )
        optimum = optimal_escort_spacing(**params)
        optimum.pop("grid", None)
        if spacing > 0:
            optimum["energy_at_requested_spacing_j_per_m"] = convoy_energy_metre(
                spacing_m=spacing,
                lead_resistance_n=params["lead_resistance_n"],
                escort_channel_resistance_n=params["escort_channel_resistance_n"],
                n_escort=params["n_escort"],
                consolidation_gain=params["consolidation_gain"],
                consolidation_length_m=params["consolidation_length_m"],
                interference_penalty=params["interference_penalty"],
                interference_length_m=params["interference_length_m"],
                reference_spacing_m=spacing_min_factor_of(params),
            )
        response["convoy"] = optimum
    return response


def spacing_min_factor_of(params: dict[str, Any]) -> float:
    """Lower bracket of the escort spacing search (0.5 ship length)."""
    return 0.5 * float(params["ship_length_m"])
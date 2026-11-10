"""Recalibration of the collision coefficient ``C`` of ``V_damage = C*M*V^2/sigma_y``.

Minorsky (1959) fixed ``E[MJ] = 47.1*R_T[m^3] + 0.5`` on 26 full-scale ship-ship
collisions of 1950s mild steel.  N01 inherited the coefficient from that line
(``C = sigma_y_ref/(2*47.1e6)``) with a ``1/sigma_y`` scaling and *no*
correction for the energy that is not absorbed by the steel.

This module generalises the coefficient to

    C(xi, ice face) = [E_steel / E_k] * sigma_y_ref / (2*47.1e6) * kappa(shape)

i.e. the fraction of the collision energy that reaches the plastically deformed
steel (the rest is ice crushing and friction) multiplies the Minorsky constant.
Equivalently, the *model-implied* coefficient of a solved case is recovered by
inverting the volume relation

    C_model = V_model * sigma_y / (M * V_impact^2),

which is the quantity this module reports for every case (``recalibrated
coefficient``).  The two reduce to N01's constant when the ice side is switched
off and the damaged volume equals the Minorsky volume.

Dataset discipline
------------------
The empirical residual of a recalibration can only be computed on a data set.
The Minorsky (1959) case table is **not present in this repository** and no
network access is available in the build environment, so
:func:`load_collision_dataset` reports
:data:`DATASET_STATUS_WITHHELD` with ``n = 0`` instead of inventing numbers.
A caller can drop a CSV with the documented columns (see
:data:`DATASET_COLUMNS`) into ``data/`` and the same code path will produce the
fitted coefficient and the median relative residual; the schema is enforced and
the regime column keeps the ship-ship baseline subset and the ice-region subset
apart (they are **never** pooled - a pooled fit mixes two different damage
mechanics and is exactly the error the node is meant to avoid).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .minorsky import MINORSKY_1959_MJ_PER_M3, REFERENCE_YIELD_STRESS_PA

__all__ = [
    "DATASET_COLUMNS",
    "DATASET_STATUS_WITHHELD",
    "DATASET_STATUS_PRESENT",
    "ship_ship_regime",
    "ice_regime",
    "collision_coefficient_from_share",
    "collision_coefficient_from_volume",
    "energy_ledger",
    "load_collision_dataset",
    "recalibrate_coefficient",
    "coefficient_uncertainty",
]

DATASET_COLUMNS = (
    "case",
    "regime",            # "ship_ship" (Minorsky baseline) or "ice"
    "mass_kg",
    "velocity_m_s",
    "sigma_y_Pa",
    "volume_m3",         # measured damaged steel volume
    "energy_J",          # measured/estimated energy loss (optional, checked if present)
)

DATASET_STATUS_WITHHELD = "WITHHELD"
DATASET_STATUS_PRESENT = "PRESENT"

MINORSKY_BASELINE_MJ_PER_M3 = MINORSKY_1959_MJ_PER_M3


def _real(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def ship_ship_regime() -> str:
    return "ship_ship"


def ice_regime() -> str:
    return "ice"


# ---------------------------------------------------------------------------
# coefficient definitions
# ---------------------------------------------------------------------------
def collision_coefficient_from_share(energy_share: float, sigma_y: float) -> float:
    """``C = psi*sigma_y/(2*47.1e6)``: the ice-aware generalisation of N01's C.

    ``energy_share = E_steel/E_k`` (``1.0`` for the rigid indenter, i.e. N01),
    ``sigma_y`` in Pa.  ``C`` is non-dimensional and multiplies ``M*V^2/sigma_y``
    to give the damaged steel volume.
    """
    psi = _real("energy_share", energy_share, nonnegative=True)
    sy = _real("sigma_y", sigma_y, positive=True)
    return psi * sy / (2.0 * MINORSKY_BASELINE_MJ_PER_M3 * 1.0e6)


def collision_coefficient_from_volume(
    volume_m3: float, mass: float, velocity: float, sigma_y: float
) -> float:
    """Invert ``V = C*M*V^2/sigma_y`` on a solved case: ``C = V*sigma_y/(M*V^2)``."""
    v = _real("volume_m3", volume_m3, nonnegative=True)
    m = _real("mass", mass, positive=True)
    u = _real("velocity", velocity, positive=True)
    sy = _real("sigma_y", sigma_y, positive=True)
    return v * sy / (m * u * u)


def energy_ledger(solution, *, effective_mass: Optional[float] = None) -> dict:
    """Energy ledger of an :class:`~tensordem.hull_collision.indentation.IndentationSolution`.

    Closes the books explicitly: ``E_k = E_steel + E_friction + E_ice + residual``
    and reports the shares (the share is what enters ``C(xi, ice)``).
    """
    e_k = _real("kinetic_energy", solution.kinetic_energy, nonnegative=True)
    structural = _real("structural_energy", solution.structural_energy, nonnegative=True)
    friction = _real("friction_energy", solution.friction_energy, nonnegative=True)
    ice = _real("ice_energy", solution.ice_energy, nonnegative=True)
    residual = solution.residual
    share = structural / e_k if e_k > 0.0 else 0.0
    out = {
        "kinetic_energy_MJ": e_k / 1e6,
        "steel_MJ": structural / 1e6,
        "friction_MJ": friction / 1e6,
        "ice_MJ": ice / 1e6,
        "residual_MJ": residual / 1e6,
        "residual_relative": abs(residual) / e_k if e_k > 0.0 else 0.0,
        "steel_share": share,
        "friction_share": friction / e_k if e_k > 0.0 else 0.0,
        "ice_share": ice / e_k if e_k > 0.0 else 0.0,
        "closes": abs(residual) <= 1e-6 * e_k if e_k > 0.0 else True,
    }
    return out


# ---------------------------------------------------------------------------
# empirical recalibration (needs a data set; withheld when absent)
# ---------------------------------------------------------------------------
def load_collision_dataset(path: Path | str) -> dict:
    """Load a collision case table, or report it WITHHELD.

    Returns ``{"status": ..., "cases": [...], "n": ..., "path": ..., "reason": ...}``.
    A missing file is *not* an error: the node must be able to state honestly
    that the empirical residual could not be computed.
    """
    p = Path(path)
    if not p.exists():
        return {
            "status": DATASET_STATUS_WITHHELD,
            "cases": [],
            "n": 0,
            "path": str(p),
            "reason": "collision case table not present in the repository and no network "
            "access in the build environment; no numbers are invented",
        }
    cases = []
    with p.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in DATASET_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"dataset {p} is missing columns {missing}")
        for row in reader:
            cases.append(
                {
                    "case": row["case"],
                    "regime": row["regime"],
                    "mass_kg": float(row["mass_kg"]),
                    "velocity_m_s": float(row["velocity_m_s"]),
                    "sigma_y_Pa": float(row["sigma_y_Pa"]),
                    "volume_m3": float(row["volume_m3"]),
                    "energy_J": float(row["energy_J"]) if row.get("energy_J") else None,
                }
            )
    return {
        "status": DATASET_STATUS_PRESENT,
        "cases": cases,
        "n": len(cases),
        "path": str(p),
        "reason": "",
    }


def recalibrate_coefficient(
    cases: Sequence[dict], *, regime: str, threshold: float = 0.15
) -> dict:
    """Least-squares-in-relative-error fit of ``C`` on one regime subset.

    The fit minimises ``sum_i (V_i/(C*x_i) - 1)^2`` with ``x_i = M_i*V_i^2/sigma_y,i``
    (closed form ``C = sum u_i^2 / sum u_i``, ``u_i = V_i/x_i``), and reports the
    **median** absolute relative residual, which is the robust summary a
    recalibration should be judged on.  Regimes are never pooled: the caller
    passes one subset at a time and the ice subset reports its own ``n``.
    """
    if not isinstance(regime, str) or not regime:
        raise ValueError("regime must be a non-empty string")
    subset = [c for c in cases if c.get("regime") == regime]
    if not subset:
        return {
            "regime": regime,
            "status": DATASET_STATUS_WITHHELD,
            "n": 0,
            "coefficient": None,
            "median_relative_residual": None,
            "max_relative_residual": None,
            "meets_threshold": None,
            "threshold": threshold,
            "residuals": [],
        }
    u = []
    for case in subset:
        m = _real("mass_kg", case["mass_kg"], positive=True)
        v = _real("velocity_m_s", case["velocity_m_s"], positive=True)
        sy = _real("sigma_y_Pa", case["sigma_y_Pa"], positive=True)
        vol = _real("volume_m3", case["volume_m3"], nonnegative=True)
        x = m * v * v / sy
        u.append(vol / x)
    sum_u = math.fsum(u)
    sum_u2 = math.fsum(value * value for value in u)
    coefficient = sum_u2 / sum_u if sum_u else 0.0
    residuals = [value / coefficient - 1.0 for value in u]
    ordered = sorted(abs(value) for value in residuals)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2 == 1
        else 0.5 * (ordered[mid - 1] + ordered[mid])
    )
    return {
        "regime": regime,
        "status": DATASET_STATUS_PRESENT,
        "n": len(subset),
        "coefficient": coefficient,
        "median_relative_residual": median,
        "max_relative_residual": ordered[-1],
        "meets_threshold": median <= threshold,
        "threshold": threshold,
        "residuals": residuals,
        "reference_minorsky_coefficient": REFERENCE_YIELD_STRESS_PA
        / (2.0 * MINORSKY_BASELINE_MJ_PER_M3 * 1e6),
    }


def coefficient_uncertainty(samples: Iterable[float]) -> dict:
    """Spread (min/median/max) of a coefficient sample set - the reported envelope."""
    values = sorted(_real("sample", v, positive=True) for v in samples)
    if not values:
        raise ValueError("at least one sample is required")
    n = len(values)
    mid = n // 2
    median = values[mid] if n % 2 == 1 else 0.5 * (values[mid - 1] + values[mid])
    return {
        "n": n,
        "min": values[0],
        "median": median,
        "max": values[-1],
        "relative_half_width": 0.5 * (values[-1] - values[0]) / median if median else None,
        "ratio_max_over_min": values[-1] / values[0] if values[0] else None,
    }
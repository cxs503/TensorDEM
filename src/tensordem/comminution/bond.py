"""Bond's third theory of comminution and the Bond grindability test.

Third theory
------------
Bond (1952) postulates that the useful work in size reduction is proportional to
the new crack length produced, which for geometrically similar breakage gives

.. math:: E = 10\\,W_i\\left(\\frac{1}{\\sqrt{P_{80}}}
                            -\\frac{1}{\\sqrt{F_{80}}}\\right)

with ``E`` the specific energy [kWh/t], ``W_i`` the Bond work index [kWh/t] and
``F80``/``P80`` the 80%-passing sizes of feed and product in micrometres.  The
factor 10 is what makes ``W_i`` the energy needed to reduce a theoretically
infinite feed to 80% passing 100 um (put ``F80 -> inf``, ``P80 = 100``:
``E = 10 W_i / 10 = W_i``); :func:`bond_work_index_from_energy` and the tests
use that identity rather than re-typing the constant.

Unit convention (important, a 10% trap)
---------------------------------------
The original equation is in **short tons** (2000 lb = 0.9072 t), so a work index
quoted in kWh/short-ton must be multiplied by ``2000/2204.62 = 0.9072`` before it
is used with metric kWh/t -- or equivalently the constant 10 becomes
``10/sqrt(1.1023) = 9.526``.  This module works in **metric kWh/t** throughout,
the convention of the platform's acceptance criterion
``E = 10 Wi (1/sqrt(P80) - 1/sqrt(F80))``, and exposes
:data:`SHORT_TON_TO_TONNE` so a caller holding short-ton indices can convert
explicitly instead of silently absorbing a 10% error.

Grindability test
-----------------
Bond's locked-cycle ball mill grindability test gives the work index from the
measured net grams per revolution ``G`` at the closing screen ``P_c``:

.. math:: W_i = \\frac{44.5}{P_c^{0.23}\\,G^{0.82}\\
                  \\left(\\frac{10}{\\sqrt{P_{80}}}-\\frac{10}{\\sqrt{F_{80}}}\\right)}

(metric form, ``P_c``, ``F80``, ``P80`` in um, ``G`` in g/rev).  It is a second,
independent published route to ``W_i`` from *experimental* numbers, which is why
it is here: it lets a report close the loop "laboratory grindability test ->
work index -> third-theory specific energy" without any hand-entered index.

Work index database
-------------------
:data:`BOND_WORK_INDEX_DB` collects literature work indices with **explicit
bands**.  Bond work indices are material indices with a real ore-to-ore spread
of roughly +/-15%, so every entry carries ``typical``, ``low``, ``high`` and a
``source`` note; comparisons in the acceptance study are made against the band
as well as the midpoint, never against a single unqualified digit.
"""
from __future__ import annotations

import math

__all__ = [
    "BOND_GRINDABILITY_CONSTANT",
    "BOND_THIRD_THEORY_CONSTANT",
    "BOND_WORK_INDEX_DB",
    "SHORT_TON_TO_TONNE",
    "bond_product_size_um",
    "bond_specific_energy_kwh_t",
    "bond_work_index_from_energy",
    "bond_work_index_from_grindability",
    "specific_energy_kwh_t",
    "work_index_band",
]

#: Multiplier of the third theory: ``E = 10 Wi (1/sqrt(P80) - 1/sqrt(F80))``.
BOND_THIRD_THEORY_CONSTANT = 10.0

#: Empirical constant of the metric grindability-test equation.
BOND_GRINDABILITY_CONSTANT = 44.5

#: 1 short ton in tonnes; a short-ton work index times this is the metric one.
SHORT_TON_TO_TONNE = 2000.0 / 2204.6226218487757

#: Literature work indices [kWh/t, metric], with explicit bands.
#:
#: ``typical`` is the tabulated value used in the acceptance study, ``low`` and
#: ``high`` bound the ore-to-ore spread quoted in the same tables.  These are
#: *material* indices: a plant-specific grindability test (see
#: :func:`bond_work_index_from_grindability`) is the only way to do better, and
#: the spread is the dominant uncertainty of any Bond-law comparison.
BOND_WORK_INDEX_DB = {
    "limestone": {
        "typical": 12.0, "low": 11.6, "high": 13.0,
        "source": "Bond (1952)/Rowland tabulated sedimentary carbonate range; "
                  "cement-industry practice quotes 10.5-13.5",
    },
    "quartz": {
        "typical": 13.0, "low": 12.8, "high": 14.0,
        "source": "Bond (1952) standard material; quartz is the reference hard "
                  "mineral in most work-index tables",
    },
    "granite": {
        "typical": 15.0, "low": 14.0, "high": 16.0,
        "source": "Bond (1952) igneous reference material range",
    },
    "cement_clinker": {
        "typical": 13.5, "low": 13.0, "high": 14.0,
        "source": "cement grinding practice (closed-circuit ball mills), "
                  "narrow band because clinker is a manufactured feed",
    },
    "copper_porphyry_ore": {
        "typical": 13.0, "low": 12.0, "high": 14.5,
        "source": "SAG/ball circuit design practice for porphyry Cu ores",
    },
}


def _positive(value, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return v


def bond_specific_energy_kwh_t(work_index: float, f80_um: float, p80_um: float) -> float:
    """Bond third-theory specific energy [kWh/t] (metric tonnes).

    ``E = 10 Wi (1/sqrt(P80) - 1/sqrt(F80))``; ``F80`` may be ``math.inf``
    (infinite feed), in which case ``E = 10 Wi / sqrt(P80)``.
    """
    wi = _positive(work_index, "work_index")
    p80 = _positive(p80_um, "p80_um")
    f80 = float(f80_um)
    if not (math.isinf(f80) or f80 > 0.0) or math.isnan(f80):
        raise ValueError("f80_um must be positive or +inf")
    if math.isfinite(f80) and f80 <= p80:
        raise ValueError("F80 must be larger than P80 for a reduction operation")
    inv_f = 0.0 if math.isinf(f80) else 1.0 / math.sqrt(f80)
    return BOND_THIRD_THEORY_CONSTANT * wi * (1.0 / math.sqrt(p80) - inv_f)


def bond_product_size_um(work_index: float, f80_um: float, energy_kwh_t: float) -> float:
    """Invert the third theory: ``P80`` reached after ``E`` [kWh/t].

    ``1/sqrt(P80) = E/(10 Wi) + 1/sqrt(F80)``.
    """
    wi = _positive(work_index, "work_index")
    e = float(energy_kwh_t)
    if not math.isfinite(e) or e <= 0.0:
        raise ValueError("energy_kwh_t must be finite and positive")
    inv_f = 0.0 if math.isinf(float(f80_um)) else 1.0 / math.sqrt(_positive(f80_um, "f80_um"))
    inv_p = e / (BOND_THIRD_THEORY_CONSTANT * wi) + inv_f
    if inv_p <= 0.0:  # pragma: no cover - defensive
        raise ValueError("inversion produced a non-positive root")
    return 1.0 / (inv_p * inv_p)


def bond_work_index_from_energy(f80_um: float, p80_um: float,
                                energy_kwh_t: float) -> float:
    """Back out ``Wi`` from a measured specific energy (inverse third theory)."""
    e = float(energy_kwh_t)
    if not math.isfinite(e) or e <= 0.0:
        raise ValueError("energy_kwh_t must be finite and positive")
    return e / bond_specific_energy_kwh_t(1.0, f80_um, p80_um)


def bond_work_index_from_grindability(g_per_rev: float, closing_screen_um: float,
                                      f80_um: float, p80_um: float,
                                      constant: float = BOND_GRINDABILITY_CONSTANT) -> float:
    """Bond locked-cycle grindability test -> work index [kWh/t] (metric form).

    ``G`` is the net grams of minus-``closing_screen`` product produced per mill
    revolution at the test's circulating load (250% plus the test's own
    screening cycle).
    """
    g = _positive(g_per_rev, "g_per_rev")
    pc = _positive(closing_screen_um, "closing_screen_um")
    denom = (pc ** 0.23) * (g ** 0.82) * bond_specific_energy_kwh_t(
        1.0, f80_um, p80_um)
    return float(constant) / denom


def work_index_band(material: str) -> dict:
    """Database entry for ``material`` (raises with the known keys if unknown)."""
    key = str(material).strip().lower().replace(" ", "_")
    if key not in BOND_WORK_INDEX_DB:
        raise KeyError(f"unknown material {material!r}; known: {sorted(BOND_WORK_INDEX_DB)}")
    return dict(BOND_WORK_INDEX_DB[key], material=key)


def specific_energy_kwh_t(power_w: float, duration_s: float, mass_kg: float) -> float:
    """Specific energy [kWh/t] deposited by ``power_w`` for ``duration_s`` into
    a ``mass_kg`` batch."""
    p = float(power_w)
    t = float(duration_s)
    m = _positive(mass_kg, "mass_kg")
    if not math.isfinite(p) or p < 0.0:
        raise ValueError("power_w must be finite and non-negative")
    if not math.isfinite(t) or t < 0.0:
        raise ValueError("duration_s must be finite and non-negative")
    return p * t / (m / 1000.0) / 3.6e6
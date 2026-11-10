"""Salinity / temperature / porosity strength model for sea ice (MYI vs FYI).

Physical picture
----------------
Sea ice is a two-phase material: a pure-ice matrix containing liquid brine
pockets whose salinity is set by the sea-ice liquidus.  The *bulk* salinity
``S`` of the ice is a conserved, measurable quantity (ice cores), and the
brine volume fraction ``nu_b`` follows from a mass balance once the brine
salinity ``S_b(T)`` is known:

    nu_b = (rho_i / rho_b) * S / S_b(T)                        (1)

``nu_b`` is the single state variable that controls the macroscopic strength:
brine pockets are the weak phase, so compressive strength collapses as
``nu_b`` approaches the porosity at which the ice matrix loses continuity.

The chain implemented here is

    T  --(liquidus)-->  S_b  --(mass balance)-->  nu_b  --(porosity law)-->  sigma_c

Each link is a published relation:

* ``S_b(T)``   -- Cox & Weeks (1983) sea-ice liquidus (freezing-point
  polynomial for seawater, inverted for brine salinity).
* ``nu_b``     -- Cox & Weeks (1983) brine-volume relation (mass balance with
  brine density ``rho_b = 1 + 0.8 S_b/1000`` g/cm^3, Weeks 2010 eq. 2.9).
* ``sigma_c``  -- Timco & Frederking (1984) uniaxial compressive strength as a
  function of brine volume fraction, written in the porosity-law form
  ``sigma_c = sigma_0 (1 - nu_b/nu_p)^m`` (Timco & Weeks 2010, Fig. 9 / eq. 21
  discussion).

Multi-year (MY) versus first-year (FY) ice
------------------------------------------
MY ice is *differentiated* from FY ice by two independent mechanisms, both of
which raise the compressive strength:

1. **Desalination (dominant).**  Repeated summer melt / winter freeze drains
   brine from the freeboard.  Typical bulk salinity drops from
   ``S_FY ~ 5-10 ppt`` to ``S_MY ~ 0.5-2 ppt`` (Cox & Weeks 1983; Weeks 2010,
   Ch. 2).  At equal temperature this lowers ``nu_b`` by the salinity ratio and
   therefore raises ``sigma_c`` through (1).
2. **Pore structure (secondary).**  MY ice has coarser columnar grains and
   fewer, more isolated brine pockets, so the same ``nu_b`` weakens the matrix
   less.  This is carried by a structural factor
   ``kappa_MY = 1.15 +- 0.05`` (engineering estimate, see
   :data:`DEFAULT_PARAMETERS`).

Because of (1)+(2), ``sigma_c_MY(T, S) > sigma_c_FY(T, S)`` holds *at equal
bulk (T, S)* through ``kappa_MY`` and the observed MY/FY strength ratio for
*typical* salinities is ``~1.3`` (lab-scale) to ``~2`` (field).
See the report's "criterion conflict" note: the campaign metadata asks for
``sigma_c_MY < sigma_c_FY`` at equal conditions, which contradicts both this
model's specification path (3) and the literature; the conflict is recorded,
not papered over.

Scope limits (stated, not hidden)
---------------------------------
* No strain-rate dependence (the Timco & Frederking data are for
  ``1e-3 1/s``); the model is a *quasi-static, high-strain-rate* strength
  envelope.  Ductile-regime strengths are lower by a factor ~2.
* No loading-direction anisotropy (columnar ice is weaker across the columns).
* No gas (air) porosity; MY ice is in reality less dense (0.87-0.90 g/cm^3),
  which acts opposite to the desalination trend and is listed as a limitation.
* The liquidus is the seawater quadratic; above ~200 ppt the real liquidus
  bends at precipitation lines (Na2SO4.10H2O at -8 C, NaCl.2H2O at -22.9 C).
  This makes low-temperature (T < -23 C) strength slightly over-predicted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "RHO_ICE_G_CM3",
    "LIQUIDUS_A1",
    "LIQUIDUS_A2",
    "StrengthParameters",
    "DEFAULT_PARAMETERS",
    "PARAMETER_PROVENANCE",
    "ICECLASS_DEFAULT_SALINITY",
    "brine_salinity",
    "brine_density",
    "brine_volume_fraction",
    "porosity_strength_factor",
    "compressive_strength",
    "strength_surface",
    "temperature_scan",
    "salinity_scan",
    "spearman_rho",
    "monte_carlo_strength",
]

# --- physical constants -----------------------------------------------------
#: Pure-ice density [g/cm^3] (Weeks 2010, Table 1.1 for bubble-free ice).
RHO_ICE_G_CM3 = 0.917

#: Cox & Weeks (1983) seawater freezing-point polynomial
#: ``T_f = -(A1*S + A2*S^2)``, T_f in degC, S in ppt.
LIQUIDUS_A1 = 0.05411
LIQUIDUS_A2 = 7.594e-5

#: Brine-density slope [g/cm^3 per ppt] in ``rho_b = 1 + B*S_b/1000``
#: (Weeks 2010, eq. 2.9; linear fit to brine densities at low temperature).
BRINE_DENSITY_SLOPE = 0.8

#: Typical bulk salinities [ppt] used for the MYI/FYI differentiation.
ICECLASS_DEFAULT_SALINITY = {"FY": 5.0, "MY": 1.0}


@dataclass(frozen=True)
class StrengthParameters:
    """Parameters of the Timco-Frederking porosity strength law.

    ``sigma_c = kappa * sigma_0 * (1 - nu_b/nu_p)^m`` for ``nu_b < nu_p``,
    else zero.
    """

    sigma0_pa: float = 5.5e6
    porosity_limit: float = 0.27
    porosity_exponent: float = 1.5
    myi_structural_factor: float = 1.15

    def as_dict(self) -> dict[str, float]:
        return {
            "sigma0_pa": self.sigma0_pa,
            "porosity_limit": self.porosity_limit,
            "porosity_exponent": self.porosity_exponent,
            "myi_structural_factor": self.myi_structural_factor,
        }


DEFAULT_PARAMETERS = StrengthParameters()

#: Item-by-item provenance for every number the model contains.
#: ``value`` is the nominal value, ``range`` the documented uncertainty band.
PARAMETER_PROVENANCE: dict[str, dict[str, Any]] = {
    "liquidus_a1": {
        "value": LIQUIDUS_A1,
        "unit": "degC/ppt",
        "source": "Cox & Weeks (1983) seawater freezing-point polynomial, "
                  "linear coefficient (also Weeks 2010, eq. 2.6)",
    },
    "liquidus_a2": {
        "value": LIQUIDUS_A2,
        "unit": "degC/ppt^2",
        "source": "Cox & Weeks (1983) freezing-point polynomial, quadratic coefficient",
    },
    "brine_density_slope": {
        "value": BRINE_DENSITY_SLOPE,
        "unit": "g/cm^3/ppt",
        "source": "Weeks (2010) eq. 2.9 linear brine-density fit",
    },
    "rho_ice": {
        "value": RHO_ICE_G_CM3,
        "unit": "g/cm^3",
        "source": "Weeks (2010) Table 1.1, bubble-free ice density",
    },
    "sigma0_pa": {
        "value": DEFAULT_PARAMETERS.sigma0_pa,
        "range": [5.0e6, 8.0e6],
        "unit": "Pa",
        "source": "Zero-porosity columnar freshwater ice strength at -10 degC and "
                  "strain rate 1e-3 1/s (Timco & Weeks 2010, Sec. 5 review range "
                  "5-8 MPa); nominal = low end to stay conservative",
    },
    "porosity_limit": {
        "value": DEFAULT_PARAMETERS.porosity_limit,
        "range": [0.25, 0.30],
        "unit": "-",
        "source": "Brine volume fraction at which sea-ice compressive strength "
                  "vanishes (Timco & Weeks 2010, Fig. 9 compilation)",
    },
    "porosity_exponent": {
        "value": DEFAULT_PARAMETERS.porosity_exponent,
        "range": [1.0, 2.0],
        "unit": "-",
        "source": "Curvature of sigma_c(nu_b) in the Timco & Frederking (1984) "
                  "data compilation (Timco & Weeks 2010)",
    },
    "myi_structural_factor": {
        "value": DEFAULT_PARAMETERS.myi_structural_factor,
        "range": [1.10, 1.20],
        "unit": "-",
        "source": "ENGINEERING ESTIMATE (not a fitted constant): strength ratio of "
                  "MY over FY ice at equal brine volume, attributed to coarser "
                  "grains / isolated pores; bracketed by the MY/FY strength ranges "
                  "of Timco & Weeks (2010) after removing the salinity effect",
    },
    "myi_salinity_ppt": {
        "value": ICECLASS_DEFAULT_SALINITY["MY"],
        "range": [0.5, 2.0],
        "unit": "ppt",
        "source": "Desalinated multi-year ice bulk salinity (Weeks 2010, Ch. 2)",
    },
    "fyi_salinity_ppt": {
        "value": ICECLASS_DEFAULT_SALINITY["FY"],
        "range": [3.0, 10.0],
        "unit": "ppt",
        "source": "First-year ice bulk salinity (Weeks 2010, Ch. 2)",
    },
}


# ---------------------------------------------------------------------------
# 1. Brine salinity from the sea-ice liquidus
# ---------------------------------------------------------------------------
def brine_salinity(temperature_c: float) -> float:
    """Brine salinity ``S_b`` [ppt] in equilibrium with ice at ``T`` [degC].

    Inverts the Cox & Weeks (1983) seawater freezing-point polynomial
    ``T_f = -(A1*S + A2*S^2)``.

    ``T >= 0`` returns ``0.0`` (no brine: all liquid).  The function is
    strictly increasing as ``T`` decreases, which is what makes the brine
    volume fraction, and hence the strength, monotone in temperature.
    """
    t = float(temperature_c)
    if not np.isfinite(t):
        raise ValueError("temperature_c must be finite")
    if t >= 0.0:
        return 0.0
    disc = LIQUIDUS_A1 * LIQUIDUS_A1 - 4.0 * LIQUIDUS_A2 * t
    return float((-LIQUIDUS_A1 + np.sqrt(disc)) / (2.0 * LIQUIDUS_A2))


def brine_density(brine_salinity_ppt: float) -> float:
    """Brine density [g/cm^3] from ``rho_b = 1 + 0.8 S_b/1000``."""
    return float(1.0 + BRINE_DENSITY_SLOPE * max(0.0, float(brine_salinity_ppt)) / 1000.0)


# ---------------------------------------------------------------------------
# 2. Brine volume fraction (Cox & Weeks 1983 mass balance)
# ---------------------------------------------------------------------------
def brine_volume_fraction(temperature_c: float, bulk_salinity_ppt: float) -> float:
    """Brine volume fraction ``nu_b`` [-].

    ``nu_b = (rho_i/rho_b) * S / S_b(T)``.

    Limits: ``S = 0`` gives ``nu_b = 0`` exactly (no salt, no brine);
    ``T >= 0`` gives ``nu_b = 1`` (fully liquid).  For ``T`` slightly below
    zero ``nu_b`` grows without bound and the caller is expected to treat
    ``nu_b >= 1`` as "no load-bearing ice matrix".
    """
    s = float(bulk_salinity_ppt)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("bulk_salinity_ppt must be finite and >= 0")
    t = float(temperature_c)
    if not np.isfinite(t):
        raise ValueError("temperature_c must be finite")
    if s == 0.0:
        return 0.0
    if t >= 0.0:
        return 1.0
    sb = brine_salinity(t)
    rb = brine_density(sb)
    return float((RHO_ICE_G_CM3 / rb) * s / sb)


# ---------------------------------------------------------------------------
# 3. Porosity -> strength (Timco & Frederking 1984)
# ---------------------------------------------------------------------------
def porosity_strength_factor(
    nu_b: float,
    *,
    porosity_limit: float = DEFAULT_PARAMETERS.porosity_limit,
    porosity_exponent: float = DEFAULT_PARAMETERS.porosity_exponent,
) -> float:
    """Normalised strength ``sigma_c/sigma_0 = (1 - nu_b/nu_p)^m``, clipped at 0."""
    nu = float(nu_b)
    lim = float(porosity_limit)
    if lim <= 0.0:
        raise ValueError("porosity_limit must be > 0")
    ratio = 1.0 - nu / lim
    if ratio <= 0.0:
        return 0.0
    return float(ratio ** float(porosity_exponent))


def compressive_strength(
    temperature_c: float,
    bulk_salinity_ppt: float,
    *,
    ice_class: str = "FY",
    params: StrengthParameters = DEFAULT_PARAMETERS,
) -> float:
    """Uniaxial compressive strength ``sigma_c`` [Pa].

    Parameters
    ----------
    temperature_c : float
        Ice temperature [degC] (negative for ice).
    bulk_salinity_ppt : float
        Bulk (core-measured) salinity [ppt].
    ice_class : {"FY", "MY"}
        Applies the structural enhancement ``kappa`` for multi-year ice.
    params : StrengthParameters
        Model constants (see :data:`PARAMETER_PROVENANCE`).
    """
    cls = str(ice_class).upper()
    if cls not in ("FY", "MY"):
        raise ValueError("ice_class must be 'FY' or 'MY'")
    nu = brine_volume_fraction(temperature_c, bulk_salinity_ppt)
    factor = porosity_strength_factor(
        nu,
        porosity_limit=params.porosity_limit,
        porosity_exponent=params.porosity_exponent,
    )
    kappa = params.myi_structural_factor if cls == "MY" else 1.0
    return float(kappa * params.sigma0_pa * factor)


# ---------------------------------------------------------------------------
# 4. Grids / trends
# ---------------------------------------------------------------------------
def strength_surface(
    temperatures_c: Sequence[float],
    salinities_ppt: Sequence[float],
    *,
    ice_class: str = "FY",
    params: StrengthParameters = DEFAULT_PARAMETERS,
) -> np.ndarray:
    """2-D ``sigma_c`` array with shape ``(n_T, n_S)``."""
    t = np.asarray(temperatures_c, dtype=float)
    s = np.asarray(salinities_ppt, dtype=float)
    out = np.empty((t.size, s.size), dtype=float)
    for i, ti in enumerate(t):
        for j, sj in enumerate(s):
            out[i, j] = compressive_strength(ti, sj, ice_class=ice_class, params=params)
    return out


def temperature_scan(
    bulk_salinity_ppt: float,
    *,
    t_min: float = -40.0,
    t_max: float = 0.0,
    n: int = 17,
    ice_class: str = "FY",
    params: StrengthParameters = DEFAULT_PARAMETERS,
) -> list[dict[str, float]]:
    """``sigma_c`` versus temperature (``t_min`` .. ``t_max``, ``n`` points)."""
    out = []
    for ti in np.linspace(t_min, t_max, int(n)):
        out.append(
            {
                "temperature_c": float(ti),
                "brine_salinity_ppt": brine_salinity(float(ti)),
                "brine_volume_fraction": brine_volume_fraction(float(ti), bulk_salinity_ppt),
                "sigma_c_pa": compressive_strength(
                    float(ti), bulk_salinity_ppt, ice_class=ice_class, params=params
                ),
            }
        )
    return out


def salinity_scan(
    temperature_c: float,
    *,
    s_min: float = 0.0,
    s_max: float = 15.0,
    n: int = 16,
    ice_class: str = "FY",
    params: StrengthParameters = DEFAULT_PARAMETERS,
) -> list[dict[str, float]]:
    """``sigma_c`` versus bulk salinity (``s_min`` .. ``s_max``, ``n`` points)."""
    out = []
    for sj in np.linspace(s_min, s_max, int(n)):
        out.append(
            {
                "salinity_ppt": float(sj),
                "brine_volume_fraction": brine_volume_fraction(temperature_c, float(sj)),
                "sigma_c_pa": compressive_strength(
                    temperature_c, float(sj), ice_class=ice_class, params=params
                ),
            }
        )
    return out


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (ties handled by average ranks)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size or xa.size < 2:
        raise ValueError("need two equally sized arrays with >= 2 points")
    rx = _rankdata(xa)
    ry = _rankdata(ya)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# 5. Uncertainty propagation (Monte Carlo)
# ---------------------------------------------------------------------------
def monte_carlo_strength(
    temperature_c: float,
    bulk_salinity_ppt: float,
    *,
    ice_class: str = "FY",
    n_samples: int = 20000,
    seed: int = 20261110,
    params: StrengthParameters = DEFAULT_PARAMETERS,
    salinity_sigma_ppt: float = 0.5,
) -> dict[str, Any]:
    """Propagate parameter uncertainty to ``sigma_c`` (percentile confidence band).

    Sampled parameters (uniform on their documented ranges in
    :data:`PARAMETER_PROVENANCE`): ``sigma_0``, ``nu_p``, ``m``, ``kappa_MY``,
    plus a normal measurement error on the core salinity.

    Uniform priors are used deliberately: the documented ranges are
    *intervals*, not standard deviations, so a normal prior would claim more
    information than exists.
    """
    rng = np.random.default_rng(int(seed))
    p = PARAMETER_PROVENANCE
    s0 = rng.uniform(*p["sigma0_pa"]["range"], size=n_samples)
    np_ = rng.uniform(*p["porosity_limit"]["range"], size=n_samples)
    m = rng.uniform(*p["porosity_exponent"]["range"], size=n_samples)
    if str(ice_class).upper() == "MY":
        kappa = rng.uniform(*p["myi_structural_factor"]["range"], size=n_samples)
    else:
        kappa = np.ones(n_samples)
    s_samp = np.maximum(0.0, rng.normal(float(bulk_salinity_ppt), float(salinity_sigma_ppt), n_samples))

    sb = brine_salinity(temperature_c)
    if sb <= 0.0:
        nu = np.ones(n_samples)
    else:
        rb = brine_density(sb)
        nu = (RHO_ICE_G_CM3 / rb) * s_samp / sb
    ratio = 1.0 - nu / np_
    factor = np.where(ratio > 0.0, np.power(np.clip(ratio, 0.0, None), m), 0.0)
    sigma = kappa * s0 * factor

    q = np.percentile(sigma, [5.0, 50.0, 95.0])
    return {
        "temperature_c": float(temperature_c),
        "bulk_salinity_ppt": float(bulk_salinity_ppt),
        "ice_class": str(ice_class).upper(),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "deterministic_sigma_c_pa": compressive_strength(
            temperature_c, bulk_salinity_ppt, ice_class=ice_class, params=params
        ),
        "p05_pa": float(q[0]),
        "p50_pa": float(q[1]),
        "p95_pa": float(q[2]),
        "mean_pa": float(sigma.mean()),
        "std_pa": float(sigma.std(ddof=1)),
    }
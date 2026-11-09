"""Mixed FY/MY ice field in the DEM section: is the linear mixture rule true?

The engineering answer for a mixed ice field is usually written as a *linear
mixture of class resistances*

    R_mix(p) = (1 - p) R_FY + p R_MY ,                                        (1)

with ``p`` the multi-year (MY) **area fraction**.  :mod:`tensordem.mixed_ice`
derives (1) for the anchors by construction (area-weighted FSD quadrature), so
within the anchor family (1) is an identity, not a prediction.  The question
this module answers is the one the identity cannot: does a *direct numerical
simulation of a spatially heterogeneous field* return (1)?

Why it can fail
---------------
Every term of the Lindqvist family is non-linear in the material state
(``R_crush ~ sigma_c h``, ``R_bend ~ sigma_f h^1.25``, friction through the
normal force) and the field is spatially coupled through the elastic lattice and
through fracture: a stiff MY patch flanked by weak FY ice loses its bending
support and fails earlier than the same MY patch inside pure MY ice.  Such
"weakest-link" coupling is exactly what (1) assumes away, and it can only be
measured with a heterogeneous simulation.

What is simulated
-----------------
:class:`tensordem.level_ice.sheet.LevelIceSheet` is used with its optional
``material_field``: every particle carries a class label (FY or MY), and the
per-particle multipliers of flexural strength, elastic modulus and crushing
strength are the *ratios* of that class' ageing-law material state to the
first-year reference state (:mod:`tensordem.mixed_ice.aging`).  Consequently

* ``p = 0`` produces multipliers of exactly 1.0 -> the sheet is bit-for-bit the
  homogeneous reference model, which is the FY-degeneration criterion;
* a bond straddling a class boundary takes the arithmetic mean of its two end
  particles (declared mixing convention);
* geometry (thickness, particle size, speed, hull) is held **fixed** across the
  sweep, so the measurement isolates the *material* part of (1).  Thickness
  heterogeneity is not resolvable in a plane-section lattice of one thickness
  and is therefore quantified separately from the anchor's thickness exponent
  (see ``docs/mixed_ice_dem_field.md``, limitation L1).

Two mixing hypotheses are tested against the same runs:

``linear``      equation (1) with ``R_FY``/``R_MY`` measured by the same model
                at ``p = 0``/``p = 1`` (the criterion the node asks for).
``series``      the harmonic (Reuss) mixture ``1/R = (1-p)/R_FY + p/R_MY``,
                the opposite extreme a coupled field would drift towards if the
                weak class dominated the load path.  It is reported as the
                "how wrong could (1) be" reference, not as a candidate model.

Units: SI (m, s, N, Pa); resistances are line loads in N/m of a 2-D section.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..level_ice.campaign import CaseSpec
from ..level_ice.formula import ICEBREAKER, IceState, lindqvist_components
from ..level_ice.sheet import LevelIceSheet
from . import aging

__all__ = [
    "DEM_FIELD_BASE",
    "MIXTURE_BAND",
    "CellScales",
    "class_scales",
    "patch_labels",
    "material_field_from_labels",
    "run_field_case",
    "linear_mixture",
    "series_mixture",
    "field_campaign",
    "mixture_fit",
    "my_fraction_effect",
    "strength_ratio_audit",
    "age_strength_curve",
]

#: ``MY`` class age used everywhere in this module [years]: an old, heavily
#: desalinated floe, the state the ageing law saturates towards by ~10 a.
MY_CLASS_AGE_YEARS = 4.0

#: tolerance on the linear-mixture error (node criterion (b)).
MIXTURE_BAND = 0.15

#: MY floe length [m] above which a transit is called *field realistic*: a real
#: MY floe is 10-100 m across, i.e. far longer than this 5 m transit, so the
#: smallest defensible reading is "the transit crosses floes, not a texture".
#: 1.5 m = 1.5 ice thicknesses is the threshold used to split the configurations
#: (declared a priori, not tuned to the result).
FIELD_REALISTIC_FLOE_LENGTH_M = 1.5

#: Base DEM settings: the repository level-ice reference case, so that the
#: ``p = 0`` field run is *identical* to ``level_ice.campaign.CaseSpec()``.
DEM_FIELD_BASE: dict[str, Any] = dict(
    hull="icebreaker",
    thickness_m=1.0,
    speed_m_s=3.0,
    layers=4,
    length_m=24.0,
    advance_m=6.0,
    bow_angle_deg=30.0,
    mass_scale=200.0,
    buoyancy=True,
    window_fraction=(0.55, 0.98),
)

#: Calibration provenance of the mixture test (documentation only).
MIXTURE_DOC = {
    "regime": "2-D plane section, crushing-dominated (crushing fraction > 0.8)",
    "my_class_age_years": MY_CLASS_AGE_YEARS,
    "bond_mixing_convention": "arithmetic mean of the two end-particle multipliers",
    "geometry_held_fixed": True,
}


# ---------------------------------------------------------------------------
# class material states -> per-particle multipliers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellScales:
    """One ice class' material multipliers relative to the first-year reference."""

    name: str
    age_years: float
    thickness_m: float
    flexural_strength_pa: float
    compressive_strength_pa: float
    elastic_modulus_pa: float
    brine_volume_fraction: float
    strength_scale: float
    modulus_scale: float
    crush_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "class": self.name,
            "age_years": self.age_years,
            "thickness_m": self.thickness_m,
            "flexural_strength_pa": self.flexural_strength_pa,
            "compressive_strength_pa": self.compressive_strength_pa,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "brine_volume_fraction": self.brine_volume_fraction,
            "strength_scale": self.strength_scale,
            "modulus_scale": self.modulus_scale,
            "crush_scale": self.crush_scale,
        }


def class_scales(age_my: float = MY_CLASS_AGE_YEARS) -> dict[str, CellScales]:
    """Multipliers of the FY and MY material states w.r.t. the FY reference.

    The first-year state is the repository reference itself (the ageing law is
    calibrated so that age 1 returns ``sigma_c = 1.5 MPa``, ``E = 5 GPa``), so
    its multipliers are exactly 1.0 and the FY class *is* the existing model.
    """
    fy = aging.material_state(1.0)
    my = aging.material_state(float(age_my))
    out = {}
    for name, state in (("FY", fy), ("MY", my)):
        out[name] = CellScales(
            name=name,
            age_years=state.age_years,
            thickness_m=state.thickness_m,
            flexural_strength_pa=state.flexural_strength_pa,
            compressive_strength_pa=state.compressive_strength_pa,
            elastic_modulus_pa=state.elastic_modulus_pa,
            brine_volume_fraction=state.brine_volume_fraction,
            strength_scale=state.flexural_strength_pa / fy.flexural_strength_pa,
            modulus_scale=state.elastic_modulus_pa / fy.elastic_modulus_pa,
            crush_scale=state.compressive_strength_pa / fy.compressive_strength_pa,
        )
    return out


# ---------------------------------------------------------------------------
# spatial class field
# ---------------------------------------------------------------------------
def patch_labels(nx: int, my_fraction: float, patch_columns: int,
                 seed: int = 0) -> np.ndarray:
    """Class label per grid column: ``True`` = MY ice.

    Exactly ``round(my_fraction * nx)`` columns are labelled MY and they are
    grouped into ``round(n_my / patch_columns)`` contiguous patches placed at
    seeded random positions.  The *realised* fraction is returned by
    :func:`run_field_case` and is what the mixture rule is evaluated at, so the
    integer rounding of the field never biases the comparison.
    """
    nx = int(nx)
    if nx <= 0:
        raise ValueError("nx must be positive")
    p = float(my_fraction)
    if not 0.0 <= p <= 1.0:
        raise ValueError("my_fraction must be in [0, 1]")
    patch_columns = max(1, int(patch_columns))
    labels = np.zeros(nx, dtype=bool)
    n_my = int(round(p * nx))
    if n_my <= 0:
        return labels
    if n_my >= nx:
        labels[:] = True
        return labels
    rng = np.random.default_rng(int(seed))
    n_patch = max(1, int(round(n_my / patch_columns)))
    sizes = np.full(n_patch, n_my // n_patch, dtype=int)
    sizes[: n_my - n_patch * (n_my // n_patch)] += 1
    # place the patches on a seeded permutation of admissible start columns;
    # retry on overlap, then fall back to a periodic layout (deterministic).
    occupied = np.zeros(nx, dtype=bool)
    starts: list[int] = []
    candidates = np.arange(nx, dtype=int)
    rng.shuffle(candidates)
    for size in sizes:
        placed = False
        for start in candidates:
            if start + size > nx:
                continue
            if occupied[start:start + size].any():
                continue
            occupied[start:start + size] = True
            starts.append(int(start))
            placed = True
            break
        if not placed:  # pragma: no cover - only for pathological fillings
            remaining = np.flatnonzero(~occupied)
            if remaining.size < size:
                break
            start = int(remaining[0])
            occupied[start:start + size] = True
            starts.append(start)
    labels[occupied] = True
    return labels


def periodic_labels(nx: int, my_fraction: float, patch_columns: int,
                    period_columns: int, *, phase: int = 0) -> np.ndarray:
    """Balanced layout: every period contains exactly the nominal MY fraction.

    When the period equals the hull advance, **every** transit sees exactly
    ``my_fraction`` of MY ice regardless of phase, so the run-to-run spread that
    remains is fracture-path chaos only -- the spatial sampling of the field is
    removed by construction.  This is the layout used to test the mixture rule
    itself; the random layout of :func:`patch_labels` is the field-realistic
    counterpart used to measure the single-transit scatter.
    """
    nx = int(nx)
    period = max(1, int(period_columns))
    if not 0.0 <= float(my_fraction) <= 1.0:
        raise ValueError("my_fraction must be in [0, 1]")
    n_my = int(round(float(my_fraction) * period))
    cell = np.zeros(period, dtype=bool)
    if n_my >= period:
        cell[:] = True
    elif n_my > 0:
        patch_columns = max(1, int(patch_columns))
        n_patch = max(1, int(math.ceil(n_my / patch_columns)))
        sizes = np.full(n_patch, n_my // n_patch, dtype=int)
        sizes[: n_my - n_patch * (n_my // n_patch)] += 1
        # patches spaced as evenly as possible inside the period
        starts = [int(round(k * period / n_patch)) for k in range(n_patch)]
        for start, size in zip(starts, sizes):
            idx = (np.arange(start, start + size)) % period
            cell[idx] = True
    reps = int(math.ceil(nx / period)) + 1
    tiled = np.tile(cell, reps)[: nx + period]
    shift = int(phase) % period
    labels = tiled[shift: shift + nx]
    return labels.astype(bool)


def traversed_mask(nx: int, spacing_m: float, thickness_m: float,
                   bow_angle_deg: float, advance_m: float) -> np.ndarray:
    """Columns swept by the hull face between t = 0 and the end of the advance.

    The hull's top vertex starts at ``x0 = -(1.5 h tan(bow) + d/2)`` (same
    expression as :class:`tensordem.level_ice.sheet.LevelIceSheet`), so the
    traversed ice is the set of column centres inside ``[x0, x0 + advance]``.
    """
    face_dx = 1.5 * float(thickness_m) * math.tan(math.radians(float(bow_angle_deg)))
    x0 = -(face_dx + 0.5 * float(spacing_m))
    x1 = x0 + float(advance_m)
    centres = (np.arange(int(nx)) + 0.5) * float(spacing_m)
    return (centres >= x0) & (centres <= x1)


def material_field_from_labels(labels: np.ndarray, nz: int,
                               scales: dict[str, CellScales] | None = None
                               ) -> dict[str, np.ndarray]:
    """Per-particle multiplier arrays from a per-column class label array."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scales = class_scales() if scales is None else scales
    nz = int(nz)
    node = np.repeat(labels, nz)          # grid_z varies fastest within a column
    fy, my = scales["FY"], scales["MY"]
    strength = np.where(node, my.strength_scale, fy.strength_scale)
    modulus = np.where(node, my.modulus_scale, fy.modulus_scale)
    crush = np.where(node, my.crush_scale, fy.crush_scale)
    return {
        "strength_scale": strength,
        "modulus_scale": modulus,
        "crush_scale": crush,
        "class_is_my": node,
    }


# ---------------------------------------------------------------------------
# one heterogeneous field case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """One DEM run of a spatially heterogeneous mixed field."""

    my_fraction: float = 0.0
    seed: int = 0
    patch_length_m: float = 2.0
    my_age_years: float = MY_CLASS_AGE_YEARS
    layout: str = "random"          # "random" | "periodic"
    label: str = ""
    base: dict[str, Any] = field(default_factory=lambda: dict(DEM_FIELD_BASE))

    def case_spec(self) -> CaseSpec:
        kwargs = dict(self.base)
        kwargs.pop("window_fraction", None)
        spec = CaseSpec(**kwargs)
        return spec

    def as_dict(self) -> dict[str, Any]:
        return {
            "my_fraction_nominal": self.my_fraction,
            "seed": int(self.seed),
            "patch_length_m": self.patch_length_m,
            "my_age_years": self.my_age_years,
            "layout": self.layout,
            "label": self.label,
            "base": {k: v for k, v in self.base.items()},
        }


def _patch_runs(labels: np.ndarray) -> list[int]:
    """Lengths of the contiguous ``True`` runs (in columns)."""
    lab = np.asarray(labels, dtype=bool)
    if not lab.any():
        return []
    padded = np.concatenate(([False], lab, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [int(edges[k + 1] - edges[k]) for k in range(0, len(edges), 2)]


def _realised_fraction(labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(labels, dtype=bool)))


def run_field_case(spec: FieldSpec) -> dict[str, Any]:
    """Run one heterogeneous field case and return the measured ledger."""
    started = time.time()
    case = spec.case_spec()
    config = case.sheet_config()
    window = tuple(spec.base.get("window_fraction", (0.55, 0.98)))
    scales = class_scales(spec.my_age_years)
    patch_columns = max(1, int(round(spec.patch_length_m / config.spacing_m)))
    advance_pre = float(spec.base["advance_m"])
    if spec.layout == "periodic":
        # period = the traversed window, so every transit is exposed to exactly
        # the nominal MY fraction whatever its phase (sampling variance -> 0).
        mask0 = traversed_mask(config.n_x, config.spacing_m, config.thickness_m,
                               spec.base["bow_angle_deg"], advance_pre)
        period_columns = max(1, int(mask0.sum()))
        labels = periodic_labels(config.n_x, spec.my_fraction, patch_columns,
                                 period_columns, phase=spec.seed)
    else:
        period_columns = 0
        labels = patch_labels(config.n_x, spec.my_fraction, patch_columns,
                              seed=spec.seed)
    material = material_field_from_labels(labels, config.n_z, scales)
    class_is_my = material.pop("class_is_my")
    sheet = LevelIceSheet(config, material_field=material)
    advance = float(spec.base["advance_m"])
    # The section only samples the ice the hull actually traverses: the columns
    # swept by the hull face between t=0 and the end of the advance.  The mixture
    # rule is therefore evaluated at the *traversed* MY fraction, which removes
    # the sampling bias of a short transit through a long field.
    x_lo = sheet.hull_position()
    x_hi = x_lo + advance
    column_x = (np.arange(config.n_x) + 0.5) * config.spacing_m
    traversed = (column_x >= x_lo) & (column_x <= x_hi)
    assert np.array_equal(traversed, traversed_mask(
        config.n_x, config.spacing_m, config.thickness_m,
        spec.base["bow_angle_deg"], advance))
    result = sheet.run(distance_m=float(spec.base["advance_m"]), record_stride=25,
                       window_fraction=window)
    steady = result.steady_resistance()
    payload: dict[str, Any] = {
        "spec": spec.as_dict(),
        "config": config.as_dict(),
        "dt_s": sheet.dt,
        "n_particles": sheet.n,
        "n_bonds": sheet.n_bonds,
        "n_steps": result.summary["n_steps"],
        "realised_my_fraction_area": _realised_fraction(labels),
        "realised_my_fraction_particles": float(np.mean(class_is_my)),
        "traversed_my_fraction": float(np.mean(labels[traversed])),
        "traversed_my_patches": len(_patch_runs(labels[traversed])),
        "traversed_my_patch_lengths_m": [
            n * config.spacing_m for n in _patch_runs(labels[traversed])],
        "traversed_my_patch_length_m": (
            float(np.mean(_patch_runs(labels[traversed])) * config.spacing_m)
            if _patch_runs(labels[traversed]) else 0.0),
        "layout": spec.layout,
        "traversed_columns": int(traversed.sum()),
        "traversed_span_m": [float(x_lo), float(x_hi)],
        "patch_columns": patch_columns,
        "period_columns": period_columns,
        "n_patches": int(np.sum(np.diff(np.concatenate(([0], labels.astype(int),
                                                         [0]))) == 1)),
        "material_summary": sheet.material_summary(),
        "mean_total_n_per_m": result.steady_mean("total_n_per_m"),
        "std_total_n_per_m": result.steady_std("total_n_per_m"),
        "crushing_n_per_m": result.steady_mean("crushing_n_per_m"),
        "flexure_n_per_m": result.steady_mean("flexure_n_per_m"),
        "friction_n_per_m": result.steady_mean("friction_n_per_m"),
        "crushing_fraction": steady["crushing_fraction"],
        "flexure_fraction": steady["flexure_fraction"],
        "broken_fraction": float(result.broken_fraction[-1]),
        "removed_particles": result.summary["removed_particles"],
        "max_contacts": result.summary["max_contacts"],
        "first_break_advance_m": result.summary["first_break_advance_m"],
        "window_m": list(result.window),
        "class_scales": {k: v.as_dict() for k, v in scales.items()},
        "labels": labels.astype(int).tolist(),
        "runtime_s": time.time() - started,
    }
    return payload


# ---------------------------------------------------------------------------
# mixture rules and their fit
# ---------------------------------------------------------------------------
def linear_mixture(r_fy: float, r_my: float, my_fraction: float) -> float:
    """Equation (1): the mixture rule under test."""
    p = float(my_fraction)
    return (1.0 - p) * r_fy + p * r_my


def series_mixture(r_fy: float, r_my: float, my_fraction: float) -> float:
    """Reuss (harmonic) mixture: the coupled-field lower reference."""
    p = float(my_fraction)
    if r_fy <= 0.0 or r_my <= 0.0:
        raise ValueError("resistances must be positive")
    return 1.0 / ((1.0 - p) / r_fy + p / r_my)


def mixture_fit(alphas: Sequence[float], values: Sequence[float]) -> dict[str, Any]:
    """Least-squares line through the measured DEM ``R(p)`` curve."""
    p = np.asarray(alphas, dtype=float)
    r = np.asarray(values, dtype=float)
    if p.size < 2:
        raise ValueError("need at least two fractions")
    A = np.stack((np.ones_like(p), p), axis=1)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - np.mean(r)) ** 2))
    return {
        "intercept_n_per_m": float(coef[0]),
        "slope_n_per_m": float(coef[1]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "max_residual_pct": float(np.max(np.abs(r - pred) / np.maximum(np.abs(pred), 1e-30)) * 100.0),
        "n_points": int(p.size),
    }


# ---------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------
def _alphas(alphas: Iterable[float]) -> list[float]:
    out = sorted(float(a) for a in alphas)
    if min(out) < 0.0 or max(out) > 1.0:
        raise ValueError("my fractions must be in [0, 1]")
    return out


def field_campaign(
    *,
    alphas: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: Sequence[int] = (11, 23, 37, 41, 53, 67),
    patch_length_m: float = 2.0,
    my_age_years: float = MY_CLASS_AGE_YEARS,
    layout: str = "periodic",
    workers: int | None = None,
    base: dict[str, Any] | None = None,
    keep_runs: bool = False,
) -> dict[str, Any]:
    """Sweep the MY area fraction in the heterogeneous DEM and test (1).

    Two statistics are reported and they answer different questions:

    ``per-alpha``   ensemble mean of the DEM resistance at each nominal
                    fraction, the *bias* against the linear mixture rule
                    evaluated at the **traversed** MY fraction of each run, and
                    a t-based 95 % confidence interval on that bias;
    ``scatter``     the run-to-run coefficient of variation of a *single*
                    realisation, i.e. how much a single heterogeneous DEM run
                    can differ from the ensemble.  Fracture patterns in a
                    heterogeneous lattice are chaotic, so this scatter is a
                    property of the measurement, not an error of the model.

    ``criterion_b_pass`` uses the ensemble bias (the quantity the mixture rule
    predicts); the single-run scatter is reported next to it so that the number
    cannot be read as a single-run accuracy claim.
    """
    alphas = _alphas(alphas)
    base_settings = dict(DEM_FIELD_BASE if base is None else base)
    specs: list[FieldSpec] = []
    for a in alphas:
        # homogeneous ends are seed independent -> one run each
        seed_list = (0,) if (a <= 0.0 or a >= 1.0) else tuple(int(s) for s in seeds)
        for s in seed_list:
            specs.append(FieldSpec(my_fraction=a, seed=s,
                                   patch_length_m=patch_length_m,
                                   my_age_years=my_age_years,
                                   layout=layout,
                                   label=f"p{a:.2f}_s{s}", base=base_settings))
    runs = _run_all(specs, workers=workers)
    by_alpha: dict[float, list[dict[str, Any]]] = {}
    for a, run in zip([s.my_fraction for s in specs], runs):
        by_alpha.setdefault(float(a), []).append(run)
    r_fy = float(np.mean([r["mean_total_n_per_m"] for r in by_alpha[min(alphas)]]))
    r_my = float(np.mean([r["mean_total_n_per_m"] for r in by_alpha[max(alphas)]]))
    rows = []
    for a in alphas:
        group = by_alpha[float(a)]
        vals = np.array([r["mean_total_n_per_m"] for r in group], dtype=float)
        p_path = np.array([r["traversed_my_fraction"] for r in group], dtype=float)
        p_field = np.array([r["realised_my_fraction_area"] for r in group])
        p_used = float(p_path.mean())
        n_patch = np.array([r["traversed_my_patches"] for r in group], dtype=float)
        patch_len = np.array([r["traversed_my_patch_length_m"] for r in group])
        pred = np.array([linear_mixture(r_fy, r_my, float(p)) for p in p_path])
        err = 100.0 * (vals - pred) / pred
        mean = float(vals.mean())
        n = len(group)
        if n > 1:
            sem = float(vals.std(ddof=1) / math.sqrt(n))
            tcrit = _t95(n - 1)
            ci = 100.0 * tcrit * sem / mean
            cv = 100.0 * float(vals.std(ddof=1)) / mean
        else:
            ci = 0.0
            cv = 0.0
        lin = linear_mixture(r_fy, r_my, p_used)
        ser = series_mixture(r_fy, r_my, p_used)
        rows.append({
            "my_fraction_nominal": float(a),
            "my_fraction_realised": float(p_field.mean()),
            "traversed_my_fraction_mean": p_used,
            "traversed_my_fraction_min": float(p_path.min()),
            "traversed_my_fraction_max": float(p_path.max()),
            "n_seeds": n,
            "layout": group[0]["layout"],
            "patch_length_nominal_m": float(specs[0].patch_length_m),
            "my_patches_in_transit_mean": float(n_patch.mean()),
            "my_patch_length_m_mean": float(patch_len.mean()),
            "my_patch_length_m_min": float(patch_len.min()),
            "my_patch_length_m_max": float(patch_len.max()),
            "dem_mean_n_per_m": mean,
            "dem_std_n_per_m": float(vals.std(ddof=1)) if n > 1 else 0.0,
            "single_run_cv_pct": cv,
            "mean_error_ci95_pct": ci,
            "dem_min_n_per_m": float(vals.min()),
            "dem_max_n_per_m": float(vals.max()),
            "linear_prediction_n_per_m": lin,
            "linear_error_pct": 100.0 * (mean - lin) / lin,
            "linear_error_of_runs_pct_mean": float(err.mean()),
            "linear_error_of_runs_pct_sd": float(err.std(ddof=1)) if n > 1 else 0.0,
            "series_prediction_n_per_m": ser,
            "series_error_pct": 100.0 * (mean - ser) / ser,
            "crushing_n_per_m": float(np.mean([r["crushing_n_per_m"] for r in group])),
            "friction_n_per_m": float(np.mean([r["friction_n_per_m"] for r in group])),
            "flexure_n_per_m": float(np.mean([r["flexure_n_per_m"] for r in group])),
            "broken_fraction": float(np.mean([r["broken_fraction"] for r in group])),
            "crushing_fraction": float(np.mean([r["crushing_fraction"] for r in group])),
        })
    criterion = _criterion(rows)
    fit = mixture_fit([row["traversed_my_fraction_mean"] for row in rows],
                      [row["dem_mean_n_per_m"] for row in rows])
    ensemble = _ensemble_regression(runs, r_fy, r_my)
    payload: dict[str, Any] = {
        "settings": {
            "alphas_nominal": list(alphas),
            "seeds": list(seeds),
            "patch_length_m": patch_length_m,
            "my_age_years": my_age_years,
            "layout": layout,
            "base": base_settings,
            "doc": dict(MIXTURE_DOC),
        },
        "class_scales": {k: v.as_dict() for k, v in class_scales(my_age_years).items()},
        "r_fy_n_per_m": r_fy,
        "r_my_n_per_m": r_my,
        "class_ratio_dem": r_my / r_fy if r_fy else float("nan"),
        "rows": rows,
        "fit": fit,
        "ensemble_regression": ensemble,
    }
    payload.update(criterion)
    payload["runs"] = runs if keep_runs else [
        {k: v for k, v in run.items() if k != "labels"} for run in runs
    ]
    payload["my_fraction_effect"] = my_fraction_effect(payload)
    return payload


def _t95(dof: int) -> float:
    """Two-sided 95 % Student-t factor (table, avoids a scipy dependency)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
             19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}
    if dof in table:
        return table[dof]
    if dof <= 0:
        return float("nan")
    return 1.960 if dof > 60 else table[min(table, key=lambda k: abs(k - dof))]


def _ensemble_regression(runs: Sequence[dict[str, Any]], r_fy: float,
                         r_my: float) -> dict[str, Any]:
    """Pool every run: regress R on the traversed MY fraction."""
    p = np.array([r["traversed_my_fraction"] for r in runs], dtype=float)
    r = np.array([r["mean_total_n_per_m"] for r in runs], dtype=float)
    interior = (p > 0.0) & (p < 1.0)
    A = np.stack((np.ones_like(p[interior]), p[interior]), axis=1)
    coef, *_ = np.linalg.lstsq(A, r[interior], rcond=None)
    resid = r[interior] - A @ coef
    dof = max(1, int(interior.sum()) - 2)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(np.sum((r[interior] - r[interior].mean()) ** 2))
    ss_res = float(resid @ resid)
    observed_slope = float(coef[1])
    ideal_slope = r_my - r_fy
    return {
        "n_runs": int(interior.sum()),
        "intercept_n_per_m": float(coef[0]),
        "intercept_se_n_per_m": float(se[0]),
        "slope_n_per_m": observed_slope,
        "slope_se_n_per_m": float(se[1]),
        "slope_ci95_n_per_m": [float(observed_slope - _t95(dof) * se[1]),
                               float(observed_slope + _t95(dof) * se[1])],
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "ideal_intercept_n_per_m": r_fy,
        "ideal_slope_n_per_m": ideal_slope,
        "slope_ratio_observed_over_ideal": observed_slope / ideal_slope
        if ideal_slope else float("nan"),
        "slope_ratio_ci95": [(float(observed_slope - _t95(dof) * se[1]) / ideal_slope),
                             (float(observed_slope + _t95(dof) * se[1]) / ideal_slope)]
        if ideal_slope else [float("nan"), float("nan")],
        "residual_cv_pct": 100.0 * math.sqrt(sigma2) / float(r[interior].mean()),
        "t95_factor": _t95(dof),
    }


def _criterion(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Criterion (b) verdict, with the physically representative regime first.

    The primary quantity is the worst ensemble bias among the configurations in
    which a transit crosses **at most one** MY patch: a real MY floe (10-100 m)
    is far larger than the 5 m of field the hull traverses, so a ship is inside
    one MY floe or one FY sheet, never in a sub-metre MY/FY texture.  The
    "intimately mixed" configurations (two or more MY patches per transit) are
    reported next to it as the sensitivity axis: they are the regime in which
    the interface seams can dominate, and they are what the strict variant tests.
    """
    interior = [row for row in rows if 0.0 < row["my_fraction_realised"] < 1.0]
    if not interior:
        return {"criterion_b_pass": False,
                "criterion_b_note": "no interior fractions measured"}
    max_abs = max(abs(row["linear_error_pct"]) for row in interior)
    worst_ci = max(row["mean_error_ci95_pct"] for row in interior)
    coarse = [row for row in interior
              if row["my_patch_length_m_mean"] >= FIELD_REALISTIC_FLOE_LENGTH_M]
    fine = [row for row in interior
            if row["my_patch_length_m_mean"] < FIELD_REALISTIC_FLOE_LENGTH_M]
    coarse_err = max((abs(row["linear_error_pct"]) for row in coarse),
                     default=float("nan"))
    fine_err = max((abs(row["linear_error_pct"]) for row in fine),
                   default=float("nan"))
    band = 100.0 * MIXTURE_BAND
    n_pass = sum(1 for row in interior if abs(row["linear_error_pct"]) < band)
    return {
        "max_abs_linear_error_pct": max_abs,
        "worst_ci95_pct": worst_ci,
        "linear_mixture_band_pct": band,
        "field_realistic_floe_length_m": FIELD_REALISTIC_FLOE_LENGTH_M,
        "coarse_floe_max_abs_error_pct": coarse_err,
        "fine_floe_max_abs_error_pct": fine_err,
        "n_coarse_configs": len(coarse),
        "n_fine_configs": len(fine),
        "n_configs_in_band": int(n_pass),
        "n_configs_total": len(interior),
        "criterion_b_pass": bool(max_abs + worst_ci < band),
        "criterion_b_pass_coarse_only": bool(
            np.isfinite(coarse_err) and coarse_err < band),
        "criterion_b_note": "pass (strict) = every measured configuration, "
                            "including its 95 % confidence half-width, inside "
                            "the band; the coarse-floe variant only looks at "
                            "transits whose crossed MY floe is at least "
                            "FIELD_REALISTIC_FLOE_LENGTH_M long.",
    }


def combine_campaigns(campaigns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge several sweeps (e.g. two patch lengths) into one criterion table."""
    if not campaigns:
        raise ValueError("need at least one campaign")
    rows = [row for campaign in campaigns for row in campaign["rows"]]
    r_fy = float(np.mean([c["r_fy_n_per_m"] for c in campaigns]))
    r_my = float(np.mean([c["r_my_n_per_m"] for c in campaigns]))
    payload: dict[str, Any] = {
        "settings": {"merged_from": [c["settings"] for c in campaigns]},
        "class_scales": campaigns[0]["class_scales"],
        "r_fy_n_per_m": r_fy,
        "r_my_n_per_m": r_my,
        "class_ratio_dem": r_my / r_fy if r_fy else float("nan"),
        "rows": rows,
        "fit": mixture_fit([r["traversed_my_fraction_mean"] for r in rows],
                           [r["dem_mean_n_per_m"] for r in rows]),
    }
    runs = [run for campaign in campaigns for run in campaign.get("runs", [])]
    if runs:
        payload["ensemble_regression"] = _ensemble_regression(runs, r_fy, r_my)
    payload.update(_criterion(rows))
    payload["my_fraction_effect"] = my_fraction_effect(payload)
    return payload


def _run_all(specs: Sequence[FieldSpec], *, workers: int | None = None
             ) -> list[dict[str, Any]]:
    if workers is None:
        workers = min(len(specs), max(1, (os_cpu_count() or 1) - 1))
    if workers <= 1 or len(specs) == 1:
        return [run_field_case(spec) for spec in specs]
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        return list(pool.map(run_field_case, specs))


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# the node's headline question: 0 % -> 60 % MY
# ---------------------------------------------------------------------------
def my_fraction_effect(campaign: dict[str, Any], target: float = 0.60) -> dict[str, Any]:
    """Resistance change from ``p = 0`` to ``p = target`` from the DEM curve.

    Two numbers are produced: the value *interpolated along the fitted line*
    (which uses every measured fraction) and, when the campaign contains a row
    whose realised fraction is within 0.02 of ``target``, that directly measured
    row.  Both are relative statements about the same fixed geometry.
    """
    r_fy = float(campaign["r_fy_n_per_m"])
    fit = campaign["fit"]
    fitted = fit["intercept_n_per_m"] + fit["slope_n_per_m"] * float(target)
    direct = None
    for row in campaign["rows"]:
        if abs(row["traversed_my_fraction_mean"] - target) <= 0.05:
            direct = row
            break
    out = {
        "target_my_fraction": float(target),
        "r_at_zero_n_per_m": r_fy,
        "r_at_target_fitted_n_per_m": float(fitted),
        "relative_change_fitted_pct": 100.0 * (fitted - r_fy) / r_fy,
        "fit_r_squared": fit["r_squared"],
    }
    if direct is not None:
        out.update({
            "r_at_target_measured_n_per_m": direct["dem_mean_n_per_m"],
            "relative_change_measured_pct":
                100.0 * (direct["dem_mean_n_per_m"] - r_fy) / r_fy,
            "measured_at_realised_fraction": direct["my_fraction_realised"],
            "measured_at_traversed_fraction": direct["traversed_my_fraction_mean"],
            "measured_vs_linear_pct": direct["linear_error_pct"],
        })
    return out


# ---------------------------------------------------------------------------
# criterion (a): MY/FY compressive-strength ratio
# ---------------------------------------------------------------------------
#: literature band for the MY/FY *compressive* strength ratio (node criterion).
STRENGTH_RATIO_BAND = (1.5, 3.0)


def strength_ratio_audit(*, age_my: float = MY_CLASS_AGE_YEARS,
                         band: tuple[float, float] = STRENGTH_RATIO_BAND
                         ) -> dict[str, Any]:
    """Compressive-strength ratio MY/FY and its sensitivity to the exponent."""
    fy = aging.material_state(1.0)
    rows = []
    for age in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0):
        state = aging.material_state(float(age))
        rows.append({
            "age_years": float(age),
            "compressive_strength_pa": state.compressive_strength_pa,
            "ratio_to_fy": state.compressive_strength_pa / fy.compressive_strength_pa,
            "flexural_ratio_to_fy": state.flexural_strength_pa / fy.flexural_strength_pa,
            "modulus_ratio_to_fy": state.elastic_modulus_pa / fy.elastic_modulus_pa,
        })
    sensitivity = []
    nominal = aging.COMPRESSIVE_TEMPERATURE_EXPONENT
    lo, hi = aging.COMPRESSIVE_TEMPERATURE_EXPONENT_BAND
    exponents = sorted({lo, nominal, hi})
    for m in exponents:
        state = _material_state_with_exponent(float(age_my), m)
        ratio = state / fy.compressive_strength_pa
        sensitivity.append({
            "temperature_exponent": m,
            "compressive_strength_pa": state,
            "ratio_to_fy": ratio,
            "in_band": bool(band[0] <= ratio <= band[1]),
        })
    target = _material_state_with_exponent(float(age_my), nominal)
    ratio = target / fy.compressive_strength_pa
    porosity_only = aging.compressive_strength_pa(
        aging.brine_volume_fraction(aging.salinity_permille(age_my),
                                    aging.temperature_c(age_my)))
    return {
        "band": list(band),
        "age_my_years": float(age_my),
        "sigma_c_fy_pa": fy.compressive_strength_pa,
        "sigma_c_my_pa": target,
        "ratio_my_over_fy": ratio,
        "ratio_porosity_only_my_over_fy": porosity_only / fy.compressive_strength_pa,
        "criterion_a_pass": bool(band[0] <= ratio <= band[1]),
        "exponent_sensitivity": sensitivity,
        "curve": rows,
        "fy_degeneration_exact": bool(
            abs(aging.compressive_strength_pa(
                aging.brine_volume_fraction(aging.FY_SALINITY_PERMILLE,
                                            aging.FY_TEMPERATURE_C),
                aging.FY_TEMPERATURE_C) - aging.FY_COMPRESSIVE_REFERENCE_PA) <= 1e-9),
    }


def _material_state_with_exponent(age: float, exponent: float) -> float:
    """MY compressive strength at a given temperature exponent (sensitivity axis)."""
    s = aging.salinity_permille(age)
    t = aging.temperature_c(age)
    v_b = aging.brine_volume_fraction(s, t)
    return float(aging.compressive_strength_pa(v_b, t, temperature_exponent=exponent))


# ---------------------------------------------------------------------------
# criterion (c): monotone, saturating sigma_c(age)
# ---------------------------------------------------------------------------
def age_strength_curve(*, ages: Sequence[float] | None = None,
                       saturation_age: float = 1.0e6) -> dict[str, Any]:
    """Check ``sigma_c(t)`` is strictly increasing and saturates as t -> infinity."""
    a = np.asarray(ages if ages is not None else (1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                                                  7.0, 10.0, 15.0, 20.0, 30.0, 50.0),
                   dtype=float)
    fy = aging.material_state(1.0)
    states = [aging.material_state(float(x)) for x in a]
    sigma = np.array([s.compressive_strength_pa for s in states])
    d1 = np.diff(sigma)
    d2 = np.diff(sigma, 2)
    sat = aging.material_state(float(saturation_age))
    residual = abs(sat.compressive_strength_pa - sigma[-1]) / sat.compressive_strength_pa
    return {
        "ages_years": a.tolist(),
        "sigma_c_pa": sigma.tolist(),
        "ratio_to_fy": (sigma / fy.compressive_strength_pa).tolist(),
        "min_increment_pa": float(d1.min()),
        "strictly_increasing": bool(np.all(d1 > 0.0)),
        "concave": bool(np.all(d2 <= 1e-9)),
        "saturation_sigma_c_pa": sat.compressive_strength_pa,
        "saturation_ratio_to_fy": sat.compressive_strength_pa / fy.compressive_strength_pa,
        "residual_at_last_age_pct": 100.0 * residual,
        "criterion_c_pass": bool(np.all(d1 > 0.0) and residual < 0.02),
    }


# ---------------------------------------------------------------------------
# anchor cross-check: what the engineering model says at the same geometry
# ---------------------------------------------------------------------------
def anchor_class_ratio(*, thickness_m: float = 1.0, speed_m_s: float = 3.0,
                       my_age_years: float = MY_CLASS_AGE_YEARS,
                       sigma_c_my_pa: float | None = None,
                       model: str = "lindqvist") -> dict[str, Any]:
    """Anchor class ratio at fixed thickness, for comparison with the DEM.

    ``sigma_c_my_pa`` overrides the MY compressive strength, which is how the
    porosity-only (pre-extension) ageing law is kept available as a contrast.
    """
    scales = class_scales(my_age_years)
    fy, my = scales["FY"], scales["MY"]
    parts = {}
    for name, cell in (("FY", fy), ("MY", my)):
        sigma_c = cell.compressive_strength_pa
        if name == "MY" and sigma_c_my_pa is not None:
            sigma_c = float(sigma_c_my_pa)
        ice = IceState(thickness_m=thickness_m,
                       flexural_strength_pa=cell.flexural_strength_pa,
                       compressive_strength_pa=sigma_c,
                       elastic_modulus_pa=cell.elastic_modulus_pa)
        if model == "lindqvist":
            parts[name] = lindqvist_components(ice, ICEBREAKER, speed_m_s)
        elif model == "riska":
            from ..level_ice.formula import riska_components

            parts[name] = riska_components(ice, ICEBREAKER, speed_m_s)
        else:
            raise ValueError("model must be 'lindqvist' or 'riska'")
    fy_total = parts["FY"]["total_N"]
    my_total = parts["MY"]["total_N"]
    return {
        "model": model,
        "thickness_m": thickness_m,
        "speed_m_s": speed_m_s,
        "sigma_c_my_pa": float(my.compressive_strength_pa if sigma_c_my_pa is None
                               else sigma_c_my_pa),
        "fy": {k: float(v) for k, v in parts["FY"].items() if isinstance(v, (int, float))},
        "my": {k: float(v) for k, v in parts["MY"].items() if isinstance(v, (int, float))},
        "class_ratio_anchor": my_total / fy_total,
        "crushing_ratio_anchor": parts["MY"]["crushing_N"] / parts["FY"]["crushing_N"],
    }
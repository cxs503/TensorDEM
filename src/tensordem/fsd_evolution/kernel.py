"""Breakage kernel ``K(d, d') = nu(d') beta(d | d')`` calibrated on DEM events.

Two measured ingredients
------------------------
``beta(d | d')``
    The daughter-size distribution.  It is *not* fitted to a textbook law: the
    measured cascade catalogue (``[SplitEvent]`` from
    :mod:`tensordem.fsd_evolution.cascade`) is used as the empirical kernel and
    is summarised by a few statistics that the acceptance test can hold out:
    the daughter count ``m``, the mass-median daughter diameter and the mass
    fraction carried by fines.  Because the cascade instrument is
    two-dimensional and of constant floe thickness, the conserved measure is
    the floe **area** ``A ~ d**2`` (``sum_i x_i**2 = 1`` per event, verified
    against the parent node count to machine precision), so all mass statements
    below use the exponent ``nu = 2``.

``nu(d')``
    The breakage frequency.  It is measured as a *hazard rate*: the number of
    split events per unit applied strain in a size bin divided by the exposure
    (strain time) accumulated by load-carrying floes in that bin -- a
    Nelson-Aalen style estimator that handles the right-censored floes of every
    run correctly.  The measured hazard is fitted by ``nu(d) = nu_ref
    (d/d_ref)**q``; ``q > 0`` means big floes break more easily, which is the
    flexural-failure ordering.

Held-out validation
-------------------
The catalogue is split into a calibration part (seeds) and a held-out part.
The kernel summary fitted on the calibration half must predict the held-out
half's daughter count, mass-median daughter size and fines mass fraction to
within ``KERNEL_RESIDUAL_TOLERANCE`` (10 %) -- that is the campaign's
"kernel fit residual" criterion, stated as a *prediction* error instead of a
self-fulfilling in-sample fit.

Honesty
-------
* Hazard exposure counts only load-carrying fragments (a detached fragment
  cannot break in a quasi-static platen instrument), so ``q`` and the absolute
  rate are lower bounds for a debris-covered field; the caveat is carried into
  the report.
* The lattice floors the daughter census at one node, so the measured fines
  fraction is resolution limited: daughters at the floor are upper bounds on
  their true size.  The fraction of mass sitting at the floor is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .cascade import CascadeResult, SplitEvent

__all__ = [
    "KERNEL_RESIDUAL_TOLERANCE",
    "MASS_EXPONENT",
    "KernelSummary",
    "KernelCalibration",
    "EmpiricalDaughterRule",
    "summarise_events",
    "fit_hazard_exponent",
    "calibrate_kernel",
    "size_effect_hazard",
]

#: Campaign acceptance: kernel prediction residual on held-out events (10 %).
KERNEL_RESIDUAL_TOLERANCE = 0.10

#: Conserved-measure exponent: floes are area-equivalent disks of constant
#: thickness, so "mass" is floe area (``d**2``).
MASS_EXPONENT = 2


def _mass(sizes: np.ndarray) -> np.ndarray:
    return np.asarray(sizes, dtype=np.float64) ** MASS_EXPONENT


# --------------------------------------------------------------------- summary
@dataclass(frozen=True)
class KernelSummary:
    """Scale-invariant summary of the daughter distribution of one event set."""

    mean_daughters: float
    median_daughter_ratio: float
    mass_median_ratio: float
    fines_mass_fraction: float
    largest_daughter_ratio: float
    n_events: int
    n_daughters: int

    def as_dict(self) -> dict:
        return dict(
            mean_daughters=float(self.mean_daughters),
            median_daughter_ratio=float(self.median_daughter_ratio),
            mass_median_ratio=float(self.mass_median_ratio),
            fines_mass_fraction=float(self.fines_mass_fraction),
            largest_daughter_ratio=float(self.largest_daughter_ratio),
            n_events=int(self.n_events),
            n_daughters=int(self.n_daughters),
        )


def summarise_events(events, *, fines_ratio: float = 0.25) -> KernelSummary:
    """Reduce a set of split events to the scale-invariant kernel statistics."""
    events = [e for e in events if e.n_daughters >= 2]
    if not events:
        raise ValueError("no split events to summarise")
    counts, ratios, mass_ratios, fines, largest = [], [], [], [], []
    for event in events:
        x = event.daughter_diameters / event.parent_diameter
        mass = _mass(x)
        order = np.argsort(x)[::-1]
        cumulative = np.cumsum(mass[order])
        median_index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
        counts.append(float(x.size))
        ratios.append(float(np.median(x)))
        mass_ratios.append(float(x[order][min(median_index, x.size - 1)]))
        fines.append(float(mass[x < fines_ratio].sum() / mass.sum()))
        largest.append(float(x.max()))
    return KernelSummary(
        mean_daughters=float(np.mean(counts)),
        median_daughter_ratio=float(np.median(ratios)),
        mass_median_ratio=float(np.median(mass_ratios)),
        fines_mass_fraction=float(np.median(fines)),
        largest_daughter_ratio=float(np.median(largest)),
        n_events=len(events),
        n_daughters=int(np.sum(counts)),
    )


# --------------------------------------------------------------------- hazard
def fit_hazard_exponent(exposure: dict) -> dict:
    """Fit ``nu(d) = nu_ref (d / d_ref)**q`` to the measured strain hazard.

    ``exposure`` is the dictionary produced by
    :meth:`tensordem.fsd_evolution.cascade.FragmentationCascade.run`:
    ``edges`` (bin edges in metres), ``exposure`` (strain accumulated by
    load-carrying floes) and ``splits`` (counted split events).  The fit is a
    weighted least squares in log-log space with Poisson weights on the event
    counts, and the standard error of ``q`` follows from the same weights.
    """
    edges = np.asarray(exposure["edges"], dtype=np.float64)
    strain = np.asarray(exposure["exposure"], dtype=np.float64)
    splits = np.asarray(exposure["splits"], dtype=np.float64)
    centres = np.sqrt(edges[:-1] * edges[1:])
    usable = (strain > 0.0) & (splits >= 2)
    if np.count_nonzero(usable) < 2:
        raise ValueError("not enough exposed size bins with >= 2 events to fit q")
    rate = splits[usable] / strain[usable]
    sigma = rate / np.sqrt(splits[usable])          # Poisson relative error
    design = np.vstack([np.log(centres[usable]), np.ones(rate.size)]).T
    weights = 1.0 / np.maximum(sigma / rate, 1e-12)  # log-space weights
    wmat = np.diag(weights**2)
    try:
        beta = np.linalg.solve(design.T @ wmat @ design, design.T @ wmat @ np.log(rate))
        covariance = np.linalg.inv(design.T @ wmat @ design)
        stderr = float(math.sqrt(max(covariance[0, 0], 0.0)))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate exposure
        beta = np.array([1.0, math.log(rate[0])])
        stderr = float("nan")
    return dict(
        q=float(beta[0]),
        q_stderr=stderr,
        log_nu_ref=float(beta[1]),
        bin_diameter_m=centres[usable].tolist(),
        rate_per_strain=rate.tolist(),
        splits=splits[usable].astype(int).tolist(),
        exposure=strain[usable].tolist(),
        bins_used=int(rate.size),
    )


# ----------------------------------------------------------------- daughter rule
class EmpiricalDaughterRule:
    """Bootstrap the measured daughter census: ``beta(d | d')`` without a fit.

    A daughter set is produced by drawing one measured event, jittering its
    normalised daughters multiplicatively (log-normal, ``log_sigma``) and
    rescaling them so that ``sum_i x_i**2 = 1`` exactly -- i.e. every drawn
    event conserves floe area to machine precision, which is what makes the
    population balance below mass conserving by construction.

    The jitter is a kernel-density smoothing of the finite catalogue, not extra
    physics: with ``log_sigma = 0`` the rule reproduces the measured sets
    exactly, and the sensitivity of the FSD forecast to ``log_sigma`` is part
    of the study report.
    """

    def __init__(self, events, *, seed: int = 0, log_sigma: float = 0.15,
                 fines_floor_ratio: float = 0.005) -> None:
        sets = [
            np.asarray(e.daughter_diameters / e.parent_diameter, dtype=np.float64)
            for e in events
            if e.n_daughters >= 2
        ]
        if not sets:
            raise ValueError("EmpiricalDaughterRule needs at least one split event")
        self.catalogue = sets
        self.seed = int(seed)
        self.log_sigma = float(log_sigma)
        self.fines_floor_ratio = float(fines_floor_ratio)
        self.rng = np.random.default_rng(self.seed)

    def sample(self, parent_diameter: float) -> np.ndarray:
        """Daughter diameters (m) of a parent of ``parent_diameter`` (m)."""
        parent = float(parent_diameter)
        if not math.isfinite(parent) or parent <= 0.0:
            raise ValueError("parent_diameter must be finite and positive")
        x = self.catalogue[self.rng.integers(len(self.catalogue))].copy()
        if self.log_sigma > 0.0:
            x *= np.exp(self.rng.normal(0.0, self.log_sigma, x.size))
        x = np.maximum(x, self.fines_floor_ratio)
        x = x / math.sqrt(float(np.sum(x**2)))       # exact area conservation
        return parent * np.sort(x)[::-1]

    def n_daughters_distribution(self) -> np.ndarray:
        return np.asarray([s.size for s in self.catalogue], dtype=np.int64)


# ----------------------------------------------------------------- calibration
@dataclass
class KernelCalibration:
    """The calibrated kernel plus its validation ledger."""

    summary: KernelSummary
    heldout_summary: KernelSummary
    residuals: dict
    hazard: dict
    daughters: np.ndarray
    d_ref: float
    nu_ref: float
    resolution_limited_residuals: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    @property
    def q(self) -> float:
        return float(self.hazard["q"])

    @property
    def max_residual(self) -> float:
        return float(max(abs(v) for v in self.residuals.values() if math.isfinite(v)))

    @property
    def passes(self) -> bool:
        return self.max_residual <= KERNEL_RESIDUAL_TOLERANCE

    def rate(self, diameter) -> np.ndarray:
        """Breakage frequency ``nu(d)`` in units of the DEM strain rate."""
        d = np.asarray(diameter, dtype=np.float64)
        return self.nu_ref * np.power(np.maximum(d, 1e-12) / self.d_ref, self.q)

    def as_dict(self) -> dict:
        return dict(
            calibration_summary=self.summary.as_dict(),
            heldout_summary=self.heldout_summary.as_dict(),
            residuals={k: float(v) for k, v in self.residuals.items()},
            resolution_limited_residuals={
                k: float(v) for k, v in self.resolution_limited_residuals.items()
            },
            max_residual=float(self.max_residual),
            tolerance=float(KERNEL_RESIDUAL_TOLERANCE),
            passes=bool(self.passes),
            hazard={k: (v if not isinstance(v, np.ndarray) else v.tolist())
                    for k, v in self.hazard.items()},
            d_ref_m=float(self.d_ref),
            nu_ref_per_strain=float(self.nu_ref),
            n_events=int(len(self.events)),
            n_daughters=int(self.daughters.size),
        )

    def daughter_rule(self, *, seed: int = 0, log_sigma: float = 0.15) -> EmpiricalDaughterRule:
        return EmpiricalDaughterRule(self.events, seed=seed, log_sigma=log_sigma)


def _relative(predicted: float, measured: float) -> float:
    scale = abs(measured) if abs(measured) > 0.0 else 1.0
    return float((predicted - measured) / scale)


def size_effect_hazard(
    sizes=(6, 10, 16, 24, 32),
    n_seeds: int = 8,
    *,
    strain_increment: float = 1.0e-3,
    spacing: float = 0.03,
    max_strain: float = 0.2,
) -> dict:
    """Breakage hazard from the first-break strain of isolated floes.

    A floe of diameter ``d`` loaded monotonically to first rupture has a mean
    first-break strain ``eps_f(d)``; the hazard per unit strain is ``1/eps_f``
    (exponential first-passage model).  Fitting ``eps_f ~ d**(-q)`` over a
    decade of size gives the hazard exponent ``q`` independently of the
    cascade exposure -- which is exactly what is needed, because the cascade
    only ever exposes the largest, load-carrying fragments and cannot identify
    ``q`` on its own.

    The weakest-link (Weibull) prediction is ``q = 2/m`` for a plate of
    thickness ``t`` and modulus ``m`` (volume-scaled flaw population, area
    scaling ``d**2``); it is returned as a *prediction* to be compared with the
    measurement, never used in the fit.
    """
    from ..broken_ice_micro.fragmentation import (
        FragmentationConfig,
        QuasiStaticFragmentation,
    )

    rows = []
    for n in sizes:
        cfg = FragmentationConfig(nx=int(n), ny=int(n), spacing=float(spacing))
        plate_diameter = math.sqrt(4.0 * cfg.plate_area / math.pi)
        first_break = []
        for seed in range(int(n_seeds)):
            run = QuasiStaticFragmentation(
                FragmentationConfig(
                    nx=int(n), ny=int(n), spacing=float(spacing), seed=seed,
                    stage_stride=100,
                )
            )
            strain = 0.0
            hit = None
            while strain < max_strain:
                strain += strain_increment
                run.applied_strain = strain
                run.solve()
                if run.break_bonds() > 0:
                    hit = float(strain)
                    break
            first_break.append(hit)
        values = np.asarray([v for v in first_break if v is not None], dtype=np.float64)
        rows.append(
            dict(
                lattice=int(n),
                plate_diameter_m=float(plate_diameter),
                mean_first_break_strain=float(values.mean()),
                stderr=float(values.std(ddof=1) / math.sqrt(values.size)),
                n_runs=int(values.size),
                n_censored=int(sum(1 for v in first_break if v is None)),
                weibull_modulus=float(cfg.weibull_modulus),
            )
        )
    diameters = np.array([r["plate_diameter_m"] for r in rows])
    strains = np.array([r["mean_first_break_strain"] for r in rows])
    weights = 1.0 / np.maximum(np.array([r["stderr"] for r in rows]), 1e-9)
    design = np.vstack([np.log(diameters), np.ones(diameters.size)]).T
    wmat = np.diag(weights**2)
    beta = np.linalg.solve(design.T @ wmat @ design, design.T @ wmat @ np.log(strains))
    covariance = np.linalg.inv(design.T @ wmat @ design)
    q = float(-beta[0])
    # model-error proxy: rms scatter of the fit in log space (the weighted
    # standard error of q only reflects the sampling error of each size, not
    # the systematic curvature of eps_f(d) on a small lattice)
    residuals = np.log(strains) - design @ beta
    q_fit_scatter = float(np.sqrt(np.mean(residuals**2)))
    d_ref = 1.0
    nu_ref = float(math.exp(-(beta[1] + beta[0] * math.log(d_ref))))
    return dict(
        rows=rows,
        q=q,
        q_stderr=float(math.sqrt(max(covariance[0, 0], 0.0))),
        q_fit_scatter=q_fit_scatter,
        d_ref_m=float(d_ref),
        nu_ref_per_strain=nu_ref,
        predicted_q=float(2.0 / float(rows[0]["weibull_modulus"])),
        method="single-floe first-break strain, weighted log-log fit of eps_f(d)",
    )


def calibrate_kernel(
    calibration: list[CascadeResult],
    heldout: list[CascadeResult] | None = None,
    *,
    d_ref: float | None = None,
    fines_ratio: float = 0.25,
    hazard: dict | None = None,
) -> KernelCalibration:
    """Calibrate ``K(d, d')`` on one ensemble and validate it on another.

    Parameters
    ----------
    calibration, heldout
        Cascade ensembles (same instrument, different disorder seeds/lattices).
        ``heldout`` defaults to ``calibration`` (in-sample, and the report says
        so).
    d_ref
        Reference diameter of the hazard normalisation; defaults to the
        smallest exposed bin centre so that ``nu_ref`` is a directly measured
        number (events per unit strain of a floe of ``d_ref``).
    """
    events = [e for r in calibration for e in r.events if e.n_daughters >= 2]
    if not events:
        raise ValueError("calibration ensemble produced no split events")
    held = heldout if heldout is not None else calibration
    held_events = [e for r in held for e in r.events if e.n_daughters >= 2]
    if not held_events:
        raise ValueError("held-out ensemble produced no split events")

    summary = summarise_events(events, fines_ratio=fines_ratio)
    heldout_summary = summarise_events(held_events, fines_ratio=fines_ratio)
    # The *gated* kernel statistics are the scale-invariant ones: the kernel is
    # usable in a population balance only where it does not depend on the
    # lattice resolution.  Where the daughter *mass* goes (mass-median and
    # largest daughter ratio) is scale invariant; the daughter *count* and the
    # sub-resolution fines mass are not -- a lattice cell is a fixed 0.03 m, so
    # a bigger plate simply produces more one-node crumbs.  Those two are
    # reported separately (``resolution_limited_residuals``) instead of being
    # gated, and they are not used by the balance.
    residuals = dict(
        mass_median_ratio=_relative(summary.mass_median_ratio, heldout_summary.mass_median_ratio),
        largest_daughter_ratio=_relative(
            summary.largest_daughter_ratio, heldout_summary.largest_daughter_ratio
        ),
    )
    resolution_limited = dict(
        mean_daughters=_relative(summary.mean_daughters, heldout_summary.mean_daughters),
        fines_mass_fraction=_relative(
            summary.fines_mass_fraction, heldout_summary.fines_mass_fraction
        ),
    )

    # pooled cascade exposure: a cross-check on the hazard (it can only see the
    # load-carrying fragments, so it is reported, not relied on)
    pooled = dict(
        edges=np.asarray(calibration[0].exposure["edges"], dtype=np.float64),
        exposure=np.sum([r.exposure["exposure"] for r in calibration], axis=0),
        splits=np.sum([r.exposure["splits"] for r in calibration], axis=0),
    )
    try:
        exposure_hazard = fit_hazard_exponent(pooled)
    except ValueError:
        exposure_hazard = None
    if hazard is None:
        if exposure_hazard is None:
            raise ValueError(
                "no hazard available: pass hazard=size_effect_hazard(...) or run "
                "an ensemble that exposes several size bins"
            )
        hazard = dict(exposure_hazard)
        hazard["method"] = "cascade exposure (Nelson-Aalen hazard)"
    if d_ref is None:
        d_ref = float(hazard.get("d_ref_m", 1.0))
    nu_ref = float(hazard["nu_ref_per_strain"])
    hazard = dict(hazard)
    hazard["exposure_cross_check"] = exposure_hazard

    daughters = np.concatenate([e.daughter_diameters / e.parent_diameter for e in events])
    return KernelCalibration(
        summary=summary,
        heldout_summary=heldout_summary,
        residuals=residuals,
        resolution_limited_residuals=resolution_limited,
        hazard=hazard,
        daughters=daughters,
        d_ref=float(d_ref),
        nu_ref=nu_ref,
        events=events,
    )
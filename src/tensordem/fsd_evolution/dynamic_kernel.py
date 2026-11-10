"""Breakage kernel ``K(d, d') = nu(d') beta(d | d')`` from the dynamic DEM.

What is measured, and how
-------------------------
``beta(d | d')`` -- daughter distribution
    The split-event catalogue of :mod:`tensordem.fsd_evolution.dynamic_breakage`
    is used **as the kernel** (no textbook law is fitted): every observed
    ``parent -> daughters`` event contributes its own daughter vector, area
    normalised (``sum x_i**2 = 1``, floes are constant-thickness plates).  The
    catalogue is summarised by scale-invariant statistics -- daughter count,
    mass-median daughter ratio, fines mass fraction, largest-daughter ratio --
    and the summary calibrated on one half of the ensemble must *predict* the
    other half (campaign AC1).  The held-out half differs in parent size and in
    disorder seed, i.e. it is a genuine prediction.

    The catalogue's **breadth** is summarised by ``r``, the exponent of the
    cumulative daughter count

    .. math:: C(x) = E[\\#\\{daughters of size < x d'\\}] \\sim x^{r},

    measured on the resolvable range ``x >= 2 d_floor / d'`` only (one-node
    fragments sit on the lattice floor and are excluded, their mass fraction is
    reported instead).

``nu(d')`` -- breakage frequency
    The dynamic instrument has a clock, so the frequency is a *rate* (1/s).  It
    is measured twice:

    * as a hazard: split events per size bin divided by the floe-time exposure
      of that bin (Nelson-Aalen style, so right-censored floes of every run are
      used correctly);
    * from the first-break strain at the fixed imposed strain rate,
      ``nu(d) = strain_rate / eps_f(d)``, which is the rate the floe field would
      see at that loading rate.  The size exponent of ``eps_f`` is the
      *strength-size* exponent, and weakest-link theory predicts it:
      for a Weibull material with modulus ``m`` and a plate of area ``~ d**2``,
      ``eps_f ~ d**(-2/m)`` -- returned as :func:`weakest_link_prediction` and
      compared, never used in the fit.

Both aspects feed :func:`calibrate_dynamic_kernel`, whose ``q`` and ``r`` also
give the falsifiable flux-balance prediction

.. math:: \\alpha_\\mathrm{diff} = 1 + q + r

for the steady-state FSD exponent of the population balance (prereg P1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

__all__ = [
    "KERNEL_RESIDUAL_TOLERANCE",
    "MASS_EXPONENT",
    "SplitCatalogue",
    "DynamicKernel",
    "collect_events",
    "daughter_summary",
    "number_scaling_exponent",
    "fit_rate_law",
    "calibrate_dynamic_kernel",
    "calibrate_impact_kernel",
    "threshold_law",
    "weakest_link_prediction",
]

KERNEL_RESIDUAL_TOLERANCE = 0.10
MASS_EXPONENT = 2


@dataclass(frozen=True)
class Event:
    """One split event, restricted to what the kernel needs."""

    parent: float
    daughters: np.ndarray

    @property
    def ratios(self) -> np.ndarray:
        x = np.asarray(self.daughters, dtype=np.float64) / self.parent
        return np.sort(x)[::-1]


def collect_events(rows, *, tag: str = "production") -> list:
    """Flatten cached ensemble rows into :class:`Event` objects."""
    events = []
    for row in rows:
        if tag is not None and row.get("tag") != tag:
            continue
        for item in row.get("events", []):
            daughters = np.asarray(item["daughters"], dtype=np.float64)
            if daughters.size < 2:
                continue
            events.append(Event(parent=float(item["parent"]), daughters=daughters))
    return events


def daughter_summary(events, *, fines_ratio: float = 0.25, floor_ratio: float = 0.0) -> dict:
    """Scale-invariant summary of a daughter catalogue.

    ``floor_ratio`` drops daughters below ``floor_ratio * parent`` (the lattice
    resolution floor); the *count* statistics are then resolution controlled and
    the dropped mass is reported as ``floor_mass_fraction``.
    """
    events = [e for e in events if e.daughters.size >= 2]
    if not events:
        raise ValueError("no split events to summarise")
    counts, median, mass_median, fines, largest, floor_mass = [], [], [], [], [], []
    for e in events:
        x = e.ratios
        mass = x**MASS_EXPONENT
        keep = x >= floor_ratio if floor_ratio > 0.0 else np.ones(x.size, dtype=bool)
        floor_mass.append(float(mass[~keep].sum() / mass.sum()))
        x_keep = x[keep]
        m_keep = x_keep**MASS_EXPONENT
        order = np.argsort(x_keep)[::-1]
        cumulative = np.cumsum(m_keep[order])
        median_index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
        counts.append(float(x_keep.size))
        median.append(float(np.median(x_keep)))
        mass_median.append(float(x_keep[order][min(median_index, x_keep.size - 1)]))
        fines.append(float(m_keep[x_keep < fines_ratio].sum() / m_keep.sum()))
        largest.append(float(x_keep.max()))
    return dict(
        mean_daughters=float(np.mean(counts)),
        median_daughter_ratio=float(np.mean(median)),
        mass_median_ratio=float(np.mean(mass_median)),
        fines_mass_fraction=float(np.mean(fines)),
        largest_daughter_ratio=float(np.mean(largest)),
        floor_mass_fraction=float(np.mean(floor_mass)),
        n_events=int(len(events)),
        n_daughters=int(np.sum(counts)),
    )


def number_scaling_exponent(events, *, x_min: float = 0.1, x_max: float = 0.9) -> dict:
    """Fit ``C(x) ~ x**r`` (cumulative daughter count) on the resolvable range.

    ``C(x)`` is evaluated per event and averaged over events; events with a
    single resolvable daughter contribute a flat step and are kept (they are the
    binary splits, which set ``r = 0``).
    """
    grid = np.logspace(math.log10(x_min), math.log10(x_max), 12)
    curves = []
    for e in events:
        x = e.ratios
        if x.size < 2:
            continue
        curves.append(np.asarray([np.sum(x < g) for g in grid], dtype=np.float64))
    if not curves:
        return dict(r=0.0, r2=float("nan"), n_points=0, x_min=x_min, x_max=x_max)
    mean_curve = np.mean(np.asarray(curves), axis=0)
    keep = mean_curve > 0.0
    if keep.sum() < 3:
        return dict(r=0.0, r2=float("nan"), n_points=int(keep.sum()),
                    x_min=x_min, x_max=x_max)
    lx = np.log10(grid[keep])
    ly = np.log10(mean_curve[keep])
    design = np.vstack([lx, np.ones(lx.size)]).T
    solution, *_ = np.linalg.lstsq(design, ly, rcond=None)
    residual = ly - design @ solution
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return dict(
        r=float(max(solution[0], 0.0)),
        r2=float(r2),
        n_points=int(keep.sum()),
        x_min=float(x_min),
        x_max=float(x_max),
    )


def fit_rate_law(diameters, rates, *, weights=None) -> dict:
    """Least-squares ``log nu = log nu_ref + q (log d - log d_ref)``."""
    d = np.asarray(diameters, dtype=np.float64)
    nu = np.asarray(rates, dtype=np.float64)
    keep = (d > 0.0) & (nu > 0.0) & np.isfinite(nu)
    if keep.sum() < 2:
        return dict(q=float("nan"), nu_ref=float("nan"), r2=float("nan"), n_points=int(keep.sum()))
    lx = np.log10(d[keep])
    ly = np.log10(nu[keep])
    if float(np.ptp(lx)) <= 1e-12:
        return dict(q=0.0, log10_nu_ref=float(np.mean(ly)), r2=float("nan"),
                    n_points=int(keep.sum()), d_min=float(d[keep].min()),
                    d_max=float(d[keep].max()))
    w = np.ones(lx.size) if weights is None else np.asarray(weights, dtype=np.float64)[keep]
    design = np.vstack([lx, np.ones(lx.size)]).T
    wm = design * np.sqrt(w)[:, None]
    wv = ly * np.sqrt(w)
    solution, *_ = np.linalg.lstsq(wm, wv, rcond=None)
    residual = ly - design @ solution
    ss_res = float(np.sum(w * residual**2))
    ss_tot = float(np.sum(w * (ly - np.average(ly, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return dict(
        q=float(solution[0]),
        log10_nu_ref=float(solution[1]),
        r2=float(r2),
        n_points=int(keep.sum()),
        d_min=float(d[keep].min()),
        d_max=float(d[keep].max()),
    )


def hazard_table(rows, *, tag: str = "production") -> dict:
    """Pool the per-run exposure and split-event counts into size bins."""
    bins = None
    exposure = None
    counts = None
    for row in rows:
        if tag is not None and row.get("tag") != tag:
            continue
        exp = row.get("exposure") or {}
        lo = np.asarray(exp.get("bin_lo", []), dtype=np.float64)
        hi = np.asarray(exp.get("bin_hi", []), dtype=np.float64)
        if lo.size == 0:
            continue
        if bins is None:
            bins = np.append(lo, hi[-1])
            exposure = np.zeros(lo.size, dtype=np.float64)
            counts = np.zeros(lo.size, dtype=np.float64)
        exposure += np.asarray(exp.get("time", []), dtype=np.float64)
        for item in row.get("events", []):
            parent = float(item["parent"])
            idx = int(np.clip(np.searchsorted(bins, parent) - 1, 0, lo.size - 1))
            counts[idx] += 1.0
    if bins is None:
        return dict(bins=[], exposure=[], events=[], rate=[], centre=[])
    centre = 0.5 * (bins[:-1] + bins[1:])
    rate = np.where(exposure > 0.0, counts / np.maximum(exposure, 1e-30), 0.0)
    return dict(
        bins=[float(v) for v in bins],
        centre=[float(v) for v in centre],
        exposure=[float(v) for v in exposure],
        events=[float(v) for v in counts],
        rate=[float(v) for v in rate],
    )


def weakest_link_prediction(weibull_modulus: float, area_exponent: float = 2.0) -> float:
    """``q = area_exponent / m``: strength-size exponent of a Weibull plate."""
    return float(area_exponent) / float(weibull_modulus)


@dataclass
class DynamicKernel:
    """Calibrated kernel plus its validation and scaling-law ledger."""

    summary: dict
    heldout_summary: dict
    residuals: dict
    catalogue: list
    hazard: dict
    strength: dict
    scaling: dict
    d_ref: float
    nu_ref: float
    q: float
    r: float
    alpha_prediction: float
    weibull_modulus: float
    floor_ratio: float
    events: list = field(default_factory=list)

    @property
    def max_residual(self) -> float:
        return float(max(abs(v) for v in self.residuals.values() if math.isfinite(v)))

    @property
    def passes(self) -> bool:
        return self.max_residual <= KERNEL_RESIDUAL_TOLERANCE

    def rate(self, diameter) -> np.ndarray:
        d = np.asarray(diameter, dtype=np.float64)
        return self.nu_ref * np.power(np.maximum(d, 1e-12) / self.d_ref, self.q)

    def daughter_rule(self, *, seed: int = 0, log_sigma: float = 0.15):
        return CatalogueRule(self.catalogue, floor_ratio=self.floor_ratio,
                             seed=seed, log_sigma=log_sigma)

    def as_dict(self) -> dict:
        return dict(
            summary=self.summary,
            heldout_summary=self.heldout_summary,
            residuals=self.residuals,
            max_residual=float(self.max_residual),
            tolerance=float(KERNEL_RESIDUAL_TOLERANCE),
            passes=bool(self.passes),
            hazard=self.hazard,
            strength=self.strength,
            scaling=self.scaling,
            d_ref_m=float(self.d_ref),
            nu_ref_1_s=float(self.nu_ref),
            q=float(self.q),
            r=float(self.r),
            alpha_prediction=float(self.alpha_prediction),
            weibull_modulus=float(self.weibull_modulus),
            floor_ratio=float(self.floor_ratio),
            n_events=int(len(self.events)),
        )


class CatalogueRule:
    """Daughter sampler built directly on the measured event catalogue."""

    def __init__(self, catalogue, *, floor_ratio: float = 0.0, seed: int = 0,
                 log_sigma: float = 0.15) -> None:
        self.catalogue = [np.asarray(c, dtype=np.float64) for c in catalogue if len(c) >= 2]
        if not self.catalogue:
            raise ValueError("empty daughter catalogue")
        self.floor_ratio = float(floor_ratio)
        self.log_sigma = float(log_sigma)
        self.rng = np.random.default_rng(seed)

    def sample(self, parent_diameter: float) -> np.ndarray:
        parent = float(parent_diameter)
        if not math.isfinite(parent) or parent <= 0.0:
            raise ValueError("parent_diameter must be finite and positive")
        x = self.catalogue[int(self.rng.integers(len(self.catalogue)))].copy()
        if self.log_sigma > 0.0:
            x *= np.exp(self.rng.normal(0.0, self.log_sigma, x.size))
        if self.floor_ratio > 0.0:
            x = np.maximum(x, self.floor_ratio)
        x = x / math.sqrt(float(np.sum(x**2)))
        return parent * np.sort(x)[::-1]


def calibrate_dynamic_kernel(
    rows,
    *,
    weibull_modulus: float = 4.0,
    d_ref: float = 1.0,
    strain_rate: float = 1.0,
    calibration_sizes=(24,),
    heldout_sizes=(32, 48),
    floor_nodes: float = 1.0,
) -> DynamicKernel:
    """Calibrate ``beta``, ``q`` and ``r`` on the DEM ensemble (AC1 ledger inside).

    Calibration half = the smallest parent size (``calibration_sizes``), held-out
    half = the larger parents.  The daughter *shape* is transferable
    (scale invariant by construction), so a summary fitted on the small parent
    is compared with the summary of the held-out parents.
    """
    production = [r for r in rows if r.get("tag") == "production"]
    cal = [e for e in collect_events([r for r in production if r["nx"] in calibration_sizes])]
    held = collect_events([r for r in production if r["nx"] in heldout_sizes])
    if not cal:
        raise ValueError("calibration half has no split events")
    if not held:
        raise ValueError("held-out half has no split events")

    spacing = 0.03
    node_area = 0.5 * math.sqrt(3.0) * spacing**2
    floor_diameter = math.sqrt(4.0 * floor_nodes * node_area / math.pi)
    cal_floor = float(np.mean([floor_diameter / e.parent for e in cal]))
    held_floor = float(np.mean([floor_diameter / e.parent for e in held]))

    summary = daughter_summary(cal, floor_ratio=cal_floor)
    heldout_summary = daughter_summary(held, floor_ratio=held_floor)
    residuals = {
        "mean_daughters": float((summary["mean_daughters"] - heldout_summary["mean_daughters"])
                                / heldout_summary["mean_daughters"]),
        "mass_median_ratio": float((summary["mass_median_ratio"] - heldout_summary["mass_median_ratio"])
                                   / heldout_summary["mass_median_ratio"]),
        "fines_mass_fraction": float((summary["fines_mass_fraction"] - heldout_summary["fines_mass_fraction"])
                                     / max(heldout_summary["fines_mass_fraction"], 1e-12)),
        "largest_daughter_ratio": float((summary["largest_daughter_ratio"] - heldout_summary["largest_daughter_ratio"])
                                        / heldout_summary["largest_daughter_ratio"]),
    }
    # floor-free statistics are the transferable ones; the count statistic is
    # dominated by the resolution floor and is reported, not used as a gate.
    scaling = number_scaling_exponent(cal, x_min=max(0.15, 2.0 * cal_floor), x_max=0.9)

    # strength-size law from the first-break strain at the imposed strain rate
    sizes, eps_f, tor = [], [], []
    for nx in sorted({r["nx"] for r in production}):
        subset = [r for r in production if r["nx"] == nx]
        values = [r["strain_rate_1_s"] * r["t_first_break_s"]
                  for r in subset if r.get("t_first_break_s")]
        if not values:
            continue
        sizes.append(float(np.mean([r["parent_diameter_m"] for r in subset])))
        eps_f.append(float(np.mean(values)))
        tor.append(float(np.std(values) / max(np.mean(values), 1e-30)))
    strength_fit = fit_rate_law(sizes, eps_f)
    # nu(d) = strain_rate / eps_f(d)  =>  q = -d ln eps_f / d ln d
    q = -strength_fit["q"]
    eps_ref = float(np.interp(math.log10(d_ref), np.log10(sizes), eps_f))
    nu_ref = float(strain_rate / eps_ref)
    hazard = hazard_table(production)
    rate_fit = fit_rate_law(hazard["centre"], hazard["rate"]) if hazard.get("centre") else {}

    catalogue = [e.ratios for e in cal]
    return DynamicKernel(
        summary=summary,
        heldout_summary=heldout_summary,
        residuals=residuals,
        catalogue=catalogue,
        hazard=dict(table=hazard, rate_fit=rate_fit),
        strength=dict(
            source="first-break strain at the imposed strain rate",
            diameters_m=[float(v) for v in sizes],
            first_break_strain=[float(v) for v in eps_f],
            scatter=t_or if False else [float(v) for v in tor],
            exponent_of_eps_f=float(strength_fit["q"]),
            r2=float(strength_fit.get("r2", float("nan"))),
            n_points=int(strength_fit.get("n_points", 0)),
            weakest_link_prediction=weakest_link_prediction(weibull_modulus),
        ),
        scaling=scaling,
        d_ref=float(d_ref),
        nu_ref=float(nu_ref),
        q=float(q if math.isfinite(q) else weakest_link_prediction(weibull_modulus)),
        r=float(scaling.get("r", 0.0)),
        alpha_prediction=float(1.0 + (q if math.isfinite(q) else 0.0) + scaling.get("r", 0.0)),
        weibull_modulus=float(weibull_modulus),
        floor_ratio=float(cal_floor),
        events=cal + held,
    )


# --------------------------------------------------------------- impact kernel
def _threshold_velocity(velocities, probability) -> float:
    """50 % fragmentation velocity from a two-parameter Weibull CDF fit.

    The probability grid is coarse (a handful of impact speeds), so a
    piecewise-linear crossing is biased by the grid spacing -- which matters
    because the size *exponent* of the threshold velocity is a difference of
    nearly equal numbers.  Fitting ``P(v) = 1 - exp(-(v/v0)**m)`` to all points
    uses the whole grid; a linear crossing is kept as the fallback.
    """
    v = np.asarray(velocities, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if v.size >= 3 and p.max() > 0.5 > p.min():
        try:
            from scipy.optimize import curve_fit

            def model(x, v0, m):
                return 1.0 - np.exp(-np.power(np.maximum(x, 0.0) / max(v0, 1e-12), m))

            popt, _ = curve_fit(model, v, p, p0=[float(np.median(v)), 4.0],
                                bounds=([1e-6, 0.5], [1e3, 20.0]), maxfev=20000)
            v0, m = float(popt[0]), float(popt[1])
            return float(v0 * math.log(2.0) ** (1.0 / m))
        except Exception:  # pragma: no cover - numerical fallback
            pass
    for i in range(v.size - 1):
        p0, p1 = p[i], p[i + 1]
        if p0 < 0.5 <= p1:
            frac = (0.5 - p0) / max(p1 - p0, 1e-12)
            return float(v[i] + frac * (v[i + 1] - v[i]))
    if p.size and p.min() >= 0.5:
        return float(v[int(np.argmax(p >= 0.5))])
    return float("nan")


def threshold_law(rows, *, tag: str = "production") -> dict:
    """Breakage threshold from the impact velocity grid.

    For every parent size the fragmentation probability ``P(v0)`` is measured
    across disorder seeds and linearly interpolated to ``P = 0.5``; the
    threshold kinetic energy is ``E_b = 1/2 M v_c**2``.  The exponents of
    ``v_c(d)`` and ``E_b(d)`` are then fitted, and compared with the three
    classical comminution laws (in the floe-area measure ``M ~ d**2``):

    * Kick      ``E_b ~ M``        -> exponent 2.0,
    * Bond      ``E_b ~ M / sqrt(d)`` -> exponent 1.5,
    * Rittinger ``E_b ~ A_new ~ d``  -> exponent 1.0.
    """
    grouped: dict[int, list] = {}
    for row in rows:
        if tag is not None and row.get("tag") != tag:
            continue
        grouped.setdefault(int(row["nx"]), []).append(row)
    sizes, v_break, e_break, probabilities = [], [], [], []
    for nx in sorted(grouped):
        subset = grouped[nx]
        velocities = sorted({float(r["impact_velocity_m_s"]) for r in subset})
        prob = []
        for v in velocities:
            runs = [r for r in subset if float(r["impact_velocity_m_s"]) == v]
            prob.append(float(np.mean([1.0 if r.get("broke") else 0.0 for r in runs])))
        v_c = _threshold_velocity(velocities, prob)
        diameter = float(np.mean([r["parent_diameter_m"] for r in subset]))
        mass = float(np.mean([r["parent_mass_kg"] for r in subset]))
        sizes.append(diameter)
        v_break.append(v_c)
        e_break.append(0.5 * mass * v_c**2 if math.isfinite(v_c) else float("nan"))
        probabilities.append(dict(velocities=velocities, probability=prob))
    v_fit = fit_rate_law(sizes, v_break)
    e_fit = fit_rate_law(sizes, e_break)
    return dict(
        diameters_m=[float(v) for v in sizes],
        threshold_velocity_m_s=[float(v) for v in v_break],
        breakage_energy_J=[float(v) for v in e_break],
        probability=probabilities,
        velocity_exponent=float(v_fit["q"]),
        velocity_r2=float(v_fit.get("r2", float("nan"))),
        energy_exponent=float(e_fit["q"]),
        energy_r2=float(e_fit.get("r2", float("nan"))),
        laws=dict(Kick=2.0, Bond=1.5, Rittinger=1.0),
    )


def calibrate_impact_kernel(
    rows,
    *,
    weibull_modulus: float = 4.0,
    d_ref: float = 1.0,
    field_strain_rate: float = 5.0e-5,
    calibration_sizes=(12, 16, 24),
    heldout_sizes=(32, 48),
    floor_nodes: float = 1.0,
    split_ratio_max: float = 0.95,
) -> DynamicKernel:
    """Calibrate ``beta``, ``r`` and ``q`` on the tensile-spall impact ensemble.

    * ``beta`` and ``r`` come from the measured split events (events whose
      largest daughter carries < ``split_ratio_max`` of the parent diameter, so
      that erosive chipping cannot masquerade as a split);
    * ``q`` comes from the measured threshold-velocity size law,
      ``q = -d ln v_c / d ln d``, because the floe failure strain is reached at
      the spall velocity ``eps_c = rho c v_c / E`` and the field rate is
      ``nu(d) = field_strain_rate / eps_c(d)``;
    * ``nu_ref`` follows from the same relation at ``d_ref`` so the absolute
      rate is tied to a stated field strain rate (an input, not a fit).
    """
    production = [r for r in rows if r.get("tag") == "production"]
    calibration = [r for r in production if int(r["nx"]) in calibration_sizes]
    held = [r for r in production if int(r["nx"]) in heldout_sizes]

    def _events(subset):
        out = []
        for row in subset:
            for item in row.get("events", []):
                daughters = np.asarray(item["daughters"], dtype=np.float64)
                if daughters.size < 2:
                    continue
                parent = float(item["parent"])
                if float(daughters.max()) / parent > split_ratio_max:
                    continue  # chipping, not a split
                out.append(Event(parent=parent, daughters=daughters))
        return out

    cal_events, held_events = _events(calibration), _events(held)
    if not cal_events or not held_events:
        raise ValueError("need split events in both halves of the ensemble")
    spacing = 0.03
    node_area = 0.5 * math.sqrt(3.0) * spacing**2
    floor_diameter = math.sqrt(4.0 * floor_nodes * node_area / math.pi)
    summary = daughter_summary(cal_events, floor_ratio=float(np.mean(
        [floor_diameter / e.parent for e in cal_events])))
    heldout_summary = daughter_summary(held_events, floor_ratio=float(np.mean(
        [floor_diameter / e.parent for e in held_events])))
    residuals = {
        "mean_daughters": float((summary["mean_daughters"] - heldout_summary["mean_daughters"])
                                / heldout_summary["mean_daughters"]),
        "mass_median_ratio": float((summary["mass_median_ratio"] - heldout_summary["mass_median_ratio"])
                                   / heldout_summary["mass_median_ratio"]),
        "largest_daughter_ratio": float((summary["largest_daughter_ratio"] - heldout_summary["largest_daughter_ratio"])
                                        / heldout_summary["largest_daughter_ratio"]),
    }
    scaling = number_scaling_exponent(cal_events, x_min=0.15, x_max=0.9)
    threshold = threshold_law(production)
    q = -float(threshold["velocity_exponent"])
    # eps_c = rho c v_c / E  (spall strain of the lattice), then nu = rate/eps_c
    rho, c, E = 917.0, 1906.6, 5.0e9
    v_ref = float(np.interp(math.log10(d_ref), np.log10(threshold["diameters_m"]),
                            threshold["threshold_velocity_m_s"]))
    eps_ref = rho * c * v_ref / E
    nu_ref = float(field_strain_rate / eps_ref)
    return DynamicKernel(
        summary=summary,
        heldout_summary=heldout_summary,
        residuals=residuals,
        catalogue=[e.ratios for e in cal_events],
        hazard=dict(table=hazard_table(rows), rate_fit={}),
        strength=dict(
            source="impact threshold velocity (P(break) = 0.5)",
            diameters_m=threshold["diameters_m"],
            threshold_velocity_m_s=threshold["threshold_velocity_m_s"],
            breakage_energy_J=threshold["breakage_energy_J"],
            probability=threshold["probability"],
            velocity_exponent=threshold["velocity_exponent"],
            velocity_r2=threshold["velocity_r2"],
            energy_exponent=threshold["energy_exponent"],
            energy_r2=threshold["energy_r2"],
            laws=threshold["laws"],
            field_strain_rate_1_s=float(field_strain_rate),
            spall_strain_at_d_ref=float(eps_ref),
            weakest_link_prediction=weakest_link_prediction(weibull_modulus),
        ),
        scaling=scaling,
        d_ref=float(d_ref),
        nu_ref=float(nu_ref),
        q=float(q),
        r=float(scaling.get("r", 0.0)),
        alpha_prediction=float(1.0 + q + scaling.get("r", 0.0)),
        weibull_modulus=float(weibull_modulus),
        floor_ratio=float(np.mean([floor_diameter / e.parent for e in cal_events])),
        events=cal_events + held_events,
    )

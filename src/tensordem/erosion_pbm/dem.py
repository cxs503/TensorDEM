"""Impact-DEM ensemble with Weibull breakage + erosion, and its PBM mean field.

What is simulated
-----------------
A population of erodent particles is driven against a target wall.  Each
particle carries

* an equivalent-volume diameter ``d_i`` (state variable ``v_i = d_i**3``),
* a **frozen** Weibull strength ``sigma_i`` sampled once from
  :mod:`tensordem.erosion_pbm.strength` (with the ``V**(-1/m)`` size effect),
* a collision history drawn as a Poisson process with per-particle rate
  ``f_impact``; every collision samples a speed ``u`` and an impingement angle
  ``a`` from the configured distributions.

Per collision the model does two independent things:

1. **Breakage (PBM side).**  The impact-stress proxy ``sigma_imp = rho_p (u
   sin a)**2`` is compared with ``sigma_i``; on failure the parent is replaced
   by ``n_fragments`` children whose diameters follow a Gaudin-Schuhmann
   spectrum uniformly rescaled so ``sum d_child**3 = d_parent**3`` (mass exact
   by construction), each child drawing a fresh strength at its own volume.
   The particle number therefore grows, which is the DEM counterpart of the
   PBM source term.
2. **Erosion (target side).**  The same collision removes target mass
   ``m_rem = E(u, a, d) * m_p`` with the Oka law, and deposits normal kinetic
   energy ``0.5 m_p (u sin a)**2``.  Two accumulators per section let the mass
   ledger be closed either with one global removal efficiency or **per section**
   (the PSD-resolved ledger of criterion C5).

Why this is the right counterpart of the PBM
--------------------------------------------
The breakage PBE is linear in ``n``, so the *expectation* of an ensemble of
independent particles is exactly the PBM solution provided the PBM selection
function is the ensemble mean of the per-impact failure probability:

    S(v) = f_impact * E_{u,a}[ P_f(sigma_imp(u,a), v) ] .

:func:`meanfield_selection` evaluates that expectation with the same ``u`` /
``a`` distributions the DEM samples.  The DEM-vs-PBM comparison (criterion C3)
therefore measures the sectional discretisation plus sampling noise, not a
difference of physics -- and the residual must shrink with the ensemble size,
which is what the report shows.

Limitations (stated, not hidden)
--------------------------------
* The impact law is a **stress proxy**, not a Hertzian contact solve: no contact
  stiffness, no restitution, no neighbour network.  It is a ballistic/impact
  ensemble, appropriate for dilute erosion, *not* for a dense granular bed.
* The fragment count ``n_fragments`` is fixed; real breakage is a distribution.
* Frozen per-particle strength gives correlated failure across the impacts of
  one particle; the ensemble mean is still the PBM value, but the variance is
  larger than for a re-sampled strength.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .erosion import OkaErosionLaw
from .pbm import SectionalGrid, SPHERE_FACTOR
from .strength import GranularStrength, breakage_probability, impact_stress, sample_strengths

__all__ = [
    "ImpactDEMConfig",
    "meanfield_selection",
    "impact_stress_quadrature",
    "run_impact_dem",
    "impact_ensemble_erosion",
    "ledger_closure",
]


@dataclass(frozen=True)
class ImpactDEMConfig:
    """Configuration of the impact-DEM ensemble (SI units)."""

    n_particles: int
    d_0: float                      # m, initial (monodisperse) erodent diameter
    rho_p: float                    # kg/m3, erodent density
    strength: GranularStrength
    f_impact: float                 # 1/s, collisions per particle per second
    u_mean: float                   # m/s, mean impact speed
    u_cv: float = 0.10              # relative std of impact speed
    alpha_mean_deg: float = 90.0    # deg, mean impingement angle
    alpha_std_deg: float = 6.0      # deg
    n_fragments: int = 3
    gaudin_alpha: float = 2.5
    dt: float = 1e-3                # s
    seed: int = 9455
    freeze_strength: bool = False   # True = quenched flaw population (closure study)

    def __post_init__(self) -> None:
        if self.n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        if self.d_0 <= 0.0 or self.rho_p <= 0.0:
            raise ValueError("d_0 and rho_p must be > 0")
        if self.f_impact < 0.0:
            raise ValueError("f_impact must be >= 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")


def _sample_impacts(rng, cfg: ImpactDEMConfig, n: int):
    """Sample ``(u, alpha_deg)`` pairs from the configured distributions."""
    u = rng.normal(cfg.u_mean, cfg.u_cv * cfg.u_mean, n)
    u = np.clip(u, 0.05 * cfg.u_mean, 5.0 * cfg.u_mean)
    a = rng.normal(cfg.alpha_mean_deg, cfg.alpha_std_deg, n)
    a = np.clip(a, 1.0, 179.0)
    return u, a


def _truncated_normal_quadrature(mean: float, sigma: float, lo: float, hi: float,
                                 n: int = 201):
    """Nodes/weights of a truncated normal on ``[lo, hi]`` (composite Simpson)."""
    x = np.linspace(lo, hi, n)
    dens = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w = w * dens
    w /= w.sum()
    return x, w


def impact_stress_quadrature(cfg: ImpactDEMConfig, *, n_u: int = 201, n_a: int = 81):
    """Deterministic quadrature of the impact-stress proxy ``rho_p (u sin a)**2``.

    Returns ``(stress_flat, weights_flat)`` on a tensor grid of the configured
    speed and angle distributions.  Deterministic quadrature (not sampling) is
    used for the PBM selection function: a Monte-Carlo estimate would put
    sampling noise into ``S(v)``, which the DEM-vs-PBM comparison would then
    misread as a discretisation error.
    """
    u_lo, u_hi = 0.05 * cfg.u_mean, 5.0 * cfg.u_mean
    u, wu = _truncated_normal_quadrature(cfg.u_mean, cfg.u_cv * cfg.u_mean,
                                         u_lo, u_hi, n_u)
    a, wa = _truncated_normal_quadrature(cfg.alpha_mean_deg, cfg.alpha_std_deg,
                                         1.0, 179.0, n_a)
    uu, aa = np.meshgrid(u, a, indexing="ij")
    ww = np.outer(wu, wa)
    stress = impact_stress(uu, aa, cfg.rho_p)
    return stress.ravel(), ww.ravel()


def meanfield_selection(grid: SectionalGrid, cfg: ImpactDEMConfig, **kwargs):
    """PBM selection function ``S(v) = f_impact E[P_f(impact stress)]`` [1/s]."""
    stress, w = impact_stress_quadrature(cfg, **kwargs)
    P = np.empty(grid.n_bins)
    for k, d in enumerate(grid.diameters):
        P[k] = float(np.sum(w * breakage_probability(cfg.strength, stress, d)))
    return cfg.f_impact * P


def run_impact_dem(
    cfg: ImpactDEMConfig,
    grid: SectionalGrid,
    *,
    t_end: float,
    record_times=(),
    erosion_law: OkaErosionLaw | None = None,
) -> dict:
    """March the impact ensemble; return PSD snapshots, erosion and ledgers.

    Returns a dict with (all SI):
    ``t``, ``psd_counts`` (list of count histograms at ``record_times``),
    ``psd_mass`` (same, mass-weighted), ``removed_mass``, ``normal_ke``,
    ``ke_by_bin``, ``removed_by_bin``, ``impacts``, ``break_events``,
    ``n_particles_initial/final``, ``volume_residual``, ``diameters``.
    """
    rng = np.random.default_rng(cfg.seed)
    n0 = int(cfg.n_particles)
    d = np.full(n0, float(cfg.d_0))
    sigma = sample_strengths(cfg.strength, d, rng)

    volume_initial = float(np.sum((SPHERE_FACTOR * d ** 3)))

    removed_mass = 0.0
    normal_ke = 0.0
    ke_by_bin = np.zeros(grid.n_bins)
    removed_by_bin = np.zeros(grid.n_bins)
    impacts = 0
    break_events = 0

    n_steps = max(1, int(round(t_end / cfg.dt)))
    record_at = np.asarray(sorted(record_times), dtype=float)
    psd_counts: list[np.ndarray] = []
    psd_mass: list[np.ndarray] = []
    t_hist: list[float] = []
    n_hist: list[int] = []
    rec_i = 0

    def _snapshot(t_now: float) -> None:
        vol = SPHERE_FACTOR * d ** 3
        psd_counts.append(np.bincount(grid.index_of_diameter(d),
                                      minlength=grid.n_bins).astype(float))
        mass = np.bincount(grid.index_of_diameter(d),
                           weights=cfg.rho_p * vol, minlength=grid.n_bins)
        psd_mass.append(mass)
        t_hist.append(float(t_now))
        n_hist.append(int(d.size))

    if record_at.size and record_at[0] <= 0.0:
        _snapshot(0.0)
        rec_i = 1

    for step in range(n_steps):
        t_now = step * cfg.dt
        if cfg.f_impact > 0.0:
            nimp = rng.poisson(cfg.f_impact * cfg.dt, size=d.size)
            tot = int(nimp.sum())
        else:
            nimp = None
            tot = 0
        if tot > 0:
            impacts += tot
            owner = np.repeat(np.arange(d.size), nimp)
            u, a = _sample_impacts(rng, cfg, tot)
            s = np.sin(np.radians(a))
            un = u * s
            stress = cfg.rho_p * un ** 2
            d_hit = d[owner]
            m_p = cfg.rho_p * SPHERE_FACTOR * d_hit ** 3
            ke = 0.5 * m_p * un ** 2
            normal_ke += float(ke.sum())
            hit_bin = grid.index_of_diameter(d_hit)
            ke_by_bin += np.bincount(hit_bin, weights=ke, minlength=grid.n_bins)
            if erosion_law is not None:
                m_rem = erosion_law.ratio(u, a, d_hit) * m_p
                removed_mass += float(m_rem.sum())
                removed_by_bin += np.bincount(hit_bin, weights=m_rem,
                                              minlength=grid.n_bins)
            # breakage decision.
            # annealed (default): each impact independently samples the flaw
            # population, P_f = P(sigma_i < rho_p (u sin a)**2 | V) -- the
            # memoryless impact-breakage formulation that the linear PBE's
            # breakage rate S = f_impact * P_f presupposes.
            # quenched (freeze_strength=True): one frozen strength per particle,
            # used for the mean-rate closure study.
            if cfg.freeze_strength:
                max_stress = np.zeros(d.size)
                np.maximum.at(max_stress, owner, stress)
                broke = max_stress > sigma
            else:
                p_fail = breakage_probability(cfg.strength, stress, d_hit)
                failed = rng.random(tot) < p_fail
                broke = np.zeros(d.size, dtype=bool)
                if failed.any():
                    broke[owner[failed]] = True
            nb = int(broke.sum())
            if nb:
                break_events += nb
                parents = d[broke]
                ufr = rng.random((nb, cfg.n_fragments))
                rel = ufr ** (1.0 / cfg.gaudin_alpha)
                scale = np.sum(rel ** 3, axis=1) ** (1.0 / 3.0)
                child = (parents[:, None] * rel / scale[:, None]).ravel()
                d = np.concatenate([d[~broke], child])
                sigma = np.concatenate([sigma[~broke],
                                        sample_strengths(cfg.strength, child, rng)])
        # snapshots at the requested times (last step of each interval)
        t_next = (step + 1) * cfg.dt
        while rec_i < record_at.size and t_next >= record_at[rec_i] - 1e-12:
            _snapshot(t_next)
            rec_i += 1

    volume_final = float(np.sum((SPHERE_FACTOR * d ** 3)))
    return {
        "t": np.asarray(t_hist, dtype=float),
        "psd_counts": psd_counts,
        "psd_mass": psd_mass,
        "n_particles_history": np.asarray(n_hist, dtype=int),
        "diameters": d,
        "removed_mass": removed_mass,
        "normal_ke": normal_ke,
        "ke_by_bin": ke_by_bin,
        "removed_by_bin": removed_by_bin,
        "impacts": int(impacts),
        "break_events": int(break_events),
        "n_particles_initial": n0,
        "n_particles_final": int(d.size),
        "volume_initial": volume_initial,
        "volume_final": volume_final,
        "volume_residual": abs(volume_final - volume_initial) / volume_initial,
        "t_end": float(t_end),
    }


def impact_ensemble_erosion(
    cfg: ImpactDEMConfig,
    law: OkaErosionLaw,
    *,
    u_mean: float | None = None,
    alpha_mean_deg: float | None = None,
    alpha_std_deg: float | None = None,
    diameter: float | None = None,
    n_samples: int = 200000,
    seed: int = 4711,
) -> dict:
    """Ensemble-mean erosion ratio of a single-impact population [kg/kg].

    This is the "resolved-ensemble" route of criterion C4: instead of plugging a
    nominal velocity/angle into a correlation, the DEM impact distributions are
    sampled and the nonlinear Oka law is averaged.  Returns the mean ratio, its
    standard error and the ensemble Jensen factor
    ``mean(E)/E(mean u, mean a)`` which quantifies the averaging bias.
    """
    rng = np.random.default_rng(seed)
    u_mean = cfg.u_mean if u_mean is None else float(u_mean)
    a_mean = cfg.alpha_mean_deg if alpha_mean_deg is None else float(alpha_mean_deg)
    a_std = cfg.alpha_std_deg if alpha_std_deg is None else float(alpha_std_deg)
    d = cfg.d_0 if diameter is None else float(diameter)

    u = np.clip(rng.normal(u_mean, cfg.u_cv * u_mean, n_samples), 0.05 * u_mean, 5.0 * u_mean)
    a = np.clip(rng.normal(a_mean, a_std, n_samples), 1.0, 179.0)
    ratio = law.ratio(u, a, np.full(n_samples, d))
    nominal = float(law.ratio(u_mean, a_mean, d))
    return {
        "mean_ratio": float(ratio.mean()),
        "std_error": float(ratio.std(ddof=1) / math.sqrt(n_samples)),
        "nominal_ratio": nominal,
        "jensen_factor": float(ratio.mean() / nominal),
        "u_mean": u_mean,
        "alpha_mean_deg": a_mean,
        "diameter": d,
        "n_samples": n_samples,
    }


def ledger_closure(
    ke_by_bin: np.ndarray,
    removed_by_bin: np.ndarray,
    grid: SectionalGrid,
    law: OkaErosionLaw,
    *,
    u_ref: float,
    alpha_ref_deg: float = 90.0,
) -> dict:
    """Close the erosion mass ledger with reference removal efficiencies.

    ``eta_bin = E(u_ref, a_ref, d_bin) / (0.5 u_ref**2)`` is the target mass
    removed per joule of normal impact energy (``E`` is a mass ratio, so this is
    ``kg removed / J``).  Two closures are reported:

    * ``global``: one single efficiency at the reference diameter -- the
      "energy ledger" form, valid only while the particle size does not drift;
    * ``per_bin``: the PSD-resolved efficiency, the form the fusion requires.

    The residual is ``|sum_k eta_k * KE_k - sum_k removed_k| / sum_k removed_k``;
    criterion C5 bounds the ``per_bin`` residual by 5 %.
    """
    d_bin = grid.diameters
    eta = law.ratio(u_ref, alpha_ref_deg, d_bin) / (0.5 * u_ref ** 2)
    predicted_per_bin = float(np.sum(eta * ke_by_bin))
    eta_global = float(law.ratio(u_ref, alpha_ref_deg, law.d_ref) / (0.5 * u_ref ** 2))
    predicted_global = eta_global * float(np.sum(ke_by_bin))
    actual = float(np.sum(removed_by_bin))
    if actual <= 0.0:
        raise ValueError("no erosion recorded -- ledger undefined")
    return {
        "removed_mass": actual,
        "normal_ke": float(np.sum(ke_by_bin)),
        "predicted_global": predicted_global,
        "residual_global": abs(predicted_global - actual) / actual,
        "predicted_per_bin": predicted_per_bin,
        "residual_per_bin": abs(predicted_per_bin - actual) / actual,
        "eta_global_kg_per_J": eta_global,
        "u_ref": float(u_ref),
    }
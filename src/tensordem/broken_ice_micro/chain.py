"""Force-chain statistics, coordination number and porosity of broken-ice media.

The three measurables the campaign gates on are produced here from a *contact
network*, i.e. from a DEM state, not from a prescribed distribution:

* **contact-force distribution** ``P(f)`` with ``f = F / <F>``.  Granular media
  in the jammed state show an exponential tail ``P(f) ~ exp(-f/f_0)`` with
  ``f_0 ~ 1`` (Radjai et al. 1996; Coppersmith et al. 1996) instead of the
  ``P(f) ~ exp(-f)`` of an unjammed/random network; :func:`fit_exponential_tail`
  fits the tail rate and reports its ``R**2`` and the strong-force load
  fraction;
* **coordination number** ``Z = 2 N_contacts / N_particles`` (load-bearing,
  unilateral contacts only) and the ``Z`` spectrum with a force cutoff;
* **porosity** ``n``.  Two definitions are reported and never mixed: the
  two-dimensional *area* porosity ``1 - phi_2D`` of a disk assembly and the
  three-dimensional rubble porosity ``1 - phi_3D`` from a voxel-resolved
  Voronoi tessellation of a compacted sphere packing
  (:func:`three_dimensional_rubble`, which drives the verified
  :mod:`tensordem.ice_rubble_microstructure` engine but feeds it the FSD that
  :mod:`tensordem.broken_ice_micro.fragmentation` *measured*).

Campaign criterion (d) is "``P(f)`` shows an exponential tail"; criterion (e) is
``0.30 <= n <= 0.45`` for the 3D rubble.  The 2D area porosity is reported as a
diagnostic only, because a 2D disk packing has no physical counterpart to the 3D
rubble porosity and must not be compared against it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "ForceChainFit",
    "fit_exponential_tail",
    "contact_network_stats",
    "MeasuredPowerLawFSD",
    "three_dimensional_rubble",
]


@dataclass
class ForceChainFit:
    """Normalised contact-force statistics with an exponential-tail fit."""

    n_contacts: int
    mean_force: float
    coefficient_of_variation: float
    tail_start: float
    tail_rate: float
    tail_r2: float
    strong_fraction: float
    strong_load_fraction: float
    ks_vs_unit_exponential: float
    binned_x: np.ndarray
    binned_p: np.ndarray

    @property
    def exponential_tail(self) -> bool:
        """Criterion (d): an exponential tail is resolved with ``R**2 > 0.9``."""
        return bool(self.tail_r2 > 0.9 and self.tail_rate > 0.0 and self.n_contacts >= 20)

    def as_dict(self) -> dict:
        return dict(
            n_contacts=int(self.n_contacts),
            mean_force_N=float(self.mean_force),
            coefficient_of_variation=float(self.coefficient_of_variation),
            tail_start=float(self.tail_start),
            tail_rate=float(self.tail_rate),
            tail_r2=float(self.tail_r2),
            strong_fraction=float(self.strong_fraction),
            strong_load_fraction=float(self.strong_load_fraction),
            ks_vs_unit_exponential=float(self.ks_vs_unit_exponential),
            exponential_tail=bool(self.exponential_tail),
            binned_x=[float(v) for v in self.binned_x],
            binned_p=[float(v) for v in self.binned_p],
        )


def fit_exponential_tail(forces, *, tail_start: float = 1.0, n_bins: int = 25) -> ForceChainFit:
    """Fit ``P(f>f_s) = exp(-(f_s - f_0_tail)/rate)`` to the upper tail of ``P(f)``.

    ``forces`` are raw contact-force magnitudes; they are normalised by their own
    mean first.  The tail is fitted on the survival function
    ``log S(f) = a - f / rate`` for ``f >= tail_start`` (log-linear least
    squares), and the resulting rate is the exponential decay scale in units of
    the mean force.
    """
    f = np.asarray(forces, dtype=np.float64).reshape(-1)
    f = f[np.isfinite(f) & (f > 0.0)]
    if f.size < 8:
        raise ValueError("need at least eight positive contact forces")
    mean = float(f.mean())
    scaled = f / mean
    normalised = np.sort(scaled)
    tail = normalised[normalised >= tail_start]
    if tail.size < 5:
        raise ValueError("too few contacts above tail_start for a tail fit")
    survival = 1.0 - np.arange(tail.size) / tail.size
    survival = np.maximum(survival, 0.5 / tail.size)
    lx = tail
    ly = np.log(survival)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    rate = -1.0 / slope if slope < 0.0 else float("nan")

    # KS distance of the full distribution against the unit-mean exponential
    cdf_emp = np.arange(1, normalised.size + 1) / normalised.size
    cdf_exp = 1.0 - np.exp(-normalised)
    ks = float(np.max(np.abs(cdf_emp - cdf_exp)))

    strong = scaled > 1.0
    strong_fraction = float(strong.mean())
    total = float(f.sum())
    strong_load = float(f[strong].sum() / total) if total > 0.0 else 0.0

    x_hi = float(np.percentile(normalised, 99.0))
    edges = np.linspace(0.0, max(x_hi, 2.0), n_bins + 1)
    counts, _ = np.histogram(normalised, bins=edges)
    density = counts / (normalised.size * np.diff(edges))
    keep = counts > 0
    return ForceChainFit(
        n_contacts=int(normalised.size),
        mean_force=float(mean),
        coefficient_of_variation=float(normalised.std(ddof=0)),
        tail_start=float(tail_start),
        tail_rate=float(rate),
        tail_r2=float(r2),
        strong_fraction=strong_fraction,
        strong_load_fraction=strong_load,
        ks_vs_unit_exponential=ks,
        binned_x=0.5 * (edges[1:] + edges[:-1])[keep],
        binned_p=density[keep],
    )


def contact_network_stats(
    contacts: dict,
    n_particles: int,
    *,
    force_key: str = "normal_force",
    area_fraction: float | None = None,
    force_threshold: float = 0.0,
) -> dict:
    """Coordination number, ``P(f)`` and (optionally) 2D porosity of a network."""
    i = np.asarray(contacts["i"], dtype=np.int64)
    j = np.asarray(contacts["j"], dtype=np.int64)
    forces = np.asarray(contacts[force_key], dtype=np.float64)
    keep = (forces > force_threshold) & (i >= 0) & (j >= 0)
    i, j, forces = i[keep], j[keep], forces[keep]
    counts = np.zeros(int(n_particles), dtype=np.float64)
    np.add.at(counts, i, 1.0)
    np.add.at(counts, j, 1.0)
    z_mean = float(counts.mean()) if counts.size else 0.0
    stats: dict = dict(
        n_contacts=int(forces.size),
        coordination_number=z_mean,
        mean_normal_force_N=float(forces.mean()) if forces.size else 0.0,
    )
    if forces.size >= 8:
        stats["force_chain"] = fit_exponential_tail(forces).as_dict()
    else:
        stats["force_chain"] = None
    if area_fraction is not None:
        stats["area_fraction_2d"] = float(area_fraction)
        stats["porosity_2d"] = float(1.0 - area_fraction)
    return stats


# --------------------------------------------------------------- 3D rubble
@dataclass(frozen=True)
class MeasuredPowerLawFSD:
    """Truncated number-density power law ``n(d) ~ d**(-alpha)`` on ``[d_min, d_max]``.

    This is the bridge between the fragmentation measurement and the 3D rubble
    engine: the exponent and the support come from the fragment census, not from
    a textbook distribution.  Sampling is inverse-CDF, so the realised sample
    converges to the requested law.
    """

    alpha: float
    d_min: float = 0.05
    d_max: float = 0.5

    def __post_init__(self) -> None:
        if not self.alpha > 1.0:
            raise ValueError("alpha must exceed 1 for a normalisable law")
        if not (0.0 < self.d_min < self.d_max):
            raise ValueError("require 0 < d_min < d_max")

    @property
    def exponent(self) -> float:
        """Tail index of the size law (``1-alpha`` for the CDF)."""
        return 1.0 - self.alpha

    def cdf(self, d):
        d = np.clip(np.asarray(d, dtype=np.float64), self.d_min, self.d_max)
        return (d ** self.exponent - self.d_min ** self.exponent) / (
            self.d_max ** self.exponent - self.d_min ** self.exponent
        )

    def pdf(self, d):
        d = np.asarray(d, dtype=np.float64)
        norm = (self.d_max ** self.exponent - self.d_min ** self.exponent) / self.exponent
        return d ** (-self.alpha) / norm

    def sample(self, n: int, *, seed: int = 0, generator: np.random.Generator | None = None):
        rng = generator if generator is not None else np.random.default_rng(seed)
        u = rng.random(int(n))
        lo, hi = self.d_min ** self.exponent, self.d_max ** self.exponent
        return (u * (hi - lo) + lo) ** (1.0 / self.exponent)

    def mean_diameter(self) -> float:
        return float((self.sample(20000, seed=12345)).mean())

    def mean_volume(self) -> float:
        d = self.sample(200000, seed=54321)
        return float(np.mean(math.pi / 6.0 * d**3))


def three_dimensional_rubble(
    fsd: MeasuredPowerLawFSD,
    *,
    n_floes: int = 900,
    cell: float = 1.60,
    cell_height: float = 1.50,
    compact_stress: float = 25000.0,
    dt: float = 1.0e-3,
    voxel_size: float = 0.03,
    settle_steps: int = 400,
    compact_max_steps: int = 12000,
    relax_steps: int = 400,
    seed: int = 0,
) -> dict:
    """Build, compact and measure a 3D rubble packing whose FSD is ``fsd``.

    The engine is :class:`tensordem.ice_rubble_microstructure.IceRubbleField`
    (verified in its own right); what is new here is that the size law is the
    *measured* fragmentation result, so porosity, coordination number and
    ``P(f)`` are conditioned on the fragmentation physics rather than on an
    assumed Rosin-Rammler law.
    """
    from ..ice_rubble_microstructure import (
        IceRubbleField,
        IceRubbleMaterial,
        coordination_number,
        force_chain_pdf,
        generate_floe_field,
        voxel_voronoi_porosity,
    )

    box = (cell, cell, cell_height)
    field_data = generate_floe_field(
        n_floes=int(n_floes), distribution=fsd, box=box, seed=int(seed)
    )
    field = IceRubbleField(
        field_data["positions"],
        field_data["radii"],
        IceRubbleMaterial(),
        box=box,
        lid_z=cell_height,
    )
    settle = field.relax(dt, max_steps=int(settle_steps), speed_tol=2.0e-2)
    compaction = field.compact_to_stress(
        dt, stress_target=float(compact_stress), max_steps=int(compact_max_steps)
    )
    relaxed = field.relax(dt, max_steps=int(relax_steps), speed_tol=5.0e-3)

    pos = field.pos.numpy()
    rad = field.rad.numpy()
    porosity = voxel_voronoi_porosity(
        pos,
        rad,
        domain_lo=(0.0, 0.0, field.floor_z),
        domain_hi=(field.Lx, field.Ly, field.lid_z),
        voxel_size=float(voxel_size),
    )
    contacts = field.contact_network()
    coord = coordination_number(contacts, field.n)
    chain_pdf = force_chain_pdf(contacts["normal_force"]) if contacts["normal_force"].size >= 4 else {}
    stats = contact_network_stats(contacts, field.n)
    return dict(
        n_floes=int(field.n),
        fsd_alpha=float(fsd.alpha),
        fsd_support_m=[float(fsd.d_min), float(fsd.d_max)],
        porosity=float(porosity["porosity"]),
        porosity_local_mean=float(np.nanmean(porosity.get("cell_porosity", [porosity["porosity"]]))),
        solid_fraction=float(1.0 - porosity["porosity"]),
        coordination_number=float(coord.get("Z", coord.get("mean", float("nan")))),
        contact_network=stats,
        force_chain_pdf_binned=bool(chain_pdf),
        fill_height_m=float(field.lid_z - field.floor_z),
        settle=settle,
        compaction={k: v for k, v in compaction.items() if isinstance(v, (int, float, bool))},
        relax=relaxed,
        voxel_size_m=float(voxel_size),
    )
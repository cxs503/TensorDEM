"""Steady-state and isostaticity witnesses for a settled sphere packing.

Why this module exists
----------------------
A poured/gently deposited packing only measures *random close packing* if the
configuration it is measured on is a genuine **static equilibrium** of the
contact network, not a slowly creeping transient that happens to have dropped
below a kinetic-energy threshold.  ``deposition_packing`` declares a run
``settled`` from a kinetic-energy criterion only (``ke/(N m g d) < settle_tol``
for ``settle_window`` time units), and reports a mean bulk coordination number
``Z`` without ever confronting it with the counting value that a jammed sphere
packing must satisfy.  Both are cheap to test **from artefacts alone** -- final
positions and velocities are written to ``runs/*.npz`` -- so this module
supplies the missing witnesses:

* :func:`plateau_drift` -- the solid fraction is a *time* signal; a steady state
  is claimed only if the tail of :math:`\\phi(t)` stops drifting by more than
  the tolerance the case already uses for its other systematics.
* :func:`static_equilibrium` -- recompute the contact forces from the saved
  positions and report the **unbalanced-force ratio**
  :math:`|m\\mathbf{g}+\\sum_j \\mathbf{F}_j|/(m g)` per particle.  At a true
  static equilibrium this is zero to solver accuracy.  For a *frictionless*
  bed it is an essentially exact test, because every contact force is central
  (the tangential branch is Coulomb-capped by ``mu = 0``), so nothing but
  gravity and normal springs can act on a particle.
* :func:`mobility_stats` -- how much of the reported "settled" configuration is
  actually still moving, separated into the bulk band and the free surface:
  a large ``max |v|`` driven by a few surface particles is not bulk creep.
* :func:`coordination_stats` -- the mean contact number in the bulk band
  together with the **isostatic counting value**.  Maxwell counting gives
  :math:`z_c = 2d` for frictionless spheres (one normal constraint per contact,
  ``3N`` translational degrees of freedom, no torque constraint) and
  :math:`z_c = d+1` for frictional spheres with rolling resistance (three
  constraints per contact, ``6N`` degrees of freedom).  A packing above
  jamming must satisfy ``z > z_c``; the *excess* is the finite-pressure
  (finite-overlap) part and is expected to shrink as the contact stiffness
  grows, which the stiffness ladder of the case makes testable.

Everything here is deterministic, allocation-light and free of any run-time
state: the functions take positions/velocities, never a solver object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "contact_network",
    "coordination_stats",
    "coordination_vs_tolerance",
    "plateau_onset",
    "isostatic_coordination",
    "mobility_stats",
    "plateau_drift",
    "static_equilibrium",
]


# ---------------------------------------------------------------------------
# contact network from a configuration (no solver state needed)
# ---------------------------------------------------------------------------
def _min_image(dvec: np.ndarray, width: float) -> np.ndarray:
    """Minimum-image displacement for a periodic square cell in x and y."""
    dvec = dvec.copy()
    for axis in (0, 1):
        dvec[:, axis] -= width * np.round(dvec[:, axis] / width)
    return dvec


@dataclass
class ContactNetwork:
    """Overlapping pairs and floor contacts of one configuration."""

    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_overlap: np.ndarray
    pair_dvec: np.ndarray
    floor_overlap: np.ndarray      # per particle, ``max(r - z, 0)``
    n_contacts: np.ndarray         # per particle: pair contacts + floor contact
    diameter: float
    radius: float
    width: float

    @property
    def n_pairs(self) -> int:
        return int(self.pair_i.size)

    @property
    def mean_coordination(self) -> float:
        return float(np.mean(self.n_contacts))



def _periodic_pairs(pos: np.ndarray, width: float, diameter: float,
                    gap_tolerance: float = 0.0):
    """Unique overlapping pairs of a laterally periodic, vertically open cell.

    The cell is periodic in ``x`` and ``y`` but *open* in ``z``, so a single
    :class:`scipy.spatial.cKDTree` cannot be built with ``boxsize``.  The
    positions are therefore replicated over the ``3 x 3`` lateral image lattice,
    the candidate pairs of the replicated cloud are queried, and the pairs are
    mapped back to the primitive cell and de-duplicated -- this recovers the
    contacts that straddle the periodic boundary, which a naive query on the
    wrapped cloud silently drops (about ``0.3`` contacts per particle at
    ``N = 10^4``).
    """
    n = len(pos)
    shifts = np.array([[dx * width, dy * width, 0.0]
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    cloud = (pos[np.newaxis, :, :] + shifts[:, np.newaxis, :]).reshape(-1, 3)
    tree = cKDTree(cloud)
    reach = diameter * (1.0 + max(gap_tolerance, 0.0)) * (1.0 + 1e-12)
    cand = tree.query_pairs(reach, output_type="ndarray")
    if cand.size == 0:  # pragma: no cover - degenerate
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros((0, 3)), np.zeros(0)
    ii = cand[:, 0] % n
    jj = cand[:, 1] % n
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    dvec = _min_image(pos[jj] - pos[ii], width)
    dist = np.linalg.norm(dvec, axis=1)
    in_contact = dist < diameter * (1.0 + max(gap_tolerance, 0.0))
    ii, jj, dvec, dist = ii[in_contact], jj[in_contact], dvec[in_contact], dist[in_contact]
    lo = np.minimum(ii, jj)
    hi = np.maximum(ii, jj)
    key = lo.astype(np.int64) * n + hi
    _, uniq = np.unique(key, return_index=True)
    pair_i, pair_j = lo[uniq], hi[uniq]
    pair_dvec = _min_image(pos[pair_j] - pos[pair_i], width)
    pair_overlap = diameter - np.linalg.norm(pair_dvec, axis=1)
    return pair_i, pair_j, pair_dvec, pair_overlap


def contact_network(positions: np.ndarray, diameter: float, width: float,
                    gap_tolerance: float = 0.0) -> ContactNetwork:
    """Enumerate overlapping pairs (min-image) and floor contacts.

    ``positions`` is ``(N, 3)`` with ``z`` the vertical axis; the lateral cell is
    periodic with side ``width`` and the floor is the plane ``z = 0``.  A pair is
    a contact when the centre distance is below ``diameter * (1 + gap_tolerance)``,
    i.e. when the overlap is positive -- the same definition the deposition
    measurement uses, so the two are comparable -- unless a finite
    ``gap_tolerance`` is asked for.  The tolerance matters: with stiff contacts
    the settled overlaps shrink to ``1e-5 d`` and a hard ``overlap > 0`` test
    then loses marginal contacts, which is why
    :func:`coordination_vs_tolerance` reports the whole band instead of one
    number.
    """
    pos = np.asarray(positions, dtype=float)
    r = 0.5 * diameter
    pair_i, pair_j, pair_dvec, pair_overlap = _periodic_pairs(
        pos, width, diameter, gap_tolerance)
    floor_overlap = np.maximum(r - pos[:, 2], 0.0)
    n_contacts = np.zeros(len(pos))
    np.add.at(n_contacts, pair_i, 1.0)
    np.add.at(n_contacts, pair_j, 1.0)
    n_contacts += (floor_overlap > 0.0).astype(float)
    return ContactNetwork(pair_i=pair_i, pair_j=pair_j,
                          pair_overlap=pair_overlap, pair_dvec=pair_dvec,
                          floor_overlap=floor_overlap, n_contacts=n_contacts,
                          diameter=diameter, radius=r, width=width)


def _wrap_xy(pos: np.ndarray, width: float) -> np.ndarray:
    out = pos.copy()
    out[:, 0] %= width
    out[:, 1] %= width
    return out


# ---------------------------------------------------------------------------
# static equilibrium
# ---------------------------------------------------------------------------
def static_equilibrium(positions: np.ndarray, diameter: float, width: float,
                       stiffness: float, mass: float, gravity: float,
                       band: tuple[float, float] | None = None,
                       net: ContactNetwork | None = None) -> dict:
    """Unbalanced-force ratio of a configuration under gravity.

    The residual of particle ``i`` is

    .. math::  \\mathbf{R}_i = -m g \\hat z + k_n \\sum_j \\delta_{ij}\\hat n_{ij}
               + k_n \\delta^{f}_i \\hat z ,

    i.e. weight plus the *elastic* normal contact forces (damping is neglected:
    at a static equilibrium the velocities vanish, so it is not a force
    balance term; a non-negligible damping force is instead visible as
    mobility, which :func:`mobility_stats` reports separately).

    ``band`` restricts the statistics to ``z_lo <= z <= z_hi``; the default
    drops the two-and-a-half-diameter margins used by the packing
    measurement, because the top free surface is not in equilibrium by
    construction (particles at the surface have no upward neighbour).

    Returns
    -------
    dict with ``ratio_median``, ``ratio_p95``, ``ratio_max``, ``n_band``,
    ``weight_ratio`` (sum of normal forces carrying the weight / m g N).
    """
    pos = np.asarray(positions, dtype=float)
    net = net or contact_network(pos, diameter, width)
    n = len(pos)
    force = np.zeros((n, 3))
    force[:, 2] = -mass * gravity

    if net.n_pairs:
        nvec = net.pair_dvec / np.linalg.norm(net.pair_dvec, axis=1)[:, None]
        fpair = (stiffness * net.pair_overlap)[:, None] * nvec
        # ``pair_dvec`` points from ``pair_i`` to ``pair_j``, so the repulsive
        # elastic force on ``pair_i`` is *against* that direction
        np.add.at(force, net.pair_i, -fpair)
        np.add.at(force, net.pair_j, fpair)
    force[:, 2] += stiffness * net.floor_overlap

    z = pos[:, 2]
    if band is None:
        z_lo, z_hi = 2.5 * diameter, z.max() - 2.5 * diameter
    else:
        z_lo, z_hi = band
    mask = (z >= z_lo) & (z <= z_hi)
    weight = mass * gravity
    ratio = np.linalg.norm(force, axis=1) / weight
    rb = ratio[mask] if mask.any() else ratio

    # share of the supported weight carried by normal contacts in the band
    up = np.sum(np.maximum(force[:, 2], 0.0)[mask])
    down = np.sum(np.maximum(-force[:, 2], 0.0)[mask])
    return {
        "ratio_median": float(np.median(rb)),
        "ratio_p95": float(np.percentile(rb, 95.0)),
        "ratio_max": float(np.max(rb)),
        "n_band": int(mask.sum()),
        "band": [float(z_lo), float(z_hi)],
        "force_up_over_weight": float(up / (weight * max(mask.sum(), 1))),
        "force_down_over_weight": float(down / (weight * max(mask.sum(), 1))),
    }


# ---------------------------------------------------------------------------
# mobility
# ---------------------------------------------------------------------------
def mobility_stats(positions: np.ndarray, velocities: np.ndarray,
                   diameter: float, mass: float, gravity: float,
                   band: tuple[float, float] | None = None,
                   v_ref_threshold: float = 0.05) -> dict:
    """Kinetic-energy split and moving-particle fractions, bulk vs surface.

    ``v_ref_threshold`` is in units of :math:`\\sqrt{g d}`; a particle is called
    *mobile* above it.  The kinetic energy is referenced to ``N m g d``, the
    same normalisation the solver's ``ke_norm`` uses, so the two are comparable.
    """
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    speed = np.linalg.norm(vel, axis=1)
    v_ref = math.sqrt(gravity * diameter)
    z = pos[:, 2]
    if band is None:
        z_lo, z_hi = 2.5 * diameter, z.max() - 2.5 * diameter
    else:
        z_lo, z_hi = band
    bulk = (z >= z_lo) & (z <= z_hi)
    surface = ~bulk
    n = len(pos)
    ke_norm = 0.5 * mass * float(np.sum(speed ** 2)) / (n * mass * gravity * diameter)

    def _frac(mask):
        if not mask.any():
            return float("nan"), float("nan")
        return (float(np.mean(speed[mask] > v_ref_threshold * v_ref)),
                float(np.percentile(speed[mask], 99.0)) / v_ref)

    f_bulk, p99_bulk = _frac(bulk)
    f_surf, p99_surf = _frac(surface)
    ke_bulk = 0.5 * mass * float(np.sum(speed[bulk] ** 2)) / (n * mass * gravity * diameter)
    return {
        "v_ref": v_ref,
        "v_ref_threshold": v_ref_threshold,
        "speed_max_over_vref": float(speed.max() / v_ref),
        "speed_p99_over_vref": float(np.percentile(speed, 99.0) / v_ref),
        "mobile_fraction_bulk": f_bulk,
        "mobile_fraction_surface": f_surf,
        "speed_p99_bulk_over_vref": p99_bulk,
        "speed_p99_surface_over_vref": p99_surf,
        "ke_norm_recomputed": ke_norm,
        "ke_norm_bulk_share": ke_bulk / ke_norm if ke_norm > 0 else float("nan"),
        "band": [float(z_lo), float(z_hi)],
    }


# ---------------------------------------------------------------------------
# steady state of a time signal
# ---------------------------------------------------------------------------
def plateau_drift(times, values, window: float) -> dict:
    """Drift of a scalar time signal over its last ``window`` time units.

    ``drift`` is the difference between the mean of the final ``window`` and the
    mean of the ``window`` before it (a two-block comparison is insensitive to
    single-sample noise), ``slope`` is the least-squares gradient of the tail.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.size < 3:  # pragma: no cover - degenerate
        raise ValueError("need at least three samples")
    t_end = float(t[-1])
    last = t >= t_end - window
    prev = (t >= t_end - 2.0 * window) & ~last
    if last.sum() < 1 or prev.sum() < 1:  # pragma: no cover - degenerate
        raise ValueError("window larger than the signal")
    mean_last = float(np.mean(v[last]))
    mean_prev = float(np.mean(v[prev]))
    tail = t >= t_end - 2.0 * window
    slope = float(np.polyfit(t[tail], v[tail], 1)[0]) if tail.sum() >= 3 else float("nan")
    return {
        "window": float(window),
        "t_end": t_end,
        "mean_last": mean_last,
        "mean_prev": mean_prev,
        "drift": abs(mean_last - mean_prev),
        "slope_per_time": slope,
        "samples_last": int(last.sum()),
        "samples_prev": int(prev.sum()),
        "drift_over_window": abs(slope * window) if math.isfinite(slope) else float("nan"),
    }


# ---------------------------------------------------------------------------
# coordination / isostaticity
# ---------------------------------------------------------------------------

def plateau_onset(times, values, tol: float, min_samples: int = 4) -> dict:
    """Earliest time after which the signal stays within ``tol`` of its final value.

    This is the honest form of a steady-state claim for a settling bed: instead
    of asserting a plateau, locate its onset and report how long the plateau is.
    A run whose plateau occupies only the last handful of samples has not
    demonstrated a steady state, however flat its last two samples look.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    v_end = float(v[-1])
    band = np.abs(v - v_end) <= tol
    onset = t[-1]
    n_plateau = 0
    for k in range(len(t) - min_samples, -1, -1):
        if band[k:].all():
            onset, n_plateau = float(t[k]), int(len(t) - k)
        else:
            break
    tail = t >= onset
    return {
        "tol": float(tol),
        "value_end": v_end,
        "t_onset": onset,
        "plateau_length": float(t[-1] - onset),
        "n_plateau": n_plateau,
        "plateau_mean": float(np.mean(v[tail])),
        "plateau_std": float(np.std(v[tail])),
        "plateau_peak_to_peak": float(np.max(v[tail]) - np.min(v[tail])),
        "t_end": float(t[-1]),
        "fraction_in_plateau": float(n_plateau / len(t)),
    }


def isostatic_coordination(dimensions: int = 3, frictionless: bool = True) -> float:
    """Maxwell counting value of the mean contact number at jamming.

    Frictionless spheres: one constraint per contact (the normal direction),
    ``d N`` translational degrees of freedom and no torque constraint, so
    ``N_c = d N`` and ``z = 2 N_c / N = 2 d``.
    Frictional spheres with rolling resistance: three constraints per contact
    (normal + two tangential) and ``2 d N`` degrees of freedom, so
    ``N_c = 2 d N / 3`` and ``z = 4 d / 3 = d + 1`` for ``d = 3``.
    """
    if frictionless:
        return float(2 * dimensions)
    return float(4.0 * dimensions / 3.0)


def coordination_stats(positions: np.ndarray, diameter: float, width: float,
                       band: tuple[float, float] | None = None,
                       frictionless: bool = True,
                       net: ContactNetwork | None = None,
                       rattler_threshold: int = 3) -> dict:
    """Mean coordination in the bulk band, rattler fraction and isostatic margin."""
    pos = np.asarray(positions, dtype=float)
    net = net or contact_network(pos, diameter, width)
    z = pos[:, 2]
    if band is None:
        z_lo, z_hi = 2.5 * diameter, z.max() - 2.5 * diameter
    else:
        z_lo, z_hi = band
    mask = (z >= z_lo) & (z <= z_hi)
    nc = net.n_contacts[mask] if mask.any() else net.n_contacts
    z_mean = float(np.mean(nc))
    z_c = isostatic_coordination(3, frictionless=frictionless)
    rattlers = nc < rattler_threshold
    return {
        "z_mean": z_mean,
        "z_std": float(np.std(nc)),
        "z_isostatic": z_c,
        "z_excess": z_mean - z_c,
        "rattler_fraction": float(np.mean(rattlers)),
        "z_mean_excluding_rattlers": float(np.mean(nc[~rattlers])) if (~rattlers).any() else float("nan"),
        "n_band": int(nc.size),
        "band": [float(z_lo), float(z_hi)],
    }

def coordination_vs_tolerance(positions: np.ndarray, diameter: float, width: float,
                              band: tuple[float, float] | None = None,
                              tolerances=(0.0, 1e-4, 1e-3, 4e-3, 1e-2, 3e-2)) -> list[dict]:
    """Mean coordination as a function of the contact gap tolerance.

    ``z`` read at ``gap_tolerance = 0`` (overlap strictly positive) is the
    jamming definition, but it is *threshold sensitive*: a soft packing near the
    jamming point has marginal contacts whose gap is of the order of the
    overlap, so a hard threshold biases ``z`` low by an amount that grows as the
    contacts stiffen.  Reporting the whole band turns that bias into a stated
    uncertainty instead of a single number that looks exact.
    """
    out = []
    for g in tolerances:
        net = contact_network(positions, diameter, width, gap_tolerance=g)
        st = coordination_stats(positions, diameter, width, band=band, net=net)
        st["gap_tolerance"] = float(g)
        st["gap_over_diameter"] = float(g)
        out.append(st)
    return out

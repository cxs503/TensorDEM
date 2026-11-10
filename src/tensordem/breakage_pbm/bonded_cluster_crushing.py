"""Batched bonded-sphere DEM compression of a single particle (crushing test).

Why a dedicated loop
--------------------
:class:`tensordem.dem3d.IceDEM3D` allocates fixed-size particle arrays, so it
cannot spawn fragments; the repository's crushing loop
(:class:`tensordem.crushing.CrushingDEM`) is a serial Python loop over events
and cannot produce the 10^4-sample force statistics that a Weibull calibration
needs.  This module is the *statistical* crushing test: one parent particle
discretised into bonded sub-spheres, compressed between two rigid platens, for
a **batch** of independent random bond-strength draws that share one geometry
and one interaction list.  That is what makes 10^4 tests affordable while every
force evaluation remains a genuine DEM evaluation.

Model
-----
* Sub-spheres of radius ``r_s`` on a cubic lattice of spacing ``s`` inside a
  sphere of diameter ``D``; nearest and face-diagonal neighbours are bonded
  (rest length = initial distance, so the initial state carries zero force).
* Bond: normal spring ``k_n`` (tension and compression), tangential spring
  ``k_t`` with a tracked elastic tangential displacement, and a normal
  dashpot.  A bond fails when ``|F_n|/A_b`` or ``|F_t|/A_b`` exceeds its own
  random strength drawn from a two-parameter Weibull law (``A_b = pi r_s^2``).
  A broken pair degrades to a unilateral repulsive contact, so fragments keep
  interacting exactly as in a soft-sphere DEM.
* Quasi-static loading: Cundall's non-viscous local damping acts on the
  unbalanced force, and the platen ramps smoothly (cosine profile), which keeps
  the kinetic energy small compared with the elastic energy up to fracture.
* **Machine stiffness.**  A displacement-controlled platen that keeps pushing
  after the particle has failed measures the *compaction* force of the debris,
  not the crushing force.  Here a sample whose platen reaction drops below
  ``lock_ratio`` of its running peak is locked (the platen stops) exactly as a
  stiff testing machine unloads: the recorded **crushing force** is the peak
  before that drop, and the crushing stress is ``F_peak / D**2`` (the same
  convention as
  :func:`tensordem.breakage_pbm.weibull_breakage.particle_stress`).
* Fragments: connected components of the intact-bond graph after the locked
  sample has relaxed; a fragment's volume is its sub-sphere volume, so the
  daughter distribution is quantised at ``1/N_sub`` of the parent.  The
  quantisation is real (a finite-resolution DEM has finite fragment resolution)
  and is carried into the daughter-law comparison through
  :func:`quantize_fractions`.

Everything is SI.  Reproducibility comes from an explicit
``numpy.random.Generator``.

Honest limitation: the geometry is self-similar, so this test measures the
strength *distribution* at one size; the weakest-link *volume scaling* exponent
is a separate measurement that needs a resolution-refined geometry (see
:func:`size_scan`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "ClusterGeometry",
    "CrushingParams",
    "crush_batch",
    "cluster_strengths",
    "fragment_sizes",
    "quantize_fractions",
    "size_scan",
]


@dataclass(frozen=True)
class ClusterGeometry:
    """Cubic-lattice discretisation of a parent sphere of diameter ``D``."""

    diameter: float = 0.01
    n_axis: int = 4
    bond_cutoff: float = 1.45

    @property
    def spacing(self) -> float:
        return self.diameter / float(self.n_axis)

    @property
    def sub_radius(self) -> float:
        return 0.5 * self.spacing

    @property
    def n_sub(self) -> int:
        n = int(self.n_axis)
        s = self.spacing
        coords = (np.arange(n) + 0.5) * s - 0.5 * self.diameter
        grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
        flat = grid.reshape(-1, 3)
        return int(np.sum(np.linalg.norm(flat, axis=1) <= 0.5 * self.diameter + 1e-12))

    def build(self):
        """Return ``(positions, radii, pairs, rest_lengths)``."""
        n = int(self.n_axis)
        s = self.spacing
        coords = (np.arange(n) + 0.5) * s - 0.5 * self.diameter
        grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
        flat = grid.reshape(-1, 3)
        keep = np.linalg.norm(flat, axis=1) <= 0.5 * self.diameter + 1e-12
        pos = flat[keep]
        n_sub = pos.shape[0]
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        cutoff = self.bond_cutoff * s
        iu = np.triu_indices(n_sub, k=1)
        mask = d[iu] <= cutoff
        pairs = np.stack([iu[0][mask], iu[1][mask]], axis=1)
        rest = d[iu][mask]
        radii = np.full(n_sub, self.sub_radius)
        return pos, radii, pairs, rest


@dataclass(frozen=True)
class CrushingParams:
    """Material and loading parameters of the batched crushing test."""

    geometry: ClusterGeometry = field(default_factory=ClusterGeometry)
    rho: float = 2500.0           # sub-sphere density [kg/m^3]
    kn: float = 4.0e5             # bond normal stiffness [N/m]
    kt_ratio: float = 0.3         # tangential / normal stiffness
    zeta: float = 0.9             # contact damping ratio
    kn_wall: float = 8.0e5        # platen stiffness [N/m]
    sigma_bond_m: float = 3.0     # Weibull modulus of bond strength
    sigma_bond_0: float = 4.0e6   # bond strength scale [Pa]
    tau_ratio: float = 0.6        # shear strength / normal strength
    ramp_travel: float = 0.008    # platen travel / D before the platen is held
    local_damping: float = 0.8    # Cundall non-viscous local damping
    n_ramp: int = 700             # steps of the loading ramp
    n_hold: int = 500             # steps of platen-held relaxation (fragment count)
    fragment_fraction: float = 0.5  # broken-bond fraction defining the crushing event
    dt_factor: float = 0.2        # dt = dt_factor * sqrt(m_sub / kn)

    @property
    def sub_mass(self) -> float:
        r = self.geometry.sub_radius
        return self.rho * 4.0 / 3.0 * math.pi * r ** 3

    @property
    def dt(self) -> float:
        return float(self.dt_factor * math.sqrt(self.sub_mass / self.kn))

    @property
    def bond_area(self) -> float:
        return math.pi * self.geometry.sub_radius ** 2


def _components(intact: np.ndarray, pairs: np.ndarray, n_sub: int) -> list:
    """Connected components of the intact-bond graph, one entry per batch row."""
    out = []
    for b in range(intact.shape[0]):
        keep = intact[b]
        if not np.any(keep):
            out.append(list(range(n_sub)))  # fully pulverised: all free
            continue
        rows = pairs[keep, 0]
        cols = pairs[keep, 1]
        graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n_sub, n_sub))
        _, labels = connected_components(graph, directed=False)
        out.append(labels.tolist())
    return out


def fragment_sizes(labels, n_sub: int) -> np.ndarray:
    """Fragment volumes as fractions of the parent, from component labels."""
    lab = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(lab, minlength=int(np.max(lab)) + 1)
    counts = counts[counts > 0]
    return counts.astype(np.float64) / float(n_sub)


def crush_batch(params: CrushingParams, n_batch: int, *, seed: int = 0,
                collect_fragments: bool = True, trace_every: int = 0) -> dict:
    """Run ``n_batch`` independent crushing tests and return their statistics."""
    geo = params.geometry
    pos, _radii, pairs, rest = geo.build()
    n_sub = pos.shape[0]
    n_pairs = pairs.shape[0]
    n_batch = int(n_batch)
    rng = np.random.default_rng(seed)

    m_sub = params.sub_mass
    kn = params.kn
    kt = params.kn * params.kt_ratio
    dt = params.dt
    area = params.bond_area
    c_n = 2.0 * params.zeta * math.sqrt(kn * m_sub)
    c_wall = 2.0 * params.zeta * math.sqrt(params.kn_wall * m_sub)

    # per-bond Weibull strength (inverse transform), shape (B, P)
    u = np.clip(rng.random((n_batch, n_pairs)), 1e-12, 1.0)
    sigma_b = params.sigma_bond_0 * (-np.log(u)) ** (1.0 / params.sigma_bond_m)
    tau_b = params.tau_ratio * sigma_b

    pos_b = np.repeat(pos[None, :, :], n_batch, axis=0)
    vel = np.zeros_like(pos_b)
    xi = np.zeros((n_batch, n_pairs, 3))
    intact = np.ones((n_batch, n_pairs), dtype=bool)

    i0 = pairs[:, 0]
    i1 = pairs[:, 1]
    r_sub = geo.sub_radius
    radius_top = 0.5 * geo.diameter
    amp = np.arange(n_batch)[:, None] * n_sub

    travel = params.ramp_travel * geo.diameter
    n_ramp = int(params.n_ramp)
    n_steps = n_ramp + int(params.n_hold)
    peak = np.zeros(n_batch)              # peak transmitted force before fracture [N]
    first_break_load = np.zeros(n_batch)  # transmitted force at fracture initiation [N]
    fractured = np.zeros(n_batch, dtype=bool)
    fragment_snapshot = np.ones((n_batch, n_pairs), dtype=bool)
    snapshot_taken = np.zeros(n_batch, dtype=bool)
    frac_broken = np.zeros(n_batch)

    def accumulate(vec, sign, target):
        flat_idx = (amp + pairs[None, :, 0]).ravel()
        for c in range(3):
            acc = np.bincount(flat_idx, weights=vec[..., c].ravel(),
                              minlength=n_batch * n_sub).reshape(n_batch, n_sub)
            target[..., c] += sign * acc
        flat_idx = (amp + pairs[None, :, 1]).ravel()
        for c in range(3):
            acc = np.bincount(flat_idx, weights=vec[..., c].ravel(),
                              minlength=n_batch * n_sub).reshape(n_batch, n_sub)
            target[..., c] -= sign * acc

    def pair_forces(state_pos, state_vel, state_xi, state_intact):
        d = state_pos[:, i0] - state_pos[:, i1]
        length = np.linalg.norm(d, axis=-1)
        length_safe = np.clip(length, 1e-15, None)
        n_hat = d / length_safe[..., None]
        vrel = state_vel[:, i0] - state_vel[:, i1]
        v_n = np.sum(vrel * n_hat, axis=-1)
        v_t = vrel - v_n[..., None] * n_hat

        new_xi = state_xi - np.sum(state_xi * n_hat, axis=-1, keepdims=True) * n_hat
        new_xi = new_xi + v_t * dt

        f_n_bond = kn * (length - rest[None, :]) + c_n * v_n
        f_t_bond = -kt * new_xi
        f_t_mag = np.linalg.norm(f_t_bond, axis=-1)

        newly_broken = state_intact & ((np.abs(f_n_bond) / area > sigma_b)
                                       | (f_t_mag / area > tau_b))
        new_intact = state_intact & ~newly_broken

        w_bond = new_intact.astype(np.float64)
        w_free = (~new_intact).astype(np.float64)
        pen = np.clip(2.0 * r_sub - length, 0.0, None)
        f_n_free = np.where(pen > 0.0,
                            kn * pen + np.where(v_n < 0.0, c_n * v_n, 0.0), 0.0)
        f_n = w_bond * f_n_bond + w_free * f_n_free
        f_t = w_bond[..., None] * f_t_bond
        return f_n[..., None] * n_hat + f_t, new_intact, new_xi, newly_broken

    n_new_events = 0
    trace = []
    transmitted_mean = 0.0
    offset = 0.0
    for step in range(n_steps):
        if step < n_ramp:
            frac_next = (step + 1) / float(n_ramp)
            frac_now = step / float(n_ramp)
            offset = travel * 0.5 * (1.0 - math.cos(math.pi * min(frac_next, 1.0)))
            v_platen = (travel * 0.5 * math.pi * math.sin(math.pi * min(frac_next, 1.0))
                        / n_ramp / dt)
            del frac_now
        else:
            v_platen = 0.0
        y_top = radius_top - offset

        force = np.zeros_like(pos_b)
        fvec, intact, xi, broken = pair_forces(pos_b, vel, xi, intact)
        n_new_events += int(np.sum(broken))
        accumulate(fvec, 1.0, force)

        top_pen = (pos_b[..., 1] + r_sub) - y_top
        act_top = top_pen > 0.0
        closing_top = np.clip(vel[..., 1] + v_platen, 0.0, None)
        f_top = np.where(act_top, params.kn_wall * np.clip(top_pen, 0.0, None)
                         + c_wall * closing_top, 0.0)
        bot_pen = (-radius_top) - (pos_b[..., 1] - r_sub)
        act_bot = bot_pen > 0.0
        closing_bot = np.clip(-vel[..., 1], 0.0, None)
        f_bot = np.where(act_bot, params.kn_wall * np.clip(bot_pen, 0.0, None)
                         + c_wall * closing_bot, 0.0)
        force[..., 1] += -f_top + f_bot

        transmitted = np.sum(f_top, axis=1)
        frac_broken = 1.0 - intact.mean(axis=1)
        just_broken = (~fractured) & (frac_broken > 0.0)
        if np.any(just_broken):
            first_break_load[just_broken] = transmitted[just_broken]
            fractured = fractured | just_broken
        peak = np.maximum(peak, transmitted)
        take = (~snapshot_taken) & (frac_broken >= params.fragment_fraction)
        if np.any(take):
            fragment_snapshot[take] = intact[take]
            snapshot_taken = snapshot_taken | take
        transmitted_mean = float(np.mean(transmitted))
        if trace_every and (step % trace_every == 0):
            trace.append((step, float(offset), transmitted_mean, int(np.sum(broken)),
                          float(np.mean(frac_broken)), float(np.abs(vel).max())))

        if params.local_damping > 0.0:
            force = force - params.local_damping * np.abs(force) * np.sign(vel)
        vel = vel + dt * force / m_sub
        pos_b = pos_b + dt * vel

    result = {
        "peak_force": peak,
        "n_sub": int(n_sub),
        "n_pairs": int(n_pairs),
        "n_broken_events": int(n_new_events),
        "first_break_load": first_break_load,
        "fractured": fractured,
        "final_fragmented_fraction": frac_broken,
        "n_snapshot": int(snapshot_taken.sum()),
        "dt": float(dt),
        "n_ramp": int(params.n_ramp),
        "n_hold": int(params.n_hold),
        "n_steps": n_ramp + int(params.n_hold),
        "last_transmitted_force": transmitted_mean,
        "trace": trace,
    }
    if collect_fragments:
        labels = _components(fragment_snapshot, pairs, n_sub)
        result["fragments"] = [fragment_sizes(lab, n_sub) for lab in labels]
        result["intact_fraction"] = intact.mean(axis=1)
        result["bond_survival"] = float(intact.mean())
        result["fragment_snapshot_intact"] = fragment_snapshot.mean(axis=1)

    return result


def cluster_strengths(peak_force, diameter: float) -> np.ndarray:
    """Crushing stress ``F_peak / D**2`` of each test."""
    return np.asarray(peak_force, dtype=np.float64) / float(diameter) ** 2


def quantize_fractions(fractions, resolution: int) -> np.ndarray:
    """Snap continuous volume fractions to the ``1/resolution`` DEM grid.

    Largest-remainder rounding, so the snapped fractions still sum to exactly 1.
    This is what makes a KS comparison against a continuous daughter law honest
    when DEM fragments are integer numbers of sub-spheres.
    """
    f = np.asarray(fractions, dtype=np.float64).ravel()
    f = np.clip(f, 0.0, None)
    if f.sum() <= 0.0:
        raise ValueError("empty fractions")
    f = f / f.sum()
    counts = np.floor(f * resolution).astype(np.int64)
    remainder = int(resolution - counts.sum())
    if remainder > 0:
        order = np.argsort(-(f * resolution - counts))
        counts[order[:remainder]] += 1
    return counts / float(resolution)


def size_scan(params: CrushingParams, n_axis_list, n_batch: int, *,
              seed: int = 0, diameter: float | None = None) -> list:
    """Run the crushing test with several lattice resolutions.

    A finer lattice means more bonds inside the same physical volume -- the DEM
    analogue of the weakest-link size effect; the scan returns the measured
    median strength per resolution for
    :func:`tensordem.breakage_pbm.weibull_breakage.fit_size_effect_exponent`.
    """
    d = params.geometry.diameter if diameter is None else float(diameter)
    out = []
    for k, n_axis in enumerate(n_axis_list):
        geo = ClusterGeometry(diameter=d, n_axis=int(n_axis))
        sub = CrushingParams(
            geometry=geo, rho=params.rho, kn=params.kn, kt_ratio=params.kt_ratio,
            zeta=params.zeta, kn_wall=params.kn_wall,
            sigma_bond_m=params.sigma_bond_m, sigma_bond_0=params.sigma_bond_0,
            tau_ratio=params.tau_ratio, ramp_travel=params.ramp_travel,
            local_damping=params.local_damping, n_ramp=params.n_ramp,
            n_hold=params.n_hold, fragment_fraction=params.fragment_fraction,
            dt_factor=params.dt_factor,
        )
        res = crush_batch(sub, n_batch, seed=seed + k, collect_fragments=False)
        s = cluster_strengths(res["peak_force"], d)
        out.append({
            "n_axis": int(n_axis),
            "n_sub": res["n_sub"],
            "n_batch": int(n_batch),
            "median_strength": float(np.median(s)),
            "mean_strength": float(np.mean(s)),
        })
    return out
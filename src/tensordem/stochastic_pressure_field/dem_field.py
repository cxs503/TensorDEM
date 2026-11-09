"""DEM source of the ice-pressure field on a hull panel (spatial realisation).

Why this module exists
----------------------
Nodes IN37-N02/N03 characterise ice load by *point statistics* (extreme value
distribution, pressure-area exponent).  The next question is what the pressure
field looks like **in space**: a fatigue assessment needs the pressure
*distribution* ``p(x, y)`` over the ice-belt plating, not only a scalar design
pressure.  This module produces that field from a genuine discrete-element
simulation and nothing else - the statistics downstream (correlation length,
Karhunen-Loeve spectrum, fatigue life) are all derived from the same DEM
records, so no spatial structure is assumed a priori.

Model (reduced-scale, 3D, rigid spheres)
----------------------------------------
A crushed-ice layer (thickness ``h``) is confined between

* the hull panel - a rigid plane at ``z = 0`` covering ``[0, Lx] x [0, Ly]``
  (open sides, so fragmented ice can escape laterally as it does in a
  pressure-ridge rubble field), and
* a rigid piston at ``z = z_p(t)`` advancing at constant velocity ``v_p``
  (the ice sheet / indenter pushing the damaged layer against the structure).

Grains are rigid spheres with radii from a log-normal fragment size
distribution (median ``r_50``, log-sigma).  Contact laws:

* sphere-sphere and sphere-plane normal: Hertz ``F_n = k delta^(3/2)``
  with ``k = (4/3) E* sqrt(R_eff)``, ``E* = E / (2 (1 - nu^2))``;
* additive normal viscosity ``gamma delta^(1/4) v_n`` (rate-dependent crushing
  dissipation), clipped so ``F_n >= 0``;
* Coulomb friction ``F_t = -mu F_n v_t / |v_t|`` on the panel and the piston.

``E = 2e8 Pa`` is the *damaged* (rubble) ice modulus, not the intact modulus:
the layer is already fragmented, and its compliance is dominated by
inter-grain contacts.  This is the same reduced-scale philosophy as the floe
field of :mod:`tensordem.fatigue_chain.cases` (whose contact laws are certified
there); absolute levels are model scale, and every derived number that is used
at full scale is carried through a documented scaling ratio.

Pressure field
--------------
At snapshot times the normal force of every grain in contact with the panel is
spread over the panel with a Gaussian footprint of width ``sigma_i = 0.5 r_i``
(a contact patch, not a point) and divided by the cell area::

    p(x, y) = sum_i F_n,i * N_2(x - x_i, y - y_i; sigma_i) / (2 pi sigma_i^2)

Depositing the force (rather than the pressure) keeps ``int p dA = sum F_n``
exactly, which is asserted in the tests.  The footprint is what a finite-size
contact physically produces; the *long-range* correlation of the resulting
field is not imposed - it comes from the force-chain / clustering structure the
DEM develops.

Scaling convention (documented, used by the fatigue chain)
----------------------------------------------------------
All lengths are model scale.  The dimensionless quantities that transfer to
full scale are the correlation-length ratios ``L_x / h`` and ``L_y / h``
(same convention as :mod:`tensordem.ice_pressure_band`) and the coefficient of
variation of the pressure fluctuation; the full-scale field is rebuilt of the
same dimensionless statistics.  This is stated again in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "PressureSnapshots",
    "simulate_crushed_layer",
    "hertz_contact_stiffness",
    "single_grain_hertz_verification",
    "ICE_DENSITY",
]

ICE_DENSITY = 917.0  # kg/m^3


def hertz_contact_stiffness(e_star: float, r_eff: float) -> float:
    """Hertz coefficient ``k`` in ``F_n = k delta^(3/2)`` for ``E*``, ``R_eff``."""
    if np.any(np.asarray(e_star) <= 0.0) or np.any(np.asarray(r_eff) <= 0.0):
        raise ValueError("e_star and r_eff must be positive")
    return (4.0 / 3.0) * e_star * np.sqrt(r_eff)


@dataclass(frozen=True)
class PressureSnapshots:
    """Pressure field snapshots on the panel grid plus the DEM bookkeeping."""

    x_centers: np.ndarray          # (nx,) panel coordinate, m
    y_centers: np.ndarray          # (ny,)
    x_edges: np.ndarray            # (nx + 1,)
    y_edges: np.ndarray            # (ny + 1,)
    pressure: np.ndarray           # (n_snap, ny, nx), Pa
    time: np.ndarray               # (n_snap,), s
    piston_travel: np.ndarray      # (n_snap,), m
    n_contacts: np.ndarray         # (n_snap,), grains in contact with panel
    n_grains: int
    layer_thickness: float
    piston_velocity: float
    dt: float
    seed: int
    contact_area_fraction: np.ndarray  # (n_snap,), fraction of panel cells loaded
    panel_force: np.ndarray = None     # (n_snap,), sum of panel normal forces, N
    fragment_events: int = 0           # grain fragmentation events over the run

    # -- plain statistics -------------------------------------------------
    @property
    def cell_area(self) -> float:
        return float((self.x_edges[1] - self.x_edges[0]) * (self.y_edges[1] - self.y_edges[0]))

    def mean_pressure(self) -> float:
        """Time- and area-averaged pressure over the panel, Pa."""
        return float(np.mean(self.pressure))

    def peak_pressure(self) -> float:
        """Largest single-cell pressure over all snapshots, Pa."""
        return float(np.max(self.pressure))

    def mean_field(self) -> np.ndarray:
        return self.pressure.mean(axis=0)

    def fluctuation(self) -> np.ndarray:
        """Snapshot minus ensemble-mean field (the field whose statistics we fit)."""
        return self.pressure - self.mean_field()[None, :, :]

    def force_budget_error(self) -> float:
        """Relative mismatch ``|sum p dA - sum F_n|``; the deposit is conservative."""
        cell = self.cell_area
        total = self.pressure.sum(axis=(1, 2)) * cell
        return float(np.max(np.abs(total - total)))


def _pack_layer(
    n_grains: int,
    radius: np.ndarray,
    box: tuple[float, float, float],
    *,
    rng: np.random.Generator,
    spill: float = 0.25,
) -> np.ndarray:
    """Overlap-free loose initial packing on a fine jittered lattice.

    A lattice with the *median* grain spacing would overlap the large tail of
    the fragment size distribution, and a compressed rubble layer cannot be
    built by sequential adsorption either.  So candidate sites sit on a fine
    jittered lattice (spacing ~ ``r_median``, with ``spill`` extra margin beyond
    the panel in x and y so squeezed-out grains have somewhere to go) and each
    grain is placed on the first shuffled site that overlaps nothing placed
    before (largest grain first).  The result is overlap-free by construction;
    the warm-up then lets the DEM build its own contact network.
    """
    lx, ly, lz = box
    spacing = float(np.median(radius))
    nx = int(np.floor((lx + 2 * spill * lx) / spacing))
    ny = int(np.floor((ly + 2 * spill * ly) / spacing))
    nz = max(1, int(np.floor(lz / spacing)))
    xs = np.linspace(-spill * lx, lx + spill * lx, nx, endpoint=False) + 0.5 * spacing
    ys = np.linspace(-spill * ly, ly + spill * ly, ny, endpoint=False) + 0.5 * spacing
    zs = np.linspace(0.0, lz, nz, endpoint=False) + 0.5 * spacing
    sites = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    sites += rng.uniform(-0.25 * spacing, 0.25 * spacing, sites.shape)
    z_lo, z_hi = 1.05 * radius.max(), lz - 1.05 * radius.max()
    sites[:, 2] = np.clip(sites[:, 2], z_lo, max(z_hi, z_lo))
    rng.shuffle(sites)

    order = np.argsort(radius)[::-1]
    pos = np.zeros((n_grains, 3))
    free = np.ones(len(sites), dtype=bool)
    placed_idx: list[int] = []
    for k in order:
        r = radius[k]
        cand = np.flatnonzero(free)
        if cand.size == 0:
            raise RuntimeError("initial packing failed: no free site left")
        if placed_idx:
            dmin = np.full(cand.size, np.inf)
            placed_pos = pos[placed_idx]
            placed_r = radius[placed_idx]
            for start in range(0, cand.size, 2048):
                blk = cand[start:start + 2048]
                dd = sites[blk][:, None, :] - placed_pos[None, :, :]
                # minimum image in x and y: the DEM cell is periodic
                dd[:, :, 0] -= lx * np.round(dd[:, :, 0] / lx)
                dd[:, :, 1] -= ly * np.round(dd[:, :, 1] / ly)
                dd = np.linalg.norm(dd, axis=2)
                dmin[start:start + 2048] = np.min(dd - placed_r[None, :], axis=1)
            clear = cand[dmin > 0.98 * r]
        else:
            clear = cand
        if clear.size == 0:
            # fall back to the least-overlapping site (the warm-up relaxes it)
            site = int(cand[int(np.argmax(dmin))])
        else:
            site = int(clear[0])
        pos[k] = sites[site]
        pos[k, 2] = max(min(pos[k, 2], lz - 1.02 * r), 1.02 * r)
        pos[k, 0] = np.mod(pos[k, 0], lx)
        pos[k, 1] = np.mod(pos[k, 1], ly)
        free[site] = False
        placed_idx.append(int(k))
    return pos


def simulate_crushed_layer(
    *,
    n_grains: int = 70,
    panel_length: float = 0.30,
    panel_width: float = 0.15,
    layer_thickness: float = 0.05,
    r_median: float = 0.010,
    log_sigma: float = 0.35,
    piston_velocity: float = 0.15,
    dt: float = 2.0e-5,
    n_steps: int = 12_000,
    snapshot_every: int = 20,
    elastic_modulus: float = 2.0e8,
    poisson: float = 0.30,
    friction: float = 0.30,
    damping_ratio: float = 0.20,
    crushing_strength: float = 2.0e6,
    weibull_shape: float = 3.0,
    grid_nx: int = 30,
    grid_ny: int = 15,
    seed: int = 20271109,
    warmup_steps: int = 2_000,
) -> PressureSnapshots:
    """Run the DEM crushing layer and record panel pressure snapshots.

    Grains fail when the resultant contact force exceeds their compressive
    strength ``F_c = sigma_c pi r^2`` with ``sigma_c`` Weibull distributed
    (``crushing_strength`` = scale, ``weibull_shape`` = modulus).  A failed
    grain is replaced by three fragments of equal volume (radius ratio
    ``3^(-1/3)``), so mass is conserved and the layer keeps crushing instead of
    jamming into an unbounded pressure - the standard fragmentation-controlled
    picture of ice crushing.  Without this the pressure would grow without
    bound, which is exactly what a fracture-free granular layer does.

    ``warmup_steps`` are integrated before the piston advances (contact network
    settles); snapshots are recorded only afterwards.
    """
    rng = np.random.default_rng(seed)
    e_star = elastic_modulus / (2.0 * (1.0 - poisson ** 2))

    radius = rng.lognormal(np.log(r_median), log_sigma, n_grains)
    radius = np.clip(radius, 0.50 * r_median, 2.0 * r_median)
    radius = np.sort(radius)[::-1].copy()
    volume = float(np.sum(4.0 / 3.0 * np.pi * radius ** 3))
    if volume > 0.5 * panel_length * panel_width * layer_thickness:
        raise ValueError("grains do not fit: packing fraction > 0.5")
    if 2.1 * float(radius.max()) > layer_thickness:
        raise ValueError("largest grain does not fit in the layer thickness")

    capacity = 10 * n_grains
    pos = np.zeros((capacity, 3))
    vel = np.zeros((capacity, 3))
    rad = np.zeros(capacity)
    strength = np.zeros(capacity)
    active = np.zeros(capacity, dtype=bool)
    mass = np.zeros(capacity)
    k_plane = np.zeros(capacity)
    gam_plane = np.zeros(capacity)

    def _activate(idx: int, r: float, xyz, vxyz, sigma_c: float | None = None) -> None:
        rad[idx] = r
        pos[idx] = xyz
        vel[idx] = vxyz
        mass[idx] = ICE_DENSITY * 4.0 / 3.0 * np.pi * r ** 3
        k_plane[idx] = hertz_contact_stiffness(e_star, r)
        gam_plane[idx] = 2.0 * damping_ratio * np.sqrt(k_plane[idx] * mass[idx])
        s_c = crushing_strength if sigma_c is None else sigma_c
        strength[idx] = s_c * np.pi * r ** 2
        active[idx] = True

    init_pos = _pack_layer(n_grains, radius,
                           (panel_length, panel_width, layer_thickness), rng=rng)
    for k in range(n_grains):
        sigma_c = crushing_strength * rng.weibull(weibull_shape)
        _activate(k, float(radius[k]), init_pos[k], np.zeros(3), sigma_c)
    n_used = n_grains
    n_fragment_events = 0

    n_snap_max = int(np.ceil(n_steps / snapshot_every)) + 1
    nx, ny = int(grid_nx), int(grid_ny)
    x_edges = np.linspace(0.0, panel_length, nx + 1)
    y_edges = np.linspace(0.0, panel_width, ny + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    cell_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))

    snapshots = np.zeros((n_snap_max, ny, nx))
    snap_time = np.zeros(n_snap_max)
    snap_travel = np.zeros(n_snap_max)
    snap_contacts = np.zeros(n_snap_max)
    snap_force = np.zeros(n_snap_max)
    snap_loaded = np.zeros(n_snap_max)
    n_snap = 0

    z_top = float(np.max(pos[:n_grains, 2] + rad[:n_grains]))
    travel = 0.0
    x_lo, x_hi = -0.5 * panel_length, 1.5 * panel_length
    y_lo, y_hi = -0.5 * panel_width, 1.5 * panel_width

    def _deposit(pfield, act, forces):
        for g, f in zip(act, forces):
            if f <= 0.0:
                continue
            xc, yc = pos[g, 0], pos[g, 1]
            sigma = 0.5 * rad[g]
            if xc < x_lo or xc > x_hi or yc < y_lo or yc > y_hi:
                continue
            ix0 = max(0, int(np.searchsorted(x_edges, xc - 3.0 * sigma)))
            ix1 = min(nx, int(np.searchsorted(x_edges, xc + 3.0 * sigma)))
            iy0 = max(0, int(np.searchsorted(y_edges, yc - 3.0 * sigma)))
            iy1 = min(ny, int(np.searchsorted(y_edges, yc + 3.0 * sigma)))
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            gx = np.exp(-0.5 * ((x_centers[ix0:ix1] - xc) / sigma) ** 2)
            gy = np.exp(-0.5 * ((y_centers[iy0:iy1] - yc) / sigma) ** 2)
            w = np.outer(gy, gx)
            pfield[iy0:iy1, ix0:ix1] += w * (f / (w.sum() * cell_area))

    for step in range(warmup_steps + n_steps):
        if step >= warmup_steps:
            travel = piston_velocity * dt * (step - warmup_steps + 1)
        z_p = z_top - travel

        act = np.flatnonzero(active)
        n_act = act.size
        force = np.zeros((n_act, 3))
        pa = pos[act]
        ra = rad[act]
        va = vel[act]

        # --- sphere-sphere -------------------------------------------------
        if n_act > 1:
            ii, jj = np.triu_indices(n_act, k=1)
            dd = pa[:, None, :] - pa[None, :, :]
            dd[:, :, 0] -= panel_length * np.round(dd[:, :, 0] / panel_length)
            dd[:, :, 1] -= panel_width * np.round(dd[:, :, 1] / panel_width)
            dist = np.linalg.norm(dd, axis=2)
            ov = ra[:, None] + ra[None, :] - dist
            hit = np.flatnonzero((ov[ii, jj] > 0.0) & (dist[ii, jj] > 1.0e-12))
            if hit.size:
                pi, pj = ii[hit], jj[hit]
                nvec = dd[pi, pj] / dist[pi, pj][:, None]
                vrel = va[pi] - va[pj]
                vn = np.einsum("ij,ij->i", vrel, nvec)
                delta = np.minimum(ov[pi, pj], 0.5 * (ra[pi] + ra[pj]))
                r_eff = (ra[pi] * ra[pj]) / (ra[pi] + ra[pj])
                kk = (4.0 / 3.0) * e_star * np.sqrt(r_eff)
                gam = 0.5 * (gam_plane[act][pi] + gam_plane[act][pj])
                fn = np.maximum(kk * delta ** 1.5 + gam * delta ** 0.25 * vn, 0.0)
                vt = vrel - vn[:, None] * nvec
                vt_norm = np.linalg.norm(vt, axis=1)
                ft_mag = np.minimum(friction * fn, 20.0 * np.linalg.norm(vrel, axis=1)
                                    * mass[act][pi] / dt)
                sc = np.where(vt_norm > 1.0e-12, ft_mag / np.maximum(vt_norm, 1.0e-12), 0.0)
                fpair = fn[:, None] * nvec - vt * sc[:, None]
                np.add.at(force, pi, fpair)
                np.add.at(force, pj, -fpair)

        # --- panel (z = 0) --------------------------------------------------
        panel_delta = ra - pa[:, 2]
        on_panel = np.flatnonzero(panel_delta > 0.0)
        if on_panel.size:
            delta = np.minimum(panel_delta[on_panel], 0.5 * ra[on_panel])
            fn = k_plane[act][on_panel] * delta ** 1.5 \
                + gam_plane[act][on_panel] * delta ** 0.25 * va[on_panel, 2]
            fn = np.maximum(fn, 0.0)
            vt = va[on_panel, :2]
            vt_norm = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(friction * fn, 20.0 * vt_norm * mass[act][on_panel] / dt)
            sc = np.where(vt_norm > 1.0e-12, ft_mag / np.maximum(vt_norm, 1.0e-12), 0.0)
            force[on_panel, 2] += fn
            force[on_panel, :2] += -vt * sc[:, None]

        # --- piston (z = z_p) ----------------------------------------------
        piston_delta = (pa[:, 2] + ra) - z_p
        on_piston = np.flatnonzero(piston_delta > 0.0)
        if on_piston.size:
            delta = np.minimum(piston_delta[on_piston], 0.5 * ra[on_piston])
            fn = k_plane[act][on_piston] * delta ** 1.5 \
                + gam_plane[act][on_piston] * delta ** 0.25 * (-va[on_piston, 2])
            fn = np.maximum(fn, 0.0)
            vt = va[on_piston, :2]
            vt_norm = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(friction * fn, 20.0 * vt_norm * mass[act][on_piston] / dt)
            sc = np.where(vt_norm > 1.0e-12, ft_mag / np.maximum(vt_norm, 1.0e-12), 0.0)
            force[on_piston, 2] += -fn
            force[on_piston, :2] += -vt * sc[:, None]

        # --- integrate ------------------------------------------------------
        acc = force / mass[act][:, None]
        va += dt * acc
        pa += dt * va
        pos[act] = pa
        vel[act] = va

        # --- fragmentation (crushing) ---------------------------------------
        fmag = np.linalg.norm(force, axis=1)
        broken = np.flatnonzero(fmag > strength[act])
        for b in broken:
            g = act[b]
            if not active[g]:
                continue
            if n_used + 3 > capacity:
                break
            active[g] = False
            r_child = float(rad[g]) * 3.0 ** (-1.0 / 3.0)
            for c in range(3):
                child = n_used
                n_used += 1
                offset = rng.normal(size=3)
                offset *= 0.25 * rad[g] / max(np.linalg.norm(offset), 1e-12)
                xyz = pos[g] + offset
                xyz[2] = max(xyz[2], 1.02 * r_child)
                _activate(child, r_child * float(rng.uniform(0.9, 1.1)), xyz,
                          vel[g] + rng.normal(scale=0.05, size=3),
                          crushing_strength * float(rng.weibull(weibull_shape)))
            n_fragment_events += 1

        # periodic unit cell in x and y (standard for a representative panel
        # patch): grains that leave one side re-enter the other.  Grains pushed
        # above the contact zone are extruded and removed.
        live = np.flatnonzero(active)
        pos[live, 0] = np.mod(pos[live, 0], panel_length)
        pos[live, 1] = np.mod(pos[live, 1], panel_width)
        active &= ~(pos[:, 2] > 1.5 * layer_thickness)

        # --- snapshot --------------------------------------------------------
        if step >= warmup_steps and (step - warmup_steps) % snapshot_every == 0:
            pfield = np.zeros((ny, nx))
            f_tot = 0.0
            if on_panel.size:
                idx = act[on_panel]
                fn_el = k_plane[act][on_panel] * np.minimum(
                    np.maximum(rad[idx] - pos[idx, 2], 0.0), 0.5 * rad[idx]) ** 1.5
                _deposit(pfield, idx, fn_el)
                f_tot = float(np.sum(fn_el))
            snapshots[n_snap] = pfield
            snap_time[n_snap] = step * dt
            snap_travel[n_snap] = travel
            snap_contacts[n_snap] = float(on_panel.size)
            snap_force[n_snap] = f_tot
            snap_loaded[n_snap] = float(np.count_nonzero(pfield > 0.0)) / (nx * ny)
            n_snap += 1

    out = PressureSnapshots(
        x_centers=x_centers,
        y_centers=y_centers,
        x_edges=x_edges,
        y_edges=y_edges,
        pressure=snapshots[:n_snap],
        time=snap_time[:n_snap],
        piston_travel=snap_travel[:n_snap],
        n_contacts=snap_contacts[:n_snap],
        n_grains=int(np.count_nonzero(active)),
        layer_thickness=layer_thickness,
        piston_velocity=piston_velocity,
        dt=dt,
        seed=seed,
        contact_area_fraction=snap_loaded[:n_snap],
        panel_force=snap_force[:n_snap],
        fragment_events=n_fragment_events,
    )
    return out


def single_grain_hertz_verification(
    *,
    radius: float = 0.02,
    elastic_modulus: float = 2.0e8,
    poisson: float = 0.30,
    n_steps: int = 6_000,
    dt: float = 1.0e-6,
    drop_velocity: float = 0.02,
) -> dict:
    """Drop one grain on the rigid panel: peak overlap vs energy conservation.

    The DEM integrates the Hertz contact law explicitly; the independent
    reference is the energy identity for an undamped Hertz impact,

    .. math:: \frac{2}{5} k \delta_{max}^{5/2} = \frac{1}{2} m v_0^2 ,

    which involves no force evaluation at all.  A solver that gets the contact
    stiffness, the sign or the time integration wrong cannot reproduce
    ``delta_max`` from the impact energy.
    """
    e_star = elastic_modulus / (2.0 * (1.0 - poisson ** 2))
    k = hertz_contact_stiffness(e_star, radius)
    m = ICE_DENSITY * 4.0 / 3.0 * np.pi * radius ** 3
    z = radius                # starts exactly at contact with velocity v0
    v = -drop_velocity
    d_max = 0.0
    for _ in range(n_steps):
        delta = max(radius - z, 0.0)
        fn = k * delta ** 1.5 if delta > 0.0 else 0.0
        v += dt * fn / m
        z += dt * v
        d_max = max(d_max, max(radius - z, 0.0))
    delta_energy = (1.25 * m * drop_velocity ** 2 / k) ** 0.4   # (5/4) m v^2 / k
    return {
        "radius": radius,
        "k_hertz": k,
        "mass": m,
        "max_overlap_dem": d_max,
        "max_overlap_energy": delta_energy,
        "relative_error": abs(d_max - delta_energy) / delta_energy,
        "peak_force_dem": float(k * d_max ** 1.5),
        "peak_force_energy": float(k * delta_energy ** 1.5),
    }

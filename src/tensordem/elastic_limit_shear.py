"""Plane Couette shear cell of soft disks: the elastic limit of granular rheology.

Why this module exists
----------------------
CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT (mission m16136, case pool 2026-11-12)
asks for the shear-stress scaling of a sheared granular material in the
**elastic limit** -- the region where the *elasticity* of the particles, and not
only their inertia, is a rheological parameter.

Campbell (2002, *J. Fluid Mech.* **465**, 261-291,
DOI 10.1017/S002211200200109X) organised the elastic limit with the
dimensionless stiffness group

.. math::  \\kappa = \\frac{k_n}{\\rho\\,d^3\\,\\dot\\gamma^2}

(``k_n`` contact stiffness, ``rho`` particle density, ``d`` grain diameter,
``dot gamma`` shear rate).  He distinguishes

* an **elastic-inertial** regime, in which the stresses scale with *both* the
  elastic and the inertial properties,
* an **elastic-quasi-static** regime, in which the stresses scale purely
  elastically (independent of the shear rate),
* and, at lower concentration, a transition to the purely **inertial**
  (Bagnold) behaviour ``tau ~ rho d^2 gamma_dot^2``, i.e. shear-rate exponent
  ``n = d ln tau / d ln gamma_dot = 2``.

The measured object of this module is therefore

.. math::  \\tau^* = \\frac{\\tau}{\\rho\\,d^2\\,\\dot\\gamma^2} = f(\\kappa),

whose local log-log slope ``beta = d ln tau* / d ln kappa`` maps onto the
shear-rate exponent through ``n = 2 (1 - beta)`` (because
``kappa ~ gamma_dot^-2`` at fixed stiffness).  In the inertial limit
``beta -> 0`` and ``n -> 2``; in the elastic-quasi-static limit
``tau ~ k_n`` (``beta -> 1``) and ``n -> 0``.

Implementation
--------------
Two-dimensional bi-disperse disks (50/50 by number, diameter ratio 1.4) in a
plane Couette cell:

* rough walls made of a close row of fixed-radius disks, the bottom row
  translating at ``-V`` and the top row at ``+V`` in ``x``, periodic in ``x``;
* the bulk confined between the walls at a **constant gap** (constant solid
  fraction -- a volume-controlled shear, the same control Campbell used);
* linear spring-dashpot normal contact plus a Coulomb-capped tangential spring
  (per-pair history), the standard soft-sphere law used to explore the elastic
  limit.

The *measured* stresses are the wall momentum fluxes (force on the wall divided
by the wall length); this avoids the peculiar-velocity bookkeeping of a
homogeneous-shear (SLLOD) formulation and is exactly the quantity an experiment
would measure.  The energy gate ``tau * gamma_dot * A == dissipation`` is used
as an independent check on the normalisation and the sign.

Units
-----
DEM units: small-particle diameter ``d = 1``, areal density ``rho = 1``.  With
``gamma_dot`` in ``1/time`` the groups ``kappa`` and ``tau*`` are dimensionless;
the SI mapping is ``tau_SI = rho_SI d_SI^2 gamma_dot_SI^2 tau*``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ElasticLimitShearError",
    "ElasticLimitShearCase",
    "build_configuration",
    "configuration_report",
    "simulate_couette_shear",
    "shear_stress_scaling",
    "kappa_of",
    "tau_star_of",
]


class ElasticLimitShearError(ValueError):
    """Raised for invalid case parameters or a failed physics gate."""


# ---------------------------------------------------------------------------
# case definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ElasticLimitShearCase:
    """One plane-Couette shear point (DEM units unless stated)."""

    n_particles: int = 400
    diameter: float = 1.0                 # small diameter = reference length d
    size_ratio: float = 1.4
    density: float = 1.0                  # areal density rho
    solid_fraction: float = 0.78          # bulk area fraction (constant volume)
    gap_over_d: float = 16.0              # wall-to-wall gap / d
    kn: float = 1.0e4
    kt_ratio: float = 0.5
    mu: float = 0.5
    restitution: float = 0.70
    wall_velocity: float = 4.0            # |V| of each wall (gamma_dot = 2V/gap)
    wall_mu: Optional[float] = None       # None -> mu
    wall_roughness: float = 0.10          # wall disk overlap fraction of d
    # numerics
    dt: Optional[float] = None            # None -> 0.05 x normal contact period
    total_strain: float = 6.0
    average_from: float = 0.5             # fraction of the run that is averaged
    sample_every: int = 20
    seed: int = 16136
    jitter: float = 0.25
    gate_tol: float = 0.30
    max_overlap: float = 0.60             # gate on max overlap / d
    max_slip: float = 0.60                # gate on wall slip / wall speed (info)

    # ------------------------------------------------------------------ props
    @property
    def radii(self) -> np.ndarray:
        r = np.empty(self.n_particles)
        r[0::2] = 0.5 * self.diameter
        r[1::2] = 0.5 * self.diameter * self.size_ratio
        return r

    @property
    def diameters(self) -> np.ndarray:
        return 2.0 * self.radii

    @property
    def masses(self) -> np.ndarray:
        return self.density * math.pi * self.radii ** 2      # 2-D disks

    @property
    def d_ref(self) -> float:
        """Reference diameter (number average) = Campbell's ``d``."""
        return float(np.mean(self.diameters))

    @property
    def wall_radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def gap(self) -> float:
        return self.gap_over_d * self.diameter

    @property
    def width(self) -> float:
        """Box width that gives the requested bulk solid fraction."""
        area = float(np.sum(math.pi * self.radii ** 2))
        return area / (self.solid_fraction * self.gap)

    @property
    def shear_rate(self) -> float:
        return 2.0 * self.wall_velocity / self.gap

    @property
    def normal_period(self) -> float:
        m_eff = 0.5 * float(np.mean(self.masses))
        return math.pi * math.sqrt(m_eff / self.kn)

    @property
    def timestep(self) -> float:
        return self.dt if self.dt is not None else 0.05 * self.normal_period

    @property
    def kappa(self) -> float:
        return kappa_of(self.kn, self.density, self.d_ref, self.shear_rate)

    @property
    def steps(self) -> int:
        return int(round(self.total_strain / (self.shear_rate * self.timestep)))

    @property
    def wall_mu_value(self) -> float:
        return self.mu if self.wall_mu is None else self.wall_mu

    def __post_init__(self) -> None:
        if self.n_particles < 16 or self.n_particles % 2:
            raise ElasticLimitShearError("n_particles must be even and >= 16")
        for name in ("diameter", "density", "kn", "wall_velocity",
                     "gap_over_d", "total_strain"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):
                raise ElasticLimitShearError(f"{name} must be finite and positive")
        if not 0.0 < self.solid_fraction < 0.85:
            raise ElasticLimitShearError("solid_fraction must lie in (0, 0.85)")
        if self.size_ratio < 1.0:
            raise ElasticLimitShearError("size_ratio must be >= 1")
        if not 0.0 <= self.restitution < 1.0:
            raise ElasticLimitShearError("restitution must lie in [0, 1)")
        if self.mu < 0.0:
            raise ElasticLimitShearError("mu must be non-negative")


# ---------------------------------------------------------------------------
# dimensionless groups
# ---------------------------------------------------------------------------
def kappa_of(kn: float, rho: float, d: float, shear_rate: float) -> float:
    """Campbell's dimensionless stiffness ``k_n / (rho d^3 gdot^2)``."""
    return kn / (rho * d ** 3 * shear_rate ** 2)


def tau_star_of(tau: float, rho: float, d: float, shear_rate: float) -> float:
    """Dimensionless shear stress ``tau / (rho d^2 gdot^2)``."""
    return tau / (rho * d ** 2 * shear_rate ** 2)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _minimum_image(d: np.ndarray, box: float) -> np.ndarray:
    return d - box * np.round(d / box)


def _wall_positions(case: ElasticLimitShearCase, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Wall disk centres and velocities for both walls (bottom, top)."""
    lx = case.width
    rw = case.wall_radius
    pitch = 2.0 * rw * (1.0 - case.wall_roughness)
    nw = max(4, int(round(lx / pitch)))
    x = (np.arange(nw) + 0.5) * lx / nw
    bottom = np.stack([x, np.full(nw, rw)], axis=1)
    top = np.stack([x, np.full(nw, case.gap + rw)], axis=1)
    pos = np.concatenate([bottom, top], axis=0)
    pos[:, 0] = np.mod(pos[:, 0] + shift, lx)
    vel = np.zeros_like(pos)
    vel[:nw, 0] = -case.wall_velocity
    vel[nw:, 0] = +case.wall_velocity
    return pos, vel


def _inside_walls(case: ElasticLimitShearCase, y: np.ndarray,
                  radii: np.ndarray) -> np.ndarray:
    """True where a bulk disk centre keeps its surface inside the wall faces."""
    lo = case.wall_radius + radii + 1e-3
    hi = case.gap + case.wall_radius - radii - 1e-3
    return (y > lo) & (y < hi)


def _lattice_in_gap(case: ElasticLimitShearCase) -> np.ndarray:
    """Jittered triangular lattice filling the gap (affinely stretched)."""
    n = case.n_particles
    lx = case.width
    gap = case.gap
    nrow = max(2, int(math.ceil(math.sqrt(n * 2.0 / math.sqrt(3.0) * gap / lx))))
    ncol = int(math.ceil(n / nrow))
    rng = np.random.default_rng(case.seed)
    pts = []
    for row in range(nrow):
        for col in range(ncol):
            pts.append((col + 0.5 * (row % 2), row * math.sqrt(3.0) / 2.0))
    pts = np.array(pts[:n], dtype=float)
    width = ncol + 0.5
    height = nrow * math.sqrt(3.0) / 2.0
    lo = case.wall_radius
    rmax = float(case.radii.max())
    pts[:, 0] = (pts[:, 0] + 0.25) * lx / width
    pts[:, 1] = lo + rmax + (pts[:, 1] + 0.1) * (gap - 2.0 * rmax) / height
    pts += rng.uniform(-1.0, 1.0, size=pts.shape) * 0.5 * case.jitter * min(lx / width, gap / height)
    pts[:, 0] = np.mod(pts[:, 0], lx)
    return pts


def _relax_overlaps(case: ElasticLimitShearCase, pos: np.ndarray, *,
                    iterations: int = 800, tol: float = 2e-3) -> np.ndarray:
    """Push overlapping disks apart at fixed volume (POCA-style relaxation).

    A lattice at a dense solids fraction interpenetrates for a size ratio > 1;
    an overlapping start injects elastic prestress unrelated to the sheared
    steady state.  This geometric relaxation gives an overlap-free dense start
    without changing the solids fraction.
    """
    lx = case.width
    gap = case.gap
    n = case.n_particles
    radii = case.radii
    iu, ju = np.triu_indices(n, 1)
    rs = radii[iu] + radii[ju]
    lo = case.wall_radius
    pos = pos.copy()
    for _ in range(iterations):
        dx = pos[ju, 0] - pos[iu, 0]
        dx -= lx * np.round(dx / lx)
        dy = pos[ju, 1] - pos[iu, 1]
        r = np.hypot(dx, dy)
        over = rs - r
        bad = over > tol * case.diameter
        if not np.any(bad):
            break
        nx = dx[bad] / np.maximum(r[bad], 1e-12)
        ny = dy[bad] / np.maximum(r[bad], 1e-12)
        push = 0.5 * over[bad]
        disp = np.zeros((n, 2))
        np.add.at(disp[:, 0], iu[bad], -push * nx)
        np.add.at(disp[:, 0], ju[bad], +push * nx)
        np.add.at(disp[:, 1], iu[bad], -push * ny)
        np.add.at(disp[:, 1], ju[bad], +push * ny)
        pos += disp
        pos[:, 0] = np.mod(pos[:, 0], lx)
        pos[:, 1] = np.clip(pos[:, 1], lo + radii + 1e-3, gap + lo - radii - 1e-3)
    return pos


def build_configuration(case: ElasticLimitShearCase) -> dict:
    """Overlap-free dense start inside the Couette gap."""
    pos = _relax_overlaps(case, _lattice_in_gap(case))
    return {"pos": pos, "vel": np.zeros_like(pos), "shift": 0.0}


def configuration_report(case: ElasticLimitShearCase, pos: np.ndarray) -> dict:
    """Overlap and wall-clearance diagnostics of a starting configuration."""
    lx = case.width
    iu, ju = np.triu_indices(case.n_particles, 1)
    dx = pos[ju, 0] - pos[iu, 0]
    dx -= lx * np.round(dx / lx)
    dy = pos[ju, 1] - pos[iu, 1]
    r = np.hypot(dx, dy)
    rs = case.radii[iu] + case.radii[ju]
    overlap = float(np.max((rs - r) / case.diameter))
    n_contacts = int(np.sum(r < rs))
    return {"width": lx, "gap": case.gap,
            "min_separation_over_d": float(r.min() / case.diameter),
            "max_initial_overlap_over_d": max(overlap, 0.0),
            "initial_contacts_per_particle": n_contacts / case.n_particles,
            "shear_rate": case.shear_rate, "kappa": case.kappa}


# ---------------------------------------------------------------------------
# integrator
# ---------------------------------------------------------------------------
def simulate_couette_shear(case: ElasticLimitShearCase, *,
                           config: Optional[dict] = None,
                           return_series: bool = False) -> dict:
    """Run one constant-gap Couette point; return the steady stress state."""
    lx = case.width
    gap = case.gap
    radii = case.radii
    masses = case.masses
    n = case.n_particles
    if config is None:
        config = build_configuration(case)
    pos = np.array(config["pos"], dtype=float, copy=True)
    vel = np.array(config["vel"], dtype=float, copy=True)
    shift = float(config.get("shift", 0.0))

    kn = case.kn
    kt = case.kt_ratio * kn
    gdot = case.shear_rate
    dt = case.timestep
    m_eff = 0.5 * float(np.mean(masses))
    e = min(max(case.restitution, 1e-6), 1.0 - 1e-12)
    zeta = abs(math.log(e)) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
    gamma_n = 2.0 * zeta * math.sqrt(kn * m_eff)
    gamma_w = 2.0 * zeta * math.sqrt(kn * m_eff)

    iu, ju = np.triu_indices(n, 1)
    rs = radii[iu] + radii[ju]
    inv_m = 1.0 / masses
    n_flat = n * n
    tanx = np.zeros(n_flat)
    tany = np.zeros(n_flat)
    prev_active = np.zeros(n_flat, dtype=bool)

    def pair_forces(pos, vel, wpos, wvel):
        """Bulk-bulk contacts; returns force, virial, dissipation, diagnostics."""
        dx = pos[ju, 0] - pos[iu, 0]
        dx -= lx * np.round(dx / lx)
        dy = pos[ju, 1] - pos[iu, 1]
        r2 = dx * dx + dy * dy
        active = r2 < rs * rs
        force = np.zeros((n, 2))
        out = {"n_contacts": 0, "max_overlap": 0.0, "dissipation": 0.0,
               "virial_xx": 0.0, "virial_yy": 0.0, "virial_xy": 0.0,
               "sliding_fraction": 0.0}
        if np.any(active):
            ia, ja = iu[active], ju[active]
            dxa, dya = dx[active], dy[active]
            r = np.sqrt(r2[active])
            nx, ny = dxa / r, dya / r
            delta = rs[active] - r
            dvx = vel[ja, 0] - vel[ia, 0]
            dvy = vel[ja, 1] - vel[ia, 1]
            vn = dvx * nx + dvy * ny
            fn_mag = np.maximum(kn * delta - gamma_n * vn, 0.0)
            tx, ty = -ny, nx
            vt = dvx * tx + dvy * ty
            pid = ia * n + ja
            active_flat = np.zeros(n_flat, dtype=bool)
            active_flat[pid] = True
            fresh = active_flat & ~prev_active
            tanx[fresh] = 0.0
            tany[fresh] = 0.0
            prev_active[:] = active_flat
            tanx[pid] += vt * tx * dt
            tany[pid] += vt * ty * dt
            ft_mag = kt * np.hypot(tanx[pid], tany[pid])
            cap = case.mu * fn_mag
            slide = ft_mag > cap
            if np.any(slide):
                scale = np.where(ft_mag > 0.0, cap / np.maximum(ft_mag, 1e-300), 1.0)
                tanx[pid] *= scale
                tany[pid] *= scale
            ftx = kt * tanx[pid]
            fty = kt * tany[pid]
            fx = fn_mag * nx + ftx
            fy = fn_mag * ny + fty
            force[:, 0] = (np.bincount(ja, weights=fx, minlength=n)
                           - np.bincount(ia, weights=fx, minlength=n))
            force[:, 1] = (np.bincount(ja, weights=fy, minlength=n)
                           - np.bincount(ia, weights=fy, minlength=n))
            diss = float(np.sum(gamma_n * np.minimum(vn, 0.0) ** 2)
                         + np.sum(np.where(slide, cap * np.abs(vt), 0.0)))
            out.update({
                "n_contacts": int(ia.size),
                "max_overlap": float(delta.max() / case.diameter),
                "dissipation": diss,
                "virial_xx": float(-np.sum(fx * dxa)),
                "virial_yy": float(-np.sum(fy * dya)),
                "virial_xy": float(-np.sum(fx * dya)),
                "sliding_fraction": float(np.mean(slide)) if ia.size else 0.0,
            })
        return force, out

    def wall_forces(pos, vel, wpos, wvel):
        """Bulk-wall contacts: force on the bulk particles + force on the walls."""
        rw = case.wall_radius
        d = pos[:, None, :] - wpos[None, :, :]
        d[:, :, 0] -= lx * np.round(d[:, :, 0] / lx)
        nw_all = wpos.shape[0]
        rs_ij = radii[:, None] + rw * np.ones((1, nw_all))
        r2 = d[:, :, 0] ** 2 + d[:, :, 1] ** 2
        mask = r2 < rs_ij ** 2
        fb = np.zeros((n, 2))
        fw = np.zeros((wpos.shape[0], 2))
        diss = 0.0
        max_overlap = 0.0
        if not np.any(mask):
            return fb, fw, diss, max_overlap
        ii, jj = np.nonzero(mask)
        dx = d[ii, jj, 0]
        dy = d[ii, jj, 1]
        r = np.sqrt(r2[ii, jj])
        nx, ny = dx / r, dy / r
        delta = rs_ij[ii, jj] - r
        dvx = vel[ii, 0] - wvel[jj, 0]
        dvy = vel[ii, 1] - wvel[jj, 1]
        vn = dvx * nx + dvy * ny
        fn_mag = np.maximum(kn * delta - gamma_w * vn, 0.0)
        # wall friction: memoryless Coulomb (the wall does not carry a spring)
        tx, ty = -ny, nx
        vt = dvx * tx + dvy * ty
        ft_mag = np.minimum(case.wall_mu_value * fn_mag,
                            np.abs(vt) * 0.5 * math.sqrt(kn * m_eff))
        ft_mag = np.where(vt > 0.0, -ft_mag, ft_mag)
        fx = fn_mag * nx + ft_mag * tx
        fy = fn_mag * ny + ft_mag * ty
        fb[:, 0] = np.bincount(ii, weights=fx, minlength=n)
        fb[:, 1] = np.bincount(ii, weights=fy, minlength=n)
        fw[:, 0] = -np.bincount(jj, weights=fx, minlength=wpos.shape[0])
        fw[:, 1] = -np.bincount(jj, weights=fy, minlength=wpos.shape[0])
        diss = float(np.sum(gamma_w * np.minimum(vn, 0.0) ** 2)
                     + np.sum(np.abs(ft_mag * vt)))
        max_overlap = float(delta.max() / case.diameter)
        return fb, fw, diss, max_overlap

    series = {"strain": [], "tau": [], "pressure": [], "tau_star": [],
              "temperature": [], "coordination": []}
    steps = case.steps
    strain = 0.0
    acc = {"tau": 0.0, "p": 0.0, "tkin": 0.0, "diss": 0.0, "work": 0.0,
           "ovl": 0.0, "contacts": 0, "slip": 0.0, "samples": 0}
    tau_hist: list[float] = []
    lx_gap = lx * gap
    for step in range(steps):
        shift += 2.0 * case.wall_velocity * dt
        wpos, wvel = _wall_positions(case, shift)
        fb, diag = pair_forces(pos, vel, wpos, wvel)
        fw_b, fw_w, diss_w, ovl_w = wall_forces(pos, vel, wpos, wvel)
        acc_total = (fb + fw_b) * inv_m[:, None]
        vel += acc_total * dt
        pos += vel * dt
        pos[:, 0] = np.mod(pos[:, 0], lx)
        strain += gdot * dt

        if step % case.sample_every == 0:
            # measurement pass on the advanced state (walls at the new shift)
            _, d_now = pair_forces(pos, vel, wpos, wvel)
            _, fw_now, diss_now, ovl_now = wall_forces(pos, vel, wpos, wvel)
            nw = fw_now.shape[0] // 2
            # momentum flux on the walls: tau = -F_x(top)/Lx, P = F_y(top)/Lx
            tau_top = -float(np.sum(fw_now[nw:, 0])) / lx
            tau_bot = -float(np.sum(fw_now[:nw, 0])) / lx
            p_top = float(np.sum(fw_now[nw:, 1])) / lx
            p_bot = -float(np.sum(fw_now[:nw, 1])) / lx
            tau_w = 0.5 * (tau_top + tau_bot)
            p_w = 0.5 * (p_top + p_bot)
            tkin = 0.5 * float(np.sum(masses * (vel[:, 0] ** 2 + vel[:, 1] ** 2))) / n
            series["strain"].append(strain)
            series["tau"].append(tau_w)
            series["pressure"].append(p_w)
            series["tau_star"].append(tau_star_of(abs(tau_w), case.density,
                                                  case.d_ref, gdot))
            series["temperature"].append(tkin)
            series["coordination"].append(d_now["n_contacts"] / n)
            if strain >= case.average_from * case.total_strain:
                acc["tau"] += tau_w
                acc["p"] += p_w
                acc["tkin"] += tkin
                acc["diss"] += d_now["dissipation"] + diss_now
                acc["work"] += tau_w * gdot * lx_gap
                acc["ovl"] += max(d_now["max_overlap"], ovl_now)
                acc["contacts"] += d_now["n_contacts"]
                acc["samples"] += 1
                tau_hist.append(abs(tau_w))

    ns = max(acc["samples"], 1)
    tau = acc["tau"] / ns
    p_mean = acc["p"] / ns
    work = acc["work"] / ns
    diss = acc["diss"] / ns
    drift = _relative_drift(tau_hist)
    gate = {
        "energy_balance_rel": float(abs(work - diss) / max(abs(work), 1e-300)),
        "max_overlap_over_d": float(acc["ovl"] / ns),
        "steady_drift": drift,
    }
    gate["energy_gate"] = bool(gate["energy_balance_rel"] <= case.gate_tol)
    gate["overlap_gate"] = bool(gate["max_overlap_over_d"] <= case.max_overlap)
    gate["steady_gate"] = bool(drift <= case.gate_tol)
    result = {
        "kappa": case.kappa,
        "kn": case.kn,
        "shear_rate": gdot,
        "wall_velocity": case.wall_velocity,
        "solid_fraction": case.solid_fraction,
        "n_particles": case.n_particles,
        "d_ref": case.d_ref,
        "dt": dt,
        "steps": steps,
        "inertial_number": (float(gdot * case.d_ref
                                  * math.sqrt(case.density / p_mean))
                            if p_mean > 0.0 else float("nan")),
        "tau": float(tau),
        "tau_abs": float(abs(tau)),
        "pressure": float(p_mean),
        "mu_macro": float(abs(tau) / p_mean) if p_mean > 0.0 else float("nan"),
        "tau_star": float(tau_star_of(abs(tau), case.density, case.d_ref, gdot)),
        "coordination": float(acc["contacts"] / (ns * n)),
        "temperature": float(acc["tkin"] / ns),
        "gate": gate,
        "gate_pass": bool(gate["energy_gate"] and gate["overlap_gate"]
                          and gate["steady_gate"]),
    }
    if return_series:
        result["series"] = series
    return result


def _relative_drift(hist: list[float]) -> float:
    """Drift of |tau| between the third quarter and the last quarter."""
    if len(hist) < 8:
        return float("nan")
    q = max(1, len(hist) // 4)
    early = float(np.mean(hist[-2 * q:-q]))
    late = float(np.mean(hist[-q:]))
    if early <= 0.0:
        return float("nan")
    return abs(late - early) / early


# ---------------------------------------------------------------------------
# scaling analysis
# ---------------------------------------------------------------------------
def shear_stress_scaling(points: list[dict], *, bagnold_tol: float = 0.20) -> dict:
    """Fit ``tau*(kappa)`` and report the exponents against the Bagnold law.

    ``beta = d ln tau* / d ln kappa`` in the plateau window and
    ``n = 2 (1 - beta)`` is the equivalent shear-rate exponent.  The regime
    boundary ``kappa_c`` is the largest ``kappa`` at which the local exponent
    leaves the Bagnold value ``n = 2`` by more than ``bagnold_tol`` (declared a
    priori in the case document).
    """
    good = [p for p in points if p.get("gate_pass", True)]
    if len(good) < 3:
        raise ElasticLimitShearError("need at least 3 gated points for a fit")
    good = sorted(good, key=lambda p: p["kappa"])
    kappa = np.array([p["kappa"] for p in good], dtype=float)
    tau_star = np.array([p["tau_star"] for p in good], dtype=float)
    lk = np.log(kappa)
    lt = np.log(np.maximum(tau_star, 1e-300))
    beta_global = float(np.polyfit(lk, lt, 1)[0])
    hi = good[-max(2, len(good) // 4):]
    plateau = float(np.mean([p["tau_star"] for p in hi]))
    scatter = float(np.std([p["tau_star"] for p in hi]) / max(plateau, 1e-300))
    beta_hi = float(np.polyfit(np.log([p["kappa"] for p in hi]),
                               np.log([p["tau_star"] for p in hi]), 1)[0])
    local = []
    for i in range(1, len(good) - 1):
        k1, k2 = good[i - 1]["kappa"], good[i + 1]["kappa"]
        t1, t2 = good[i - 1]["tau_star"], good[i + 1]["tau_star"]
        b = (math.log(t2) - math.log(t1)) / (math.log(k2) - math.log(k1))
        local.append({"kappa": good[i]["kappa"],
                      "tau_star": good[i]["tau_star"],
                      "beta_local": b, "n_local": 2.0 * (1.0 - b)})
    kappa_c = None
    for entry in sorted(local, key=lambda e: -e["kappa"]):
        if abs(entry["n_local"] - 2.0) / 2.0 > bagnold_tol:
            kappa_c = entry["kappa"]
            break
    return {
        "beta_global": beta_global,
        "n_global": 2.0 * (1.0 - beta_global),
        "beta_plateau": beta_hi,
        "n_plateau": 2.0 * (1.0 - beta_hi),
        "tau_star_plateau": plateau,
        "tau_star_plateau_scatter": scatter,
        "plateau_kappa_range": [float(hi[0]["kappa"]), float(hi[-1]["kappa"])],
        "local": local,
        "kappa_c_leaving_bagnold": kappa_c,
        "points": good,
    }
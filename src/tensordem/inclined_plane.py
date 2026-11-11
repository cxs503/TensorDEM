"""Chute flow: granular flow down an inclined, rough plane.

Reproduction target: Silbert, Ertaş, Grest, Halsey, Levine & Plimpton,
*Granular flow down an inclined plane: Bagnold scaling and rheology*,
Phys. Rev. E **64**, 051302 (2001) -- see
``benchmarks/repro_dem_inclined_plane_m15145/``.

Geometry (paper convention, rotated so that ``y`` is height above the base):

* ``x`` -- down-slope, periodic (period ``lx``),
* ``z`` -- spanwise, periodic (period ``lz``),
* ``y`` -- normal to the base, open free surface,
* base  -- a layer of *glued* (frozen) spheres sitting on the ``y = 0`` plane,
  i.e. the paper's "rough base"; gravity ``g`` has components
  ``g_x = g sin(theta)`` and ``g_y = -g cos(theta)``.

Contact law: the shared :mod:`tensordem.granular3d_core` kernel (linear
spring-dashpot normal force, Cundall-Strack tangential spring saturated at
``mu |F_n|``, optional rolling resistance), which is the paper's "Model L"
(``k_n``, ``k_t/k_n = 2/7``, ``mu = 0.50``, restitution ``e = 0.88``,
no rolling resistance).

Periodic images use the ``box`` argument added to
:func:`tensordem.granular3d_core.build_pairs` /
:func:`tensordem.granular3d_core.contact_forces`; a Verlet skin keeps the
neighbour list valid for a few steps.

Units: lengths in particle diameters ``d``, masses in particle masses ``m``,
times in ``sqrt(d/g)`` -- set ``radius = 0.5``, ``mass = 1``, ``g = 1`` and
every reported number is dimensionless in that system (as in the paper).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .granular3d_core import (
    ContactState,
    Material3D,
    build_pairs,
    contact_forces,
    minimum_image,
)

PHI_INIT = 0.575          # initial/steady bulk packing fraction target

__all__ = [
    "InclinedPlaneCase",
    "build_chute",
    "simulate_chute",
    "flow_metrics",
    "fit_bagnold_exponent",
    "fit_near_base_power_law",
    "fit_stress_rate_exponent",
    "PHI_INIT",
    "density_velocity_profile",
    "main",
]


@dataclass(frozen=True)
class InclinedPlaneCase:
    """Geometry + material of one chute-flow run (dimensionless units)."""

    theta_deg: float = 22.0
    #: dimensionless pile height H = N d^2 / (lx lz)
    height: float = 20.0
    lx: float = 16.0                     # down-slope period (diameters)
    lz: float = 8.0                      # spanwise period (diameters)
    #: mobile particles are (re)packed on an hcp lattice; jitter breaks layering
    packing_gap: float = 0.04
    packing_jitter: float = 0.03
    seed: int = 20010513
    # ---- particle / contact model (paper Model L3 defaults) ----------------
    radius: float = 0.5
    density: float = 1.909859317102744     # -> particle mass = 1 in d=m=g=1 units
    kn: float = 2.0e4                     # paper: 2e5 (softened, see doc)
    kt_ratio: float = 2.0 / 7.0           # paper: 2/7
    mu: float = 0.50                      # paper: 0.50
    mu_rolling: float = 0.0               # paper: none
    restitution: float = 0.88             # paper: 0.88
    gravity: float = 1.0
    # ---- base plate -------------------------------------------------------
    base_layers: int = 1
    base_gap: float = 0.06                # lattice gap of the glued layer
    # ---- integration --------------------------------------------------------
    dt: float | None = None               # default: material.timestep / 2
    skin: float = 0.35                    # Verlet skin (diameters)
    pair_refresh: int = 8                 # rebuild neighbour list every n steps

    # ------------------------------------------------------------- derived
    @property
    def material(self) -> Material3D:
        return Material3D(
            radius=self.radius,
            density=self.density,
            mu=self.mu,
            mu_rolling=self.mu_rolling,
            restitution=self.restitution,
            kn=self.kn,
            kt_ratio=self.kt_ratio,
            gravity=self.gravity,
        )

    @property
    def box(self) -> tuple[float, float, float]:
        """Periodic box: x and z periodic, y open (``inf``)."""
        return (self.lx, math.inf, self.lz)

    @property
    def n_mobile(self) -> int:
        return int(round(self.height * self.lx * self.lz))

    @property
    def gravity_vector(self) -> np.ndarray:
        th = math.radians(self.theta_deg)
        return np.array([self.gravity * math.sin(th),
                         -self.gravity * math.cos(th), 0.0])

    @property
    def timestep(self) -> float:
        return self.dt if self.dt is not None else self.material.timestep


def build_chute(case: InclinedPlaneCase) -> dict:
    """Build the glued base + the mobile hcp packing.

    Returns ``pos (n,3)``, ``vel``, ``omega``, ``mobile`` (bool mask) and the
    index of the first mobile particle (the base occupies ``[0, n_base)``).
    """
    mat = case.material
    r = mat.radius
    # glue the base on a triangular lattice that tiles the periodic cell exactly
    base = _triangular_layer(r, case.lx, case.lz, case.base_gap,
                             layers=case.base_layers)

    # mobile particles: hcp slab, then trim to the requested particle count
    gen = _hcp_slab(mat.radius, case.lx, case.lz, case.height,
                    case.packing_gap, case.packing_jitter, case.seed)
    if len(gen) > case.n_mobile:
        order = np.argsort(gen[:, 1])[: case.n_mobile]
        gen = gen[np.sort(order)]
    # lift above the glued layer
    gen[:, 1] += 2.0 * r
    pos = np.vstack([base, gen])
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    mobile = np.zeros(len(pos), dtype=bool)
    mobile[len(base):] = True
    return {"pos": pos, "vel": vel, "omega": omega, "mobile": mobile,
            "n_base": len(base)}


def _layer_min_dist(lx: float, lz: float, points: np.ndarray) -> float:
    """Minimum periodic in-plane distance of a set of centres."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points, boxsize=np.array([lx, np.inf, lz]))
    pairs = tree.query_pairs(2.0 * 1.05, output_type="ndarray")
    if len(pairs) == 0:
        return float("inf")
    d = minimum_image(points[pairs[:, 1]] - points[pairs[:, 0]],
                      (lx, np.inf, lz))
    return float(np.linalg.norm(d, axis=1).min())


def _layer_lattice(radius: float, lx: float, lz: float, gap: float):
    """In-plane triangular lattice that tiles the periodic cell without overlap.

    Returns ``(nx, nz, sq, sz)``.  ``nz`` is the largest row count for which the
    periodic layer still keeps every centre pair at least one diameter apart
    (the seam row is checked too -- a naive ``round(lz/sz_ideal)`` leaves a
    short seam gap that overlaps).
    """
    d = 2.0 * radius
    spacing = d * (1.0 + gap)
    nx = max(1, int(math.floor(lx / spacing)))
    sq = lx / nx
    nz_max = max(1, int(math.ceil(lz / (sq * math.sqrt(3.0) / 2.0))) + 2)
    for nz in range(nz_max, 0, -1):
        sz = lz / nz
        pts = np.array([((ix * sq + 0.5 * sq * (iz % 2)) % lx, 0.0, iz * sz)
                        for iz in range(nz) for ix in range(nx)])
        if _layer_min_dist(lx, lz, pts) >= d * (1.0 - 1e-9):
            return nx, nz, sq, sz
    return 1, 1, lx, lz


def _triangular_layer(radius: float, lx: float, lz: float, gap: float,
                      layers: int = 1) -> np.ndarray:
    """Triangular lattice filling the periodic cell ``lx x lz`` exactly."""
    d = 2.0 * radius
    nx, nz, sq, sz = _layer_lattice(radius, lx, lz, gap)
    pts: list[tuple[float, float, float]] = []
    for layer in range(max(1, layers)):
        y = radius + layer * d
        for iz in range(nz):
            for ix in range(nx):
                x = (ix * sq + 0.5 * sq * (iz % 2)) % lx
                pts.append((x, y, iz * sz))
    return np.asarray(pts, dtype=np.float64)


def _hcp_slab(radius: float, lx: float, lz: float, height: float,
              gap: float, jitter: float, seed: int) -> np.ndarray:
    """Layered hcp-ish slab filling ``lx x lz`` up to ``height`` (y vertical).

    The vertical layer gap is chosen from the target packing fraction
    ``PHI_INIT`` (paper's 3-D bulk packing is ~0.57-0.59), so the bed starts
    close to its steady density instead of collapsing/dilating violently.
    """
    d = 2.0 * radius
    nx, nz, sq, sz = _layer_lattice(radius, lx, lz, gap)
    vp = (4.0 / 3.0) * math.pi * radius ** 3
    dy = max(d, vp / (PHI_INIT * sq * sz))
    n_layers = max(1, int(math.ceil((height + radius) / dy)))
    pts: list[np.ndarray] = []
    for layer in range(n_layers):
        pts.append(np.array([((ix * sq + 0.5 * sq * (iz % 2)) % lx,
                              radius + layer * dy, iz * sz)
                             for iz in range(nz) for ix in range(nx)]))
    pos = np.vstack(pts)
    if jitter:
        rng = np.random.default_rng(seed)
        pos += rng.standard_normal(pos.shape) * (jitter * radius)
        pos[:, 0] %= lx
        pos[:, 2] %= lz
    pos[:, 1] = np.maximum(pos[:, 1], radius * 1.001)
    return pos


def simulate_chute(case: InclinedPlaneCase, *, t_end: float = 60.0,
                   steady_frac: float = 0.4, sample_every: int = 40,
                   bin_width: float = 1.0, init: dict | None = None,
                   verbose: bool = False) -> dict:
    """Integrate the chute flow and return profiles + history.

    ``steady_frac``: fraction of ``t_end`` at the end over which the reported
    (time-averaged) profiles are taken.  A flow is classified as *flowing* when
    the late-time kinetic energy per particle stays above ``tol_ke`` (see
    :func:`flow_metrics`), otherwise it has *stopped*.
    """
    mat = case.material
    dt = case.timestep
    st = build_chute(case) if init is None else init
    pos, vel, omega = st["pos"].copy(), st["vel"].copy(), st["omega"].copy()
    mobile, n_base = st["mobile"], st["n_base"]
    mass, inertia = mat.mass, mat.inertia
    gvec = case.gravity_vector
    box = case.box
    state = ContactState()
    n = len(pos)
    y_bins = np.arange(0.0, max(4.0, case.height + 6.0) + bin_width, bin_width)
    n_bins = len(y_bins) - 1
    bin_vol = bin_width * case.lx * case.lz

    prof = {"count": np.zeros(n_bins), "vx": np.zeros(n_bins),
            "vy": np.zeros(n_bins), "vx2": np.zeros(n_bins),
            "vz2": np.zeros(n_bins), "tau": np.zeros(n_bins),
            "tau_n": np.zeros(n_bins), "samples": 0}
    history: list[dict] = []
    iu = ju = None
    ref_pos = pos.copy()
    t0 = time.time()
    step = 0
    t = 0.0
    cutoff_radius = mat.radius * (1.0 + case.skin / (2.0 * mat.radius))
    print_every = max(1, int(round(0.05 * t_end / dt)))   # every 5% of t_end
    while t < t_end - 1e-12:
        # ---- neighbour list (Verlet skin) ---------------------------------
        if iu is None or step % case.pair_refresh == 0:
            disp = np.linalg.norm(pos - ref_pos, axis=1)
            if iu is None or disp.max() > 0.5 * case.skin:
                iu, ju = build_pairs(pos, cutoff_radius, box=box)
                ref_pos = pos.copy()
        # ---- forces --------------------------------------------------------
        force, torque = contact_forces(pos, vel, omega, iu, ju, state, mat, dt,
                                       box=box)
        # ---- integrate (semi-implicit Euler) -------------------------------
        acc = force / mass
        acc[mobile] += gvec
        vel += acc * dt
        vel[~mobile] = 0.0
        omega += torque / inertia * dt
        omega[~mobile] = 0.0
        pos += vel * dt
        pos[:, 0] %= case.lx
        pos[:, 2] %= case.lz
        pos[~mobile, 1] = st["pos"][~mobile, 1]
        if not np.isfinite(pos).all():
            raise RuntimeError(f"blow-up at t={t:.3f} (non-finite positions)")
        if np.any(pos[mobile, 1] < -mat.radius):
            raise RuntimeError(
                f"particle escaped the base at t={t:.3f} (y<0): geometry gap")
        t += dt
        step += 1

        if t >= t_end * (1.0 - steady_frac) and step % sample_every == 0:
            _accumulate(prof, pos, vel, iu, ju, force, mobile, y_bins, bin_vol,
                        box)
        if step % max(1, int(round(t_end / dt / 120))) == 0 or t >= t_end - 1e-12:
            ym = pos[mobile, 1]
            history.append({
                "t": t,
                "ke_per_particle": float(0.5 * mass * (vel[mobile] ** 2).sum()
                                         / max(1, mobile.sum())),
                "vx_mean": float(vel[mobile, 0].mean()),
                "vx_max": float(vel[mobile, 0].max()),
                "y_mean": float(ym.mean()),
                "y_p95": float(np.percentile(ym, 95)),
            })
        if verbose and step % print_every == 0:
            h = history[-1]
            print(f"t={t:7.2f} KE/N={h['ke_per_particle']:.4f} "
                  f"<vx>={h['vx_mean']:.2f} y95={h['y_p95']:.1f}", flush=True)

    wall = time.time() - t0
    profiles = _finalise_profiles(prof, y_bins, case)
    metrics = flow_metrics(history)
    fits = fit_bagnold_exponent(profiles, case)
    return {
        "case": {k: v for k, v in asdict(case).items()},
        "n_particles": int(mobile.sum()),
        "n_base": int(n_base),
        "dt": dt,
        "steps": step,
        "wall_seconds": wall,
        "history": history,
        "profiles": profiles,
        "metrics": metrics,
        "fits": fits,
    }


def _accumulate(prof, pos, vel, iu, ju, force, mobile, y_bins, bin_vol, box):
    n_bins = len(y_bins) - 1
    y = pos[mobile, 1]
    idx = np.clip(np.digitize(y, y_bins) - 1, 0, n_bins - 1)
    np.add.at(prof["count"], idx, 1.0)
    np.add.at(prof["vx"], idx, vel[mobile, 0])
    np.add.at(prof["vy"], idx, vel[mobile, 1])
    np.add.at(prof["vx2"], idx, vel[mobile, 0] ** 2)
    np.add.at(prof["vz2"], idx, vel[mobile, 2] ** 2)
    # virial shear stress: tau = sum_pairs d_y F_x / V_bin  (moduli per unit area)
    if len(iu):
        dvec = minimum_image(pos[ju] - pos[iu], box)
        fx = force[ju, 0]
        yc = 0.5 * (pos[iu, 1] + pos[ju, 1])
        bidx = np.clip(np.digitize(yc, y_bins) - 1, 0, n_bins - 1)
        # sigma_xy with tension-positive: sigma_xy = -(1/V) sum d_y f_x
        np.add.at(prof["tau"], bidx, -dvec[:, 1] * fx)
        np.add.at(prof["tau_n"], bidx, 1.0)
    prof["samples"] += 1


def _finalise_profiles(prof: dict, y_bins: np.ndarray, case) -> dict:
    n_bins = len(y_bins) - 1
    n_samp = max(1, prof["samples"])
    count = prof["count"] / n_samp
    vx = np.divide(prof["vx"], np.maximum(prof["count"], 1.0))
    vy = np.divide(prof["vy"], np.maximum(prof["count"], 1.0))
    vx2 = np.divide(prof["vx2"], np.maximum(prof["count"], 1.0))
    vz2 = np.divide(prof["vz2"], np.maximum(prof["count"], 1.0))
    vp = (4.0 / 3.0) * math.pi * case.radius ** 3
    phi = count * vp / (case.lx * case.lz * (y_bins[1] - y_bins[0]))
    tau = prof["tau"] / n_samp / (case.lx * case.lz
                                  * (y_bins[1] - y_bins[0]))
    return {
        "y_mid": (0.5 * (y_bins[1:] + y_bins[:-1])).tolist(),
        "y_edges": y_bins.tolist(),
        "phi": phi.tolist(),
        "vx": vx.tolist(),
        "vy": vy.tolist(),
        "granular_temperature": (vx2 + vz2 + vy ** 2).tolist(),
        "tau_xy": tau.tolist(),
        "n_samples": int(prof["samples"]),
    }


def flow_metrics(history: list[dict], tail_frac: float = 0.25) -> dict:
    """Steady-state vs stopped classification from the KE history."""
    if not history:
        return {"flowing": None, "ke_steady": float("nan"),
                "ke_decay_ratio": float("nan")}
    ke = np.array([h["ke_per_particle"] for h in history], dtype=float)
    t = np.array([h["t"] for h in history], dtype=float)
    n_tail = max(3, int(round(tail_frac * len(ke))))
    ke_tail = ke[-n_tail:]
    # decay = KE at the end divided by the peak of the early flow phase
    early = ke[: max(1, len(ke) // 4)].max()
    ratio = float(ke_tail.mean() / early) if early > 0 else float("nan")
    vx_tail = float(np.mean([h["vx_mean"] for h in history[-n_tail:]]))
    return {
        "flowing": bool(ratio > 0.02 and ke_tail.mean() > 1e-4),
        "ke_steady": float(ke_tail.mean()),
        "ke_peak": float(ke.max()),
        "ke_decay_ratio": ratio,
        "vx_mean_steady": vx_tail,
        "vx_max_steady": float(np.mean([h["vx_max"] for h in history[-n_tail:]])),
        "y_p95_steady": float(np.mean([h["y_p95"] for h in history[-n_tail:]])),
        "t_end": float(t[-1]),
    }


def fit_bagnold_exponent(profiles: dict, case: InclinedPlaneCase,
                         y_lo: float | None = None,
                         y_hi_frac: float = 0.85) -> dict:
    """Fit ``v_x(y) = V [1 - (1 - (y-y0)/h)^p]`` (paper Eq. 22) for ``p``.

    The paper's Bagnold profile is Eq. (22)::

        v_x(y) = A_Bag h^{3/2} (2/3) sqrt(rho g sin(theta))
                 [1 - ((h - y)/h)^{3/2}]

    i.e. ``p = 3/2`` with ``V`` the free-surface velocity.  We fit ``p``
    nonlinearly (grid search + linear amplitude) on the layers between the
    base (``y_lo``, default ``1.5 d``) and ``y_hi_frac * h``; the topmost
    dilating layers and the base layering are excluded, as in the paper.
    """
    y = np.asarray(profiles["y_mid"], dtype=float)
    vx = np.asarray(profiles["vx"], dtype=float)
    cnt = np.asarray(profiles["phi"], dtype=float)
    valid = cnt > 1e-6
    if valid.sum() < 4:
        return {"p": float("nan"), "note": "insufficient occupied bins"}
    phi_bulk = float(np.percentile(cnt[valid], 75))
    dense = cnt > 0.5 * phi_bulk
    # the free surface is where the bed stops being dense; above it the splash
    # ("kinetic gas") tail is not part of the flow profile
    y_surf = float(y[dense].max() + 0.5)
    y0 = 1.0 if y_lo is None else y_lo
    sel = valid & (y >= y0) & (y <= y_hi_frac * y_surf)
    if sel.sum() < 4:
        return {"p": float("nan"), "note": "insufficient bins in fit window"}
    yy = y[sel] - y0
    h = y_surf - y0
    vv = vx[sel]
    best = None
    for p in np.arange(0.5, 3.0, 0.005):
        shape = 1.0 - np.clip(1.0 - yy / h, 0.0, 1.0) ** p
        denom = float(shape @ shape)
        if denom <= 0:
            continue
        amp = float(shape @ vv) / denom
        sse = float(((vv - amp * shape) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, p, amp)
    sse, p, amp = best
    var = float(((vv - vv.mean()) ** 2).sum())
    # reference: the paper's p = 3/2 (Eq. 22), amplitude fitted the same way
    shape15 = 1.0 - np.clip(1.0 - yy / h, 0.0, 1.0) ** 1.5
    amp15 = float(shape15 @ vv) / max(float(shape15 @ shape15), 1e-30)
    sse15 = float(((vv - amp15 * shape15) ** 2).sum())
    return {
        "p": float(p),
        "V_surface": float(amp),
        "y0": y0,
        "y_surface": y_surf,
        "h": float(h),
        "n_bins": int(sel.sum()),
        "r2": float(1.0 - sse / var) if var > 0 else float("nan"),
        "p_target": 1.5,
        "rel_error": float(abs(p - 1.5) / 1.5),
        "V_surface_p15": float(amp15),
        "r2_p15": float(1.0 - sse15 / var) if var > 0 else float("nan"),
        "r2_best": float(1.0 - sse / var) if var > 0 else float("nan"),
        "stress_rate": fit_stress_rate_exponent(profiles, y_lo=y0,
                                                y_hi_frac=y_hi_frac),
        "near_base": fit_near_base_power_law(y, vx, cnt, y_lo=y0,
                                             y_hi=y0 + 6.0),
    }


def fit_stress_rate_exponent(profiles: dict, *, y_lo: float = 1.5,
                              y_hi_frac: float = 0.85) -> dict:
    """Bagnold rheology test: fit ``tau ~ |dvx/dy|^n`` (paper Eq. 21, ``n = 2``).

    Only contact (virial) stresses are used -- the streaming/kinetic part is
    negligible in the dense interior; the fitted window is the same bulk window
    as the velocity-profile fit.
    """
    y = np.asarray(profiles["y_mid"], dtype=float)
    vx = np.asarray(profiles["vx"], dtype=float)
    phi = np.asarray(profiles["phi"], dtype=float)
    tau = np.asarray(profiles["tau_xy"], dtype=float)
    phi_bulk = float(np.percentile(phi[phi > 1e-6], 75)) if (phi > 1e-6).any() \
        else 0.0
    dense = phi > 0.5 * phi_bulk
    y_surf = float(y[dense].max() + 0.5) if dense.any() else float(y.max())
    rate = np.gradient(vx, y)
    sel = (phi > 1e-6) & (y >= y_lo) & (y <= y_hi_frac * y_surf) \
        & (rate > 1e-6) & (tau > 1e-9)
    if sel.sum() < 3:
        return {"n": float("nan"), "n_bins": int(sel.sum())}
    lr = np.log(rate[sel])
    lt = np.log(tau[sel])
    a, b = np.polyfit(lr, lt, 1)
    pred = a * lr + b
    var = float(((lt - lt.mean()) ** 2).sum())
    return {"n": float(a), "log_amplitude": float(b),
            "n_bins": int(sel.sum()),
            "r2": float(1.0 - ((lt - pred) ** 2).sum() / var) if var > 0 else
            float("nan"),
            "tau_max": float(tau[sel].max())}


def fit_near_base_power_law(y: np.ndarray, vx: np.ndarray, phi: np.ndarray,
                            *, y_lo: float = 1.0, y_hi: float = 7.0) -> dict:
    """Local power law ``v_x ~ (y - y_lo)^q`` in the layers next to the base.

    ``q = 3/2`` is the constant-shear-stress Bagnold result; the paper's
    linear-stress profile Eq. (22) gives ``q = 1`` close to the base and
    ``q -> 3/2`` only in the "delta-free" asymptote, so both are reported.
    """
    sel = (phi > 1e-6) & (y >= y_lo) & (y <= y_hi) & (vx > 0)
    if sel.sum() < 3:
        return {"q": float("nan"), "n_bins": int(sel.sum())}
    ly = np.log(y[sel] - y_lo + 1e-12)
    lv = np.log(vx[sel])
    a, b = np.polyfit(ly, lv, 1)
    pred = a * ly + b
    var = float(((lv - lv.mean()) ** 2).sum())
    return {"q": float(a), "amplitude": float(math.exp(b)),
            "n_bins": int(sel.sum()),
            "r2": float(1.0 - ((lv - pred) ** 2).sum() / var) if var > 0 else
            float("nan")}


def density_velocity_profile(result: dict, *, dense_frac: float = 0.5) -> dict:
    """Bulk packing fraction + velocity of the dense interior (paper Fig. 12).

    "Dense" bins are those above ``dense_frac`` times the 75th percentile of the
    packing-fraction profile, which removes both the base layering and the
    dilute splash tail above the free surface.  The bulk value is the mean over
    the middle 60 % of that window (away from both boundaries).
    """
    phi = np.asarray(result["profiles"]["phi"], dtype=float)
    y = np.asarray(result["profiles"]["y_mid"], dtype=float)
    vx = np.asarray(result["profiles"]["vx"], dtype=float)
    tau = np.asarray(result["profiles"]["tau_xy"], dtype=float)
    occ = phi > 1e-6
    if not occ.any():
        return {"phi_bulk": float("nan"), "phi_surface": float("nan")}
    phi_hi = float(np.percentile(phi[occ], 75))
    dense = np.nonzero(phi > dense_frac * phi_hi)[0]
    lo_i, hi_i = dense[0], dense[-1]
    n = hi_i - lo_i + 1
    lo = lo_i + max(1, int(round(0.2 * n)))
    hi = hi_i - max(1, int(round(0.2 * n)))
    hi = max(hi, lo)
    return {
        "phi_bulk": float(phi[lo:hi + 1].mean()),
        "phi_bulk_window": [float(y[lo]), float(y[hi])],
        "phi_surface": float(phi[dense[-1]]),
        "phi_max": float(phi[occ].max()),
        "vx_surface": float(vx[dense[-1]]),
        "vx_bulk_mean": float(vx[lo:hi + 1].mean()),
        "tau_bulk_mean": float(tau[lo:hi + 1].mean()),
        "y_surface": float(y[dense[-1]] + 0.5),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chute flow (Silbert et al. 2001)")
    ap.add_argument("--theta", type=float, default=22.0)
    ap.add_argument("--height", type=float, default=20.0)
    ap.add_argument("--lx", type=float, default=16.0)
    ap.add_argument("--lz", type=float, default=8.0)
    ap.add_argument("--t-end", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--kn", type=float, default=2.0e4)
    ap.add_argument("--mu", type=float, default=0.50)
    ap.add_argument("--restitution", type=float, default=0.88)
    ap.add_argument("--steady-frac", type=float, default=0.4)
    ap.add_argument("--out", type=str, default="chute.json")
    args = ap.parse_args(argv)
    case = InclinedPlaneCase(theta_deg=args.theta, height=args.height,
                             lx=args.lx, lz=args.lz, dt=args.dt, kn=args.kn,
                             mu=args.mu, restitution=args.restitution)
    res = simulate_chute(case, t_end=args.t_end, steady_frac=args.steady_frac,
                         verbose=True)
    res["bulk"] = density_velocity_profile(res)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"theta": args.theta, "H": args.height,
                      "flowing": res["metrics"]["flowing"],
                      "ke": res["metrics"]["ke_steady"],
                      "p_bagnold": res["fits"].get("p"),
                      "phi_bulk": res["bulk"].get("phi_bulk"),
                      "wall_s": round(res["wall_seconds"], 1)}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
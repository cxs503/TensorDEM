"""Bonded-granular ice-ridge keel as a *foundation*: plane-strain DEM specimen.

Purpose
=======
:mod:`tensordem.ridge_keel_grounding` reduces the keel to a Kerr three-parameter
foundation.  This module produces the *independent* granular measurement of the
same object, so the reduction can be tested instead of assumed:

``confined_compression``  wide rigid lid pushed into the sample -> subgrade modulus
                          ``K`` [N/m^3] (the series modulus of the Kerr model) and
                          the whole pressure-settlement curve.
``simple_shear``          lid sheared -> skeleton shear modulus ``G_s`` [Pa] and
                          the foundation shear parameter ``G_f = G_s h`` [N/m].
``footing_test``          rigid footing of width ``b`` settling by ``w0`` -> total
                          footing resistance ``R`` [N/m], the quantity the Kerr
                          footing closed form predicts.

Model (documented idealisation, no hidden calibration)
======================================================
Plane strain, unit thickness.  The keel skeleton is a **bonded lattice** of
circular blocks on a jittered triangular lattice with solid fraction
``phi = 1 - n``: every pair closer than ``cutoff * (r_i + r_j)`` receives one
freeze bond.  A bond is a two-node elastic element with axial stiffness
``kn = E_ice w_b / l0`` and shear stiffness ``kt = SHEAR_RATIO kn``
(``w_b = 2 sqrt(r_i r_j)``), i.e. a freeze bond modelled as an elastic ice bridge
of Young's modulus ``E_ice`` -- the same idealisation the freeze-bond module uses
for a cemented contact.  Quasi-static loading is dynamic relaxation (velocity
Verlet, viscous damping at the critical damping of the stiffest bond, unit
masses: no inertia physics is claimed, only equilibrium).  Lateral boundaries are
**periodic**, so the specimen is a representative patch of an extended keel and
the footing sees no artificial free edge.

Why a lattice rather than a contact-detecting DEM
-------------------------------------------------
The quantities under test are small-strain elastic (settlements of a few mm in a
2 m keel, strain ~1e-3).  In that regime no contact is created or lost, so a
frozen bond topology is not an approximation of the contact problem -- it *is*
the contact problem.  The gain is that one relaxation step is a handful of vector
operations on the bond arrays, which is what makes the 25k / 50k / 100k particle
refinement study affordable.  Bond breakage, contact creation at large strain and
block rotation are outside the model; they are listed in the artifact
``limitations`` rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable

import numpy as np

try:  # neighbour search, build time only
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

__all__ = [
    "KeelSample",
    "build_keel_sample",
    "bond_forces",
    "relax",
    "confined_compression",
    "simple_shear",
    "footing_test",
]

ICE_YOUNG = 1.0e9            # Pa, freeze-bond (granular ice) modulus
SHEAR_RATIO = 0.25           # kt / kn of a bonded element
PLATE_LAYERS = 2             # lid / footing thickness in lattice spacings
REFERENCE_LENGTH = 1.0      # m, unit out-of-plane thickness of the plane-strain bond


@dataclass
class KeelSample:
    """A bonded plane-strain keel patch (SI, unit thickness)."""

    x: np.ndarray
    r: np.ndarray
    length: float
    height: float
    bi: np.ndarray
    bj: np.ndarray
    l0: np.ndarray
    n0: np.ndarray
    t0: np.ndarray
    kn: np.ndarray
    kt: np.ndarray
    spacing: float
    solid_fraction: float
    rng_seed: int
    meta: dict[str, Any] = field(default_factory=dict)
    x0: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.x0 is None:
            self.x0 = self.x.copy()

    @property
    def n_particles(self) -> int:
        return int(self.x.shape[0])

    @property
    def coordination(self) -> float:
        return float(2 * self.bi.size / self.n_particles)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_particles": self.n_particles, "n_bonds": int(self.bi.size),
            "coordination": self.coordination, "length_m": self.length,
            "height_m": self.height, "spacing_m": self.spacing,
            "mean_radius_m": float(np.mean(self.r)),
            "solid_fraction": self.solid_fraction,
            "kPa_per_bond": float(np.mean(self.kn) / 1e3),
            "seed": self.rng_seed, **self.meta,
        }


def build_keel_sample(n_particles: int, *, height: float = 2.0, cell_ratio: float = 8.0,
                      solid_fraction: float = 0.62, seed: int = 0,
                      jitter: float = 0.12, cutoff: float = 1.06,
                      ice_young: float = ICE_YOUNG,
                      bond_reference_length: float | None = None) -> KeelSample:
    """Jittered triangular bonded lattice filling ``height`` x ``cell_ratio*height``.

    ``n_particles`` fixes the lattice spacing (both directions refine together,
    which is the refinement direction reported in the GCI table).

    ``bond_reference_length`` selects the bond stiffness law:

    * ``None`` (W27 legacy law) ``kn = E w_b / l0`` -- the bond length ``l0`` scales
      with the particle size, so ``kn`` is particle-size independent and the lattice
      subgrade modulus ``k ~ kn / a`` grows like ``1/a``: no particle-converged
      continuum limit exists.
    * ``Lambda`` (spacing-consistent law) ``kn = E w_b / Lambda`` with a fixed length
      ``Lambda`` -- ``w_b ~ a``, hence ``kn ~ a`` and ``k ~ E / Lambda`` independent
      of the particle size (this is what ``REFERENCE_LENGTH`` documents).
    """
    if cKDTree is None:  # pragma: no cover
        raise RuntimeError("scipy.spatial.cKDTree is required to build the bond network")
    if n_particles < 200:
        raise ValueError("n_particles must be >= 200")
    if not (0.2 < solid_fraction < 0.95):
        raise ValueError("solid_fraction must be in (0.2, 0.95)")
    length = cell_ratio * height
    spacing = math.sqrt(length * height / ((math.sqrt(3.0) / 2.0) * n_particles))
    radius = spacing * math.sqrt(solid_fraction * math.sqrt(3.0) / (2.0 * math.pi))
    rng = np.random.default_rng(seed)
    ny = max(4, int(round(height / (spacing * math.sqrt(3.0) / 2.0))))
    dy = height / ny
    dx = dy / (math.sqrt(3.0) / 2.0)
    nx = max(4, int(round(length / dx)))
    pts = []
    for row in range(ny):
        y = (row + 0.5) * dy
        for col in range(nx):
            pts.append(((col + 0.5 * (row % 2)) * dx, y))
    pts = np.array(pts, dtype=float)
    pts += rng.normal(scale=jitter * min(dx, dy), size=pts.shape)
    pts[:, 0] = np.mod(pts[:, 0], length)
    pts[:, 1] = np.clip(pts[:, 1], 0.5 * radius, height - 0.5 * radius)
    n = pts.shape[0]
    radii = np.clip(radius * (1.0 + 0.08 * rng.standard_normal(n)), 0.7 * radius, 1.3 * radius)

    shifted = pts.copy()
    shifted[:, 0] += length
    allp = np.vstack([pts, shifted])
    tree = cKDTree(allp, boxsize=[3.0 * length, 10.0 * height])
    pairs = tree.query_pairs(cutoff * 2.0 * float(radii.max()), output_type="ndarray")
    keep = (pairs[:, 0] < n) & (pairs[:, 1] < n)
    pairs = pairs[keep]
    bi = pairs[:, 0].astype(int)
    bj = pairs[:, 1].astype(int)
    dxv = pts[bj] - pts[bi]
    dxv[:, 0] -= length * np.round(dxv[:, 0] / length)     # minimum image in x
    l0 = np.linalg.norm(dxv, axis=1)
    n0 = dxv / l0[:, None]
    t0 = np.column_stack([-n0[:, 1], n0[:, 0]])
    w_b = 2.0 * np.sqrt(radii[bi] * radii[bj])
    # Bond stiffness = E w_b / l0 (an elastic ice bridge of length l0 and width
    # w_b).  NOTE, measured in the W27 study: the lattice subgrade modulus then
    # scales like kn/a, i.e. it is *not* particle-size invariant (see the artifact
    # `B_bonded_keel_dem` and the limitations list); `REFERENCE_LENGTH` documents
    # the spacing-consistent alternative (kn = E w_b / 1 m) that must be used
    # before the granular block can claim a converged continuum limit.
    if bond_reference_length is None:
        kn = ice_young * w_b / l0
        bond_law = "legacy_bond_length"
    else:
        if not (math.isfinite(bond_reference_length) and bond_reference_length > 0.0):
            raise ValueError("bond_reference_length must be finite and > 0")
        kn = ice_young * w_b / float(bond_reference_length)
        bond_law = "spacing_consistent"
    kt = SHEAR_RATIO * kn
    return KeelSample(x=pts, r=radii, length=length, height=height, bi=bi, bj=bj,
                      l0=l0, n0=n0, t0=t0, kn=kn, kt=kt, spacing=spacing,
                      solid_fraction=solid_fraction, rng_seed=seed,
                      meta={"ice_young_Pa": ice_young, "cutoff": cutoff,
                            "jitter": jitter, "cell_ratio": cell_ratio,
                            "bond_law": bond_law,
                            "bond_reference_length_m": bond_reference_length})


# ---------------------------------------------------------------------------
# bond forces / relaxation
# ---------------------------------------------------------------------------
def bond_forces(s: KeelSample, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(f_i, f_j, fn)`` for every bond (``fn`` positive in tension)."""
    d = x[s.bj] - x[s.bi]
    d[:, 0] -= s.length * np.round(d[:, 0] / s.length)      # minimum image
    l = np.linalg.norm(d, axis=1)
    fn = s.kn * (l - s.l0)                       # tension positive along n0
    ft = s.kt * np.einsum("ij,ij->i", d, s.t0)
    f = fn[:, None] * s.n0 + ft[:, None] * s.t0
    fi = np.zeros_like(x)
    fj = np.zeros_like(x)
    np.add.at(fi, s.bi, f)
    np.add.at(fj, s.bj, -f)
    return fi, fj, fn


def _internal_force(s: KeelSample) -> np.ndarray:
    """Total bond force per particle (vectorised, ``bincount`` accumulation)."""
    n = s.n_particles
    d = s.x[s.bj] - s.x[s.bi]
    d[:, 0] -= s.length * np.round(d[:, 0] / s.length)
    dx, dy = d[:, 0], d[:, 1]
    l = np.sqrt(dx * dx + dy * dy)
    fn = s.kn * (l - s.l0)
    ft = s.kt * (dx * s.t0[:, 0] + dy * s.t0[:, 1])
    fx = fn * s.n0[:, 0] + ft * s.t0[:, 0]
    fy = fn * s.n0[:, 1] + ft * s.t0[:, 1]
    f = np.empty_like(s.x)
    f[:, 0] = (np.bincount(s.bi, fx, minlength=n) - np.bincount(s.bj, fx, minlength=n))
    f[:, 1] = (np.bincount(s.bi, fy, minlength=n) - np.bincount(s.bj, fy, minlength=n))
    return f


def relax(s: KeelSample, steps: int, *, fixed: np.ndarray, tol: float | None = None,
          record_every: int = 0, damping_ratio: float = 0.7) -> dict[str, Any]:
    """Semi-implicit dynamic relaxation with the ``fixed`` particles held.

    Update per step (unit masses): ``v += dt f``; ``v /= 1 + dt 2 xi omega``;
    ``x += dt v``.  The damping factor is implicit, so the scheme is
    unconditionally stable in the damping term; ``dt = 1/omega`` keeps the
    explicit part inside its stability limit (``omega = sqrt(4 k_max)`` for ~4
    bonds per particle).  No inertia physics is claimed -- this is an equilibrium
    solver.
    """
    k_max = float(np.max(s.kn + s.kt))
    omega = math.sqrt(4.0 * k_max)
    dt = 1.0 / omega
    damp = dt * 2.0 * damping_ratio * omega
    tol = (1e-8 * float(np.mean(s.kn)) * s.spacing) if tol is None else float(tol)
    v = np.zeros_like(s.x)
    free = ~fixed
    hist: list[float] = []
    res = float("inf")
    for step in range(steps):
        f = _internal_force(s)
        f[fixed] = 0.0
        v[free] = (v[free] + dt * f[free]) / (1.0 + damp)
        s.x[free] += dt * v[free]
        res = float(np.max(np.abs(f)))
        if record_every and step % record_every == 0:
            hist.append(res)
        if res < tol:
            break
    return {"steps": step + 1, "residual": res, "tol": tol, "dt_s": dt,
            "omega_rad_s": omega, "history": hist, "converged": bool(res < tol)}


def _reaction(s: KeelSample, group: np.ndarray, axis: int = 1) -> float:
    """Total internal force on ``group`` from bonds crossing the group boundary."""
    n = s.n_particles
    d = s.x[s.bj] - s.x[s.bi]
    d[:, 0] -= s.length * np.round(d[:, 0] / s.length)
    dx, dy = d[:, 0], d[:, 1]
    l = np.sqrt(dx * dx + dy * dy)
    fn = s.kn * (l - s.l0)
    ft = s.kt * (dx * s.t0[:, 0] + dy * s.t0[:, 1])
    fx = fn * s.n0[:, 0] + ft * s.t0[:, 0]
    fy = fn * s.n0[:, 1] + ft * s.t0[:, 1]
    comp = fx if axis == 0 else fy
    gi, gj = group[s.bi], group[s.bj]
    cross = gi ^ gj
    acc = (np.bincount(s.bi[gi & cross], comp[gi & cross], minlength=n)
           - np.bincount(s.bj[gj & cross], comp[gj & cross], minlength=n))
    return float(np.sum(acc[group]))


def assemble_stiffness(s: KeelSample) -> "object":
    """Global stiffness matrix (2 DOF per particle, ``csr``)."""
    from scipy.sparse import coo_matrix

    n = s.n_particles
    nb = s.bi.size
    a = np.zeros((nb, 2, 2))
    a[:, 0, 0] = s.kn * s.n0[:, 0] ** 2 + s.kt * s.t0[:, 0] ** 2
    a[:, 0, 1] = s.kn * s.n0[:, 0] * s.n0[:, 1] + s.kt * s.t0[:, 0] * s.t0[:, 1]
    a[:, 1, 0] = a[:, 0, 1]
    a[:, 1, 1] = s.kn * s.n0[:, 1] ** 2 + s.kt * s.t0[:, 1] ** 2
    rows, cols, vals = [], [], []
    for (ni, nj, sign) in ((s.bi, s.bj, 1.0), (s.bi, s.bi, -1.0),
                           (s.bj, s.bj, 1.0), (s.bj, s.bi, -1.0)):
        for di in (0, 1):
            for dj in (0, 1):
                rows.append(2 * ni + di)
                cols.append(2 * nj + dj)
                vals.append(sign * a[:, di, dj])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    k = coo_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
    k.sum_duplicates()
    return k


def solve_static(s: KeelSample, *, prescribed: np.ndarray,
                 values: np.ndarray, extra_fixed: np.ndarray | None = None
                 ) -> dict[str, Any]:
    """Solve ``K u = 0`` with Dirichlet data on the ``prescribed`` particles.

    Exact linear-elastic equilibrium (no relaxation): ``prescribed`` is a boolean
    mask whose ``values[mask]`` give the imposed ``(ux, uy)``.  ``extra_fixed``
    (e.g. the rigid base) is clamped to zero.  Returns the displacement field and
    the total reaction on the prescribed set.
    """
    from scipy.sparse.linalg import spsolve

    k = assemble_stiffness(s)
    n = s.n_particles
    deg = np.bincount(np.concatenate([s.bi, s.bj]), minlength=n)
    fixed = np.zeros(n, dtype=bool)
    fixed |= prescribed
    fixed |= deg == 0          # unbonded particle: no stiffness, clamp it
    if extra_fixed is not None:
        fixed |= extra_fixed
    u = np.zeros((n, 2))
    u[prescribed] = values[prescribed]
    free_dof = np.ones(2 * n, dtype=bool)
    free_dof[np.repeat(fixed, 2)] = False
    fixed_dof = ~free_dof
    if not np.any(free_dof):
        return {"u": u, "reaction": np.zeros(2)}
    k_ff = k[free_dof][:, free_dof]
    k_fp = k[free_dof][:, fixed_dof]
    rhs = -(k_fp @ u.reshape(-1)[fixed_dof])
    sol = spsolve(k_ff.tocsc(), rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("static solve produced non-finite displacements")
    u.reshape(-1)[free_dof] = sol
    full = k @ u.reshape(-1)
    reaction = full.reshape(-1, 2)[prescribed].sum(axis=0)
    return {"u": u, "reaction": reaction, "n_free_dof": int(free_dof.sum()),
            "residual_inf": float(np.max(np.abs(full[free_dof])))}


def _group_masks(s: KeelSample, *, plate_half_width: float | None = None,
                 plate_depth: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    depth = plate_depth if plate_depth is not None else PLATE_LAYERS * s.spacing
    base = s.x0[:, 1] <= depth
    top = s.x0[:, 1] >= s.height - depth
    if plate_half_width is not None:
        xc = 0.5 * s.length
        top &= np.abs(s.x0[:, 0] - xc) <= plate_half_width
    return base, top


def _push(s: KeelSample, *, plate: np.ndarray, fixed_extra: np.ndarray,
          direction: np.ndarray, total: float, increments: int,
          steps_per_increment: int) -> dict[str, Any]:
    """Incremental quasi-static displacement of ``plate`` along ``direction``."""
    fixed = fixed_extra.copy()
    fixed[plate] = True
    base_pos = s.x0[plate].copy()
    hist = []
    res_hist = []
    for inc in range(1, increments + 1):
        tgt = base_pos + direction[None, :] * (total * inc / increments)
        s.x[plate] = tgt
        out = relax(s, steps_per_increment, fixed=fixed)
        res_hist.append(out["residual"])
        hist.append(float(out["residual"]))
    return {"increments": increments, "steps_per_increment": steps_per_increment,
            "residual_final": res_hist[-1], "residual_history": hist}


def confined_compression(s: KeelSample, *, w_max: float | None = None,
                         increments: int = 10, steps_per_increment: int = 300,
                         solver: str = "relaxation") -> dict[str, Any]:
    """Wide lid pushed down: pressure-settlement curve and subgrade modulus."""
    w_max = w_max if w_max is not None else 4.0e-3 * s.height
    base, top = _group_masks(s)
    ws = w_max * np.arange(1, increments + 1) / increments
    pressures = []
    base_pos = s.x0[top].copy()
    fixed = base.copy()
    fixed[top] = True
    for w in ws:
        if solver == "static":
            vals = np.zeros_like(s.x)
            vals[top, 1] = -w
            out = solve_static(s, prescribed=top, values=vals, extra_fixed=base)
            pressures.append(-out["reaction"][1] / s.length)
        else:
            s.x[top] = base_pos + np.array([0.0, -w])
            relax(s, steps_per_increment, fixed=fixed)
            pressures.append(_reaction(s, top, axis=1) / s.length)
    pressures = np.array(pressures)
    return {"test": "confined_compression", "settlements_m": ws.tolist(),
            "pressures_Pa": pressures.tolist(),
            "k_subgrade_small_strain_N_per_m3": float(pressures[0] / ws[0]),
            "k_subgrade_N_per_m3": float(pressures[-1] / ws[-1]), "solver": solver,
            "sample": s.as_dict()}


def simple_shear(s: KeelSample, *, gamma_max: float = 4.0e-3, increments: int = 8,
                 steps_per_increment: int = 300, solver: str = "relaxation"
                 ) -> dict[str, Any]:
    """Lid sheared horizontally: skeleton shear modulus ``G_s`` and ``G_f``."""
    base, top = _group_masks(s)
    gammas, taus = [], []
    base_pos = s.x0[top].copy()
    fixed = base.copy()
    fixed[top] = True
    for inc in range(1, increments + 1):
        disp = gamma_max * s.height * inc / increments
        if solver == "static":
            vals = np.zeros_like(s.x)
            vals[top, 0] = disp
            out = solve_static(s, prescribed=top, values=vals, extra_fixed=base)
            taus.append(abs(out["reaction"][0]) / s.length)
        else:
            s.x[top] = base_pos + np.array([disp, 0.0])
            relax(s, steps_per_increment, fixed=fixed)
            taus.append(abs(_reaction(s, top, axis=0)) / s.length)
        gammas.append(disp / s.height)
    gammas = np.array(gammas)
    taus = np.array(taus)
    g_s = float(taus[0] / gammas[0])
    return {"test": "simple_shear", "gammas": gammas.tolist(), "taus_Pa": taus.tolist(),
            "G_skeleton_Pa": g_s,
            "G_skeleton_fit_Pa": float(np.polyfit(gammas, taus, 1)[0]),
            "G_foundation_shear_N_per_m": g_s * s.height, "solver": solver,
            "sample": s.as_dict()}


def footing_test(s: KeelSample, *, footing_width: float, w0: float | None = None,
                 increments: int = 10, steps_per_increment: int = 300,
                 solver: str = "relaxation") -> dict[str, Any]:
    """Rigid footing of width ``footing_width`` settling by ``w0``: resistance curve.

    Quasi-static loading by dynamic relaxation (default) or, with
    ``solver="static"``, by an exact sparse linear solve.  The static path is
    available but *not* used for the W27 numbers: the released bond model is a
    central/shear spring without rotational DOF, whose global operator has very
    soft modes for the jittered lattice (coordination ~4.8), so the direct solve is
    ill-conditioned.  See the artifact limitations.
    """
    w0 = w0 if w0 is not None else 4.0e-3 * s.height
    base, top = _group_masks(s, plate_half_width=0.5 * footing_width)
    ws = w0 * np.arange(1, increments + 1) / increments
    forces = []
    base_pos = s.x0[top].copy()
    fixed = base.copy()
    fixed[top] = True
    for w in ws:
        if solver == "static":
            vals = np.zeros_like(s.x)
            vals[top, 1] = -w
            out = solve_static(s, prescribed=top, values=vals, extra_fixed=base)
            forces.append(-out["reaction"][1])
        else:
            s.x[top] = base_pos + np.array([0.0, -w])
            relax(s, steps_per_increment, fixed=fixed)
            forces.append(_reaction(s, top, axis=1))
    forces = np.array(forces)
    return {"test": "footing", "footing_width_m": footing_width,
            "settlements_m": ws.tolist(), "resistance_N_per_m": forces.tolist(),
            "secant_stiffness_N_per_m2": float(forces[0] / ws[0]),
            "n_footing_particles": int(np.count_nonzero(top)),
            "solver": solver, "sample": s.as_dict()}

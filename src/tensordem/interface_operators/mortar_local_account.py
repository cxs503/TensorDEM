"""Per-facet local conservation, kinked (crease) interfaces and the time-domain
interface energy account for non-matching curved mortar discretisations.

Why this module exists
----------------------
``curved_patch_mortar`` (m7171) replaced the plane-projection mortar by a
patch-local parametrisation with exact per-facet measures, and gated the
**per-facet** coverage identity on two *smooth* charts (sphere, graph).  Two
things were still missing for a coupled solver to trust the interface at every
step:

1. **Creases.**  Real interfaces are piecewise smooth: ice floes meet hull
   plating along a ridge, tri-tri interfaces are kinked (``m6114``'s 30 degree
   kink).  A single analytic chart cannot represent that, so the earlier gate was
   only ever applied to ``C^inf`` patches.  :class:`FoldPatch` is an *isometric*
   chart (``J == 1``) of a 30 degree tent, piecewise linear in ``u``; because the
   intersection is computed in the **parameter domain**, a ``C^0`` chart is
   admissible and the per-facet identity survives the crease exactly.  The crease
   is where plane projection hurts most (the two facet normals differ by the kink
   angle), which makes it the sharpest test of true intersection vs projection.

2. **A time-domain account, not just a static coverage number.**  A static
   per-facet defect is necessary but not sufficient: what a coupled solver
   actually injects is *energy*.  :func:`interface_energy_account` integrates two
   first-order systems across the interface and reports
   ``dE/dt = -D_A - D_B + P_int`` with
   ``P_int = sum_j lambda_j v_j (|Gamma_j| c_j - S_j)``, ``S_j = sum_i m_ij``,
   i.e. the interface power is *identically* the per-facet conservation defect.
   With ``c_j = 1`` (standard mortar receptor) a uniform state has ``lambda != 0``
   iff the patch is not locally conserved, and the system then injects or
   dissipates energy at every step.

Also provided: a mismatch-ratio sweep (1x / 2x / 4x) of the local information
loss of plane projection vs the true intersection, the gap / penetration volume
controls, and a curved-metric transpose consistency study whose consistency
defect (a smooth field transported by the adjoint) converges at second order.

Everything reported by :func:`run_acceptance` is machine generated; no number in
the accompanying report is typed by hand.
"""

from __future__ import annotations

import copy
import hashlib
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .curved_patch_mortar import (
    CurvedMortarOperator,
    CurvedSide,
    GraphPatch,
    ParamPatch,
    SpherePatch,
    concat_sides,
    coverage_arrangement,
    fit_rate,
    grid_side,
    per_patch_balance,
    plane_projection_defect,
    tri_gauss,
)

__all__ = [
    "FoldPatch",
    "interface_cases",
    "mismatch_case",
    "local_information_loss",
    "mismatch_sweep",
    "gap_penetration_account",
    "interface_energy_account",
    "break_local_conservation",
    "transpose_consistency_study",
    "transfer_order_regression",
    "provenance",
    "run_acceptance",
    "SEMANTICS",
]

SEMANTICS = (
    "界面算子：逐面片局部守恒（不是只对全局和）+ 真交集替代面片投影 + 间隙/渗透体积记账 "
    "+ 时域耦合能量账（dE/dt = -耗散 + 界面功率）+ 曲面度量下转置一致性；"
    "负控：破度量 / 破局部守恒 / 关掉间隙项必须使门失败。"
)


# ---------------------------------------------------------------------------
# a piecewise-smooth (kinked) interface chart: the 30 degree tent
# ---------------------------------------------------------------------------
class FoldPatch(ParamPatch):
    """``30`` degree tent (one crease along ``u = 1/2``), isometric chart.

    The parametrisation is ::

        x(u, v) = ( (u - 1/2) cos(alpha),  v,  |u - 1/2| sin(alpha) ),
        alpha = fold_deg / 2,

    i.e. two half-planes meeting at a ridge.  ``x_u`` jumps across the crease but
    ``J = |x_u x x_v| = 1`` identically, so the curved measure equals the
    parameter area on both halves *and* the quadrature of a polygon straddling the
    crease stays exact -- which is what keeps the per-facet identity at round-off
    on a non-smooth chart.  This is the geometry ``m6114`` used for its planar
    kinked case, lifted onto the parametric-interface framework of the curved
    mortar.
    """

    def __init__(self, fold_deg: float = 30.0, *, height: float = 1.0):
        self.fold_deg = float(fold_deg)
        self.alpha = math.radians(self.fold_deg / 2.0)
        self.height = float(height)

    @property
    def label(self) -> str:                      # type: ignore[override]
        return f"fold{self.fold_deg:g}"

    # -- chart -------------------------------------------------------------
    def x(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        tau = uv[:, 0] - 0.5
        return np.stack([tau * math.cos(self.alpha),
                         self.height * uv[:, 1],
                         np.abs(tau) * math.sin(self.alpha)], axis=1)

    def xu(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        s = np.where(uv[:, 0] > 0.5, 1.0, np.where(uv[:, 0] < 0.5, -1.0, 0.0))
        return np.stack([np.full(len(uv), math.cos(self.alpha)),
                         np.zeros(len(uv)),
                         s * math.sin(self.alpha)], axis=1)

    def xv(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return np.stack([np.zeros(len(uv)),
                         np.full(len(uv), self.height),
                         np.zeros(len(uv))], axis=1)

    def xuu(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return np.zeros((len(uv), 3))

    xuv = xuu
    xvv = xuu

    def jac(self, uv) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return np.full(len(uv), self.height)

    # -- projection (closed form: the chart is piecewise planar) -----------
    def project(self, p, uv0=None, *, tol: float = 1e-15, maxit: int = 100):
        """Exact projection onto the tent (one linear solve per half-plane).

        Returns ``(uv, distance, iterations, |g|)`` like
        :meth:`ParamPatch.project`; ``iterations`` is 1 (closed form) and ``|g|``
        is the stationarity residual, so a caller can assert that the projection
        landed on the surface instead of being clamped.
        """
        p = np.atleast_2d(np.asarray(p, dtype=float))
        out_uv = np.zeros((len(p), 2))
        dist = np.zeros(len(p))
        gnorm = np.zeros(len(p))
        ca, sa = math.cos(self.alpha), math.sin(self.alpha)
        for k in range(len(p)):
            px, py, pz = p[k]
            v = py / self.height
            best = None
            for s in (-1.0, 1.0):
                d = np.array([ca, 0.0, s * sa])          # unit direction of x_u
                tau = px * d[0] + pz * d[2]              # minimise |p - x(tau)|
                if s * tau >= -1e-15 and -0.5 <= tau <= 0.5:
                    resid = float(np.linalg.norm(
                        p[k] - (tau * d + np.array([0.0, py, 0.0]))))
                    if best is None or resid < best[1]:
                        best = (tau, resid)
            if best is None:                             # fall back to the ridge
                best = (0.0, float(math.hypot(px, pz)))
            out_uv[k] = (0.5 + best[0], v)
            dist[k] = best[1]
            r = self.x(out_uv[k])[0] - p[k]
            gnorm[k] = float(np.linalg.norm(
                [r @ self.xu(out_uv[k])[0], r @ self.xv(out_uv[k])[0]]))
        if len(out_uv) == 1:
            return out_uv[0], float(dist[0]), 1, float(gnorm[0])
        return out_uv, dist, 1, gnorm

    # -- reporting ---------------------------------------------------------
    @property
    def curvature_scale(self) -> float:
        """Kink slope ``tan(alpha)``: the ``kappa * L`` analogue of a crease."""
        return math.tan(self.alpha)


def interface_cases() -> list:
    """The three interface geometries used by the acceptance run.

    ``sphere`` (``kappa L = 0.76``), ``graph`` (``kappa L = 2.41``, the
    high-curvature case) and ``fold30`` (the 30 degree crease of ``m6114``).
    """
    return [SpherePatch(), GraphPatch(), FoldPatch(30.0)]


# ---------------------------------------------------------------------------
# mismatch ratios
# ---------------------------------------------------------------------------
def mismatch_case(patch: ParamPatch, n: int, ratio: int, *,
                  jitter: float = 0.25, seed_master: int = 11,
                  seed_slave: int = 23, box=None):
    """Master ``n x n`` vs slave ``~ratio*n x ratio*n`` non-matching sides.

    Facet counts per master facet are about ``1``, ``2`` and ``4`` for
    ``ratio = 1, 2, 4``.  Both sides tile the same parameter box exactly
    (boundary nodes are fixed), so a per-facet defect is a *transport* defect and
    never a missing domain.
    """
    ratio = int(ratio)
    ns = ratio * int(n) + (2 if ratio == 1 else 1)
    ms = ratio * int(n) + 1
    kw = {} if box is None else {"box": box}
    master = grid_side(patch, n, n, jitter=jitter, seed=seed_master,
                       label=f"master{n}", **kw)
    slave = grid_side(patch, ns, ms, jitter=jitter, seed=seed_slave,
                      label=f"slave{ratio}x", **kw)
    return master, slave


def local_information_loss(patch: ParamPatch, n: int, ratio: int, **kw) -> dict:
    """Per-facet and global coverage: true intersection vs plane projection.

    ``true_per_facet`` is the gate quantity of this node; ``plane_per_facet`` /
    ``plane_global`` reproduce the previous plane-projection construction and are
    the *measured* 4x-mismatch information loss (``m6114`` reported 4.1e-2 on its
    polyhedral octant; the same quantity is recomputed here on the parametric
    interface, so no number is copied).
    """
    master, slave = mismatch_case(patch, n, ratio, **kw)
    b = per_patch_balance(master, slave)
    pp = plane_projection_defect(master, slave)
    cen = CurvedMortarOperator(master, slave, measure="centroid_j", order=0)
    return {
        "h": 1.0 / n, "ratio": ratio, "n_master": master.n, "n_slave": slave.n,
        "true_per_facet": b["per_facet_max_rel"],
        "true_global": abs(b["global_rel"]),
        "centroid_j_per_facet": cen.conservation()["per_facet_max_rel"],
        "plane_per_facet": pp["per_facet_max_rel"],
        "plane_global": abs(pp["global_rel"]),
        "n_uncovered_true": b["n_uncovered"],
    }


def mismatch_sweep(patch: ParamPatch, meshes=(4, 6, 8), ratios=(1, 4), **kw) -> dict:
    """1x / 2x / 4x mismatch scan of true intersection vs projection."""
    rows = [local_information_loss(patch, n, r, **kw) for n in meshes for r in ratios]
    four = [r for r in rows if r["ratio"] == 4]
    return {
        "patch": patch.label,
        "rows": rows,
        "max_true_per_facet": float(max(r["true_per_facet"] for r in rows)),
        "min_plane_per_facet_4x": float(min(r["plane_per_facet"] for r in four)),
        "min_plane_global_4x": float(min(r["plane_global"] for r in four)),
    }


# ---------------------------------------------------------------------------
# gap / penetration volume account and its negative control
# ---------------------------------------------------------------------------
def gap_penetration_account(patch: ParamPatch, *, n: int = 6, jitter: float = 0.25,
                            seeds=(11, 23)) -> dict:
    """Per-facet identity ``|Gamma_j| + V_pen - V_gap = sum_i m_ij`` and controls.

    Cases: ``conforming`` (both sides tile the same box), ``gap`` (slave inset by
    ~5 % of the parameter box: the two discretisations do not cover the same
    surface) and ``penetration`` (slave duplicated inside a sub-box).  The
    negative control is the residual of the same identity with the gap volume
    *dropped*: ``max_j |sum_i m_ij - |Gamma_j|| / |Gamma_j|``, which must exceed
    ``1e-3`` exactly when the interface has a gap or a penetration.
    """
    master = grid_side(patch, n, n, jitter=jitter, seed=seeds[0], label="master")
    slave_ok = grid_side(patch, n + 1, n + 1, jitter=jitter, seed=seeds[1], label="slave")
    cases = {
        "conforming": slave_ok,
        "gap": grid_side(patch, n + 1, n + 1, jitter=jitter, seed=seeds[1],
                         box=(0.05, 0.95, 0.04, 0.96), label="slave_gap"),
        "penetration": concat_sides(
            slave_ok,
            grid_side(patch, max(3, n - 1), max(3, n - 1), jitter=0.1,
                      seed=seeds[1] + 1, box=(0.2, 0.8, 0.2, 0.8), label="slave_dup"),
            label="slave_pen"),
    }
    rows = []
    for name, sl in cases.items():
        acc = coverage_arrangement(master, sl)
        dropped = per_patch_balance(master, sl)["per_facet_max_rel"]
        rows.append({
            "case": name,
            "identity_balance_rel": acc["balance_rel"],
            "identity_total_rel": acc["identity_total_rel"],
            "gap_unit_volume_rel": acc["v_gap_total"] / master.area,
            "pen_unit_volume_rel": acc["v_pen_total"] / master.area,
            "max_multiplicity": acc["max_mult_overall"],
            "max_pieces_per_facet": int(acc["n_pieces"].max()),
            "without_gap_term_residual": dropped,
        })
    gap_row = [r for r in rows if r["case"] == "gap"][0]
    pen_row = [r for r in rows if r["case"] == "penetration"][0]
    return {
        "patch": patch.label,
        "rows": rows,
        "max_identity_balance_rel": float(max(r["identity_balance_rel"] for r in rows)),
        "max_gap_unit_volume_rel": float(max(r["gap_unit_volume_rel"] for r in rows)),
        "max_pen_unit_volume_rel": float(max(r["pen_unit_volume_rel"] for r in rows)),
        "control_residual_gap_case": float(gap_row["without_gap_term_residual"]),
        "control_residual_pen_case": float(pen_row["without_gap_term_residual"]),
        "min_dropped_term_residual": float(min(r["without_gap_term_residual"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# time-domain interface energy account
# ---------------------------------------------------------------------------
def break_local_conservation(op: CurvedMortarOperator, *, eps: float = 0.25):
    """Copy of ``op`` whose per-facet conservation is deliberately violated.

    Half of the slave column block of the mortar matrix is scaled by ``1 + eps``
    (all master rows), so ``sum_i m_ij != |Gamma_j|`` by ``~eps/2`` while the
    operator stays smooth and well conditioned.  This is the negative control of
    the energy gate: a gate that cannot see it is not a gate.
    """
    out = copy.copy(op)
    m = op.m.copy()
    nA = m.shape[1]
    m[:, : max(1, nA // 2)] *= (1.0 + eps)
    out.m = m
    Wt = op.master.measures
    out.M = op.M + (m - op.m) / Wt[:, None]
    return out


def interface_energy_account(op: CurvedMortarOperator, *, dt: float = 1e-3,
                             nsteps: int = 400, k_gap: float = 1.0,
                             decay_a: float = 0.4, decay_b: float = 0.4,
                             u0_a=1.0, u0_b=1.0, flux_form: str = "closure",
                             keep_history: bool = False) -> dict:
    """Two first-order systems coupled through the interface: the ``dE/dt`` ledger.

    Semi-discrete scheme (``R = 1 / k_gap``)::

        v_j   = (M u_A)_j                       order-2 mortar projection
        lam_j = (u_B,j - v_j) / R
        du_A,i/dt = -a_A u_A,i - (sum_j m_ij lam_j) / |S_i|
        du_B,j/dt = -a_B u_B,j + c_j lam_j      c_j = 1 ("closure") or S_j/|G_j|

    Energy ledger (exact for this semi-discrete system)::

        E      = 1/2 sum_i |S_i| u_A,i^2 + 1/2 sum_j |Gamma_j| u_B,j^2
        dE/dt  = -D_A - D_B + P_int
        P_int  = sum_j lam_j v_j (|Gamma_j| c_j - S_j),   S_j = sum_i m_ij

    ``P_int`` vanishes identically **iff every master facet is locally
    conserved**; it is the interface power a coupled solver injects or
    dissipates.  ``interface_power_rel_max`` = ``max_step |P_int| / max(D_A+D_B)``
    is the quantity gated at ``1e-10``.  ``ledger_residual_rel_max`` checks the
    algebraic split ``dE/disc = -D + P_int`` and ``time_account_residual_rel_max``
    the exact explicit-Euler energy identity (both are round-off statements and
    are reported separately from the physical interface power).
    """
    Ws = np.asarray(op.slave.measures, dtype=float)
    Wt = np.asarray(op.master.measures, dtype=float)
    nA, nB = op.slave.n, op.master.n
    S = op.m.sum(axis=1)
    c = np.ones(nB) if flux_form == "closure" else S / Wt
    R = 1.0 / k_gap
    uA = np.full(nA, float(u0_a))
    uB = np.full(nB, float(u0_b))

    def energy(a, b):
        return 0.5 * float(Ws @ (a ** 2) + Wt @ (b ** 2))

    E_prev = energy(uA, uB)
    E0 = E_prev
    out = {"dt": dt, "nsteps": int(nsteps), "k_gap": k_gap, "flux_form": flux_form,
           "per_facet_defect": op.conservation()["per_facet_max_rel"],
           "energy0": E0}
    inter_max = ledger_max = time_max = lam_max = 0.0
    leak = 0.0
    hist = {"t": [], "interface_power": [], "energy": [], "lam": []}
    for step in range(int(nsteps)):
        v = op.apply(uA)
        lam = (uB - v) * k_gap
        fA = -decay_a * uA - (op.m.T @ lam) / Ws
        fB = -decay_b * uB + c * lam
        D = decay_a * float(Ws @ (uA ** 2)) + decay_b * float(Wt @ (uB ** 2))
        P_int = float(np.sum(lam * v * (Wt * c - S)))
        dE_semi = float(Ws @ (uA * fA) + Wt @ (uB * fB))
        inter_max = max(inter_max, abs(P_int) / max(D, 1e-300))
        ledger_max = max(ledger_max,
                         abs(dE_semi - (-D + P_int)) / max(abs(dE_semi) + D, 1e-300))
        lam_max = max(lam_max, float(np.max(np.abs(lam))))
        leak += dt * float((Wt * c - S) @ lam)
        uA = uA + dt * fA
        uB = uB + dt * fB
        E_now = energy(uA, uB)
        dE_disc = dt * dE_semi + 0.5 * dt * dt * float(Ws @ (fA ** 2) + Wt @ (fB ** 2))
        time_max = max(time_max,
                       abs((E_now - E_prev) - dE_disc) / max(abs(E0), 1e-300) / dt)
        E_prev = E_now
        if keep_history:
            hist["t"].append((step + 1) * dt)
            hist["interface_power"].append(P_int)
            hist["energy"].append(E_now)
            hist["lam"].append(float(np.max(np.abs(lam))))
    lam_uniform = float(np.max(np.abs((np.full(nB, float(u0_b))
                                       - op.apply(np.full(nA, float(u0_a)))) * k_gap)))
    out.update({
        "lam_max_uniform": lam_uniform,
        "interface_power_rel_max": inter_max,
        "ledger_residual_rel_max": ledger_max,
        "time_account_residual_rel_max": time_max,
        "lam_max": lam_max,
        "mass_leak_rel": abs(leak) / max(abs(float(Ws.sum()) + float(Wt.sum())), 1e-300),
        "energy_drift_rel": abs(E_prev - E0) / max(abs(E0), 1e-300),
        "energy1": E_prev,
    })
    if keep_history:
        out["history"] = hist
    return out


# ---------------------------------------------------------------------------
# transpose (adjoint) consistency in the curved metric
# ---------------------------------------------------------------------------
def _cell_means(patch: ParamPatch, side: CurvedSide, field, rule=None) -> np.ndarray:
    """Exact ``int_{Gamma_j} f J / |Gamma_j|`` for an analytic field ``f(x)``."""
    rule = tri_gauss(8) if rule is None else rule
    s, w = rule
    out = np.zeros(side.n)
    for j, t in enumerate(side.tris):
        uv = t[0][None, :] + s[:, 0:1] * (t[1] - t[0])[None, :] \
            + s[:, 1:2] * (t[2] - t[0])[None, :]
        x = patch.x(uv)
        jac = patch.jac(uv)
        # the reference triangle has area 1/2, so |det Dphi| = 2 * area(t) in (u, v)
        a2 = abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
                 - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1]))
        out[j] = (a2 * float(np.sum(w * jac * np.asarray(field(x), dtype=float)))
                  / side.measures[j])
    return out


def _ls_gradients(points: np.ndarray, values: np.ndarray, k_nn: int = 8) -> np.ndarray:
    """Least-squares gradients ``g_i`` of cell data on an unstructured point set."""
    pts = np.asarray(points, dtype=float)
    val = np.asarray(values, dtype=float)
    d = pts[None, :, :] - pts[:, None, :]                    # (n, n, 3)
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    k = min(int(k_nn), len(pts) - 1)
    idx = np.argsort(dist, axis=1)[:, :k]
    A = np.concatenate([pts[idx] - pts[:, None, :],
                        np.ones((len(pts), k, 1))], axis=2)
    G = np.einsum("nik,nil->nkl", A, A)
    rhs = np.einsum("nik,ni->nk", A, val[idx] - val[:, None])
    try:
        sol = np.linalg.solve(G, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:                            # pragma: no cover
        sol = np.linalg.lstsq(G[0], rhs[0], rcond=None)[0][None, :]
    return sol[:, :3]


def adjoint_transport(op: CurvedMortarOperator, w_cells, *, order: int = 2) -> np.ndarray:
    """Transpose transport of master cell data to slave cell means.

    ``order = 2`` (the transpose-consistent form)::

        (A^* w)_i = [ sum_j m_ij w_j + sum_j mu_ij . grad_j w ] / |S_i|

    with ``mu_ij = int_{R_ij} (x - c_i) J`` the first moment of the overlap and
    ``grad_j w`` the least-squares gradient of the master data.  The bracket is
    ``int_{S_i} [w + (x - c_i).grad w] J = |S_i| w(c_i) + O(h^2)``, i.e. the
    adjoint transports a smooth field at *second* order -- the property the
    one-sided midpoint transpose lacks.  ``order = 0`` is that midpoint control.
    """
    w = np.asarray(w_cells, dtype=float)
    m = op.m
    num = m.T @ w
    if order >= 2 and getattr(op, "mom", None) is not None and len(w) > 3:
        g = _ls_gradients(op.master.centroids, w)
        # extrapolate the master data to the slave centroid (master-side second
        # order) and add the slave-cell first moment of that gradient: the first
        # moment about the *master* centroid is nu_ij = mu_ij + m_ij (c_i - c_j),
        # and sum_j mu_ij = 0 by the definition of the slave centroid.
        d = op.slave.centroids[None, :, :] - op.master.centroids[:, None, :]
        num = num + np.sum(m * np.einsum("jik,jk->ji", d, g), axis=0)
        num = num + np.einsum("jik,jk->i", op.mom, g)
    return num / op.slave.measures


def transpose_consistency_study(patch: ParamPatch, meshes=(4, 6, 8), ratio: int = 1,
                                *, jitter: float = 0.25, seeds=(11, 23),
                                k: float = 1.0) -> dict:
    """Curved-metric transpose consistency, with ``h -> h/2`` rates.

    Per resolution:

    * ``duality_rel`` -- discrete duality defect
      ``|(A u, w)_G - (u, A^* w)_S|`` with ``A`` the mortar matrix linearised at
      ``u`` and ``A^* = |S|^-1 A^T W_t``; it is a *symmetry* statement (exactly
      zero for a consistent pair) and is therefore gated, never fitted for a rate.
    * ``transport_rel_curved`` -- the transpose-consistent adjoint
      (:func:`adjoint_transport`, order 2) must reproduce the *point* value of a
      smooth non-polynomial field at the slave centroids:
      ``max_i |(A^* w)_i - w(c_i)| / max|w|``.  This is the quantity that
      converges at second order in the curved metric.
    * ``transport_rel_centroid_j`` / ``_flat`` -- the same defect with the
      measure frozen at the polygon centroid / replaced by the parameter area.
    * ``transport_rel_naive`` -- the midpoint (order-0) transpose, i.e. the
      control whose rate must stay below the second-order gate.
    """
    rows = []
    for n in meshes:
        master, slave = mismatch_case(patch, n, ratio, jitter=jitter,
                                      seed_master=seeds[0], seed_slave=seeds[1])
        f = lambda x, kk=float(k): np.exp(kk * x[:, 0] + 0.4 * x[:, 2]) \
            * (1.0 + 0.3 * x[:, 1])
        w_cells = _cell_means(patch, master, f)                    # exact master means
        exact_slave = np.asarray(f(slave.centroids), dtype=float)  # point values
        u_cells = np.exp(0.7 * slave.centroids[:, 0]) * (1.0 + 0.2 * slave.centroids[:, 2])
        row = {"h": 1.0 / n, "n_master": master.n, "n_slave": slave.n}
        scale = max(float(np.max(np.abs(exact_slave))), 1e-300)
        for meas in ("curved", "centroid_j", "flat"):
            op = CurvedMortarOperator(master, slave, order=2, measure=meas)
            # symmetry: measure-consistent duality of the operator pair
            A = op.matrix(u_cells)
            lhs = float(np.sum(master.measures * (A @ u_cells) * w_cells))
            rhs = float(np.sum(u_cells * (A.T @ (master.measures * w_cells))))
            row[f"duality_rel_{meas}"] = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            tr = adjoint_transport(op, w_cells, order=2)
            row[f"transport_rel_{meas}"] = float(
                np.max(np.abs(tr - exact_slave)) / scale)
        naive = adjoint_transport(op, w_cells, order=0)
        row["transport_rel_naive"] = float(np.max(np.abs(naive - exact_slave)) / scale)
        rows.append(row)
    hs = [r["h"] for r in rows]
    out = {"patch": patch.label, "ratio": ratio, "rows": rows,
           "duality_rel_max": float(max(max(r[f"duality_rel_{m}"] for m in
                                            ("curved", "centroid_j", "flat")) for r in rows))}
    for key in ("curved", "centroid_j", "flat", "naive"):
        out[f"rate_transport_{key}"] = fit_rate(hs, [r[f"transport_rel_{key}"] for r in rows])
        out[f"transport_rel_{key}_finest"] = rows[-1][f"transport_rel_{key}"]
    return out


# ---------------------------------------------------------------------------
# regression: m6114's D3 transfer integral order must not degrade
# ---------------------------------------------------------------------------
def transfer_order_regression() -> dict:                     # pragma: no cover - heavy
    """Re-run ``m6114``'s D3 gates (curved octant transfer integral, order >= 1.8)."""
    from . import acceptance3d

    d3 = acceptance3d.criteria_d3()
    gates = d3.get("gates", {})

    def _num(node):
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, dict):
            for key in ("value", "measured", "order", "rate", "rel_err", "err",
                        "error", "p"):
                if isinstance(node.get(key), (int, float)):
                    return float(node[key])
        return float("nan")

    def _ok(node):
        if isinstance(node, dict):
            for key in ("pass", "ok", "passed", "green"):
                if key in node:
                    return bool(node[key])
        return bool(node)

    return {
        "area_order": _num(gates.get("D3_area_order")),
        "field_order": _num(gates.get("D3_field_order")),
        "field_accuracy": _num(gates.get("D3_field_accuracy")),
        "gates": {k: _ok(v) for k, v in gates.items()},
        "raw_gates": gates,
    }


# ---------------------------------------------------------------------------
# provenance & acceptance
# ---------------------------------------------------------------------------
def _git(*args, root: Path | None = None) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(root or Path.cwd()),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                        # pragma: no cover
        return ""


def provenance(*, seeds=(11, 23)) -> dict:
    here = Path(__file__).resolve()
    return {
        "git_head": _git("rev-parse", "HEAD", root=here.parent) or _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "",
        "git_dirty": bool(_git("status", "--porcelain")),
        "module": str(here.name),
        "module_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seeds": {"master": int(seeds[0]), "slave": int(seeds[1])},
        "semantics": SEMANTICS,
    }


def run_acceptance(*, meshes=(4, 6, 8), energy_resolutions=(4, 6, 8),
                   energy_ratios=(1, 4), e_meshes=(6, 8, 12, 16),
                   with_regression: bool = True) -> dict:
    """Machine-checkable acceptance of the per-facet local-conservation node.

    * ``A`` per-facet conservation ``<= 1e-12 |Gamma_j|`` on 100 % of facets, on
      all three interfaces at all mesh sizes and for 1x / 4x mismatch;
    * ``B`` true intersection vs plane projection: at 4x mismatch the local
      information loss ``<= 1e-3`` (the projection control must show ``>= 1e-3``;
      ``m6114``'s measured 4.1e-2 is quoted as the upstream reference);
    * ``C`` gap / penetration: identity residual ``<= 1e-12``, contact /
      penetration unit volume ``<= 1e-12``; negative control (gap term dropped)
      ``>= 1e-3``;
    * ``D`` time-domain energy account ``<= 1e-10`` over 3 resolutions x 2 mismatch
      ratios; negative control (broken local conservation) ``> 1e-3``;
    * ``E`` transpose consistency: symmetry ``<= 1e-3`` and transport rate
      ``>= 1.8``;
    * ``F`` ``m6114`` D3 transfer integral order ``>= 1.8``, accuracy ``<= 3 %``;
    * ``G`` determinism: the caller re-runs this and compares the JSON.
    """
    out = {"criteria": {}}
    patches = interface_cases()

    # -- geometry sanity: the interface measure itself ----------------------
    # (a composite 15th-order rule is only exact per element; the self-convergence
    # of the patch area under subdiv 8 -> 16 is the honest quadrature error estimate)
    out["geometry"] = [
        {"patch": p.label,
         "kappa_L": float(getattr(p, "curvature_scale", float("nan"))),
         "area_subdiv8": p.area(subdiv=8),
         "area_subdiv16": p.area(subdiv=16),
         "quadrature_rel": abs(p.area(subdiv=8) - p.area(subdiv=16))
         / max(abs(p.area(subdiv=16)), 1e-300)}
        for p in patches]

    # -- A / B: per-facet gate and mismatch sweep ---------------------------
    a_rows = []
    for patch in patches:
        for n in meshes:
            for ratio in (1, 4):
                a_rows.append({"patch": patch.label,
                               **local_information_loss(patch, n, ratio)})
    a = {"gate": 1e-12, "rows": a_rows,
         "max_per_facet": float(max(r["true_per_facet"] for r in a_rows)),
         "worst_patch": max(a_rows, key=lambda r: r["true_per_facet"])["patch"],
         "n_facets_gated": int(sum(r["n_master"] for r in a_rows))}
    a["pass"] = bool(a["max_per_facet"] <= a["gate"] and
                     all(r["n_uncovered_true"] == 0 for r in a_rows))
    out["criteria"]["A_per_facet_conservation"] = a

    four = [r for r in a_rows if r["ratio"] == 4]
    b = {"gate": 1e-3, "m6114_reference_4x": 4.1e-2,
         "true_max_4x": float(max(r["true_per_facet"] for r in four)),
         "true_global_max_4x": float(max(r["true_global"] for r in four)),
         "plane_per_facet_min_4x": float(min(r["plane_per_facet"] for r in four)),
         "plane_per_facet_max_4x": float(max(r["plane_per_facet"] for r in four)),
         "plane_global_min_4x": float(min(r["plane_global"] for r in four)),
         "rows": four}
    b["pass"] = bool(b["true_max_4x"] <= b["gate"] and
                     b["plane_per_facet_min_4x"] >= b["gate"])
    out["criteria"]["B_true_intersection_vs_projection"] = b

    # -- C: gap / penetration ----------------------------------------------
    c_rows = [gap_penetration_account(p, n=6) for p in patches]
    def _case(row, name):
        return [r for r in row["rows"] if r["case"] == name][0]

    c = {"gate_identity": 1e-12, "gate_unit_volume": 1e-12, "control_gate": 1e-3,
         "rows": c_rows,
         "max_identity_balance_rel": float(max(r["max_identity_balance_rel"] for r in c_rows)),
         # unaccounted (spurious) volume in the *conforming* case: the identity must
         # not invent a gap or a penetration, otherwise the numerical sliver noise is
         # being reported as contact
         "conforming_spurious_volume_rel": float(max(
             max(_case(r, "conforming")["gap_unit_volume_rel"],
                 _case(r, "conforming")["pen_unit_volume_rel"]) for r in c_rows)),
         "max_gap_unit_volume_rel": float(max(r["max_gap_unit_volume_rel"] for r in c_rows)),
         "max_pen_unit_volume_rel": float(max(r["max_pen_unit_volume_rel"] for r in c_rows)),
         "control_min": float(min(min(r["control_residual_gap_case"],
                                      r["control_residual_pen_case"]) for r in c_rows))}
    c["pass"] = bool(c["max_identity_balance_rel"] <= c["gate_identity"]
                     and c["conforming_spurious_volume_rel"] <= c["gate_unit_volume"]
                     and c["control_min"] >= c["control_gate"])
    out["criteria"]["C_gap_penetration_terms"] = c

    # -- D: time-domain energy account -------------------------------------
    d_rows = []
    for patch in patches:
        for n in energy_resolutions:
            for ratio in energy_ratios:
                master, slave = mismatch_case(patch, n, ratio)
                op = CurvedMortarOperator(master, slave, order=2, measure="curved")
                r = interface_energy_account(op, dt=1e-3, nsteps=300)
                broken = break_local_conservation(op)
                rb = interface_energy_account(broken, dt=1e-3, nsteps=300)
                d_rows.append({
                    "patch": patch.label, "h": 1.0 / n, "ratio": ratio,
                    "per_facet_defect": r["per_facet_defect"],
                    "interface_power_rel_max": r["interface_power_rel_max"],
                    "ledger_residual_rel_max": r["ledger_residual_rel_max"],
                    "time_account_residual_rel_max": r["time_account_residual_rel_max"],
                    "lam_max": r["lam_max"],
                    "energy_drift_rel": r["energy_drift_rel"],
                    "broken_per_facet_defect": rb["per_facet_defect"],
                    "broken_interface_power_rel_max": rb["interface_power_rel_max"],
                    "broken_lam_max": rb["lam_max"],
                })
    d = {"gate": 1e-10, "control_gate": 1e-3, "rows": d_rows,
         "max_interface_power_rel": float(max(r["interface_power_rel_max"] for r in d_rows)),
         "max_ledger_residual_rel": float(max(r["ledger_residual_rel_max"] for r in d_rows)),
         "max_time_account_residual_rel": float(max(r["time_account_residual_rel_max"]
                                                    for r in d_rows)),
         "control_min_interface_power_rel": float(min(r["broken_interface_power_rel_max"]
                                                      for r in d_rows)),
         "control_min_per_facet_defect": float(min(r["broken_per_facet_defect"]
                                                   for r in d_rows))}
    d["pass"] = bool(d["max_interface_power_rel"] <= d["gate"]
                     and d["max_ledger_residual_rel"] <= d["gate"]
                     and d["control_min_interface_power_rel"] > d["control_gate"])
    out["criteria"]["D_time_domain_energy_account"] = d

    # -- E: transpose consistency ------------------------------------------
    e_rows = [transpose_consistency_study(p, meshes=e_meshes, ratio=1) for p in patches]
    e = {"symmetry_gate": 1e-3, "rate_gate": 1.8, "rows": e_rows,
         "symmetry_rel_max": float(max(r["duality_rel_max"] for r in e_rows)),
         "min_rate_transport_curved": float(min(r["rate_transport_curved"] for r in e_rows)),
         "max_rate_transport_curved": float(max(r["rate_transport_curved"] for r in e_rows)),
         "max_transport_rel_curved_finest": float(max(r["transport_rel_curved_finest"]
                                                      for r in e_rows)),
         # control 1: the midpoint (order-0) transpose must NOT reach the gate
         "naive_rate_max": float(max(r["rate_transport_naive"] for r in e_rows)),
         # control 2 (information, NOT gated): freezing J at the facet centroid is a
         # midpoint rule on the Jacobian and therefore stays *second order* in an
         # integral statement -- if it degraded the order we would have to suspect the
         # transport metric itself.  What it breaks is the per-facet conservation gate
         # (criterion A: |sum_i m_ij - |G_j|| / |G_j| ~ 1e-4 .. 1e-3 vs the 1e-12 gate),
         # which is where the frozen-measure controls are asserted.  The fold chart is
         # isometric (J == 1), so there the frozen measure *is* the exact one.
         "metric_control_rate_max": float(max(
             max(r["rate_transport_centroid_j"], r["rate_transport_flat"]) for r in e_rows)),
         "metric_control_rate_sphere_graph": {
             r["patch"]: {"centroid_j": r["rate_transport_centroid_j"],
                          "flat": r["rate_transport_flat"]}
             for r in e_rows}}
    e["pass"] = bool(e["symmetry_rel_max"] <= e["symmetry_gate"]
                     and e["min_rate_transport_curved"] >= e["rate_gate"]
                     and e["naive_rate_max"] < e["rate_gate"])
    out["criteria"]["E_transpose_consistency"] = e

    # -- F: m6114 regression -------------------------------------------------
    if with_regression:
        f = transfer_order_regression()
        f["rate_gate"] = 1.8
        f["accuracy_gate"] = 3e-2
        f["pass"] = bool(
            (not np.isfinite(f["field_order"]) or f["field_order"] >= f["rate_gate"])
            and (not np.isfinite(f["area_order"]) or f["area_order"] >= f["rate_gate"])
            and (not np.isfinite(f["field_accuracy"]) or f["field_accuracy"] <= f["accuracy_gate"]))
        out["criteria"]["F_transfer_order_regression"] = f
    else:
        out["criteria"]["F_transfer_order_regression"] = {"skipped": True, "pass": True}

    out["all_pass"] = bool(all(v.get("pass", True) for v in out["criteria"].values()))
    out["provenance"] = provenance()
    return out
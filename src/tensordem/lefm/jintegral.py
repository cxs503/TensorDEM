"""J-integral, equivalent-domain (EDI) J and the interaction (M) integral.

Three independent ways of getting the crack-driving force of a 2-D lattice:

``contour_j``
    the textbook line integral

        J = oint (W dy - T . du/dx ds),     T = sigma . n

    evaluated on a rectangular contour that hugs the crack faces and goes around
    the tip.  ``W`` (strain energy density), ``sigma`` (virial stress) and
    ``du/dx`` (least-squares gradient over *bonded* neighbours) are interpolated
    bilinearly from the lattice nodes.

``domain_j``
    the equivalent-domain form (Li, Shih & Needleman 1985)

        J = (1/B) int_A [ sigma_ij u_i,1 - W delta_1j ] q_,j dA

    with a C1 ring function ``q`` (1 inside ``r_in``, 0 outside ``r_out``).
    Robust against lattice noise, and path independence means independence of the
    annulus radii.

``interaction_integral``
    the two-state M-integral, which separates the modes:

        M = (1/B) int_A [ sigma_ij u_i,1^aux + sigma_ij^aux u_i,1
                          - sigma_ik^aux eps_ik delta_1j ] q_,j dA
          = 2 (K_I K_I^aux + K_II K_II^aux) / E'

    With a unit mode-I auxiliary field ``M = 2 K_I / E'``; with a unit mode-II
    field ``M = 2 K_II / E'``.  ``E' = E`` (plane stress), ``E/(1-nu^2)`` (plane
    strain).  The auxiliary in-plane derivatives are analytic (no finite
    differences), see :func:`auxiliary_field`.

All lengths in m, stresses in Pa, ``B`` = out-of-plane thickness in m.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .lattice2d import BondedLattice2D

__all__ = [
    "contour_j",
    "domain_j",
    "interaction_integral",
    "stress_intensity_from_j",
    "AuxiliaryField",
    "effective_modulus",
]


def effective_modulus(young_modulus: float, poisson_ratio: float = 0.25,
                      plane: str = "stress") -> float:
    """LEFM effective modulus ``E'``."""
    if plane == "stress":
        return float(young_modulus)
    if plane == "strain":
        return float(young_modulus) / (1.0 - poisson_ratio**2)
    raise ValueError("plane must be 'stress' or 'strain'")


def stress_intensity_from_j(J: float, young_modulus: float,
                            poisson_ratio: float = 0.25,
                            plane: str = "stress") -> float:
    """``K = sqrt(J E')``."""
    return float(np.sqrt(abs(J) * effective_modulus(young_modulus, poisson_ratio, plane)))


# --------------------------------------------------------------------- aux field
@dataclass
class AuxiliaryField:
    """Unit-K Williams near-tip field sampled at arbitrary points."""

    mode: str
    kappa: float
    shear_modulus: float
    young_modulus: float
    poisson_ratio: float

    @classmethod
    def create(cls, mode: str, young_modulus: float, poisson_ratio: float = 0.25,
               plane: str = "stress") -> "AuxiliaryField":
        if mode not in ("I", "II"):
            raise ValueError("mode must be 'I' or 'II'")
        kappa = (3.0 - poisson_ratio) / (1.0 + poisson_ratio) if plane == "stress" \
            else 3.0 - 4.0 * poisson_ratio
        mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
        return cls(mode, kappa, mu, young_modulus, poisson_ratio)

    def sample(self, dx: np.ndarray, dy: np.ndarray, k_aux: float = 1.0):
        """Return ``(sigma (...,2,2), u (...,2), du_dx (...,2), eps (...,2,2))``."""
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)
        r = np.hypot(dx, dy)
        r = np.where(r < 1e-12, 1e-12, r)
        theta = np.arctan2(dy, dx)
        c, s = np.cos(theta), np.sin(theta)
        ch, sh = np.cos(0.5 * theta), np.sin(0.5 * theta)
        c3, s3 = np.cos(1.5 * theta), np.sin(1.5 * theta)
        kk = float(k_aux)
        pref = kk / np.sqrt(2.0 * np.pi * r)
        sig = np.zeros(dx.shape + (2, 2))
        if self.mode == "I":
            sig[..., 0, 0] = pref * ch * (1.0 - sh * s3)
            sig[..., 1, 1] = pref * ch * (1.0 + sh * s3)
            sig[..., 0, 1] = sig[..., 1, 0] = pref * ch * sh * c3
        else:
            sig[..., 0, 0] = -pref * sh * (2.0 + ch * c3)
            sig[..., 1, 1] = pref * sh * ch * c3
            sig[..., 0, 1] = sig[..., 1, 0] = pref * ch * (1.0 - sh * s3)

        # displacements and their analytic x-derivative
        A = kk / (2.0 * self.shear_modulus) * np.sqrt(r / (2.0 * np.pi))
        kap = self.kappa
        if self.mode == "I":
            f1 = kap - 1.0 + 2.0 * sh**2
            f2 = kap + 1.0 - 2.0 * ch**2
            ux, uy = A * ch * f1, A * sh * f2
            dudx_r_x = A * ch * f1 / (2.0 * r)
            dudx_t_x = A * (-0.5 * sh * f1 + 2.0 * sh * ch**2)
            dudx_r_y = A * sh * f2 / (2.0 * r)
            dudx_t_y = A * (0.5 * ch * f2 + 2.0 * sh**2 * ch)
        else:
            g1 = kap + 1.0 + 2.0 * ch**2
            g2 = kap - 1.0 - 2.0 * sh**2
            ux, uy = A * sh * g1, -A * ch * g2
            dudx_r_x = A * sh * g1 / (2.0 * r)
            dudx_t_x = A * (0.5 * ch * g1 - 2.0 * sh**2 * ch)
            dudx_r_y = -A * ch * g2 / (2.0 * r)
            dudx_t_y = -A * (-0.5 * sh * g2 - 2.0 * s * ch**2)
        # d/dx = cos(theta) d/dr - sin(theta)/r d/dtheta
        du_dx = np.stack([c * dudx_r_x - s / r * dudx_t_x,
                          c * dudx_r_y - s / r * dudx_t_y], axis=-1)
        u = np.stack([ux, uy], axis=-1)
        eps = self.strain(sig)
        return sig, u, du_dx, eps

    def strain(self, sig: np.ndarray) -> np.ndarray:
        """Plane-stress/strain elastic strain of a 3-D stress state."""
        nu, E = self.poisson_ratio, self.young_modulus
        tr = sig[..., 0, 0] + sig[..., 1, 1]
        eps = np.empty_like(sig)
        eps[..., 0, 0] = (sig[..., 0, 0] - nu * sig[..., 1, 1]) / E
        eps[..., 1, 1] = (sig[..., 1, 1] - nu * sig[..., 0, 0]) / E
        eps[..., 0, 1] = eps[..., 1, 0] = (1.0 + nu) * sig[..., 0, 1] / E
        return eps


# ------------------------------------------------------------------ grid helpers
def _fields(lat: BondedLattice2D):
    """Interpolators for W, sigma, du/dx on the lattice grid."""
    W = lat.node_energy_density()
    sig = lat.node_stress()
    grad = lat.node_displacement_gradient()
    nrows, ncols = lat.nrows, lat.ncols
    xs = np.arange(ncols) * lat.cfg.spacing
    ys = np.arange(nrows) * lat.cfg.spacing

    def make(vals):
        return RegularGridInterpolator((ys, xs), vals.reshape(nrows, ncols),
                                       bounds_error=False, fill_value=None)

    interp = {
        "W": make(W),
        "sxx": make(sig[:, 0, 0]), "syy": make(sig[:, 1, 1]),
        "sxy": make(sig[:, 0, 1]),
        "uxx": make(grad[:, 0, 0]), "uxy": make(grad[:, 0, 1]),
        "uyx": make(grad[:, 1, 0]), "uyy": make(grad[:, 1, 1]),
    }
    return interp


def _sample(interp, pts: np.ndarray) -> dict[str, np.ndarray]:
    return {k: f(pts) for k, f in interp.items()}


# ------------------------------------------------------------------- contour J
def contour_j(lat: BondedLattice2D, tip: Sequence[float], radius: float,
              behind: float | None = None, n_per_leg: int = 121,
              include_face_legs: bool = True) -> float:
    """Textbook contour J-integral ``oint (W dy - T . du/dx ds)``.

    The contour is a U-shape that encircles the tip and returns along the two
    crack faces (``y = y_tip -/+ delta``).  Traversal is counter-clockwise for
    the enclosed region, so the outward normal is ``n = (t_y, -t_x)`` with ``t``
    the unit tangent.  The face legs are kept in the path (their contribution is
    zero in the continuum: ``dy = 0`` and ``T = 0``) but they are offset by one
    lattice row from the slit so that the bilinear interpolation never straddles
    the crack-face discontinuity.
    """
    a = lat.cfg.spacing
    xt, yt = float(tip[0]), float(tip[1])
    behind = 2.0 * a if behind is None else float(behind)
    r = float(radius)
    d_lo = 0.5 * a
    d_hi = 1.5 * a
    corners = [
        (xt - behind, yt - d_lo),
        (xt + r, yt - d_lo),
        (xt + r, yt + r),
        (xt - r, yt + r),
        (xt - r, yt + d_hi),
        (xt - behind, yt + d_hi),
    ]
    if not include_face_legs:
        # the two crack-face legs carry zero contribution in the continuum
        # (dy = 0, T = 0), so they may be dropped; the lattice face nodes are
        # half-weighted and dropping them removes that bias
        corners = corners[1:-1]
    interp = _fields(lat)
    J = 0.0
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        t = np.linspace(0.0, 1.0, n_per_leg)
        pts = np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1)
        seg = np.hypot(x1 - x0, y1 - y0)
        tx, ty = (x1 - x0) / seg, (y1 - y0) / seg
        nx, ny = ty, -tx
        mid = 0.5 * (pts[1:] + pts[:-1])
        ds = seg / (n_per_leg - 1)
        vals = _sample(interp, mid)
        trac_x = vals["sxx"] * nx + vals["sxy"] * ny
        trac_y = vals["sxy"] * nx + vals["syy"] * ny
        integrand = vals["W"] * ny - (trac_x * vals["uxx"] + trac_y * vals["uyx"])
        J += float(np.sum(integrand) * ds)
    return J


# -------------------------------------------------------------------- domain J
def _ring_function(lat: BondedLattice2D, tip: Sequence[float], r_in: float,
                   r_out: float, band: float | None = None):
    """Ring q and its gradient, plus the node mask of the integration domain."""
    a = lat.cfg.spacing
    xt, yt = float(tip[0]), float(tip[1])
    dx = lat.x - xt
    dy = lat.y - yt
    r = np.hypot(dx, dy)
    band = 1.5 * a if band is None else float(band)
    inside = (r >= r_in) & (r <= r_out)
    # exclude the crack faces (the displacement field is discontinuous there)
    inside &= ~((np.abs(dy) < band) & (dx < 0.0))
    t = np.clip((r_out - r) / max(r_out - r_in, 1e-30), 0.0, 1.0)
    q = t * t * (3.0 - 2.0 * t)  # C1 smoothstep
    dqdt = 6.0 * t * (1.0 - t)
    dqdr = dqdt * (-1.0 / max(r_out - r_in, 1e-30))
    safe = np.where(r < 1e-30, 1e-30, r)
    qx = dqdr * dx / safe
    qy = dqdr * dy / safe
    qx = np.where(inside, qx, 0.0)
    qy = np.where(inside, qy, 0.0)
    return q, qx, qy, inside


def domain_j(lat: BondedLattice2D, tip: Sequence[float], r_in: float,
             r_out: float, band: float | None = None) -> float:
    """Equivalent-domain J-integral with a smooth ring function."""
    a, B = lat.cfg.spacing, lat.cfg.thickness
    W = lat.node_energy_density()
    sig = lat.node_stress()
    grad = lat.node_displacement_gradient()
    _, qx, qy, inside = _ring_function(lat, tip, r_in, r_out, band)
    V = a * a * B
    sxx, syy, sxy = sig[:, 0, 0], sig[:, 1, 1], sig[:, 0, 1]
    uxx, uxy = grad[:, 0, 0], grad[:, 1, 0]  # du_x/dx, du_y/dx
    integrand = (sxx * uxx + sxy * uxy - W) * qx + (sxy * uxx + syy * uxy) * qy
    return float(np.sum(integrand[inside]) * a * a / B)


def interaction_integral(lat: BondedLattice2D, tip: Sequence[float], mode: str,
                         r_in: float, r_out: float, k_aux: float = 1.0,
                         band: float | None = None) -> float:
    """Two-state M-integral for mode ``'I'`` or ``'II'``; returns M [J/m]."""
    a, B = lat.cfg.spacing, lat.cfg.thickness
    cfg = lat.cfg
    sig = lat.node_stress()
    grad = lat.node_displacement_gradient()
    _, qx, qy, inside = _ring_function(lat, tip, r_in, r_out, band)
    dx = lat.x - float(tip[0])
    dy = lat.y - float(tip[1])
    aux = AuxiliaryField.create(mode, cfg.young_modulus, cfg.poisson_ratio, "stress")
    s_aux, u_aux, du_aux, e_aux = aux.sample(dx, dy, k_aux=k_aux)
    ux, uy = lat.u[0::2], lat.u[1::2]
    uxx, uyx = grad[:, 0, 0], grad[:, 1, 0]        # du_x/dx, du_y/dx (total)
    uxx_a, uyx_a = du_aux[:, 0], du_aux[:, 1]      # aux du_x/dx, du_y/dx
    sxx, syy, sxy = sig[:, 0, 0], sig[:, 1, 1], sig[:, 0, 1]
    axx, ayy, axy = s_aux[:, 0, 0], s_aux[:, 1, 1], s_aux[:, 0, 1]
    # interaction strain energy density W12 = sigma_ij eps_ij^aux
    W12 = sxx * e_aux[:, 0, 0] + syy * e_aux[:, 1, 1] + 2.0 * sxy * e_aux[:, 0, 1]
    term_x = (sxx * uxx_a + sxy * uyx_a + axx * uxx + axy * uyx - W12)
    term_y = (sxy * uxx_a + syy * uyx_a + axy * uxx + ayy * uyx)
    integrand = term_x * qx + term_y * qy
    return float(np.sum(integrand[inside]) * a * a / B)


def stress_intensity_from_m(lat: BondedLattice2D, tip: Sequence[float], mode: str,
                            r_in: float, r_out: float) -> float:
    """``K_mode = M E' / 2`` for a unit auxiliary field."""
    M = interaction_integral(lat, tip, mode, r_in, r_out)
    return float(M * effective_modulus(lat.cfg.young_modulus, lat.cfg.poisson_ratio,
                                       "stress") / 2.0)
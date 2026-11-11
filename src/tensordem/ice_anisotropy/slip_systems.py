"""Slip-system (crystal-plasticity) reference for columnar ice: the independent
check the continuum yield surface is scored against.

Why this module exists
----------------------
A transverse-isotropy yield surface (Hill / Tsai-Wu) is a *quadratic envelope*:
it is calibrated with two or three uniaxial strengths and then interpolates the
remaining directions with a fixed functional form.  Nothing in that form knows
why columnar ice is weak at 45 degrees to its columns.  To score the envelope at
45 degrees we therefore need a reference built from *different* physics, not
from the same envelope fitted differently:

    the Schmid (resolved-shear) law on the slip systems of the ice crystal,
    averaged over the measured c-axis ODF.

That reference contains no quadratic interaction at all -- the angular shape of
the strength curve comes from the *geometry* of the slip planes relative to the
loading axis.  Once its single free ratio (pyramidal : basal critical resolved
shear stress, CRSS) is fixed by the 0/90 strengths, the 45 degree value is a
genuine prediction of the reference model, and the useful comparison is
*continuum envelope vs slip model* at the direction neither of them was fitted
to.

Mechanisms
----------
* **basal** slip -- glide plane normal = c-axis, CRSS ``tau_basal``.  For an S2
  fabric (c-axes in the horizontal plane) a *vertical* uniaxial load has zero
  resolved shear on every basal plane, which is the micro-mechanical reason the
  column direction is the strong one.
* **pyramidal** slip -- glide planes inclined by ``tilt`` (default 45 deg) to
  the c-axis, CRSS ``tau_pyramidal``.  This is the mechanism that carries the
  column-parallel load, and its CRSS ratio to the basal one is the
  single anisotropy parameter of the reference model.

Resolved-shear factor
---------------------
For a unit uniaxial stress along ``d`` and a glide plane of normal ``p`` the
maximum resolved shear over the slip directions in that plane is

    m(d, p) = |d.p| * sqrt(1 - (d.p)^2)      (Schmid factor, maximum over s)

and for a general unit stress tensor ``S`` it is the component of ``S p``
perpendicular to ``p``, ``m = |S p - (p.S p) p|`` (:func:`resolved_shear_factor`).

Two averaging conventions are reported, and they bracket the polycrystal
response:

* ``"mean"`` -- mean-field: the macroscopic strength is ``tau / <m>`` with ``<m>``
  the ODF-weighted mean Schmid factor (the classical mean-field estimate used by
  Vogler & Clayton 2008 for columnar ice).
* ``"max"`` -- Sachs (uniform-stress) lower bound: the most favourably oriented
  grain yields first, ``tau / max m``.  It is a rigorous lower bound, and for a
  single-system material it collapses the 45 degree dip (see the study report).

References
----------
- Schmid, E. & Boas, W. (1935). *Kristallplastizitaet*. Springer.
- Kocks, U. F., Tome, C. N. & Wenk, H.-R. (1998). *Texture and Anisotropy*.
  Cambridge University Press.  (Sachs vs Taylor vs self-consistent bounds.)
- Vogler, T. J. & Clayton, J. D. (2008). "Heterogeneous deformation in ductile
  columnar ice." Philosophical Magazine 88(3), 341-378.
- Schulson, E. M. & Duval, P. (2009). *Creep and Fracture of Ice*. CUP.
  (basal / prismatic / pyramidal CRSS hierarchy of Ih ice.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .fabric import FabricODF, _orthonormal_basis

__all__ = [
    "SlipSystemModel",
    "resolved_shear_factor",
    "schmid_factor",
    "uniaxial_stress_tensor",
    "load_direction",
]


def load_direction(theta_deg: float, *, azimuth_deg: float = 0.0) -> np.ndarray:
    """Unit load direction at ``theta`` from the column axis (+z), in the x-z plane."""
    th = np.radians(float(theta_deg))
    az = np.radians(float(azimuth_deg))
    return np.array([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)])


def uniaxial_stress_tensor(direction: Sequence[float], magnitude: float = 1.0) -> np.ndarray:
    """Unit (or scaled) uniaxial stress tensor ``sigma d (x) d``."""
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    return magnitude * np.outer(d, d)


def resolved_shear_factor(plane_normals: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Maximum resolved shear (per unit stress magnitude) on each glide plane.

    ``|S p - (p.S p) p|`` -- the part of the traction ``S p`` that can drive
    glide, maximised over slip directions ``s`` perpendicular to ``p``.
    """
    p = np.asarray(plane_normals, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("plane_normals must have shape (N, 3)")
    S = np.asarray(stress, dtype=float).reshape(3, 3)
    trac = p @ S.T                      # (N, 3) = S p
    normal_part = np.einsum("ni,ni->n", trac, p)
    shear = trac - normal_part[:, None] * p
    return np.linalg.norm(shear, axis=1)


def schmid_factor(plane_normals: np.ndarray, direction: Sequence[float]) -> np.ndarray:
    """Maximum Schmid factor for a unit uniaxial load (closed form)."""
    p = np.asarray(plane_normals, dtype=float)
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    c = np.abs(p @ d)
    return c * np.sqrt(np.maximum(0.0, 1.0 - c * c))


def _pyramidal_normals(odf: FabricODF, direction: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Most favourably oriented pyramidal plane normal, per ODF sample.

    The pyramidal family is the cone of plane normals at ``tilt_deg`` from the
    c-axis; for a fixed load direction the member of the family that maximises
    the resolved shear lies in the plane spanned by the c-axis and the load.
    """
    c = odf.directions
    cd = c @ direction
    perp = direction[None, :] - cd[:, None] * c
    nrm = np.linalg.norm(perp, axis=1)
    fallback = np.tile(np.array([0.0, 0.0, 1.0]), (len(c), 1))
    e = np.where(nrm[:, None] > 1e-12, perp / np.maximum(nrm, 1e-12)[:, None], fallback)
    alpha = np.radians(tilt_deg)
    return np.cos(alpha) * c + np.sin(alpha) * e


@dataclass
class SlipSystemModel:
    """Two-mechanism (basal + pyramidal) Schmid-law reference for columnar ice.

    Parameters
    ----------
    basal_tau
        Basal CRSS.  Only ratios matter for the strength *ratios*; absolute
        values matter for the absolute strengths.
    pyramids_tau_ratio
        ``tau_pyramidal / tau_basal``.  With a columnar ODF this single number
        sets the column-parallel : transverse strength ratio.
    pyramidal_tilt_deg
        Inclination of the pyramidal glide planes to the c-axis (45 deg is the
        geometric optimum for a resolved shear of 0.5).
    convention
        ``"mean"`` (mean-field, default) or ``"max"`` (Sachs lower bound).
    """

    basal_tau: float = 1.0
    pyramids_tau_ratio: float = 3.5
    pyramidal_tilt_deg: float = 45.0
    convention: str = "mean"

    def __post_init__(self) -> None:
        if self.basal_tau <= 0.0:
            raise ValueError("basal_tau must be positive")
        if self.pyramids_tau_ratio <= 0.0:
            raise ValueError("pyramids_tau_ratio must be positive")
        if self.convention not in ("mean", "max"):
            raise ValueError("convention must be 'mean' or 'max'")

    # -- properties --------------------------------------------------------
    @property
    def pyramids_tau(self) -> float:
        return self.basal_tau * self.pyramids_tau_ratio

    def _reduce(self, m: np.ndarray, weights: np.ndarray) -> float:
        m = np.asarray(m, dtype=float)
        if self.convention == "mean":
            return float((weights * m).sum())
        return float(m.max())

    # -- mechanism factors -------------------------------------------------
    def mechanism_factors(self, odf: FabricODF, direction: Sequence[float]) -> dict[str, float]:
        """Reduced resolved-shear factors for both mechanisms along ``direction``."""
        d = np.asarray(direction, dtype=float).reshape(3)
        d = d / np.linalg.norm(d)
        w = odf.weights
        m_basal = schmid_factor(odf.directions, d)
        m_pyr = schmid_factor(_pyramidal_normals(odf, d, self.pyramidal_tilt_deg), d)
        return {"basal": self._reduce(m_basal, w), "pyramidal": self._reduce(m_pyr, w)}

    def mechanism_strengths(self, odf: FabricODF,
                            direction: Sequence[float]) -> dict[str, float | None]:
        """Uniaxial strength (Pa) each mechanism allows along ``direction``.

        ``None`` means the mechanism has no resolved shear in this direction
        (inactive, not "infinitely strong" in any physical sense).
        """
        fac = self.mechanism_factors(odf, direction)
        out: dict[str, float | None] = {}
        for name, tau in (("basal", self.basal_tau), ("pyramidal", self.pyramids_tau)):
            m = fac[name]
            out[name] = None if m <= 1e-12 else tau / m
        return out

    def uniaxial_strength(self, odf: FabricODF, direction: Sequence[float]) -> float:
        """Governing uniaxial strength (Pa): the weakest active mechanism."""
        vals = [v for v in self.mechanism_strengths(odf, direction).values() if v is not None]
        if not vals:
            raise ValueError("no active slip mechanism along this direction")
        return float(min(vals))

    def governing_mechanism(self, odf: FabricODF, direction: Sequence[float]) -> str:
        vals = {k: v for k, v in self.mechanism_strengths(odf, direction).items() if v is not None}
        return min(vals, key=lambda k: vals[k])

    # -- strength curves ---------------------------------------------------
    def strength_curve(self, odf: FabricODF, angles_deg: Sequence[float]) -> list[dict[str, Any]]:
        """Uniaxial strength vs load angle from the column axis (degrees)."""
        rows = []
        for th in angles_deg:
            d = load_direction(th)
            fac = self.mechanism_factors(odf, d)
            strengths = self.mechanism_strengths(odf, d)
            rows.append({
                "theta_deg": float(th),
                "schmid_basal": fac["basal"],
                "schmid_pyramidal": fac["pyramidal"],
                "strength_basal_Pa": strengths["basal"],
                "strength_pyramidal_Pa": strengths["pyramidal"],
                "strength_Pa": self.uniaxial_strength(odf, d),
                "governing": self.governing_mechanism(odf, d),
            })
        return rows

    def strength_ratio(self, odf: FabricODF) -> float:
        """``sigma(0 deg) / sigma(90 deg)`` -- the column/transverse strength ratio."""
        return float(self.uniaxial_strength(odf, load_direction(0.0))
                     / self.uniaxial_strength(odf, load_direction(90.0)))

    # -- shear -------------------------------------------------------------
    def transverse_shear_strength(self, odf: FabricODF, *,
                                  loading_axis: Sequence[float] = (0.0, 1.0, 0.0),
                                  shear_normal: Sequence[float] = (1.0, 0.0, 0.0)) -> dict[str, Any]:
        """Strength under pure shear ``tau_{xz}`` (a transverse shear for a +z column axis).

        The stress state is the symmetric unit shear
        ``S = n (x) a + a (x) n`` (``a`` = shear direction, ``n`` = plane normal);
        each mechanism yields when its resolved shear reaches its CRSS.
        """
        a = np.asarray(loading_axis, dtype=float).reshape(3)
        n = np.asarray(shear_normal, dtype=float).reshape(3)
        a = a / np.linalg.norm(a)
        n = n / np.linalg.norm(n)
        if abs(float(a @ n)) > 1e-9:
            raise ValueError("shear direction and plane normal must be orthogonal")
        S = 0.5 * (np.outer(n, a) + np.outer(a, n))
        S = S / np.abs(S).max()          # unit shear magnitude
        w = odf.weights
        fac_b = self._reduce(resolved_shear_factor(odf.directions, S), w)
        fac_p = self._reduce(
            resolved_shear_factor(_pyramidal_normals(odf, a, self.pyramidal_tilt_deg), S), w)
        cands = []
        if fac_b > 1e-12:
            cands.append(self.basal_tau / fac_b)
        if fac_p > 1e-12:
            cands.append(self.pyramids_tau / fac_p)
        return {
            "shear_strength_Pa": float(min(cands)),
            "schmid_basal": fac_b,
            "schmid_pyramidal": fac_p,
        }


def calibrate_pyramidal_ratio(odf: FabricODF, target_ratio: float, *,
                              convention: str = "mean", tilt_deg: float = 45.0,
                              tol: float = 1e-10, max_iter: int = 200) -> float:
    """Solve ``tau_pyramidal / tau_basal`` so that ``sigma(0)/sigma(90) == target_ratio``.

    Bisection on a monotone problem: raising the pyramidal CRSS only raises the
    column-parallel strength, so the ratio is strictly increasing in the free
    parameter.
    """
    if target_ratio <= 1.0:
        raise ValueError("target_ratio must exceed 1")

    def ratio_of(r: float) -> float:
        model = SlipSystemModel(pyramids_tau_ratio=r, convention=convention,
                                pyramidal_tilt_deg=tilt_deg)
        return model.strength_ratio(odf)

    lo, hi = 1.0, 2.0
    while ratio_of(hi) < target_ratio:
        lo, hi = hi, hi * 2.0
        if hi > 1e8:
            raise RuntimeError("could not bracket the pyramidal CRSS ratio")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if ratio_of(mid) < target_ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)
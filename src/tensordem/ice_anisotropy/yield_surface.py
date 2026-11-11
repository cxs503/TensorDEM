"""Transversely isotropic (columnar-ice) strength surface: Hill/Tsai-Wu class.

Why this module exists
----------------------
A columnar ice sheet is transversely isotropic about its column (growth) axis:
the fabric is random within the plane perpendicular to that axis but very
different along it, so two independent uniaxial strengths exist --

    sigma_parallel  : load along the columns (c-axes perpendicular to the load)
    sigma_perp      : load across the columns

and the yield surface must interpolate everything in between.  The classical
quadratic form for a transversely isotropic metal (Hill 1948, specialised with
F = G for in-plane isotropy) is

    f = F (s2 - s3)^2 + G (s3 - s1)^2 + H (s1 - s2)^2
        + 2L t23^2 + 2M t31^2 + 2N t12^2 = 1

with axis 3 = column axis.  Uniaxial strengths fix the normal coefficients,

    sigma_parallel = 1/sqrt(F+G),   sigma_perp = 1/sqrt(G+H),   F = G,

the in-plane isotropy requirement N = F + 2H follows algebraically, and the
transverse shear strength ``tau_transverse`` (yield under ``t31``/``t23``) fixes
L = M = 1/(2 tau_transverse^2).  One strength ratio is therefore not enough --
the surface needs (sigma_parallel, sigma_perp, tau_transverse) and the third
quantity is exactly where a quadratic envelope can disagree with a
slip-system-resolved model (see :mod:`tensordem.ice_anisotropy.slip_systems`).

Isotropic degeneration (the negative control)
---------------------------------------------
With ``sigma_parallel = sigma_perp = sigma_y`` and ``tau_transverse = sigma_y /
sqrt(3)`` the three normal coefficients become equal, ``F = G = H =
1/(2 sigma_y^2)``, and ``2L = 2M = 2N = 3/sigma_y^2``, i.e. :class:`HillTI`
reduces **identically** to the von Mises surface (``3 J2 = sigma_y^2``).  That
identity is asserted to machine precision in the tests, and it is what makes the
uniform-fabric limit of :func:`strength_surface_from_fabric` a meaningful
negative control rather than a cosmetic one (the uniform limit is reached by the
slip-system reference's own isotropy, not by forcing the parameters).

References
----------
- Hill, R. (1948). "A theory of the yielding and plastic flow of anisotropic
  metals." Proc. R. Soc. Lond. A 193, 281-297.
- Tsai, S. W. & Wu, E. M. (1971). "A general theory of strength for anisotropic
  materials." J. Compos. Mater. 5(1), 58-80.
- Weeks, W. F. & Assur, A. (1967). CRREL Monograph II-C3 (columnar S2 ice
  strength anisotropy).
- Timco, G. W. & Weeks, W. F. (2010). "A review of the engineering properties of
  sea ice in the twenty-first century." Cold Reg. Sci. Technol. 60, 107-129.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .fabric import FabricODF, _orthonormal_basis
from .slip_systems import SlipSystemModel, calibrate_pyramidal_ratio, load_direction

__all__ = [
    "HillTI",
    "calibrated_slip_reference",
    "strength_surface_from_fabric",
    "COLUMNAR_RATIO_RANGE",
]

#: Literature range of the column-parallel : transverse uniaxial strength ratio
#: for columnar S2 sea ice (Weeks & Assur 1967; Timco & Weeks 2010 review the
#: observed 2-5 band; the campaign anchor is 3-5).
COLUMNAR_RATIO_RANGE: tuple[float, float] = (3.0, 5.0)


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n <= 0.0:
        raise ValueError("axis must be nonzero")
    return a / n


@dataclass
class HillTI:
    """Quadratic transversely isotropic (Hill) strength surface.

    Parameters
    ----------
    sigma_parallel
        Uniaxial strength (Pa) with the load along the column axis (material
        axis 3).
    sigma_perp
        Uniaxial strength (Pa) with the load normal to the column axis.
    tau_transverse
        Transverse (interlaminar) shear strength (Pa): yield under ``t31``.
    axis
        Column axis in the specimen frame (default ``+z``).
    label
        Provenance tag.
    """

    sigma_parallel: float
    sigma_perp: float
    tau_transverse: float
    axis: Sequence[float] = (0.0, 0.0, 1.0)
    label: str = "hill-ti"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sigma_parallel", "sigma_perp", "tau_transverse"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite stress")
            setattr(self, name, value)
        self.axis = _unit(self.axis)
        e1, e2 = _orthonormal_basis(np.asarray(self.axis))
        self._basis = np.stack([e1, e2, np.asarray(self.axis)], axis=1)  # columns e1,e2,e3

    # -- coefficients ------------------------------------------------------
    @property
    def coefficients(self) -> dict[str, float]:
        """Hill coefficients ``(F, G, H, L, M, N)`` in the material frame."""
        sp, st, tau = self.sigma_parallel, self.sigma_perp, self.tau_transverse
        F = 1.0 / (2.0 * sp * sp)
        G = F
        H = 1.0 / (st * st) - F
        M = 1.0 / (2.0 * tau * tau)
        return {"F": F, "G": G, "H": H, "L": M, "M": M, "N": F + 2.0 * H}

    @property
    def strength_ratio(self) -> float:
        """``sigma_parallel / sigma_perp``."""
        return self.sigma_parallel / self.sigma_perp

    # -- frame handling ----------------------------------------------------
    def to_material_frame(self, stress: Sequence[Sequence[float]]) -> np.ndarray:
        """Rotate a specimen-frame stress tensor into the material frame."""
        S = np.asarray(stress, dtype=float).reshape(3, 3)
        return self._basis.T @ S @ self._basis

    def from_material_frame(self, stress: Sequence[Sequence[float]]) -> np.ndarray:
        S = np.asarray(stress, dtype=float).reshape(3, 3)
        return self._basis @ S @ self._basis.T

    # -- criterion ---------------------------------------------------------
    def hill_form(self, stress: Sequence[Sequence[float]], *, material_frame: bool = False) -> float:
        """Hill quadratic form ``f`` (``f = 1`` on the surface, ``f > 1`` inside-out)."""
        S = np.asarray(stress, dtype=float).reshape(3, 3)
        if not material_frame:
            S = self.to_material_frame(S)
        c = self.coefficients
        s1, s2, s3 = S[0, 0], S[1, 1], S[2, 2]
        t23, t31, t12 = S[1, 2], S[2, 0], S[0, 1]
        return float(c["F"] * (s2 - s3) ** 2 + c["G"] * (s3 - s1) ** 2
                     + c["H"] * (s1 - s2) ** 2
                     + 2.0 * c["L"] * t23 ** 2 + 2.0 * c["M"] * t31 ** 2
                     + 2.0 * c["N"] * t12 ** 2)

    def yield_index(self, stress: Sequence[Sequence[float]], *,
                    material_frame: bool = False) -> float:
        """``sqrt(f)``: 1.0 on the surface, < 1 elastic, > 1 failed."""
        f = self.hill_form(stress, material_frame=material_frame)
        if f <= 0.0:
            raise ValueError("Hill form is non-positive: stress state is outside the "
                             "criterion's range (check the coefficients)")
        return float(np.sqrt(f))

    def is_failed(self, stress: Sequence[Sequence[float]]) -> bool:
        return self.yield_index(stress) >= 1.0

    # -- uniaxial strengths ------------------------------------------------
    def uniaxial_stress(self, theta_deg: float, *, azimuth_deg: float = 0.0,
                        magnitude: float = 1.0) -> np.ndarray:
        """Specimen-frame unit uniaxial stress at ``theta`` from the column axis."""
        d = load_direction(theta_deg, azimuth_deg=azimuth_deg)
        return magnitude * np.outer(d, d)

    def uniaxial_strength(self, theta_deg: float, *, azimuth_deg: float = 0.0) -> float:
        """Uniaxial strength (Pa) for a load at ``theta`` degrees from the column axis."""
        S = self.uniaxial_stress(theta_deg, azimuth_deg=azimuth_deg, magnitude=1.0)
        f = self.hill_form(S)
        if f <= 0.0:
            raise ValueError("uniaxial direction has no finite strength for these coefficients")
        return float(1.0 / np.sqrt(f))

    def strength_curve(self, angles_deg: Sequence[float]) -> list[dict[str, float]]:
        return [{"theta_deg": float(th),
                 "strength_Pa": self.uniaxial_strength(th),
                 "strength_over_perp": self.uniaxial_strength(th) / self.sigma_perp}
                for th in angles_deg]

    def biaxial_locus(self, sigma1_over_perp: Sequence[float]) -> list[dict[str, float]]:
        """Plane-stress (s3 = 0) yield locus: axial critical stress vs lateral stress.

        ``s1`` is the stress in the transverse plane and ``s3`` the column-axis
        stress; for each prescribed ``s1`` the root ``s3`` of ``f = 1`` is
        returned (the first root on each branch).
        """
        c = self.coefficients
        rows = []
        for s1 in sigma1_over_perp:
            s1a = float(s1) * self.sigma_perp
            # f(s3) = (F+G) s3^2 - (G s1 + F s1) s3 + (G+H... ) -- expand:
            # F(0-s3)^2 + G(s3-s1)^2 + H(s1-0)^2 = 1
            a = c["F"] + c["G"]
            b = -2.0 * c["G"] * s1a
            cc = c["G"] * s1a ** 2 + c["H"] * s1a ** 2 - 1.0
            disc = b * b - 4.0 * a * cc
            row = {"sigma1_Pa": s1a, "sigma1_over_perp": float(s1)}
            if disc < 0.0:
                row["sigma3_Pa"] = float("nan")
                row["feasible"] = False
            else:
                root = (-b + np.sqrt(disc)) / (2.0 * a)
                row["sigma3_Pa"] = float(root)
                row["sigma3_over_parallel"] = float(root / self.sigma_parallel)
                row["feasible"] = True
            rows.append(row)
        return rows

    # -- consistency -------------------------------------------------------
    def isotropic_deviation(self) -> dict[str, float]:
        """Distance from the von Mises limit (all zero => exactly isotropic)."""
        vm_tau = self.sigma_perp / np.sqrt(3.0)
        return {
            "strength_ratio_minus_one": float(self.strength_ratio - 1.0),
            "tau_ratio_minus_one": float(self.tau_transverse / vm_tau - 1.0),
        }

    def is_isotropic(self, tol: float = 1e-9) -> bool:
        dev = self.isotropic_deviation()
        return (abs(dev["strength_ratio_minus_one"]) <= tol
                and abs(dev["tau_ratio_minus_one"]) <= tol)

    def von_mises_deviation(self, angles_deg: Sequence[float] = (0.0, 45.0, 90.0)) -> float:
        """Max relative deviation of the uniaxial strength from von Mises (unitless)."""
        if not self.is_isotropic(tol=1.0):
            raise ValueError("surface is not isotropic; deviation from von Mises is meaningless")
        vals = [self.uniaxial_strength(th) for th in angles_deg]
        ref = np.sqrt(3.0) * self.tau_transverse
        return float(max(abs(v / ref - 1.0) for v in vals))

    def hill_matrix(self) -> np.ndarray:
        """6x6 Hill matrix in Voigt (11,22,33,23,31,12) ordering for PSD checks."""
        c = self.coefficients
        F, G, H = c["F"], c["G"], c["H"]
        M = np.zeros((6, 6))
        M[:3, :3] = np.array([[G + H, -H, -G],
                              [-H, F + H, -F],
                              [-G, -F, F + G]])
        M[3, 3] = 2.0 * c["L"]
        M[4, 4] = 2.0 * c["M"]
        M[5, 5] = 2.0 * c["N"]
        return M

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of :meth:`hill_matrix`; all >= 0 => closed convex surface."""
        return np.linalg.eigvalsh(self.hill_matrix())

    def is_convex(self, tol: float = 1e-12) -> bool:
        return bool(self.eigenvalues().min() >= -tol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "hill-ti",
            "sigma_parallel_Pa": self.sigma_parallel,
            "sigma_perp_Pa": self.sigma_perp,
            "tau_transverse_Pa": self.tau_transverse,
            "strength_ratio": self.strength_ratio,
            "axis": list(map(float, self.axis)),
            "coefficients": self.coefficients,
            "eigenvalues": self.eigenvalues().tolist(),
            "is_convex": self.is_convex(),
            "is_isotropic": self.is_isotropic(tol=1e-9),
            "label": self.label,
            "metadata": dict(self.metadata),
        }


def calibrated_slip_reference(reference_odf: FabricODF, *,
                              ratio_columnar: float = 3.5,
                              convention: str = "mean",
                              tilt_deg: float = 45.0,
                              sigma_perp: float | None = None) -> SlipSystemModel:
    """Slip-system reference calibrated to a target column/transverse ratio.

    The single free parameter of the reference model (pyramidal : basal CRSS) is
    solved so that ``sigma(0 deg)/sigma(90 deg) = ratio_columnar`` on
    ``reference_odf`` (the ideal columnar fabric by default).
    """
    if not (COLUMNAR_RATIO_RANGE[0] <= ratio_columnar <= COLUMNAR_RATIO_RANGE[1]):
        raise ValueError(f"ratio_columnar={ratio_columnar} is outside the literature "
                         f"band {COLUMNAR_RATIO_RANGE}")
    r_pyr = calibrate_pyramidal_ratio(reference_odf, ratio_columnar,
                                      convention=convention, tilt_deg=tilt_deg)
    basal_tau = 1.0
    if sigma_perp is not None:
        unit = SlipSystemModel(pyramids_tau_ratio=r_pyr, convention=convention,
                               pyramidal_tilt_deg=tilt_deg)
        basal_tau = float(sigma_perp) / unit.uniaxial_strength(
            reference_odf, load_direction(90.0))
    return SlipSystemModel(basal_tau=basal_tau, pyramids_tau_ratio=r_pyr,
                           convention=convention, pyramidal_tilt_deg=tilt_deg)


def strength_surface_from_fabric(odf: FabricODF, *, sigma_perp: float,
                                 ratio_columnar: float = 3.5,
                                 convention: str = "mean",
                                 tilt_deg: float = 45.0,
                                 reference_odf: FabricODF | None = None,
                                 label: str = "hill-ti-from-odf") -> HillTI:
    """Build the TI strength surface that belongs to a c-axis ODF.

    What comes from where
    ---------------------
    The *shape* of the surface is taken from the slip-system reference, not
    fitted: the reference is calibrated **once** on the ideal columnar fabric so
    that its column/transverse ratio equals ``ratio_columnar`` (the literature
    anchor, default 3.5), and then evaluated on the ODF actually supplied --

        r(ODF)        = sigma_par / sigma_perp   (uniaxial, mean-field Schmid)
        t(ODF)        = tau_transverse / sigma_perp

    The absolute scale is the measured/declared transverse strength
    ``sigma_perp``, so ``sigma_parallel = sigma_perp * r(ODF)`` and
    ``tau_transverse = sigma_perp * t(ODF)``.

    Consequences the tests exploit:

    * **Uniform ODF** -> the slip reference is isotropic, so ``r = 1`` and
      ``t = <m_n>/<m_shear>``.  For an isotropic crystal population that ratio
      comes out at ``1/sqrt(3)`` (checked numerically to ~1e-3 in the study), so
      the surface lands on von Mises: ``sigma_parallel = sigma_perp`` and
      ``tau_transverse = sigma_perp/sqrt(3)``.  The isotropic degeneration is
      therefore a *prediction* of the reference model, and the strength ratio
      goes to 1 within the 5% control tolerance.
    * **Ideal columnar ODF** -> ``r = ratio_columnar`` by construction and
      ``t = 1/2`` (mean Schmid factor ``2/pi`` for transverse shear vs ``1/pi``
      for horizontal uniaxial loading).

    Parameters
    ----------
    odf
        Fabric the surface is required for (its column axis is used).
    sigma_perp
        Transverse uniaxial strength of that fabric (Pa) -- the measured anchor
        that sets the absolute scale.
    ratio_columnar
        Target column/transverse ratio on the ideal columnar fabric (3-5 band).
    """
    if sigma_perp <= 0.0:
        raise ValueError("sigma_perp must be positive")
    ref = reference_odf if reference_odf is not None else FabricODF.columnar(
        axis=odf.axis, label="reference-columnar")
    model = calibrated_slip_reference(ref, ratio_columnar=ratio_columnar,
                                      convention=convention, tilt_deg=tilt_deg)
    sig_par = model.uniaxial_strength(odf, load_direction(0.0))
    sig_perp_slip = model.uniaxial_strength(odf, load_direction(90.0))
    tau_slip = model.transverse_shear_strength(odf)["shear_strength_Pa"]
    r_odf = sig_par / sig_perp_slip
    t_odf = tau_slip / sig_perp_slip
    return HillTI(
        sigma_parallel=float(sigma_perp * r_odf),
        sigma_perp=float(sigma_perp),
        tau_transverse=float(sigma_perp * t_odf),
        axis=odf.axis,
        label=label,
        metadata={
            "fabric_index_g": odf.anisotropy_index(),
            "fabric_kind": odf.kind,
            "fabric_label": odf.label,
            "ratio_columnar_target": float(ratio_columnar),
            "ratio_from_slip_odf": float(r_odf),
            "tau_over_perp_from_slip": float(t_odf),
            "pyramidal_over_basal_crss": float(model.pyramids_tau_ratio),
            "slip_convention": convention,
            "pyramidal_tilt_deg": float(tilt_deg),
            "closure": "shape from the Schmid slip-system reference; scale from sigma_perp",
        },
    )

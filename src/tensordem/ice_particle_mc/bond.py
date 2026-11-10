"""Mohr-Coulomb bonded contact law for ice particles (micro-to-macro consistency).

Physical problem
----------------
Level-ice and broken-ice DEM models need a cohesive contact law whose *bond*
failure reproduces the macroscopic Mohr-Coulomb (MC) strength envelope
``tau <= c + sigma_n tan(phi)`` of the parent ice.  A bonded particle model
whose bond strength is set independently of the macro parameters cannot do
that: the uniaxial (UCS) peak stress comes out wherever the calibration puts
it.  This module implements the identifiability-preserving choice:

**the bond carries the continuum stress measure of the plane it represents.**

For a bond set of ``M`` bonds in specimen volume ``V`` with bond lengths
``L_b``, the volumetric stress of the lattice is

    sigma = (1/V) * sum_b F_b (x) n_b L_b .

If each bond is assigned the *volume-consistent* representative area

    A_b = V / (M * L_b)                                    (1)

then the bond traction ``F_b / A_b`` is exactly the same stress measure as the
volumetric average, and the bond-level MC criterion

    |F_t| <= c A_b + max(F_n, 0) tan(phi)                  (2)

is satisfied at the *same* instant as the continuum criterion.  With (2) and
(1), a homogeneous, uniformly stressed specimen therefore reaches

    sigma_peak = 2 c cos(phi) / (1 - sin(phi))              (3)

(the classical MC uniaxial strength) exactly, up to the discretisation error of
the bond-orientation set and the stress heterogeneity of a finite specimen.

Derivation of (3) from (2), for the record
------------------------------------------
With a uniaxial macro stress ``sigma`` and plane normal ``n``:
``sigma_n = sigma n_z^2`` and ``tau = sigma |n_z| sqrt(1 - n_z^2)``.  Failure
first occurs on the plane minimising
``h(n_z) = sqrt(x(1-x)) - x tan(phi),  x = n_z^2``, whose minimum sits at
``x = (1 - sin phi)/2`` with ``h_min = (1 - sin phi)/(2 cos phi)``; hence
``sigma = c / h_min = 2 c cos(phi)/(1 - sin(phi))``.  This is asserted in the
unit tests (``test_bond.py``) against the closed form for several ``phi``.

This module is state-free: it exposes the criterion, the strength constants, the
volume-consistent area, the Weibull bond-strength scatter, and the affine
lattice modulus predictor used to score the measured elastic modulus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

__all__ = [
    "MohrCoulombStrength",
    "bond_areas",
    "bond_shear_capacity",
    "bond_failure_mask",
    "mohr_coulomb_ucs",
    "mohr_coulomb_critical_normal_fraction",
    "affine_stiffness_matrix",
    "affine_uniaxial_modulus",
    "weibull_scale_from_median",
    "voigt_basis",
]


@dataclass(frozen=True)
class MohrCoulombStrength:
    """Macroscopic MC cohesion / friction angle plus a tensile cut-off.

    ``tensile_strength_pa`` is the bond tensile stress limit: ice fails in
    tension far below its compressive strength, so the MC cone is truncated by
    ``sigma_n <= sigma_t``.  It is an *input*, never fitted.
    """

    cohesion_pa: float
    friction_angle_deg: float
    tensile_strength_pa: float

    def __post_init__(self) -> None:
        if not (self.cohesion_pa > 0.0 and math.isfinite(self.cohesion_pa)):
            raise ValueError("cohesion_pa must be finite and positive")
        if not (0.0 < self.friction_angle_deg < 90.0):
            raise ValueError("friction_angle_deg must lie in (0, 90)")
        if not (self.tensile_strength_pa > 0.0):
            raise ValueError("tensile_strength_pa must be positive")

    @property
    def phi_rad(self) -> float:
        return math.radians(self.friction_angle_deg)

    @property
    def tan_phi(self) -> float:
        return math.tan(self.phi_rad)

    @property
    def ucs_pa(self) -> float:
        """Classical MC uniaxial compressive strength ``2c cos(phi)/(1-sin(phi))``."""
        return mohr_coulomb_ucs(self.cohesion_pa, self.friction_angle_deg)


def mohr_coulomb_ucs(cohesion_pa: float, friction_angle_deg: float) -> float:
    """``sigma_MC = 2 c cos(phi) / (1 - sin(phi))`` (positive c, 0 < phi < 90)."""
    phi = math.radians(friction_angle_deg)
    return 2.0 * float(cohesion_pa) * math.cos(phi) / (1.0 - math.sin(phi))


def mohr_coulomb_critical_normal_fraction(friction_angle_deg: float) -> float:
    """``n_z^2`` of the critical plane for a uniaxial test: ``(1 - sin phi)/2``."""
    return 0.5 * (1.0 - math.sin(math.radians(friction_angle_deg)))


def bond_areas(volume: float, n_bonds: int, lengths: torch.Tensor) -> torch.Tensor:
    """Volume-consistent representative bond area ``A_b = V / (M L_b)`` (Eq. 1)."""
    if n_bonds <= 0:
        return torch.zeros(0, dtype=torch.float64)
    return (float(volume) / (float(n_bonds) * lengths)).to(torch.float64)


def bond_shear_capacity(normal_force: torch.Tensor, area: torch.Tensor,
                        strength: MohrCoulombStrength,
                        tan_phi: torch.Tensor | None = None) -> torch.Tensor:
    """MC shear capacity ``c A_b + max(F_n, 0) tan(phi)`` (Eq. 2).

    ``normal_force`` is compression positive, so only compression increases the
    shear capacity (tension is handled by the tensile cut-off).
    """
    tan_phi = strength.tan_phi if tan_phi is None else tan_phi
    return (strength.cohesion_pa * area
            + torch.clamp(normal_force, min=0.0) * tan_phi)


def bond_failure_mask(normal_force: torch.Tensor, shear_norm: torch.Tensor,
                      area: torch.Tensor, strength: MohrCoulombStrength,
                      tan_phi: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Failure mask per bond with the two mechanisms kept mutually exclusive.

    Returns ``{"tensile": ..., "shear": ..., "any": ...}``.  ``normal_force`` is
    **compression positive** (the engine convention), so the tensile stress is
    ``-F_n / A_b``.  Tensile failure is checked first; shear failure is the MC
    criterion (2).
    """
    tensile_stress = -normal_force / area
    tensile = tensile_stress > strength.tensile_strength_pa
    shear = shear_norm > bond_shear_capacity(normal_force, area, strength, tan_phi)
    shear = shear & ~tensile
    return {"tensile": tensile, "shear": shear, "any": tensile | shear}


def weibull_scale_from_median(median: float, modulus: float) -> float:
    """Weibull scale so that ``P(X > median) = 0.5`` exactly.

    ``median = scale * ln(2)**(1/m)``  =>  ``scale = median / ln(2)**(1/m)``.
    """
    if modulus <= 0.0:
        raise ValueError("modulus must be positive")
    return float(median) / math.log(2.0) ** (1.0 / modulus)


def voigt_basis() -> list[torch.Tensor]:
    """Symmetric strain basis ``S_p`` with engineering shear (``e_4 = 2 eps_23``)."""
    basis = []
    for i in range(3):
        s = torch.zeros(3, 3, dtype=torch.float64)
        s[i, i] = 1.0
        basis.append(s)
    for i, j in ((1, 2), (0, 2), (0, 1)):
        s = torch.zeros(3, 3, dtype=torch.float64)
        s[i, j] = 0.5
        s[j, i] = 0.5
        basis.append(s)
    return basis


def affine_stiffness_matrix(unit_normals: torch.Tensor, lengths: torch.Tensor,
                            normal_stiffness: torch.Tensor,
                            shear_stiffness: torch.Tensor,
                            volume: float) -> torch.Tensor:
    """Affine (uniform-strain) 6x6 stiffness of a bonded lattice, MPa-free SI.

    Energy of one bond for a uniform strain ``eps``:
    ``U_b = 0.5 L_b^2 [ k_n (n.eps.n)^2 + k_s ( |eps n|^2 - (n.eps.n)^2 ) ]``,
    so the stiffness tensor is the exact second derivative sum used here.
    """
    basis = voigt_basis()
    c = torch.zeros(6, 6, dtype=torch.float64)
    if unit_normals.numel() == 0:
        return c
    a = torch.stack([torch.einsum("mi,ij,mj->m", unit_normals, s, unit_normals)
                     for s in basis])                                  # (6, M)
    b = torch.stack([unit_normals @ s.T for s in basis])                  # (6, M, 3)
    weight = (lengths.square() / float(volume))
    kn = normal_stiffness * weight
    ks = shear_stiffness * weight
    # normal part: sum_b k_n (a_p a_q); shear part: sum_b k_s (b_p.b_q - a_p a_q)
    gram = torch.einsum("pmk,qmk->pqm", b, b)
    a_outer = a[:, None, :] * a[None, :, :]
    c = torch.tensordot((kn * a).reshape(6, -1).T.new_zeros(0, 0), a, dims=0) \
        if False else ((a * kn) @ a.T)
    c = c + torch.tensordot(gram - a_outer, ks, dims=([2], [0]))
    return 0.5 * (c + c.T)


def affine_uniaxial_modulus(stiffness: torch.Tensor, axis: int = 2) -> float:
    """Young's modulus from the affine stiffness with free lateral stress.

    ``E = 1 / (C^-1)[axis, axis]`` -- the axial compliance of the *drained*
    (lateral-stress-free) uniaxial state, which is what the platen test measures.
    """
    regularised = stiffness + torch.eye(6, dtype=torch.float64) * 1e-12
    compliance = torch.linalg.inv(regularised)
    return float(1.0 / compliance[axis, axis])


def normal_stiffness_from_modulus(young_modulus: float, area: torch.Tensor,
                                  lengths: torch.Tensor) -> torch.Tensor:
    """Bond normal stiffness ``k_n = E A_b / L_b`` (bar/bond axial stiffness)."""
    return young_modulus * area / lengths


def shear_stiffness_from_normal(normal_stiffness: torch.Tensor,
                                ratio: float) -> torch.Tensor:
    """Bond shear stiffness ``k_s = ratio * k_n`` (PFC parallel-bond convention)."""
    if not (0.0 < ratio <= 1.0):
        raise ValueError("shear stiffness ratio must be in (0, 1]")
    return float(ratio) * normal_stiffness


def critical_timestep(mass: torch.Tensor, normal_stiffness: torch.Tensor,
                      safety: float = 0.2) -> float:
    """``dt = safety * min sqrt(m_i / sum k_i)`` -- the bonded-lattice bound."""
    if mass.numel() == 0 or normal_stiffness.numel() == 0:
        return float("inf")
    stiffness_sum = torch.zeros_like(mass)
    return float(safety * float((mass / (stiffness_sum + 4.0 * normal_stiffness.mean())).min()
                                .sqrt()))
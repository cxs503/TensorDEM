"""Analytic centre field of the Brazilian disc loaded over a *finite* contact arc.

Why this module exists
----------------------
`docs/dem3d_brazilian_hondros_closure.md` (§5, §6) judged the lattice Brazilian
specimen against the *point*-load Hondros field ``sigma_t = 2P/(pi D t)`` and
reported a 30% miss, leaving one mitigation open: "replace the baseline with a
finite-contact-width analytic reference".  That mitigation is quantified here in
closed form, so the residual can be attributed to the lattice structure instead
of to an idealised load patch.

Derivation (documented so the numbers can be re-derived, not trusted)
--------------------------------------------------------------------
A diametral pair of concentrated forces ``dF`` acting along a material axis
``n`` of an isotropic disc produces, at the centre,

    sigma = (dF / (pi R t)) * (m (x) m - 3 n (x) n),     m = n rotated by 90 deg

because the centre field of a diametral point-load pair is ``+dF/(pi R t)``
across the load axis and ``-3 dF/(pi R t)`` along it (Hondros 1959; ASTM D3967),
and the disc is rotationally symmetric, so the pair may be tilted freely.  A
uniform radial pressure ``p`` applied over two opposed arcs of half-angle
``alpha`` is a continuum of such pairs with ``dF = p t R dtheta`` and total load

    P = 2 p t R sin(alpha).

Integrating the tensor over the two arcs (the two arcs together supply both
members of every pair, hence the factor 2) gives

    sigma_xx(0) = + (P / (pi R t)) * (sin 2a - a) / sin a
    sigma_zz(0) = - (P / (pi R t)) * (sin 2a + a) / sin a
    sigma_zz / sigma_xx = -(sin 2a + a) / (sin 2a - a)

with ``a = alpha``.  Both reduce exactly to the classical point-load values
``+2P/(pi D t)`` and ``-6P/(pi D t)`` (ratio ``-3``) as ``alpha -> 0``, which is
the check asserted in ``tests/test_brazilian_resolution_sweep.py``.

Geometry of the contact patch: a load distributed over an arc of half-angle
``alpha`` on a disc of radius ``R`` touches a flat platen over the chord width
``w = 2 R sin(alpha)``, so the measured contact-arc ratio ``w / D`` equals
``sin(alpha)`` and ``alpha = arcsin(w / D)``.

Scope: this is a closed-form reference field for apparatus verification.  It
carries no ice material calibration and says nothing about a physical tensile
strength.
"""
from __future__ import annotations

import math
from typing import Any

__all__ = [
    "contact_half_angle",
    "point_load_centre_field",
    "finite_arc_centre_field",
    "finite_arc_centre_field_discrete",
    "reference_comparison",
]


def contact_half_angle(contact_arc_ratio: float) -> float:
    """Half-angle of the contact arc from the measured ``w / D`` ratio.

    A flat platen touching an arc of half-angle ``alpha`` of a disc of radius
    ``R`` sees the chord ``w = 2 R sin(alpha)``; since ``D = 2 R`` the measured
    ratio ``w / D`` is ``sin(alpha)``.
    """
    ratio = float(contact_arc_ratio)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
        raise ValueError("contact arc ratio must be finite and in [0, 1)")
    return math.asin(ratio)


def point_load_centre_field(load_N: float, diameter_m: float, thickness_m: float) -> dict[str, float]:
    """Classical Hondros / ASTM D3967 point-load centre field.

    ``sigma_xx = +2P/(pi D t)`` (indirect tension), ``sigma_zz = -6P/(pi D t)``.
    """
    _check_positive(load_N=load_N, diameter_m=diameter_m, thickness_m=thickness_m)
    split = 2.0 * float(load_N) / (math.pi * float(diameter_m) * float(thickness_m))
    return {
        "sigma_xx_Pa": split,
        "sigma_zz_Pa": -3.0 * split,
        "principal_ratio": -3.0,
        "split_stress_Pa": split,
        "contact_half_angle_rad": 0.0,
    }


def finite_arc_centre_field(load_N: float, diameter_m: float,
                            contact_arc_ratio: float,
                            thickness_m: float | None = None,
                            *, half_angle_rad: float | None = None) -> dict[str, float]:
    """Closed-form centre field for a uniform load over an arc.

    Either the measured contact-arc ratio ``w / D`` (preferred: it is what the
    simulation reports) or the half-angle directly may be supplied.  ``P`` is the
    total platen load and ``t`` the specimen thickness; because both stresses
    scale with ``P / (pi R t)``, the principal ratio is load independent.
    """
    if thickness_m is None:
        raise ValueError("thickness_m is required")
    _check_positive(load_N=load_N, diameter_m=diameter_m, thickness_m=thickness_m)
    alpha = (contact_half_angle(contact_arc_ratio) if half_angle_rad is None
             else float(half_angle_rad))
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 0.5 * math.pi:
        raise ValueError("contact half-angle must be finite and in (0, pi/2)")
    radius = 0.5 * float(diameter_m)
    scale = float(load_N) / (math.pi * radius * float(thickness_m))
    sin_a = math.sin(alpha)
    sin_2a = math.sin(2.0 * alpha)
    sigma_xx = scale * (sin_2a - alpha) / sin_a
    sigma_zz = -scale * (sin_2a + alpha) / sin_a
    point = point_load_centre_field(load_N, diameter_m, thickness_m)
    return {
        "sigma_xx_Pa": sigma_xx,
        "sigma_zz_Pa": sigma_zz,
        "principal_ratio": sigma_zz / sigma_xx,
        "split_stress_Pa": point["split_stress_Pa"],
        "contact_half_angle_rad": alpha,
        # Correction the finite patch applies to the point-load baseline.
        "tension_correction_factor": sigma_xx / point["sigma_xx_Pa"],
        "compression_correction_factor": sigma_zz / point["sigma_zz_Pa"],
        "ratio_shift": (sigma_zz / sigma_xx) - point["principal_ratio"],
    }


def finite_arc_centre_field_discrete(load_N: float, diameter_m: float,
                                     thickness_m: float, half_angle_rad: float,
                                     segments: int = 20001) -> dict[str, float]:
    """Independent summation check of :func:`finite_arc_centre_field`.

    Instead of the closed-form integral, the two arcs are diced into ``segments``
    uniform-pressure elements, each contributing its own diametral pair
    ``(dF / (pi R t)) (m (x) m - 3 n (x) n)`` at the centre, and the tensor is
    summed.  Agreement to the discretisation error proves the closed form is the
    integral of the same physical model (and not, say, a mis-normalised load).
    """
    if segments < 3:
        raise ValueError("segments must be >= 3")
    _check_positive(load_N=load_N, diameter_m=diameter_m, thickness_m=thickness_m)
    alpha = float(half_angle_rad)
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 0.5 * math.pi:
        raise ValueError("contact half-angle must be finite and in (0, pi/2)")
    radius = 0.5 * float(diameter_m)
    # Uniform pressure consistent with P = 2 p t R sin(alpha).
    pressure = float(load_N) / (2.0 * float(thickness_m) * radius * math.sin(alpha))
    dtheta = 2.0 * alpha / segments
    d_pair_force = pressure * float(thickness_m) * radius * dtheta
    # Sweeping the upper arc once counts every diametral pair exactly once: each
    # element contributes the pair tensor of magnitude ``dF`` together with its
    # opposed partner on the lower arc.
    sigma_xx = 0.0
    sigma_zz = 0.0
    for k in range(segments):
        theta = -alpha + (k + 0.5) * dtheta
        n_x, n_z = math.sin(theta), math.cos(theta)
        # m = (cos, -sin) is the direction *across* the tilted load axis n = (sin, cos).
        sigma_xx += (d_pair_force / (math.pi * radius * thickness_m)) * (n_z * n_z - 3.0 * n_x * n_x)
        sigma_zz += (d_pair_force / (math.pi * radius * thickness_m)) * (n_x * n_x - 3.0 * n_z * n_z)
    return {
        "sigma_xx_Pa": sigma_xx,
        "sigma_zz_Pa": sigma_zz,
        "principal_ratio": sigma_zz / sigma_xx,
        "contact_half_angle_rad": alpha,
        "segments": segments,
    }


def reference_comparison(measured: dict[str, Any], *, load_N: float, diameter_m: float,
                         thickness_m: float, contact_arc_ratio: float) -> dict[str, Any]:
    """Distances of a measured lattice field from the point-load and finite-arc references."""
    point = point_load_centre_field(load_N, diameter_m, thickness_m)
    finite = finite_arc_centre_field(load_N, diameter_m, contact_arc_ratio, thickness_m)
    sigma_xx = float(measured["centre_sigma_xx_Pa"])
    ratio = float(measured["centre_principal_ratio"])
    return {
        "point_load": {
            **point,
            "sigma_xx_rel_error": (sigma_xx - point["sigma_xx_Pa"]) / abs(point["sigma_xx_Pa"]),
            "principal_ratio_rel_error": (ratio - point["principal_ratio"]) / abs(point["principal_ratio"]),
        },
        "finite_arc": {
            **finite,
            "sigma_xx_rel_error": (sigma_xx - finite["sigma_xx_Pa"]) / abs(finite["sigma_xx_Pa"]),
            "principal_ratio_rel_error": (ratio - finite["principal_ratio"]) / abs(finite["principal_ratio"]),
        },
        # How much of the point-load miss the finite patch can possibly explain.
        "attributable_to_contact_width": {
            "sigma_xx": (finite["sigma_xx_Pa"] - point["sigma_xx_Pa"]) / abs(point["sigma_xx_Pa"]),
            "principal_ratio_rel": (finite["principal_ratio"] - point["principal_ratio"])
            / abs(point["principal_ratio"]),
        },
        "unexplained_sigma_xx_rel_error": (sigma_xx - finite["sigma_xx_Pa"]) / abs(finite["sigma_xx_Pa"]),
    }


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
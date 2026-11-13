"""Rolling-resistance laws transcribed from LIGGGHTS-PUBLIC (OSS DEM reference).

Provenance (fetched 2026-11-13 through the GitHub REST contents API, sha256 of
every byte stored in ``docs/xval/oss_snapshot/PROVENANCE.json``):

* repo ``CFDEMproject/LIGGGHTS-PUBLIC`` (GPL-2.0), head commit
  ``3d5c00f20519e6bb6eb6756f51f1ad36564e649d`` (2024-06-07)
* ``src/rolling_model_cdt.h``    — constant directional torque model
* ``src/rolling_model_epsd2.h``  — elastic-plastic spring-dashpot (EPSD2) model
* ``src/fix_check_timestep_gran.cpp`` — Rayleigh timestep estimator

Why rolling resistance: the round-2 method selection for
``CASE-COLUMN-COLLAPSE`` uses the rolling coefficient ``mu_r`` as the *shape
surrogate* for angular grains (``oss_solutions`` row
``TensorDEM|CASE-COLUMN-COLLAPSE|round2-method-selection``, item 4).  The
projection into a contact law is exactly the degree of freedom the two codes
implement differently, so it has to be cross-validated before any calibrated
``mu_r`` is quoted next to a literature number.

Transcription notes (zero handwriting, every formula has its source line)
-------------------------------------------------------------------------
CDT, ``rolling_model_cdt.h:118-120`` (wall) and ``:145`` (pair)::

    r_torque = mu_r * Fn * reff * (w_rel / |w_rel|)

with ``reff = radi`` for a wall and ``reff = radi*radj/(radi+radj)`` for a pair
(``:98``), ``Fn = kn*deltan`` (``:117``), and the normal (torsion) component of
the *torque* removed unless ``torsionTorque`` is on (``:123-131``).  There is
no contact history: the full Coulomb-level torque appears as soon as the
relative spin is non-zero, and vanishes when it is zero.

EPSD2, ``rolling_model_epsd2.h:186-206``::

    kr      = kt * reff * reff
    wr_tot  = w_rel - (w_rel . n) n          # torsion removed unless torsionTorque
    M       = M_old + kr * wr_tot * dt
    |M|    <= mu_r * |Fn| * reff             # Coulomb cap on the torque

and ``M`` is stored back as contact history (``:217-224``); there is no
dashpot in EPSD2 (``:226`` "dashpot only for the original epsd model").

TensorDEM ``granular_column`` kernel (``kernel.py::pair_contact_forces``)::

    roll     = roll_old + w * dt             # full spin, torsion included
    cap      = mu_r * r * f_n / kr           # cap on the *angle*
    torque   = -kr * roll

so the *family* is EPSD2 (linear rolling spring + Coulomb cap), and the two
implementation differences are (i) the lever arm ``reff``: kernel uses the
particle radius ``r`` for both wall and pair, LIGGGHTS uses ``r`` for a wall but
the reduced radius ``r/2`` for equal spheres; (ii) the kernel does not remove
the normal (torsion/drilling) part of the spin.  ``SpringRolling`` below makes
both switches explicit, which is what the cross-validation measures.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "OVERLAP_TOL",
    "SpringRolling",
    "cdt_torque",
    "hertz_equivalent_shear_modulus",
    "oss_epsd2",
    "rayleigh_timestep",
    "tangential_part",
    "tensordem_kernel",
]

#: below this |spin| the OSS models have no preferred direction (``wrmag > 0``
#: guards in ``rolling_model_cdt.h:110/138``); the guard is reproduced verbatim.
OVERLAP_TOL = 1e-300


def tangential_part(vec: np.ndarray, normal: np.ndarray | None) -> np.ndarray:
    """``w - (w.n) n`` — the torsion-free part of a relative spin.

    ``rolling_model_epsd2.h:173-186`` (and the torque-side removal in
    ``rolling_model_cdt.h:123-131``).  ``normal=None`` returns ``vec``.
    """
    vec = np.asarray(vec, dtype=np.float64)
    if normal is None:
        return vec
    normal = np.asarray(normal, dtype=np.float64)
    if normal.ndim == 1:
        normal = np.broadcast_to(normal, vec.shape)
    proj = np.einsum("ij,ij->i", vec, normal)[:, None] * normal
    return vec - proj


def _mag(vec: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", vec, vec))


def cdt_torque(spin_rel: np.ndarray, fn, reff, mu_r, normal=None,
               torsion_torque: bool = False) -> np.ndarray:
    """Torque **on particle i** of the CDT model (``rolling_model_cdt.h``).

    ``|M| = mu_r |Fn| reff`` along the (torsion-free) relative spin axis, i.e.
    a constant-direction Coulomb-level couple; zero spin gives zero torque.
    """
    w = np.atleast_2d(np.asarray(spin_rel, dtype=np.float64))
    fn = np.broadcast_to(np.asarray(fn, dtype=np.float64), (w.shape[0],))
    reff = np.broadcast_to(np.asarray(reff, dtype=np.float64), (w.shape[0],))
    mu_r = np.broadcast_to(np.asarray(mu_r, dtype=np.float64), (w.shape[0],))
    if not torsion_torque:
        w = tangential_part(w, normal)
    wmag = _mag(w)
    r_torque = np.zeros_like(w)
    live = wmag > 0.0
    r_torque[live] = (w[live] / wmag[live, None]) * (
        (mu_r[live] * np.abs(fn[live]) * reff[live])[:, None])
    if not torsion_torque:
        r_torque = tangential_part(r_torque, normal)   # :148-156
    return -r_torque        # torque on i is -r_torque (:160-162)


class SpringRolling:
    """Linear rolling spring with a Coulomb cap — EPSD2 and the TensorDEM kernel.

    ``torsion=True`` reproduces the ``granular_column`` kernel (full spin in the
    rolling spring); ``torsion=False`` reproduces ``rolling_model_epsd2.h``
    (normal part removed, ``:173-186``).  The caller decides the lever arm
    ``reff``: the kernel passes ``r`` for both contact kinds, LIGGGHTS passes
    ``r`` for a wall and ``radi*radj/(radi+radj)`` (``= r/2`` for equal spheres)
    for a pair (``rolling_model_cdt.h:98``, ``rolling_model_epsd2.h:187``).

    The kernel caps the accumulated *angle* while EPSD2 caps the accumulated
    *torque*; with a linear spring ``torque = kr * angle`` the two are the same
    constraint, and :meth:`step` reproduces the kernel bit for bit when
    ``reff`` is the particle radius (see ``tests/test_xval_oss_rolling.py``).
    """

    def __init__(self, *, name: str, torsion: bool = True) -> None:
        self.name = name
        self.torsion = bool(torsion)
        self.roll = np.zeros((0, 3), dtype=np.float64)

    # -- state ---------------------------------------------------------
    def reset(self, n: int) -> None:
        self.roll = np.zeros((int(n), 3), dtype=np.float64)

    def step(self, spin_rel: np.ndarray, fn, reff, kt, mu_r, dt: float,
             normal=None) -> dict:
        """Advance the rolling spring one step; return torque on i and slip."""
        w = np.atleast_2d(np.asarray(spin_rel, dtype=np.float64))
        n = w.shape[0]
        if self.roll.shape[0] != n:
            self.reset(n)
        fn = np.broadcast_to(np.asarray(fn, dtype=np.float64), (n,))
        reff = np.broadcast_to(np.asarray(reff, dtype=np.float64), (n,))
        mu_r = np.broadcast_to(np.asarray(mu_r, dtype=np.float64), (n,))
        kt = np.broadcast_to(np.asarray(kt, dtype=np.float64), (n,))
        if not self.torsion:
            w = tangential_part(w, normal)
        kr = kt * reff * reff
        roll_new = self.roll + w * dt
        r_norm = _mag(roll_new)
        cap = np.where(kr > 0.0, mu_r * reff * np.maximum(fn, 0.0) / kr, 0.0)
        scale = np.where(r_norm > cap, cap / np.maximum(r_norm, 1e-30), 1.0)
        roll_new = roll_new * scale[:, None]
        torque = -kr[:, None] * roll_new
        slip = w * dt - (roll_new - self.roll)
        stored_energy = 0.5 * kr * np.einsum("ij,ij->i", roll_new, roll_new)
        self.roll = roll_new
        return {"torque": torque, "roll": roll_new, "slip": slip,
                "dissipation": float(-np.einsum("ij,ij->i", torque, slip).sum()),
                "energy": float(stored_energy.sum()), "kr": kr}


def tensordem_kernel() -> SpringRolling:
    """The TensorDEM ``granular_column`` law (full spin, caller passes ``r``)."""
    return SpringRolling(name="tensordem_kernel", torsion=True)


def oss_epsd2() -> SpringRolling:
    """LIGGGHTS ``rolling_model epsd2`` (torsion removed by default)."""
    return SpringRolling(name="oss_epsd2", torsion=False)


def rayleigh_timestep(radius: float, density: float, shear_modulus: float,
                      poisson: float) -> float:
    """Rayleigh-wave timestep of ``fix_check_timestep_gran.cpp:262``.

    ``dt_R = pi r sqrt(rho/G) / (0.1631 nu + 0.8766)`` with
    ``G = Y / (2 (1 + nu))`` (``:261``).
    """
    radius = float(radius)
    density = float(density)
    shear_modulus = float(shear_modulus)
    poisson = float(poisson)
    for name, value in (("radius", radius), ("density", density),
                        ("shear_modulus", shear_modulus)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if not (-0.99 < poisson < 0.5):
        raise ValueError("poisson must lie in (-0.99, 0.5)")
    return math.pi * radius * math.sqrt(density / shear_modulus) / (
        0.1631 * poisson + 0.8766)


def hertz_equivalent_shear_modulus(fn_ref: float, kn: float, radius: float,
                                   poisson: float) -> float:
    """Shear modulus ``G`` of a Hertz contact that matches a linear ``kn``.

    A linear spring has no unique elastic-modulus equivalent, so the mapping is
    declared: the representative overlap is ``delta_ref = fn_ref / kn`` (one
    grain weight on the case's normal stiffness) and the Hertz law is used at
    that overlap, ``fn = (4/3) E* sqrt(R_eff) delta^(3/2)`` with
    ``R_eff = radius/2`` (equal spheres) and ``E* = E / (2 (1 - nu^2))``.
    The assumption travels with every number derived from it.
    """
    fn_ref = float(fn_ref)
    kn = float(kn)
    radius = float(radius)
    if not (fn_ref > 0.0 and kn > 0.0 and radius > 0.0):
        raise ValueError("fn_ref, kn and radius must be positive")
    delta = fn_ref / kn
    r_eff = 0.5 * radius
    e_star = fn_ref / ((4.0 / 3.0) * math.sqrt(r_eff) * delta ** 1.5)
    young = 2.0 * (1.0 - poisson * poisson) * e_star
    return young / (2.0 * (1.0 + poisson))
"""Regression + unit tests for the m5481 Beverloo silo-discharge benchmark.

Three defects are pinned here, all found while attempting to *verify* the
module against the literature it claims (Beverloo 1961, C ~ 0.58, k ~ 1.4):

1. the plate-with-a-hole contact treated the *rim cylinder* as an infinite
   cylinder, so a particle resting on the plate **top** at ``radial ~ a`` was
   pushed sideways by up to ``kn * r`` (~250 N on a 1.3 g particle) and the
   run diverged within a few dozen steps;
2. the "underside" (``z = -t``) face was applied inside the aperture
   (``radial < a``), where there is a hole and therefore no material, and
   without the ``z <= -t`` gate, so particles far above the plate were
   pushed down with ``kn * (r + z + t)``;
3. the tangential surface-velocity term mixed the rotation of the two
   contacting spheres with the wrong sign (``+ (omega_j + omega_i) x r n``
   instead of ``-``), which pumps energy into the spin instead of damping it.

The physical tests are deliberately small so the module stays in the fast CI
path; the literature comparison lives in
``scripts/m6296_silo_beverloo_sweep.py`` + ``docs/silo_discharge_beverloo_verification.md``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular3d_core import ContactState, build_pairs, contact_forces
from tensordem.silo_discharge import (
    SiloCase,
    SiloError,
    _initial_packing,
    _plate_forces,
    beverloo_fit,
    beverloo_prediction,
    simulate_silo,
)

G = 9.81


def _case(**kw) -> SiloCase:
    base = dict(radius=0.005, silo_radius=0.045, fill_height=0.03,
                orifice_diameter=0.03, settle_time=0.01,
                discharge_time=0.02, window=0.004, n_windows=5)
    base.update(kw)
    return SiloCase(**base)


# ---------------------------------------------------------------------------
# 1. plate contact geometry
# ---------------------------------------------------------------------------
def test_plate_top_face_does_not_kick_particles_sideways():
    """A particle resting on the plate top near the rim gets an axial force.

    Regression for defect 1: before the fix this particle (centre at
    ``z = r``, i.e. touching the plate top, at ``radial = a + 0.05 d``)
    received ``|f_xy| ~ kn * (r - |radial - a|)`` ~ 200 N.
    """
    case = _case()
    mat = case.material
    r, a = case.radius, case.orifice_radius
    # resting exactly on the top face, close to the rim
    pos = np.array([[a + 0.1 * r, 0.0, r * 0.999]])
    vel = np.zeros((1, 3))
    f = np.zeros((1, 3))
    _plate_forces(pos, vel, mat, case, f)
    # support force ~ kn * (r - z) = 1e5 * 5e-6 = 0.5 N; *never* ~250 N
    assert f[0, 2] > 0.0, "top face must push up"
    assert abs(f[0, 2]) < 10.0, f"spurious normal force {f[0, 2]:.3g} N"
    assert abs(f[0, 0]) < 1e-2 and abs(f[0, 1]) < 1e-2, (
        "top-face contact must not push radially near the rim: "
        f"f_xy = ({f[0, 0]:.4g}, {f[0, 1]:.4g}) N")


def test_plate_top_face_force_matches_spring_penetration():
    case = _case()
    mat = case.material
    r, a = case.radius, case.orifice_radius
    pen = 1.0e-6
    pos = np.array([[a + 2.0 * r, 0.0, r - pen]])
    f = np.zeros((1, 3))
    _plate_forces(pos, np.zeros((1, 3)), mat, case, f)
    assert f[0, 2] == pytest.approx(mat.kn * pen, rel=1e-9)


def test_aperture_is_open_below_the_plate():
    """Defect 2: below the plate *inside* the aperture is a hole, not a face."""
    case = _case()
    mat = case.material
    r, a, t = case.radius, case.orifice_radius, case.plate_thickness
    # centre inside the aperture, just below the plate bottom -> free fall
    pos = np.array([[0.0, 0.0, -t - 0.5 * r]])
    f = np.zeros((1, 3))
    _plate_forces(pos, np.zeros((1, 3)), mat, case, f)
    assert np.allclose(f, 0.0), f"particle in the hole must be free, got {f}"

    # centre outside the aperture, just below the plate bottom -> pushed down
    pos = np.array([[a + 2.0 * r, 0.0, -t - 0.99 * r]])
    f = np.zeros((1, 3))
    _plate_forces(pos, np.zeros((1, 3)), mat, case, f)
    assert f[0, 2] < 0.0, "underside outside the aperture must push -z"


def test_particle_far_above_plate_feels_nothing():
    """Defect 2 (no z-gate): a particle one fill-height above was pushed down."""
    case = _case()
    mat = case.material
    pos = np.array([[0.0, 0.0, 0.5 * case.fill_height]])
    f = np.zeros((1, 3))
    _plate_forces(pos, np.zeros((1, 3)), mat, case, f)
    assert np.allclose(f, 0.0)


def test_plate_forces_do_not_diverge_on_the_settling_column():
    """The full start-up must stay bounded (defect 1 made it blow up)."""
    case = _case(settle_time=0.02, discharge_time=0.02, window=0.004)
    out = simulate_silo(case, return_history=True)
    pos = out["positions"]
    assert np.isfinite(pos).all()
    assert np.abs(pos).max() < 1.0, "column exploded"


# ---------------------------------------------------------------------------
# 2. tangential surface velocity sign
# ---------------------------------------------------------------------------
def test_tangential_contact_opposes_the_spin_pair():
    """Defect 3: friction must oppose the rotational surface velocity.

    Two overlapping spheres, i at the origin and j at ``+x`` (so ``n = +x``),
    no translation, equal spin ``omega = +z``.  The relative surface velocity
    at the contact is ``-(omega_j + omega_i) x (r n) = -2 omega r y``, so the
    tangential force on j must point ``+y`` (opposing -y motion).
    """
    case = _case()
    mat = case.material
    r = case.radius
    pos = np.array([[0.0, 0.0, 0.0], [2.0 * r - 1.0e-5, 0.0, 0.0]])
    vel = np.zeros((2, 3))
    omega = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    iu, ju = build_pairs(pos, r)
    f, tq = contact_forces(pos, vel, omega, iu, ju, ContactState(), mat,
                           case.timestep)
    overlap = 2.0 * r - (2.0 * r - 1.0e-5)
    assert f[1, 0] == pytest.approx(mat.kn * overlap, rel=1e-6)   # normal, +x
    assert f[1, 1] > 0.0, f"tangential force on j should be +y, got {f[1]}"
    # the tangential force on i is the opposite
    assert f[0, 1] < 0.0


def test_tangential_contact_does_not_inject_energy_into_spin():
    """A pure spin pair must lose rotational energy to friction, never gain."""
    case = _case()
    mat = case.material
    r = case.radius
    pos = np.array([[0.0, 0.0, 0.0], [2.0 * r - 1.0e-5, 0.0, 0.0]])
    vel = np.zeros((2, 3))
    omega = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
    state = ContactState()
    iu, ju = build_pairs(pos, r)
    e0 = 0.5 * mat.inertia * float((omega ** 2).sum())
    for _ in range(50):
        _, tq = contact_forces(pos, vel, omega, iu, ju, state, mat, case.timestep)
        omega = omega + tq * (case.timestep / mat.inertia)
    e1 = 0.5 * mat.inertia * float((omega ** 2).sum())
    assert e1 <= e0 * (1.0 + 1e-9), f"spin energy grew {e0:.3e} -> {e1:.3e}"


# ---------------------------------------------------------------------------
# 3. Beverloo inversion
# ---------------------------------------------------------------------------
def test_beverloo_fit_recovers_exact_coefficients():
    C, k, d, rho_b = 0.58, 1.4, 0.01, 1600.0
    D = np.array([0.03, 0.04, 0.05, 0.06, 0.07])
    W = np.array([beverloo_prediction(C, k, Di, d, rho_b, G) for Di in D])
    fit = beverloo_fit(W, D, d, rho_b, G, k_range=(1.0, 2.0), n_k=401)
    assert fit["C"] == pytest.approx(C, rel=2e-3)
    assert fit["k"] == pytest.approx(k, abs=5e-3)
    assert fit["max_rel_err"] < 5e-3


def test_beverloo_fit_flags_an_inconsistent_exponent():
    """If the data follow D^2.0 instead of (D - k d)^2.5 the fit must be poor."""
    d, rho_b, C = 0.01, 1600.0, 0.58
    D = np.array([0.03, 0.04, 0.05, 0.06, 0.07])
    W = C * rho_b * math.sqrt(G) * D ** 2.0
    fit = beverloo_fit(W, D, d, rho_b, G)
    assert fit["max_rel_err"] > 0.10


def test_beverloo_prediction_zero_below_critical_orifice():
    assert beverloo_prediction(0.58, 1.4, 0.01, 0.01, 1600.0, G) == 0.0


def test_silo_case_validates_orifice_size():
    with pytest.raises(SiloError):
        _case(orifice_diameter=2.0 * 0.005)          # < 3 d
    with pytest.raises(SiloError):
        _case(orifice_diameter=2.0 * 0.045)          # >= silo diameter
    with pytest.raises(SiloError):
        _case(n_windows=4)


# ---------------------------------------------------------------------------
# 4. end-to-end sanity of the physical run
# ---------------------------------------------------------------------------
def test_discharge_is_positive_and_monotone():
    case = _case(settle_time=0.02, discharge_time=0.03, window=0.006)
    out = simulate_silo(case)
    assert out["n_initial"] == len(_initial_packing(case))
    assert out["n_discharged"] >= 1
    assert out["flow_rate_kg_s"] >= 0.0
    assert out["packing_fraction"] == pytest.approx(0.60, abs=0.12)
    assert out["discharged_mass_kg"] == pytest.approx(
        out["n_discharged"] * case.material.mass, rel=1e-12)
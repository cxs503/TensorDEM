"""Tests for the anisotropic plane-stress continuum reference of a Brazilian disc.

The suite pins, in order of decreasing generality:

1. the constitutive helpers (plane-stress reduction, Zener ratio, isotropic
   consistency, single-element affine patch test);
2. the finite-element machinery itself, against the *exact* isotropic point-load
   field ``sigma_xx = +2P/(pi D t)``, ``sigma_zz = -6P/(pi D t)`` and against the
   symmetry of the problem (reaction sums must vanish);
3. the two effects the reference exists to separate -- window averaging and
   constitutive anisotropy -- and the closing statement of the m2543 debt: the
   DEM static centre ratio agrees with the *anisotropic* continuum within 3 %
   while it is more than 15 % away from the isotropic reference.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.anisotropic_continuum import (  # noqa: E402
    HONDROS_SPLIT_FACTOR,
    _gradients,
    _strain_displacement,
    isotropic_plane_stress,
    plane_stress_from_cubic,
    q4_disc_mesh,
    solve_diametral_disc,
    stiffness_matrix_2d,
    zener_anisotropy,
)

# the manifest case `tensordem-dem3d-brazilian-v1 / brazil_elastic_field`
LATTICE = {"c11": 433333.0, "c12": 233333.0, "c44": 233333.0}
CASE = {"diameter": 0.30, "thickness": 0.06, "load": 3.37231}
WINDOW = 0.1 * CASE["diameter"]
# static-relaxation centre field of the DEM lattice, docs/dem3d_brazilian_hondros_closure.md
DEM = {"sigma_xx": 155.58995, "sigma_zz": -313.49622, "ratio": -2.0149}


def _iso():
    return isotropic_plane_stress(270000.0, 0.35)


def _anis():
    return plane_stress_from_cubic(**LATTICE)


def _solve(d, n_radial=15, window=WINDOW, **kw):
    return solve_diametral_disc(
        d, CASE["load"], CASE["diameter"], CASE["thickness"],
        n_radial=n_radial, window_half_width=window, **kw
    )


# --------------------------------------------------------------- constitutive
def test_plane_stress_reduction_matches_the_isotropic_sheet():
    """Reducing an isotropic cubic crystal must reproduce the isotropic sheet."""
    young, poisson = 270000.0, 0.35
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    c11, c12, c44 = lam + 2 * mu, lam, mu
    reduced = plane_stress_from_cubic(c11, c12, c44)
    expected = _iso()
    assert np.allclose(reduced, expected, rtol=1e-12, atol=1e-9)


def test_zener_ratio_of_the_brazilian_lattice():
    zener = zener_anisotropy(LATTICE["c11"], LATTICE["c12"], LATTICE["c44"])
    # 233333 / ((433333 - 233333)/2) = 2.33333; the isotropic-equivalent set is exactly 1
    assert zener == pytest.approx(7.0 / 3.0, rel=2e-6)
    young, poisson = 270000.0, 0.35
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    assert zener_anisotropy(lam + 2 * mu, lam, mu) == pytest.approx(1.0, rel=1e-9)


def test_sheet_stiffness_rejects_unstable_inputs():
    with pytest.raises(ValueError):
        stiffness_matrix_2d(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        stiffness_matrix_2d(1.0, 0.5, 0.0)


def test_single_element_affine_patch():
    """A linear displacement field must produce exactly ``sigma = D eps``."""
    d = _iso()
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    gx, gz, det = _gradients(xy, 0.0, 0.0)
    b = _strain_displacement(gx, gz)
    # xi, eta in [-1, 1] over a unit square: det(J) = (1/2)(1/2) = 1/4
    assert det == pytest.approx(0.25, rel=1e-12)
    ue = np.zeros(8)
    ue[0::2] = 0.01 * xy[:, 0]
    strain = b @ ue
    assert strain == pytest.approx([0.01, 0.0, 0.0], abs=1e-14)
    stress = d @ strain
    assert stress == pytest.approx([d[0, 0] * 0.01, d[1, 0] * 0.01, 0.0], rel=1e-12)


# --------------------------------------------------------- FE vs Hondros limit
def test_mesh_puts_the_load_node_exactly_on_the_rim():
    """The loading diameter must equal D exactly, or the analytic check is moot."""
    mesh = q4_disc_mesh(0.15, 20, 0.06)
    node, weight = mesh.load_nodes("point", axis="z")
    assert weight.sum() == pytest.approx(1.0)
    assert mesh.coords[node[0], 1] == pytest.approx(0.15, abs=1e-12)
    assert abs(mesh.coords[node[0], 0]) < 1e-12


def test_isotropic_disc_reproduces_the_hondros_point_load_field():
    s = _solve(_iso(), n_radial=20)
    assert s.split_stress == pytest.approx(2 * CASE["load"] / (math.pi * CASE["diameter"] * CASE["thickness"]), rel=1e-12)
    assert s.sigma_xx_center / s.split_stress == pytest.approx(1.0, rel=0.01)
    assert s.sigma_zz_center / s.split_stress == pytest.approx(HONDROS_SPLIT_FACTOR, rel=0.01)
    assert s.ratio_center == pytest.approx(HONDROS_SPLIT_FACTOR, rel=0.01)
    assert abs(s.tau_xz_center) < 1e-6 * s.split_stress


def test_symmetry_constraints_carry_no_reaction():
    s = _solve(_iso(), n_radial=15)
    assert abs(s.axial_reaction) / CASE["load"] < 1e-9
    assert abs(s.transverse_reaction) / CASE["load"] < 1e-9


def test_solution_is_mesh_convergent():
    coarse = _solve(_iso(), n_radial=15)
    fine = _solve(_iso(), n_radial=25)
    assert abs(fine.sigma_xx_center / fine.split_stress - 1.0) <= abs(
        coarse.sigma_xx_center / coarse.split_stress - 1.0
    ) + 1e-4


# ------------------------------------------- window averaging and anisotropy
def test_window_average_lowers_the_split_tension():
    """The DEM reports a D/10 box average; that alone moves the centre numbers."""
    centre = _solve(_iso(), n_radial=20, window=0.0)
    windowed = _solve(_iso(), n_radial=20, window=WINDOW)
    assert centre.sigma_xx_window == pytest.approx(centre.sigma_xx_center, rel=1e-12)
    drop = 1.0 - windowed.sigma_xx_window / windowed.sigma_xx_center
    assert 0.02 < drop < 0.10
    assert windowed.ratio_window < centre.ratio_center  # ratio pushed further from -3


def test_anisotropy_moves_the_centre_ratio_towards_the_dem_value():
    iso = _solve(_iso(), n_radial=20)
    anis = _solve(_anis(), n_radial=20)
    assert abs(anis.ratio_center - DEM["ratio"]) < abs(iso.ratio_center - DEM["ratio"])
    assert anis.ratio_window == pytest.approx(-2.0244, rel=0.01)
    assert anis.sigma_xx_window / anis.split_stress == pytest.approx(1.184, rel=0.01)


def test_anisotropic_reference_is_mesh_insensitive():
    coarse = _solve(_anis(), n_radial=15)
    fine = _solve(_anis(), n_radial=25)
    assert coarse.ratio_window == pytest.approx(fine.ratio_window, rel=0.005)
    assert coarse.ratio_center == pytest.approx(fine.ratio_center, rel=0.005)


# --------------------------------------------------- the closed verification debt
def test_dem_static_field_matches_the_anisotropic_reference():
    """m2543 FAIL was a benchmark choice: same DEM data, right benchmark."""
    anis = _solve(_anis(), n_radial=20)
    iso = _solve(_iso(), n_radial=20)

    err_anis = (DEM["ratio"] - anis.ratio_window) / abs(anis.ratio_window)
    err_iso = (DEM["ratio"] - iso.ratio_window) / abs(iso.ratio_window)
    assert abs(err_anis) < 0.03, f"DEM vs anisotropic continuum ratio {100*err_anis:+.2f}%"
    assert abs(err_iso) > 0.15, f"DEM vs isotropic continuum ratio {100*err_iso:+.2f}%"

    err_xx = (DEM["sigma_xx"] / anis.split_stress - anis.sigma_xx_window / anis.split_stress) / abs(
        anis.sigma_xx_window / anis.split_stress
    )
    assert abs(err_xx) < 0.15, f"DEM split stress vs anisotropic continuum {100*err_xx:+.2f}%"


def test_finite_bearing_arc_is_a_second_order_effect():
    point = _solve(_anis(), n_radial=20)
    arc = _solve(_anis(), n_radial=20, load_mode="patch", patch_length=0.067 * CASE["diameter"])
    assert abs(arc.ratio_window - point.ratio_window) / abs(point.ratio_window) < 0.02


# ------------------------------------------------------------- published study
def test_published_reference_json_is_consistent():
    """Guard the artifact written by scripts/brazilian_anisotropic_reference.py."""
    path = ROOT / "artifacts" / "brazilian_anisotropic" / "reference.json"
    if not path.exists():  # pragma: no cover - artifact regenerated on demand
        pytest.skip("reference.json not generated in this checkout")
    import json

    data = json.loads(path.read_text())
    assert data["anisotropic_reference"]["zener"] == pytest.approx(7.0 / 3.0, rel=2e-6)
    budget = {row["benchmark"]: row for row in data["error_budget"]["rows"]}
    assert abs(budget["anisotropic_fe_same_window"]["err_ratio"]) < 0.03
    assert abs(budget["hondros_point_load"]["err_ratio"]) > 0.15
    iso_last = data["machinery_validation"]["rows"][-1]
    assert abs(iso_last["err_ratio"]) < 0.01  # FE machinery vs the exact solution
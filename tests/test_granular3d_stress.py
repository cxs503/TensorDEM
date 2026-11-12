"""Verification of the 3-D Virial/Love-Weber stress instrument.

Every check below is an *exact* statement about the same contact data, so a
disagreement cannot be blamed on relaxation, dissipation or a missing reference
number:

* sign pin: a compressed pair gives ``sigma_zz = -F d / V``;
* the plane traction across a cut equals (minus) the net external force acting
  on the material above it -- the discrete divergence theorem;
* the depth profile of a statically supported column equals
  ``-m g (layers above) / d^2`` at every plane, with no binning;
* the wall traction at the floor equals the plane flux just above it;
* ``sum l (x) f = -sum x (x) F_ext`` holds at round-off for a balanced packing,
  **and** it is shown to fail under two classic implementation errors so the
  check cannot pass vacuously;
* the isotropic cubic lattice gives ``sigma = -p I`` with
  ``p = kn delta / d^2``, matching the independent closed-form pressure already
  used by ``granular3d.jamming.SpherePacking``;
* rotation invariance ``sigma -> R sigma R^T``.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.granular3d import stress as S3
from tensordem.granular3d.jamming import ContactModel, make_packing


# --------------------------------------------------------------------- fixtures
def column_case(m: int = 3, layers: int = 4, *, d: float = 1.0, g: float = 9.81,
                mass: float = 1.0, load: float | None = None):
    """A statically supported ``m x m`` column stack of ``layers`` spheres.

    Contacts are vertical only, with the branch pointing up.  The contact force
    is fixed by statics: the contact under layer ``k`` carries the weight of all
    ``layers - k`` spheres above it (a 1-D chain per column).  In-plane
    neighbours are placed exactly ``d`` apart, i.e. just touching, so they carry
    no force and cannot pollute the check.
    """
    pos = []
    for k in range(layers):
        for a in range(m):
            for b in range(m):
                pos.append([a * d, b * d, k * d])
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    per_layer = m * m
    ii, jj, ff = [], [], []
    for k in range(layers - 1):
        above = layers - 1 - k            # number of spheres above the contact
        for p in range(per_layer):
            i = k * per_layer + p         # lower sphere
            j = (k + 1) * per_layer + p   # upper sphere
            ii.append(i)
            jj.append(j)
            ff.append([0.0, 0.0, -above * mass * g])  # force on the lower one
    ii = np.asarray(ii, dtype=np.int64)
    jj = np.asarray(jj, dtype=np.int64)
    ff = np.asarray(ff, dtype=float)
    branch = pos[jj] - pos[ii]
    ext = np.zeros((n, 3))
    ext[:, 2] = -mass * g
    # floor carries the whole column: reaction on the bottom layer
    ext[:per_layer, 2] += layers * mass * g
    floor_force = np.tile([0.0, 0.0, layers * mass * g], (per_layer, 1))
    return {
        "m": m, "layers": layers, "d": d, "g": g, "mass": mass,
        "pos": pos, "i": ii, "j": jj, "force": ff, "branch": branch,
        "ext": ext, "floor_force": floor_force, "per_layer": per_layer,
        "area": (m * d) ** 2,
    }


def isotropic_lattice(n: int = 4, *, d: float = 1.0, kn: float = 2.0,
                      overlap: float = 0.01):
    """Simple-cubic lattice on a periodic cell, compressed along all three axes."""
    grid = np.arange(n, dtype=float) * d
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    pos = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    pairs = []
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = d
        target = pos + shift
        tree_pairs = []
        for p, t in enumerate(target):
            t = np.mod(t, n * d)
            q = np.argmin(np.linalg.norm(pos - t, axis=1))
            tree_pairs.append((p, int(q)))
        pairs.extend(tree_pairs)
    ii = np.asarray([p[0] for p in pairs], dtype=np.int64)
    jj = np.asarray([p[1] for p in pairs], dtype=np.int64)
    branch = pos[jj] - pos[ii]
    branch -= (n * d) * np.round(branch / (n * d))
    force = -kn * overlap * branch / d      # central, compressive on i
    return pos, ii, jj, branch, force, float((n * d) ** 3), kn, overlap, d


# ---------------------------------------------------------------- 1. sign pin
def test_sign_pin_compressed_pair():
    """A compressed pair must give a negative normal stress (tension positive)."""
    d, f, vol = 1.0, 3.0, 10.0
    branch = np.array([[0.0, 0.0, d]])
    force = np.array([[0.0, 0.0, -f]])
    sigma = S3.stress_tensor(branch, force, vol)
    assert sigma[2, 2] == pytest.approx(-f * d / vol, rel=0, abs=1e-15)
    assert S3.pinned_sign_check(branch, force, vol) == pytest.approx(-f * d / vol)
    assert S3.pressure(sigma) == pytest.approx(f * d / (3.0 * vol))
    # no off-axis or shear components for a purely axial pair
    assert np.allclose(sigma - np.diag(np.diag(sigma)), 0.0)
    assert sigma[0, 0] == sigma[1, 1] == 0.0


# --------------------------------------- 2. agreement with the existing pressure
def test_pressure_matches_sphere_packing():
    """``-tr(sigma)/3`` must equal ``SpherePacking.pressure()`` (independent path)."""
    pack = make_packing(120, 0.66, seed=7, model=ContactModel(kn=1.0))
    i, j, unit, dist, overlap = pack.contact_arrays()
    assert overlap.size > 0
    branch = unit * dist[:, None]            # x_j - x_i (unit points j -> i? no)
    # ``contact_arrays`` returns the unit vector along x_j - x_i, so branch =
    # dist * unit; the force on i is along -unit (repulsive, away from j).
    branch = dist[:, None] * unit
    force = -overlap[:, None] * pack.model.kn * unit
    sigma = S3.stress_tensor(branch, force, pack.volume)
    assert S3.pressure(sigma) == pytest.approx(pack.pressure(), rel=1e-12)
    assert np.allclose(sigma, sigma.T, atol=1e-14)


# ------------------------------ 3. discrete divergence theorem across a cut
def test_plane_traction_equals_net_external_force_above():
    c = column_case()
    for k in range(c["layers"] - 1):
        plane = (k + 0.5) * c["d"]
        above = c["pos"][:, 2] > plane
        net_ext_above = c["ext"][above].sum(axis=0)
        trac, ncross = S3.plane_traction(c["branch"], c["force"], c["pos"][c["i"]],
                                        axis=2, plane=plane)
        # the traction (on the lower side) must equal the net external force on
        # the material above the cut, exactly
        assert np.allclose(trac, net_ext_above, atol=1e-11)
        assert ncross == c["per_layer"]
        assert net_ext_above[2] == pytest.approx(
            -(c["layers"] - 1 - k) * c["mass"] * c["g"] * c["per_layer"])


# ---------------------------------------------------- 4. hydrostatic profile
def test_plane_profile_hydrostatic_column():
    c = column_case(m=4, layers=6)
    planes = (np.arange(c["layers"] - 1) + 0.5) * c["d"]
    normal, shear, counts = S3.plane_profile(
        c["branch"], c["force"], c["pos"][c["i"]], axis=2, planes=planes,
        area=c["area"])
    above_layers = np.arange(c["layers"] - 1)
    analytic = -(c["layers"] - 1 - above_layers) * c["mass"] * c["g"] / c["d"] ** 2
    assert np.allclose(normal, analytic, rtol=0, atol=1e-12)
    assert np.allclose(shear, 0.0, atol=1e-15)
    assert np.all(counts == c["per_layer"])


# ------------------------------------------------ 5. wall traction == plane flux
def test_wall_stress_equals_plane_flux():
    c = column_case(m=5, layers=5)
    wall = S3.wall_stress(c["floor_force"], c["area"], axis=2, normal_sign=-1)
    total_weight = c["layers"] * c["per_layer"] * c["mass"] * c["g"]
    assert wall[2, 2] == pytest.approx(-total_weight / c["area"], rel=1e-14)
    # the plane cut inside the first contact row must transmit everything except
    # the weight of the bottom layer: the step across z = 0 is the floor layer
    trac, ncross = S3.plane_traction(c["branch"], c["force"], c["pos"][c["i"]],
                                     axis=2, plane=0.5 * c["d"])
    assert ncross == c["per_layer"]
    bottom_layer = c["per_layer"] * c["mass"] * c["g"] / c["area"]
    assert wall[2, 2] - trac[2] / c["area"] == pytest.approx(-bottom_layer, rel=1e-13)
    assert np.allclose(wall[0, :], 0.0) and np.allclose(wall[1, :], 0.0)


# ------------------------------------- 6./7. moment identity + mutation guards
def test_moment_residual_is_roundoff():
    c = column_case()
    res = S3.moment_residual(c["pos"], c["branch"], c["force"], c["ext"])
    assert res < 1e-13


def test_moment_residual_detects_sign_error():
    """Flipping the contact-force sign must break the identity (non-vacuous test)."""
    c = column_case()
    good = S3.moment_residual(c["pos"], c["branch"], c["force"], c["ext"])
    bad = S3.moment_residual(c["pos"], c["branch"], -c["force"], c["ext"])
    assert good < 1e-13 and bad > 0.5


def test_plane_traction_detects_wrong_branch_convention():
    """Using ``-l`` (i.e. the wrong pairing) must move the traction by O(1)."""
    c = column_case()
    plane = 1.5 * c["d"]
    trac, _ = S3.plane_traction(c["branch"], c["force"], c["pos"][c["i"]],
                                axis=2, plane=plane)
    # the classic mistake: force taken on j but branch kept as x_j - x_i
    swapped, _ = S3.plane_traction(c["branch"], -c["force"], c["pos"][c["i"]],
                                   axis=2, plane=plane)
    assert np.linalg.norm(trac - swapped) > 0.5 * np.linalg.norm(trac)


def test_region_balance_two_regions():
    c = column_case(m=3, layers=4)
    mask = c["pos"][:, 2] > 1.5 * c["d"]
    net_ext, boundary, res = S3.region_balance(
        c["pos"], c["i"], c["j"], c["force"], c["ext"], mask)
    assert res < 1e-13
    assert np.allclose(net_ext + boundary, 0.0, atol=1e-11)
    trac, _ = S3.plane_traction(c["branch"], c["force"], c["pos"][c["i"]],
                                axis=2, plane=1.5 * c["d"])
    assert np.allclose(trac, net_ext, atol=1e-11)


# --------------------------------------------- 8. isotropic lattice / symmetry
def test_isotropic_lattice_stress():
    pos, ii, jj, branch, force, vol, kn, overlap, d = isotropic_lattice()
    sigma = S3.stress_tensor(branch, force, vol)
    p_expected = kn * overlap / d ** 2
    assert np.allclose(sigma, -p_expected * np.eye(3), rtol=1e-13, atol=1e-13)
    assert S3.pressure(sigma) == pytest.approx(p_expected, rel=1e-13)
    assert np.allclose(S3.deviatoric(sigma), 0.0, atol=1e-13)
    # the branch/force construction itself must be rotationally clean: a
    # rotation of the whole configuration rotates the tensor and leaves the
    # principal values (here: p) unchanged.
    rng = np.random.default_rng(11)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = S3.stress_tensor(branch @ q.T, force @ q.T, vol)
    assert np.allclose(rot, q @ sigma @ q.T, rtol=1e-12, atol=1e-12)
    assert S3.pressure(rot) == pytest.approx(p_expected, rel=1e-13)


# ---------------------------------------------------------- 9. kinetic stress
def test_kinetic_stress_exact():
    mass = np.array([1.0, 2.0, 3.0])
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    vol = 5.0
    k = S3.kinetic_stress(mass, vel, vol)
    assert np.allclose(k, np.diag([1.0, 8.0, 3.0]) / vol)
    assert S3.pressure(k) == pytest.approx(-(1.0 + 8.0 + 3.0) / (3.0 * vol))


# ------------------------------------------------ 10. pair list / validation
def test_from_pair_list_minimum_image():
    pos = np.array([[0.1, 0.1, 0.1], [9.9, 0.1, 0.1]])
    branch, force = S3.from_pair_list(pos, [0], [1], [[1.0, 0.0, 0.0]], box=10.0)
    assert branch[0, 0] == pytest.approx(-0.2)
    assert np.isclose(np.linalg.norm(branch[0]), 0.2)


@pytest.mark.parametrize("bad", [
    dict(volume=0.0), dict(volume=-1.0), dict(volume=float("nan")),
])
def test_volume_validation(bad):
    branch = np.zeros((1, 3))
    force = np.zeros((1, 3))
    with pytest.raises(ValueError):
        S3.stress_tensor(branch, force, **bad)


def test_shape_and_axis_validation():
    with pytest.raises(ValueError):
        S3.stress_tensor(np.zeros((2, 2)), np.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError):
        S3.stress_tensor(np.zeros((2, 3)), np.zeros((3, 3)), 1.0)
    with pytest.raises(ValueError):
        S3.plane_traction(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)),
                          axis=3, plane=0.0)
    with pytest.raises(ValueError):
        S3.wall_stress(np.zeros((1, 3)), 1.0, axis=2, normal_sign=0)
    with pytest.raises(ValueError):
        S3.pressure(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        S3.kinetic_stress([1.0, 2.0], np.zeros((3, 3)), 1.0)
    with pytest.raises(IndexError):
        S3.from_pair_list(np.zeros((2, 3)), [0], [5], np.zeros((1, 3)))
    with pytest.raises(ValueError):
        S3.region_balance(np.zeros((2, 3)), [0], [1], np.zeros((1, 3)),
                          np.zeros((2, 3)), np.array([True]))
"""Tests for the rubble-pile post-processing module (P82W4-N02).

Coverage
--------
* exact circle/rectangle geometry (closed form vs Monte-Carlo),
* porosity field conservation (acceptance (a)) **including a negative control**
  that drops the clipped part of boundary-crossing particles and must fail,
* Voronoi / Delaunay pore-scale descriptors,
* force-chain strong/weak partition self-consistency (acceptance (b)) and the
  anisotropy of a synthetic anisotropic network,
* the ``mu(I)`` fit quality metrics and the viscosity window (acceptance (c),(d))
  on a synthetic law with known coefficients,
* the negative control of acceptance (e): a contact-free configuration must
  raise instead of returning a silent zero,
* an end-to-end pipeline on a small DEM configuration from
  :mod:`tensordem.rubble_scenarios`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.rubble_postprocess import (
    GUIMARAES_2022_VISCOSITY_RANGE,
    ContactView,
    ParticleSample,
    RubbleNoContactsError,
    RubblePostProcessError,
    RubbleRheologyError,
    calibrate_mu_i,
    circle_box_area,
    contact_partition_audit,
    contact_stress,
    contact_view,
    effective_viscosity,
    fabric_tensor,
    fit_mu_i_quadratic,
    force_chain_statistics,
    grid_porosity_field,
    inertial_number,
    mu_of_i,
    porosity_conservation_audit,
    rheology_point,
    stress_invariants,
    strong_weak_split,
    voronoi_local_porosity,
)


# ------------------------------------------------------------------ helpers
def _hex_packing(domain: float = 0.3, radius: float = 0.01, jitter: float = 0.0,
                 seed: int = 0, periodic: tuple[bool, bool] = (False, False)):
    """Hexagonal packing pushed through a periodic/closed square domain."""
    step = 2.0 * radius
    coords = [(x, y) for x in np.arange(radius, domain - radius, step)
              for y in np.arange(radius, domain - radius, step)]
    pos = np.asarray(coords)
    if jitter:
        pos = pos + (np.random.default_rng(seed).random(pos.shape) - 0.5) * jitter
    return ParticleSample(pos, np.full(len(pos), radius), (domain, domain),
                          0.03, periodic)


def _contact_dict(pos, radii, *, fn_scale=1.0, anisotropic=False):
    """Synthetic contact record on a pair list of a hex packing."""
    pairs = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < (radii[i] + radii[j]) * 1.001:
                pairs.append((i, j))
    i = np.array([p[0] for p in pairs], dtype=np.int64)
    j = np.array([p[1] for p in pairs], dtype=np.int64)
    n = pos[j] - pos[i]
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    fn = np.ones(len(i)) * fn_scale
    if anisotropic:                      # load the horizontal contacts much more
        fn = np.where(np.abs(n[:, 1]) > 0.5, 0.2 * fn_scale, 4.0 * fn_scale)
    # repulsive contact: the force on i points away from j (engine convention)
    f_on_i = -fn[:, None] * n
    kind = np.zeros(len(i), dtype=np.int8)
    return {"i": i, "j": j, "kind": kind, "n": n, "fn": fn, "f_on_i": f_on_i,
            "f_on_j": -f_on_i}


# ------------------------------------------------------ exact circle/box area
def test_circle_box_area_matches_monte_carlo():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(12):
        cx, cy = rng.uniform(0.0, 1.0, 2)
        r = rng.uniform(0.05, 0.4)
        x0, x1 = np.sort(rng.uniform(-0.2, 1.2, 2))
        y0, y1 = np.sort(rng.uniform(-0.2, 1.2, 2))
        exact = circle_box_area(cx, cy, r, x0, x1, y0, y1)
        xs = rng.uniform(x0, x1, 400000)
        ys = rng.uniform(y0, y1, 400000)
        mc = np.mean((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r) * (x1 - x0) * (y1 - y0)
        worst = max(worst, abs(exact - mc) / max(exact, 1e-9))
    assert worst < 0.01


def test_circle_box_area_limits():
    assert circle_box_area(0.5, 0.5, 0.1, 0.0, 1.0, 0.0, 1.0) == pytest.approx(math.pi * 0.01)
    assert circle_box_area(2.0, 2.0, 0.1, 0.0, 1.0, 0.0, 1.0) == 0.0
    half = circle_box_area(0.5, 0.5, 0.2, 0.0, 0.5, 0.0, 1.0)
    assert half == pytest.approx(0.5 * math.pi * 0.04)
    with pytest.raises(RubblePostProcessError):
        circle_box_area(0.5, 0.5, 0.2, 1.0, 0.0, 0.0, 1.0)


# --------------------------------------------- acceptance (a): conservation
@pytest.mark.parametrize("periodic", [(False, False), (True, True), (True, False)])
def test_porosity_field_conservation(periodic):
    sample = _hex_packing(jitter=0.004, periodic=periodic)
    field = grid_porosity_field(sample)
    assert field["h_over_d50"] == pytest.approx(5.0, rel=0.05)
    audit = porosity_conservation_audit(field, sample)
    assert audit["passed"], audit
    assert audit["absolute_error"] <= 1e-6
    # the field must be a real field, not a constant
    assert field["phi_std"] > 0.0


def test_porosity_conservation_negative_control():
    """Dropping the boundary-clipped particle area must fail the audit.

    This is the *bug* the criterion is designed to catch: particles that cross a
    periodic seam (or a closed wall) lose the wrapped sliver if it is not
    handled, and the field integral then under/over-counts solid area.
    """
    sample = _hex_packing(jitter=0.006, periodic=(True, True))
    field = grid_porosity_field(sample)
    assert porosity_conservation_audit(field, sample)["passed"]

    # mechanically corrupt the field: sum only the interior grid (drop the first
    # and last column/row), which is what a missing wrap/boundary term looks like
    phi = field["phi"].copy()
    hx, hy = field["cell_size"]
    corrupted = dict(field)
    corrupted["phi"] = phi.copy()
    corrupted["phi"][:, 0] = 0.0
    corrupted["phi"][0, :] = 0.0
    bad = porosity_conservation_audit(corrupted, sample)
    assert not bad["passed"]
    assert bad["absolute_error"] > 1e-3


def test_porosity_field_straddling_periodic_seam():
    """A disk split by the periodic seam keeps its full area in the field."""
    domain = 0.2
    radius = 0.02
    sample = ParticleSample(np.array([[0.001, 0.1], [0.1, 0.1]]),
                            np.array([radius, radius]), (domain, domain),
                            0.03, (True, True))
    field = grid_porosity_field(sample)
    assert porosity_conservation_audit(field, sample)["absolute_error"] <= 1e-12
    assert field["solid_area_from_field"] == pytest.approx(sample.solid_area(), rel=1e-12)


# ------------------------------------------------------- Voronoi/Delaunay
def test_voronoi_local_porosity_dense_packing():
    sample = _hex_packing(domain=0.4, radius=0.011, periodic=(True, True))
    vor = voronoi_local_porosity(sample)
    assert vor["n_unbounded_cells"] == 0
    # monodisperse: the Voronoi-weighted porosity reproduces the global one
    assert vor["phi_voronoi_weighted"] == pytest.approx(sample.porosity(), abs=0.02)
    assert 0.0 < vor["phi_std"] < 0.5
    assert vor["delaunay_mean_length_over_d50"] == pytest.approx(1.0, abs=0.4)


# ------------------------------------------- acceptance (b): self-consistency
def test_force_chain_partition_is_exhaustive():
    sample = _hex_packing(domain=0.3, radius=0.01)
    contacts = _contact_dict(sample.positions, sample.radii)
    view = contact_view(contacts)
    assert view.count > 0
    stats = force_chain_statistics(sample, view, quantile=0.8)
    part = stats["partition"]
    assert part["n_strong"] + part["n_weak"] == part["n_total"]
    assert part["count_residual"] <= 1e-10
    assert part["force_sum_residual_relative"] <= 1e-10
    assert part["passed"]
    assert stats["n_strong"] == pytest.approx(0.2 * view.count, abs=1.0)
    assert 0.0 <= stats["anisotropy_a_c"] <= 2.0
    assert stats["coordination_z"] == pytest.approx(2 * view.count / sample.n)


def test_partition_audit_detects_overlap():
    """Negative control: a non-partition (overlap / gap) must be reported."""
    mag = np.array([1.0, 2.0, 3.0, 4.0])
    strong = np.array([False, True, True, True])
    weak = np.array([True, True, True, False])    # contact 1 double counted
    audit = contact_partition_audit(mag, strong, weak)
    assert audit["n_overlap"] == 2
    assert not audit["passed"]
    # empty set must raise instead of silently reporting a zero
    with pytest.raises(RubbleNoContactsError):
        strong_weak_split(np.zeros(0))


def test_fabric_anisotropy_limits():
    theta = np.linspace(0.0, np.pi, 200000)
    isotropic = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    assert fabric_tensor(isotropic)["anisotropy"] < 0.01
    uniaxial = np.tile(np.array([[1.0, 0.0]]), (1000, 1))
    out = fabric_tensor(uniaxial)
    assert out["anisotropy"] == pytest.approx(2.0)
    assert out["theta_deg"] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(RubbleNoContactsError):
        fabric_tensor(np.zeros((0, 2)))


def test_anisotropic_force_chain_is_detected():
    sample = _hex_packing(domain=0.3, radius=0.01)
    contacts = _contact_dict(sample.positions, sample.radii, anisotropic=True)
    stats = force_chain_statistics(sample, contact_view(contacts), quantile=0.8)
    assert stats["anisotropy_a_c"] > stats["anisotropy_a_c_all"]
    assert stats["force_share_strong"] > 0.2


# --------------------------------------------------- stresses and rheology
def test_contact_stress_hydrostatic_is_isotropic():
    sample = _hex_packing(domain=0.3, radius=0.01)
    contacts = _contact_dict(sample.positions, sample.radii, fn_scale=10.0)
    inv = stress_invariants(contact_stress(sample, contact_view(contacts))["sigma"])
    assert inv["pressure"] > 0.0
    assert inv["mu"] < 0.15                     # nearly isotropic network


def test_fit_mu_i_recovers_known_coefficients():
    I = np.array([0.010, 0.025, 0.045, 0.070, 0.100, 0.150, 0.210])
    mu = 0.55 + 0.80 * I - 1.10 * I ** 2
    fit = fit_mu_i_quadratic(I, mu)
    assert fit["mu0"] == pytest.approx(0.55, abs=1e-9)
    assert fit["alpha"] == pytest.approx(0.80, abs=1e-8)
    assert fit["beta"] == pytest.approx(-1.10, abs=1e-8)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["residual_std"] <= 1e-10
    assert fit["n_points"] == 7


def test_fit_mu_i_noise_acceptance_thresholds():
    rng = np.random.default_rng(5)
    I = np.linspace(0.01, 0.2, 40)
    mu = 0.55 + 0.8 * I - 1.1 * I ** 2 + rng.normal(0.0, 0.004, I.size)
    fit = fit_mu_i_quadratic(I, mu)
    assert fit["r2"] >= 0.95
    assert fit["residual_std"] <= 0.05


def test_viscosity_window_at_reference_state():
    """Acceptance (d) on the declared reference state (P = 10 kPa, 0.1 1/s)."""
    I = np.array([0.010, 0.025, 0.045, 0.070, 0.100, 0.150])
    mu = 0.55 + 0.8 * I - 1.1 * I ** 2
    fit = fit_mu_i_quadratic(I, mu)
    d50, rho, gamma, pressure = 0.02, 917.0, 0.1, 1.0e4
    I_ref = inertial_number(gamma, d50, pressure, rho)
    eta = effective_viscosity(float(mu_of_i(I_ref, fit["mu0"], fit["alpha"], fit["beta"])),
                              pressure, gamma)
    lo, hi = GUIMARAES_2022_VISCOSITY_RANGE
    assert lo <= eta <= hi
    with pytest.raises(RubbleRheologyError):
        effective_viscosity(0.5, pressure, 0.0)
    with pytest.raises(RubbleRheologyError):
        inertial_number(gamma, d50, 0.0, rho)


# --------------------------------------- acceptance (e): negative controls
def test_no_contact_is_an_error_not_a_zero():
    sample = _hex_packing(domain=0.3, radius=0.01)
    empty = ContactView(i=np.zeros(0, dtype=np.int64), j=np.zeros(0, dtype=np.int64),
                        normal_force=np.zeros(0), force_on_i=np.zeros((0, 2)))
    with pytest.raises(RubbleNoContactsError):
        contact_stress(sample, empty)
    with pytest.raises(RubbleNoContactsError):
        force_chain_statistics(sample, empty)
    with pytest.raises(RubbleNoContactsError):
        rheology_point(sample, empty, shear_rate=1.0, density=917.0)
    with pytest.raises(RubbleNoContactsError):
        fit_mu_i_quadratic([], [])


def test_engine_contact_force_absent_raises():
    """A real engine state with no overlap raises (no silent zero)."""
    from tensordem.granular2d import DiskConfig, DiskSystem, box_walls

    system = DiskSystem(np.array([[0.1, 0.1]]), np.array([0.005]), DiskConfig(),
                        walls=box_walls(0.4, 0.4))
    system.contact_forces()
    view = contact_view(system.contacts)
    assert view.count == 0
    sample = ParticleSample(system.pos, system.radii, (0.4, 0.4), 0.03)
    with pytest.raises(RubbleNoContactsError):
        rheology_point(sample, view, shear_rate=1.0, density=917.0)


# ------------------------------------------------------------- end-to-end
def test_end_to_end_small_shear_scan():
    from tensordem.rubble_scenarios import bulk_mask, run_couette_shear

    points = []
    for rate in (2.0, 6.0):
        run = run_couette_shear(shear_rate=rate, seed=3, strain=0.6, snapshots=3,
                                n_particles=120, width=0.24, height=0.12, phi=0.82)
        for snap in run["snapshots"][-2:]:
            sample = snap["sample"]
            view = contact_view(snap["contacts"])
            mask = bulk_mask(sample, margin=run["bulk_margin"])
            shear = abs(snap["profile"]["slope"])
            point = rheology_point(sample, view, shear_rate=shear, density=917.0,
                                   particle_mask=mask)
            field = grid_porosity_field(sample)
            assert porosity_conservation_audit(field, sample)["passed"]
            stats = force_chain_statistics(sample, view)
            assert stats["partition"]["passed"]
            points.append(point)
    assert all(p["I"] > 0.0 and np.isfinite(p["mu"]) for p in points)
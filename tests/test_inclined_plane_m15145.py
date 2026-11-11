"""Tests for the chute-flow (inclined plane) reproduction: Silbert et al. 2001.

Covers the two additive kernel capabilities this needs (periodic minimum-image
neighbour search / pair forces in :mod:`tensordem.granular3d_core`) and the
:mod:`tensordem.inclined_plane` model itself (geometry, gravity decomposition,
short dynamics, profile fits).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular3d_core import (
    ContactState,
    Material3D,
    build_pairs,
    contact_forces,
    minimum_image,
)
from tensordem.inclined_plane import (
    InclinedPlaneCase,
    build_chute,
    density_velocity_profile,
    fit_bagnold_exponent,
    flow_metrics,
    simulate_chute,
)

BOX = (10.0, math.inf, 6.0)


class TestPeriodicKernel:
    def test_minimum_image_wraps_periodic_axes_only(self):
        d = np.array([[9.0, 3.0, -5.5], [0.5, 2.0, 1.0]])
        out = minimum_image(d, BOX)
        assert out[0, 0] == pytest.approx(-1.0)     # 9 -> 9-10
        assert out[0, 1] == pytest.approx(3.0)      # y open -> untouched
        assert out[0, 2] == pytest.approx(0.5)      # -5.5 -> +0.5
        assert out[1, 0] == pytest.approx(0.5)

    def test_build_pairs_finds_contacts_across_seam(self):
        r = 0.5
        pos = np.array([[0.05, 1.0, 1.0], [10.0 - 0.05, 1.0, 1.0]])
        iu, ju = build_pairs(pos, r)
        assert len(iu) == 0, "open box must not pair across the periodic seam"
        iu, ju = build_pairs(pos, r, box=BOX)
        assert len(iu) == 1 and (iu[0], ju[0]) == (0, 1)
        dvec = minimum_image(pos[ju] - pos[iu], BOX)
        assert np.linalg.norm(dvec[0]) == pytest.approx(0.1, abs=1e-12)

    def test_seam_contact_force_matches_open_box(self):
        mat = Material3D(radius=0.5, mu=0.5, restitution=0.88)
        state = ContactState()
        pos = np.array([[0.05, 1.0, 1.0], [10.0 - 0.05, 1.0, 1.0]])
        iu, ju = build_pairs(pos, 0.5, box=BOX)
        f, _ = contact_forces(pos, np.zeros((2, 3)), np.zeros((2, 3)),
                              iu, ju, state, mat, 1e-4, box=BOX)
        # overlap = 1.0 - 0.1 = 0.9 -> f_n = k_n * 0.9 along the wrapped normal
        assert np.linalg.norm(f[0]) == pytest.approx(0.9 * mat.kn, rel=1e-9)
        # across the seam particle 1 sits to the *left* of particle 0
        assert f[0, 0] > 0.0 and f[1, 0] < 0.0
        assert f.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-9)

    def test_periodic_pair_force_is_newtonian(self):
        mat = Material3D(radius=0.5, mu=0.3)
        rng = np.random.default_rng(3)
        pos = rng.uniform([0, 0, 0], [10.0, 4.0, 6.0], size=(40, 3))
        vel = rng.normal(scale=0.1, size=(40, 3))
        omega = rng.normal(scale=0.1, size=(40, 3))
        iu, ju = build_pairs(pos, 0.5, box=BOX)
        f, t = contact_forces(pos, vel, omega, iu, ju, ContactState(), mat, 1e-4,
                              box=BOX)
        assert f.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-9)
        d = minimum_image(pos[ju] - pos[iu], BOX)
        assert np.all(np.linalg.norm(d, axis=1) <= 1.0 + 1e-12)


class TestChuteGeometry:
    def test_particle_count_and_base(self):
        case = InclinedPlaneCase(height=6.0, lx=8.0, lz=4.0)
        st = build_chute(case)
        assert int(st["mobile"].sum()) == case.n_mobile == 192
        assert st["n_base"] > 0
        # the glued base sits on the y=0 plane, mobile particles above it
        assert np.allclose(st["pos"][: st["n_base"], 1], case.radius)
        assert st["pos"][st["mobile"], 1].min() > case.radius

    def test_no_initial_overlap_beyond_one_percent(self):
        case = InclinedPlaneCase(height=6.0, lx=8.0, lz=4.0, packing_jitter=0.0)
        st = build_chute(case)
        iu, ju = build_pairs(st["pos"], case.radius * 1.001, box=case.box)
        d = np.linalg.norm(minimum_image(st["pos"][ju] - st["pos"][iu], case.box),
                           axis=1)
        overlap = np.maximum(0.0, 2 * case.radius - d).max()
        assert overlap < 0.02 * 2 * case.radius

    def test_gravity_decomposition(self):
        case = InclinedPlaneCase(theta_deg=30.0, gravity=1.0)
        g = case.gravity_vector
        assert g[0] == pytest.approx(0.5)
        assert g[1] == pytest.approx(-math.sqrt(3.0) / 2.0)
        assert g[2] == 0.0
        # steeper plane -> larger driving component, smaller normal load
        steep = InclinedPlaneCase(theta_deg=45.0).gravity_vector
        assert steep[0] > g[0] and steep[1] > g[1]

    def test_pile_height_matches_H_times_density(self):
        case = InclinedPlaneCase(height=8.0, lx=8.0, lz=4.0, packing_gap=0.0)
        st = build_chute(case)
        y = st["pos"][st["mobile"], 1]
        h_meas = y.max() + case.radius - case.radius
        # H = N d^2 / A;  geometric height h ~ H / phi_pack
        phi = math.pi / 6.0 / (1.0 ** 3) * 1.0
        assert case.n_mobile == 256
        assert h_meas > 8.0 and h_meas < 8.0 / 0.60


class TestChuteDynamics:
    def test_short_run_is_stable_and_flows_downhill(self):
        case = InclinedPlaneCase(theta_deg=26.0, height=6.0, lx=8.0, lz=4.0,
                                 kn=1.0e4)
        res = simulate_chute(case, t_end=1.5, steady_frac=0.5,
                             sample_every=20, bin_width=1.0)
        assert np.isfinite(res["metrics"]["ke_steady"])
        assert res["n_particles"] == case.n_mobile
        # gravity drives the slab down-slope: positive mean v_x by the end
        assert res["history"][-1]["vx_mean"] > 0.0
        # ... and it does not sink through the glued base
        assert res["history"][-1]["y_p95"] > case.radius
        assert res["steps"] > 0 and res["wall_seconds"] >= 0.0

    def test_profile_shapes(self):
        case = InclinedPlaneCase(theta_deg=26.0, height=6.0, lx=8.0, lz=4.0,
                                 kn=1.0e4)
        res = simulate_chute(case, t_end=1.5, steady_frac=0.5, sample_every=20)
        prof = res["profiles"]
        assert len(prof["y_mid"]) == len(prof["phi"]) == len(prof["vx"])
        phi = np.asarray(prof["phi"])
        vx = np.asarray(prof["vx"])
        assert phi.max() <= 0.75 + 1e-9          # cannot exceed hcp packing
        assert np.all(phi >= 0.0)
        bulk = density_velocity_profile(res)
        assert 0.3 < bulk["phi_bulk"] < 0.75
        assert np.nanmax(vx) > 0.0


class TestProfileFits:
    def test_bagnold_exponent_recovered_from_synthetic_profile(self):
        y = np.arange(0.5, 20.5, 1.0)
        h, y0 = 18.0, 1.0
        shape = 1.0 - np.clip(1.0 - (y - y0) / (h - y0), 0.0, 1.0) ** 1.5
        vx = 12.0 * shape
        prof = {"y_mid": y.tolist(),
                "phi": np.where(y < 17.0, 0.58, 0.1).tolist(),
                "vx": vx.tolist(),
                "tau_xy": (0.4 * np.clip(18.0 - y, 0.0, None)).tolist()}
        case = InclinedPlaneCase()
        fit = fit_bagnold_exponent(prof, case)
        assert fit["p"] == pytest.approx(1.5, abs=0.06)
        assert fit["r2"] > 0.995
        assert fit["near_base"]["q"] > 0.8      # linear close to the base

    def test_flow_metrics_flags_stopped_flow(self):
        hist = [{"t": 0.1 * i, "ke_per_particle": 5.0 * math.exp(-i / 3.0),
                 "vx_mean": 0.0, "vx_max": 0.0, "y_mean": 5.0, "y_p95": 6.0}
                for i in range(1, 40)]
        m = flow_metrics(hist)
        assert m["flowing"] is False
        steady = [{"t": 0.1 * i, "ke_per_particle": 5.0, "vx_mean": 1.0,
                   "vx_max": 3.0, "y_mean": 5.0, "y_p95": 6.0}
                  for i in range(1, 40)]
        assert flow_metrics(steady)["flowing"] is True
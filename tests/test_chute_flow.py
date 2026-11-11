"""Unit tests for the chute cross-validation infrastructure.

These check the *infrastructure* invariants that make a three-way comparison
meaningful; they do not replace the (unfinished) production runs.
"""
import math

import numpy as np

from tensordem.chute_flow import analysis
from tensordem.chute_flow.engine import ChuteEngine, periodic_pair_candidates
from tensordem.chute_flow.spec import (DEFAULT_CASE, ChuteCase, make_base,
                                       make_initial_particles, write_case_spec)
from tensordem.granular3d_core import Material3D, build_pairs, contact_forces
from tensordem.granular_column.kernel import PairSpringState, pair_contact_forces


def _case(**kw):
    p = dict(DEFAULT_CASE)
    p.update(kw)
    return ChuteCase.from_dict(p)


def test_base_and_slab_are_periodic_and_counted():
    c = _case(lx_d=8.0, ly_d=8.0, n_target=400)
    base = make_base(c)
    init = make_initial_particles(c)
    # every coordinate lies inside the periodic cell
    assert (base[:, 0] >= 0).all() and (base[:, 0] < c.lx + 1e-12).all()
    assert (base[:, 1] >= 0).all() and (base[:, 1] < c.ly + 1e-12).all()
    assert (init[:, 0] >= 0).all() and (init[:, 0] < c.lx + 1e-12).all()
    # base spheres sit on the datum plane and touch the top of the roughness
    assert np.allclose(base[:, 2], c.base_center_z_d * c.d)
    assert math.isclose(c.z_wall, (c.base_center_z_d + 0.5) * c.d)
    # counts are integer multiples of a commensurate layer
    per_layer = int(round(c.lx / (c.init_gap * c.d))) * int(round(c.ly / (c.init_gap * c.d)))
    assert init.shape[0] % per_layer == 0
    # all particles are above the base
    assert (init[:, 2] > c.z_wall).all()


def test_min_image_pairs_reproduce_wrapped_geometry():
    """box= minimum image must equal explicitly shifted (open) geometry."""
    c = _case(lx_d=4.0, ly_d=4.0)
    mat = Material3D(radius=c.radius, density=c.density, mu=c.mu,
                     mu_rolling=0.0, restitution=c.restitution, kn=c.kn,
                     kt_ratio=c.kt_ratio, gravity=c.gravity)
    r = c.radius
    # two grains that only touch across the periodic x boundary
    pos = np.array([[0.2 * r, c.ly / 2, 0.01], [c.lx - 0.2 * r, c.ly / 2, 0.01]])
    iu, ju = periodic_pair_candidates(pos, 2, (c.lx, c.ly), 2.0 * r)
    assert iu.size == 1 and ju.size == 1
    vel = np.array([[0.0, 0.0, -0.01], [0.0, 0.0, 0.01]])
    om = np.zeros_like(pos)
    st_a, st_b = PairSpringState(), PairSpringState()
    a = pair_contact_forces(pos, vel, om, iu, ju, st_a, mat, c.dt,
                            box=(c.lx, c.ly))
    shifted = pos.copy()
    shifted[1, 0] -= c.lx            # bring j next to i, open geometry
    b = pair_contact_forces(shifted, vel, om, iu, ju, st_b, mat, c.dt)
    # normal force magnitude must match; the direction differs by the wrap
    assert np.isclose(np.linalg.norm(a["force"][0]), np.linalg.norm(b["force"][0]),
                      rtol=1e-10, atol=1e-12)
    assert a["n_contacts"] == b["n_contacts"] == 1


def test_kernel_box_kwarg_is_backward_compatible():
    """box=None must reproduce granular3d_core.contact_forces bit for bit."""
    c = _case()
    mat = Material3D(radius=c.radius, density=c.density, mu=c.mu,
                     mu_rolling=0.0, restitution=c.restitution, kn=c.kn,
                     kt_ratio=c.kt_ratio, gravity=c.gravity)
    rng = np.random.default_rng(3)
    pos = np.array([[0, 0, 0], [1.9 * c.radius, 0, 0], [0.0, 1.8 * c.radius, 0.0]])
    pos += rng.normal(scale=1e-6, size=pos.shape)
    vel = rng.normal(scale=1e-3, size=pos.shape)
    om = rng.normal(scale=1e-1, size=pos.shape)
    iu, ju = build_pairs(pos, c.radius)
    f_ref, t_ref = contact_forces(pos, vel, om, iu, ju, _DictState(), mat, c.dt)
    res = pair_contact_forces(pos, vel, om, iu, ju, PairSpringState(), mat, c.dt,
                              contact_mode="reference")
    assert np.allclose(res["force"], f_ref, atol=1e-12)
    assert np.allclose(res["torque"], t_ref, atol=1e-12)


class _DictState:
    """Adapter exposing the dict-based state of granular3d_core."""

    def __init__(self):
        from tensordem.granular3d_core import ContactState
        self._s = ContactState()

    def __getattr__(self, item):
        return getattr(self._s, item)


def test_bagnold_fit_recovers_exponent():
    c = _case()
    h_d = 10.0
    z = np.arange(0.25, h_d, 0.5)
    C = 0.8 * c.v_ref
    vx = C * (h_d ** 1.5 - np.maximum(h_d - z, 0.0) ** 1.5)
    fit = analysis.bagnold_fit(z * c.d, vx, d=c.d, z_wall=0.0, h_d=h_d)
    assert abs(fit["p"] - 1.5) < 0.02
    assert fit["r2"] > 0.999
    assert abs(fit["C"] / c.v_ref - 0.8) < 1e-2


def test_classify_and_h_stop_bracket():
    assert analysis.classify_run(v_rms_over_vref=0.005, vx_mean_over_vref=0.0) == "arrest"
    assert analysis.classify_run(v_rms_over_vref=0.5, vx_mean_over_vref=0.4) == "flow"
    recs = [
        {"theta_deg": 18.0, "h_over_d": 1.0, "state": "arrest"},
        {"theta_deg": 18.0, "h_over_d": 2.0, "state": "flow"},
    ]
    out = analysis.h_stop_from_scan(recs)
    assert math.isclose(out[18.0]["h_stop_over_d"], 1.5)
    assert out[18.0]["bracket_resolved"] is True


def test_engine_runs_and_stays_stable():
    c = _case(lx_d=5.0, ly_d=5.0, n_target=120, theta_deg=24.0)
    base = make_base(c)
    init = make_initial_particles(c)
    eng = ChuteEngine(c, init, base)
    eng.run(40)
    diag = eng.diagnostics()
    assert np.isfinite(diag["v_rms_over_vref"])
    assert diag["max_overlap_over_d"] < 0.05
    assert diag["steps"] == 40


def test_case_spec_manifest_has_hashes(tmp_path):
    c = _case(lx_d=5.0, ly_d=5.0, n_target=120)
    man = write_case_spec(c, tmp_path)
    assert man["n_particles"] > 0
    assert len(man["files"]["base_spheres.csv"]["sha256"]) == 64
    assert set(man["sampling"]) == {"z_binning", "profile_average", "thickness",
                                    "bagnold_fit"}

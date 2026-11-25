"""Gates for the disordered (settle-then-release) column wave (m20811)."""
from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pytest

from tensordem.granular_column.pluviation import (EQUILIBRIUM_OVERLAP_FRAC,
                                                  RelaxCase,
                                                  cylinder_wall_contact_forces,
                                                  disorder_metrics,
                                                  prepare_disordered_column,
                                                  settle_report)
from tensordem.granular_column.axisym import (AxisymCase, build_axisymmetric_column,
                                             simulate_axisymmetric_column)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "docs", "artifacts",
                      "column_collapse_disordered_m20811")


def _material(radius=0.03):
    case = RelaxCase(radius=radius)
    return case.material()


# --- cylindrical side wall -------------------------------------------------

def test_wall_force_is_zero_outside_the_container():
    mat = _material()
    pos = np.array([[0.10, 0.0, 0.03]])
    out = cylinder_wall_contact_forces(pos, np.zeros((1, 3)), np.zeros((1, 3)),
                                       mat, 4.8e-4, wall_radius=0.24)
    assert out["n_contacts"] == 0
    assert np.allclose(out["force"], 0.0)


def test_wall_force_points_inward_and_scales_with_penetration():
    mat = _material()
    pos = np.array([[0.215, 0.0, 0.03]])
    pen = 0.215 + 0.03 - 0.24
    out = cylinder_wall_contact_forces(pos, np.zeros((1, 3)), np.zeros((1, 3)),
                                       mat, 4.8e-4, wall_radius=0.24,
                                       contact_mode="one_sided")
    assert out["n_contacts"] == 1
    assert out["force"][0, 0] == pytest.approx(-mat.kn * pen, rel=1e-9)
    assert out["force"][0, 1] == pytest.approx(0.0, abs=1e-12)


def test_wall_rejects_invalid_radius():
    mat = _material()
    pos = np.array([[0.1, 0.0, 0.03]])
    with pytest.raises(ValueError):
        cylinder_wall_contact_forces(pos, np.zeros((1, 3)), np.zeros((1, 3)),
                                     mat, 4.8e-4, wall_radius=0.0)


# --- equilibrium gate ------------------------------------------------------

def test_settle_report_flags_overlap_and_aspect():
    r = 0.03
    pos = np.array([[0.0, 0.0, r], [2.0 * r - 1e-4, 0.0, r], [0.0, 0.0, 3 * r]])
    rep = settle_report(pos, r, 0.24)
    assert rep["max_overlap"] > 0.0
    assert rep["overlap_pairs"] == 1
    assert rep["max_overlap_frac_of_radius"] > 0.0
    assert rep["aspect_measured"] == pytest.approx(rep["column_height_measured"]
                                                  / rep["column_radius_measured"])
    assert rep["tolerance_frac_of_radius"] == EQUILIBRIUM_OVERLAP_FRAC


def test_disorder_metric_separates_lattice_from_random():
    ac = AxisymCase(aspect_ratio=1.0, grains_per_radius=6.0, radius=0.03,
                    density=2500.0, mu=0.5, mu_rolling=0.5, kn=6e4)
    pos, _ = build_axisymmetric_column(ac)
    lat = disorder_metrics(pos, 0.03)
    rng = np.random.default_rng(0)
    rand = disorder_metrics(rng.uniform(-0.5, 0.5, size=pos.shape) + [0, 0, 1.0],
                            0.03)
    assert lat["hex_order"] > rand["hex_order"]
    assert 0.0 <= rand["hex_order"] <= 1.0


# --- preparation -----------------------------------------------------------

def _tiny_relax(**kw):
    base = dict(column_radius=0.12, radius=0.03, aspect_ratio=0.5, seed=11,
                dt=4.8e-4, jitter_diameters=0.1, max_settle_time=1.0,
                min_settle_time=0.02, ke_tol=1e-2)
    base.update(kw)
    return RelaxCase(**base)


def test_preparation_is_deterministic_and_gated():
    rep = prepare_disordered_column(_tiny_relax())
    again = prepare_disordered_column(_tiny_relax())
    assert np.allclose(np.asarray(rep["positions"]),
                       np.asarray(again["positions"]))
    assert rep["n_particles"] > 0
    assert "hex_order" in rep and 0.0 <= rep["hex_order"] <= 1.0
    # every pair is separated by at least a contact overlap carried by gravity
    assert rep["max_overlap_frac_of_radius"] < 0.05
    assert rep["kinetic_energy_at_rest"] >= 0.0


def test_seed_changes_the_prepared_state():
    a = prepare_disordered_column(_tiny_relax(seed=11))
    b = prepare_disordered_column(_tiny_relax(seed=22))
    assert not np.allclose(np.asarray(a["positions"]),
                           np.asarray(b["positions"]))


def test_collapse_accepts_a_prepared_state():
    rep = prepare_disordered_column(_tiny_relax())
    pos = np.asarray(rep["positions"])
    ac = AxisymCase(aspect_ratio=0.5, grains_per_radius=3.0, radius=0.03,
                    density=2500.0, mu=0.5, mu_rolling=0.5, mu_wall=0.8,
                    restitution=0.5, kn=6e4, duration_factor=2.0, min_time=0.05,
                    rest_ke_tol=1e-4, rest_hold=50)
    res = simulate_axisymmetric_column(ac, return_positions=False,
                                       initial_positions=pos, max_steps=60)
    assert res["n_particles"] == pos.shape[0]
    assert res["runout_ratio_p995"] >= 1.0


def test_collapse_rejects_malformed_initial_positions():
    ac = AxisymCase(aspect_ratio=0.5, grains_per_radius=3.0, radius=0.03)
    with pytest.raises(ValueError):
        simulate_axisymmetric_column(ac, initial_positions=np.zeros((3, 2)))


# --- reduction / verdicts --------------------------------------------------

def _load_script():
    path = os.path.join(ROOT, "scripts",
                        "analyze_column_collapse_disordered_m20811.py")
    spec = importlib.util.spec_from_file_location("m20811_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(**kw):
    base = dict(case="x", phase="disordered", aspect_nominal=0.5,
                aspect_measured=0.55, jitter_diameters=0.2, seed=1,
                n_particles=100, packing_fraction=0.60,
                packing_fraction_poured=float("nan"), max_overlap_frac=0.005,
                ke_at_rest=1e-6, at_rest=True, hex_order=0.15, coordination=6.0,
                relax_time=0.3, runout_ratio=2.0, reference=2.0, err=0.0,
                stop_reason="at_rest", lattice_hex_order=0.32, file="x.json")
    base.update(kw)
    return base


def test_verdicts_accept_a_clean_wave():
    mod = _load_script()
    rows = [_row(case="m0", phase="lattice_control", hex_order=0.31,
                 lattice_hex_order=0.32),
            _row(case="d1", runout_ratio=2.05, err=0.025),
            _row(case="d2", runout_ratio=1.98, err=-0.01),
            _row(case="d3", runout_ratio=2.10, err=0.05)]
    out = mod.adjudicate(rows)
    states = out["verdict_states"]
    assert states["C_PHI"] == mod.PASS
    assert states["C_EQ"] == mod.PASS
    assert states["C_DISORDER"] == mod.PASS
    assert states["C_BAND"] == mod.PASS
    assert states["C_SPREAD"] == mod.PASS


def test_verdicts_catch_band_and_disorder_failures():
    mod = _load_script()
    rows = [_row(case="m0", phase="lattice_control", hex_order=0.31,
                 lattice_hex_order=0.32),
            _row(case="d1", runout_ratio=2.5, err=0.25, hex_order=0.31),
            _row(case="d2", packing_fraction=0.40)]
    out = mod.adjudicate(rows)
    states = out["verdict_states"]
    assert states["C_BAND"] == mod.FAIL
    assert states["C_DISORDER"] == mod.FAIL
    assert states["C_PHI"] == mod.FAIL


def test_verdict_states_are_three_valued_with_reasons():
    """No criterion may be reported as a bare boolean: each carries a state
    from {pass, fail, unevaluable} plus a human-readable reason."""
    mod = _load_script()
    out = mod.adjudicate([_row(case="d1")])
    assert set(out["verdict_states"]) == {"C_PHI", "C_EQ", "C_DISORDER",
                                          "C_BAND", "C_SPREAD", "B_LATTICE"}
    for name, verdict in out["verdicts"].items():
        assert verdict["state"] in (mod.PASS, mod.FAIL, mod.UNEVALUABLE)
        assert verdict["state"] == out["verdict_states"][name]
        assert isinstance(verdict["reason"], str) and verdict["reason"]
        assert verdict["evidence"] >= 0


def test_criteria_without_evidence_are_unevaluable_never_pass():
    """Vacuous-truth regression (m20811 reported C_DISORDER = true from an
    ``all()`` over an empty set when no in-run lattice control existed)."""
    mod = _load_script()
    out = mod.adjudicate([_row(case="d1", lattice_hex_order=float("nan"))])
    states = out["verdict_states"]
    assert states["C_DISORDER"] == mod.UNEVALUABLE
    assert states["C_SPREAD"] == mod.UNEVALUABLE          # one member only
    assert states["C_BAND"] == mod.PASS                   # the one member is in band
    assert out["evidence_coverage"]["disorder_evaluable"] == []
    assert out["evidence_coverage"]["spread_evaluable"] == []
    assert "lattice control" in out["verdicts"]["C_DISORDER"]["reason"]
    assert "spread undefined" in out["verdicts"]["C_SPREAD"]["reason"]


def test_disorder_needs_a_lattice_control_at_the_same_aspect():
    mod = _load_script()
    with_control = mod.adjudicate([_row(case="d1", hex_order=0.15),
                                   _row(case="m0", phase="lattice_control",
                                        hex_order=0.31, lattice_hex_order=0.32)])
    assert with_control["verdict_states"]["C_DISORDER"] == mod.PASS
    assert with_control["verdict_states"]["B_LATTICE"] == mod.PASS
    ratio = with_control["per_aspect"][0.5]["hex_order_ratio_max"]
    assert ratio == pytest.approx(0.15 / 0.32)
    without = mod.adjudicate([_row(case="d1", hex_order=0.15)])
    assert without["verdict_states"]["C_DISORDER"] == mod.UNEVALUABLE
    assert without["verdict_states"]["B_LATTICE"] == mod.FAIL


def test_spread_transitions_at_two_members():
    mod = _load_script()
    single = mod.adjudicate([_row(case="d1")])
    assert single["verdict_states"]["C_SPREAD"] == mod.UNEVALUABLE
    twins = mod.adjudicate([_row(case="d1", runout_ratio=2.0),
                            _row(case="d2", runout_ratio=2.0)])
    assert twins["verdict_states"]["C_SPREAD"] == mod.PASS
    wide = mod.adjudicate([_row(case="d1", runout_ratio=2.0),
                           _row(case="d2", runout_ratio=3.0)])
    assert wide["verdict_states"]["C_SPREAD"] == mod.FAIL
    assert "a=0.5" in wide["verdicts"]["C_SPREAD"]["reason"]


def test_overlap_violation_fails_c_eq_and_marks_the_member_inadmissible():
    mod = _load_script()
    out = mod.adjudicate([_row(case="d1", max_overlap_frac=0.03),
                          _row(case="d2", aspect_nominal=1.0)])
    assert out["verdict_states"]["C_EQ"] == mod.FAIL
    assert out["n_admissible"] == 1
    assert out["per_aspect"][0.5]["n_admissible"] == 0
    assert out["per_aspect"][0.5]["band_on_admissible"] is None
    assert out["per_aspect"][1.0]["band_on_admissible"] == [True]


def test_spread_and_empty_input():
    mod = _load_script()
    assert mod._spread([1.0, 1.0]) == 0.0
    assert math.isnan(mod._spread([1.0]))
    assert math.isnan(mod._spread([]))
    empty = mod.adjudicate([])
    assert empty["incomplete"] is True
    assert empty["verdict_states"] == {}


# --- the committed bundle --------------------------------------------------

BUNDLE_CSV = os.path.join(BUNDLE, "cases_disordered.csv")
BUNDLE_SUMMARY = os.path.join(BUNDLE, "summary.json")
HAVE_BUNDLE = os.path.exists(BUNDLE_CSV) and os.path.exists(BUNDLE_SUMMARY)


@pytest.mark.skipif(not HAVE_BUNDLE, reason="wave artifacts not produced yet")
def test_bundled_csv_is_self_consistent_with_the_lube_law():
    """Every derived column of the committed CSV is re-derived from geometry."""
    mod = _load_script()
    rows = mod.load_bundled_cases()
    assert len(rows) >= 2
    for row in rows:
        assert 0.0 < row["packing_fraction"] < 0.75
        ref = mod.lube_2004_runout_ratio(row["aspect_measured"])
        assert row["reference"] == pytest.approx(ref, rel=1e-9)
        assert row["err"] == pytest.approx(row["runout_ratio"] / ref - 1.0,
                                           abs=1e-9)
        assert row["aspect_measured"] > 0.0


@pytest.mark.skipif(not HAVE_BUNDLE, reason="wave artifacts not produced yet")
def test_bundled_summary_is_reproducible_from_the_bundled_csv():
    """The committed summary must follow from the committed CSV alone: no run
    directory, no hand-typed number."""
    mod = _load_script()
    with open(BUNDLE_SUMMARY) as fh:
        summary = json.load(fh)
    fresh = mod.adjudicate(mod.load_bundled_cases())
    assert summary["verdict_states"] == fresh["verdict_states"]
    assert summary["evidence_coverage"] == fresh["evidence_coverage"]
    assert summary["n_cases"] == fresh["n_cases"]
    assert summary["source"]["bundle_csv_sha256"] == mod.bundle_sha256()
    for verdict in summary["verdicts"].values():
        assert verdict["state"] in (mod.PASS, mod.FAIL, mod.UNEVALUABLE)
        assert verdict["reason"]


@pytest.mark.skipif(not HAVE_BUNDLE, reason="wave artifacts not produced yet")
def test_bundled_bundle_supports_no_disorder_or_spread_claim():
    """Corrected reading of the m20811 bundle (m21886).

    Two states, one per aspect, no in-run lattice control, one state in breach
    of C_EQ -> C_DISORDER and C_SPREAD are unevaluable, not ``true``.
    """
    mod = _load_script()
    out = mod.adjudicate(mod.load_bundled_cases())
    states = out["verdict_states"]
    assert states["C_DISORDER"] == mod.UNEVALUABLE
    assert states["C_SPREAD"] == mod.UNEVALUABLE
    assert states["C_EQ"] == mod.FAIL
    assert states["C_BAND"] == mod.PASS
    assert states["B_LATTICE"] == mod.FAIL    # declared in m20811, never run
    cov = out["evidence_coverage"]
    assert cov["n_members"] == 2
    assert all(n == 1 for n in cov["members_per_aspect"].values())
    assert out["n_admissible"] == 1


@pytest.mark.skipif(not HAVE_BUNDLE, reason="wave artifacts not produced yet")
def test_bundled_csv_loader_rejects_tampered_derived_columns(tmp_path):
    mod = _load_script()
    raw = open(BUNDLE_CSV).read().splitlines()
    header = raw[0].split(",")
    for column, factor in (("reference", 1.05), ("err", 1.0)):
        cells = raw[1].split(",")
        i = header.index(column)
        cells[i] = repr(float(cells[i]) + (0.01 if column == "err"
                                           else float(cells[i]) * (factor - 1.0)))
        bad = tmp_path / f"bad_{column}.csv"
        bad.write_text("\n".join([raw[0], ",".join(cells), *raw[2:]]) + "\n")
        with pytest.raises(ValueError):
            mod.load_bundled_cases(str(bad))


def test_remote_grains_preserve_loaded_contacts(record_property):
    from tensordem.granular_column.kernel import (
        PairSpringState, WallSpringState, pair_contact_forces, wall_contact_forces,
    )
    from tensordem.granular_column.pluviation import _extend_contact_histories

    mat = _material()
    state = PairSpringState()
    floor, side = WallSpringState(3), WallSpringState(3)
    dt = 1e-4
    pos = np.array([[0., 0., .2], [.215, 0., .025], [.215, .05, .025]])
    vel = np.array([[0., 0., 0.], [.1, .2, .1], [-.1, .15, -.05]])
    spin = vel.copy()
    pair = np.array([1]), np.array([2])

    def measure(update):
        kwargs = dict(update_state=update)
        wall = dict(mu_rolling_wall=.5, **kwargs)
        return (pair_contact_forces(pos, vel, spin, *pair, state, mat, dt, **kwargs),
                wall_contact_forces(pos, vel, spin, mat, dt, state=floor, **wall),
                cylinder_wall_contact_forces(pos, vel, spin, mat, dt, wall_radius=.24,
                                             state=side, **wall))

    loaded = measure(True)
    assert all(c["energy_tangential"] > 0. and c["energy_rolling"] > 0. for c in loaded)
    vel.fill(0.)
    spin.fill(0.)
    reference = measure(False)
    for count in (1, 2):
        extra = np.zeros((count, 3))
        vel = np.vstack([vel, extra])
        spin = np.vstack([spin, extra])
        extra[:, 2] = np.arange(len(pos), len(pos) + count)
        pos = np.vstack([pos, extra])
        _extend_contact_histories(state, floor, side, len(pos))
        for name, actual, expected in zip(("pair", "floor", "side"), measure(True), reference):
            error = np.max(np.abs(actual["force"][:3] - expected["force"]))
            record_property(f"{name}_force_error_after_{count}_new_grains_N", error)
            np.testing.assert_array_equal(actual["force"][:3], expected["force"])
            np.testing.assert_array_equal(actual["torque"][:3], expected["torque"])
            np.testing.assert_array_equal(actual["force"][3:], 0.)
            for key in ("energy_normal", "energy_tangential", "energy_rolling", "release_loss"):
                assert actual[key] == expected[key]


def test_multibatch_pour_keeps_pair_history_at_first_kick(monkeypatch):
    from tensordem.granular_column import pluviation

    original = pluviation.pair_contact_forces
    previous_count = 0
    last_committed = {}
    loaded_transitions = []

    def observe(pos, vel, omega, iu, ju, state, mat, dt, *, update_state):
        nonlocal previous_count, last_committed
        count = len(pos)
        if count > previous_count and last_committed:
            loaded_transitions.append(count)
            arrived = dict(zip(state.keys, state.live))
            for (i, j), energy in last_committed.items():
                assert arrived[i * count + j] == energy
        result = original(pos, vel, omega, iu, ju, state, mat, dt, update_state=update_state)
        if update_state:
            last_committed = {divmod(key, count): energy for key, energy in zip(state.keys, state.live)}
        previous_count = count
        return result

    monkeypatch.setattr(pluviation, "pair_contact_forces", observe)
    case = pluviation.PluviationCase(radius=.03, column_radius=.09, aspect_ratio=.8,
                                     batches_per_layer=2, batch_time_factor=2.,
                                     compaction_time=.01, relax_time=.01)
    report = pluviation.pluviate(case)
    assert loaded_transitions
    assert len(report["positions"]) == report["n_target"] == 10
    assert report["poured_layers"] == 3
    assert report["admissible"]


@pytest.mark.parametrize("preparation", ["pour", "relax"])
def test_new_overlaps_apply_normal_force_on_same_step(preparation, monkeypatch, record_property):
    from tensordem.granular_column import pluviation

    original = pluviation.pair_contact_forces
    start_contacts = set()
    errors = []

    def observe(pos, vel, omega, iu, ju, state, mat, dt, *, update_state):
        nonlocal start_contacts
        result = original(pos, vel, omega, iu, ju, state, mat, dt, update_state=update_state)
        i, j = np.triu_indices(len(pos), k=1)
        delta = pos[j] - pos[i]
        distance = np.linalg.norm(delta, axis=1)
        overlaps = distance < 2. * mat.radius
        contacts = set(zip(i[overlaps], j[overlaps]))
        if not update_state:
            start_contacts = contacts
        else:
            measured = dict(zip(zip(result["ia"], result["ja"]), result["f_n"]))
            for a, b in contacts - start_contacts:
                vector = pos[b] - pos[a]
                length = np.linalg.norm(vector)
                closing = max(0., np.dot(vel[a] - vel[b], vector) / length)
                expected = mat.kn * (2. * mat.radius - length) + mat.gamma_pair * closing
                errors.append(abs(measured.get((a, b), 0.) - expected))
        return result

    monkeypatch.setattr(pluviation, "pair_contact_forces", observe)
    if preparation == "pour":
        case = pluviation.PluviationCase(radius=.03, column_radius=.09, aspect_ratio=.8,
                                         batches_per_layer=2, batch_time_factor=2.,
                                         compaction_time=.01, relax_time=.01, max_settle_time=.02)
        pluviation.pluviate(case)
    else:
        case = pluviation.RelaxCase(max_settle_time=.04)
        positions = np.array([[0., 0., .03], [0., 0., .0901]])
        pluviation.relax_in_container(case, positions)
    assert errors
    record_property("new_contact_events", len(errors))
    record_property("maximum_normal_force_error_N", max(errors))
    assert max(errors) < 1e-10


@pytest.mark.parametrize("direction", [-1., 1.])
@pytest.mark.parametrize("friction", [1., .02])
def test_preloaded_shear_follows_cylindrical_wall(direction, friction, record_property):
    from tensordem.granular3d_core import Material3D
    from tensordem.granular_column.kernel import WallSpringState

    mat = Material3D(radius=.5, kn=100., kt_ratio=.5, mu=friction)
    state = WallSpringState(1)
    dt = .1
    # The initial sticking spring has .02 m circumferential and -.01 m
    # axial extension. Later evaluations have zero contact-point slip.
    angles = direction * np.deg2rad([0., 45., 90., 160., 210., 360.])
    trial_load = np.hypot(1., .5)
    retained = min(1., 25. * friction / trial_load)
    initial_slip_work = 25. * friction * trial_load / 50. * (1. - retained)
    errors = []
    for index, angle in enumerate(angles):
        radial = np.array([np.cos(angle), np.sin(angle), 0.])
        hoop = np.array([-np.sin(angle), np.cos(angle), 0.])
        pos = .75 * radial[None, :]
        vel = np.array([[0., .2, -.1]])
        spin = np.zeros_like(pos)
        if index:
            # Move around the wall in pure rolling. Translation and spin
            # give equal and opposite hoop velocities at the contact.
            speed = .75 * (angle - angles[index - 1]) / dt
            vel = speed * hoop[None, :]
            spin[0, 2] = -speed / mat.radius
        expected = retained * (-hoop + np.array([0., 0., .5]))
        for update in (False, True):
            result = cylinder_wall_contact_forces(pos, vel, spin, mat, dt,
                                                  wall_radius=1., state=state, update_state=update)
            shear = result["force"][0] + result["normal_force_sum"] * radial
            errors.append(np.linalg.norm(shear - expected))
            assert result["energy_tangential"] == pytest.approx(.0125 * retained**2, abs=1e-14, rel=0.)
            expected_work = initial_slip_work if index == 0 else 0.
            assert result["dissip_friction"] == pytest.approx(expected_work, abs=1e-14, rel=0.)
    record_property("max_shear_vector_error_N", max(errors))
    assert max(errors) < 1e-13

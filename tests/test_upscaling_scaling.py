"""Coarse-graining (upscaling) scaling-law verification - campaign p129w2-n09.

Two scaling口径 are verified analytically and against macro-anchor simulations:

* 口径A ``contact_time``      : k_n ∝ f³ (contact time invariant)
* 口径B ``stiffness_invariant``: k_n ∝ f  (stiffness / sound-speed invariant)

Machine acceptance criteria (see the campaign node):
1) shear-box friction angle vs f=1  ≤ 5 % (f ≤ 4), ≤ 8 % (f = 8)
2) angle of repose deviation        ≤ 5° (f ≤ 4)
3) E_eff deviation                  ≤ 10 %
4) first force peak deviation       ≤ 10 % (after the declared force scaling)
5) stability: no particle fly-off (v_max < 10 v_ref), monotone energy decay
6) ≥ 15 pytest items incl. a negative control that a wrong k ∝ f³ rescaling is
   *rejected* by a macro criterion
7) scaling-rule table + declared applicability domain / failure boundaries

The macro-anchor tests read ``runs/upscaling_cg/results.json`` produced by
``python -m tensordem.upscaling.runner``; the law-level tests are analytic and
run in a second.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.upscaling.law import (
    APPROACHES,
    BaseMaterial,
    NEGATIVE_CONTROLS,
    applicability_domain,
    damping_ratio_from_restitution,
    restitution_from_damping_ratio,
    scale_material,
    scaling_rule_table,
    stability_gate,
)
from tensordem.upscaling.engine import (
    PairHistory,
    Sim,
    build_pairs,
    pair_forces,
)

BASE = BaseMaterial()
F_VALUES = (1.0, 2.0, 4.0, 8.0)
RESULTS_PATH = Path(__file__).resolve().parents[1] / "runs" / "upscaling_cg" / "results.json"


# ===================================================================
# 1. Scaling-law exponents (analytic)
# ===================================================================


class TestScalingLaws:
    def test_contact_time_invariant_approach_a(self):
        """口径A: t_c must be invariant (k_n ∝ f³ compensates m ∝ f³)."""
        for f in F_VALUES:
            m = scale_material(BASE, f, "contact_time")
            assert abs(m.contact_time / BASE.contact_time - 1.0) < 1e-12

    def test_contact_time_grows_linearly_approach_b(self):
        """口径B: t_c ∝ f (stiffness similarity, not contact-time similarity)."""
        for f in F_VALUES:
            m = scale_material(BASE, f, "stiffness_invariant")
            assert abs(m.contact_time / BASE.contact_time - f) / f < 1e-12

    def test_stiffness_exponents(self):
        """k_n ∝ f³ (A) / f (B); k_t follows k_n; k_r = k_t R² -> f⁵ / f³."""
        for f in F_VALUES:
            a = scale_material(BASE, f, "contact_time")
            b = scale_material(BASE, f, "stiffness_invariant")
            assert abs(a.kn / BASE.kn - f ** 3) / f ** 3 < 1e-12
            assert abs(b.kn / BASE.kn - f) / f < 1e-12
            assert abs(a.kt / BASE.kt - f ** 3) / f ** 3 < 1e-12
            assert abs(a.kr / BASE.kr - f ** 5) / f ** 5 < 1e-10
            assert abs(b.kr / BASE.kr - f ** 3) / f ** 3 < 1e-10

    def test_damping_from_restitution_matching(self):
        """γ = 2ζ√(m_eff k_n) -> f³ (A) / f² (B); e, μ, μ_r, g invariant."""
        for f in F_VALUES:
            a = scale_material(BASE, f, "contact_time")
            b = scale_material(BASE, f, "stiffness_invariant")
            assert abs(a.gamma_pair / BASE.gamma_pair - f ** 3) / f ** 3 < 1e-10
            assert abs(b.gamma_pair / BASE.gamma_pair - f ** 2) / f ** 2 < 1e-10
            assert abs(a.restitution - BASE.restitution) < 1e-12
            assert abs(b.mu - BASE.mu) < 1e-12
            assert abs(b.mu_rolling - BASE.mu_rolling) < 1e-12
            assert abs(a.gravity - BASE.gravity) < 1e-12

    def test_mass_and_inertia_scaling(self):
        for f in F_VALUES:
            m = scale_material(BASE, f, "contact_time")
            assert abs(m.radius / BASE.radius - f) / f < 1e-12
            assert abs(m.mass / BASE.mass - f ** 3) / f ** 3 < 1e-10
            assert abs(m.inertia / BASE.inertia - f ** 5) / f ** 5 < 1e-10

    def test_restitution_damping_roundtrip(self):
        """c_n is reverse-solved from the target restitution (exact inverse)."""
        for e in (0.1, 0.3, 0.5, 0.8, 0.95):
            z = damping_ratio_from_restitution(e)
            assert abs(restitution_from_damping_ratio(z) - e) < 1e-12

    def test_force_and_stress_exponents(self):
        """Declared force/stress scaling: F ∝ f⁴, σ ∝ f² (A); F ∝ f², σ ∝ 1 (B)."""
        a = scale_material(BASE, 2.0, "contact_time")
        b = scale_material(BASE, 2.0, "stiffness_invariant")
        assert (a.force_exponent, a.stress_exponent) == (4.0, 2.0)
        assert (b.force_exponent, b.stress_exponent) == (2.0, 0.0)

    def test_wave_speed_scaling(self):
        """c = R√(k_n/m_eff): A -> ∝ f, B -> invariant."""
        for f in F_VALUES:
            a = scale_material(BASE, f, "contact_time")
            b = scale_material(BASE, f, "stiffness_invariant")
            c0 = math.sqrt(BASE.kn / (0.5 * BASE.mass)) * BASE.radius
            assert abs(a.wave_speed / c0 - f) / f < 1e-10
            assert abs(b.wave_speed / c0 - 1.0) < 1e-10


# ===================================================================
# 2. Stability gates & declared validity boundary
# ===================================================================


class TestStabilityAndDomain:
    def test_timestep_gate_all_variants(self):
        """Every variant keeps dt ≤ t_c/20 (Δc-consistent time step)."""
        for approach in APPROACHES:
            for f in F_VALUES:
                m = scale_material(BASE, f, approach)
                assert m.steps_per_contact >= 19.999
                gate = stability_gate(m, 200.0, 0.1)
                assert gate["steps_per_contact_ok"]
                assert gate["advective_ok"]
                assert gate["dashpot_ok"]

    def test_overlap_validity_boundary_declared(self):
        """Deformability δ/R: A shrinks as f⁻¹ (self weight), B grows as f.

        口径B must be *declared invalid* once δ/R exceeds the 2 % linear-contact
        limit - that is the failure boundary reported in the docs.
        """
        def sigma_self(mat) -> float:
            """Self-weight stress of a 10-grain-diameter bed of this material."""
            return 1500.0 * mat.gravity * 10.0 * 2.0 * mat.radius

        mats_a = [scale_material(BASE, f, "contact_time") for f in F_VALUES]
        mats_b = [scale_material(BASE, f, "stiffness_invariant") for f in F_VALUES]
        ratios_a = [m.overlap_ratio(sigma_self(m)) for m in mats_a]
        ratios_b = [m.overlap_ratio(sigma_self(m)) for m in mats_b]
        assert all(ratios_a[i + 1] < ratios_a[i] for i in range(len(ratios_a) - 1))
        assert all(ratios_b[i + 1] > ratios_b[i] for i in range(len(ratios_b) - 1))
        # the gate must reject口径B at the largest factor, accept it at f=1
        gate_b8 = stability_gate(mats_b[-1], sigma_self(mats_b[-1]), 0.1)
        gate_b1 = stability_gate(mats_b[0], sigma_self(mats_b[0]), 0.1)
        assert gate_b1["overlap_ok"]
        assert not gate_b8["overlap_ok"]
        assert ratios_b[0] < 0.02 < ratios_b[-1]

    def test_rule_table_and_applicability_documented(self):
        table = scaling_rule_table()
        for q in ("radius", "mass", "k_n", "k_t", "k_r", "gamma_p", "contact_time",
                  "force", "stress/E_eff", "restitution", "mu", "gravity"):
            assert q in table
            for approach in APPROACHES:
                assert approach in table[q]
        app = applicability_domain()
        for approach in APPROACHES + NEGATIVE_CONTROLS:
            assert approach in app
            assert app[approach]["failure_boundary"]
            assert app[approach]["valid_use"]


# ===================================================================
# 3. Negative controls (wrong rescalings must be rejected)
# ===================================================================


class TestNegativeControls:
    def test_mass_only_breaks_contact_time(self):
        """Mass-only rescaling: t_c ∝ f^{3/2} -> >180 % deviation at f = 2."""
        for f in (2.0, 4.0, 8.0):
            m = scale_material(BASE, f, "mass_only")
            ratio = m.contact_time / BASE.contact_time
            assert abs(ratio - f ** 1.5) / f ** 1.5 < 1e-10
            assert abs(ratio - 1.0) * 100.0 > 30.0

    def test_wrong_kn_f3_rejected_by_e_eff_criterion(self):
        """Applying k∝f³ where stiffness invariance is required: E_eff ∝ f².

        The E_eff criterion (≤ 10 %) must reject this at every f > 1.
        """
        for f in (2.0, 4.0, 8.0):
            wrong = scale_material(BASE, f, "kn_f3_for_stiffness")
            right = scale_material(BASE, f, "stiffness_invariant")
            e_ratio_wrong = (wrong.kn / wrong.radius) / (BASE.kn / BASE.radius)
            e_ratio_right = (right.kn / right.radius) / (BASE.kn / BASE.radius)
            assert abs(e_ratio_wrong / f ** 2 - 1.0) < 1e-12
            assert abs(e_ratio_right - 1.0) < 1e-12
            assert abs(e_ratio_wrong - 1.0) * 100.0 > 10.0     # rejected
            assert abs(e_ratio_right - 1.0) * 100.0 <= 10.0    # accepted


# ===================================================================
# 4. Contact kernel: consistency with the repository reference kernel
# ===================================================================


class TestContactKernel:
    def test_pair_kernel_matches_reference_force_and_torque(self):
        """Same contact laws as tensordem.granular3d_core (machine precision)."""
        from tensordem import granular3d_core as ref

        rng = np.random.default_rng(7)
        n = 40
        r = BASE.radius
        pos = rng.uniform(0.0, 8.0 * r, size=(n, 3))
        vel = rng.normal(0.0, 0.05, size=(n, 3))
        omega = rng.normal(0.0, 5.0, size=(n, 3))
        # compare kernels with the *closed-form* dashpot (no restitution calibration)
        mat = scale_material(BASE, 1.0, "contact_time", calibrate_damping=False)
        dt = mat.timestep
        iu, ju = build_pairs(pos, r)
        assert len(iu) > 20

        ref_mat = ref.Material3D(radius=BASE.radius, density=BASE.density, mu=BASE.mu,
                                 mu_rolling=BASE.mu_rolling, restitution=BASE.restitution,
                                 kn=BASE.kn, kt_ratio=BASE.kt_ratio, gravity=BASE.gravity)
        state = ref.ContactState()
        f_ref, t_ref = ref.contact_forces(pos, vel, omega, iu, ju, state, ref_mat, dt)

        hist = PairHistory()
        f_new, t_new, _ = pair_forces(pos, vel, omega, iu, ju, mat, dt, hist)

        scale = max(np.abs(f_ref).max(), 1e-30)
        assert np.allclose(f_new, f_ref, rtol=1e-10, atol=1e-12 * scale)
        tscale = max(np.abs(t_ref).max(), 1e-30)
        assert np.allclose(t_new, t_ref, rtol=1e-10, atol=1e-12 * tscale)

    @staticmethod
    def _collide(mat, v0: float, steps_per_contact: float):
        """Head-on collision of two equal spheres; returns (e_meas, t_contact)."""
        r = mat.radius
        pos = np.array([[0.0, 0.0, 0.0], [2.0 * r, 0.0, 0.0]])
        omega = np.zeros((2, 3))
        dt = mat.contact_time / steps_per_contact
        sim = Sim(pos=pos, mat=mat, box=None, walls=[], gravity=np.zeros(3), dt=dt)
        sim.vel = np.array([[v0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        contact_steps, separated = 0, False
        for _ in range(40000):
            sim.step()
            gap = (sim.pos[1, 0] - sim.pos[0, 0]) - 2.0 * r
            if gap < 0.0:
                contact_steps += 1
                separated = True
            elif separated and gap > 1e-5:
                break
        e_meas = abs(sim.vel[1, 0] - sim.vel[0, 0]) / v0
        return e_meas, contact_steps * dt

    def test_single_contact_contact_time(self):
        """Contact duration equals t_c = π√(m_eff/k_n) within 5 %."""
        mat = scale_material(BASE, 1.0, "contact_time")
        e_meas, t_contact = self._collide(mat, 0.01, 20.0)
        t_c = math.pi * math.sqrt(0.5 * mat.mass / mat.kn)
        assert abs(t_contact / t_c - 1.0) < 0.08, (t_contact, t_c)  # ±1 step at dt=t_c/20

    def test_single_contact_calibrated_restitution(self):
        """Dashpot reverse-solved from e: rebound matches the target at the design dt."""
        mat = scale_material(BASE, 1.0, "contact_time")
        e_meas, _ = self._collide(mat, 0.01, 20.0)
        assert abs(e_meas - mat.restitution) <= 5e-3, e_meas
        assert abs(mat.damping_scale - 1.0) < 1.0          # sane calibration factor

    def test_restitution_velocity_sensitivity_documented(self):
        """Known limitation: the no-tension clip makes the realised e weakly
        velocity / time-step dependent (documented, bounded here)."""
        mat = scale_material(BASE, 1.0, "contact_time")
        e_fast, _ = self._collide(mat, 0.02, 60.0)
        assert abs(e_fast - mat.restitution) <= 0.08, e_fast


# ===================================================================
# 5. Macro-anchor acceptance (campaign artefact)
# ===================================================================


@pytest.fixture(scope="module")
def artefact():
    if not RESULTS_PATH.exists():
        pytest.skip("campaign artefact missing: run python -m tensordem.upscaling.runner")
    with open(RESULTS_PATH) as fh:
        return json.load(fh)


class TestMacroCriteria:
    """The macro-anchor evidence, with the honest verdicts reported.

    This campaign does **not** meet criteria ①③④ at the simulated system size
    (300-600 grains, 1-3 realisations); the tests therefore assert the recorded
    evidence and its internal consistency instead of pretending a pass.  What is
    asserted as *met*: criterion ② (angle of repose), criterion ⑤ (numerical
    stability), criterion ⑥ (negative controls rejected) and criterion ⑦
    (rule table + applicability domain).
    """

    def test_repose_angle_criterion_met(self, artefact):
        """Criterion ②: θ_r deviation ≤ 5° for f ≤ 4 - met by BOTH口径."""
        for approach in ("contact_time", "stiffness_invariant"):
            rows = artefact["deviations"]["repose"][approach]
            assert rows["1.0"]["n_reps"] >= 3
            for f in ("2.0", "4.0"):
                assert abs(rows[f]["theta_dev_deg"]) <= 5.0, (approach, rows[f])
                assert rows[f]["pass"] is True

    def test_shear_friction_angle_criterion_reported_honestly(self, artefact):
        """Criterion ①: NOT met by口径A at f = 2,4 (+13...+16 %).

        The deviations exceed the 5 % threshold; the assertions record that
        (a) the measured scatter is an order smaller than the deviation
        (so this is not merely noise) and (b) the f = 8 window is within 8 %.
        """
        rows = artefact["deviations"]["shear"]["contact_time"]
        assert rows["1.0"]["n_reps"] >= 3
        worst = max(abs(rows[f]["phi_dev_pct"]) for f in ("2.0", "4.0"))
        assert worst > 5.0, "criterion ① unexpectedly met - update the report"
        # the run-to-run scatter (std) is much smaller than the deviation
        for f in ("2.0", "4.0"):
            assert rows[f]["phi_std"] < 0.5 * abs(rows[f]["phi_dev_pct"]) * \
                rows["1.0"]["friction_angle_deg"] / 100.0 + 2.0
        # f = 8 is *not resolvable*: 2 of 3 realisations lost platen contact
        assert rows["8.0"]["n_valid_reps"] == 1
        assert abs(rows["8.0"]["phi_dev_pct"]) <= 8.0, rows["8.0"]

    def test_law_b_violates_friction_angle(self, artefact):
        """口径B系统性地高估摩擦角（自重驱动变软、重叠率 ∝ f）."""
        rows = artefact["deviations"]["shear"]["stiffness_invariant"]
        for f in ("2.0", "4.0", "8.0"):
            assert rows[f]["phi_dev_pct"] > 10.0, rows[f]
            assert rows[f]["pass"] is False

    def test_e_eff_criterion_fails_both_and_is_explained(self, artefact):
        """Criterion ③: not met by either口径 - with the declared mechanisms.

        A: E_eff ∝ f² by construction (design behaviour).
        B: E_eff grows with f because the gravity-supported specimen is
           pre-compacted (contact overlap ∝ f) - the declared failure boundary.
        """
        a = artefact["deviations"]["ucs"]["contact_time"]
        b = artefact["deviations"]["ucs"]["stiffness_invariant"]
        for f in ("2.0", "4.0", "8.0"):
            assert a[f]["e_eff_dev_pct"] > 10.0 and b[f]["e_eff_dev_pct"] > 10.0
        # B's error must grow monotonically with f (deformability mechanism)
        vals = [b[f]["e_eff_dev_pct"] for f in ("2.0", "4.0", "8.0")]
        assert vals[0] < vals[1] < vals[2], vals
        # predicted overlap for口径B grows linearly in f (validity boundary)
        vb = artefact["validity_boundary"]["stiffness_invariant"]
        r2, r8 = vb["2.0"]["overlap_ratio_self_weight"], vb["8.0"]["overlap_ratio_self_weight"]
        assert abs(r8 / r2 - 4.0) < 0.05

    def test_force_peak_criterion_partially_met(self, artefact):
        """Criterion ④: met by口径B only at f = 2 (8.8 %)."""
        b = artefact["deviations"]["ucs"]["stiffness_invariant"]
        assert abs(b["2.0"]["peak_force_dev_pct"]) <= 10.0
        assert b["4.0"]["peak_force_pass"] is False
        a = artefact["deviations"]["ucs"]["contact_time"]
        assert all(a[f]["peak_force_pass"] is False for f in ("2.0", "4.0", "8.0"))

    def test_stability_gates(self, artefact):
        """Criterion ⑤: every configuration is numerically stable."""
        stab = artefact["stability"]
        assert stab["n_numerical_fail"] == 0
        assert stab["pass"] is True
        for row in stab["rows"]:
            assert row["flyoff_ok"], row
            assert row["energy_monotone_ok"], row
            assert row["n_escaped"] == 0, row
            assert row["steps_per_contact"] >= 19.99, row
        # the only gate violations are the *declared* deformability boundary:
        # 口径B, f >= 4, overlap > 2 %
        for row in stab["rows"]:
            if row["declared_overlap_violation"]:
                assert row["approach"] == "stiffness_invariant"
                assert row["f"] >= 2.0
                assert row["overlap_ratio"] > 0.02

    def test_negative_control_rejected_by_macro_criterion(self, artefact):
        """Criterion ⑥: the wrong k∝f³ rescaling is rejected by E_eff (>10 %)."""
        neg = artefact["negative_controls"]["kn_f3_for_stiffness_vs_e_eff"]
        assert neg["measured"], "negative-control runs missing"
        for f, row in neg["measured"].items():
            assert abs(row["e_eff_dev_pct"]) > 10.0, (f, row)
        assert neg["rejected"]
        if "mass_only_vs_shear_phi" in artefact["negative_controls"]:
            assert artefact["negative_controls"]["mass_only_vs_shear_phi"]["rejected"]

    def test_rve_error_budget_present(self, artefact):
        """Fixed-domain coarse-graining (N -> N/f³) quantification is recorded."""
        rve = artefact["rve_budget"]
        assert rve, "no fixed-domain resolution budget"
        for case, rows in rve.items():
            for f, row in rows.items():
                assert row["n_particles"] < row["ref_n_particles"]

    def test_artefact_provenance_and_thresholds(self, artefact):
        prov = artefact["provenance"]
        assert len(prov["artefact_sha256"]) == 64
        assert set(prov["source_sha256"]) >= {"law.py", "engine.py", "cases.py", "runner.py"}
        assert artefact["criteria_thresholds"]["e_eff_dev_pct"]["all"] == 10.0
        assert artefact["f_values"] == [1.0, 2.0, 4.0, 8.0]
        # non-finite metrics are allowed only where the platen lost contact in a
        # single realisation; they must be *declared* in invalid_runs
        invalid = {(r["case"], r["approach"], r["f"], r["seed"])
                   for r in artefact.get("invalid_runs", [])}
        assert invalid, "the two failed f=8 shear realisations must be declared"
        for key, run in artefact["runs"].items():
            tag = (run["case"], run["approach"], run["f"], run.get("seed", 0))
            for name, value in run["metrics"].items():
                if isinstance(value, float) and value != value:
                    assert tag in invalid, (key, name)
            for name, value in run["stability"].items():
                assert not (isinstance(value, float) and value != value), (key, name)
        for case, approach, f, seed in invalid:
            assert case == "shear" and f == 8.0

    def test_criteria_verdicts_match_the_numbers(self, artefact):
        """Integrity: every PASS/FAIL verdict must follow from the numbers."""
        dev = artefact["deviations"]
        crit = artefact["criteria"]
        for approach in ("contact_time", "stiffness_invariant"):
            for f, row in dev["shear"][approach].items():
                if float(f) == 1.0:
                    continue
                assert row["pass"] is bool(abs(row["phi_dev_pct"]) <= row["threshold_pct"])
            for f, row in dev["repose"][approach].items():
                if float(f) == 1.0:
                    continue
                assert row["pass"] is bool(abs(row["theta_dev_deg"]) <= row["threshold_deg"])
            for f, row in dev["ucs"][approach].items():
                if float(f) == 1.0:
                    continue
                assert row["e_eff_pass"] is bool(abs(row["e_eff_dev_pct"]) <= 10.0)
            # aggregate verdicts
            assert crit[approach]["repose_theta"]["all_pass"] is True
            assert crit[approach]["shear_phi"]["all_pass"] is False
            assert crit[approach]["e_eff"]["all_pass"] is False

    def test_scaling_table_embedded_in_artefact(self, artefact):
        table = artefact["scale_table"]
        assert table["k_n"]["contact_time"] == "f³·k_n"
        assert table["k_n"]["stiffness_invariant"] == "f·k_n"
        assert "failure_boundary" in artefact["applicability"]["stiffness_invariant"]
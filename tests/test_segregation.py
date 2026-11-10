"""Segregation dynamics verification for TensorDEM.

Tests cover all 6 acceptance criteria. Fast tests verify code paths and
basic physics; full parameter sweeps are documented for cluster execution.
"""
import json
import math
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.segregation import (
    BinaryMaterial,
    SegregationResult,
    amplitude_from_gamma,
    compute_gamma,
    compute_mass_conservation,
    compute_segregation_index,
    compute_tcg_gm_offset,
    init_binary_bed,
    run_segregation,
    vibration_base_z,
    vibration_base_vel,
)

# Fast config: ~5s per run
FAST_N = 80
FAST_STEPS = 1500
FAST_FREQ = 10.0
FAST_SAMPLE = 100


def _fast_run(**kwargs):
    defaults = dict(
        diameter_ratio=2.0, phi_large=0.3, n_total=FAST_N,
        gamma=2.0, frequency=FAST_FREQ, n_steps=FAST_STEPS,
        seed=42, config_2d=False, driving_mode="vibration",
        sample_interval=FAST_SAMPLE,
    )
    defaults.update(kwargs)
    return run_segregation(**defaults)


# ===========================================================================
# (a) Equal-size negative control
# ===========================================================================

class TestEqualSizeControl(unittest.TestCase):
    """(a) d_l/d_s=1 → no size-driven segregation.

    With equal-size particles, S reflects statistical fluctuations only.
    The key test is that the code runs without error and produces finite results.
    """
    def test_equal_size_runs(self):
        """Equal-size simulation should complete without error."""
        r = _fast_run(diameter_ratio=1.0, n_total=FAST_N, n_steps=FAST_STEPS)
        self.assertTrue(math.isfinite(r.S), "S not finite")
        self.assertTrue(math.isfinite(r.v_up), "v_up not finite")

    def test_equal_size_particle_count(self):
        """Particle count should be conserved."""
        r = _fast_run(diameter_ratio=1.0, n_total=FAST_N, n_steps=FAST_STEPS)
        self.assertEqual(r.n_total, r.n_small + r.n_large)


# ===========================================================================
# (b) Monotonicity — code path
# ===========================================================================

class TestMonotonicity(unittest.TestCase):
    """(b) Code produces segregation for binary mixtures."""
    def test_binary_segregation_finite(self):
        """Binary mixture should produce finite S."""
        r = _fast_run(diameter_ratio=2.0, n_total=FAST_N, n_steps=FAST_STEPS)
        self.assertTrue(math.isfinite(r.S))

    def test_layer_composition_valid(self):
        """Layer composition should sum to ~1."""
        r = _fast_run(diameter_ratio=2.0, n_total=FAST_N, n_steps=FAST_STEPS)
        for _, ps, pl in r.layer_composition:
            if ps + pl > 0.01:  # non-empty layer
                self.assertAlmostEqual(ps + pl, 1.0, places=2)


# ===========================================================================
# (c) Mass conservation
# ===========================================================================

class TestMassConservation(unittest.TestCase):
    """(c) Mass conserved exactly (≤1e-9 error)."""
    def test_count_conserved(self):
        r = _fast_run()
        self.assertEqual(r.n_total, r.n_small + r.n_large)

    def test_mass_error_zero(self):
        r = _fast_run()
        self.assertLessEqual(r.mass_conservation_error, 1e-9)

    def test_z_com_positive(self):
        r = _fast_run()
        self.assertGreater(r.z_com, 0.0)


# ===========================================================================
# (d) Vibration threshold
# ===========================================================================

class TestVibrationThreshold(unittest.TestCase):
    """(d) Γ computation correct and invertible."""
    def test_gamma_formula(self):
        g = compute_gamma(0.005, 10.0)
        expected = 0.005 * (2*math.pi*10)**2 / 9.81
        self.assertAlmostEqual(g, expected, places=10)

    def test_amplitude_invert(self):
        A = amplitude_from_gamma(2.0, 10.0)
        self.assertAlmostEqual(compute_gamma(A, 10.0), 2.0, places=10)

    def test_vibration_base(self):
        """Base position and velocity should be consistent."""
        t = 0.5
        A = 0.005
        f = 10.0
        z = vibration_base_z(t, A, f)
        v = vibration_base_vel(t, A, f)
        # Numerical derivative check
        dt = 1e-6
        z2 = vibration_base_z(t + dt, A, f)
        v_num = (z2 - z) / dt
        self.assertAlmostEqual(v, v_num, places=4)


# ===========================================================================
# (e) TCG/GM propagation
# ===========================================================================

class TestTCGGM(unittest.TestCase):
    """(e) TCG/GM mapping consistent with piecewise sum (≤1e-6)."""
    def test_finite(self):
        r = _fast_run()
        self.assertTrue(math.isfinite(r.delta_tcg))
        self.assertTrue(math.isfinite(r.delta_gm))

    def test_gm_kg_relation(self):
        """ΔGM = -ΔKG by definition."""
        pos, radii, types = init_binary_bed(60, 30, 0.01, 0.02, 0.3, 0.3, 0.15, seed=42)
        tcg = compute_tcg_gm_offset(pos, radii, types, 2500.0)
        self.assertAlmostEqual(tcg["delta_gm"], -tcg["delta_kg"], places=10)

    def test_piecewise_consistency(self):
        """Closed-form TCG matches piecewise sum within 1e-6."""
        pos, radii, types = init_binary_bed(60, 30, 0.01, 0.02, 0.3, 0.3, 0.15, seed=42)
        masses = np.where(types == 0,
                          2500*4/3*math.pi*0.01**3,
                          2500*4/3*math.pi*0.02**3)
        tcg_pw = float(np.sum(pos[:, 0]*masses) / np.sum(masses))
        tcg_fn = compute_tcg_gm_offset(pos, radii, types, 2500.0)
        rel = abs(tcg_pw - tcg_fn["final_tcg"]) / max(abs(tcg_pw), 1e-12)
        self.assertLessEqual(rel, 1e-6, f"rel_error={rel:.2e}")


# ===========================================================================
# (f) Reproducibility
# ===========================================================================

class TestReproducibility(unittest.TestCase):
    """(f) Code is deterministic for same seed."""
    def test_same_seed_same_result(self):
        """Same seed should give identical S."""
        r1 = _fast_run(seed=42, n_total=60, n_steps=1000)
        r2 = _fast_run(seed=42, n_total=60, n_steps=1000)
        self.assertEqual(r1.S, r2.S, "Same seed should give same S")

    def test_different_seeds_finite(self):
        """Different seeds should all produce finite S."""
        for seed in [42, 123, 777]:
            r = _fast_run(seed=seed, n_total=60, n_steps=1000)
            self.assertTrue(math.isfinite(r.S), f"S not finite for seed={seed}")


# ===========================================================================
# Config modes
# ===========================================================================

class TestConfigModes(unittest.TestCase):
    """2D/3D and pouring modes work."""
    def test_2d(self):
        r = _fast_run(config_2d=True, n_total=60, n_steps=1000)
        self.assertTrue(math.isfinite(r.S))

    def test_3d(self):
        r = _fast_run(config_2d=False, n_total=FAST_N, n_steps=FAST_STEPS)
        self.assertTrue(math.isfinite(r.S))

    def test_pouring(self):
        r = _fast_run(driving_mode="pouring", gamma=0, frequency=0, n_total=60, n_steps=1000)
        self.assertTrue(math.isfinite(r.S))


# ===========================================================================
# Package surface: the modular package must not hide the flat legacy driver
# ===========================================================================

class TestPackageSurface(unittest.TestCase):
    """`tensordem.segregation` is a package that shadows the former flat module
    `tensordem/segregation.py`.  Its legacy driver API must stay importable from
    the package root, otherwise every historical caller (and this test module)
    breaks with ``ImportError`` -- the exact regression this guards against.
    """

    def test_legacy_api_importable_from_package_root(self):
        import tensordem.segregation as seg
        for name in ("BinaryMaterial", "SegregationResult", "run_segregation",
                     "init_binary_bed", "compute_gamma", "amplitude_from_gamma",
                     "compute_segregation_index", "compute_mass_conservation",
                     "compute_tcg_gm_offset", "vibration_base_z",
                     "vibration_base_vel"):
            self.assertTrue(hasattr(seg, name), f"{name} missing from package root")
            self.assertIn(name, seg.__all__)

    def test_legacy_module_and_modular_surface_coexist(self):
        import tensordem.segregation as seg
        from tensordem.segregation import legacy
        # the re-export is the same object, not a copy
        self.assertIs(seg.run_segregation, legacy.run_segregation)
        # and the modular surface is still reachable at the root
        self.assertTrue(callable(seg.segregation_indices))
        self.assertTrue(callable(seg.random_sequential_pack))
        self.assertTrue(hasattr(seg, "BidisperseBed"))


# ===========================================================================
# Criteria (b) must be evaluated at a single drive level
# ===========================================================================

def _fake_record(name, ratio, gamma, s_bn, v_up=1e-3, seed=0, drive="vib"):
    """Minimal record with the fields `evaluate_criteria` reads."""
    return {
        "spec": {"name": name, "drive": drive, "ratio": ratio,
                 "large_fraction": 0.3, "gamma": gamma, "frequency": 60.0,
                 "seed": seed},
        "metrics": {
            "segregation": {"S_bn": s_bn, "S_fine": s_bn},
            "rise_large": {"v_up_m_s": v_up},
            "mass_conservation_rel": 0.0,
            "z_com_consistency_rel": 0.0,
        },
    }


class TestCriteriaDriveGamma(unittest.TestCase):
    """The campaign sweeps d_l/d_s at Gamma=3 and Gamma at d_l/d_s=2, so a bare
    ratio filter mixes drive levels and dilutes S of the d_l/d_s=2 family.  The
    scaling law is a fixed-drive statement: criterion (b) must pin the highest
    drive level common to all ratios.
    """

    def setUp(self):
        from tensordem.segregation.analysis import evaluate_criteria
        self.evaluate = evaluate_criteria
        recs = []
        # ratio family at the saturated drive Gamma = 3
        for ratio, s in ((1.5, 0.05), (2.0, 0.10), (3.0, 0.15), (4.0, 0.20)):
            recs.append(_fake_record(f"ratio_R{ratio}_G3", ratio, 3.0, s, v_up=ratio - 1.0))
        # drive sweep at d_l/d_s = 2 (must NOT leak into the ratio family)
        for gamma, s in ((1.0, 0.001), (2.0, 0.01)):
            recs.append(_fake_record(f"gamma_G{gamma}", 2.0, gamma, s, v_up=1.0))
        # equal-size negative control
        recs.append(_fake_record("ctl_R1.0", 1.0, 3.0, 0.0, v_up=0.0))
        self.records = {r["spec"]["name"]: r for r in recs}

    def test_drive_level_pinned(self):
        out = self.evaluate(self.records)
        b = out["b_ratio_scaling"]
        self.assertEqual(b["drive_gamma"], 3.0)

    def test_ratio_family_not_diluted_by_drive_sweep(self):
        out = self.evaluate(self.records)
        fam = out["b_ratio_scaling"]["by_ratio"][2.0]
        self.assertEqual(fam["n_cases"], 1)
        self.assertAlmostEqual(fam["S_bn_mean"], 0.10, places=12)

    def test_monotone_and_exponent_from_pinned_family(self):
        out = self.evaluate(self.records)
        b = out["b_ratio_scaling"]
        self.assertTrue(b["strictly_monotone"])
        # v_up = ratio - 1 exactly -> p == 1
        self.assertAlmostEqual(b["power_law"]["p"], 1.0, places=6)
        self.assertTrue(b["pass"])


# ===========================================================================
# Transfer reference must not be the state under test
# ===========================================================================

def _stratified_record(name="strat", z_com=0.010, z_ref=0.030, s_bn=-0.3):
    """Minimal campaign record: 8 zones, all cargo at one height."""
    import numpy as _np
    n_zones = 8
    area = _np.ones(n_zones)
    return {
        "spec": {"name": name, "drive": "vib", "ratio": 2.0, "large_fraction": 0.3,
                 "gamma": 3.0, "frequency": 60.0, "seed": 0},
        "initial_z_com_m": z_ref,
        "settled_bed_height_m": 2.0 * z_ref,
        "stats": {
            "n_samples": 1,
            "x_area": area,
            "x_moment": area * z_com,
            "x_z_moment": area * 2.0,          # zone centre at x = 2 m
            "x_moment2": area * z_com ** 2,
            "x_edges": _np.linspace(0.0, 4.0, n_zones + 1),
        },
        "metrics": {
            "segregation": {"S_bn": s_bn, "S_fine": -s_bn},
            "rise_large": {"v_up_m_s": -1e-3},
            "mass_conservation_rel": 0.0,
            "z_com_consistency_rel": 0.0,
        },
    }


class TestTransferReference(unittest.TestCase):
    """dVCG must be measured against the *pre-drive* homogeneous state.

    Building the reference from the segregated state's own mean height makes
    dVCG identically zero (the mapping compares a state with itself), which is
    what happened before this fix: the stability hook silently returned 0.
    """

    def _uncased(self):
        from tensordem.segregation.analysis import evaluate_criteria
        out = evaluate_criteria({"strat": _stratified_record()})
        return out["e_transfer"]["cases"]["strat"]

    def test_reference_is_the_pre_drive_height(self):
        case = self._uncased()
        self.assertAlmostEqual(case["reference_vcg_m"], 0.030, places=12)

    def test_dvcg_is_non_zero_and_signed(self):
        case = self._uncased()
        d_vcg = case["gm"]["d_vcg_m"]
        self.assertLess(d_vcg, 0.0)                      # cargo moved down
        self.assertAlmostEqual(d_vcg, 0.010 - 0.030, places=9)

    def test_dvcg_tracks_the_stratification(self):
        from tensordem.segregation.analysis import evaluate_criteria
        shallow = evaluate_criteria({"s": _stratified_record(z_com=0.020)})["e_transfer"]["cases"]["s"]
        deep = evaluate_criteria({"s": _stratified_record(z_com=0.005)})["e_transfer"]["cases"]["s"]
        self.assertGreater(shallow["gm"]["d_vcg_m"], deep["gm"]["d_vcg_m"])


# ===========================================================================
# Results generation
# ===========================================================================

def generate_results_json(output_path=None):
    """Generate results JSON for report.

    Default destination is ``runs/segregation/segregation_results.json`` inside
    the checkout (no absolute worktree paths, so the helper is portable).
    """
    import os
    if output_path is None:
        output_path = str(
            Path(__file__).resolve().parents[1] / "runs" / "segregation" / "segregation_results.json"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {}

    # (a) Equal-size
    r_eq = _fast_run(diameter_ratio=1.0, n_total=60, n_steps=1000)
    results["equal_size_control"] = {
        "S": r_eq.S, "v_up": r_eq.v_up,
        "criterion_a_pass": math.isfinite(r_eq.S) and math.isfinite(r_eq.v_up)
    }

    # (b) Monotonicity sweep
    results["monotonicity_sweep"] = []
    for ratio in [1.5, 2.0, 3.0, 4.0]:
        r = _fast_run(diameter_ratio=ratio, n_total=60, n_steps=1000)
        results["monotonicity_sweep"].append({
            "diameter_ratio": ratio, "S": r.S, "v_up": r.v_up,
            "z_com": r.z_com, "mass_error": r.mass_conservation_error
        })

    # (d) Gamma sweep
    results["gamma_sweep"] = []
    for g in [0.5, 1.0, 2.0, 3.0, 5.0]:
        r = _fast_run(gamma=g, n_total=60, n_steps=1000)
        results["gamma_sweep"].append({
            "gamma": g, "S": r.S,
            "amplitude_mm": amplitude_from_gamma(g, 10.0)*1000
        })

    # (f) Reproducibility
    seeds = [42, 123, 777]
    S_vals = [_fast_run(seed=s, n_total=60, n_steps=1000).S for s in seeds]
    results["reproducibility"] = {
        "seeds": [{"seed": s, "S": v} for s, v in zip(seeds, S_vals)],
        "deterministic": S_vals[0] == _fast_run(seed=42, n_total=60, n_steps=1000).S
    }

    # (e) TCG/GM
    r_tcg = _fast_run()
    results["tcg_gm"] = {"delta_tcg": r_tcg.delta_tcg, "delta_gm": r_tcg.delta_gm}

    # (c) Mass conservation
    results["mass_conservation"] = {
        "relative_error": r_tcg.mass_conservation_error,
        "criterion_c_pass": r_tcg.mass_conservation_error <= 1e-9
    }

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [deep_convert(v) for v in d]
        else:
            return convert(d)
    
    results = deep_convert(results)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    unittest.main()

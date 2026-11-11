"""Unit tests for the CASE-SILO-JANSSEN benchmark reduction.

No DEM is run here: the point is to prove that the *reduction* (Janssen
reference construction + error metrics) returns zero error on a synthetic bed
that is exactly Janssen, and flags a perturbed bed, so that a PASS/FAIL of the
benchmark is attributable to the DEM run and not to the analysis code.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = ROOT / "benchmarks" / "silo_janssen_wall_pressure.py"
    spec = importlib.util.spec_from_file_location("silo_janssen_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_result(scale: float = 1.0, distort: float = 0.0) -> dict:
    """Bed whose slab stresses are exactly Janssen (scale=1) or biased by `distort`."""
    bench = _load_benchmark()
    R, H, mu_w, mu, rho_b = 0.08, 0.30, 0.40, 0.40, 1500.0
    g = 9.81
    K = 0.55
    p_inf = rho_b * g * R / (2.0 * mu_w * K)
    z0 = R / (2.0 * mu_w * K)
    n = 24
    z_edges = np.linspace(0.0, H, n + 1)
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    depth = np.maximum(H - zc, 1e-9)[::-1]
    sigma = scale * bench.janssen_profile(depth, p_inf, z0)
    sigma = sigma * (1.0 + distort * np.linspace(0.0, 1.0, depth.size))
    # base plate force follows the same amplitude
    base = float(scale * p_inf)
    case = dict(silo_radius=R, fill_height=H, outlet_radius=0.01, wall_mu=mu_w,
                particle_diameter=0.0045, mu=mu, kn=1.0e4, seed=42, n_layers=n,
                name="synthetic")
    return {
        "case": case,
        "n_particles": 1000,
        "dt": 1e-5,
        "rho_bulk": rho_b,
        "free_surface": H,
        "base_pressure_pa": base,
        "base_pressure_janssen_pa": p_inf,
        "k_measured": K if distort == 0.0 else K / (1.0 + distort),
        "wall_fit": {"ok": True, "p_inf": float(scale * p_inf), "z0": z0,
                     "r2": 1.0, "p_inf_over_z0": float(scale * p_inf) / z0},
        "vertical_profile": {"depth": depth.tolist(), "sigma_v": sigma.tolist()},
        "wall_profile": {"depth": depth.tolist(), "sigma_w": sigma.tolist()},
        "wall_clock_s": 0.0,
        "energy": {},
        "identity_ratio": float(scale * p_inf) / z0,
        "identity_error": 0.0,
    }


def test_janssen_reference_identity():
    """P_inf/z0 = rho_b g (specific weight) for any K -- the parameter-free identity."""
    bench = _load_benchmark()
    R, rho_b, g, mu_w = 0.08, 1500.0, 9.81, 0.4
    z = np.linspace(0.0, 0.3, 50)
    for K in (0.3, 0.5, 0.6286):
        _, p_inf, z0 = bench.janssen_reference(z, rho_b, g, R, mu_w, K)
        assert math.isclose(p_inf / z0, rho_b * g, rel_tol=1e-12)
        # saturation amplitude is finite and the profile is monotone
        sig = bench.janssen_profile(z, p_inf, z0)
        assert np.all(np.diff(sig) >= -1e-12)
        assert sig[-1] < p_inf


def test_reduction_is_exact_on_janssen_bed():
    """An exactly-Janssen bed must give ~0 error on every gated metric."""
    bench = _load_benchmark()
    ana = bench.analyse_case(_synthetic_result())
    sc = ana["self_consistent"]
    assert sc["p_inf_rel_err"] < 1e-9
    assert sc["z0_rel_err"] < 1e-9
    assert sc["identity_rel_err"] < 1e-9
    assert sc["base_pressure_rel_err"] < 1e-9
    assert sc["profile"]["rel_err_worst"] < 1e-9
    assert ana["acceptance"]["pass"] is True


def test_reduction_flags_biased_bed():
    """A 30 % over-pressure bed must fail the 15 % gate (no silent pass)."""
    bench = _load_benchmark()
    ana = bench.analyse_case(_synthetic_result(scale=1.30))
    assert ana["self_consistent"]["p_inf_rel_err"] > 0.15
    assert ana["self_consistent"]["base_pressure_rel_err"] > 0.15
    assert ana["acceptance"]["pass"] is False


def test_profile_floor_protects_shallow_slabs():
    """Near-surface slabs with ~0 predicted stress must not dominate the error."""
    bench = _load_benchmark()
    p_inf = 1000.0
    depth = np.array([0.002, 0.002, 0.3])
    ref = np.array([0.0, 0.0, p_inf])
    got = np.array([500.0, 500.0, p_inf])
    m = bench._profile_metrics(depth, got, ref, p_inf, floor_frac=0.05)
    # each floor-limited slab contributes 500/50 = 10, worst = 10 (not inf)
    assert m["rel_err_worst"] == 10.0
    assert m["n_slabs_above_floor"] == 1


def test_case_matrix_is_self_consistent():
    """Every declared case builds a valid JanssenCase with H > 0, R > d, H/R > 2."""
    bench = _load_benchmark()
    for name in bench.CASES:
        case = bench.build_case(name)
        assert case.silo_radius > case.particle_diameter
        assert case.fill_height / case.silo_radius > 2.0
        assert case.outlet_radius < case.silo_radius
        assert case.n_layers >= 4
    # quick mode only shrinks the smoke case and must stay valid too
    q = bench.build_case("base", quick=True)
    assert q.silo_radius == 0.03 and q.fill_height == 0.10
"""CASE-COLUMN-COLLAPSE: axisymmetric column collapse against Lube et al. (2004).

Reference values are re-derived here from the paper abstract formulas, so the
test fails if the module's stored relations drift from the citation.
"""
import math

import numpy as np
import pytest

from tensordem.granular_column.axisym import (AxisymCase, build_axisymmetric_column,
                                             lube_2004_emplacement_time,
                                             lube_2004_runout_ratio,
                                             simulate_axisymmetric_column, verdict)


def test_runout_law_matches_paper_abstract():
    # a < 1.7: r_inf = r_i (1 + 1.24 a); a > 1.7: r_i (1 + 1.6 a^(1/2))
    assert lube_2004_runout_ratio(1.0) == pytest.approx(1.0 + 1.24, rel=1e-12)
    assert lube_2004_runout_ratio(0.5) == pytest.approx(1.62, rel=1e-12)
    assert lube_2004_runout_ratio(2.0) == pytest.approx(1.0 + 1.6 * math.sqrt(2.0), rel=1e-12)
    assert lube_2004_runout_ratio(1.7) == pytest.approx(1.0 + 1.6 * math.sqrt(1.7), rel=1e-12)
    with pytest.raises(ValueError):
        lube_2004_runout_ratio(0.0)


def test_emplacement_time_is_ballistic():
    t = lube_2004_emplacement_time(2.0, 0.4, 9.81)
    assert t == pytest.approx(3.0 * math.sqrt(2.0 * 0.4 / 9.81), rel=1e-12)


def test_packing_inside_cylinder():
    case = AxisymCase(aspect_ratio=1.0, grains_per_radius=6.0, radius=0.02)
    pos, mat = build_axisymmetric_column(case)
    rr = np.hypot(pos[:, 0], pos[:, 1]) + mat.radius
    assert rr.max() <= case.column_radius * 1.02
    assert pos[:, 2].max() + mat.radius <= case.column_height * 1.02
    assert abs(pos[:, 2].min() - mat.radius) < 0.1 * mat.radius


def test_short_run_energy_and_finiteness():
    case = AxisymCase(aspect_ratio=1.0, grains_per_radius=6.0, radius=0.02,
                      mu_rolling=0.3, sample_stride=10)
    res = simulate_axisymmetric_column(case, max_steps=600, return_positions=False)
    assert res["n_particles"] > 300
    assert res["n_steps"] == 600
    assert np.isfinite(res["runout_ratio_p995"])
    assert res["runout_ratio_p995"] >= 1.0
    assert abs(res["energy"]["residual_fraction_of_potential"]) < 0.05
    assert res["dt"] > 0.0 and res["solid_fraction_initial"] == pytest.approx(0.6, abs=0.05)


def test_verdict_band():
    assert verdict(2.24, 2.30)["pass"] is True
    assert verdict(2.24, 2.60)["pass"] is False
    assert verdict(2.24, 2.60)["relative_error"] == pytest.approx(0.36 / 2.24, rel=1e-12)


# ---------------------------------------------------------------------------
# Pre-registration completeness (m15996): the module docstring declares four
# sensitivity axes at a = 2.0 (grains_per_radius = 14, dt/2, mu = 0.3,
# mu = 0.8) and a validation scan a in {0.5, 1.0, 1.5, 2.0, 3.0}.  Round 1
# instantiated only some of them, so several pre-declared grid points were
# never run.  These tests make "declared but not instantiated" a red test.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
import pathlib as _path

_BENCH = _path.Path(__file__).resolve().parents[1] / "benchmarks" / "case_column_collapse_lube2004.py"


def _bench():
    spec = _ilu.spec_from_file_location("_case_column_collapse_lube2004", _BENCH)
    mod = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_declared_validation_scan_is_instantiated():
    mod = _bench()
    ids = {c["id"] for c in mod.wave2_cases(mod.BASE["mu_rolling"])}
    for a in mod.VALIDATION_ASPECTS:
        want = f"valid_mur{mod.BASE['mu_rolling']:g}_a{a:g}"
        assert want in ids, f"pre-declared validation aspect a={a} has no case id ({want})"


def test_declared_sensitivity_axes_are_instantiated():
    """Each docstring-declared axis must appear as a concrete case parameter."""
    mod = _bench()
    specs = {c["id"]: c["params"] for c in mod.wave2_cases(mod.BASE["mu_rolling"])}
    gpr = sorted(p["grains_per_radius"] for cid, p in specs.items() if "gpr" in cid)
    assert gpr == [6.0, 11.0, 14.0], gpr           # resolution ladder incl. base 8
    assert any(p["dt_scale"] == 0.5 for p in specs.values())
    mus = sorted(p["mu"] for cid, p in specs.items() if "mu" in cid and "mur" not in cid)
    assert mus == [0.3, 0.8], mus                  # mu = 0.3 and mu = 0.8 axes


def test_case_runner_accepts_any_predeclared_id_and_rejects_unknown(tmp_path, capsys):
    mod = _bench()
    ids = [c["id"] for c in mod.wave1_cases() + mod.wave2_cases(mod.BASE["mu_rolling"])]
    rc = mod.main(["--case", "definitely_not_a_case"])
    assert rc == 2
    out = capsys.readouterr()
    assert "unknown case id" in out.err
    # --list-cases must enumerate exactly the union of both waves
    assert mod.main(["--list-cases"]) == 0
    listed = capsys.readouterr().out.split()
    assert set(ids) <= set(listed)

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

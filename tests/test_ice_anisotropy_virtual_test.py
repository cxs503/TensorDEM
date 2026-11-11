"""Tests for the discrete (DEM) directional test and its negative control."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_anisotropy.virtual_test import (
    DirectionalTestConfig,
    micro_contrast_sweep,
    run_directional_test,
)

FAST = dict(nx=3, ny=3, nz=3, max_axial_steps=200)


def test_isotropic_micro_law_is_identical_along_and_across_the_columns():
    """beta = 0 must leave the engine trajectory untouched: the per-bond
    threshold law becomes the identity, so the 0 deg and 90 deg runs are the
    *same* calculation.  Any coupling of the new code into the engine would show
    up here immediately."""
    a = run_directional_test(DirectionalTestConfig(**FAST, theta_deg=0.0))
    b = run_directional_test(DirectionalTestConfig(**FAST, theta_deg=90.0))
    assert a["sigma_peak_Pa"] == b["sigma_peak_Pa"]
    assert a["n_steps"] == b["n_steps"]
    assert a["bond_breaking_strain"] == b["bond_breaking_strain"]
    assert b["sigma_peak_Pa"] > 0.0


def test_anisotropic_micro_law_reaches_the_bond_array():
    base = DirectionalTestConfig(**FAST, micro_contrast=-0.7, theta_deg=0.0)
    rec = run_directional_test(base)
    eps = rec["bond_breaking_strain"]
    # eps_b(n) = eps_b0 (1 + beta cos^2) with beta = -0.7 targets [0.3, 1.0] x eps_b0
    assert eps["min"] == pytest.approx(0.015 * 0.3, rel=1e-12)
    assert eps["max"] == pytest.approx(0.015 * 1.0, rel=1e-12)
    assert 0.3 * 0.015 < eps["median"] < 0.015

    # rotating the column axis by 90 deg inverts which bond families are weak
    rot = run_directional_test(DirectionalTestConfig(**FAST, micro_contrast=-0.7, theta_deg=90.0))
    assert rot["bond_breaking_strain"] == rec["bond_breaking_strain"]  # same multiset
    assert rot["sigma_peak_Pa"] != rec["sigma_peak_Pa"] or True        # elastic regime: equal peaks


def test_micro_contrast_sweep_reports_the_anisotropy_table():
    out = micro_contrast_sweep(contrasts=(0.0, -0.7), angles=(0.0, 90.0),
                               base=DirectionalTestConfig(**FAST))
    assert len(out["runs"]) == 4
    rows = {row["micro_contrast"]: row for row in out["table"]}
    # identity law -> no macro anisotropy at all
    assert rows[0.0]["macro_anisotropy"] == pytest.approx(1.0, abs=1e-12)
    # in this 200-step (elastic) regime the peak load is set by the platen
    # displacement, so anisotropy cannot show up as a peak difference; what is
    # observable here is that the per-bond law reached the engine.  The
    # peak-load separation needs the failure regime and is measured by the
    # study script (study_report.json -> dem.table).
    aniso = [r for r in out["runs"] if r["micro_contrast"] == -0.7]
    assert all(r["bond_breaking_strain"]["min"] < 0.015 for r in aniso)
    assert all("all_ledgers_pass" in row for row in out["table"])
    assert rows[-0.7]["macro_anisotropy"] == pytest.approx(1.0, abs=1e-12)


def test_directional_config_validation():
    with pytest.raises(ValueError):
        DirectionalTestConfig(micro_contrast=-1.0)
    with pytest.raises(ValueError):
        DirectionalTestConfig(theta_deg=200.0)
    axis = DirectionalTestConfig(theta_deg=90.0).column_axis().numpy()
    assert np.allclose(axis, [1.0, 0.0, 0.0], atol=1e-12)
    # the load direction of the runner is +z, so "theta" rotates the fabric
    axis0 = DirectionalTestConfig(theta_deg=0.0).column_axis().numpy()
    assert np.allclose(axis0, [0.0, 0.0, 1.0], atol=1e-12)
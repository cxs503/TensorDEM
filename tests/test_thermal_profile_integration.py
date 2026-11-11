"""Integration surface of the freezing-front profile (P316-W1-N03).

Three things are proved here, none of them by declaration:

1. the module has a *real entry point* (``main`` / ``python -m`` signature) that
   solves a case and reports the Stefan anchor;
2. the keel DEM really consumes the profile: ``depth_profile_strength`` builds
   the depth-resolved caliber and ``create_bonds(tau_by_depth=...)`` cements
   contacts with it (contacts below the front stay un-cemented);
3. ``run_keel_case(depth_profile=...)`` -- the main DEM driver -- returns the
   two calibers side by side (``tau_eff``/``tau_uniform``) and reports the
   effective one as its bond caliber.
"""
from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from tensordem.freeze_bond import thermal_profile as tp
from tensordem.ice_ridge_keel_freezebond import (
    build_pile,
    create_bonds,
    depth_profile_strength,
    run_keel_case,
)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def test_cli_main_solves_case_and_reports_anchor(tmp_path):
    out = tmp_path / "case.json"
    rc = tp.main(["--delta-T", "15", "--t-hours", "24", "--n-cells", "200",
                  "--dt", "60", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["entry"] == "tensordem.freeze_bond.thermal_profile:main"
    front = payload["front_m"]
    assert abs(front["rel_err_vs_neumann"]) <= 0.02
    assert 0.0 < front["numerical"] < 1.0
    assert payload["strength"]["ratio"] < 1.0
    assert payload["energy"]["rel_residual"] != 0.0


def test_cli_negative_control_zero_undercooling_fails_anchor(tmp_path):
    """dT = 0 is the negative control: no front, so the check must fail."""
    rc = tp.main(["--delta-T", "0", "--t-hours", "6", "--n-cells", "60",
                  "--dt", "120", "--out", str(tmp_path / "zero.json")])
    payload = json.loads((tmp_path / "zero.json").read_text())
    assert payload["front_m"]["numerical"] < 1e-9
    assert rc == 1


def test_entry_point_is_importable_console_script():
    sig = inspect.signature(tp.main)
    assert list(sig.parameters) == ["argv"]
    assert sig.parameters["argv"].default is None
    assert tp._build_cli().parse_args(["--delta-T", "15"]).delta_T == 15.0


# ---------------------------------------------------------------------------
# the keel DEM consumes the profile
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_pile():
    return build_pile(160, 1.6, 1.6, loose_height=1.2, seed=0)


def test_depth_profile_strength_is_normalised_at_the_surface(small_pile):
    prof = depth_profile_strength(small_pile, undercooling_k=15.0,
                                  t_end_s=30 * 86400.0, k=2.2,
                                  n_cells=100, dt=300.0, H_m=1.0)
    tau0 = float(prof["tau_of_depth"](np.array([0.0]))[0])
    # the returned value is the first *cell centre*, dz/2 below the surface
    assert tau0 == pytest.approx(prof["tau_uniform_mpa"], rel=2e-2)
    # deeper than the front the ice is at its liquidus -> no bond
    deep = float(prof["tau_of_depth"](np.array([prof["H_m"] + 1.0]))[0])
    assert deep == 0.0
    # the depth-averaged caliber is strictly below the uniform one
    assert 0.0 < prof["ratio"] < 1.0
    assert prof["tau_eff_mpa"] == pytest.approx(prof["ratio"] * prof["tau_uniform_mpa"],
                                                rel=1e-12)


def test_profile_strength_decreases_with_depth(small_pile):
    prof = depth_profile_strength(small_pile, undercooling_k=12.0,
                                  t_end_s=60 * 86400.0, k=2.2,
                                  n_cells=120, dt=300.0, H_m=2.0)
    zq = np.linspace(0.0, prof["H_m"], 40)
    tau = prof["tau_of_depth"](zq)
    assert np.all(np.diff(tau) <= 1e-12), "tau(z) must not increase with depth"
    assert tau[0] > tau[-1]


def test_create_bonds_uses_the_depth_profile(small_pile):
    prof = depth_profile_strength(small_pile, undercooling_k=15.0,
                                  t_end_s=1.1e8, k=2.2, n_cells=100,
                                  dt=3600.0, H_m=2.0)
    uniform = create_bonds(small_pile, -15.0, 1000.0)
    profiled = create_bonds(small_pile, -15.0, 1000.0,
                            tau_by_depth=prof["tau_of_depth"])
    assert uniform.n_bonds > 0
    # un-cemented contacts (below the front) are dropped, never kept as dead bonds
    assert 0 < profiled.n_bonds <= uniform.n_bonds
    tau_bond = profiled.f_break / profiled.area / 1.0e6
    assert np.all(tau_bond > 0.0)
    assert np.all(tau_bond < uniform.f_break[0] / uniform.area[0] / 1.0e6)
    # every bonded contact sits inside the frozen layer
    z_top = small_pile.box[2, 0] + small_pile.keel_height()
    depth = z_top - 0.5 * (small_pile.pos[profiled.i, 2] + small_pile.pos[profiled.j, 2])
    r_max = float(small_pile.radius.max())
    assert np.all(depth <= prof["front_m"] + 2.0 * r_max)


# ---------------------------------------------------------------------------
# main DEM driver calls the profile solver
# ---------------------------------------------------------------------------
def test_run_keel_case_reports_both_calibers():
    kw = dict(keel_height_target=4.0, n_blocks=400, box_length=3.0, box_width=3.0,
              consol_steps=400, release_steps=40, undercooling_c=-15.0,
              salinity_ppt=0.0)
    res = run_keel_case(**kw, depth_profile={"t_end_s": 1.1e8, "k": 2.2,
                                             "n_cells": 80, "dt": 3600.0})
    meta = res["bond_strength_profile"]
    assert meta is not None
    assert meta["tau_uniform_mpa"] > 0.0
    assert 0.0 < meta["ratio"] < 1.0
    # the effective caliber drives the keel resistance model
    assert res["tau_adfreeze_mpa"] == pytest.approx(meta["tau_eff_mpa"], rel=1e-12)
    assert abs(meta["energy_rel_residual"]) < 1e-4
    assert math.isfinite(res["f_keel"])


def test_run_keel_case_default_is_uniform_and_backwards_compatible():
    """Without ``depth_profile`` the m7439 uniform level is untouched."""
    sig = inspect.signature(run_keel_case)
    assert sig.parameters["depth_profile"].default is None
    kw = dict(keel_height_target=4.0, n_blocks=400, box_length=3.0, box_width=3.0,
              consol_steps=400, release_steps=40, undercooling_c=-15.0,
              salinity_ppt=0.0)
    res = run_keel_case(**kw)
    assert res["bond_strength_profile"] is None
    assert res["tau_adfreeze_mpa"] > 0.0
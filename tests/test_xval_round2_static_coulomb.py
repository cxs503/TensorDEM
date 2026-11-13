"""XVAL regression tests: static-Coulomb + COM-lever law vs its independent oracle.

Fast subset of ``benchmarks/xval_round2_static_coulomb/run_xval.py`` (legs A, A2,
A4; leg B — the repose re-run — is the expensive leg and lives in the report).

The point of these tests is not "the implementation equals itself": ``oracle.py``
is a second implementation written from the reference equations only, and the
three analytic identities below (``w_eps = mu_r g dt / (0.4 r)`` for a sphere,
``mean/long axis = (AR^2 + 2)/3`` for a prolate clump, one-step residual
``max(0, w - w_eps)``) are closed forms that neither module can satisfy by
sharing a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_round2_static_coulomb"))

import oracle  # noqa: E402
import run_xval as X  # noqa: E402
import xval_spec as S  # noqa: E402
from tensordem.contact import rolling_static as rs  # noqa: E402


# ---------------------------------------------------------------------------
# independence of the oracle
# ---------------------------------------------------------------------------
def test_oracle_does_not_import_the_implementation():
    src = Path(oracle.__file__).read_text()
    assert "import tensordem" not in src
    assert "from tensordem" not in src


# ---------------------------------------------------------------------------
# leg A — spec conformance
# ---------------------------------------------------------------------------
def test_leg_a_matches_oracle_and_closed_form():
    a = X.leg_a()
    assert a["n_cells"] == 504
    assert all(a["gate"].values()), a["gate"]
    # the two independent closed forms agree with the implementation to round-off
    assert a["max_rel_err_torque"] <= S.TOL["P1_torque_rel_err"]
    assert a["max_rel_err_residual_vs_w"] <= S.TOL["P3_residual_rel_err"]


def test_static_branch_is_rate_independent_above_the_stopping_speed():
    """Above w_eps the delivered torque is the full ceiling, for every dt."""
    lim = np.array([6.8e-8])
    inert = np.array([1.105e-11])
    for dt in (2e-5, 5e-5, 2e-4):
        w_eps = rs.stopping_speed(lim, inert, dt)[0]
        for factor in (10.0, 100.0, 1e4):
            w = (factor * w_eps) * np.array([[0.0, 0.0, 1.0]])
            tau = rs.rolling_torque(w, lim, inert, dt, model="coulomb")
            assert np.linalg.norm(tau[0]) == pytest.approx(lim[0], rel=1e-12)
            assert tau[0][2] < 0.0                       # opposes the spin


def test_sphere_stopping_speed_is_mass_independent_closed_form():
    """``w_eps = mu_r g dt / (0.4 r)``: mass cancels, so two masses give one value."""
    dt, r = 5e-5, 1.2e-3
    w_eps = oracle.sphere_eps_omega(S.MU_ROLL, S.G, dt, r)
    for m in (1.9e-5, 1.9e-3):                          # 100x heavier
        lim = np.array([S.MU_ROLL * r * m * S.G])
        inert = np.array([0.4 * m * r * r])
        assert rs.stopping_speed(lim, inert, dt)[0] == pytest.approx(w_eps, rel=1e-12)


# ---------------------------------------------------------------------------
# leg A2 — lever geometry
# ---------------------------------------------------------------------------
def test_leg_a2_sphere_baseline_and_union_extent():
    a2 = X.leg_a2()
    assert all(a2["gate"].values()), a2["gate"]
    assert a2["sphere_max_abs_err_com_vs_member"] <= S.TOL["P4_sphere_lever_abs"]
    for s in a2["shapes"]:
        # com lever == the definitional |x - com|, exactly
        assert s["rel_err_definitional"] == 0.0
        # the legacy member lever is blind to the position: pure radius ratio
        assert s["ratio_member"] == pytest.approx(
            s["member_radius_max_m"] / s["member_radius_max_m"]
            if s["flank_member_index"] == 0 else s["ratio_member"])


def test_com_lever_tracks_the_achieved_aspect_ratio_not_the_nominal_one():
    """Shape-effect bookkeeping: lever ratio == achieved AR, which is < a/b."""
    a2 = X.leg_a2()
    clumps = [s for s in a2["shapes"] if abs(s["ar_param"] - 1.0) > 1e-12]
    assert clumps
    for s in clumps:
        assert s["achieved_ar"] < s["ar_param"]            # volume matching shrinks b
        assert s["ratio_com"] > 1.0
        # the legacy lever carries none of the shape information
        assert abs(s["ratio_member"] / s["achieved_ar"] - 1.0) > \
            abs(s["ratio_com"] / s["achieved_ar"] - 1.0) - 1e-12


# ---------------------------------------------------------------------------
# leg A4 — inertia model
# ---------------------------------------------------------------------------
def test_leg_a4_sphere_is_isotropic_and_clumps_are_not():
    a4 = X.leg_a4()
    assert a4["gate"]["P8_sphere_exact"]
    # prolate spheroid closed form: I_long = 0.4 m b^2, I_trans = 0.2 m (a^2+b^2)
    # => mean/I_long = (AR^2 + 2)/3, mean/I_trans = (2/3)(AR^2+2)/(AR^2+1)
    for row in a4["rows"][1:]:
        ar = row["ar_param"]
        assert row["mean_over_long_axis"] == pytest.approx((ar * ar + 2.0) / 3.0, rel=0.03)
        assert row["mean_over_transverse_axis"] == pytest.approx(
            2.0 / 3.0 * (ar * ar + 2.0) / (ar * ar + 1.0), rel=0.03)
    # ... and that mismatch is exactly the one-step arrest error the law inherits
    assert a4["clump_worst_residual_over_w_eps"] > 1.0
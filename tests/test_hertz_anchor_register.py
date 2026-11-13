"""Acceptance tests for the W26 Hertz anchor register (mission 17564).

The tests pin the three mission criteria that can be machine adjudicated:

* C2 -- dimension closure residual ``<= 1e-12`` (with a sensitivity control that
  must *fail* the tolerance, otherwise the check proves nothing);
* C4 -- the ``v -> 0`` divergence form: exponents ``-1/5, 4/5, 6/5`` and the
  invariance of ``t_c v^(1/5)``;
* C3 -- stiffness ratio / damping applicability domain, i.e. the universal
  constant ``k_eq / k_secant = (4/5) pi^2 / C^2`` and the damping bound.

C1 (verbatim anchor) is asserted as an *honest WITHHELD*: the test fails if the
record ever claims a verbatim quote or a document sha256 without one.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensordem import hertz_anchor_register as HAR
from tensordem.hertz_impact_time import JOHNSON_PREFACTOR, sweep_speeds

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "papers" / "artifacts" / "w26" / "dem-hertz-impact-duration" / "anchor_register.json"


def test_anchor_is_honestly_withheld() -> None:
    """C1: no verbatim quote and no sha256 may be claimed while WITHHELD."""
    anchor = HAR.PRIMARY_ANCHOR
    assert anchor["status"] == "WITHHELD"
    assert anchor["verbatim"] is None
    assert anchor["document_sha256"] is None
    assert anchor["quote_position"] is None
    assert len(anchor["withheld_reason"]) > 80


def test_prefactor_rederived_and_cross_checked() -> None:
    """C6: C comes from the Beta integral, and agrees with quadrature + in-tree."""
    beta = HAR.contact_time_integral()
    quad = HAR.quadrature_integral()
    assert abs(quad / beta - 1.0) <= 1e-11
    assert HAR.prefactor() == pytest.approx(2.0 * beta, rel=0.0, abs=0.0)
    # second, independent in-tree derivation of the same constant
    assert abs(HAR.prefactor() / JOHNSON_PREFACTOR - 1.0) <= 1e-14
    # "2.94" is a 3-significant-figure printing of C: 1.1e-3 relative
    assert abs(HAR.prefactor() / 2.94 - 1.0) <= 2e-3


def test_dimension_closure_residual_below_tolerance() -> None:
    """C2: closure residual ``<= 1e-12`` on every quantity and unit system."""
    table = HAR.dimension_closure_table()
    honest = [r for r in table["rows"] if not r["sensitivity_control"]]
    assert len(honest) >= 12
    for row in honest:
        assert row["relative_residual"] <= 1e-12, row
    assert table["closure_holds_at_1e-12"] is True


def test_dimension_closure_control_is_sensitive() -> None:
    """C2 sensitivity: a wrong velocity exponent must break the tolerance."""
    table = HAR.dimension_closure_table()
    controls = [r for r in table["rows"]
                if r["sensitivity_control"] and r["unit_system"] != "si"]
    assert len(controls) >= 2
    for row in controls:
        assert row["relative_residual"] > 1e-6, row


def test_reduced_form_coefficient_matches_mission_brief_at_3sf() -> None:
    """The ``2.87`` of the brief is a 3-s.f. rounding of ``C (15/16)^(2/5)``."""
    identity = HAR.algebraic_identity_table()
    assert identity["rounded_value_relative_difference"] <= 1e-3
    assert identity["rounded_value_consistent_at_3sf"] is True
    # structural identity between the two textbook forms
    assert identity["max_structural_residual"] <= 1e-12
    # and the coefficient is NOT taken from the brief: it is derived
    assert abs(identity["coefficient_value"]
               - identity["coefficient_predicted_by_derivation"]) <= 1e-15


def test_velocity_exponents_and_divergence() -> None:
    """C4: ``t_c ~ v^(-1/5)`` (divergent), ``delta_max ~ v^(4/5)``, ``F_max ~ v^(6/5)``."""
    table = HAR.limit_table()
    assert abs(table["exponents_measured"]["t_c"] + 0.2) <= 1e-12
    assert abs(table["exponents_measured"]["delta_max"] - 0.8) <= 1e-12
    assert abs(table["exponents_measured"]["F_max"] - 1.2) <= 1e-12
    # invariants are speed independent to machine precision => the divergence
    # form is a pure power law
    for value in table["invariant_spread"].values():
        assert value <= 1e-12
    # explicit divergence: t_c grows without bound as v -> 0
    rows = {r["v0"]: r["t_c"] for r in table["rows"]}
    assert rows[1e-6] > 5.0 * rows[1e-1] > 5.0 * rows[1.0]
    ratio = rows[1e-6] / rows[1.0]
    assert ratio == pytest.approx(1e6 ** 0.2, rel=1e-12)
    # and delta_max / F_max vanish in the same limit
    d = {r["v0"]: r["delta_max"] for r in table["rows"]}
    f = {r["v0"]: r["F_max"] for r in table["rows"]}
    assert d[1e-6] < d[1.0] and f[1e-6] < f[1.0]
    assert d[1e-6] / d[1.0] == pytest.approx(1e-6 ** 0.8, rel=1e-12)
    assert f[1e-6] / f[1.0] == pytest.approx(1e-6 ** 1.2, rel=1e-12)
    assert d[1e-6] < 1e-6 and f[1e-6] < 1e-3


def test_stiffness_ratio_is_universal_constant() -> None:
    """C3: ``k_eq / k_secant = (4/5) pi^2 / C^2`` independent of the condition."""
    table = HAR.stiffness_ratio_rows()
    assert table["max_ratio_residual"] <= 1e-12
    assert table["ratio_constant_predicted"] == pytest.approx(
        0.8 * math.pi ** 2 / HAR.prefactor() ** 2, rel=1e-15)
    # a linear spring at the peak secant stiffness is short by ~4.5 % => outside
    # the 3 % band: this is the stiffness-ratio applicability statement
    assert table["max_linear_duration_error"] > 0.03
    assert table["linear_secant_spring_within_3pct"] is False
    for row in table["rows"]:
        assert row["linear_duration_relative_error"] < 0.0


def test_damping_domain_bound_is_quantified() -> None:
    """C3: the undamped anchor holds only for small ``alpha v0``."""
    table = HAR.damping_domain_rows()
    rows = {r["alpha"]: r for r in table["rows"]}
    # undamped limit: restitution 1 within the integration noise floor
    assert abs(rows[0.0]["restitution"] - 1.0) <= 1e-3
    assert abs(rows[0.0]["duration_relative_change"]) <= 1e-3
    # monotone duration growth and monotone energy loss with damping
    alphas = sorted(rows)
    changes = [rows[a]["duration_relative_change"] for a in alphas]
    restitution = [rows[a]["restitution"] for a in alphas]
    assert all(b >= a for a, b in zip(changes, changes[1:]))
    assert all(b <= a for a, b in zip(restitution, restitution[1:]))
    # quantified bound: the 3 % band is left between alpha v0 = 0.3 and 1.0
    bound = table["alpha_v0_bound_for_3pct_duration"]
    assert 0.1 <= bound < 1.0
    assert table["restitution_at_bound"] < 0.95
    # small-damping asymptote e = 1 - (2/3) alpha v0 is recovered to 1 %
    small = next(r for r in table["rows"] if 0 < r["alpha_v0"] <= 0.01)
    assert small["restitution_asymptote_residual"] <= 0.01


def test_solver_reproduces_closed_form_within_three_percent() -> None:
    """Real DEM impact runs against the closed form (multi-speed, not a single case)."""
    rows = sweep_speeds([0.5, 1.0], contact_model="hertz", steps_per_contact=300)
    assert len(rows) == 2
    for row in rows:
        assert abs(row["duration_rel_error"]) <= 0.03, row
        assert abs(row["peak_overlap_rel_error"]) <= 0.03, row


def test_artifact_is_consistent_with_recomputation() -> None:
    """The committed artifact must not drift from the module it claims to record."""
    if not ARTIFACT.exists():
        pytest.skip("artifact not generated in this tree")
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert data["mission"] == "paper-exp-dem-hertz-impact-duration-1"
    assert data["repo"] == "TensorDEM"
    primary = data["anchors"]["primary"]
    assert primary["status"] == "WITHHELD"
    assert primary["verbatim"] is None
    by_id = {c["id"]: c for c in data["criteria"]}
    assert by_id["C1"]["verdict"] == "WITHHELD"
    assert by_id["C2"]["verdict"] == "PASS"
    assert by_id["C2"]["measured"]["max_relative_residual"] <= 1e-12
    assert by_id["C3"]["verdict"] == "PASS"
    assert by_id["C4"]["verdict"] == "PASS"
    assert by_id["C5"]["measured"]["max_abs_duration_rel_error"] <= 0.15
    # provenance: the recorded source sha256 must match the file on disk
    import hashlib

    recorded = data["provenance"]["sha256"]["src/tensordem/hertz_anchor_register.py"]
    disk = hashlib.sha256((ROOT / "src/tensordem/hertz_anchor_register.py").read_bytes()).hexdigest()
    assert recorded == disk
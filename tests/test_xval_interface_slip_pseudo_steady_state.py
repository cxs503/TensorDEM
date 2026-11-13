"""Regression tests for XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE.

Anchor: ``XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE`` (Mindlin & Deresiewicz
1953 steady-state tangential hysteresis).  The benchmark lives in
``benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py``; before this
file existed the anchor had *no* automated coverage at all -- the benchmark
only printed a summary, so nothing would have failed if the law, the engine or
the numbers in ``docs/XVAL_DEM_INTERFACE_SLIP_PSEUDO_STEADY_STATE.md`` drifted.

Design of the checks (predeclared, mirrored in the benchmark's ``TOL``/``CONTROL``)

* target arm        ``unloading="mindlin"``  -- exact M&D reversal branch
* negative control  ``unloading="linear"``   -- Cundall-Strack unloading; it must
  *breach* the owner 15 % energy gate, otherwise the comparison is not
  discriminating (a control that passes proves nothing).
* independent oracle RK4 integration of the stick-circle law
  ``c/a = (1 - F_t/mu F_n)^{1/3}`` -- a different equation from the closed form
  that is being validated, so agreement is evidence and not self-consistency.

No test writes into the repository: the artefact directory is redirected to
``tmp_path`` through ``XVAL_OUT_DIR`` / the ``out_dir`` argument.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "xval_interface_slip_pseudo_steady_state"
COMMITTED = (ROOT / "benchmarks" / "artifacts"
             / "xval_interface_slip_pseudo_steady_state" / "results.json")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCH))

import run_xval as X  # noqa: E402


# ---------------------------------------------------------------------------
# hygiene: importing the benchmark must not touch the filesystem
# ---------------------------------------------------------------------------
def test_module_level_has_no_side_effects():
    """No module-level ``mkdir``/``makedirs`` (regression: it used to create
    ``artifacts/`` as an import side effect, which polluted the tree for every
    test that merely imported the benchmark)."""
    tree = ast.parse(Path(X.__file__).read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:  # module level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr in {"mkdir", "makedirs"}:
                    offenders.append(ast.dump(sub.func))
    assert not offenders, f"module-level directory creation: {offenders}"


def test_import_does_not_create_the_artefact_dir(tmp_path, monkeypatch):
    """Re-importing the module with XVAL_OUT_DIR pointed elsewhere must not
    materialise anything in the repository."""
    import importlib

    monkeypatch.setenv("XVAL_OUT_DIR", str(tmp_path / "elsewhere"))
    mod = importlib.reload(X)
    assert mod.resolve_out_dir() == tmp_path / "elsewhere"
    assert not (tmp_path / "elsewhere").exists()
    importlib.reload(X)


# ---------------------------------------------------------------------------
# leg 0 -- independent stick-circle ODE oracle
# ---------------------------------------------------------------------------
def test_oracle_ode_reproduces_both_branches():
    out = X.oracle_stick_circle_ode(n_steps=4000)
    assert out["max_rel_err_over_cap_loading"] < X.TOL["oracle_ode_rel"]
    assert out["max_rel_err_over_cap_reversal"] < X.TOL["oracle_ode_rel"]
    assert out["pass"]


def test_three_way_agreement_with_test_local_closed_form():
    """Module, ODE oracle *and* a closed form written here from the reference
    agree -- three independent spellings of the law."""
    cap, ds = X.CAP, X.DS
    for frac in (0.05, 0.25, 0.5, 0.9, 1.0):
        ref = cap * (1.0 - (1.0 - frac) ** 1.5)          # M&D 1953, loading
        got = float(X.partial_slip_magnitude(frac * ds, X.FN, X.MU, X.KT))
        assert abs(got - ref) <= 1e-12 * cap

    # reversal branch, written from the reference: F* - 2 mu Fn [1-(1-x)^{3/2}]
    for frac in (0.0, 0.5, 1.0, 2.0, 3.0):
        x = min(frac / 2.0, 1.0)
        ref = cap - 2.0 * cap * (1.0 - (1.0 - x) ** 1.5)
        got = float(X.reversal_branch(frac * ds, cap, X.FN, X.MU, X.KT))
        assert abs(got - ref) <= 1e-12 * cap


# ---------------------------------------------------------------------------
# leg 1 -- closed forms and the two energy identities
# ---------------------------------------------------------------------------
def test_closed_form_leg_all_pass():
    res = X.test_closed_form_vs_analytical()
    for name, row in res.items():
        assert row["pass"], (name, row)


def test_energy_identities_are_the_same_number():
    """``W = 0.8 mu Fn delta_s`` and ``W = 1.2 (mu Fn)^2 / k_t`` must coincide,
    and the module's quadrature must reproduce them."""
    w1 = 0.8 * X.CAP * X.DS
    w2 = 1.2 * X.CAP**2 / X.KT
    assert abs(w1 - w2) / w1 < 1e-12
    got = float(X.steady_state_cycle_area(X.FN, X.MU, X.KT))
    assert abs(got - w1) / w1 < 1e-6
    # closed-form displacement delta_s = 3 mu Fn / (2 k_t)
    assert abs(X.DS - 3.0 * X.CAP / (2.0 * X.KT)) / X.DS < 1e-12
    # the law starts at the origin and its virgin slope is the no-slip k_t
    assert float(X.partial_slip_magnitude(0.0, X.FN, X.MU, X.KT)) == 0.0
    h = 1e-6 * X.DS
    slope = float(X.partial_slip_magnitude(h, X.FN, X.MU, X.KT)) / h
    assert abs(slope - X.KT) / X.KT < 1e-3


# ---------------------------------------------------------------------------
# leg 2 -- engine: target arm and negative control
# ---------------------------------------------------------------------------
def test_target_arm_matches_exact_mindlin():
    res = X.test_engine_vs_closed_form()["mindlin"]
    assert res["role"] == "target"
    assert res["max_abs_err_over_cap"] < X.TOL["engine_point_over_cap"]
    assert res["loop_area_rel_err"] < X.TOL["engine_area_rel"]
    assert res["pass"]


def test_negative_control_breaches_the_owner_gate():
    """Cundall-Strack unloading must lose ~23 % of the cycle energy.  If this
    ever starts passing, the harness stopped discriminating and every other
    green check in this file becomes meaningless."""
    res = X.test_engine_vs_closed_form()["linear"]
    lo, hi = X.CONTROL["linear_expected_area_rel"]
    assert res["role"] == "negative_control"
    assert res["loop_area_rel_err"] > X.OWNER_GATE == 0.15
    assert lo <= res["loop_area_signed_rel_err"] <= hi
    assert res["max_abs_err_over_cap"] > 0.1
    assert res["pass"]


# ---------------------------------------------------------------------------
# leg 3/4 -- dt refinement and the scaling laws
# ---------------------------------------------------------------------------
def test_dt_refinement_converges():
    """The law is rate independent (pointwise deviation stays at round-off for
    every dt) while the *enclosed area* is a quadrature and must converge as
    dt -> 0, monotonically and at least first order."""
    res = X.timestep_convergence(n_steps=200)
    assert res["pass"]
    assert res["max_area_rel_err"] < X.TOL["dt_area_rel"]
    rows = sorted(res["rows"], key=lambda r: r["dt_factor"])  # fine -> coarse
    areas = [r["area_rel_err"] for r in rows]
    assert areas == sorted(areas), f"area error must grow with dt: {areas}"
    # each dt doubling must multiply the area error by ~4 (first order)
    for coarse, fine in zip(areas[1:], areas[:-1]):
        assert 2.0 <= coarse / fine <= 8.0, areas
    # rate independence: the force law itself is sampled, not integrated
    assert max(r["max_abs_err_over_cap"] for r in res["rows"]) < 1e-12
    assert rows[0]["n_samples"] > rows[-1]["n_samples"]


@pytest.mark.parametrize("leg, exponent", [("mu_sensitivity", 1.0),
                                           ("fn_sensitivity", 2.0 / 3.0),
                                           ("E_sensitivity", -2.0 / 3.0)])
def test_scaling_laws(leg, exponent):
    res = X.parameter_sensitivity()[leg]
    assert res["pass"], (leg, res["max_rel_err"])
    assert res["exponent_theory"] == pytest.approx(exponent)
    # refit the exponent from the reported sweep: slope of log(delta_s) vs log(x)
    xs = res["{}_values".format(leg.split("_")[0])]
    ds = res["delta_s_values"]
    xs_log = [math.log(v) for v in xs]
    ds_log = [math.log(v) for v in ds]
    n = len(xs_log)
    mx = sum(xs_log) / n
    my = sum(ds_log) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs_log, ds_log))
    den = sum((a - mx) ** 2 for a in xs_log)
    assert abs(num / den - exponent) < 1e-9


# ---------------------------------------------------------------------------
# verdict logic -- control-aware, cannot report PASS when blind
# ---------------------------------------------------------------------------
def test_verdict_is_control_aware():
    results = _fresh()
    v = results["verdict"]
    assert v["status"] == "ALL PASS"
    assert v["control_is_discriminating"] is True
    assert all(v["checks"].values())

    # a blind harness (control passing the gate) must NOT be reported as PASS
    blind = copy.deepcopy(results)
    blind["engine_vs_closed_form"]["linear"]["pass"] = False
    v2 = X.verdict(blind)
    assert v2["status"] != "ALL PASS" or "control not discriminating" in v2["status"]
    assert v2["control_is_discriminating"] is False


_FRESH: dict = {}


def _fresh(no_figures=True, tmp_path=None):
    """Run the benchmark legs without writing into the repository.

    Cached for the session: the full leg set (20 000-step ODE oracle included)
    costs ~25 s, and two tests need the identical payload.
    """
    import os
    import tempfile

    key = bool(no_figures)
    if key not in _FRESH:
        if tmp_path is None:
            tmp_path = Path(tempfile.mkdtemp(prefix="xval_slip_",
                                             dir=os.environ.get("TMPDIR")))
        _FRESH[key] = X.run(out_dir=tmp_path, make_figures=not no_figures)
    return _FRESH[key]


# ---------------------------------------------------------------------------
# anti-drift gate: the committed artefact must equal a fresh run
# ---------------------------------------------------------------------------
def test_committed_artefact_is_not_stale():
    """Every scalar quoted in the report is read from this file, so it must be
    reproducible: a stale/hand-edited ``results.json`` fails here."""
    assert COMMITTED.exists(), f"missing committed artefact {COMMITTED}"
    old = json.loads(COMMITTED.read_text(encoding="utf-8"))
    new = _fresh()

    assert old["verdict"]["status"] == new["verdict"]["status"] == "ALL PASS"

    def close(a, b, tol=1e-10):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= tol * max(1.0, abs(b))
        return a == b

    for key in ("gross_slip_displacement", "partial_slip_magnitude",
                "reversal_branch", "steady_state_cycle_area", "stick_radius_ratio"):
        for field, value in new["closed_form_vs_analytical"][key].items():
            if isinstance(value, (int, float, bool)):
                assert close(old["closed_form_vs_analytical"][key][field], value), (key, field)

    for arm in ("mindlin", "linear"):
        for field in ("max_abs_err_over_cap", "loop_area_J", "loop_area_rel_err",
                      "loop_area_signed_rel_err"):
            assert close(old["engine_vs_closed_form"][arm][field],
                         new["engine_vs_closed_form"][arm][field]), (arm, field)
        assert old["engine_vs_closed_form"][arm]["role"] == \
            new["engine_vs_closed_form"][arm]["role"]

    assert old["oracle_ode"]["pass"] is True
    assert close(old["oracle_ode"]["max_rel_err_over_cap_loading"],
                 new["oracle_ode"]["max_rel_err_over_cap_loading"], tol=1e-6)
    assert close(old["oracle_ode"]["max_rel_err_over_cap_reversal"],
                 new["oracle_ode"]["max_rel_err_over_cap_reversal"], tol=1e-6)

    for leg in ("mu_sensitivity", "fn_sensitivity", "E_sensitivity"):
        assert old["parameter_sensitivity"][leg]["pass"] is True
        assert old["parameter_sensitivity"][leg]["delta_s_values"] == \
            pytest.approx(new["parameter_sensitivity"][leg]["delta_s_values"], rel=1e-12)

    assert old["design"]["tolerances"] == new["design"]["tolerances"]
    assert (json.loads(json.dumps(old["design"]["control_expectation"]))
            == json.loads(json.dumps(new["design"]["control_expectation"])))
    assert not any(p.startswith("/") for p in [f["path"] for f in old["figures"]]), \
        "artefact must store repo-relative figure paths"
"""Regression tests for tools/self_verification_scan.py.

The scanner exists because `audit_report.md` §4-F1 found a *pass* that was
decided by a constant typed into a test rather than by the solver (seepage
C1-C3: `u_analytic * (1 + noise)`, `i_c_analytic * 1.10`, `k * 1.15`, reported
as "DEM-measured").  These tests pin the detector's behaviour on synthetic
fixtures -- including the negative controls, so the check cannot pass by being a
silent no-op -- and then require the real repository to be clean.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "self_verification_scan", ROOT / "tools" / "self_verification_scan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve their module here
    spec.loader.exec_module(mod)
    return mod


sv = _load_tool()


def _scan(tmp_path: Path, body: str):
    p = tmp_path / "test_fixture.py"
    p.write_text(body, encoding="utf8")
    allow = sv.load_allowlist(ROOT / "tools" / "self_verification_allowlist.json")
    return sv.scan_file(p, allow).sites


# ------------------------------------------------------------------ positives
def test_detects_typed_scale_as_sev1(tmp_path):
    sites = _scan(tmp_path, """
def test_c2():
    i_c_analytic = proto.terzaghi_i_c
    i_c_measured = i_c_analytic * 1.10
    check(i_c_measured)
""")
    assert [s.severity for s in sites] == ["SEV1"]
    assert sites[0].target == "i_c_measured"
    assert sites[0].kind == "analytic_scale"
    assert not sites[0].declared


def test_detects_noise_surrogate_as_sev2(tmp_path):
    sites = _scan(tmp_path, """
def test_c1(z, u_analytic):
    u_measured = u_analytic * (1.0 + 0.05 * rng.standard_normal(len(z)))
    check(u_measured)
""")
    assert [s.severity for s in sites] == ["SEV2"]
    assert sites[0].kind == "analytic_noise"


def test_detects_alias_chain(tmp_path):
    """The surrogate is not always on the same line as the analytic call."""
    sites = _scan(tmp_path, """
def test_c3(k_analytic):
    k_ref = k_analytic
    k_measured = [k * 1.15 for k in k_ref]
    check(k_measured)
""")
    assert len(sites) == 1 and sites[0].severity == "SEV1"


# --------------------------------------------------------- declarations/negatives
def test_inband_marker_declares_the_site(tmp_path):
    sites = _scan(tmp_path, """
def test_c1(z, u_analytic):
    # self-verification: synthetic surrogate -- not a measurement
    u_measured = u_analytic * (1.0 + 0.05 * rng.standard_normal(len(z)))
    check(u_measured)
""")
    assert len(sites) == 1 and sites[0].declared


def test_synthetic_prefix_declares_the_site(tmp_path):
    sites = _scan(tmp_path, """
def test_c1(u_analytic):
    synthetic_u_measured = u_analytic * 1.05
    check(synthetic_u_measured)
""")
    assert sites and all(s.declared for s in sites)


def test_real_solver_output_is_not_flagged(tmp_path):
    """A value read from the solver is the whole point of a measurement."""
    sites = _scan(tmp_path, """
def test_dem():
    ref = reference_case()
    measured = float(result["theta_c_deg"])
    check(measured)
""")
    assert sites == []


def test_mixed_deficit_is_advisory_not_surrogate(tmp_path):
    """solver_work - analytic_work is a cross-check, not a stand-in."""
    sites = _scan(tmp_path, """
def test_closure(sim):
    measured = sim.work_exact_J - sim.quadrature_work_J
    check(measured)
""")
    assert [s.severity for s in sites] == ["INFO"]
    assert sites[0].mixed


# ------------------------------------------------------------------- the repo
def test_repository_has_no_undeclared_surrogate():
    """Gate: no undeclared SEV1/SEV2 site anywhere in tests/ or scripts/."""
    assert sv.main(["--quiet"]) == 0


def test_scan_report_is_machine_readable(tmp_path):
    out = tmp_path / "scan.json"
    sv.main(["--json", str(out), "--quiet"])
    body = json.loads(out.read_text(encoding="utf8"))
    assert body["summary"]["blocking_sev1_sev2"] == 0
    assert body["files_scanned"] > 500
    # the historical offender is still *visible*, just declared
    seepage = [s for s in body["sites"] if s["path"].endswith("test_seepage_piping.py")]
    assert seepage and all(s["declared"] for s in seepage)
    assert any(s["allowlisted"] for s in body["sites"])

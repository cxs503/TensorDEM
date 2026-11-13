"""``tensordem h1-broken-ice`` must be usable as a CI gate.

The H1-BROKEN-ICE-DEM acceptance chain has two entry points: the benchmark
runner (``benchmarks/h1_broken_ice_dem/run_validation.py``) and the CLI
subcommand.  The runner was given a non-zero exit code on a failing case in
audit m15192; the CLI kept printing ``cases: {...False}`` and exiting ``0``, so
a pipeline that used the CLI could not see a regression.  These tests pin the
exit status, the report file, and the interaction between ``--quick`` and the
explicitly typed case sizes -- all with a stubbed ``validate`` so the assertions
are about the gate, not about DEM arithmetic (the cases themselves are covered
by ``tests/test_h1_broken_ice_dem.py``).
"""
from __future__ import annotations

import json

import pytest

from tensordem import cli
from tensordem.polyhedron_floe3d import h1_cases


def _stub_validate(monkeypatch, passed: dict, record: dict | None = None):
    """Replace ``h1_cases.validate`` and capture how it was called."""

    def fake_validate(*, cases=(), quick=False, **kwargs):
        if record is not None:
            record.update({"cases": list(cases), "quick": quick, "kwargs": kwargs})
        return {"passed": dict(passed), "cases": {}, "quick": bool(quick),
                "runtime_s": 0.0}

    monkeypatch.setattr(h1_cases, "validate", fake_validate)


def _run(argv, tmp_path):
    return cli.main(argv + ["--output", str(tmp_path / "report.json")])


def test_failing_case_exits_nonzero(monkeypatch, tmp_path, capsys):
    """A requested case that reports ``passed = False`` must fail the process."""
    _stub_validate(monkeypatch, {"buoyancy": False})
    with pytest.raises(SystemExit) as excinfo:
        _run(["h1-broken-ice", "--cases", "buoyancy", "--quick"], tmp_path)
    assert excinfo.value.code == 1
    assert "FAILED cases: buoyancy" in capsys.readouterr().err
    # the report is still written, so the failing run can be audited
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] == {"buoyancy": False}


def test_all_cases_passing_exits_zero(monkeypatch, tmp_path):
    _stub_validate(monkeypatch, {"buoyancy": True, "channel": True})
    assert _run(["h1-broken-ice", "--cases", "buoyancy,channel", "--quick"],
                tmp_path) is None


def test_mixed_verdict_still_fails(monkeypatch, tmp_path):
    """One FAILED case out of two is enough -- no majority vote, no silent skip."""
    _stub_validate(monkeypatch, {"buoyancy": True, "repose": False})
    with pytest.raises(SystemExit) as excinfo:
        _run(["h1-broken-ice", "--cases", "buoyancy,repose", "--quick"], tmp_path)
    assert excinfo.value.code == 1


def test_quick_mode_does_not_invent_case_sizes(monkeypatch, tmp_path):
    """``--quick`` must reach ``validate`` without the full-size kwargs.

    Passing ``n_steps``/``n_bodies`` unconditionally would defeat the quick
    presets (``validate`` applies them with ``setdefault``), so the flag would
    quietly run the full case.
    """
    seen: dict = {}
    _stub_validate(monkeypatch, {"buoyancy": True}, seen)
    _run(["h1-broken-ice", "--cases", "buoyancy,repose,channel", "--quick"], tmp_path)
    assert seen["quick"] is True
    assert seen["kwargs"] == {}


def test_quick_mode_honours_explicit_sizes(monkeypatch, tmp_path):
    """A size the caller typed wins over the quick preset (no silently ignored flag)."""
    seen: dict = {}
    _stub_validate(monkeypatch, {"repose": True, "channel": True}, seen)
    _run(["h1-broken-ice", "--cases", "repose,channel", "--quick",
          "--repose-steps", "123", "--channel-steps", "7"], tmp_path)
    assert seen["kwargs"]["repose"]["n_steps"] == 123
    assert seen["kwargs"]["channel"]["n_steps"] == 7


def test_full_mode_uses_the_documented_defaults(monkeypatch, tmp_path):
    seen: dict = {}
    _stub_validate(monkeypatch, {"repose": True, "channel": True}, seen)
    _run(["h1-broken-ice", "--cases", "repose,channel"], tmp_path)
    assert seen["quick"] is False
    assert seen["kwargs"]["repose"] == {"n_steps": 6000, "n_bodies": 45}
    assert seen["kwargs"]["channel"] == {"n_steps": 900}


def test_real_quick_buoyancy_case_passes_through_the_cli(tmp_path):
    """End-to-end smoke on the cheapest real case: the C1 gate is genuinely green.

    This is the only test here that runs DEM arithmetic; it is the check that the
    stubbed gate above is wired to the real ``validate`` signature.
    """
    assert _run(["h1-broken-ice", "--cases", "buoyancy", "--quick"], tmp_path) is None
    report = json.loads((tmp_path / "report.json").read_text())
    case = report["cases"]["buoyancy"]
    assert report["passed"]["buoyancy"] is True
    assert case["error"]["rel_err_submerged_fraction"] < case["tolerance"]
    assert case["error"]["rel_err_draft"] < case["tolerance"]
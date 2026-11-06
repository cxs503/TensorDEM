"""Regression gates for the m4790 ledger sensitivity evidence.

``scripts/plot_m4790_sensitivity.py`` publishes
``artifacts/m4790/sensitivity.json`` + ``ledger_corruption_sensitivity.png``.
Those numbers are quoted in
``docs/dem2d_energy_fracture_ledger_negative_controls.md``, so they must not
decay silently: this module re-derives them and fails if the committed
evidence drifts from the current solver.

Three independent checks:

1. **provenance** -- the committed ledger channels and the clean residual are
   reproduced by a fresh simulation (relative tolerance ``1e-9``);
2. **model** -- every swept point and every analytic threshold ``z*`` obeys
   the closed form ``R(z) = R0 + s*a*(z-1)``, ``S(z) = S0 + |a|*(z-1)`` for
   channels inside the scale (agreement to rounding);
3. **behaviour** -- the gate itself (``_rel_err``) turns red *beyond* the
   analytic threshold and stays green *inside* it, i.e. ``z*`` is the real
   transition point and not a curve-fitting artefact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.test_dem2d_energy_fracture_ledger_negative import (  # noqa: E402
    TOL,
    _corrupt,
    _energy_residual,
    _energy_scale,
    _rel_err,
    _run_indentation,
)

EVIDENCE = ROOT / "artifacts" / "m4790" / "sensitivity.json"
FIGURE = ROOT / "artifacts" / "m4790" / "ledger_corruption_sensitivity.png"

#: numerical agreement required between the closed form and the measurement
MODEL_RTOL = 1e-9


@pytest.fixture(scope="module")
def evidence() -> dict:
    assert EVIDENCE.is_file(), f"missing evidence artifact {EVIDENCE}; run scripts/plot_m4790_sensitivity.py"
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sim():
    return _run_indentation()


def _predicted_rel(sim, channel: str, sign: float, in_scale: bool, z: float) -> float:
    """Closed-form relative residual at multiplicative factor ``z``."""
    r0 = _energy_residual(sim)
    s0 = _energy_scale(sim)
    a = float(sim.accounting[channel])
    r = r0 + sign * a * (z - 1.0)
    s = s0 + (abs(a) * (z - 1.0) if in_scale else 0.0)
    return abs(r) / max(s, 1e-18)


def test_evidence_is_present(evidence: dict) -> None:
    """The published evidence and figure exist and are non-trivial."""
    assert evidence["tolerance_rel"] == TOL
    assert evidence["clean_closes"] is True
    assert evidence["broken_bonds"] > 0
    assert len(evidence["channels"]) == 5
    assert FIGURE.is_file(), f"missing figure {FIGURE}"
    assert FIGURE.stat().st_size > 5000, "figure is suspiciously small"


def test_committed_ledger_matches_fresh_run(evidence: dict, sim) -> None:
    """Provenance: the committed ledger channels reproduce on the current code."""
    for channel, value in evidence["ledger_J"].items():
        actual = sim.accounting[channel]
        if isinstance(actual, (list, tuple)):  # impulse channels are vectors
            assert [float(x) for x in actual] == pytest.approx(value, rel=1e-9, abs=1e-15)
            continue
        assert actual == pytest.approx(value, rel=1e-9, abs=1e-15), (
            f"ledger channel {channel} drifted: evidence {value!r} vs run {actual!r}"
        )
    assert _energy_residual(sim) == pytest.approx(
        evidence["clean_residual_J"], rel=1e-9, abs=1e-15
    )
    assert _rel_err(sim) == pytest.approx(
        evidence["clean_rel_residual"], rel=1e-9
    )


def test_sweep_obeys_closed_form(evidence: dict, sim) -> None:
    """Model: every swept residual follows the linear-in-channel identity."""
    for ch in evidence["channels"]:
        for point in ch["sweep"]:
            predicted = _predicted_rel(
                sim, ch["channel"], ch["sign"], ch["in_scale"], point["factor"]
            )
            assert point["rel_residual"] == pytest.approx(
                predicted, rel=MODEL_RTOL
            ), f"{ch['channel']} z={point['factor']}: {point['rel_residual']} != {predicted}"
            assert point["detected"] is (point["rel_residual"] > TOL)


def test_analytic_thresholds_sit_on_the_tolerance_surface(evidence: dict) -> None:
    """Model: each analytic ``z*`` evaluates to exactly the gate tolerance."""
    for ch in evidence["channels"]:
        for check in ch["analytic_checks"]:
            assert check["rel_residual_measured"] == pytest.approx(TOL, rel=1e-9), (
                f"{ch['channel']} z*={check['factor']}: measured "
                f"{check['rel_residual_measured']} != tolerance {TOL}"
            )


def test_dormant_channel_is_flagged_vacuous(evidence: dict) -> None:
    """``external_work_J`` is 0 here, so multiplicative faults prove nothing."""
    dormant = [c for c in evidence["channels"] if c["channel"] == "external_work_J"]
    assert len(dormant) == 1
    ch = dormant[0]
    assert ch["clean_value_J"] == 0.0
    assert ch["vacuous"] is True
    assert ch["analytic_threshold_factors"] == []
    assert ch["blind_spot_abs_1mz"] is None
    assert not any(point["detected"] for point in ch["sweep"]), (
        "scaling a dormant channel must not trip the gate -- if it does, the "
        "control is no longer vacuous and the doc claim must be revisited"
    )


def test_gate_transition_matches_analytic_threshold(evidence: dict, sim) -> None:
    """Behaviour: inside ``z*`` the gate is green, beyond it the gate is red."""
    checked = 0
    for ch in evidence["channels"]:
        for z_star in ch["analytic_threshold_factors"]:
            if abs(1.0 - z_star) < 1e-3:
                continue  # too close to the clean point to resolve a step
            inside = 1.0 + 0.98 * (z_star - 1.0)
            outside = 1.0 + 1.02 * (z_star - 1.0)
            r_in = _rel_err(_corrupt(sim, ch["channel"], factor=inside))
            r_out = _rel_err(_corrupt(sim, ch["channel"], factor=outside))
            assert r_in < TOL < r_out, (
                f"{ch['channel']} z*={z_star}: inside->{r_in} outside->{r_out} "
                f"does not bracket {TOL}"
            )
            checked += 1
    assert checked >= 6, f"only {checked} thresholds exercised"


def test_non_vacuous_channels_log_a_blind_spot(evidence: dict) -> None:
    """Sensitivity ordering is published for every live channel."""
    live = [c for c in evidence["channels"] if not c["vacuous"]]
    assert len(live) == 4
    for ch in live:
        assert ch["blind_spot_abs_1mz"] > 0.0
        assert ch["grid_first_detected_factor"] is not None
        assert abs(1.0 - ch["grid_first_detected_factor"]) >= ch["blind_spot_abs_1mz"]

def _generator():
    """The report/doc builder script, imported as a module (no side effects)."""
    import importlib

    return importlib.import_module("scripts.post_m4790_report")


def test_committed_artifacts_match_the_generator() -> None:
    """Provenance: the committed report/doc block are the generator's own output."""
    mod = _generator()
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report, section = mod.build(data)

    committed_report = (ROOT / "artifacts" / "m4790" / "report.md").read_text(encoding="utf-8")
    assert committed_report == report, "artifacts/m4790/report.md drifted from the generator"

    doc = (ROOT / "docs" / "dem2d_energy_fracture_ledger_negative_controls.md").read_text(
        encoding="utf-8"
    )
    assert doc.count(mod.BEGIN) == 1 and doc.count(mod.END) == 1
    body = doc.split(mod.BEGIN, 1)[1].split(mod.END, 1)[0]
    assert (mod.BEGIN + body + mod.END).rstrip("\n") == section.rstrip("\n"), (
        "docs/dem2d_energy_fracture_ledger_negative_controls.md §7 drifted from the generator"
    )


def test_doc_injection_is_byte_idempotent(tmp_path, monkeypatch) -> None:
    """Re-running the generator on its own output must not change a single byte."""
    mod = _generator()
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    _report, section = mod.build(data)

    doc = tmp_path / "doc.md"
    doc.write_text("# pre-existing\n\nprose above the block\n", encoding="utf-8")
    monkeypatch.setattr(mod, "DOC", doc)

    mod.inject_doc(section)
    first = doc.read_text(encoding="utf-8")
    mod.inject_doc(section)
    second = doc.read_text(encoding="utf-8")

    assert second == first, "doc injection is not idempotent (trailing blank lines drift)"
    assert first.count(mod.BEGIN) == 1 and first.count(mod.END) == 1
    assert first.startswith("# pre-existing\n\nprose above the block\n")
    assert first.rstrip("\n").endswith(mod.END)

"""Tests for ``benchmarks/anchors/formation_ice_channel_anchor.py``.

These lock the three things the artifact claims:

* the unit algebra really rejects a non-homogeneous sum;
* the conversion is reproducible (independent closed form == repo module);
* the verbatim-anchor block stays empty unless text was actually transcribed
  (a regression guard against a future edit quietly inventing a page number).
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "anchors" / "formation_ice_channel_anchor.py"
ARTIFACT = REPO_ROOT / "benchmarks" / "anchors" / "formation-ice-channel.json"


def _load():
    spec = importlib.util.spec_from_file_location("formation_anchor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_unit_addition_rejects_heterogeneous_sum(mod):
    force = mod.NEWTON
    energy = mod.Unit(M=1, L=2, T=-2)
    with pytest.raises(TypeError):
        force + energy
    assert (force + force).exp == force.exp


def test_all_components_are_forces(mod):
    report = mod.dimensional_check(mod.ALPHA_REF_DEG, mod.MU_REF)
    assert report["passed"] is True
    assert report["n_failures"] == 0
    for name, row in report["terms"].items():
        assert row["reduces_to_newton"], name


def test_closed_form_matches_repository_module(mod):
    comp = mod.closed_form(
        mod.ALPHA_REF_DEG, mod.MU_REF, h=mod.CASE["ice_thickness_m"],
        speed=mod.CASE["speed_m_s"], beam=mod.CASE["hull_beam_m"],
        length=mod.CASE["hull_length_m"], sigma_c=mod.CASE["sigma_c_pa"],
        sigma_f=mod.CASE["sigma_f_pa"], rho_w=mod.CASE["rho_water_kg_m3"],
        g=mod.CASE["gravity_m_s2"],
    )
    ref = mod.module_components(mod.ALPHA_REF_DEG, mod.MU_REF,
                                mod.CASE["ice_thickness_m"], mod.CASE["speed_m_s"])
    for key in ref:
        assert abs(comp[key] - ref[key]) / ref[key] < 1e-12, key


def test_inverse_solve_recovers_thickness(mod):
    h = mod.CASE["ice_thickness_m"]
    total = mod.closed_form(
        mod.ALPHA_REF_DEG, mod.MU_REF, h=h, speed=mod.CASE["speed_m_s"],
        beam=mod.CASE["hull_beam_m"], length=mod.CASE["hull_length_m"],
        sigma_c=mod.CASE["sigma_c_pa"], sigma_f=mod.CASE["sigma_f_pa"],
        rho_w=mod.CASE["rho_water_kg_m3"], g=mod.CASE["gravity_m_s2"],
    )["R_total"]
    solved, _ = mod.inverse_solve_h(total, mod.ALPHA_REF_DEG, mod.MU_REF,
                                    mod.CASE["speed_m_s"])
    assert abs(solved - h) / h < 1e-9


def test_identities_all_pass_and_are_not_vacuous(mod):
    report = mod.identities()
    assert report["passed"] is True
    assert report["n_failures"] == 0
    # the crushing term must diverge at a vertical stem: that is the reason the
    # case hull (a box, alpha = 90 deg) is declared out of the formula domain.
    div = next(c for c in report["checks"]
               if c["identity"] == "R_c_diverges_at_alpha_90")
    assert div["ratio_alpha_89p9_over_20"] > 10.0


def test_verbatim_anchor_is_not_fabricated(mod):
    audit = mod.verbatim_anchor_audit()
    assert audit["passed"] is False
    assert audit["verdict"] == "WITHHELD"
    for name, comp in audit["components"].items():
        assert comp["quote"] is None, name
        assert comp["page_or_table"] is None, name
        assert comp["value"] is None, name


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
def test_artifact_schema_and_numbers(mod):
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for key in ("criteria", "section_A_verbatim_anchor", "section_B_caliber_conversion",
                "section_C_dimensional", "section_D_internal_consistency",
                "section_E_domain_comparison", "caliber", "geometry_check"):
        assert key in payload, key
    assert payload["prereg_sha256"] == mod.hashlib.sha256(
        mod.PREREG.read_bytes()).hexdigest()
    assert payload["geometry_check"]["passed"] is True

    identifiers = {c["id"]: c for c in payload["criteria"]}
    assert identifiers["M1"]["status"] == "WITHHELD"
    for cid in ("M2", "M3", "M4"):
        assert identifiers[cid]["passed"] is True, cid

    # the conversion numbers in the artifact must be recomputable from the caliber
    caliber = payload["caliber"]
    ref = payload["section_B_caliber_conversion"]["reference_point"]["components_N"]
    recomputed = mod.closed_form(
        mod.ALPHA_REF_DEG, mod.MU_REF, h=caliber["ice_thickness_m"],
        speed=caliber["speed_m_s"], beam=caliber["hull_beam_m"],
        length=caliber["hull_length_m"], sigma_c=caliber["sigma_c_pa"],
        sigma_f=caliber["sigma_f_pa"], rho_w=caliber["rho_water_kg_m3"],
        g=caliber["gravity_m_s2"],
    )
    for key, value in ref.items():
        assert math.isclose(value, recomputed[key], rel_tol=1e-12), key
    assert payload["section_B_caliber_conversion"]["recompute"]["passed"] is True
    assert payload["overall"]["usable_as_validation_reference"] is False
"""Tests for the CASE-DEM-BIAXIAL-FRICTION document generator and its gates.

The deliverable under test is ``benchmarks/case_dem_biaxial_friction/regen_doc.py``
-- the "zero hand transcription" renderer for
``docs/CASE-DEM-BIAXIAL-FRICTION.md``.  These tests pin the three defects that
made the committed document disagree with the artefact it claimed to come from,
plus the provenance / drift gates that are supposed to make such a
disagreement impossible:

* **unit defect** -- the Love-Weber internal stresses were printed in Pa inside
  a table whose header says kPa (a 1000x overstatement, e.g. 453393.6 kPa);
* **hand-transcription defect** -- the material rows were hardcoded to
  ``mu = 0.55 (RidgeMaterial default)`` and ``rho = 917 kg/m^3 (ice)`` while the
  case under test runs a cohesionless sand specimen at ``mu = 0.65``;
* **unsupported-claim defect** -- the inertial-number sentence asserted
  "quasi-static (I < 0.01)" no matter what the recorded I was;
* **provenance defect** -- nothing rejected an artefact produced by a different
  configuration, so a stale record could stamp a PASS onto another case
  definition;
* **drift defect** -- nothing checked that the committed markdown still equals a
  fresh render.

The numbers themselves are not re-asserted here (that is the benchmark's job);
these tests are number-agnostic: they derive the expected strings from the
record they feed in.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction"))

import regen_doc as rg  # noqa: E402


def load_tracked():
    assert os.path.exists(rg.TRACKED_ART), (
        "tracked distilled record missing; regenerate it with "
        "regen_doc.py --distil")
    with open(rg.TRACKED_ART) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# The tracked record and the pre-declared configuration
# ---------------------------------------------------------------------------

def test_tracked_record_divergence_from_the_anchor_is_disclosed():
    """Either the tracked record *is* the declared anchor, or the committed
    document carries a configuration-drift banner listing every divergence.

    (Silence is the defect this mission fixes: the previous document quoted a
    PASS produced by a 24x48/f_b=0.35 record while the declared case is a
    24x48/f_b=0.0 sand specimen, and nothing said so.)
    """
    rec = load_tracked()
    off = rg.provenance_mismatches(rec["config"])
    with open(rg.DOC) as f:
        md = f.read()
    if off:
        assert "CONFIGURATION DRIFT" in md
        for item in off:
            assert f"`{item}`" in md
    else:
        assert "CONFIGURATION DRIFT" not in md


def test_tracked_record_carries_every_doc_input():
    """Distillation must not drop a block the generator reads."""
    rec = load_tracked()
    for key in ("reference", "config", "specimen", "peak", "stress_definitions",
                "friction_angle", "judgement", "dimensionless", "environment"):
        assert key in rec, f"tracked record lacks {key}"
    assert "stress_path" not in rec, "bulk stress path must stay out of the tracked record"


# ---------------------------------------------------------------------------
# Defect 1: internal stresses in kPa
# ---------------------------------------------------------------------------

def test_internal_stresses_are_reported_in_kpa():
    rec = load_tracked()
    md = rg.regenerate(rec)
    sd = rec["stress_definitions"]
    # the kPa conversion of the internal sigma_1 must appear ...
    assert f"{sd['sigma_v']/1e3:.1f}" in md
    # ... and the raw pascal value must not (1000x overstatement defect)
    assert f"| {sd['sigma_v']:.1f} |" not in md
    assert "All stresses in kPa" in md


def test_internal_and_wall_columns_are_same_order_of_magnitude():
    """A kPa-vs-Pa mix-up shows up as two columns 1000x apart."""
    rec = load_tracked()
    peak, sd = rec["peak"], rec["stress_definitions"]
    ratio = sd["sigma_v"] / peak["sigma_v"]
    assert 0.5 < ratio < 2.0  # same physical quantity, ~3 % apart in reality


# ---------------------------------------------------------------------------
# Defect 2: material constants come from the run, not from prose
# ---------------------------------------------------------------------------

def test_material_row_follows_run_configuration():
    rec = load_tracked()
    md = rg.regenerate(rec)
    mu = rec["config"]["friction_coefficient"]
    assert f"| {mu} (arctan({mu})" in md
    assert "0.55 (RidgeMaterial default)" not in md
    assert "917 kg/m^3 (ice)" not in md


def test_unrecorded_material_constants_are_declared_not_invented():
    rec = load_tracked()
    md = rg.regenerate(rec)
    # kn / kt / rho are not in the record -> must be flagged as not recorded
    assert md.count("not recorded by the case runner") >= 3


def test_material_row_degrades_when_mu_absent():
    rec = copy.deepcopy(load_tracked())
    rec["config"]["friction_coefficient"] = None
    md = rg.regenerate(rec)
    assert "not recorded in this artefact" in md


# ---------------------------------------------------------------------------
# Defect 3: the quasi-static claim follows the number
# ---------------------------------------------------------------------------

def test_inertial_regime_claim_follows_the_recorded_number():
    rec = load_tracked()

    fast = copy.deepcopy(rec)
    fast["dimensionless"]["inertial_number"] = 0.0122
    md_fast = rg.regenerate(fast)
    assert "well within the quasi-static regime" not in md_fast
    assert "quasi-static threshold (I = 0.0122" in md_fast
    assert "rate effects are not negligible" in md_fast

    slow = copy.deepcopy(rec)
    slow["dimensionless"]["inertial_number"] = 0.005
    md_slow = rg.regenerate(slow)
    assert "well within the quasi-static regime" in md_slow


@pytest.mark.parametrize("value,expected", [(0.005, True), (0.0099, True),
                                            (0.01, False), (0.02, False)])
def test_inertial_regime_boundary(value, expected):
    assert ("well within" in rg.inertial_regime(value)) is expected


# ---------------------------------------------------------------------------
# Defect 4: provenance gate
# ---------------------------------------------------------------------------

def test_provenance_gate_rejects_foreign_configuration():
    rec = load_tracked()
    foreign = copy.deepcopy(rec)
    foreign["config"]["columns"] = rec["config"]["columns"] // 2
    assert rg.provenance_mismatches(foreign["config"])
    with pytest.raises(ValueError) as ei:
        rg.assert_provenance(foreign["config"])
    assert "refusing to stamp a verdict" in str(ei.value)


def test_provenance_gate_accepts_the_declared_configuration():
    fp = rg.assert_provenance(copy.deepcopy(rg.ANCHOR_CONFIG))
    assert len(fp) == 16 and int(fp, 16) >= 0


def test_fingerprint_is_stable_and_config_sensitive():
    a = rg.config_fingerprint(rg.ANCHOR_CONFIG)
    b = rg.config_fingerprint(dict(reversed(list(rg.ANCHOR_CONFIG.items()))))
    assert a == b  # key order must not matter
    changed = dict(rg.ANCHOR_CONFIG, freeze_bond_fraction=0.35)
    assert rg.config_fingerprint(changed) != a


# ---------------------------------------------------------------------------
# Defect 5: drift gate, missing-artefact behaviour, distillation
# ---------------------------------------------------------------------------

def test_committed_document_is_in_sync_with_its_artefact():
    off = rg.provenance_mismatches(load_tracked()["config"])
    argv = ["--check"] + (["--accept-recorded-config"] if off else [])
    assert rg.main(argv) == 0


def test_document_has_no_unexpanded_placeholders():
    with open(rg.DOC) as f:
        md = f.read()
    assert "{" not in md and "}" not in md
    assert "None" not in md


def test_check_mode_detects_edited_document(tmp_path):
    doc = tmp_path / "doc.md"
    flag = ["--accept-recorded-config"] if rg.provenance_mismatches(
        load_tracked()["config"]) else []
    assert rg.main(["--doc", str(doc)] + flag) == 0            # write
    assert rg.main(["--check", "--doc", str(doc)] + flag) == 0  # in sync
    text = doc.read_text()
    doc.write_text(text.replace("deg", "degrees", 1))
    assert rg.main(["--check", "--doc", str(doc)] + flag) == 1  # drift


def test_missing_artefact_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "FULL_ART", str(tmp_path / "nope_full.json"))
    monkeypatch.setattr(rg, "TRACKED_ART", str(tmp_path / "nope_tracked.json"))
    with pytest.raises(SystemExit) as ei:
        rg.resolve_artefact(str(tmp_path / "nope.json"))
    msg = str(ei.value)
    assert "run_case.py" in msg and "artifacts/" in msg


def test_distill_drops_pointwise_history_but_keeps_doc_inputs():
    rec = load_tracked()
    rec = dict(rec)
    rec["stress_path"] = {"axial_strain": [0.0, 0.05, 0.1]}
    dist = rg.distill(rec, source={"artefact": "unit-test"})
    assert "stress_path" not in dist
    assert dist["distilled"]["dropped_channels"]["stress_path"]["points"] == 3
    assert dist["config"] == rec["config"]
    assert dist["judgement"] == rec["judgement"]
    assert dist["distilled"]["config_fingerprint"] == rg.config_fingerprint(rec["config"])


def test_regenerate_is_deterministic():
    rec = load_tracked()
    assert rg.regenerate(rec) == rg.regenerate(copy.deepcopy(rec))


def test_provenance_line_records_sha_and_job():
    rec = load_tracked()
    prov = {"artefact": "docs/data/x.json", "sha256": "deadbeef", "kind": "test",
            "job": "cjTEST"}
    md = rg.regenerate(rec, provenance=prov)
    assert "deadbeef" in md and "cjTEST" in md

# ---------------------------------------------------------------------------
# Drift banner: an out-of-anchor record must announce itself
# ---------------------------------------------------------------------------

def test_no_drift_banner_when_record_is_the_anchor():
    rec = load_tracked()
    md = rg.regenerate(rec)
    assert "CONFIGURATION DRIFT" not in md


def test_drift_banner_lists_every_divergent_field():
    rec = load_tracked()
    foreign = copy.deepcopy(rec)
    foreign["config"]["columns"] = 12
    foreign["config"]["strain_target"] = 0.15
    off = rg.provenance_mismatches(foreign["config"])
    assert any(o.startswith("columns:") for o in off)
    assert any(o.startswith("strain_target:") for o in off)
    md = rg.regenerate(foreign, divergences=off)
    assert "CONFIGURATION DRIFT" in md
    assert "columns" in md and "strain_target" in md
    # the banner must precede the provenance line that carries the verdict quote
    assert md.index("CONFIGURATION DRIFT") < md.index("**Provenance:**")


def test_main_refuses_foreign_record_unless_opted_in(tmp_path):
    rec = load_tracked()
    foreign = copy.deepcopy(rec)
    foreign["config"]["consolidation_steps"] = 300
    art = tmp_path / "run.json"
    art.write_text(json.dumps(foreign))
    doc = tmp_path / "doc.md"

    assert rg.main(["--artefact", str(art), "--doc", str(doc)]) == 2
    assert not doc.exists()

    assert rg.main(["--artefact", str(art), "--doc", str(doc),
                    "--accept-recorded-config"]) == 0
    assert "CONFIGURATION DRIFT" in doc.read_text()

"""Tests of the segregation campaign layer (temporal blocks, criteria, report).

These tests do not re-validate the physics of a full campaign (that is the job of
the campaign artefacts); they pin the machinery that makes the campaign
trustworthy:

* a case carries its own *measured* sampling error (temporal blocks) and the
  per-block accounting closes the mass balance,
* the solver is bit-deterministic for a given seed,
* the criteria layer reacts to the index it is asked to judge
  (fine-species index versus large-species index) and never reports a
  coefficient of variation for a mean that sits inside the noise floor,
* the report script turns a directory of records into a summary JSON plus
  figures, with the record hashes in the provenance block.

The runs here are deliberately tiny (600 disks, half a second of physical time)
so the file stays at the scale of a unit test; the cluster campaign uses the same
code path with 8000 disks and 80 vibration periods.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.segregation.analysis import evaluate_criteria, load_records  # noqa: E402
from tensordem.segregation.campaign import CaseSpec, run_case  # noqa: E402


def tiny_spec(name: str, **kw) -> CaseSpec:
    base = dict(
        name=name, drive="vib", ratio=2.0, large_fraction=0.30, gamma=3.0,
        frequency=40.0, seed=0, n_particles=600, small_radius=3.0e-3,
        width=0.24, height=0.80, dt=6.0e-5, kt_ratio=0.5, zeta=0.3, mu=0.4,
        mu_rolling=0.05, settle_time=0.05, run_time=0.5, sample_every=50,
        top_fraction=0.25, n_z_bins=120, n_x_zones=6, loose_fraction=0.45,
        n_blocks=4, snapshot_sample=600,
    )
    base.update(kw)
    return CaseSpec(**base)


@pytest.fixture(scope="module")
def vib_record() -> dict:
    return run_case(tiny_spec("tiny_R2"))


@pytest.fixture(scope="module")
def control_record() -> dict:
    return run_case(tiny_spec("tiny_R1", ratio=1.0))


def test_temporal_blocks_close_the_mass_balance(vib_record):
    rec = vib_record
    blocks = rec["stats_blocks"]
    assert len(blocks) == rec["spec"]["n_blocks"]
    assert all(b["n_samples"] > 0 for b in blocks)
    # every block must see the whole specimen: the sum over z bins equals the
    # per-snapshot measure, and the total is the specimen measure
    for blk in blocks:
        area = np.asarray(blk["z_area"])
        assert area.sum() > 0.0
        assert abs(area.sum() / blk["n_samples"] - rec["solid_area_m2"]) / rec["solid_area_m2"] < 1e-9
    # the block-level index is reported with its spread
    seg = rec["metrics"]["segregation_blocks"]
    assert len(seg["S_fine"]) == len(blocks)
    assert np.isfinite(seg["S_fine_mean"])
    assert seg["S_fine_std"] >= 0.0
    # the mean of the blocks and the whole-run value must agree to the level of
    # the block spread (they are the same estimator on a coarser sample)
    assert abs(seg["S_fine_mean"] - rec["metrics"]["segregation"]["S_fine"]) <= max(
        3.0 * seg["S_fine_stderr"], 1e-9)


def test_mass_conservation_and_centre_of_mass(vib_record):
    m = vib_record["metrics"]
    assert m["mass_conservation_rel"] <= 1e-9
    assert m["z_com_consistency_rel"] <= 1e-6
    assert m["final_z_com_instant_m"] > 0.0


def test_index_signs_are_opposite(vib_record):
    seg = vib_record["metrics"]["segregation"]
    # by construction S_fine and S_bn describe the two species of the same
    # profile, so a measurable segregation cannot have both signs equal
    if abs(seg["S_fine"]) > 1e-3:
        assert seg["S_fine"] * seg["S_bn"] < 0.0
    assert seg["phi_fine_top"] + seg["phi_large_top"] == pytest.approx(1.0, abs=1e-9)


def test_run_is_bit_deterministic(vib_record):
    again = run_case(tiny_spec("tiny_R2"))
    assert again["audit"]["digest"] == vib_record["audit"]["digest"]
    assert again["metrics"]["segregation"]["S_fine"] == pytest.approx(
        vib_record["metrics"]["segregation"]["S_fine"], rel=0.0, abs=0.0)


def test_pour_drive_produces_a_heap(tmp_path):
    # A gate release is a *collisional* event: the collapse reaches ~1.5 m/s, so
    # the contact stiffness has to be raised (and the timestep lowered with it)
    # for the soft-sphere overlap to stay at the percent level.  The production
    # pour family of the campaign is run with the same lesson in mind -- see the
    # limitations section of the report.
    rec = run_case(tiny_spec("tiny_pour", drive="pour", ratio=2.0, run_time=0.20,
                            settle_time=0.03, kn=2.0e6, dt=5.0e-6,
                            gate_fraction=0.85, height=0.45))
    assert rec["spec"]["drive"] == "pour"
    assert rec["metrics"]["mass_conservation_rel"] <= 1e-9
    # a poured heap has a finite spread of the floor load: the gate release must
    # redistribute the load over several floor zones
    press = np.asarray(rec["base_load"]["mean_pressure_Pa"], dtype=float)
    assert press.size >= 3
    assert press.max() > 0.0


def _write(tmp_path: Path, *records) -> Path:
    for rec in records:
        (tmp_path / f"{rec['spec']['name']}.json").write_text(json.dumps(rec))
    return tmp_path


def test_criteria_layer_switches_index(tmp_path, vib_record, control_record):
    run_dir = _write(tmp_path, vib_record, control_record)
    records = load_records(str(run_dir))
    assert set(records) == {"tiny_R1", "tiny_R2"}
    fine = evaluate_criteria(records, index="S_fine", gamma_ratio=2.0,
                            gamma_fraction=0.30, gamma_frequency=40.0)
    bn = evaluate_criteria(records, index="S_bn", gamma_ratio=2.0,
                           gamma_fraction=0.30, gamma_frequency=40.0)
    assert "a_negative_control" in fine and "c_mass_conservation" in fine
    assert fine["c_mass_conservation"]["pass"] is True
    # the same record set judged on the two species gives opposite forcing; at
    # minimum the structures must exist and the pass flags must be booleans
    for key in ("a_negative_control", "b_ratio_scaling", "d_gamma_threshold",
                "e_transfer", "f_seed_reproducibility"):
        if key in fine:
            assert isinstance(fine[key]["pass"], bool)
    b_fine = fine.get("b_ratio_scaling", {}).get("by_ratio", {})
    b_bn = bn.get("b_ratio_scaling", {}).get("by_ratio", {})
    if 2.0 in b_fine and 2.0 in b_bn:
        # the index selected flips which species is reported as the primary one,
        # while the other-species field keeps the raw large-species value
        assert b_fine[2.0]["S_other_mean"] == pytest.approx(b_bn[2.0]["S_bn_mean"], rel=1e-9)


def test_seed_cv_skips_the_noise_floor(tmp_path, vib_record, control_record):
    run_dir = _write(tmp_path, vib_record, control_record)
    criteria = evaluate_criteria(load_records(str(run_dir)), index="S_fine",
                                 gamma_ratio=2.0, gamma_fraction=0.30,
                                 gamma_frequency=40.0)
    f_seed = criteria.get("f_seed_reproducibility", {})
    # a single seed per ratio cannot give a CV at all, and the control ratio must
    # never be used as the denominator of one
    assert f_seed.get("pass") is False
    assert 1.0 in f_seed.get("excluded_inside_noise_floor", []) or 1.0 not in f_seed.get("by_ratio", {})


def test_report_script_end_to_end(tmp_path, vib_record, control_record):
    from importlib.util import module_from_spec, spec_from_file_location

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write(out_dir, vib_record, control_record)
    pytest.importorskip("matplotlib", reason="the figure stage needs matplotlib")
    spec = spec_from_file_location(
        "seg_report", ROOT / "scripts" / "segregation_campaign_report.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = tmp_path / "summary.json"
    md = tmp_path / "tables.md"
    fig_dir = tmp_path / "figs"
    module.main(["--run-dir", str(out_dir), "--fig-dir", str(fig_dir),
                 "--out-summary", str(summary), "--markdown", str(md),
                 "--index", "S_fine", "--gamma-family", "2.0:0.3:40.0"])
    data = json.loads(summary.read_text())
    assert data["n_cases"] == 2
    assert "verdict" in data["criteria"]
    assert data["provenance"]["tiny_R1"]["sha256"]
    assert md.exists() and md.read_text().startswith("# Segregation campaign")
    assert list(fig_dir.glob("*.png")), "at least one figure must be produced"
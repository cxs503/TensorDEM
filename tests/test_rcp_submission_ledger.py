"""The submission ledger must be gated against artefacts, not against the queue.

Motivation (m22977): the compute gate once reported six RCP deposition jobs as
``succeeded`` (rc=0) while all six had died at the staging step and produced no
file.  A ledger that says "submitted" therefore proves nothing; only the
``.json`` on disk does.  These tests pin that behaviour down on synthetic run
directories: no simulation, no cluster, no fixtures from the real campaign.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "case_rcp_deposition"))

import submission_ledger as sl  # noqa: E402


def _run(path, n=10000, mu=0.1, kn=64000.0, seed=2, e=0.1, dt_frac=0.05,
         settled=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"spec": {"n_particles": n, "mu": mu, "kn": kn, "seed": seed,
                            "restitution": e, "dt_frac": dt_frac},
                   "metrics": {"phi_bulk": 0.62},
                   "impl_sha256": "deadbeef", "settled": settled}, fh)


def test_parse_tag_round_trips_the_naming_convention():
    spec = sl.parse_tag("deposition_n10000_mu0.1_kn64000_e0.1_seed2_dtf0.025")
    assert spec == {"n_particles": 10000, "mu": 0.1, "kn": 64000.0,
                    "restitution": 0.1, "seed": 2, "dt_frac": 0.025}
    assert sl.parse_tag("deposition_n10000_mu0_kn4000_e0.1_seed1")["dt_frac"] == 0.05
    assert sl.parse_tag("deposition_n10000_mu0_kn4000_e0.1_seed1")["mu"] == 0.0


def test_unparsable_tag_is_never_counted_as_present(tmp_path):
    res = sl.audit_ledger([{"jid": "cj1", "tag": "queued-run"}], [str(tmp_path)])
    assert res["rows"][0]["status"] == sl.UNPARSED
    assert res["complete"] is False


def test_missing_artefact_is_missing(tmp_path):
    res = sl.audit_ledger(
        [{"jid": "cj4411", "tag": "deposition_n10000_mu0.1_kn64000_e0.1_seed3"}],
        [str(tmp_path)])
    assert res["rows"][0]["status"] == sl.MISSING
    assert res["complete"] is False
    assert res["missing"] == ["deposition_n10000_mu0.1_kn64000_e0.1_seed3"]


def test_matching_artefact_is_present_and_completes(tmp_path):
    tag = "deposition_n10000_mu0.1_kn64000_e0.1_seed2_dtf0.025"
    _run(tmp_path / (tag + ".json"), seed=2, dt_frac=0.025)
    res = sl.audit_ledger([{"jid": "cj4497", "tag": tag}], [str(tmp_path)])
    assert res["rows"][0]["status"] == sl.PRESENT, res["rows"][0]["reasons"]
    assert res["complete"] is True
    assert res["n_digests"] == 1


def test_spec_contradicting_its_own_tag_is_mismatch(tmp_path):
    """A file named for mu=0.1 whose spec says mu=0.0 is not that run."""
    tag = "deposition_n10000_mu0.1_kn64000_e0.1_seed2"
    _run(tmp_path / (tag + ".json"), mu=0.0, seed=2)
    row = sl.check_row({"jid": "cj1", "tag": tag}, [str(tmp_path)])
    assert row["status"] == sl.MISMATCH
    assert any("mu" in r for r in row["reasons"])


def test_timestep_contradiction_is_mismatch_not_noise(tmp_path):
    tag = "deposition_n10000_mu0_kn64000_e0.1_seed1_dtf0.025"
    _run(tmp_path / (tag + ".json"), mu=0.0, seed=1, dt_frac=0.05)
    assert sl.check_row({"jid": "cj1", "tag": tag},
                        [str(tmp_path)])["status"] == sl.MISMATCH


def test_unsettled_run_is_not_evidence(tmp_path):
    tag = "deposition_n10000_mu0_kn4000_e0.1_seed1"
    _run(tmp_path / (tag + ".json"), mu=0.0, kn=4000.0, seed=1, settled=False)
    assert sl.check_row({"jid": "cj1", "tag": tag},
                        [str(tmp_path)])["status"] == sl.UNSETTLED
    assert sl.check_row({"jid": "cj1", "tag": tag}, [str(tmp_path)],
                        require_settled=False)["status"] == sl.PRESENT


def test_truncated_artefact_is_missing_not_a_crash(tmp_path):
    tag = "deposition_n10000_mu0_kn4000_e0.1_seed1"
    with open(tmp_path / (tag + ".json"), "w") as fh:
        fh.write('{"spec": {"n_particles": 10000')
    row = sl.check_row({"jid": "cj1", "tag": tag}, [str(tmp_path)])
    assert row["status"] == sl.MISSING
    assert "unreadable" in row["reasons"][0]


def test_search_order_is_respected(tmp_path):
    """Production runs win over a repro copy of the same tag."""
    tag = "deposition_n10000_mu0_kn4000_e0.1_seed1"
    prod, repro = tmp_path / "runs", tmp_path / "repro"
    _run(prod / (tag + ".json"), mu=0.0, kn=4000.0, seed=1)
    _run(repro / (tag + ".json"), mu=0.0, kn=4000.0, seed=1, settled=False)
    row = sl.check_row({"jid": "cj1", "tag": tag}, [str(prod), str(repro)])
    assert row["status"] == sl.PRESENT
    assert row["path"].startswith("benchmarks") is False  # relpath from ROOT


def test_mixed_digests_are_surfaced(tmp_path):
    tag_a = "deposition_n10000_mu0_kn4000_e0.1_seed1"
    tag_b = "deposition_n10000_mu0_kn64000_e0.1_seed1"
    _run(tmp_path / (tag_a + ".json"), mu=0.0, kn=4000.0, seed=1)
    _run(tmp_path / (tag_b + ".json"), mu=0.0, kn=64000.0, seed=1)
    with open(tmp_path / (tag_b + ".json")) as fh:
        d = json.load(fh)
    d["impl_sha256"] = "cafebabe"
    with open(tmp_path / (tag_b + ".json"), "w") as fh:
        json.dump(d, fh)
    res = sl.audit_ledger([{"jid": "a", "tag": tag_a},
                           {"jid": "b", "tag": tag_b}], [str(tmp_path)])
    assert res["complete"] is True
    assert res["n_digests"] == 2


def test_repro_row_pinned_to_its_own_directory(tmp_path):
    """A reproduction row must not be satisfied by the archived run."""
    tag = "deposition_n10000_mu0_kn4000_e0.1_seed1"
    prod, repro = tmp_path / "runs", tmp_path / "runs_repro_m22824"
    _run(prod / (tag + ".json"), mu=0.0, kn=4000.0, seed=1)
    dirs = [str(prod), str(repro)]
    loose = sl.check_row({"jid": "cj4499", "tag": tag}, dirs)
    assert loose["status"] == sl.PRESENT          # found the archive...
    pinned = sl.check_row({"jid": "cj4499", "tag": tag, "dir": "runs_repro_m22824"}, dirs)
    assert pinned["status"] == sl.MISSING          # ...but not in the pinned dir
    _run(repro / (tag + ".json"), mu=0.0, kn=4000.0, seed=1)
    ok = sl.check_row({"jid": "cj4499", "tag": tag, "dir": "runs_repro_m22824"}, dirs)
    assert ok["status"] == sl.PRESENT
    assert "runs_repro_m22824" in ok["path"]


def test_empty_ledger_is_not_complete():
    assert sl.audit_ledger([], ["."])["complete"] is False


def test_cli_exit_code_follows_the_gate(tmp_path, capsys):
    tag = "deposition_n10000_mu0_kn4000_e0.1_seed1"
    _run(tmp_path / (tag + ".json"), mu=0.0, kn=4000.0, seed=1)
    ledger = tmp_path / "plan.json"
    ledger.write_text(json.dumps({"jobs": [{"jid": "cj1", "tag": tag}]}))
    assert sl.main([str(ledger), "--runs", str(tmp_path)]) == 0
    ledger.write_text(json.dumps({"jobs": [{"jid": "cj2",
                                            "tag": tag + "_dtf0.025"}]}))
    assert sl.main([str(ledger), "--runs", str(tmp_path)]) == 1
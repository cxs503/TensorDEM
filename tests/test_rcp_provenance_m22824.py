"""Tests for the RCP campaign implementation-provenance audit (m22824).

The audit's whole point is to refuse two shortcuts: "the diff looks harmless"
and "the run is on disk, therefore it is comparable".  These tests pin the
decision rules (not just the code paths) with synthetic artefacts:

* a digest change may only be called inert when the settled positions and the
  scalar metrics are *both* reproduced (``EQUIVALENT``) or agree to 1e-12
  (``NUMERICALLY_EQUAL``) -- everything else is ``DIVERGENT``/``MISSING``;
* a timestep pair that mixes digests is reported as *unclean*, so a caller can
  never mistake it for a timestep-only difference;
* a run produced by the *current* digest needs no certificate, one produced by
  an older digest does -- even when the reproduction has not arrived yet
  (``MISSING`` is not a pass).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "case_rcp_deposition"))

import provenance_audit as pa  # noqa: E402


# ---------------------------------------------------------------------------
# decision rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scalar,dpos,expected", [
    (0.0, 0.0, "EQUIVALENT"),
    (0.0, 1e-14, "NUMERICALLY_EQUAL"),
    (1e-13, 0.0, "NUMERICALLY_EQUAL"),
    (1e-13, 1e-13, "NUMERICALLY_EQUAL"),
    (1e-9, 0.0, "DIVERGENT"),
    (0.0, 3e-8, "DIVERGENT"),
    (0.0, None, "MISSING"),
])
def test_verdict_rules(scalar, dpos, expected):
    assert pa.verdict_for(scalar, dpos) == expected


def test_digest_change_never_passes_without_reproduction():
    """A missing reproduction is ``MISSING`` -- never silently equivalent."""
    assert pa.verdict_for(0.0, None) == "MISSING"


# ---------------------------------------------------------------------------
# inventory / confound census
# ---------------------------------------------------------------------------
def _run(tag, *, mu=0.0, kn=4000.0, seed=1, dt=1 / 20, digest="a" * 64,
         phi=0.64):
    return {"tag": tag, "path": tag + ".json", "n": 10000, "mu": mu, "kn": kn,
            "seed": seed, "dt_frac": dt, "impl_sha256": digest,
            "phi_bulk": phi, "phi_bulk_overlap_corrected": phi,
            "coordination_bulk": 6.0, "steps": 100, "final_time": 20.0,
            "settled": True}


def test_hash_classes_counts_cells():
    runs = [_run("a", digest="x" * 64), _run("b", seed=2, digest="x" * 64),
            _run("c", digest="y" * 64, dt=1 / 40)]
    cls = pa.hash_classes(runs)
    assert set(cls) == {"x" * 64, "y" * 64}
    assert cls["x" * 64]["n_runs"] == 2
    assert cls["y" * 64]["dt_fracs"] == [1 / 40]


def test_ladder_pair_mixing_digests_is_flagged_unclean():
    runs = [_run("c1", seed=1, digest="old" + "0" * 61),
            _run("c2", seed=2, digest="new" + "0" * 61),
            _run("f1", seed=1, dt=1 / 40, digest="new" + "0" * 61)]
    (lvl,) = pa.ladder_confounds(runs)
    assert lvl["mixed_digest"] is True
    assert lvl["clean_for_timestep_test"] is False
    assert lvl["n_digests"] == 2
    assert lvl["n_coarse"] == 2 and lvl["n_fine"] == 1


def test_ladder_pair_single_digest_is_clean():
    new = "new" + "0" * 61
    runs = [_run("c1", seed=1, digest=new), _run("c2", seed=2, digest=new),
            _run("f1", seed=1, dt=1 / 40, digest=new)]
    (lvl,) = pa.ladder_confounds(runs)
    assert lvl["mixed_digest"] is False
    assert lvl["clean_for_timestep_test"] is True
    assert lvl["shared_digest_on_both_levels"] is True


def test_ladder_requires_both_timestep_levels():
    runs = [_run("c1"), _run("c2", seed=2)]
    assert pa.ladder_confounds(runs) == []


# ---------------------------------------------------------------------------
# reproduction certificate
# ---------------------------------------------------------------------------
def _write_run(directory, run, positions=None, prefix="deposition"):
    os.makedirs(directory, exist_ok=True)
    payload = {
        "spec": {"n_particles": run["n"], "mu": run["mu"], "kn": run["kn"],
                 "seed": run["seed"], "dt_frac": run["dt_frac"],
                 "restitution": 0.1},
        "metrics": {k: run[k] for k in ("phi_bulk", "phi_bulk_overlap_corrected",
                                        "coordination_bulk")},
        "settled": run["settled"], "final_time": run["final_time"],
        "steps": run["steps"], "impl_sha256": run["impl_sha256"],
    }
    tag = f"{prefix}_{run['tag']}"
    with open(os.path.join(directory, tag + ".json"), "w") as fh:
        json.dump(payload, fh)
    if positions is not None:
        np.savez_compressed(os.path.join(directory, tag + ".npz"),
                            positions=np.asarray(positions, dtype=float))
    return tag


def test_compare_repro_equivalent_when_positions_identical(tmp_path):
    pos = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    a, b = tmp_path / "a", tmp_path / "b"
    ta = _write_run(str(a), _run("t1"), positions=pos)
    tb = _write_run(str(b), _run("t1"), positions=pos,
                    prefix="deposition")
    rec = pa.compare_repro(pa.load_provenance(str(a))[0],
                           pa.load_provenance(str(b))[0],
                           os.path.join(a, ta + ".npz"),
                           os.path.join(b, tb + ".npz"))
    assert rec["same_spec"] is True
    assert rec["verdict"] == "EQUIVALENT"
    assert rec["positions_max_abs_diff"] == 0.0


def test_compare_repro_divergent_on_metric_change(tmp_path):
    pos = np.array([[0.0, 0.0, 1.0]])
    a, b = tmp_path / "a", tmp_path / "b"
    ta = _write_run(str(a), _run("t1"), positions=pos)
    shifted = _run("t1")
    shifted["phi_bulk"] = 0.6405
    tb = _write_run(str(b), shifted, positions=pos)
    rec = pa.compare_repro(pa.load_provenance(str(a))[0],
                           pa.load_provenance(str(b))[0],
                           os.path.join(a, ta + ".npz"),
                           os.path.join(b, tb + ".npz"))
    assert rec["verdict"] == "DIVERGENT"
    assert rec["max_scalar_abs_diff"] == pytest.approx(5e-4)


def test_compare_repro_flags_spec_mismatch(tmp_path):
    pos = np.zeros((1, 3))
    a, b = tmp_path / "a", tmp_path / "b"
    ta = _write_run(str(a), _run("t1"), positions=pos)
    other = _run("t1", seed=7)
    tb = _write_run(str(b), other, positions=pos)
    rec = pa.compare_repro(pa.load_provenance(str(a))[0],
                           pa.load_provenance(str(b))[0],
                           os.path.join(a, ta + ".npz"),
                           os.path.join(b, tb + ".npz"))
    assert rec["same_spec"] is False


# ---------------------------------------------------------------------------
# end-to-end audit bookkeeping
# ---------------------------------------------------------------------------
def test_audit_skips_current_digest_and_marks_old_one_missing(tmp_path,
                                                              monkeypatch):
    runs_dir, repro_dir = tmp_path / "runs", tmp_path / "repro"
    _write_run(str(runs_dir), _run("old1", digest="old" + "0" * 61))
    _write_run(str(runs_dir), _run("new1", digest="new" + "0" * 61))
    monkeypatch.setattr(pa, "current_digest", lambda: "new" + "0" * 61)
    res = pa.audit(str(runs_dir), str(repro_dir))
    assert res["n_runs"] == 2
    assert res["n_runs_off_current_digest"] == 1
    assert res["n_repro_runs"] == 0
    tags = [c["stored_tag"] for c in res["certificates"]]
    assert len(tags) == 1
    assert res["certificates"][0]["verdict"] == "MISSING"
    assert res["all_certified_inert"] is False


def test_audit_certifies_inert_digest_change(tmp_path, monkeypatch):
    runs_dir, repro_dir = tmp_path / "runs", tmp_path / "repro"
    pos = np.array([[0.0, 0.0, 1.0]])
    _write_run(str(runs_dir), _run("old1", digest="old" + "0" * 61),
               positions=pos)
    _write_run(str(repro_dir), _run("old1", digest="new" + "0" * 61),
               positions=pos)
    monkeypatch.setattr(pa, "current_digest", lambda: "new" + "0" * 61)
    res = pa.audit(str(runs_dir), str(repro_dir))
    assert res["certificate_verdicts"] == {"EQUIVALENT": 1}
    assert res["all_certified_inert"] is True
    assert res["certificates"][0]["same_spec"] is True
"""The deposited-case artefacts must be regenerable, not typed.

``docs/CASE_RCP_DEPOSITION.md`` and ``runs/summary.json`` are *outputs* of
``benchmarks/case_rcp_deposition/aggregate.py``.  Two failure modes are
possible and both have happened in this case's history:

* the committed artefacts drift from the run set (a probe wave lands, nobody
  re-runs the aggregator, and the document keeps quoting the older numbers);
* the document points at figures that were never added to the tree.

These tests rebuild the artefacts from ``runs/*.json`` in a throw-away mirror
of the repository layout and compare byte for byte with what is committed.
Nothing here re-simulates: the mirror only symlinks ``src/``, the reference
anchor ``benchmarks/rcp_jamming_audit/runs/audit.json`` and the run JSONs, so a
mismatch means a stale artefact, never a flaky rerun.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(HERE, "benchmarks", "case_rcp_deposition")
DOC = os.path.join(HERE, "docs", "CASE_RCP_DEPOSITION.md")
SUMMARY = os.path.join(CASE, "runs", "summary.json")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _mirror(root: str) -> str:
    """Build a minimal repository layout that ``aggregate.py`` can run in."""
    os.makedirs(os.path.join(root, "benchmarks", "case_rcp_deposition", "runs"))
    os.makedirs(os.path.join(root, "benchmarks", "rcp_jamming_audit", "runs"))
    os.makedirs(os.path.join(root, "docs"))
    os.symlink(os.path.join(HERE, "src"), os.path.join(root, "src"))
    for name in ("aggregate.py", "plan.json"):
        os.symlink(os.path.join(CASE, name),
                   os.path.join(root, "benchmarks", "case_rcp_deposition", name))
    anchor = os.path.join(HERE, "benchmarks", "rcp_jamming_audit", "runs", "audit.json")
    if os.path.exists(anchor):
        os.symlink(anchor,
                   os.path.join(root, "benchmarks", "rcp_jamming_audit", "runs",
                                "audit.json"))
    runs_src = os.path.join(CASE, "runs")
    for name in sorted(os.listdir(runs_src)):
        if name.startswith("deposition_") and name.endswith(".json"):
            os.symlink(os.path.join(runs_src, name),
                       os.path.join(root, "benchmarks", "case_rcp_deposition",
                                    "runs", name))
    return root


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("rcp_deposition_artifacts"))
    if not os.listdir(os.path.join(CASE, "runs")):
        pytest.skip("no deposition runs in the checkout")
    _mirror(root)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable,
         os.path.join(root, "benchmarks", "case_rcp_deposition", "aggregate.py")],
        cwd=root, env=env, capture_output=True, text=True)
    if proc.returncode != 0:                    # pragma: no cover
        pytest.fail(f"aggregate.py failed:\n{proc.stdout}\n{proc.stderr}")
    return {"root": root, "stdout": proc.stdout}


def test_summary_json_is_the_aggregator_output(rebuilt):
    """A run set change must reach `runs/summary.json`, not just `runs/`."""
    fresh = os.path.join(rebuilt["root"], "benchmarks", "case_rcp_deposition",
                         "runs", "summary.json")
    got = json.load(open(fresh))
    committed = json.load(open(SUMMARY))
    if got != committed:
        diffs = sorted(k for k in set(got) | set(committed)
                       if got.get(k) != committed.get(k))
        pytest.fail("runs/summary.json is stale vs the run set; differing keys: "
                    + ", ".join(diffs))
    shutil.rmtree(os.path.join(rebuilt["root"], "benchmarks", "case_rcp_deposition",
                               "runs"), ignore_errors=True)


def test_document_is_the_aggregator_output(rebuilt):
    """Every number in the document is generated; the text must match."""
    fresh = os.path.join(rebuilt["root"], "docs", "CASE_RCP_DEPOSITION.md")
    want = open(fresh).read()
    have = open(DOC).read()
    if want != have:
        want_lines, have_lines = want.splitlines(), have.splitlines()
        first = next((i for i, (a, b) in enumerate(zip(want_lines, have_lines))
                      if a != b), min(len(want_lines), len(have_lines)))
        pytest.fail(
            f"docs/CASE_RCP_DEPOSITION.md is stale vs aggregate.py "
            f"(first differing line {first + 1}):\n"
            f"  committed: {have_lines[first] if first < len(have_lines) else '<eof>'}\n"
            f"  regenerated: {want_lines[first] if first < len(want_lines) else '<eof>'}")


def test_every_referenced_figure_is_in_the_tree(rebuilt):
    """A `![...](...)` link into a file that does not exist is a broken case."""
    text = open(DOC).read()
    refs = []
    for chunk in text.split("![")[1:]:
        target = chunk.split("](", 1)[1].split(")", 1)[0]
        refs.append(target)
    assert refs, "the document references no figures"
    for ref in refs:
        path = os.path.normpath(os.path.join(HERE, "docs", ref))
        assert os.path.isfile(path), f"document references a missing figure: {ref}"
        rel = os.path.relpath(path, HERE).replace(os.sep, "/")
        assert rel.startswith("benchmarks/case_rcp_deposition/figures/"), rel
        blob = open(path, "rb").read()
        assert blob.startswith(PNG_MAGIC), f"{ref} is not a PNG"
        assert len(blob) > 5000, f"{ref} looks empty ({len(blob)} bytes)"
    figures = json.load(open(SUMMARY))["figures"]
    assert figures, "summary.json lists no figures"
    for f in figures:
        assert any(f.endswith(os.path.basename(r)) for r in refs), \
            f"summary.json lists {f} but the document never shows it"


def test_gate_flags_cannot_disagree_with_the_gate_they_summarise():
    """`all_gates_pass` is the AND of the gate list; the document repeats it."""
    summary = json.load(open(SUMMARY))
    gates = summary["gates"]
    assert gates, "no gates"
    ids = [g["id"] for g in gates]
    assert len(ids) == len(set(ids)), "duplicate gate id"
    for g in gates:
        assert {"id", "criterion", "got", "pass"} <= set(g)
        assert isinstance(g["pass"], bool)
    assert summary["all_gates_pass"] == all(g["pass"] for g in gates)
    text = open(DOC).read()
    for g in gates:
        assert g["id"] in text, f"gate {g['id']} is not in the document table"


def test_production_arm_carries_a_single_kernel_revision(rebuilt):
    """The document quotes one kernel stamp: it must be the production one.

    The run set spans several module revisions (the module grew analysis
    helpers between waves), so a stamp taken from an arbitrary run -- the old
    ``runs[0]`` behaviour -- would describe a run the case does not rest on.
    """
    summary = json.load(open(SUMMARY))
    runs = summary["runs"]
    assert len(summary["impl_hashes"]) >= 1
    assert sum(summary["impl_hashes"].values()) == len(runs)
    prod = [r for r in runs
            if r["n"] == summary["n_primary"]
            and abs(r["kn"] - summary["kn_production"]) < 1e-9
            and abs(r["dt_frac"] - 1.0 / 20.0) < 1e-12]
    assert prod, "no production run in the summary"
    assert len({r["impl"] for r in prod}) == 1, "production arm mixes kernels"
    assert summary["impl_sha256"] == prod[0]["impl"]
    assert summary["impl_sha256"] in open(DOC).read()


def test_production_stiffness_is_the_stiffest_seeded_tier():
    """The escalation rule is read off the artefacts, not written down by hand.

    Two levels carrying the seed coverage is only legitimate if the softer one
    is still there as a reference -- otherwise the escalation would be a
    substitution.
    """
    summary = json.load(open(SUMMARY))
    kn_prod = summary["kn_production"]
    n = summary["n_primary"]
    runs = summary["runs"]
    for mu in (0.0, 0.1):
        seeds = {r["seed"] for r in runs
                 if r["n"] == n and abs(r["mu"] - mu) < 1e-12
                 and abs(r["kn"] - kn_prod) < 1e-9
                 and abs(r["dt_frac"] - 1.0 / 20.0) < 1e-12}
        assert len(seeds) >= 3, f"tier mu={mu} has {len(seeds)} seeds at k_n={kn_prod}"
    levels = summary["kn_levels"]
    # the ladder must stay non-degenerate: either the production stiffness sits
    # above a softer arm that is kept as the softness reference, or (production
    # on the softest level) the stiffer levels above it quantify the systematic
    # an escalation would remove.  A single level would support neither claim.
    if min(levels) != kn_prod:
        assert [k for k in levels if k < kn_prod], \
            "no softer arm kept in the run set"
    assert [k for k in levels if k > kn_prod], \
        "no stiffer level to extrapolate the hard-sphere limit from"
    stiff = summary["stiffness"]
    for key, s in stiff.items():
        assert len(s["kn"]) >= 3, f"{key} ladder lost its levels"
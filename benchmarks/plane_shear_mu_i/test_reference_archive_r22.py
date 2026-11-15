"""Round-22 tests: the archived provenance, the parsed law, and the re-judgement.

These tests are the acceptance gate for round 22.  They are offline: they read
``runs/provenance_archive/sources.json`` (with its per-source excerpt and
decimal-inventory files) and ``runs/round22/analysis_r22.json``.

The load-bearing assertions are:

1. every constant the law model uses is a **decimal literal that occurs in the
   archived paper text** (token membership in the inventory) - so no number in
   the benchmark is hand-copied;
2. **neither cited paper contains the two in-repo triples** - the citation
   defect is falsified by the archive, not by opinion;
3. the archived linear law is *not* representable by either triple (they miss it
   by far more than the 10 % tolerance) although the Jop form itself can
   represent it to a few percent with a different triple;
4. the re-judged errors in ``analysis_r22.json`` recompute exactly from the
   archived inputs.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plane_shear_mu_i.reference_archive import (  # noqa: E402
    ATTRIBUTED_TRIPLES, JOP_FORMS, SOURCES_JSON, archived_law, fit_jop_to_law,
    jop_mu, load_manifest, parse_dacruz, parse_gdr_midi,
)
from plane_shear_mu_i import analyse_r22  # noqa: E402

ARCHIVE_DIR = os.path.join(HERE, "plane_shear_mu_i", "runs", "provenance_archive")


@pytest.fixture(scope="module")
def manifest():
    assert os.path.exists(SOURCES_JSON), "run build_reference_archive first"
    return load_manifest()


@pytest.fixture(scope="module")
def law(manifest):
    return archived_law(manifest)


def _inventory(manifest, key):
    for src in manifest["sources"]:
        if src["key"] == key:
            path = os.path.join(HERE, "plane_shear_mu_i", src["numbers_inventory_file"])
            with open(path, encoding="utf-8") as fh:
                return set(fh.read().split())
    raise KeyError(key)


def _floats(manifest, key):
    return {float(t) for t in _inventory(manifest, key)}


# --------------------------------------------------------------------------- #
# 1. archive integrity
# --------------------------------------------------------------------------- #
def test_both_cited_papers_are_archived(manifest):
    keys = {s["key"] for s in manifest["sources"]}
    assert keys == {"dacruz2005", "gdr_midi2004"}
    for s in manifest["sources"]:
        assert re.fullmatch(r"[0-9a-f]{64}", s["pdf_sha256"]), "pdf identity not pinned"
        assert s["n_chars_extracted"] > 50_000
        assert s["arxiv_id"]
        assert s["excerpts"], "no verbatim excerpts"
        for ex in s["excerpts"]:
            assert ex["anchor"] in ex["text"], "excerpt does not contain its anchor"
            assert os.path.exists(os.path.join(HERE, "plane_shear_mu_i", s["excerpts_file"]))


def test_parsed_constants_occur_in_the_paper_text(manifest):
    """No hand-copying: every parsed constant is a literal of the paper."""
    dc = parse_dacruz(manifest)
    inv = _floats(manifest, "dacruz2005")
    for name in ("mu_min", "b", "i_saturation", "nu_max", "a",
                 "mu_min_frictionless"):
        assert float(dc[name]) in inv, (
            "%s=%s is not a number printed in da Cruz 2005" % (name, dc[name]))

    gd = parse_gdr_midi(manifest)
    ginv = _floats(manifest, "gdr_midi2004")
    for name in ("phi_max_quasi_static", "phi_at_I", "I_of_second_point"):
        assert float(gd[name]) in ginv, name


def test_parsing_is_reproducible_from_the_committed_excerpts(manifest):
    """Re-running the regex on the stored excerpt gives the same numbers."""
    dc = parse_dacruz(manifest)
    assert (dc["mu_min"], dc["b"]) == (0.22, 1.0)
    assert (dc["nu_max"], dc["a"]) == (0.81, 0.3)
    assert dc["i_saturation"] == 0.2
    gd = parse_gdr_midi(manifest)
    assert gd["phi_max_quasi_static"] == 0.85
    assert gd["phi_at_I"] == 0.80 and gd["I_of_second_point"] == 0.10


# --------------------------------------------------------------------------- #
# 2. the citation defect
# --------------------------------------------------------------------------- #
def test_neither_cited_paper_contains_the_in_repo_triples(manifest):
    """The two triples are mis-attributed, not merely unverified.

    Not every number of a triple is decisive: da Cruz *does* print ``0.84``
    (the packing fraction at mu_p = 0) and ``0.3`` (the dilatancy slope ``a``),
    but never as the ``mu_2`` / ``I_0`` of a friction law.  What is decisive is
    the defining ``mu_s`` of each triple, which neither paper prints at all, and
    the complete absence of the pytest-suite triple from the paper it names.
    """
    mu_s_of = {"PRIMARY_REPO": 0.36, "TEST_GDR_MIDI": 0.379}
    for src in manifest["sources"]:
        inv = _floats(manifest, src["key"])
        for name, triple in ATTRIBUTED_TRIPLES.items():
            assert mu_s_of[name] not in inv, (
                "%s prints %s (the mu_s of %s) - the mis-attribution claim "
                "would be wrong, re-adjudicate" % (src["key"], mu_s_of[name], name))
            # the paper's own parsed fit is not this triple either
            if src["key"] == "dacruz2005":
                dc = parse_dacruz(manifest)
                assert (dc["mu_min"], dc["b"]) != tuple(triple[:2])
    inv_dc = _floats(manifest, "dacruz2005")
    for v in ATTRIBUTED_TRIPLES["TEST_GDR_MIDI"]:
        assert v not in inv_dc, "%.3f is printed in da Cruz 2005" % v


def test_the_shared_numbers_are_a_documented_false_positive(manifest):
    """0.84 and 0.3 *are* in da Cruz - as phi and as the dilatancy slope."""
    dc = parse_dacruz(manifest)
    inv = _floats(manifest, "dacruz2005")
    assert 0.84 in inv, "packing fraction at mu_p=0 is expected in the text"
    assert float(dc["a"]) == 0.3
    assert "Eq. (13)" in dc["locators"]["dilatancy"], (
        "0.3 must be attributed to Eq. (13) (slope a), never to a friction law")
    assert dc["locators"]["mu_law"].endswith("Eq. (14)")


def test_gdr_midi_publishes_no_fitted_triple(manifest):
    gd = parse_gdr_midi(manifest)
    assert "no fitted mu(I) triple published" in gd["law"]
    assert "mu_s" not in gd


# --------------------------------------------------------------------------- #
# 3. law model
# --------------------------------------------------------------------------- #
def test_archived_law_matches_the_paper_and_refuses_extrapolation(law):
    assert law.mu_min == 0.22 and law.b == 1.0
    assert law.mu(0.01) == pytest.approx(0.23)
    assert law.mu(0.2) == pytest.approx(0.42)
    assert law.mu(0.3) is None, "I > I_sat must not return a law value"
    assert law.mu_extrapolated(0.3) == pytest.approx(0.52)
    assert law.phi(0.0) == pytest.approx(0.81)
    assert law.phi(0.1) == pytest.approx(0.78)


def test_archived_law_is_decreasing_in_phi_and_increasing_in_mu(law):
    I = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1]
    mus = [law.mu(i) for i in I]
    phis = [law.phi(i) for i in I]
    assert all(b > a for a, b in zip(mus, mus[1:]))
    assert all(b < a for a, b in zip(phis, phis[1:]))
    assert all(0.79 < p <= 0.81 for p in phis[:2])


def test_jop_form_can_represent_the_archived_law_but_not_with_either_triple(law):
    fit = fit_jop_to_law(law)
    assert fit["reproduces_law_within_tol"], (
        "the Jop form %s fails to represent da Cruz's own law - then the "
        "triples cannot be checked this way" % JOP_FORMS)
    for name, triple in ATTRIBUTED_TRIPLES.items():
        worst = max(abs(jop_mu(I, *triple) - (law.mu_min + law.b * I)) /
                    (law.mu_min + law.b * I) for I in fit["I_grid"])
        assert worst > 0.10, (
            "%s reproduces the archived da Cruz law to %.1f%% - the "
            "mis-attribution claim would be undermined" % (name, 100 * worst))


# --------------------------------------------------------------------------- #
# 4. the re-judgement
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report():
    return json.load(open(os.path.join(HERE, "plane_shear_mu_i", "runs", "round22",
                                       "analysis_r22.json"), encoding="utf-8"))


def test_rejudged_errors_recompute_from_the_archived_inputs(report, law):
    for row in report["points"]:
        I = row["I_bulk"]
        mu = row["mu"]["mu_full_zz_final"]
        target = law.mu(I)
        entry = row["rel_err"]["DACRUZ_LINEAR"]
        if target is None:
            assert entry["law_in_validity"] is False
            target = law.mu_extrapolated(I)
        else:
            assert entry["law_in_validity"] is True
        assert entry["rel_err"]["mu_full_zz_final"] == pytest.approx((mu - target) / target)
        for name, triple in ATTRIBUTED_TRIPLES.items():
            assert row["rel_err"][name]["rel_err"]["mu_full_zz_final"] == pytest.approx(
                (mu - jop_mu(I, *triple)) / jop_mu(I, *triple))


def test_the_round_does_not_claim_a_pass(report):
    """Honest-FAIL guard: the archived law does not close the ledger either."""
    v = report["verdict"]
    assert v["ledger"] == "FAIL"
    assert v["archived_law_passes_in_validity"] is False
    assert v["phi_passes_in_validity"] is False
    dc = report["candidates"]["DACRUZ_LINEAR"]["mu_full_zz_final"]
    assert dc["n_within_tol_in_validity"] < dc["n"]


def test_analysis_is_reproducible(report):
    fresh = analyse_r22.evaluate()
    assert json.loads(json.dumps(fresh, sort_keys=True)) == \
           json.loads(json.dumps(report, sort_keys=True))
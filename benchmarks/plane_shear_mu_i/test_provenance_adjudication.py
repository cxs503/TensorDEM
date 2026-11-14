"""Tests for the round-13 provenance / consistency adjudication.

These are *verification* tests, not smoke tests: each one pins a statement that
the round-13 document and the ledger verdict rest on.  If a future edit moves
the reference constants, changes a pressure normaliser or re-freezes the
archive, one of these fails and the round-13 claims must be re-derived.

The load-bearing assertion is ``test_no_envelope_corner_passes``: it is what
makes "the provenance dispute is not the blocker" a computed fact rather than an
opinion.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reference_provenance as rp          # noqa: E402
import adjudicate_provenance as adj        # noqa: E402

I_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


# --------------------------------------------------------------------------- #
# single source of truth
# --------------------------------------------------------------------------- #
def test_reference_law_matches_primary_candidate():
    """``reference_law.py`` must be the triple the registry calls PRIMARY.

    ``reference_law.py`` bills itself as the "single source of truth (m17026)";
    this test is the enforcement.
    """
    import reference_law

    assert (reference_law.MU_S, reference_law.MU_2, reference_law.I_0) == rp.PRIMARY_REPO.triple


def test_second_reference_triple_survives_elsewhere():
    """The pytest suite carries a *different* triple.  Record it: as long as it
    exists, "the reference law" is ambiguous and the envelope must be used."""
    src = (HERE.parent.parent / "tests" / "test_mu_i_rheology.py").read_text()
    for name, value in (("GDR_MIDI_MU_S", 0.379),
                        ("GDR_MIDI_MU_2", 0.641),
                        ("GDR_MIDI_I_0", 0.279)):
        assert f"{name} = {value}" in src, f"{name} drifted from the registry"
    assert rp.TEST_GDR_MIDI.triple == (0.379, 0.641, 0.279)


def test_candidates_disagree_materially():
    """The dispute is worth adjudicating only if the two readings differ by more
    than the 10 % acceptance band.  If this ever fails, collapse the registry."""
    spread = rp.envelope_spread(0.3)
    assert spread > 0.10, f"envelope spread at I=0.3 is only {spread:.1%}"


def test_envelope_is_the_full_corner_box():
    assert len(rp.ENVELOPE) == 2 * 2 * 2
    assert len({c.triple for c in rp.ENVELOPE}) == 8


# --------------------------------------------------------------------------- #
# the law itself
# --------------------------------------------------------------------------- #
def test_jop_limits():
    assert rp.jop_mu_I(0.0) == pytest.approx(rp.jop_mu_I(1e-300))
    assert rp.jop_mu_I(1e-12) == pytest.approx(rp.PRIMARY_REPO.mu_s, abs=1e-9)
    assert rp.jop_mu_I(1e12) == pytest.approx(rp.PRIMARY_REPO.mu_2, abs=1e-9)
    # at I = I_0 the law is exactly half-way
    half = rp.PRIMARY_REPO.mu_s + (rp.PRIMARY_REPO.mu_2 - rp.PRIMARY_REPO.mu_s) / 2
    assert rp.jop_mu_I(rp.PRIMARY_REPO.I_0) == pytest.approx(half)


def test_reverse_fit_recovers_a_known_law():
    truth = (0.379, 0.641, 0.279)
    mus = [rp.jop_mu_I(I, *truth) for I in I_GRID]
    fit = rp.fit_jop_law(I_GRID, mus)
    assert fit["identified"]
    assert not fit["degenerate"]
    assert fit["mu_s"] == pytest.approx(truth[0], abs=2e-3)
    assert fit["mu_2"] == pytest.approx(truth[1], abs=2e-3)
    assert fit["I_0"] == pytest.approx(truth[2], rel=0.05)
    assert fit["rms"] < 1e-4


def test_reverse_fit_refuses_to_identify_two_points():
    fit = rp.fit_jop_law([0.01, 0.1], [0.38, 0.45])
    assert not fit["identified"]
    assert "reason" in fit


# --------------------------------------------------------------------------- #
# pressure-normaliser consistency
# --------------------------------------------------------------------------- #
def test_consistent_pair_does_not_mix_normalisers():
    """The two admissible readings differ by exactly the anisotropy ratio, and
    mixing them inflates mu by that ratio without moving the abscissa."""
    out = rp.consistent_pressure_pair(sigma_zz=-62500.0, pressure=50000.0, tau=20000.0)
    assert out["zz"]["mu"] == pytest.approx(20000.0 / 62500.0)
    assert out["p"]["mu"] == pytest.approx(20000.0 / 50000.0)
    assert out["mixing_inflation"] == pytest.approx(1.25)
    assert out["p"]["mu"] / out["zz"]["mu"] == pytest.approx(out["mixing_inflation"])


def test_consistent_pair_rejects_zero_pressure():
    with pytest.raises(ValueError):
        rp.consistent_pressure_pair(sigma_zz=0.0, pressure=1.0, tau=1.0)


def test_inertial_number_scaling():
    """I scales as sqrt(rho/p): doubling p divides I by sqrt(2)."""
    I1 = adj.inertial_number(1.0, 1.0e4)
    I2 = adj.inertial_number(1.0, 2.0e4)
    assert I1 / I2 == pytest.approx(math.sqrt(2.0), rel=1e-12)


# --------------------------------------------------------------------------- #
# the archive, re-derived
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report():
    return adj.main()


def test_archived_abscissa_uses_the_imposed_rate(report):
    """Round-12's ``I_measured`` is built from the imposed platen rate, not from
    the measured bulk rate - the very thing ``rate_ratio`` contradicts.  This is
    the consistency defect the round documents; pin it down numerically."""
    for c in report["consistency"]:
        assert c["I_archived_vs_imposed_rel"] < 0.05, c


def test_rate_deficit_is_recorded(report):
    """At least one point shears far slower than commanded (wall slip/plug flow)."""
    ratios = [c["rate_ratio"] for c in report["consistency"]]
    assert min(ratios) < 0.50
    assert sum(1 for c in report["consistency"] if c["kinematically_admissible"]) == 1


def test_no_envelope_corner_passes(report):
    """THE load-bearing result.

    Not one corner of the box spanned by the two disputed triples puts every
    point inside the 10 % band - under *any* of the admissible pairings.  So the
    ledger row cannot be closed by resolving the citation; something else is
    binding.
    """
    for a in report["arms"]:
        ec = a["envelope_corners"]
        assert ec["n_corners_passing_all"] == 0, (a["label"], ec)
        assert ec["best_corner_n_within_tol"] < ec["n_points"], a["label"]


def test_law_consistent_pairing_beats_the_archived_convention(report):
    """Normalising mu and I by the same pressure is not cosmetic: the
    law-consistent arm is strictly better than the archived mixed convention."""
    by_label = {a["label"]: a for a in report["arms"]}
    archived = by_label["zz-consistent, nominal (imposed) rate  [archived convention]"]
    consistent = by_label["p-consistent, measured bulk rate  [law-consistent]"]
    best_arch = min(v["max_abs_rel_err"] for v in archived["verdicts"])
    best_cons = min(v["max_abs_rel_err"] for v in consistent["verdicts"])
    assert best_cons < best_arch, (best_cons, best_arch)


def test_oracle_fit_is_not_identified(report):
    """A Jop law can be forced under 10 % - but only by pinning mu_2 to the edge
    of the search box, i.e. outside both candidate papers.  Not a rescue."""
    h = report["headline"]
    assert h["min_max_abs_rel_err_over_all_laws"] <= adj.TOL
    assert h["oracle_boundary_hit"], "oracle optimum moved off the box edge"
    assert h["oracle_provenance_can_rescue"] is False


def test_published_json_is_current(report):
    """The committed artefact must equal a fresh recomputation (no stale tables)."""
    on_disk = json.loads((adj.RUNS / "provenance_adjudication.json").read_text())
    assert on_disk["headline"] == report["headline"]
    assert len(on_disk["arms"]) == len(report["arms"])


def test_tables_are_rendered(report):
    md = (adj.RUNS / "provenance_adjudication_tables.md").read_text()
    for heading in ("Table A", "Table B", "Table C", "Table D", "Table D2", "Table E"):
        assert heading in md
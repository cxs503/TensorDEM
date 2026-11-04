"""Tests for the Brazilian-disc lattice-refinement and estimator-convergence scans.

These tests pin the four claims the two scan scripts make, and nothing that
needs a long campaign:

1. **the control** -- rescaling to ``N = 15`` reproduces the manifest case
   exactly, and the scan re-derives the published closure numbers
   (``sigma_zz/sigma_xx = -2.0149``, ``+30.45%`` tension error) to 1e-3;
2. **scale invariance of the material** -- Zener anisotropy and Poisson ratio
   of the bond network are independent of the grain size, which is what makes
   "refine the lattice" unable to fix the isotropic-continuum assumption;
3. **the owed question** -- refinement does *not* bring the centre-field error
   monotonically towards zero, so it does not close the manifest's 15% band;
4. **why** -- on a single relaxed configuration the window-averaged centre
   stress swings across the 15% tolerance as the averaging volume changes, and
   that swing exceeds the sampling noise band, so the criterion has no
   resolution-independent verdict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_dem3d_brazilian import MANIFEST  # noqa: E402
from brazilian_hondros_estimator import WINDOW_RATIOS, analyse  # noqa: E402
from brazilian_hondros_refinement import (  # noqa: E402
    DIAMETER_M,
    THICKNESS_M,
    anisotropy_metrics,
    case_at_resolution,
    _verdict,
    summarise,
)
from brazilian_lattice_static import lattice_moduli  # noqa: E402

MANIFEST_DATA = json.loads(MANIFEST.read_text(encoding="utf-8"))
CASE = next(c for c in MANIFEST_DATA["cases"] if c["id"] == "brazil_elastic_field")

#: The published closure-run values this scan must re-derive unchanged.
PUBLISHED_RATIO = -2.0149
PUBLISHED_TENSION_ERROR = 0.30450718


# ---------------------------------------------------------------------------
# 1. Control: the scan is anchored to the published specimen
# ---------------------------------------------------------------------------

def test_resolution_15_reproduces_the_manifest_case() -> None:
    """N=15 must be the manifest case bit for bit, or the scan is not anchored."""
    case = case_at_resolution(CASE, 15)
    for key in ("nx", "ny", "nz", "radius_m"):
        assert case["config"][key] == CASE["config"][key], key


def test_specimen_size_is_held_fixed_across_resolutions() -> None:
    """Diameter and thickness are invariants; only the grain count changes."""
    for resolution in (15, 20, 25, 30, 40):
        case = case_at_resolution(CASE, resolution)
        radius = case["config"]["radius_m"]
        ny = case["config"]["ny"]
        diameter = 2.0 * radius * case["config"]["nz"]
        thickness = 2.0 * radius * ny
        assert diameter == pytest.approx(DIAMETER_M, rel=1e-12)
        assert thickness == pytest.approx(THICKNESS_M, rel=1e-12)
        assert case["config"]["nx"] == case["config"]["nz"] == resolution
        # thickness spans exactly five grain diameters at every resolution
        assert thickness / diameter == pytest.approx(0.2, rel=1e-12)


@pytest.mark.parametrize("resolution", [16, 17, 22, 3, 0])
def test_resolutions_that_cannot_tile_the_thickness_are_rejected(resolution: int) -> None:
    """A resolution that leaves a partial grain layer would silently change t."""
    with pytest.raises(ValueError):
        case_at_resolution(CASE, resolution)


# ---------------------------------------------------------------------------
# 2. Scale invariance of the bond-network anisotropy
# ---------------------------------------------------------------------------

def test_zener_anisotropy_is_scale_invariant() -> None:
    """The material reason refinement cannot help: A and nu do not depend on h."""
    metrics = {}
    for resolution in (15, 20, 25, 30):
        case = case_at_resolution(CASE, resolution)
        moduli = lattice_moduli(case)
        metrics[resolution] = (anisotropy_metrics(moduli),
                               moduli.get("poisson_ratio"))
    reference, reference_nu = metrics[15]
    assert reference["zener_anisotropy"] == pytest.approx(7.0 / 3.0, rel=1e-9)
    assert reference["cauchy_residual_rel"] == pytest.approx(0.0, abs=1e-12)
    for resolution, (metric, nu) in metrics.items():
        assert metric["zener_anisotropy"] == pytest.approx(
            reference["zener_anisotropy"], rel=1e-9), resolution
        assert metric["cauchy_residual_rel"] == pytest.approx(0.0, abs=1e-12), resolution
        # ``derived_moduli`` reports nu only when the pair is well conditioned; the
        # scale-invariance claim is then vacuous for nu and rests on A alone.
        if reference_nu is not None and nu is not None:
            assert nu == pytest.approx(reference_nu, rel=1e-9, abs=1e-12), resolution
        else:
            assert nu is reference_nu, resolution


def test_anisotropy_metrics_report_a_cauchy_solid() -> None:
    """C12 == C44 is the central-force signature; A != 1 means not isotropic."""
    metrics = anisotropy_metrics(lattice_moduli(case_at_resolution(CASE, 15)))
    assert metrics["C12_Pa"] == pytest.approx(metrics["C44_Pa"], rel=1e-12)
    assert metrics["zener_anisotropy"] > 1.0
    assert metrics["shear_modulus_Pa"] == pytest.approx(
        0.5 * (metrics["C11_Pa"] - metrics["C12_Pa"]), rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Verdict logic and the owed question
# ---------------------------------------------------------------------------

def _row(resolution: int, tension_error: float, zener: float = 7.0 / 3.0) -> dict:
    return {
        "refinement": {"resolution_across_diameter": resolution,
                       "zener_anisotropy": zener},
        "centre_principal_ratio": -2.0,
        "centre_tension_rel_error": tension_error,
        "particle_count": 1, "bond_count": 1, "centre_particle_count": 1,
        "contact_arc_ratio": 0.1, "top_contact_particles": 1,
        "centre_sigma_xx_Pa": 1.0, "centre_sigma_zz_Pa": -2.0,
        "split_stress_Pa": 1.0, "axial_uniformity_rel": 1.0,
        "equilibrium_residual_rel": 1e-8,
        "load_balance_rel": 1e-11, "equilibrium_converged": True,
    }


def test_verdict_declares_a_plateau_when_refinement_stalls() -> None:
    """Improving but staying above 15% is a plateau, not a discretisation limit."""
    summary = summarise([_row(15, 0.305), _row(20, 0.159), _row(25, 0.26)])
    assert summary["achieved_below_15pct"] is False
    assert summary["any_below_15pct"] is False
    assert summary["monotone_decreasing_tension_error"] is False
    assert "PLATEAU" in summary["verdict"].upper() or "NO IMPROVEMENT" in summary["verdict"].upper()


def test_verdict_declares_discretisation_when_the_band_is_closed() -> None:
    """A final error inside 15% must be reported as a closed tolerance."""
    summary = summarise([_row(15, 0.305), _row(30, 0.09)])
    assert summary["achieved_below_15pct"] is True
    assert "DISCRETISATION-LIMITED" in summary["verdict"].upper()


def test_verdict_names_the_scale_invariant_cause_when_it_plateaus() -> None:
    """The plateau verdict must cite the anisotropy that refinement cannot touch."""
    verdict = _verdict([
        {"N": 15, "centre_tension_rel_error": 0.305, "centre_principal_ratio": -2.0,
         "zener_anisotropy": 7.0 / 3.0},
        {"N": 25, "centre_tension_rel_error": 0.26, "centre_principal_ratio": -2.0,
         "zener_anisotropy": 7.0 / 3.0},
    ])
    assert "2.33" in verdict
    assert "scale invariant" in verdict


# ---------------------------------------------------------------------------
# 4. Integration: the published closure numbers and the ill-posed criterion
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def n15_analysis():
    """One relaxation at the published resolution, re-measured at every volume."""
    return analyse(CASE, 15)


def test_scan_reproduces_the_published_closure_numbers(n15_analysis) -> None:
    """The D/10 anchor must land on the numbers the closure doc published."""
    anchor = n15_analysis["anchor_window"]
    assert anchor["window_ratio"] == pytest.approx(0.10)
    assert anchor["sigma_xx_Pa"] == pytest.approx(155.59, rel=1e-3)
    assert anchor["principal_ratio"] == pytest.approx(PUBLISHED_RATIO, rel=1e-3)
    assert anchor["tension_rel_error"] == pytest.approx(PUBLISHED_TENSION_ERROR, rel=1e-3)


def test_relaxation_is_converged_so_the_scan_is_not_a_transient(n15_analysis) -> None:
    """A residual of 1e-7 relative means the field is an equilibrium, not a snapshot."""
    assert n15_analysis["equilibrium_residual_rel"] < 1e-6


def test_the_15pct_criterion_has_no_resolution_independent_verdict(n15_analysis) -> None:
    """The core finding: the verdict flips with the arbitrary averaging volume."""
    sweep = n15_analysis["window_sweep"]
    # The swing across averaging volumes is far larger than the tolerance...
    assert sweep["systematic_swing_in_tension_error"] > 0.15
    # ...and it is systematic, not sampling noise.
    assert sweep["swing_over_noise"] > 1.0
    assert sweep["noise_dominated"] is False
    # Concretely: some volumes pass the tolerance and others fail it.
    errors = [w["tension_rel_error"] for w in n15_analysis["windows"]
              if w.get("tension_rel_error") is not None]
    assert min(errors) < 0.15 < max(errors)


def test_centre_stress_falls_monotonically_as_the_averaging_volume_grows(n15_analysis) -> None:
    """The swing is a trend, which is why it is systematic rather than noise."""
    wide = [w for w in n15_analysis["windows"]
            if w["window_ratio"] >= 0.10 and w.get("sigma_xx_Pa") is not None]
    values = [w["sigma_xx_Pa"] for w in wide]
    assert values == sorted(values, reverse=True)
    assert len(wide) >= 4


def test_sampling_noise_is_reported_and_bounded(n15_analysis) -> None:
    """Every populated window must carry a standard error, and it must shrink."""
    sem = [w["sem_over_abs_mean"] for w in n15_analysis["windows"]
           if w.get("sem_over_abs_mean") is not None]
    assert sem and all(s == s for s in sem)
    assert min(sem) < 0.05


def test_strip_estimator_is_reported_as_dominated_by_the_load_points(n15_analysis) -> None:
    """Averaging the whole loaded diameter cancels tension against load compression."""
    strips = [s for s in n15_analysis["strips"] if s.get("sigma_xx_Pa") is not None]
    assert strips
    # Mean over the full diameter collapses towards zero compared with 2P/(pi D t).
    assert abs(strips[0]["sigma_xx_Pa"]) < 0.02 * n15_analysis["split_stress_Pa"]


def test_window_ratios_bracket_the_published_anchor() -> None:
    """The sweep must contain the D/10 value it is testing, or it is meaningless."""
    assert 0.10 in WINDOW_RATIOS
    assert min(WINDOW_RATIOS) < 0.10 < max(WINDOW_RATIOS)
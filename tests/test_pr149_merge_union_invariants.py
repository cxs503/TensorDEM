"""PR #149 union invariants: neither side of the cure may be re-overlaid away.

The branch ``agent/m2772`` and ``main`` independently produced two studies of the
same DEM3D direct-shear specimen (a wide ``1/n`` series on ``main`` and an
interior-window traction census on the branch).  The cure of PR #149 merged them
by re-homing the branch's artifacts into their own namespace instead of letting
one ``report.json`` clobber the other.  These tests pin that property so a later
mechanical overlay cannot silently delete either side again:

* both artifact namespaces exist and still hold *different* datasets,
* both documentation pages exist and cross-reference each other (symmetric),
* the two audit scripts write to disjoint trees and each path is repo-anchored,
* ``static_direct_shear`` still exposes *both* opt-in flags,
* the union test module still contains every test from both sides.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
WIDE_DIR = REPO / "docs" / "dem3d-shear-cross-section-convergence"
WINDOW_DIR = REPO / "docs" / "dem3d-shear-interior-window-convergence"
WIDE_PAGE = REPO / "docs" / "dem3d_shear_cross_section_convergence.md"
WINDOW_PAGE = REPO / "docs" / "dem3d_shear_interior_window_convergence.md"
WIDE_SCRIPT = REPO / "scripts" / "audit_dem3d_shear_cross_section_convergence.py"
WINDOW_SCRIPT = REPO / "scripts" / "audit_dem3d_shear_interior_window.py"
UNION_TESTS = REPO / "tests" / "test_dem3d_shear_cross_section_convergence.py"

# test names that came from each side of the fork; dropping either group is the
# silent-overlay failure mode this file guards against
MAIN_SIDE_TESTS = {
    "test_series_is_monotone_and_conserves_the_affine_census",
    "test_boundary_layer_fit_is_first_order_with_no_asymptotic_bias",
    "test_gate_prediction_sits_above_the_fitted_range",
    "test_wider_cross_section_reduces_the_deficit_by_the_predicted_ratio",
    "test_surface_lag_decays_inside_a_few_lattice_spacings",
    "test_displacement_field_is_optional_and_consistent",
    "test_invalid_inputs_are_rejected",
}
BRANCH_SIDE_TESTS = {
    "test_interior_window_is_exactly_c44_under_affine_shear",
    "test_whole_layer_modulus_increases_with_cross_section",
    "test_quadratic_extrapolation_recovers_the_analytic_c44",
    "test_thin_specimen_interior_window_recovers_c44_within_one_percent",
    "test_window_measure_is_linear_in_gamma",
    "test_thick_specimen_modulus_is_a_specimen_property_not_a_material_constant",
    "test_layer_argument_rejects_out_of_range_indices",
    "test_fit_boundary_layer_recovers_a_synthetic_decay_length",
    "test_margin_is_clamped_on_specimens_too_small_for_the_window",
    "test_lattice_spacing_and_c44_reference_are_the_documented_closed_forms",
}


def _out_constant(script: pathlib.Path) -> str:
    """Return the literal the script assigns to ``OUT`` (repo-anchored form)."""
    source = script.read_text()
    match = re.search(r'^OUT = (.+)$', source, flags=re.MULTILINE)
    assert match, f"{script.name} defines no OUT constant"
    return match.group(1)


def test_both_artifact_namespaces_survive_and_stay_distinct():
    for name in ("report.json", "summary.md"):
        assert (WIDE_DIR / name).is_file(), f"main's {name} was lost"
        assert (WINDOW_DIR / name).is_file(), f"branch's {name} was lost"
    assert (WIDE_DIR / "series.csv").is_file(), "main's series.csv was lost"
    assert (WIDE_DIR / "figures" / "convergence.png").is_file()
    assert (WINDOW_DIR / "figures" / "cross_section_convergence.png").is_file()

    wide = json.loads((WIDE_DIR / "report.json").read_text())
    window = json.loads((WINDOW_DIR / "report.json").read_text())
    # they are different studies, not two copies of the same file
    assert wide["schema"] == "tensordem-dem3d-shear-cross-section-convergence-v1"
    assert "series" in wide and "fit_square_quadratic" in window
    assert set(wide) != set(window)


def test_documentation_pages_cross_reference_each_other():
    wide_text = WIDE_PAGE.read_text()
    window_text = WINDOW_PAGE.read_text()
    assert "dem3d_shear_interior_window_convergence.md" in wide_text, (
        "the wide-series page must point at the interior-window page"
    )
    assert "dem3d_shear_cross_section_convergence.md" in window_text, (
        "the interior-window page must point at the wide-series page"
    )
    # both sides' headline numbers survive in the merged prose
    for number in ("0.99899", "0.14 %", "0.19 %", "0.63 a", "0.790"):
        assert number in wide_text or number in window_text


@pytest.mark.parametrize("script", [WIDE_SCRIPT, WINDOW_SCRIPT], ids=lambda p: p.name)
def test_audit_scripts_parse_and_anchor_their_output_tree(script: pathlib.Path):
    ast.parse(script.read_text(), filename=str(script))
    literal = _out_constant(script)
    assert literal.startswith("pathlib.Path(__file__)"), (
        f"{script.name} must anchor OUT to the repo, not to the current directory"
    )
    assert "/" in literal


def test_audit_scripts_write_disjoint_trees():
    wide, window = _out_constant(WIDE_SCRIPT), _out_constant(WINDOW_SCRIPT)
    assert "dem3d-shear-cross-section-convergence" in wide
    assert "dem3d-shear-interior-window-convergence" in window
    assert "interior-window" not in wide and "cross-section-convergence" not in window


def test_static_direct_shear_keeps_both_opt_in_flags():
    import inspect

    from tensordem.shear_static import static_direct_shear

    params = inspect.signature(static_direct_shear).parameters
    assert "return_displacement" in params, "main's flag was dropped"
    assert "include_state" in params, "the branch's flag was dropped"
    assert params["return_displacement"].default is False
    assert params["include_state"].default is False


def test_union_test_module_keeps_every_test_from_both_sides():
    tree = ast.parse(UNION_TESTS.read_text(), filename=str(UNION_TESTS))
    present = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }
    missing_main = MAIN_SIDE_TESTS - present
    missing_branch = BRANCH_SIDE_TESTS - present
    assert not missing_main, f"tests from main silently dropped: {sorted(missing_main)}"
    assert not missing_branch, f"tests from the branch silently dropped: {sorted(missing_branch)}"
    assert len(present) == len(MAIN_SIDE_TESTS) + len(BRANCH_SIDE_TESTS)


def test_union_audit_tool_is_committed_with_its_reviewed_exceptions():
    tool = REPO / "scripts" / "audit_pr149_merge_union.py"
    exceptions = REPO / "docs" / "pr149_merge_review_exceptions.json"
    assert tool.is_file() and exceptions.is_file()
    ast.parse(tool.read_text(), filename=str(tool))
    reviewed = json.loads(exceptions.read_text())
    for path, entry in reviewed.items():
        assert (REPO / path).is_file(), f"review exception refers to a missing file {path}"
        assert entry["reason"].strip(), f"exception for {path} carries no reason"
        assert entry["lines"], f"exception for {path} lists no lines"
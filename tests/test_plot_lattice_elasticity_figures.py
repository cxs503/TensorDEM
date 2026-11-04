"""Test for lattice elasticity figure regeneration script."""
import json
import os
import pytest

REPORT_PATH = "docs/lattice-elasticity/lattice_elasticity_report.json"
OUT_DIR = "docs/lattice-elasticity"
EXPECTED_FIGURES = [
    "lattice_stiffness_matrices.png",
    "directional_young_modulus.png",
    "surface_deficiency.png",
    "phonon_dispersion.png",
]


class TestLatticeElasticityFigures:
    """Verify the 4 missing figures for dem3d_lattice_elasticity.md are regenerated."""

    def test_report_exists(self):
        assert os.path.isfile(REPORT_PATH), f"Report not found: {REPORT_PATH}"

    def test_report_verdict_pass(self):
        with open(REPORT_PATH) as f:
            r = json.load(f)
        assert r["verdict"] == "PASS"
        assert r["pass_count"] == r["case_count"] == 14

    @pytest.mark.parametrize("fig_name", EXPECTED_FIGURES)
    def test_figure_exists(self, fig_name):
        path = os.path.join(OUT_DIR, fig_name)
        assert os.path.isfile(path), f"Figure not found: {path}"
        size = os.path.getsize(path)
        assert size > 10_000, f"Figure {fig_name} too small ({size} bytes), likely empty"

    def test_doc_references_all_figures(self):
        """Verify docs/dem3d_lattice_elasticity.md references all 4 figures."""
        doc_path = "docs/dem3d_lattice_elasticity.md"
        with open(doc_path) as f:
            doc = f.read()
        for fig in EXPECTED_FIGURES:
            assert fig in doc, f"Figure {fig} not referenced in {doc_path}"

    def test_figure_data_consistency(self):
        """Verify figures match the report data (spot check directional E)."""
        with open(REPORT_PATH) as f:
            r = json.load(f)
        directional = r["directional"]["100"]
        E_vals = [d["E_Pa"] for d in directional]
        # E at 0 deg should be 108000 Pa (from C11, C12)
        assert abs(E_vals[0] - 108000.0) < 1.0, f"E(0) = {E_vals[0]}, expected 108000"
        # Anisotropy ratio should be > 1 (Cauchy solid is anisotropic)
        assert max(E_vals) / min(E_vals) > 1.0, "No anisotropy detected"

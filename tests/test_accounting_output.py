"""Tests for accounting output in diagnostics and CLI."""

import json
import math
import tempfile
from pathlib import Path

import torch

from tensordem import DEMConfig, IceDEM
from tensordem.cli import main


def test_diagnostics_accounting_fields():
    """Verify diagnostics() returns flattened accounting fields with finite values."""
    config = DEMConfig(nx=5, ny=3, tool_speed=0.2)
    simulation = IceDEM(config)
    
    # Run a few steps
    for _ in range(10):
        simulation.step()
    
    diag = simulation.diagnostics()
    
    # Check new accounting fields exist
    required_fields = [
        "tool_impulse_on_ice_x_Ns", "tool_impulse_on_ice_y_Ns",
        "support_impulse_x_Ns", "support_impulse_y_Ns",
        "drag_impulse_x_Ns", "drag_impulse_y_Ns",
        "external_impulse_x_Ns", "external_impulse_y_Ns",
        "fracture_release_J",
        "tool_work_J", "damping_dissipation_J", "external_work_J",
    ]
    
    for field in required_fields:
        assert field in diag, f"Missing field: {field}"
        assert math.isfinite(diag[field]), f"Field {field} is not finite: {diag[field]}"
    
    # Check existing fields still present
    existing_fields = [
        "time", "tool_x", "tool_y", "reaction_x", "reaction_y",
        "boundary_reaction_x", "boundary_reaction_y", "broken_bonds", "kinetic_energy",
    ]
    for field in existing_fields:
        assert field in diag, f"Missing existing field: {field}"


def test_accounting_json_output():
    """Verify CLI writes accounting.json with all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        
        # Run CLI
        main([
            "--nx", "5", "--ny", "3", "--steps", "10",
            "--output", str(output_dir),
        ])
        
        # Check accounting.json exists
        accounting_file = output_dir / "accounting.json"
        assert accounting_file.exists(), "accounting.json not created"
        
        # Load and validate
        with accounting_file.open() as f:
            accounting = json.load(f)
        
        # Check required fields from self.accounting
        required_accounting_fields = [
            "external_work_J", "fracture_release_J", "tool_work_J",
            "damping_dissipation_J", "external_impulse_Ns",
            "support_impulse_Ns", "drag_impulse_Ns", "tool_impulse_on_ice_Ns",
        ]
        for field in required_accounting_fields:
            assert field in accounting, f"Missing accounting field: {field}"
        
        # Check extra fields
        required_extra_fields = [
            "total_bonds", "broken_bonds", "failure_energy_sum_J",
            "total_mass_kg", "dt", "steps",
        ]
        for field in required_extra_fields:
            assert field in accounting, f"Missing extra field: {field}"
        
        # Check values are finite
        for key, value in accounting.items():
            if isinstance(value, (int, float)):
                assert math.isfinite(value), f"Field {key} is not finite: {value}"
            elif isinstance(value, list):
                for v in value:
                    assert math.isfinite(v), f"Field {key} contains non-finite value: {v}"
        
        # Check types
        assert isinstance(accounting["total_bonds"], int)
        assert isinstance(accounting["broken_bonds"], int)
        assert isinstance(accounting["steps"], int)
        assert isinstance(accounting["failure_energy_sum_J"], float)
        assert isinstance(accounting["total_mass_kg"], float)
        assert isinstance(accounting["dt"], float)
        
        # Check fracture_release_J >= 0
        assert accounting["fracture_release_J"] >= 0, "fracture_release_J must be >= 0"
        
        # Check failure_energy_sum_J == fracture_release_J (within tolerance)
        assert abs(accounting["failure_energy_sum_J"] - accounting["fracture_release_J"]) <= 1e-10, \
            f"failure_energy_sum_J ({accounting['failure_energy_sum_J']}) != fracture_release_J ({accounting['fracture_release_J']})"


def test_accounting_json_with_hull_profile():
    """Verify accounting.json works with hull profile mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        hull_csv = Path(tmpdir) / "hull.csv"
        
        # Create a simple hull profile
        with hull_csv.open("w") as f:
            f.write("x,y\n")
            f.write("0.0,0.0\n")
            f.write("0.1,0.0\n")
            f.write("0.1,0.1\n")
        
        # Run CLI with hull profile
        main([
            "--nx", "5", "--ny", "3", "--steps", "10",
            "--hull-profile", str(hull_csv),
            "--hull-velocity-x", "0.2",
            "--output", str(output_dir),
        ])
        
        # Check accounting.json exists and is valid
        accounting_file = output_dir / "accounting.json"
        assert accounting_file.exists()
        
        with accounting_file.open() as f:
            accounting = json.load(f)
        
        # Basic validation
        assert "total_bonds" in accounting
        assert "fracture_release_J" in accounting
        assert accounting["fracture_release_J"] >= 0
        assert abs(accounting["failure_energy_sum_J"] - accounting["fracture_release_J"]) <= 1e-10

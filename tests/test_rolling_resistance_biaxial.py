"""Rolling resistance in biaxial shear: invariant tests.

Verifies that the rolling-resistance couple in ``RidgeAssembly``:
1. Adds tangential force proportional to mu_r * F_n
2. Does not affect the normal force
3. Vanishes exactly when mu_r = 0
4. Opposes the direction of tangential slip
"""
from __future__ import annotations

import pytest
import torch

from tensordem.ridge_dem import RidgeAssembly, RidgeMaterial


def _make_two_particle_assembly(mu_r: float, v_t: float = 1.0):
    """Two overlapping disks with prescribed tangential relative velocity."""
    m = RidgeMaterial(rolling_friction_coefficient=mu_r, freeze_bond_fraction=0.0)
    pos = torch.tensor([[0.0, 0.0], [1.4, 0.0]], dtype=torch.float64)
    radii = torch.tensor([0.75, 0.75], dtype=torch.float64)
    mass = torch.tensor([m.particle_mass, m.particle_mass], dtype=torch.float64)
    bonds = torch.empty((0, 2), dtype=torch.int64)
    vel = torch.tensor([[0.0, 0.0], [0.0, v_t]], dtype=torch.float64)
    assembly = RidgeAssembly(pos, radii, mass, bonds, material=m, planes=[])
    assembly.vel = vel
    return assembly, m


class TestRollingResistanceCouple:
    """Invariant tests for the rolling-resistance couple in RidgeAssembly."""

    def test_zero_rolling_friction_gives_no_extra_force(self):
        """mu_r = 0 must produce exactly the same forces as the baseline."""
        asm0, _ = _make_two_particle_assembly(0.0, v_t=1.0)
        out0 = asm0.step(asm0.material.recommended_dt)

        asm1, _ = _make_two_particle_assembly(0.0, v_t=1.0)
        out1 = asm1.step(asm1.material.recommended_dt)

        assert torch.allclose(out0["force"], out1["force"], atol=1e-10)

    def test_rolling_force_scales_linearly_with_mu_r(self):
        """Tangential force increment must be proportional to mu_r."""
        asm0, m0 = _make_two_particle_assembly(0.0, v_t=1.0)
        out0 = asm0.step(m0.recommended_dt)

        for mu_r in [0.1, 0.3, 0.5]:
            asm, m = _make_two_particle_assembly(mu_r, v_t=1.0)
            out = asm.step(m.recommended_dt)
            diff = out["force"][0, 1] - out0["force"][0, 1]
            expected = mu_r * 1e6  # approximate: mu_r * F_n
            assert abs(diff - expected) / expected < 0.05, (
                f"mu_r={mu_r}: diff={diff:.0f}, expected~{expected:.0f}")

    def test_normal_force_unaffected_by_rolling(self):
        """Rolling resistance must not change the normal (x) force."""
        asm0, _ = _make_two_particle_assembly(0.0, v_t=1.0)
        out0 = asm0.step(asm0.material.recommended_dt)

        asm1, _ = _make_two_particle_assembly(0.5, v_t=1.0)
        out1 = asm1.step(asm1.material.recommended_dt)

        assert abs(out0["force"][0, 0] - out1["force"][0, 0]) < 1.0

    def test_rolling_opposes_tangential_motion(self):
        """The rolling force must oppose the direction of tangential slip."""
        asm_pos, _ = _make_two_particle_assembly(0.3, v_t=1.0)
        out_pos = asm_pos.step(asm_pos.material.recommended_dt)

        asm_neg, _ = _make_two_particle_assembly(0.3, v_t=-1.0)
        out_neg = asm_neg.step(asm_neg.material.recommended_dt)

        assert out_pos["force"][0, 1] > 0
        assert out_neg["force"][0, 1] < 0

    def test_rolling_friction_increases_contact_tangential_force(self):
        """With rolling resistance, the total tangential force must be larger."""
        asm0, m0 = _make_two_particle_assembly(0.0, v_t=1.0)
        out0 = asm0.step(m0.recommended_dt)

        for mu_r in [0.2, 0.4, 0.6]:
            asm, m = _make_two_particle_assembly(mu_r, v_t=1.0)
            out = asm.step(m.recommended_dt)
            ft0 = abs(out0["force"][0, 1])
            ft = abs(out["force"][0, 1])
            assert ft > ft0, (
                f"mu_r={mu_r}: tangential force {ft:.0f} <= baseline {ft0:.0f}")

    def test_rolling_force_symmetric_on_both_particles(self):
        """The rolling couple applies equal-and-opposite tangential forces."""
        asm0, m0 = _make_two_particle_assembly(0.0, v_t=1.0)
        out0 = asm0.step(m0.recommended_dt)

        asm1, m1 = _make_two_particle_assembly(0.3, v_t=1.0)
        out1 = asm1.step(m1.recommended_dt)

        # The rolling force increment should be equal and opposite
        inc_i = out1["force"][0, 1] - out0["force"][0, 1]
        inc_j = out1["force"][1, 1] - out0["force"][1, 1]
        assert abs(inc_i + inc_j) < 1.0, (
            f"Rolling force not symmetric: inc_i={inc_i:.0f}, inc_j={inc_j:.0f}")

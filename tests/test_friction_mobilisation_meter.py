"""Invariant tests for the unbonded-contact friction mobilisation meter (m21958).

The meter exists to make one claim auditable: the rotationless disk engine's
tangential contact law is the *regularised* law

    F_t = -mu * F_n * tanh(v_t / v_reg)

so the mobilised fraction of the Coulomb capacity is ``tanh(v_t/v_reg)`` -- a
velocity ratio, i.e. a numerical parameter sits inside the constitutive law.
These tests pin (a) the contact-level identity, (b) that the meter is pure
instrumentation (identical forces with the meter on and off) and (c) that the
biaxial driver reports it at the right place in the record.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.biaxial_shear_band import run_biaxial_shear_band
from tensordem.ridge_dem import RidgeAssembly, RidgeMaterial


def _two_disk(velocity, mu=0.55, v_reg=0.01):
    """Two overlapping disks in a vertical contact, relative velocity prescribed."""
    material = RidgeMaterial(friction_coefficient=mu, velocity_regularisation=v_reg,
                             freeze_bond_fraction=0.0)
    radius = material.particle_radius
    overlap = 1.0e-4 * radius
    positions = torch.tensor([[0.0, 0.0], [2.0 * radius - overlap, 0.0]],
                             dtype=torch.float64)
    radii = torch.full((2,), radius, dtype=torch.float64)
    masses = torch.full((2,), float(torch.pi * radius ** 2 * material.ice_density),
                        dtype=torch.float64)
    assembly = RidgeAssembly(positions, radii, masses,
                             torch.zeros((0, 2), dtype=torch.long), material)
    assembly.vel = torch.tensor(velocity, dtype=torch.float64)
    return assembly, material


def test_contact_law_is_regularised_coulomb():
    """F_t / (mu F_n) equals -tanh(v_t/v_reg) exactly (not sign(v_t)).

    The assembly force also carries the global velocity drag, which points along
    the velocity and therefore contaminates the tangential component.  The drag
    is removed by differencing against a ``mu = 0`` twin: the normal component is
    untouched by drag (contacts are loaded along x, drag acts along y) and the
    difference of the two forces is the pure contact friction.
    """
    v_reg = 0.01
    v_t = 0.5 * v_reg                      # deep inside the linear regime
    velocity = [[0.0, 0.5 * v_t], [0.0, -0.5 * v_t]]
    assembly, material = _two_disk(velocity, mu=0.0, v_reg=v_reg)
    force_nofric = assembly.forces()["force"][0].clone()
    assembly, material = _two_disk(velocity, mu=0.55, v_reg=v_reg)
    assembly.enable_friction_meter()
    force_fric = assembly.forces()["force"][0]
    meter = assembly.read_friction_meter()

    assert meter["contacts"] == pytest.approx(1.0)
    expected = math.tanh(v_t / v_reg)
    assert meter["mean_abs_tanh"] == pytest.approx(expected, rel=1e-9)
    assert meter["saturated_fraction"] == 0.0

    # outward normal from j towards i: i sits at the origin, j at +x
    normal = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    tangent = torch.tensor([0.0, -1.0], dtype=torch.float64)
    fn = float(force_nofric @ normal)
    ft = float((force_fric - force_nofric) @ tangent)
    assert fn > 0.0
    assert abs(ft) == pytest.approx(material.friction_coefficient * fn * expected,
                                    rel=1e-9)
    # the mobilised friction is HALF the Coulomb capacity here: a pure Coulomb
    # law would sit at 1.0 * mu * F_n
    assert abs(ft) < 0.9 * material.friction_coefficient * fn


def test_meter_saturates_when_regularisation_is_small():
    """v_t >> v_reg drives the law to Coulomb (|tanh| -> 1)."""
    v_reg = 1.0e-3
    assembly, material = _two_disk([[0.0, 0.05], [0.0, -0.05]], v_reg=v_reg)
    assembly.enable_friction_meter()
    assembly.forces()
    meter = assembly.read_friction_meter()
    assert meter["saturated_fraction"] == pytest.approx(1.0)
    assert meter["mean_abs_tanh"] > 0.99


def test_meter_is_pure_instrumentation():
    """Enabling the meter must not change a single force."""
    assembly_a, _ = _two_disk([[0.0, 1.0e-3], [0.0, -2.0e-3]])
    out_a = assembly_a.forces()
    assembly_b, _ = _two_disk([[0.0, 1.0e-3], [0.0, -2.0e-3]])
    assembly_b.enable_friction_meter()
    out_b = assembly_b.forces()
    assert torch.equal(out_a["force"], out_b["force"])
    assert torch.equal(out_a["virial"], out_b["virial"])


def test_read_meter_without_enable_raises():
    assembly, _ = _two_disk([[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(RuntimeError):
        assembly.read_friction_meter()


def test_biaxial_mobilisation_record_shape_and_no_force_change():
    """The driver reports a per-window mobilisation record; off-path is identical."""
    kwargs = dict(confining_stress=100.0e3, columns=6, rows=12, strain_rate=0.05,
                  strain_target=0.03, consolidation_steps=200, snapshot_strain=1e9,
                  wall_friction=0.0, lateral_control="isochoric",
                  material_overrides={"freeze_bond_fraction": 0.0})
    on = run_biaxial_shear_band(mobilisation_every=100, **kwargs)
    off = run_biaxial_shear_band(**kwargs)
    assert off["mobilisation"] == []
    assert on["mobilisation"], "meter produced no samples"
    for sample in on["mobilisation"]:
        assert 0.0 <= sample["mean_abs_tanh"] <= 1.0
        assert 0.0 <= sample["saturated_fraction"] <= 1.0
        assert sample["contacts"] > 0.0
        assert np.isfinite(sample["fn_weighted_mean_abs_tanh"])
        assert np.isfinite(sample["axial_strain"])
    assert np.all(np.diff([s["axial_strain"] for s in on["mobilisation"]]) >= 0.0)
    # the meter is diagnostic only: the stress path must be bit-identical
    for key in ("sigma_v", "sigma_h", "axial_strain", "volumetric_strain_log"):
        assert np.array_equal(np.asarray(on["stress_path"][key]),
                              np.asarray(off["stress_path"][key]))

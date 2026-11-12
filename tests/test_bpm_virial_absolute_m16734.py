"""m16734: absolute-level audit of the BPM virial stress instrument.

The Brazilian-disc AC5 gate only consumes a *ratio*, so the absolute level of
``virial_stress_tensor`` was never checked.  The independent path is the plane-cut
traction of :mod:`tensordem.granular3d.stress` (sum of the forces crossing a plane
/ area), which shares no algebra with the volume average.

Claims pinned here (numbers from ``scripts/verify_bpm_virial_absolute_m16734.py``,
artifacts in ``artifacts/m16734_virial_absolute/``):

* the per-particle tensor must satisfy ``sum_p V_p sigma_p == sum_c l (x) f`` (the
  historical implementation returned exactly ``2x`` that);
* with the gauge specific volume ``V/N`` the volume average must equal the plane-cut
  traction times the stretch factor ``(1 + strain)`` -- the Cauchy-vs-nominal
  difference, linear in the strain to 1e-9 relative (``strain_sweep``);
* both defects are non-vacuous: the historical factors are ``2 (n-1)/n``
  (measured 1.336 / 1.603 / 1.781 / 1.886 for n = 3/5/9/17) and the cell-volume
  normalisation error is ``1/n``;
* the AC5 ratio is invariant under either choice (it is a ratio of two principal
  stresses of the same tensor).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.bpm_calibration_feasibility import (
    LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR,
    PARTICLE_RADIUS,
    REFERENCE_VECTOR,
    SpecimenGeometry,
    build_axial,
    pair_stress_state,
    region_stress,
    specific_volume,
    virial_stress_tensor,
)
from tensordem.granular3d.stress import contact_outer, plane_stress, stress_tensor

CELL = LATTICE_CELL_VOLUME_PER_PARTICLE_FACTOR * PARTICLE_RADIUS ** 3


def affine_engine(n_layers: int = 5, cross_n: int = 4, strain: float = 2.0e-3, *,
                  kill_fraction: float = 0.0, seed: int = 7):
    """Homogeneous affine state (hold every particle -> statics, no dynamics)."""
    geo = SpecimenGeometry(n_layers=n_layers, cross_n=cross_n, radius=PARTICLE_RADIUS)
    engine, _, _, _ = build_axial(REFERENCE_VECTOR, geo, mass_scaling=100.0,
                                 decay_steps=15)
    base = engine.positions.clone()
    lever = (base[:, 2] - float(base[:, 2].min()))[:, None]
    engine.positions = base + strain * lever * torch.tensor(
        [0.0, 0.0, 1.0], dtype=engine.dtype)
    if kill_fraction > 0.0:
        rng = np.random.default_rng(seed)
        alive = engine.alive.clone()
        dead = rng.choice(int(alive.numel()),
                          size=int(round(kill_fraction * int(alive.numel()))),
                          replace=False)
        alive[torch.from_numpy(np.sort(dead))] = False
        engine.alive = alive
    engine.fixed = torch.ones(engine.positions.shape[0], dtype=torch.bool)
    return engine, geo


def plane_cut(engine, geo):
    state = pair_stress_state(engine)
    center = engine.positions[torch.from_numpy(state["i"])].detach().cpu().numpy()
    z = engine.positions[:, 2].detach().cpu().numpy()
    plane = 0.5 * (float(z.min()) + float(z.max()))
    return plane_stress(state["branch"], state["force_on_i"], center,
                        axis=2, plane=plane, area=float(geo.area))


# ---------------------------------------------------------------- C1: counting
def test_pair_counting_identity_is_exact_and_legacy_mode_doubles_it():
    engine, _ = affine_engine(strain=-1.0e-3, kill_fraction=0.25)
    state = pair_stress_state(engine)
    reference = contact_outer(state["branch"], state["force_on_i"]).sum(axis=0)
    unit_volume = virial_stress_tensor(engine, volume_per_particle=1.0)
    legacy = virial_stress_tensor(engine, volume_per_particle=1.0, scatter="legacy")
    assert np.allclose(unit_volume.sum(dim=0).numpy(), reference, atol=1e-12)
    assert np.allclose(legacy.sum(dim=0).numpy(), 2.0 * reference, atol=1e-12)


def test_legacy_double_count_matches_its_analytic_factor():
    for n_layers in (3, 5, 9):
        engine, geo = affine_engine(n_layers=n_layers, strain=+2.0e-3)
        plane = float(plane_cut(engine, geo)[2, 2])
        legacy = float(virial_stress_tensor(
            engine, volume_per_particle=CELL, scatter="legacy").mean(dim=0)[2, 2])
        predicted = 2.0 * (n_layers - 1) / n_layers * (1.0 + 2.0e-3)
        assert legacy / plane == pytest.approx(predicted, rel=1.0e-3)


# ------------------------------------------------------- C2: normalisation
def test_gauge_specific_volume_reproduces_the_plane_cut():
    """The volume average equals the plane-cut traction up to the stretch factor."""
    for strain in (+2.0e-3, -2.0e-3, +1.0e-4):
        engine, geo = affine_engine(n_layers=5, strain=strain)
        state = pair_stress_state(engine)
        V_total = float(geo.area) * float(geo.thickness)
        V_p = specific_volume(V_total, state["n_particles"])
        plane = float(plane_cut(engine, geo)[2, 2])
        average = float(virial_stress_tensor(engine, volume_per_particle=V_p)
                        .mean(dim=0)[2, 2])
        exact = float(stress_tensor(state["branch"], state["force_on_i"],
                                    V_total)[2, 2])
        assert average == pytest.approx(exact, rel=1e-12)
        # Cauchy (current configuration) vs nominal (reference area): (1 + strain)
        assert average / plane == pytest.approx(1.0 + strain, rel=1.0e-9)


def test_cell_volume_normalisation_error_is_one_over_n_layers():
    for n_layers in (3, 5, 9):
        engine, geo = affine_engine(n_layers=n_layers, strain=+2.0e-3)
        plane = float(plane_cut(engine, geo)[2, 2])
        cell = float(virial_stress_tensor(engine, volume_per_particle=CELL)
                     .mean(dim=0)[2, 2])
        # V_p(true)/V_cell = (n-1)/n and the volume average carries (1 + strain)
        predicted = abs(1.0 - (n_layers - 1) / n_layers * (1.0 + 2.0e-3))
        assert abs(cell / plane - 1.0) == pytest.approx(predicted, rel=1.0e-3)


# ------------------------------------------------------ non-vacuity (mutations)
@pytest.mark.parametrize("kind,expected_rel_change", [
    ("force_sign_flipped", 2.0),
    ("pair_counted_twice", 1.0),
    ("dead_bonds_kept", 0.20),
    ("contacts_not_gated", 0.30),
    ("cell_volume_normalisation", 0.20),
])
def test_classic_bugs_are_detectable(kind, expected_rel_change):
    engine, geo = affine_engine(n_layers=5, strain=-2.0e-3, kill_fraction=0.25)
    state = pair_stress_state(engine)
    V_total = float(geo.area) * float(geo.thickness)
    branch, force = state["branch"], state["force_on_i"]
    bonded, contact = state["bonded"], state["contact"]
    reference = float(contact_outer(branch, force).sum(axis=0)[2, 2] / V_total)

    if kind == "force_sign_flipped":
        value = float(contact_outer(branch, -force).sum(axis=0)[2, 2] / V_total)
    elif kind == "pair_counted_twice":
        value = 2.0 * reference
    elif kind == "dead_bonds_kept":
        pos = engine.positions
        i, j = engine.pairs.T
        d = pos[j] - pos[i]
        length = torch.linalg.vector_norm(d, dim=1).clamp_min(1e-30)
        unit = (d / length[:, None]).detach().cpu().numpy()
        f_bond = float(engine.config.bond_stiffness) * (
            length - engine.rest_lengths).detach().cpu().numpy()
        overlap = 2.0 * float(engine.config.radius) - length.detach().cpu().numpy()
        f_contact = np.where(contact, float(engine.config.contact_stiffness) * overlap, 0.0)
        value = float(contact_outer(branch, (f_bond - f_contact)[:, None] * unit)
                      .sum(axis=0)[2, 2] / V_total)
    elif kind == "contacts_not_gated":
        pos = engine.positions
        i, j = engine.pairs.T
        d = pos[j] - pos[i]
        length = torch.linalg.vector_norm(d, dim=1).clamp_min(1e-30)
        unit = (d / length[:, None]).detach().cpu().numpy()
        f_bond = float(engine.config.bond_stiffness) * (
            length - engine.rest_lengths).detach().cpu().numpy()
        f_bond = np.where(bonded, f_bond, 0.0)
        overlap = 2.0 * float(engine.config.radius) - length.detach().cpu().numpy()
        f_contact = float(engine.config.contact_stiffness) * np.clip(overlap, 0.0, None)
        value = float(contact_outer(branch, (f_bond - f_contact)[:, None] * unit)
                      .sum(axis=0)[2, 2] / V_total)
    else:                                    # cell_volume_normalisation
        value = float(contact_outer(branch, force).sum(axis=0)[2, 2]
                      / (state["n_particles"] * CELL))
    change = abs(value - reference) / abs(reference)
    assert change > 0.5 * expected_rel_change     # detected, with margin
    assert change < 2.0 * expected_rel_change


def test_transpose_convention_is_invisible_for_central_forces():
    """Honest negative result: ``l (x) f`` and ``f (x) l`` cannot be told apart here.

    Both force branches of the instrument are central, so the transpose mutation
    is *not* detectable -- the instrument therefore cannot measure the friction
    asymmetry ``sigma_ab - sigma_ba`` at all; that limitation is in the docstring.
    """
    engine, geo = affine_engine(n_layers=5, strain=-2.0e-3, kill_fraction=0.25)
    state = pair_stress_state(engine)
    V_total = float(geo.area) * float(geo.thickness)
    left = contact_outer(state["branch"], state["force_on_i"]).sum(axis=0) / V_total
    right = np.einsum("ma,mb->mab", state["force_on_i"], state["branch"]).sum(axis=0) / V_total
    assert np.allclose(left, right, atol=1e-6 * abs(left).max())


# ------------------------------------------------- AC5 consumer is unaffected
def test_ac5_ratio_is_invariant_under_the_correction():
    engine, _ = affine_engine(n_layers=5, strain=-2.0e-3, kill_fraction=0.25)
    centre = (0.0, 0.0, float(engine.positions[:, 2].mean()))
    legacy = region_stress(engine, center=centre, radius=0.03,
                           scatter="legacy")["principal_in_plane"]
    fixed = region_stress(engine, center=centre, radius=0.03,
                          scatter="half")["principal_in_plane"]
    ratio_legacy = max(legacy) / min(legacy)
    ratio_fixed = max(fixed) / min(fixed)
    assert ratio_fixed == pytest.approx(ratio_legacy, rel=1e-12)
    assert abs(fixed[0]) == pytest.approx(0.5 * abs(legacy[0]), rel=1e-12)


# --------------------------------------------------------------- validation
def test_option_validation():
    engine, _ = affine_engine(n_layers=3)
    with pytest.raises(ValueError):
        virial_stress_tensor(engine, scatter="quadruple")
    with pytest.raises(ValueError):
        virial_stress_tensor(engine, volume_per_particle=-1.0)
    with pytest.raises(ValueError):
        virial_stress_tensor(engine, volume_per_particle=float("nan"))
    with pytest.raises(ValueError):
        specific_volume(0.0, 10)
    with pytest.raises(ValueError):
        specific_volume(1.0, 0)
    assert specific_volume(2.0, 4) == pytest.approx(0.5)
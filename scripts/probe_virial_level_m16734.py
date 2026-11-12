"""Probe: does the BPM virial stress instrument report the *absolute* stress level?

Independent path: the nominal stress from the driven-platen reaction (P/A, the
path every strength number of this module already uses) versus the interior
Love-Weber average from ``virial_stress_tensor`` / ``region_stress``.

No changes to the library here -- this only measures.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from tensordem.bpm_calibration_feasibility import (
    PARTICLE_RADIUS,
    REFERENCE_VECTOR,
    DiscGeometry,
    SpecimenGeometry,
    build_axial,
    build_disc,
    default_protocol,
    virial_stress_tensor,
)
from tensordem.bpm_identifiability import LoadProtocol
from tensordem.granular3d.stress import contact_outer


def lattice_specific_volume(engine, volume_total: float) -> float:
    return float(volume_total) / float(engine.positions.shape[0])


def total_virial_sum(engine):
    """Independent algebraic reference: sum_c l (x) f over the same pair list."""
    pos = engine.positions
    i, j = engine.pairs.T
    direction = pos[j] - pos[i]
    length = torch.linalg.vector_norm(direction, dim=1).clamp_min(
        torch.finfo(pos.dtype).eps)
    unit = direction / length[:, None]
    extension = length - engine.rest_lengths
    f_bond = float(engine.config.bond_stiffness) * extension
    f_bond = torch.where(engine.alive, f_bond, torch.zeros_like(f_bond))
    overlap = 2.0 * float(engine.config.radius) - length
    f_contact = torch.where((~engine.alive) & (overlap > 0.0),
                            float(engine.config.contact_stiffness) * overlap,
                            torch.zeros_like(overlap))
    force_on_i = (f_bond - f_contact)[:, None] * unit
    branch = direction.detach().cpu().numpy()
    force = force_on_i.detach().cpu().numpy()
    return contact_outer(branch, force).sum(axis=0), float(
        torch.linalg.vector_norm(force_on_i, dim=1).sum() * 0.0)


def report_axial(n_layers: int, cross_n: int, strain: float, steps: int):
    geo = SpecimenGeometry(n_layers=n_layers, cross_n=cross_n, radius=PARTICLE_RADIUS)
    engine, top, bottom, meta = build_axial(REFERENCE_VECTOR, geo,
                                           mass_scaling=100.0, decay_steps=15)
    # settle, then an affine uniaxial compression of the whole gauge: this makes
    # the specimen homogeneous so the interior average and the platen reaction
    # refer to the same stress state.
    for _ in range(int(steps)):
        engine.step()
    base = engine.positions.clone()
    lever = (base[:, 2] - float(base[:, 2].min()))[:, None]
    if strain > 0:
        engine.positions = base - strain * lever * torch.tensor(
            [0.0, 0.0, 1.0], dtype=engine.dtype)
    for _ in range(int(steps)):
        engine.step()

    V_p_legacy = 8.0 * PARTICLE_RADIUS ** 3
    n = int(engine.positions.shape[0])
    v_spec = geo.volume
    V_p_true = lattice_specific_volume(engine, v_spec)

    stress = virial_stress_tensor(engine)
    mean_tensor = stress.mean(dim=0).detach().cpu().numpy()
    sum_tensor = (stress.sum(dim=0) * V_p_legacy).detach().cpu().numpy()
    ref_sum, _ = total_virial_sum(engine)

    # independent boundary path: reaction on the driven top layer
    f, _ = engine.forces(update_fracture=False, update_history=False)
    reaction = -float(f[top, 2].sum())
    sigma_app = reaction / meta["area_m2"]

    return {
        "n_layers": n_layers, "cross_n": cross_n, "n_particles": n,
        "strain": strain,
        "V_p_legacy": V_p_legacy, "V_p_specific": V_p_true,
        "V_p_ratio": V_p_legacy / V_p_true,
        "sigma_zz_legacy_mean": float(mean_tensor[2, 2]),
        "sigma_zz_region_corrected": float(mean_tensor[2, 2] * 0.5 * V_p_legacy / V_p_true),
        "sigma_applied": sigma_app,
        "ratio_legacy_over_applied": float(mean_tensor[2, 2]) / sigma_app,
        "ratio_corrected_over_applied": float(
            mean_tensor[2, 2] * 0.5 * V_p_legacy / V_p_true) / sigma_app,
        "sum_V_sigma_over_ref_sum": float(np.linalg.norm(sum_tensor) / np.linalg.norm(ref_sum)),
        "ref_sum_zz": float(ref_sum[2, 2]),
    }


def main() -> None:
    out = {}
    t0 = time.time()
    for n_layers in (5, 9):
        out[f"axial_{n_layers}"] = report_axial(n_layers, 4, 2.0e-3, 300)
        print(json.dumps(out[f"axial_{n_layers}"], indent=2))
    # disc: what is the specific volume of the cylinder_xz lattice?
    engine, drive, support, meta = build_disc(REFERENCE_VECTOR,
                                             DiscGeometry(n_lattice=7, n_layers=4),
                                             mass_scaling=100.0, decay_steps=15)
    n = int(engine.positions.shape[0])
    V_cyl = math.pi * (meta["diameter_m"] ** 2) / 4.0 * meta["thickness_m"]
    out["disc"] = {
        "n_particles": n,
        "V_p_legacy": 8.0 * PARTICLE_RADIUS ** 3,
        "V_p_specific_cylinder": V_cyl / n,
        "V_p_ratio": (8.0 * PARTICLE_RADIUS ** 3) / (V_cyl / n),
    }
    print(json.dumps(out["disc"], indent=2))
    out["wall_s"] = time.time() - t0
    pathlib.Path("runs/m16734_virial_probe.json").parent.mkdir(
        parents=True, exist_ok=True)
    pathlib.Path("runs/m16734_virial_probe.json").write_text(json.dumps(out, indent=2))
    print("wall", out["wall_s"])


if __name__ == "__main__":
    main()
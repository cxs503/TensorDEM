"""Mode-II (direct-shear) specimen protocol for the 3-D bonded-sphere lattice.

Why this module exists
----------------------
``docs/dem3d_material_calibration_roadmap.md`` lists a *shear specimen protocol*
as an open fracture-validation item, and ``docs/dem3d_bond_fracture_energy.md``
states explicitly that the shear criterion is "a strain criterion without a
mode-II energy release rate".  ``tensordem.dem3d`` implements that criterion as

    shear invariant   J2 = sqrt(2/3 * dev(E) : dev(E))        (per particle)
    bond shear strain     = 0.5 * (J2[i] + J2[j])
    failure               when bond shear strain > shear_breaking_strain

with ``E`` the Green-Lagrange strain of a least-squares local deformation
gradient.  Nothing in the repository mapped the parameter
``shear_breaking_strain`` onto a measurable macroscopic quantity, and nothing
drove a specimen in shear to observe the mode-II sequence.  This module closes
that gap with:

* closed forms for the invariant of a prescribed simple shear and of a
  prescribed uniaxial strain, plus the inverted threshold mapping
  ``gamma_c(shear_breaking_strain)``;
* ``affine_shear_probe`` / ``shear_failure_onset`` which verify those closed
  forms against the solver's own criterion code path to machine precision;
* ``run_direct_shear`` which drives a block specimen with a rigid lateral
  platen (kinematic top layer, fixed bottom layer) and records the
  engineering shear response, the first mode-II bond event, the local strain
  invariant at that event and the energy bookkeeping.

The module does not change the solver.  All measurements come from public
state (``positions``, ``velocities``, ``alive``, ``failure_mode``,
``failure_energy_J``) plus the solver's own strain measure
``IceDEM3D._objective_shear_strain``, so the audit exercises the same code path
that decides failure.

Scope limits (do not read more into the numbers)
------------------------------------------------
* ``run_direct_shear`` measures an *apparent* modulus of a small lattice block
  with a clamped bottom, a rigidly driven top and free lateral faces.  There is
  no closed-form elastic solution for that boundary-value problem on a
  26-neighbour lattice with a few spacings per dimension, so the apparent
  modulus is reported against the material constant ``C44`` as a specimen
  characterisation, not as a validated material property.
* The affine probes are linear-kinematics statements about the failure
  criterion only: no dynamics, no fracture propagation, no ice-material claim.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "affine_shear_invariant",
    "affine_shear_probe",
    "apparent_shear_modulus",
    "apparent_shear_modulus_from_history",
    "bond_shear_invariant",
    "critical_engineering_shear_strain",
    "direct_shear_specimen_area",
    "direct_shear_specimen_height",
    "lattice_spacing",
    "run_direct_shear",
    "shear_modulus_closed_form",
    "shear_failure_onset",
    "uniaxial_shear_criterion_strain",
    "uniaxial_failure_onset",
]

_PLANES = {
    # name: (displacement axis, gradient axis) for u_a = gamma * x_b
    "xy": (0, 1),
    "xz": (0, 2),
    "yx": (1, 0),
    "yz": (1, 2),
    "zx": (2, 0),
    "zy": (2, 1),
}


def lattice_spacing(config: DEM3DConfig) -> float:
    """Lattice spacing ``a = 2 * radius`` used by the simple-cubic builder."""
    return 2.0 * config.radius


def shear_modulus_closed_form(config: DEM3DConfig) -> float:
    """Affine shear constant of the 26-neighbour lattice, ``C44 = 7 k / (3 a)``.

    Same closed form as ``docs/dem3d_lattice_elastic_constants.md``; valid for
    the central-force bond law of ``tensordem.dem3d`` under affine strain.
    """
    return 7.0 * config.bond_stiffness / (3.0 * lattice_spacing(config))


def affine_shear_invariant(gamma: float) -> float:
    """Exact shear invariant of simple shear ``u_a = gamma * x_b``.

    With ``F = I + gamma e_a (x) e_b`` the Green-Lagrange tensor has
    ``E_ab = gamma/2`` and ``E_bb = gamma^2/2``, hence

        sqrt(2/3 dev(E):dev(E)) = sqrt(gamma^2/3 + gamma^4/9).

    The leading term is ``gamma/sqrt(3)``; the quadratic term matters only at
    the percent level for ``gamma`` of order 0.05.
    """
    return math.sqrt(gamma * gamma / 3.0 + gamma ** 4 / 9.0)


def critical_engineering_shear_strain(shear_breaking_strain: float) -> float:
    """Engineering shear strain at which the shear criterion first fires.

    Solving ``s^2 = gamma^2/3 + gamma^4/9`` for ``gamma^2`` gives

        gamma_c = sqrt(1.5 * (sqrt(1 + 4 s^2) - 1)),

    which reduces to ``sqrt(3) * s`` to first order (0.04 % lower at s = 0.03).
    """
    s = float(shear_breaking_strain)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("shear_breaking_strain must be finite and positive")
    return math.sqrt(1.5 * (math.sqrt(1.0 + 4.0 * s * s) - 1.0))


def uniaxial_shear_criterion_strain(shear_breaking_strain: float) -> float:
    """Uniaxial strain at which the deviatoric (shear) criterion fires.

    For ``F = diag(1 + eps, 1, 1)`` the deviatoric invariant is
    ``(2/3) * (eps + eps^2/2)``, so ``s = (2/3)(eps + eps^2/2)`` has the
    positive root ``eps = -1 + sqrt(1 + 3 s)``.  This is the quantitative
    statement that the criterion is a *deviatoric strain* criterion and not a
    mode-II-specific one: it also fires in uniaxial compression/tension well
    before the axial failure strain ``3 s``.
    """
    s = float(shear_breaking_strain)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("shear_breaking_strain must be finite and positive")
    return math.sqrt(1.0 + 3.0 * s) - 1.0


def bond_shear_invariant(sim: IceDEM3D) -> torch.Tensor:
    """Per-bond shear invariant exactly as ``IceDEM3D._commit_fracture`` uses it."""
    particle = sim._objective_shear_strain()
    bi, bj = sim.bond_indices.T
    return 0.5 * (particle[bi] + particle[bj])


def _affine_shear_positions(sim: IceDEM3D, gamma: float, plane: str) -> torch.Tensor:
    if plane not in _PLANES:
        raise ValueError(f"plane must be one of {sorted(_PLANES)}")
    axis, gradient = _PLANES[plane]
    positions = sim.initial_positions.clone()
    positions[:, axis] += float(gamma) * sim.initial_positions[:, gradient]
    return positions


def _affine_uniaxial_positions(sim: IceDEM3D, strain: float) -> torch.Tensor:
    positions = sim.initial_positions.clone()
    positions[:, 0] *= 1.0 + float(strain)
    return positions


def _rigid_rotation() -> torch.Tensor:
    a, b, c = 0.31, -0.22, 0.17
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)],
                       [0.0, math.sin(a), math.cos(a)]], dtype=torch.float64)
    ry = torch.tensor([[math.cos(b), 0.0, math.sin(b)], [0.0, 1.0, 0.0],
                       [-math.sin(b), 0.0, math.cos(b)]], dtype=torch.float64)
    rz = torch.tensor([[math.cos(c), -math.sin(c), 0.0], [math.sin(c), math.cos(c), 0.0],
                       [0.0, 0.0, 1.0]], dtype=torch.float64)
    return rz @ ry @ rx


def _affine_config(**overrides: Any) -> DEM3DConfig:
    values: dict[str, Any] = {
        "nx": 3, "ny": 3, "nz": 3, "drag": 0.0, "fix_x_edges": False,
        "tool_gap": 20.0, "breaking_strain": 1.0e3, "shear_breaking_strain": 0.03,
    }
    values.update(overrides)
    return DEM3DConfig(**values)


def affine_shear_probe(
    config: DEM3DConfig | None = None,
    gammas: Sequence[float] = (1.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, -2.0e-2),
    plane: str = "xz",
) -> dict[str, Any]:
    """Compare the solver's shear invariant with the closed form under affine shear."""
    config = config or _affine_config()
    sim = IceDEM3D(config)
    rows = []
    for gamma in gammas:
        sim.alive[:] = True
        sim.positions = _affine_shear_positions(sim, gamma, plane)
        invariant = sim._objective_shear_strain()
        expected = affine_shear_invariant(float(gamma))
        measured_min = float(invariant.min())
        measured_max = float(invariant.max())
        scale = max(expected, 1.0e-30)
        rows.append({
            "gamma": float(gamma),
            "measured_min": measured_min,
            "measured_max": measured_max,
            "closed_form": expected,
            "max_relative_error": max(abs(measured_max - expected), abs(measured_min - expected)) / scale,
            "particle_spread": measured_max - measured_min,
        })
    # Isotropy of the {100} shear planes: the invariant of u_a = gamma x_b must not
    # depend on which cubic plane is sheared.
    gamma = 0.02
    isotropy = {}
    for name in ("xy", "yz", "xz"):
        sim.alive[:] = True
        sim.positions = _affine_shear_positions(sim, gamma, name)
        isotropy[name] = float(sim._objective_shear_strain().max())
    isotropic_spread = max(isotropy.values()) - min(isotropy.values())
    # Rigid rotation must not register as shear strain.
    sim.alive[:] = True
    sim.positions = sim.initial_positions @ _rigid_rotation().T + torch.tensor(
        [1.7, -2.1, 3.4], dtype=torch.float64)
    rotation_invariant = float(sim._objective_shear_strain().max())
    max_error = max(row["max_relative_error"] for row in rows)
    return {
        "plane": plane,
        "rows": rows,
        "max_relative_error": max_error,
        "isotropy_at_gamma_0_02": isotropy,
        "isotropy_absolute_spread": isotropic_spread,
        "rotation_invariant": rotation_invariant,
        "closed_form_invariant_at_0_02": affine_shear_invariant(gamma),
    }


def shear_failure_onset(
    config: DEM3DConfig | None = None,
    *,
    hi: float = 0.3,
    iterations: int = 80,
    plane: str = "xz",
) -> dict[str, Any]:
    """Bisect the affine shear strain at which the first bond fails."""
    config = config or _affine_config()
    sim = IceDEM3D(config)
    threshold = float(config.shear_breaking_strain)

    def breaks(gamma: float) -> bool:
        sim.alive[:] = True
        sim.failure_mode.zero_()
        sim.positions = _affine_shear_positions(sim, gamma, plane)
        sim.forces(update_fracture=True)
        return sim.broken_bonds > 0

    if breaks(0.0):
        raise ValueError("specimen already broken at zero strain")
    if not breaks(hi):
        raise ValueError(f"no shear failure below gamma = {hi}")
    lo = 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if breaks(mid):
            hi = mid
        else:
            lo = mid
    onset = hi
    # Sample the criterion at the onset to confirm the recorded mode and the
    # local invariant that triggered it.
    sim.alive[:] = True
    sim.failure_mode.zero_()
    sim.positions = _affine_shear_positions(sim, onset, plane)
    sim.forces(update_fracture=True)
    modes = sorted({int(v) for v in sim.failure_mode[~sim.alive].tolist()})
    local = bond_shear_invariant(sim)
    expected = critical_engineering_shear_strain(threshold)
    return {
        "shear_breaking_strain": threshold,
        "onset_gamma": onset,
        "closed_form_gamma": expected,
        "first_order_sqrt3_s": math.sqrt(3.0) * threshold,
        "relative_error": abs(onset - expected) / expected,
        "broken_bonds": int(sim.broken_bonds),
        "failure_mode_codes": modes,
        "local_bond_invariant_at_onset": float(local.max()),
    }


def uniaxial_failure_onset(
    config: DEM3DConfig | None = None, *, hi: float = 0.3, iterations: int = 80
) -> dict[str, Any]:
    """Bisect the uniaxial strain at which the deviatoric criterion fires."""
    config = config or _affine_config()
    sim = IceDEM3D(config)
    threshold = float(config.shear_breaking_strain)

    def breaks(strain: float) -> bool:
        sim.alive[:] = True
        sim.failure_mode.zero_()
        sim.positions = _affine_uniaxial_positions(sim, strain)
        sim.forces(update_fracture=True)
        return sim.broken_bonds > 0

    if breaks(0.0):
        raise ValueError("specimen already broken at zero strain")
    if not breaks(hi):
        raise ValueError(f"no failure below uniaxial strain {hi}")
    lo = 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if breaks(mid):
            hi = mid
        else:
            lo = mid
    onset = hi
    expected = uniaxial_shear_criterion_strain(threshold)
    sim.alive[:] = True
    sim.failure_mode.zero_()
    sim.positions = _affine_uniaxial_positions(sim, onset)
    sim.forces(update_fracture=True)
    return {
        "shear_breaking_strain": threshold,
        "onset_strain": onset,
        "closed_form_strain": expected,
        "relative_error": abs(onset - expected) / expected,
        "broken_bonds": int(sim.broken_bonds),
        "failure_mode_codes": sorted({int(v) for v in sim.failure_mode[~sim.alive].tolist()}),
        "axial_failure_strain_3s": 3.0 * threshold,
    }


def direct_shear_specimen_height(config: DEM3DConfig) -> float:
    """Distance between the driven top layer and the fixed bottom layer."""
    return (config.nz - 1) * lattice_spacing(config)


def direct_shear_specimen_area(config: DEM3DConfig) -> float:
    """Nominal sheared cross-section ``nx * a * ny * a`` (population convention)."""
    a = lattice_spacing(config)
    return config.nx * a * config.ny * a


def _specimen_masks(sim: IceDEM3D) -> tuple[torch.Tensor, torch.Tensor]:
    z = sim.initial_positions[:, 2]
    top = torch.isclose(z, z.clone().max())
    bottom = torch.isclose(z, z.clone().min())
    return top, bottom


def run_direct_shear(
    config: DEM3DConfig | None = None,
    *,
    velocity: float = 0.1,
    gamma_max: float = 0.04,
    sample_every: int | None = None,
    record_histories: bool = True,
) -> dict[str, Any]:
    """Drive a block specimen in shear and record the engineering response.

    The bottom particle layer is a hard fixed support (``fix_bottom``), the top
    layer is displaced laterally at constant speed (a rigid kinematic platen) and
    the lateral faces are free (``fix_x_edges = False``).  The shear traction is
    the reaction on the fixed layer divided by the nominal cross-section; the
    engineering strain is the platen displacement divided by the layer spacing.
    """
    base = config or DEM3DConfig()
    config = replace(base, fix_bottom=True, fix_x_edges=False)
    if velocity <= 0.0:
        raise ValueError("velocity must be positive")
    sim = IceDEM3D(config)
    a = lattice_spacing(config)
    height = direct_shear_specimen_height(config)
    area = direct_shear_specimen_area(config)
    top, bottom = _specimen_masks(sim)
    xhat = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    dt = sim.dt
    steps = max(int(round(gamma_max * height / (velocity * dt))), 1)
    if sample_every is None:
        sample_every = max(1, steps // 60)
    history: list[dict[str, float]] = []
    first_failure: dict[str, Any] | None = None
    previous_broken = 0
    for k in range(steps):
        displacement = velocity * (k + 1) * dt
        sim.positions[top] = sim.initial_positions[top] + displacement * xhat
        sim.velocities[top] = velocity * xhat
        sim.step()
        broken = sim.broken_bonds
        if first_failure is None and broken > previous_broken:
            first_failure = {
                "gamma": float(displacement / height),
                "time_s": float(sim.time),
                "broken_bonds": int(broken),
                "max_bond_invariant": float(bond_shear_invariant(sim).max()),
                "failure_mode_codes": sorted(
                    {int(v) for v in sim.failure_mode[~sim.alive].tolist()}),
            }
        previous_broken = broken
        if (k + 1) % sample_every == 0 or k == steps - 1:
            force, _ = sim.forces(update_fracture=False)
            reaction = abs(float(force[bottom].sum(dim=0)[0]))
            free = ~top
            kinetic = 0.5 * config.mass * float(sim.velocities[free].square().sum())
            distance = torch.linalg.vector_norm(sim.positions[sim.pairs[:, 1]]
                                               - sim.positions[sim.pairs[:, 0]], dim=1)
            extension = distance - sim.rest_lengths
            elastic = float((0.5 * config.bond_stiffness * extension.square() * sim.alive).sum())
            bond_invariant = bond_shear_invariant(sim)
            history.append({
                "time_s": float(sim.time),
                "gamma": float(displacement / height),
                "tau_Pa": reaction / area,
                "broken_bonds": int(broken),
                "kinetic_energy_J": kinetic,
                "bond_elastic_energy_J": elastic,
                "released_bond_energy_J": float(sim.failure_energy_J.sum()),
                "max_bond_invariant": float(bond_invariant.max()),
                "max_alive_bond_invariant": float(bond_invariant[sim.alive].max())
                if bool(sim.alive.any()) else 0.0,
            })
    modes = sim.failure_mode[~sim.alive]
    code_counts: dict[int, int] = {}
    for code in modes.tolist():
        code_counts[int(code)] = code_counts.get(int(code), 0) + 1
    mode_energies = {}
    for code in sorted(code_counts):
        mask = (~sim.alive) & (sim.failure_mode == code)
        mode_energies[str(code)] = {
            "count": int(mask.sum()),
            "mean_recorded_release_J": float(sim.failure_energy_J[mask].mean()),
            "max_recorded_release_J": float(sim.failure_energy_J[mask].max()),
        }
    result: dict[str, Any] = {
        "config": {
            "nx": config.nx, "ny": config.ny, "nz": config.nz,
            "radius_m": config.radius, "bond_stiffness_N_per_m": config.bond_stiffness,
            "breaking_strain": config.breaking_strain,
            "shear_breaking_strain": config.shear_breaking_strain,
            "drag_Ns_per_m": config.drag,
            "contact_damping_Ns_per_m": config.contact_damping,
        },
        "protocol": {
            "velocity_m_per_s": velocity,
            "gamma_max": gamma_max,
            "dt_s": dt,
            "steps": steps,
            "sample_every": sample_every,
            "particle_count": int(len(sim.positions)),
            "specimen_height_m": height,
            "specimen_area_m2": area,
            "lattice_spacing_m": a,
        },
        "C44_Pa": shear_modulus_closed_form(config),
        "first_failure": first_failure,
        "failure_mode_counts": code_counts,
        "failure_mode_energy": mode_energies,
        "final": {
            "gamma": history[-1]["gamma"],
            "tau_Pa": history[-1]["tau_Pa"],
            "broken_bonds": history[-1]["broken_bonds"],
            "released_bond_energy_J": history[-1]["released_bond_energy_J"],
            "bond_elastic_energy_J": history[-1]["bond_elastic_energy_J"],
        },
    }
    if record_histories:
        result["history"] = history
    return result


def apparent_shear_modulus_from_history(
    history: Sequence[dict[str, float]], g_lo: float, g_hi: float
) -> dict[str, float]:
    """Least-squares slope of ``tau`` against ``gamma`` inside ``[g_lo, g_hi]``.

    A free intercept is kept because the driven specimen carries the transient of
    the load path from the platen into the block; the slope, not the ratio
    ``tau/gamma``, is the elastic response once the load has been transmitted.
    """
    rows = [row for row in history
            if g_lo <= row["gamma"] <= g_hi and row["broken_bonds"] == 0]
    if len(rows) < 3:
        raise ValueError("window contains fewer than three intact samples")
    gammas = [row["gamma"] for row in rows]
    taus = [row["tau_Pa"] for row in rows]
    n = float(len(rows))
    mean_g = sum(gammas) / n
    mean_t = sum(taus) / n
    sxx = sum((g - mean_g) ** 2 for g in gammas)
    sxy = sum((g - mean_g) * (t - mean_t) for g, t in zip(gammas, taus))
    slope = sxy / sxx
    intercept = mean_t - slope * mean_g
    residual = max(abs(t - (slope * g + intercept)) for g, t in zip(gammas, taus))
    return {
        "apparent_shear_modulus_Pa": slope,
        "intercept_Pa": intercept,
        "sample_count": len(rows),
        "max_abs_residual_Pa": residual,
        "gamma_lo": gammas[0],
        "gamma_hi": gammas[-1],
        "relative_residual": residual / max(abs(slope * gammas[-1]), 1.0e-30),
    }


def apparent_shear_modulus(
    config: DEM3DConfig | None = None,
    *,
    velocity: float = 0.1,
    gamma_max: float = 0.04,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run a direct-shear ramp and return the apparent modulus of the specimen."""
    result = run_direct_shear(config, velocity=velocity, gamma_max=gamma_max,
                              record_histories=True)
    history = result["history"]
    if window is None:
        # Late elastic window: after the load path has been established and
        # before the first event.
        upper = 0.95 * (result["first_failure"]["gamma"]
                        if result["first_failure"] else history[-1]["gamma"])
        window = (0.6 * upper, upper)
    fit = apparent_shear_modulus_from_history(history, *window)
    c44 = result["C44_Pa"]
    return {
        "window": list(window),
        "fit": fit,
        "C44_Pa": c44,
        "ratio_to_C44": fit["apparent_shear_modulus_Pa"] / c44,
        "first_failure": result["first_failure"],
        "history": history,
        "protocol": result["protocol"],
        "failure_mode_counts": result["failure_mode_counts"],
    }
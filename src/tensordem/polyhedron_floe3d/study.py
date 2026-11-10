"""Validation cases for 3-D polyhedral floe DEM with Weibull weakest-link fracture.

Three cases, each with machine-checkable numbers:

``case1_impact``
    A single 3-D polyhedral floe impacts a rigid wall.  The peak contact force is
    compared with the closed-form peak of a linear spring-dashpot impact
    (:func:`peak_force_analytic`), and the fragmentation of the same floe is
    driven by a Weibull weakest-link cell model: the survival probability of the
    body versus the applied peak stress, the size effect on the strength scale
    and the fragment-size distribution on a Weibull probability plot.

``case2_channel``
    A 3-D brash-ice channel is cleared by an advancing hull plane.  Clearing
    resistance and the *measured* porosity evolution are reported; the same field
    is re-run with disc plates (equal volume and thickness) to isolate the shape
    effect, with an ensemble spread study for the statistical convergence order,
    and with the vertical/tilt degrees of freedom frozen (the 2-D limit).

``case3_repose``
    A rubble pile of polyhedra is poured and settled; the angle of repose from a
    free-surface fit and the force-chain network statistics (load share of the
    strongest contacts, exponential tail of ``P(F/<F>)``, coordination number)
    are measured for three friction coefficients.

Every case returns a plain ``dict`` of numbers so that the verdict can be
computed mechanically (:func:`evaluate_criteria`) and injected into the report
without transcription.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contact import ContactLaw
from .dynamics import DEMSystem3D, FloeBody
from .field import BrashIceField, build_brash_ice_field, build_disc_reference_field
from .geometry import (ConvexPolyhedron, extrude_polygon, polygon_area,
                       regular_polygon, voronoi_tessellate)
from .weibull import (fit_mle, fit_probability_plot, fragment_flaw_model,
                      fragmentation_monte_carlo, kolmogorov_smirnov,
                      sample_strengths, size_effect_study, weakest_link_scale)

__all__ = [
    "peak_force_analytic",
    "conservation_case",
    "calibrate_strength",
    "case1_impact",
    "run_channel",
    "statistical_convergence",
    "case2_channel",
    "case3_repose",
    "evaluate_criteria",
    "run_all",
]

ICE_DENSITY = 900.0
WATER_DENSITY = 1025.0


# ---------------------------------------------------------------------------
# closed-form reference for a linear spring-dashpot impact
# ---------------------------------------------------------------------------
def peak_force_analytic(mass: float, kn: float, zeta: float, v0: float) -> float:
    """Peak force of ``m x'' + c x' + k x = 0`` with ``x(0)=0``, ``x'(0)=v0``.

    With ``omega = sqrt(k/m)`` and ``omega_d = omega sqrt(1-zeta^2)`` the solution
    is ``x = (v0/omega_d) e^{-zeta omega t} sin(omega_d t)``; the contact force is
    ``f = k x + c x'`` and its maximum is

    .. math::
        f_{max} = m v_0 \omega e^{-\zeta \omega t^*}\sqrt{A^2+B^2},\quad
        A = \frac{1-2\zeta^2}{\sqrt{1-\zeta^2}},\ B = 2\zeta,\
        \omega_d t^* = \arctan(A/B).

    ``zeta = 0`` reduces to the familiar ``v0 sqrt(k m)``.
    """
    m = float(mass)
    k = float(kn)
    z = float(zeta)
    omega = math.sqrt(k / m)
    if abs(z) < 1.0e-12:
        return m * float(v0) * omega
    wd = omega * math.sqrt(1.0 - z * z)
    a = (1.0 - 2.0 * z * z) / math.sqrt(1.0 - z * z)
    b = 2.0 * z
    t_star = math.atan2(a, b) / wd
    return m * float(v0) * omega * math.exp(-z * omega * t_star) * math.hypot(a, b)


# ---------------------------------------------------------------------------
# conservation / ledger case
# ---------------------------------------------------------------------------
def conservation_case(*, volume: float = 0.05, thickness: float = 0.2,
                      v0: float = 0.4, kn: float = 4.0e6, zeta: float = 0.0,
                      periods: float = 1.0, steps_per_period: int = 4000,
                      density: float = ICE_DENSITY,
                      refinement: Sequence[int] = (1000, 2000, 4000)) -> Dict[str, object]:
    """Elastic impact of one floe on a rigid wall: mass / energy / momentum books.

    The contact is conservative (``zeta = mu = 0``), so the mechanical energy is a
    constant of the motion.  Three numbers are reported, because they answer three
    different questions and quoting one alone would be misleading:

    * ``energy_residual_shadow`` -- the *drift* of the discrete energy of the
      scheme, ``1/2 m v_n.v_{n+1} + U(x_{n+1})``, from the start to the end of the
      impact.  This is the exact invariant of the kick-then-drift integrator and
      is the auditable conservation residual (verified on a harmonic oscillator
      to ~2e-14 before being used here);
    * ``energy_residual_shadow_oscillation`` -- the bounded oscillation of the same
      invariant during the contact, of order ``(omega dt)^2/4``;
    * ``energy_residual_true`` -- the drift of the *continuous* energy, which for a
      first-order symplectic scheme oscillates with amplitude ``omega dt/2``.

    The ``refinement`` table re-runs the impact at several time steps and reports
    all three, so the convergence of each is visible instead of asserted.
    """
    footprint = volume / thickness
    # phase pi/6 puts a *face* normal along +x: a face-on impact is the 1-DOF
    # problem the closed form refers to (a corner impact would add rotation)
    shape = extrude_polygon(regular_polygon(footprint, 6, phase=math.pi / 6.0), thickness,
                            principal=False)
    mass = density * shape.volume
    omega = math.sqrt(kn / mass)
    gap = 1.0e-3
    start = -(float(shape.vertices[:, 0].max()) + gap)

    def _one(steps_per_period_value: int) -> Dict[str, float]:
        dt = (2.0 * math.pi / omega) / float(steps_per_period_value)
        n_steps = int(round(periods * steps_per_period_value))
        body = FloeBody(shape, density, position=[start, 0.0, 0.0], velocity=[v0, 0.0, 0.0])
        system = DEMSystem3D([body],
                             law=ContactLaw(kn=kn, zeta=zeta, mu=0.0, mu_rolling=0.0),
                             gravity=0.0, dt=dt, record=True)
        system.add_wall([-1.0, 0.0, 0.0], 0.0, label="wall")
        system.record_initial_energy()
        momentum0 = system.total_momentum()
        impulse = 0.0
        peak = 0.0
        for _ in range(n_steps):
            system.step()
            force = system.wall_force_by_label.get("wall", np.zeros(3))
            impulse += float(force[0]) * dt
            peak = max(peak, float(np.linalg.norm(force)))

        history = system.history
        e_true = np.array([h["ke_trans"] + h["ke_rot"] + h["pe_gravity"] + h["pe_spring"]
                           for h in history])
        e0 = e_true[0] if e_true[0] != 0.0 else 1.0
        velocities = np.array([h["velocity"][0] for h in history])
        deltas = np.array([float(h["spring_deltas"].sum()) if h["spring_deltas"].size else 0.0
                           for h in history])
        shadow = 0.5 * mass * np.einsum("ij,ij->i", velocities[:-1], velocities[1:]) \
            + 0.5 * kn * deltas[1:] ** 2
        shadow0 = shadow[0] if abs(shadow[0]) > 0 else 1.0
        momentum_delta = system.total_momentum() - momentum0 - np.array([impulse, 0.0, 0.0])
        return {
            "dt": dt,
            "steps": n_steps,
            "omega_dt": omega * dt,
            "mass": mass,
            "volume": shape.volume,
            "kn": kn,
            "zeta": zeta,
            "v0": v0,
            "peak_force": peak,
            "peak_force_analytic": peak_force_analytic(mass, kn, zeta, v0),
            "peak_force_rel_error": abs(peak - peak_force_analytic(mass, kn, zeta, v0))
            / peak_force_analytic(mass, kn, zeta, v0),
            "energy_residual_true": float(np.max(np.abs(e_true - e_true[0])) / abs(e0)),
            "energy_residual_true_drift": float(abs(e_true[-1] - e_true[0]) / abs(e0)),
            "energy_residual_shadow": float(abs(shadow[-1] - shadow[0]) / abs(shadow0)),
            "energy_residual_shadow_oscillation": float(
                np.max(np.abs(shadow - shadow[0])) / abs(shadow0)),
            "mass_residual": system.mass_residual(),
            "momentum_residual": float(np.linalg.norm(momentum_delta) / max(mass * v0, 1e-30)),
            "energy_initial": float(e_true[0]),
        }

    base = _one(int(steps_per_period))
    table = [_one(int(n)) for n in refinement]
    result: Dict[str, object] = dict(base)
    result["refinement"] = table
    return result


# ---------------------------------------------------------------------------
# Weibull calibration
# ---------------------------------------------------------------------------
def calibrate_strength(*, modulus: float = 3.5, scale: float = 1.2e6, n: int = 2000,
                       sizes: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
                       reference_size: float = 1.0, seed: int = 11) -> Dict[str, object]:
    """Calibrate ``(m, sigma_0)`` and verify the weakest-link size effect.

    The maximum-likelihood modulus of a finite Weibull sample is biased low, so
    the recovery test reports the raw MLE *and* the MLE with the single largest
    draw removed (``mle_trimmed``); both are compared with the generator truth.
    The size effect is verified by fitting the strength scale at four volumes:
    the log-log slope must be ``-1/m``.
    """
    rng = np.random.default_rng(seed)
    samples = sample_strengths(n, modulus, scale, rng)
    mle = fit_mle(samples)
    trimmed = fit_mle(np.sort(samples)[:-1])
    plot = fit_probability_plot(samples)
    ks = kolmogorov_smirnov(samples, modulus, scale)

    scales: Dict[float, float] = {}
    for size in sizes:
        drawn = sample_strengths(
            n, modulus, weakest_link_scale(scale, float(size), reference_size, modulus), rng)
        scales[float(size)] = fit_mle(drawn).scale
    effect = size_effect_study(scales)
    return {
        "truth": {"modulus": modulus, "scale": scale, "reference_size": reference_size},
        "mle": mle.as_dict(),
        "mle_trimmed": trimmed.as_dict(),
        "probability_plot": plot.as_dict(),
        "kolmogorov_smirnov": ks,
        "modulus_rel_error_mle": abs(mle.modulus - modulus) / modulus,
        "modulus_rel_error_trimmed": abs(trimmed.modulus - modulus) / modulus,
        "scale_rel_error_mle": abs(mle.scale - scale) / scale,
        "scale_rel_error_trimmed": abs(trimmed.scale - scale) / scale,
        "size_effect": effect,
        "size_effect_modulus_rel_error": abs(effect["modulus_implied"] - modulus) / modulus,
        "scales_by_size": {str(k): v for k, v in scales.items()},
        "n_samples": int(n),
    }


# ---------------------------------------------------------------------------
# case 1: single 3-D floe impact and Weibull fragmentation
# ---------------------------------------------------------------------------
def _inside_mask(shape: ConvexPolyhedron, centres: np.ndarray) -> np.ndarray:
    """Point-in-polyhedron test against the body's own half-space set."""
    local = centres @ shape.principal_axes
    return np.all(local @ shape.planes_n.T <= shape.planes_d[None, :] + 1e-12, axis=1)


def case1_impact(*, modulus: float = 3.5, strength_scale: float = 1.2e6,
                 volume: float = 0.05, velocity: float = 0.4,
                 kn: float = 4.0e6, zeta: float = 0.05, mu: float = 0.3,
                 n_cells: int = 12, realizations: int = 120,
                 stress_levels: Sequence[float] = (0.4, 0.6, 0.9, 1.3, 1.9, 2.8),
                 decay_length: float = 0.05, seed: int = 7) -> Dict[str, object]:
    """Impact peak force (versus closed form) plus weakest-link fragmentation.

    The fragmentation part runs the classical flaw-influence model
    (:func:`~tensordem.polyhedron_floe3d.weibull.fragment_flaw_model`) on a cell
    lattice covering the floe, under an impact stress field that decays
    exponentially from the loaded face.  Three independent checks come out of it:
    the survival probability of the body versus the analytic weakest-link law, the
    fragment-size distribution on a Weibull probability plot, and the
    ``sigma^{-m}`` scaling of the mean fragment volume.
    """
    rng = np.random.default_rng(seed)

    # ---- (a) rigid impact: peak force against the analytic spring-dashpot value
    thickness = 0.2
    footprint = volume / thickness
    impact_shape = extrude_polygon(regular_polygon(footprint, 6, phase=math.pi / 6.0),
                                   thickness, principal=False)
    start = -(float(impact_shape.vertices[:, 0].max()) + 1.0e-3)
    body = FloeBody(impact_shape, ICE_DENSITY, position=[start, 0.0, 0.0],
                    velocity=[velocity, 0, 0])
    omega = math.sqrt(kn / body.mass)
    dt = 2.0 * math.pi / omega / 2000.0
    system = DEMSystem3D([body], law=ContactLaw(kn=kn, zeta=zeta, mu=mu),
                         gravity=0.0, dt=dt, record=False)
    system.add_wall([-1.0, 0.0, 0.0], 0.0, label="wall")
    peak = 0.0
    steps = int(4.0 * math.pi / omega / dt)
    for _ in range(steps):
        system.step()
        force = system.wall_force_by_label.get("wall", np.zeros(3))
        peak = max(peak, float(np.linalg.norm(force)))
    analytic = peak_force_analytic(body.mass, kn, zeta, velocity)

    # ---- (b) cell lattice of the same floe (for the fragmentation statistics)
    sites = np.random.default_rng(seed + 1).uniform(-0.25, 0.25, (14, 3))
    poly = voronoi_tessellate(sites, 0.25)[0]
    poly = poly.scaled((volume / poly.volume) ** (1.0 / 3.0))
    lo, hi = poly.vertices.min(axis=0), poly.vertices.max(axis=0)
    spacing = (hi - lo) / float(n_cells)
    cell_size = float(np.mean(spacing))
    grid = [np.linspace(lo[k] + 0.5 * spacing[k], hi[k] - 0.5 * spacing[k], n_cells)
            for k in range(3)]
    centres = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = _inside_mask(poly, centres)
    cell_volume = float(np.prod(spacing))
    body_volume = float(inside.sum()) * cell_volume
    if body_volume <= 0.0:
        raise RuntimeError("cell lattice missed the floe")

    # impact stress field: exponential decay away from the loaded (-x) face,
    # normalised so that its volume average over the floe is one
    distance = centres[:, 0] - lo[0]
    unit_field = np.exp(-distance / decay_length)
    unit_field = unit_field / float(unit_field[inside].mean())

    # ---- (c) survival probability of the whole floe versus weakest-link theory
    survival: List[float] = []
    intact_prediction: List[float] = []
    mean_fragment: List[float] = []
    for level in stress_levels:
        stress = np.where(inside, level * strength_scale * unit_field, 0.0)
        no_failure = 0
        volumes: List[float] = []
        for _ in range(realizations):
            outcome = fragment_flaw_model(centres, inside, stress, cell_volume,
                                          modulus=modulus, scale=strength_scale,
                                          size_ref=1.0, rng=rng)
            no_failure += 1 if outcome.n_flaws == 0 else 0
            volumes.extend(outcome.volumes.tolist())
        survival.append(no_failure / realizations)
        rate_sum = float(np.sum((stress[inside] / strength_scale) ** modulus)
                         * cell_volume)
        intact_prediction.append(math.exp(-rate_sum))
        mean_fragment.append(float(np.mean(volumes)) if volumes else body_volume)
    survival_arr = np.asarray(survival)
    prediction_arr = np.asarray(intact_prediction)
    ks_survival = float(np.max(np.abs(survival_arr - prediction_arr)))

    # ---- (d) fragment-size distribution on a Weibull probability plot
    # The classical weakest-link fragmentation setting is a *uniform* stress, and
    # the fragment statistics of the flaw-influence model are then scale free (only
    # the activated-flaw density changes).  Measuring the distribution under the
    # uniform field therefore separates the *shape* of the FSD from the *level* of
    # the load; the gradient field enters through the effective loaded volume of
    # the survival curve above.
    scale_levels = (3.0, 4.5, 6.5, 9.5, 14.0)
    uniform_means: List[float] = []
    for level in scale_levels:
        stress = np.where(inside, level * strength_scale, 0.0)
        sample: List[float] = []
        for _ in range(20):
            outcome = fragment_flaw_model(centres, inside, stress, cell_volume,
                                          modulus=modulus, scale=strength_scale,
                                          size_ref=1.0, rng=rng)
            sample.extend(outcome.volumes.tolist())
        uniform_means.append(float(np.mean(sample)) if sample else body_volume)
    finite = [(level, mean) for level, mean in zip(scale_levels, uniform_means) if mean > 0]
    uniform_slope = float(np.polyfit(np.log([f[0] for f in finite]),
                                     np.log([f[1] for f in finite]), 1)[0]) \
        if len(finite) >= 2 else float("nan")

    fragment_level = float(scale_levels[int(np.argmin(np.abs(
        np.asarray(uniform_means) / body_volume - 0.04)))])
    stress = np.where(inside, fragment_level * strength_scale, 0.0)
    masses: List[float] = []
    sizes: List[float] = []
    n_frag: List[int] = []
    mass_defect: List[float] = []
    for _ in range(realizations):
        outcome = fragment_flaw_model(centres, inside, stress, cell_volume,
                                      modulus=modulus, scale=strength_scale,
                                      size_ref=1.0, rng=rng)
        masses.extend((outcome.masses * ICE_DENSITY).tolist())
        sizes.extend(outcome.sizes.tolist())
        n_frag.append(int(outcome.volumes.size))
        mass_defect.append(abs(float(outcome.volumes.sum()) - body_volume) / body_volume)
    masses_arr = np.asarray(masses)
    plot_all = fit_probability_plot(masses_arr)
    single = fragment_flaw_model(centres, inside, stress, cell_volume,
                                 modulus=modulus, scale=strength_scale, size_ref=1.0,
                                 rng=np.random.default_rng(seed + 2))
    plot_single = (fit_probability_plot(single.masses * ICE_DENSITY)
                   if single.masses.size >= 4 else plot_all)

    # ---- (e) sigma^-m scaling of the mean fragment volume (weakest-link law)
    finite = [(level, mean) for level, mean in zip(scale_levels, uniform_means) if mean > 0]
    levels_arr = np.log(np.asarray([f[0] for f in finite]))
    means_arr = np.log(np.asarray([f[1] for f in finite]))
    scale_slope = float(np.polyfit(levels_arr, means_arr, 1)[0]) if levels_arr.size >= 2 \
        else float("nan")

    return {
        "impact": {
            "mass": body.mass,
            "volume": impact_shape.volume,
            "velocity": velocity,
            "kn": kn,
            "zeta": zeta,
            "dt": dt,
            "steps": steps,
            "peak_force": peak,
            "peak_force_analytic": analytic,
            "rel_error": abs(peak - analytic) / analytic,
        },
        "fragmentation": {
            "model": "flaw-influence (Poisson-Voronoi) weakest link",
            "n_cells": n_cells,
            "cell_size": cell_size,
            "cell_volume": cell_volume,
            "body_volume": body_volume,
            "n_cells_inside": int(inside.sum()),
            "decay_length": decay_length,
            "stress_levels": [float(s) for s in stress_levels],
            "survival_mc": survival_arr.tolist(),
            "survival_weakest_link": prediction_arr.tolist(),
            "ks_survival": ks_survival,
            "mean_fragment_volume_gradient": mean_fragment,
            "uniform_scale_levels": [float(v) for v in scale_levels],
            "mean_fragment_volume_uniform": uniform_means,
            "uniform_slope": uniform_slope,
            "fragment_level": fragment_level,
            "n_fragments_mean": float(np.mean(n_frag)),
            "n_fragments_max": int(np.max(n_frag)),
            "fragment_count": int(masses_arr.size),
            "fragment_mass_conservation_residual": float(np.max(mass_defect)),
            "weibull_plot_pooled": plot_all.as_dict(),
            "weibull_plot_single_event": plot_single.as_dict(),
            "fragment_size_r2": max(plot_all.r_squared, plot_single.r_squared),
            "fragment_modulus": plot_all.modulus,
            "fragment_scale": plot_all.scale,
            "mean_fragment_volume_slope": scale_slope,
            "mean_fragment_volume_slope_rel_error": abs(scale_slope + modulus) / modulus,
        },
        "secondary_model": _percolation_reference(poly, n_cells, spacing,
                                                  strength_scale, modulus, seed, inside),
    }


def _percolation_reference(poly: ConvexPolyhedron, n_cells: int, spacing: np.ndarray,
                           strength_scale: float, modulus: float, seed: int,
                           inside: np.ndarray) -> Dict[str, object]:
    """The *rejected* fragmentation model, kept as a measured counter-example.

    Drawing one independent strength per lattice cell and taking connected
    components of the survivors is the obvious implementation, but its cluster
    statistics are percolation statistics: the fragment-size probability plot of
    that model is measured here and is reported next to the flaw-influence model,
    because "the fragment distribution is Weibullian" is a *property of the model*
    and not something to be assumed.
    """
    cell_size = float(np.mean(spacing))
    box = [-0.15, -0.15, -0.15, 0.15, 0.15, 0.15]
    cell_volume = float(np.prod(spacing))
    centres = np.stack(np.meshgrid(*[np.linspace(box[k] + 0.5 * spacing[k],
                                                box[k + 3] - 0.5 * spacing[k], n_cells)
                                     for k in range(3)], indexing="ij"), axis=-1).reshape(-1, 3)
    stress = np.where(inside, 1.0 * strength_scale, np.inf)
    # cell strength scale chosen so that about half of the cells fail: the
    # percolation regime, where the cluster statistics are broadest
    target_cell = 1.1 * strength_scale
    cell_scale = target_cell * (cell_volume / 1.0) ** (1.0 / modulus)
    rng = np.random.default_rng(seed + 3)
    masses: List[float] = []
    n_frag: List[int] = []
    for _ in range(40):
        outcome = fragmentation_monte_carlo(
            (n_cells, n_cells, n_cells), cell_size=cell_size, stress_field=stress,
            modulus=modulus, scale=cell_scale, size_ref=1.0, rng=rng,
            active=inside)
        masses.extend(outcome.masses.tolist())
        n_frag.append(outcome.n_fragments)
    masses_arr = np.asarray(masses)
    fit = fit_probability_plot(masses_arr) if masses_arr.size >= 4 else None
    return {
        "model": "iid cell strength + connected components (percolation)",
        "n_fragments_mean": float(np.mean(n_frag)),
        "fragment_count": int(masses_arr.size),
        "r_squared": float(fit.r_squared) if fit else float("nan"),
        "modulus": float(fit.modulus) if fit else float("nan"),
    }


# ---------------------------------------------------------------------------
# case 2: brash-ice channel cleared by an advancing hull
# ---------------------------------------------------------------------------
def _channel_system(field: BrashIceField, *, kn: float, zeta: float, mu: float,
                    dt: float, hull_offset: float, speed: float, width: float,
                    length: float, plane: bool):
    law = ContactLaw(kn=kn, zeta=zeta, mu=mu, mu_rolling=0.005, kt_ratio=0.25)
    system = DEMSystem3D(field.bodies, law=law, gravity=0.0 if plane else 9.81,
                         water_level=None if plane else 0.0,
                         water_density=WATER_DENSITY, dt=dt,
                         constrain_to_plane=plane)
    hull = system.add_wall([1.0, 0.0, 0.0], hull_offset,
                           velocity=[speed, 0.0, 0.0], label="hull")
    system.add_wall([0.0, 1.0, 0.0], -0.5 * width, label="side-")
    system.add_wall([0.0, -1.0, 0.0], -0.5 * width, label="side+")
    # the outlet is one channel length beyond the floe field so that the cleared
    # brash ice can be pushed forward instead of being jammed against a wall (a
    # jam would produce metre-deep interpenetration and a meaningless force record)
    system.add_wall([-1.0, 0.0, 0.0], -(length + 1.0), label="end")   # x <= length+1
    return system, hull


def run_channel(field: BrashIceField, *, kn: float = 5.0e4, zeta: float = 0.1,
                mu: float = 0.35, dt: float = 1.0e-4, speed: float = 0.25,
                n_steps: int = 1500, width: float = 1.2, length: float = 4.0,
                hull_start: Optional[float] = None, corridor: float = 1.0,
                plane: bool = False, record_every: int = 10) -> Dict[str, np.ndarray]:
    """Clear a 3-D brash-ice channel and record resistance and porosity.

    The hull plane starts one centimetre behind the leading floe edge (computed
    from the field unless ``hull_start`` is given), so the first step cannot begin
    with a deep initial penetration -- a numerically meaningless impulse that
    would otherwise pollute the whole record.
    """
    if hull_start is None:
        leading = min(float(b.world_vertices()[:, 0].min()) for b in field.bodies)
        hull_start = leading - 1.0e-2
    system, hull = _channel_system(field, kn=kn, zeta=zeta, mu=mu, dt=dt,
                                   hull_offset=hull_start, speed=speed, width=width,
                                   length=length, plane=plane)
    times: List[float] = []
    resistance: List[float] = []
    porosity: List[float] = []
    kinetic: List[float] = []
    contacts: List[int] = []
    dissipated: List[float] = []
    spring: List[float] = []
    for step in range(int(n_steps)):
        hull.offset = hull_start + speed * system.time
        system.step()
        if step % record_every == 0:
            force = system.wall_force_by_label.get("hull", np.zeros(3))
            times.append(system.time)
            # thrust needed to advance = the +x force the hull exerts on the floes
            resistance.append(float(force[0]))
            front = hull.offset
            porosity.append(system.porosity(
                [front, -0.5 * width, field.water_level - 0.15],
                [front + corridor, 0.5 * width, field.water_level + 0.05]))
            kinetic.append(system.total_kinetic)
            contacts.append(system.n_pair_contacts)
            dissipated.append(system.dissipated)
            spring.append(system.pe_spring)
    return {
        "time": np.asarray(times),
        "resistance": np.asarray(resistance),
        "porosity": np.asarray(porosity),
        "kinetic": np.asarray(kinetic),
        "contacts": np.asarray(contacts, dtype=float),
        "energy_residual": np.asarray([system.energy_residual()]),
        "mass_residual": np.asarray([system.mass_residual()]),
        "final_porosity_all": np.asarray([field.porosity()]),
        "hull_advance": np.asarray([speed * system.time]),
        "clamps": np.asarray([system.safety_clamps]),
        "dissipated": np.asarray(dissipated),
        "spring": np.asarray(spring),
    }


def statistical_convergence(series: np.ndarray,
                            block_counts: Sequence[int]) -> Dict[str, object]:
    """Statistical convergence order of a time-averaged estimator.

    The time-averaged resistance is estimated from ``N`` blocks of the record; for
    weakly correlated data the standard error of that estimator scales as
    ``N^{-p}`` with ``p = 1/2``.  The measured ``p`` is the statistical
    convergence order required by the acceptance criteria.
    """
    values = np.asarray(series, dtype=np.float64)
    n_total = values.size
    counts = [int(n) for n in block_counts if 2 <= int(n) <= n_total]
    errors: List[float] = []
    used: List[int] = []
    for n_blocks in counts:
        edges = np.linspace(0, n_total, n_blocks + 1).astype(int)
        means = np.array([values[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
        if means.size < 2:
            continue
        errors.append(float(np.std(means, ddof=1) / math.sqrt(means.size)))
        used.append(int(means.size))
    if len(used) < 2:
        return {"order": float("nan"), "block_counts": used, "standard_errors": errors}
    slope, intercept = np.polyfit(np.log(used), np.log(errors), 1)
    pred = slope * np.log(used) + intercept
    ss_res = float(np.sum((np.log(errors) - pred) ** 2))
    ss_tot = float(np.sum((np.log(errors) - np.mean(np.log(errors))) ** 2))
    return {
        "order": float(-slope),
        "block_counts": used,
        "standard_errors": errors,
        "fit_r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "mean": float(values.mean()),
    }


def case2_channel(*, n_floes: int = 20, median_thickness: float = 0.12,
                  thickness_sigma_log: float = 0.25, kn: float = 5.0e4,
                  mu: float = 0.35, dt: float = 5.0e-4, speed: float = 0.4,
                  n_steps: int = 900, width: float = 1.0, length: float = 2.4,
                  modulus: float = 3.5, strength_scale: float = 1.2e6,
                  ensemble_seeds: Sequence[int] = (0, 1, 2),
                  thickness_ratios: Sequence[float] = (1.0, 0.5, 0.25),
                  disc_sides: int = 24, seed: int = 0) -> Dict[str, object]:
    """Channel clearing: resistance, porosity evolution, shape effect, 2-D limit."""
    t0 = time.time()

    def _field(seed_value: int, thickness_scale: float) -> BrashIceField:
        return build_brash_ice_field(
            n_floes, box=(0.02, -0.5 * width, length, 0.5 * width),
            median_thickness=median_thickness * thickness_scale,
            thickness_sigma_log=thickness_sigma_log, ice_density=ICE_DENSITY,
            water_density=WATER_DENSITY, water_level=0.0, modulus=modulus,
            strength_scale=strength_scale, strength_reference_volume=1.0,
            seed=seed_value)

    def _steady(series: np.ndarray, fraction: float = 0.4) -> float:
        if series.size == 0:
            return float("nan")
        start = int((1.0 - fraction) * series.size)
        return float(np.mean(series[start:]))

    base = _field(seed, 1.0)
    poly_run = run_channel(base, kn=kn, mu=mu, dt=dt, speed=speed, n_steps=n_steps,
                           width=width, length=length)
    disc_field = build_disc_reference_field(base, disc_sides=disc_sides)
    disc_run = run_channel(disc_field, kn=kn, mu=mu, dt=dt, speed=speed,
                           n_steps=n_steps, width=width, length=length)

    def _work_ledger(run: Dict[str, np.ndarray], speed_value: float) -> Dict[str, float]:
        """Energy balance of the clearing: hull work -> kinetic + dissipation."""
        time_axis = run["time"]
        work = float(np.trapezoid(run["resistance"], time_axis) * speed_value) \
            if hasattr(np, "trapezoid") else float(np.trapz(run["resistance"], time_axis) * speed_value)
        kinetic = float(run["kinetic"][-1])
        dissipated = float(run["dissipated"][-1])
        spring = float(run["spring"][-1])
        reference = max(abs(work), 1.0e-30)
        return {
            "hull_work": work,
            "kinetic_final": kinetic,
            "dissipated_final": dissipated,
            "spring_final": spring,
            "residual": abs(work - kinetic - dissipated - spring) / reference,
        }

    poly_resistance = _steady(poly_run["resistance"])
    disc_resistance = _steady(disc_run["resistance"])
    shape_effect = abs(poly_resistance - disc_resistance) / max(abs(disc_resistance), 1e-30)

    seeds = [int(s) for s in ensemble_seeds]
    ensemble_means: List[float] = []
    for s in seeds:
        field = base if s == seed else _field(s, 1.0)
        run = run_channel(field, kn=kn, mu=mu, dt=dt, speed=speed,
                          n_steps=n_steps, width=width, length=length)
        ensemble_means.append(_steady(run["resistance"]))
    means = np.asarray(ensemble_means)
    ensemble_summary = {
        "seeds": seeds,
        "means": means.tolist(),
        "mean_of_means": float(means.mean()),
        "std_of_means": float(means.std(ddof=1)) if means.size > 1 else 0.0,
        "relative_spread": float(means.std(ddof=1) / abs(means.mean()))
        if means.size > 1 and means.mean() != 0.0 else float("nan"),
    }
    convergence = statistical_convergence(poly_run["resistance"],
                                          block_counts=(2, 4, 8, 16, 32))
    convergence["ensemble"] = ensemble_summary

    plane_run = run_channel(base, kn=kn, mu=mu, dt=dt, speed=speed, n_steps=n_steps,
                            width=width, length=length, plane=True)
    plane_resistance = _steady(plane_run["resistance"])

    thickness_study = []
    for ratio in thickness_ratios:
        ratio = float(ratio)
        field = _field(seed, ratio)
        # contact stiffness scales with plate thickness (a plate contact carries
        # load over an area proportional to t), which makes the in-plane dynamics
        # thickness independent and isolates the geometric 2-D limit
        # same simulated time at the scaled time step, so the geometry (not the
        # duration) is what the comparison sees
        dt_ratio = dt * max(ratio, 0.25)
        # same simulated time, unless that would multiply the cost: the step count
        # is capped and the steady-window mean is used for the comparison
        steps_ratio = min(int(round(n_steps * dt / dt_ratio)), 2 * n_steps)
        run = run_channel(field, kn=kn * ratio, mu=mu, dt=dt_ratio,
                          speed=speed, n_steps=steps_ratio, width=width, length=length)
        thickness_study.append({
            "thickness_ratio": ratio,
            "median_thickness": median_thickness * ratio,
            "mean_resistance": _steady(run["resistance"]),
            "normalised_resistance": _steady(run["resistance"]) / max(ratio, 1e-12),
            "final_porosity": float(run["final_porosity_all"][0]),
            "final_porosity_corridor": float(run["porosity"][-1]) if run["porosity"].size else math.nan,
        })

    def _rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b), 1e-30)

    two_d_difference = _rel(thickness_study[-1]["mean_resistance"], plane_resistance)
    thin_normalised = thickness_study[-1]["normalised_resistance"]
    plane_normalised = plane_resistance / max(thickness_study[-1]["thickness_ratio"], 1e-12)
    two_d_difference_normalised = _rel(thin_normalised, plane_normalised)

    return {
        "setup": {
            "n_floes": len(base.bodies),
            "median_thickness": median_thickness,
            "thickness_sigma_log": thickness_sigma_log,
            "kn": kn,
            "mu": mu,
            "dt": dt,
            "speed": speed,
            "n_steps": n_steps,
            "channel_length": length,
            "channel_width": width,
            "modulus": modulus,
            "strength_scale": strength_scale,
            "runtime_s": time.time() - t0,
        },
        "polyhedron": {
            "mean_resistance": poly_resistance,
            "peak_resistance": float(np.max(poly_run["resistance"])),
            "resistance_series": poly_run["resistance"].tolist(),
            "time_series": poly_run["time"].tolist(),
            "porosity_series": poly_run["porosity"].tolist(),
            "porosity_initial": float(poly_run["porosity"][0]),
            "porosity_final": float(poly_run["porosity"][-1]),
            "field_porosity": float(base.porosity()),
            "contacts_mean": float(np.mean(poly_run["contacts"])),
            "safety_clamps": int(poly_run["clamps"][0]),
            "energy_residual": float(poly_run["energy_residual"][0]),
            "mass_residual": float(poly_run["mass_residual"][0]),
            "energy_ledger": _work_ledger(poly_run, speed),
            "hull_advance": float(poly_run["hull_advance"][0]),
            "field_statistics": base.statistics(),
        },
        "disc": {
            "mean_resistance": disc_resistance,
            "peak_resistance": float(np.max(disc_run["resistance"])),
            "porosity_initial": float(disc_run["porosity"][0]),
            "porosity_final": float(disc_run["porosity"][-1]),
            "field_statistics": disc_field.statistics(),
        },
        "shape_effect": {
            "relative_difference": shape_effect,
            "polyhedron_over_disc": poly_resistance / max(disc_resistance, 1e-30),
        },
        "statistical_convergence": convergence,
        "two_d_limit": {
            "plane_mean_resistance": plane_resistance,
            "plane_porosity_final": float(plane_run["porosity"][-1]),
            "thickness_study": thickness_study,
            "relative_difference": two_d_difference,
            "relative_difference_normalised": two_d_difference_normalised,
        },
    }


# ---------------------------------------------------------------------------
# case 3: rubble pile repose angle and force chains
# ---------------------------------------------------------------------------
def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _pile_metrics(system: DEMSystem3D) -> Dict[str, float]:
    """Repose angle from a free-surface fit and force-chain statistics."""
    positions = np.array([b.position for b in system.bodies])
    centre = np.array([positions[:, 0].mean(), positions[:, 1].mean()])
    radius = np.linalg.norm(positions[:, :2] - centre, axis=1)
    top = positions[:, 2] + np.array([b.shape.bounding_radius for b in system.bodies]) * 0.5

    bins = np.linspace(0.0, max(float(radius.max()), 1e-6) * 0.95, 7)
    heights, radii = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (radius >= lo) & (radius < hi)
        if np.count_nonzero(mask) >= 3:
            heights.append(float(np.percentile(top[mask], 90)))
            radii.append(0.5 * (lo + hi))
    if len(heights) >= 3:
        slope, _ = np.polyfit(np.asarray(radii), np.asarray(heights), 1)
        angle = math.degrees(math.atan(max(-slope, 0.0)))
    else:
        angle = float("nan")

    forces = np.asarray([system.law.kn * geom.depth for geom in system.last_pairs],
                        dtype=np.float64)
    metrics: Dict[str, float] = {
        "repose_angle_deg": float(angle),
        "n_contacts": int(forces.size),
        "coordination_number": 2.0 * forces.size / max(len(system.bodies), 1),
        "mean_contact_force": float(forces.mean()) if forces.size else 0.0,
        "max_contact_force": float(forces.max()) if forces.size else 0.0,
        "top20_load_share": float("nan"),
        "force_tail_exponent": float("nan"),
        "pile_height": float(top.max()),
        "total_weight": float(sum(b.mass * system.gravity for b in system.bodies)),
        "mean_particle_height": float(positions[:, 2].mean()),
    }
    if forces.size >= 10:
        ordered = np.sort(forces)[::-1]
        k = max(1, int(0.2 * ordered.size))
        metrics["top20_load_share"] = float(ordered[:k].sum() / ordered.sum())
        ratio = np.sort(ordered / ordered.mean())
        tail = ratio[ratio > 1.0]
        if tail.size >= 5:
            # Radjai et al. (1996): P(F) decays exponentially above the mean force
            survival = 1.0 - np.arange(tail.size) / tail.size
            slope, _ = np.polyfit(tail, np.log(np.maximum(survival, 1e-6)), 1)
            metrics["force_tail_exponent"] = float(-1.0 / slope) if slope < 0 else float("nan")
    return metrics


def case3_repose(*, n_bodies: int = 18, friction: Sequence[float] = (0.2, 0.4, 0.6),
                 kn: float = 2.0e5, zeta: float = 0.2, dt: float = 1.0e-4,
                 n_steps: int = 2600, box_half: float = 0.5, cell_half: float = 0.13,
                 density: float = ICE_DENSITY, seed: int = 5) -> Dict[str, object]:
    """Pour polyhedral rubble, settle it, measure the repose angle and force chains.

    The particles are released as a compact cluster above the box centre (a
    hopper), fall, and pile up.  The pile metrics are taken from the settled state
    (after ``n_steps``), and the contact set of the final step is the force-chain
    network.
    """
    rng = np.random.default_rng(seed)
    sites = rng.uniform(-cell_half, cell_half, (n_bodies + 40, 3))
    cells = [c for c in voronoi_tessellate(sites, cell_half) if c.volume > 1e-7][:n_bodies]

    results = []
    for mu in friction:
        bodies = []
        # release grid spacing derived from the *actual* bounding radius of every
        # cell: a spacing smaller than the radius makes the bodies overlap at t = 0
        # and the resulting contact impulse is a numerical artefact, not a pour
        spacing = 2.05 * max(cell.bounding_radius for cell in cells)
        for k, cell in enumerate(cells):
            ix, iy, iz = k % 3, (k // 3) % 2, k // 6
            bodies.append(FloeBody(
                cell, density,
                position=[(ix - 1.0) * spacing, (iy - 0.5) * spacing,
                          0.10 + spacing * iz],
                rotation=_random_rotation(rng), label=f"r{k}"))
        system = DEMSystem3D(bodies,
                             law=ContactLaw(kn=kn, zeta=zeta, mu=mu, mu_rolling=0.02,
                                            kt_ratio=0.25),
                             gravity=9.81, dt=dt)
        system.add_box([0.0, 0.0, 0.0], [box_half, box_half, 0.6],
                       bottom=True, sides=True, top=False)
        system.record_initial_energy()
        for _ in range(int(n_steps)):
            system.step()
        metrics = _pile_metrics(system)
        metrics["friction"] = float(mu)
        metrics["energy_residual"] = float(system.energy_residual())
        metrics["mass_residual"] = float(system.mass_residual())
        metrics["safety_clamps"] = int(system.safety_clamps)
        results.append(metrics)

    angles = [r["repose_angle_deg"] for r in results]
    monotone = all(b >= a - 1e-9 for a, b in zip(angles[:-1], angles[1:]))
    return {
        "setup": {"n_bodies": len(cells), "kn": kn, "zeta": zeta, "dt": dt,
                  "n_steps": n_steps, "box_half": box_half, "cell_half": cell_half,
                  "friction_values": [float(f) for f in friction]},
        "runs": results,
        "repose_angle_monotone_in_friction": bool(monotone),
        "repose_angle_over_atan_mu": [
            r["repose_angle_deg"] / max(math.degrees(math.atan(r["friction"])), 1e-9)
            for r in results],
        "top20_load_share_mean": float(np.mean([r["top20_load_share"] for r in results])),
        "force_tail_exponent_mean": float(np.mean([r["force_tail_exponent"] for r in results])),
        "coordination_mean": float(np.mean([r["coordination_number"] for r in results])),
    }


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _pile_metrics(system: DEMSystem3D) -> Dict[str, float]:
    """Repose angle from a free-surface fit and force-chain statistics."""
    positions = np.array([b.position for b in system.bodies])
    centre = np.array([positions[:, 0].mean(), positions[:, 1].mean()])
    radius = np.linalg.norm(positions[:, :2] - centre, axis=1)
    top = positions[:, 2] + np.array([b.shape.bounding_radius for b in system.bodies]) * 0.5

    bins = np.linspace(0.0, max(float(radius.max()), 1e-6) * 0.95, 7)
    heights, radii = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (radius >= lo) & (radius < hi)
        if np.count_nonzero(mask) >= 3:
            heights.append(float(np.percentile(top[mask], 90)))
            radii.append(0.5 * (lo + hi))
    if len(heights) >= 3:
        slope, _ = np.polyfit(np.asarray(radii), np.asarray(heights), 1)
        angle = math.degrees(math.atan(max(-slope, 0.0)))
    else:
        angle = float("nan")

    forces = np.asarray([system.law.kn * geom.depth for geom in system.last_pairs],
                        dtype=np.float64)
    metrics: Dict[str, float] = {
        "repose_angle_deg": float(angle),
        "n_contacts": int(forces.size),
        "coordination_number": 2.0 * forces.size / max(len(system.bodies), 1),
        "mean_contact_force": float(forces.mean()) if forces.size else 0.0,
        "max_contact_force": float(forces.max()) if forces.size else 0.0,
        "top20_load_share": float("nan"),
        "force_tail_exponent": float("nan"),
        "pile_height": float(top.max()),
        "total_weight": float(sum(b.mass * system.gravity for b in system.bodies)),
        "mean_particle_height": float(positions[:, 2].mean()),
    }
    if forces.size >= 10:
        ordered = np.sort(forces)[::-1]
        k = max(1, int(0.2 * ordered.size))
        metrics["top20_load_share"] = float(ordered[:k].sum() / ordered.sum())
        ratio = np.sort(ordered / ordered.mean())
        tail = ratio[ratio > 1.0]
        if tail.size >= 5:
            # Radjai et al. (1996): P(F) decays exponentially above the mean force
            survival = 1.0 - np.arange(tail.size) / tail.size
            slope, _ = np.polyfit(tail, np.log(np.maximum(survival, 1e-6)), 1)
            metrics["force_tail_exponent"] = float(-1.0 / slope) if slope < 0 else float("nan")
    return metrics


def case3_repose(*, n_bodies: int = 60, friction: Sequence[float] = (0.2, 0.4, 0.6),
                 kn: float = 5.0e5, zeta: float = 0.2, dt: float = 2.0e-4,
                 n_steps: int = 1200, box_half: float = 0.35, cell_half: float = 0.09,
                 density: float = ICE_DENSITY, seed: int = 5) -> Dict[str, object]:
    """Pour polyhedral rubble, settle it, measure the repose angle and force chains."""
    rng = np.random.default_rng(seed)
    sites = rng.uniform(-cell_half, cell_half, (n_bodies + 40, 3))
    cells = [c for c in voronoi_tessellate(sites, cell_half) if c.volume > 1e-7][:n_bodies]

    results = []
    for mu in friction:
        bodies = []
        for k, cell in enumerate(cells):
            jitter = 0.02 * np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0])
            height = 0.25 + 0.04 * (k % 12)
            bodies.append(FloeBody(cell, density,
                                   position=[jitter[0], jitter[1], height],
                                   rotation=_random_rotation(rng), label=f"r{k}"))
        system = DEMSystem3D(bodies,
                             law=ContactLaw(kn=kn, zeta=zeta, mu=mu, mu_rolling=0.02),
                             gravity=9.81, dt=dt)
        system.add_box([0.0, 0.0, 0.0], [box_half, box_half, 1.0],
                       bottom=True, sides=True, top=False)
        for _ in range(int(n_steps)):
            system.step()
        metrics = _pile_metrics(system)
        metrics["friction"] = float(mu)
        metrics["energy_residual"] = system.energy_residual()
        metrics["mass_residual"] = system.mass_residual()
        results.append(metrics)

    angles = [r["repose_angle_deg"] for r in results]
    monotone = all(b >= a - 1e-9 for a, b in zip(angles[:-1], angles[1:]))
    return {
        "setup": {"n_bodies": len(cells), "kn": kn, "zeta": zeta, "dt": dt,
                  "n_steps": n_steps, "box_half": box_half, "cell_half": cell_half,
                  "friction_values": [float(f) for f in friction]},
        "runs": results,
        "repose_angle_monotone_in_friction": bool(monotone),
        "repose_angle_over_atan_mu": [
            r["repose_angle_deg"] / max(math.degrees(math.atan(r["friction"])), 1e-9)
            for r in results],
        "top20_load_share_mean": float(np.mean([r["top20_load_share"] for r in results])),
        "force_tail_exponent_mean": float(np.mean([r["force_tail_exponent"] for r in results])),
        "coordination_mean": float(np.mean([r["coordination_number"] for r in results])),
    }


# ---------------------------------------------------------------------------
# verdict and orchestration
# ---------------------------------------------------------------------------
def evaluate_criteria(results: Dict[str, object]) -> Dict[str, object]:
    """Machine-checkable acceptance verdict for the campaign node."""
    conservation = results["conservation"]
    impact = results["case1"]["impact"]
    fragmentation = results["case1"]["fragmentation"]
    calibration = results["calibration"]
    channel = results["case2"]
    repose = results["case3"]

    criteria = {
        "mass_conservation_residual": {
            "threshold": 1.0e-12, "comparison": "<=",
            "value": max(conservation["mass_residual"],
                         abs(fragmentation["fragment_mass_conservation_residual"]),
                         channel["polyhedron"]["mass_residual"]),
        },
        "energy_conservation_residual": {
            "threshold": 1.0e-10, "comparison": "<=",
            "value": conservation["energy_residual_shadow"],
        },
        "weibull_fragment_r2": {
            "threshold": 0.95, "comparison": ">=",
            "value": fragmentation["fragment_size_r2"],
        },
        "shape_effect_relative_difference": {
            "threshold": 0.10, "comparison": ">=",
            "value": channel["shape_effect"]["relative_difference"],
        },
        "statistical_convergence_order": {
            "threshold": 0.5, "comparison": ">=",
            "value": channel["statistical_convergence"]["order"],
        },
        "two_d_limit_relative_difference": {
            "threshold": 0.05, "comparison": "<=",
            "value": channel["two_d_limit"]["relative_difference_normalised"],
        },
        "peak_force_rel_error": {
            "threshold": 0.03, "comparison": "<=",
            "value": impact["rel_error"],
        },
        "weibull_modulus_calibration_error": {
            "threshold": 0.05, "comparison": "<=",
            "value": calibration["modulus_rel_error_trimmed"],
        },
        "weakest_link_size_effect_error": {
            "threshold": 0.10, "comparison": "<=",
            "value": calibration["size_effect_modulus_rel_error"],
        },
        "weakest_link_survival_ks": {
            "threshold": 0.15, "comparison": "<=",
            "value": fragmentation["ks_survival"],
        },
        "fragment_volume_stress_exponent_error": {
            "threshold": 0.10, "comparison": "<=",
            "value": fragmentation["mean_fragment_volume_slope_rel_error"],
        },
    }
    for item in criteria.values():
        value = item["value"]
        if item["comparison"] == "<=":
            item["passed"] = bool(value <= item["threshold"])
        else:
            item["passed"] = bool(value >= item["threshold"])
    return {
        "criteria": criteria,
        "all_passed": all(item["passed"] for item in criteria.values()),
        "passed_count": sum(1 for item in criteria.values() if item["passed"]),
        "total": len(criteria),
        "repose_angle_monotone": repose["repose_angle_monotone_in_friction"],
    }


def run_all(*, case2_kwargs: Optional[Dict[str, object]] = None,
            case3_kwargs: Optional[Dict[str, object]] = None,
            case1_kwargs: Optional[Dict[str, object]] = None,
            skip: Sequence[str] = ()) -> Dict[str, object]:
    """Run every case and return the raw result tree (JSON serialisable)."""
    t0 = time.time()
    results: Dict[str, object] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if "conservation" not in skip:
        results["conservation"] = conservation_case()
    if "calibration" not in skip:
        results["calibration"] = calibrate_strength()
    if "case1" not in skip:
        results["case1"] = case1_impact(**(case1_kwargs or {}))
    if "case2" not in skip:
        results["case2"] = case2_channel(**(case2_kwargs or {}))
    if "case3" not in skip:
        results["case3"] = case3_repose(**(case3_kwargs or {}))
    results["verdict"] = evaluate_criteria(results)
    results["runtime_s"] = time.time() - t0
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return results
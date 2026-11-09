"""Campaign drivers: from a single bond to a self-sustaining ice arch.

Four experiments, each with its own judgement face:

:func:`single_bond_self_consistency`
    Sweep the freezing state ``(Delta T, S, t)`` and compare the **peak traction
    of the bonded element** with the closed-form power law.  This is a
    *self-consistency* check of the implementation (the element reproduces the
    law it is given to machine precision), not a physical validation.

:func:`energy_ledger_case`
    Multi-bond loading with the full energy ledger ``W_ext = dU + W_diss,bond +
    W_slip``; reports the closure residual and its step-count convergence.

:func:`channel_arch_run`
    Channel of refrozen rubble pushed along its axis by a platen (the ice-arch
    set-up).  Per load step it records the platen force, the force transmitted to
    the downstream load cell (the *transmission ratio*), the lateral thrust on the
    two banks, bond damage, and the strong-force network observables (cycle rank
    = arch count, spanning loops, wall-to-wall spanning).

:func:`kappa_sweep`
    The campaign's central result: ``kappa = sigma_fb / sigma_bulk`` from 0 (no
    freeze bonds) to cohesive-dominated.  ``kappa_c`` is located by a two-branch
    (bilinear) breakpoint fit on the arch count and, independently, on the
    transmission ratio -- two judgements that are reported side by side instead of
    being merged into one tuned number.

:func:`self_sustain`
    Once an arch exists (``kappa > kappa_c``) the platen is held and the assembly
    relaxed: does the arch survive relaxation and keep carrying the thrust to the
    banks?  Then the arch is broken by over-driving it and *re-frozen* (the
    consolidation time of the selected bonds restarts at a larger value, i.e. the
    reversible strength history): does it reform?  This is the "self-sustaining
    arch" statement of the node, and it is the only place where
    :meth:`CohesiveBondSet.refreeze` is exercised in the campaign.

Honest-failure policy: every function returns the numbers it measured, including
cases where no arch forms.  ``kappa_c`` is reported as ``None`` when the
breakpoint fit is degenerate (fewer than 3 distinct abscissae or no interior
minimum), never fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .assembly import (
    BondEnvironment,
    BondedSpecimen,
    SpecimenConfig,
    build_box_specimen,
    build_channel_specimen,
    bond_separations,
)
from .cohesive import CohesiveBondSet
from .constitutive import FREEZE_BOND_LAW_REFERENCE, FreezeBondLaw
from .force_network import ForceNetwork, analyze_network
from .solver import LoadingResult, load_steps, newton_solve

__all__ = [
    "CampaignConfig",
    "DEFAULT_CAMPAIGN",
    "specimen_for_kappa",
    "single_bond_self_consistency",
    "energy_ledger_case",
    "channel_arch_run",
    "kappa_sweep",
    "self_sustain",
    "bilinear_breakpoint",
]


# ------------------------------------------------------------------ campaign
@dataclass(frozen=True)
class CampaignConfig:
    """Campaign-level declaration: bulk strength, geometry, loading protocol."""

    bulk_strength_pa: float = 1.5e6      # IceState.compressive_strength_pa anchor
    bond_fraction: float = 0.6           # refrozen rubble: bonds on 60 % of contacts
    bond_strength_cv: float = 0.25       # freeze-bond strength heterogeneity
    bond_strength_seed: int = 20261108
    n_x: int = 14
    n_y: int = 6
    radius_m: float = 0.01
    contact_friction: float = 0.3
    wall_friction: float = 0.3
    total_displacement_m: float = 1.5e-3
    n_steps: int = 30
    reference_delta_T_K: float = 5.0
    reference_salinity_ppt: float = 0.0
    reference_consolidation_s: float = 3600.0

    def specimen_config(self, *, seed: int | None = None) -> SpecimenConfig:
        return SpecimenConfig(
            n_x=self.n_x, n_y=self.n_y, radius_m=self.radius_m,
            seed=self.bond_strength_seed if seed is None else seed,
            bond_fraction=self.bond_fraction,
            contact_friction=self.contact_friction,
            wall_friction=self.wall_friction,
        )

    def strength_kappa(self, kappa: float) -> float:
        """Freeze-bond tensile strength for a strength ratio ``kappa`` [Pa]."""
        return float(kappa) * self.bulk_strength_pa


DEFAULT_CAMPAIGN = CampaignConfig()


def _environment(n_bonds: int, strength_pa: float, cfg: CampaignConfig,
                 *,
                 law: FreezeBondLaw = FREEZE_BOND_LAW_REFERENCE,
                 cv: float | None = None, seed: int | None = None) -> BondEnvironment:
    """Bond freezing state producing ``strength_pa`` with a declared scatter.

    The scatter is applied to the *temperature drop* (the physical driver that
    varies most between bonds), with a lognormal factor of coefficient of
    variation ``cv``; ``cv = 0`` and ``strength = 0`` give a completely
    unbonded (purely frictional) specimen.
    """
    if strength_pa <= 0.0:
        return BondEnvironment.uniform(n_bonds, delta_T_K=0.0,
                                       salinity_ppt=cfg.reference_salinity_ppt,
                                       consolidation_time_s=cfg.reference_consolidation_s)
    sigma_ref = float(law.tension_strength(cfg.reference_delta_T_K,
                                          cfg.reference_salinity_ppt,
                                          cfg.reference_consolidation_s))
    if sigma_ref <= 0.0:
        raise ValueError("reference law gives zero strength at the reference state")
    # sigma ~ Delta T ** alpha, so the required temperature drop follows from the
    # requested strength ratio.
    scale = (strength_pa / sigma_ref) ** (1.0 / law.temperature_exponent)
    scatter_cv = cfg.bond_strength_cv if cv is None else float(cv)
    rng = np.random.default_rng(cfg.bond_strength_seed if seed is None else seed)
    factor = np.ones(n_bonds)
    if scatter_cv > 0.0:
        sigma_ln = np.sqrt(np.log1p(scatter_cv ** 2))
        factor = np.exp(rng.normal(0.0, sigma_ln, size=n_bonds) - 0.5 * sigma_ln ** 2)
    return BondEnvironment(
        delta_T_K=cfg.reference_delta_T_K * scale * factor,
        salinity_ppt=np.full(n_bonds, cfg.reference_salinity_ppt),
        consolidation_time_s=np.full(n_bonds, cfg.reference_consolidation_s),
    )


def specimen_for_kappa(
    kappa: float,
    *,
    campaign: CampaignConfig = DEFAULT_CAMPAIGN,
    law: FreezeBondLaw = FREEZE_BOND_LAW_REFERENCE,
    geometry: str = "channel",
    cv: float | None = None,
    seed: int | None = None,
) -> BondedSpecimen:
    """Build a channel (or box) specimen with freeze bonds of strength ``kappa``.

    ``kappa = sigma_fb(Delta T_ref, S_ref, t_ref) / sigma_bulk``.  ``kappa = 0``
    sets ``Delta T = 0``, i.e. no freezing at all: the specimen is the purely
    frictional reference used for the "no freeze bond" arm of the campaign.
    """
    cfg = campaign.specimen_config(seed=seed)
    builder = build_channel_specimen if geometry == "channel" else build_box_specimen
    dummy = builder(law, config=cfg)
    env = _environment(dummy.topology.n_bonds, campaign.strength_kappa(kappa),
                       campaign, law=law, cv=cv, seed=seed)
    spec = builder(law, config=cfg, environment=env)
    # Rescale the law so that the reference strength is exactly kappa * sigma_bulk
    # (the scatter is already in the environment, so the law keeps A at the
    # reference anchor).
    return spec


# --------------------------------------------------- (2) single-bond element
def single_bond_self_consistency(
    *,
    law: FreezeBondLaw = FREEZE_BOND_LAW_REFERENCE,
    stiffness_pa_per_m: float = 1.5e9,
    delta_T_grid: Iterable[float] = (1.0, 2.0, 5.0, 10.0),
    salinity_grid: Iterable[float] = (0.0, 5.0, 20.0),
    time_grid: Iterable[float] = (600.0, 3600.0, 86400.0),
) -> dict[str, Any]:
    """Peak traction of the bonded element vs the analytic power law.

    The element is loaded in 400 monotone steps beyond its peak and its maximum
    traction is compared with ``law.tension_strength`` at the same
    ``(Delta T, S, t)``.  A second pass checks the *shear* branch against
    ``kappa_s * sigma_fb``.  Returns per-point relative errors and the maximum.
    """
    rows: list[dict[str, float]] = []
    worst = 0.0
    for dT in delta_T_grid:
        for s in salinity_grid:
            for t in time_grid:
                env = BondEnvironment.uniform(1, delta_T_K=dT, salinity_ppt=s,
                                             consolidation_time_s=t)
                element = CohesiveBondSet(law, env, stiffness_pa_per_m=stiffness_pa_per_m)
                sig_analytic = float(law.tension_strength(dT, s, t))
                d0 = sig_analytic / stiffness_pa_per_m
                span = max(20.0 * d0, 1e-12)
                peak = 0.0
                for delta in np.union1d(np.linspace(0.0, span, 400), [d0]):
                    state = element.trial(np.array([delta]), np.array([0.0]))
                    peak = max(peak, float(state["traction"][0]))
                    element.commit(state)
                # shear branch
                sig_shear = float(law.shear_strength(dT, s, t))
                element_s = CohesiveBondSet(law, env, stiffness_pa_per_m=stiffness_pa_per_m)
                d0s = sig_shear / stiffness_pa_per_m
                peak_s = 0.0
                for delta in np.union1d(np.linspace(0.0, span, 400),
                                        [d0s] if d0s > 0 else []):
                    state = element_s.trial(np.array([0.0]), np.array([delta]))
                    peak_s = max(peak_s, float(state["traction"][0]))
                    element_s.commit(state)
                err = abs(peak - sig_analytic) / sig_analytic if sig_analytic else 0.0
                err_s = abs(peak_s - sig_shear) / sig_shear if sig_shear else 0.0
                worst = max(worst, err, err_s)
                rows.append({
                    "delta_T_K": float(dT), "salinity_ppt": float(s),
                    "consolidation_s": float(t),
                    "sigma_analytic_pa": sig_analytic,
                    "sigma_element_pa": peak,
                    "rel_error": err,
                    "sigma_shear_analytic_pa": sig_shear,
                    "sigma_shear_element_pa": peak_s,
                    "rel_error_shear": err_s,
                })
    return {
        "n_points": len(rows),
        "max_rel_error": worst,
        "law": law.as_dict(),
        "stiffness_pa_per_m": stiffness_pa_per_m,
        "rows": rows,
        "verdict": "self-consistent" if worst < 1e-9 else "INCONSISTENT",
    }


# ------------------------------------------------------- (3) energy ledger
def energy_ledger_case(
    *,
    kappa: float = 0.02,
    campaign: CampaignConfig = DEFAULT_CAMPAIGN,
    n_steps: int = 40,
    geometry: str = "channel",
    cv: float | None = 0.0,
) -> dict[str, Any]:
    """Run one loading case and report the closed energy ledger."""
    spec = specimen_for_kappa(kappa, campaign=campaign, geometry=geometry, cv=cv)
    result = load_steps(spec, total_displacement_m=campaign.total_displacement_m,
                        n_steps=n_steps)
    return {
        "kappa": float(kappa),
        "geometry": geometry,
        "n_bonds": spec.topology.n_bonds,
        "ledger": result.ledger,
        "final": {
            "platen_force_N": result.steps[-1].platen_force_N,
            "load_cell_force_N": result.steps[-1].load_cell_force_N,
            "broken_bonds": result.steps[-1].broken_bonds,
            "mean_damage": result.steps[-1].mean_damage,
        },
        "result": result,
    }


# -------------------------------------------------------- (4) channel arch
def _network(spec: BondedSpecimen, step, *, k_strong: float = 1.0,
             span_fraction: float = 0.5) -> ForceNetwork:
    data = step.contact_forces
    edges = np.concatenate([data["bond_pairs"], data["contact_pairs"]], axis=0)
    magnitudes = np.concatenate([data["bond_magnitudes"],
                                 data["contact_magnitudes"]], axis=0)
    radius = float(np.mean(spec.radii))
    y = data["positions"][:, 1]
    tag = np.zeros(spec.n_particles, dtype=int)
    tag[y <= y.min() + 1.2 * radius] = -1
    tag[y >= y.max() - 1.2 * radius] = 1
    return analyze_network(data["positions"], edges, magnitudes, k_strong=k_strong,
                           width=spec.height_m, transverse_axis=1,
                           span_fraction=span_fraction, wall_tag=tag)


def channel_arch_run(
    kappa: float,
    *,
    campaign: CampaignConfig = DEFAULT_CAMPAIGN,
    k_strong: float = 1.0,
    span_fraction: float = 0.5,
    cv: float | None = None,
    seed: int | None = None,
    n_steps: int | None = None,
    total_displacement_m: float | None = None,
) -> dict[str, Any]:
    """Channel push at a given strength ratio; per-step arch observables."""
    spec = specimen_for_kappa(kappa, campaign=campaign, geometry="channel",
                              cv=cv, seed=seed)
    result = load_steps(
        spec,
        total_displacement_m=(campaign.total_displacement_m
                              if total_displacement_m is None else total_displacement_m),
        n_steps=campaign.n_steps if n_steps is None else n_steps)
    rows = []
    for step in result.steps:
        net = _network(spec, step, k_strong=k_strong, span_fraction=span_fraction)
        transmission = (step.load_cell_force_N / step.platen_force_N
                        if step.platen_force_N else 0.0)
        rows.append({
            "step": step.step,
            "platen_force_N": step.platen_force_N,
            "load_cell_force_N": step.load_cell_force_N,
            "lateral_force_N": step.lateral_force_N,
            "transmission": float(transmission),
            "arch_count": net.cycle_rank,
            "n_spanning_loops": net.n_spanning_loops,
            "max_span_ratio": net.max_span_ratio,
            "spanning": bool(net.spanning),
            "strong_edges": net.n_strong_edges,
            "broken_bonds": step.broken_bonds,
            "mean_damage": step.mean_damage,
        })
    final = rows[-1]
    return {
        "kappa": float(kappa),
        "n_bonds": spec.topology.n_bonds,
        "channel_height_m": spec.height_m,
        "channel_width_m": spec.width_m,
        "rows": rows,
        "final": final,
        "ledger": result.ledger,
        "result": result,
        "specimen": spec,
    }


# ------------------------------------------------------------- (4b) kappa_c
def bilinear_breakpoint(x: np.ndarray, y: np.ndarray,
                        *, min_points: int = 2) -> dict[str, Any]:
    """Least-squares two-branch fit ``y = a1 + b1 x`` / ``a2 + b2 x``.

    Returns the interior breakpoint that minimises the total squared error, the
    two slopes, and the slope change ``b2 - b1`` (the "jump" that a threshold is
    judged on).  Degenerate inputs return ``breakpoint = None`` instead of a
    fabricated number.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique = np.unique(x)
    if unique.size < 3 or x.size < 2 * min_points:
        return {"breakpoint": None, "slope_left": None, "slope_right": None,
                "slope_change": None, "r2": None,
                "reason": "fewer than 3 distinct abscissae"}
    best: dict[str, Any] | None = None
    for i in range(min_points, x.size - min_points + 1):
        left, right = slice(0, i), slice(i, x.size)
        if x[left].max() >= x[right].min():
            continue
        p1 = np.polyfit(x[left], y[left], 1)
        p2 = np.polyfit(x[right], y[right], 1)
        pred = np.concatenate([np.polyval(p1, x[left]), np.polyval(p2, x[right])])
        sse = float(np.sum((pred - y) ** 2))
        if best is None or sse < best["sse"]:
            best = {
                "sse": sse,
                "breakpoint_index": i,
                "breakpoint": float(0.5 * (x[i - 1] + x[i])),
                "slope_left": float(p1[0]),
                "slope_right": float(p2[0]),
                "slope_change": float(p2[0] - p1[0]),
            }
    total = float(np.sum((y - np.mean(y)) ** 2))
    best["r2"] = float(1.0 - best["sse"] / total) if total > 0 else None
    # Is the two-branch description actually better than one straight line?
    p1 = np.polyfit(x, y, 1)
    sse_single = float(np.sum((np.polyval(p1, x) - y) ** 2))
    best["sse_single_branch"] = sse_single
    best["sse_reduction"] = (float(1.0 - best["sse"] / sse_single)
                             if sse_single > 0 else 0.0)
    # The threshold must also be *visible*: either a level jump at the breakpoint
    # or a slope change, both relative to the data range.  A straight line then
    # returns ``breakpoint = None`` instead of a spurious split.
    i = best["breakpoint_index"]
    p1 = np.polyfit(x[:i], y[:i], 1)
    p2 = np.polyfit(x[i:], y[i:], 1)
    x_b = best["breakpoint"]
    span = float(np.max(y) - np.min(y)) if np.ptp(y) > 0 else 1.0
    best["level_jump"] = float(abs(np.polyval(p2, x_b) - np.polyval(p1, x_b)) / span)
    best["slope_change_relative"] = float(
        abs(best["slope_change"]) / max(abs(p1[0]), abs(p2[0]), 1e-300))
    visible = best["level_jump"] > 0.05 or best["slope_change_relative"] > 0.25
    best["threshold_detected"] = bool(
        best["sse"] < 0.5 * sse_single and visible)
    if not best["threshold_detected"]:
        best["breakpoint"] = None
    return best


def kappa_sweep(
    kappas: Iterable[float],
    *,
    campaign: CampaignConfig = DEFAULT_CAMPAIGN,
    k_strong: float = 1.0,
    span_fraction: float = 0.5,
    cv: float | None = None,
) -> dict[str, Any]:
    """Strength-ratio sweep; returns per-``kappa`` observables and two ``kappa_c`` s."""
    runs = [channel_arch_run(float(kappa), campaign=campaign, k_strong=k_strong,
                             span_fraction=span_fraction, cv=cv)
            for kappa in kappas]
    kappa_arr = np.array([r["kappa"] for r in runs])
    arch = np.array([r["final"]["arch_count"] for r in runs], dtype=float)
    damage = np.array([r["final"]["mean_damage"] for r in runs], dtype=float)
    broken = np.array([r["final"]["broken_bonds"] for r in runs], dtype=float)
    n_bonds = float(np.mean([r["n_bonds"] for r in runs]))
    spanning = np.array([r["final"]["n_spanning_loops"] for r in runs], dtype=float)
    transmission = np.array([r["final"]["transmission"] for r in runs])
    fit_arch = bilinear_breakpoint(kappa_arr, arch)
    fit_transmission = bilinear_breakpoint(kappa_arr, transmission)
    fit_damage = bilinear_breakpoint(kappa_arr, damage)
    return {
        "policy": "two independent judgements (arch count and load transmission); "
                  "no merged or tuned threshold",
        "kappas": kappa_arr.tolist(),
        "arch_count": arch.tolist(),
        "spanning_loops": spanning.tolist(),
        "transmission": transmission.tolist(),
        "platen_force_N": [r["final"]["platen_force_N"] for r in runs],
        "broken_bonds": [r["final"]["broken_bonds"] for r in runs],
        "mean_damage": damage.tolist(),
        "broken_bond_fraction": (broken / n_bonds).tolist(),
        "kappa_c_arch": fit_arch.get("breakpoint"),
        "kappa_c_transmission": fit_transmission.get("breakpoint"),
        "kappa_c_damage": fit_damage.get("breakpoint"),
        "fit_arch": fit_arch,
        "fit_transmission": fit_transmission,
        "fit_damage": fit_damage,
        "runs": runs,
    }


# --------------------------------------------------------------- self-sustain
def self_sustain(
    kappa: float,
    *,
    campaign: CampaignConfig = DEFAULT_CAMPAIGN,
    hold_relax_steps: int = 3,
    refreeze_time_s: float = 86400.0,
    cv: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Does an arch survive relaxation, and can a broken arch re-freeze?

    Protocol:

    1. load the channel to the campaign displacement (arch forms if
       ``kappa > kappa_c``);
    2. **hold** the platen and re-equilibrate: an arch is self-sustaining if it
       survives relaxation, keeps a positive platen reaction and keeps carrying
       the thrust to the banks (``transmission`` stays low);
    3. **over-drive** the platen by another 60 % of the campaign displacement to
       destroy the arch (bonds break);
    4. **re-freeze**: the broken bonds get a fresh neck with a consolidation time
       of ``refreeze_time_s`` (:meth:`CohesiveBondSet.refreeze`), the platen is
       held and the assembly relaxed again -- does the arch reform?

    The three numbers reported are ``transmission``, ``arch_count`` and
    ``platen_force`` at stages 2, 3 and 4; the "self-sustaining" verdict requires
    stage 2 to keep a spanning arch, and the "re-freezing" verdict requires stage
    4 to recover a spanning arch after stage 3 lost it.
    """
    run = channel_arch_run(kappa, campaign=campaign, cv=cv, seed=seed)
    spec: BondedSpecimen = run["specimen"]
    result: LoadingResult = run["result"]
    friction = result.friction
    f_scale = max(1.0, abs(result.steps[-1].platen_force_N))
    stages: list[dict[str, Any]] = []

    def _record(name: str, pos: np.ndarray, steps: int) -> dict[str, Any]:
        pos, info, iters, f_max = newton_solve(spec, pos, friction,
                                               f_scale=f_scale, tol_rel=1e-5,
                                               max_iter=steps)
        spec.positions = pos.copy()
        dn, dt = bond_separations(spec, pos)
        trial = spec.cohesive.trial(dn, dt)
        spec.cohesive.commit(trial)
        platen = float(info["wall_normal_forces"].get("platen", 0.0))
        load_cell = float(info["wall_normal_forces"].get("load_cell", 0.0))
        lateral = float(info["wall_normal_forces"].get("wall_top", 0.0)
                        + info["wall_normal_forces"].get("wall_bottom", 0.0))
        edges = np.concatenate([np.stack([spec.topology.i, spec.topology.j], axis=1),
                                np.asarray(info["contact_edges"]).reshape(-1, 2)], axis=0)
        magnitudes = np.concatenate([trial["traction"] * spec.topology.area,
                                     np.asarray(info["contact_magnitudes"]), ], axis=0)
        radius = float(np.mean(spec.radii))
        tag = np.zeros(spec.n_particles, dtype=int)
        tag[pos[:, 1] <= pos[:, 1].min() + 1.2 * radius] = -1
        tag[pos[:, 1] >= pos[:, 1].max() - 1.2 * radius] = 1
        net = analyze_network(pos, edges, magnitudes, width=spec.height_m,
                              transverse_axis=1, wall_tag=tag)
        record = {
            "stage": name,
            "platen_force_N": platen,
            "load_cell_force_N": load_cell,
            "lateral_force_N": lateral,
            "transmission": float(load_cell / platen) if platen else 0.0,
            "arch_count": net.cycle_rank,
            "n_spanning_loops": net.n_spanning_loops,
            "spanning": bool(net.spanning),
            "broken_bonds": int(np.count_nonzero(trial["broken"])),
            "mean_damage": float(np.mean(trial["damage"])),
            "newton_iterations": int(iters),
            "residual_N": float(f_max),
        }
        stages.append(record)
        return record

    _record("loaded", spec.positions.copy(), 40)
    _record("held_relaxed", spec.positions.copy(), int(hold_relax_steps) * 20)

    # stage 3: over-drive the platen until the arch is damaged.  The escalation is
    # explicit in the report (``overdrive_factor``) so the reader can see how hard
    # the arch had to be pushed before it broke.
    platen_wall = spec.wall_by_name("platen")
    broken_before = int(np.count_nonzero(spec.cohesive.was_broken))
    overdrive = 0.0
    sub = {"positions": spec.positions.copy()}
    for attempt in range(1, 5):
        extra = 0.6 * attempt * campaign.total_displacement_m
        sub = load_steps_from(spec, platen_wall, friction,
                              extra - overdrive,
                              n_steps=max(4, campaign.n_steps // 3), f_scale=f_scale)
        overdrive = extra
        spec.positions = sub["positions"]
        dn, dt = bond_separations(spec, spec.positions)
        trial_now = spec.cohesive.trial(dn, dt)
        broken_now = int(np.count_nonzero(trial_now["broken"]))
        if broken_now - broken_before >= 3:
            break
    record_over = _record("overdriven", sub["positions"], 40)
    record_over["overdrive_displacement_m"] = float(overdrive)

    # stage 4: re-freeze the broken bonds and relax again
    # Re-freeze every bond that reached complete failure: the damage history
    # variable of a broken bond is at (or beyond) its softening length.
    healed = spec.cohesive.refreeze(spec.cohesive.was_broken.copy(), refreeze_time_s)
    record = _record("refrozen", sub["positions"], 60)
    record["healed_bonds"] = int(healed)

    sustained = bool(stages[1]["spanning"] and stages[1]["platen_force_N"] > 0.0)
    reformed = bool(stages[3]["spanning"] and not stages[2]["spanning"])
    return {
        "kappa": float(kappa),
        "stages": stages,
        "self_sustaining": sustained,
        "arch_lost_on_overdrive": not stages[2]["spanning"],
        "reformed_after_refreeze": reformed,
        "refreeze_time_s": float(refreeze_time_s),
        "verdict": _sustain_verdict(sustained, stages, reformed),
    }


def _sustain_verdict(sustained: bool, stages: list[dict[str, Any]], reformed: bool) -> str:
    if sustained and reformed:
        return "arch survives relaxation and re-forms after re-freezing"
    if sustained:
        return ("arch survives relaxation; no break-then-re-freeze cycle observed "
                "at this strength ratio")
    return "no self-sustaining arch at this strength ratio (honest negative)"


def load_steps_from(spec: BondedSpecimen, wall, friction, extra_m: float,
                    *, n_steps: int = 10, f_scale: float = 1.0) -> dict[str, Any]:
    """Minimal continuation loader used by :func:`self_sustain` stage 3."""
    from .assembly import commit_friction

    du = extra_m / n_steps
    for _ in range(n_steps):
        wall.point = wall.point + du * wall.normal
        pos_old = spec.positions.copy()
        pos, info, iters, f_max = newton_solve(spec, pos_old, friction,
                                               f_scale=f_scale, tol_rel=1e-5,
                                               max_iter=40)
        dn, dt = bond_separations(spec, pos)
        spec.cohesive.commit(spec.cohesive.trial(dn, dt))
        commit_friction(spec, pos, friction)
        spec.positions = pos.copy()
    return {"positions": spec.positions.copy(), "info": info}
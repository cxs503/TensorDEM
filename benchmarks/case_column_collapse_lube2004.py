"""CASE-COLUMN-COLLAPSE — axisymmetric granular-column collapse vs Lube et al. (2004).

算例池 case card (owner 2026-11-12).  Reference experiment
--------------------------------------------------------
Lube, Huppert, Sparks & Freundt (2004), *Axisymmetric collapses of granular
columns*, J. Fluid Mech. **508**, 175-199, doi:10.1017/S0022112004009036.
Targets (paper abstract, quoted verbatim in
:mod:`tensordem.granular_column.axisym`):

* ``r_inf = r_i (1 + 1.24 a)``            for ``a < 1.7``
* ``r_inf = r_i (1 + 1.6 a^(1/2))``       for ``1.7 < a``
* ``t_inf = 3 (h_i/g)^(1/2) = 3 (r_i/g)^(1/2) a^(1/2)``

**evaluated from the formula at run time** (:func:`lube_2004_runout_ratio`,
:func:`lube_2004_emplacement_time`) — no reference number is typed by hand.

Usage
-----
::

    python benchmarks/case_column_collapse_lube2004.py --list
    python benchmarks/case_column_collapse_lube2004.py --wave1        # control + calibration
    python benchmarks/case_column_collapse_lube2004.py --shard 2/6    # wave-2 worker
    python benchmarks/case_column_collapse_lube2004.py --collect      # verdicts

``--case ID`` runs a single pre-declared case id (wave-1 or wave-2, at
``--mu-rolling``); ``--list-cases`` prints every id — this is how cluster
shards fill the grid point by point.  ``--wave1`` runs the negative control
(``mu_r = 0``, no rolling resistance) and
the one-parameter calibration grid at ``a = 0.5``; ``--shard i/n`` runs one
worker of the wave-2 grid (validation + sensitivity) at a given ``mu_r``
(``--mu-rolling``, read off the calibration by ``--collect``).
``--collect`` evaluates the predeclared criteria (< 15 % relative error on the
runout radius ratio and on the emplacement time) over every stored case and
writes ``summary.json`` + ``cases.csv``.  Nothing is recomputed from a
hard-coded number, so the per-case JSONs are the provenance.

Predeclared protocol (fixed before the runs, see
``docs/case_column_collapse_lube2004.md``)
------------------------------------------
control     ``a = 1.0``, ``mu_rolling = 0`` — demonstrates that sphere DEM
            without rolling resistance never arrests (degenerate deposit);
calibration ``a = 0.5``, ``mu_rolling in {0.2, 0.3, 0.4}`` — **one** grain-shape
            parameter, fitted on the smallest aspect ratio only;
validation  ``a in {0.5, 1.0, 1.5, 2.0, 3.0}`` at the calibrated
            ``mu_rolling`` — out-of-sample for ``a != 0.5``;
sensitivity ``a = 2.0``: ``grains_per_radius = 14``, ``dt/2``, ``mu = 0.3``,
            ``mu = 0.8`` (all at the calibrated ``mu_rolling``).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.granular_column.inertial_friction import JOP_2006
from tensordem.granular_column.axisym import (LUBE2004_DOI, LUBE2004_SOURCE,  # noqa: E402
                                             AxisymCase, lube_2004_emplacement_time,
                                             lube_2004_runout_ratio, verdict,
                                             simulate_axisymmetric_column)

OUT_DIR = os.path.join("runs", "case_column_collapse_lube2004")
CASE_DIR = os.path.join(OUT_DIR, "cases")
TOLERANCE = 0.15

BASE = dict(radius=0.03, density=2500.0, mu=0.5, mu_rolling=0.3,
            restitution=0.5, kn=6.0e4, kt_ratio=0.5, gravity=9.81,
            packing_gap=0.02, jitter=0.0, seed=12345,
            dt_scale=1.0, duration_factor=8.0, min_time=0.5,
            sample_stride=20, rest_ke_tol=1.0e-4, rest_hold=200,
            grains_per_radius=8.0)

VALIDATION_ASPECTS = (1.0, 1.5, 2.0, 3.0)

# --- round 3 (m16534): contact-friction *curve* grid -----------------------
# Round 2 established that mu is the dominant lever at a = 2 (mu = 0.8 gave
# +0.03 % while mu = 0.5 gave +22.8 %), but that is a single grid point, so it
# cannot be distinguished from a point-wise coincidence.  Round 3 therefore
# asks a curve-level question, pre-registered before the runs: does *one* mu
# value reproduce the Lube (2004) runout law at every aspect ratio
# a in SWEEP_ASPECTS simultaneously (< 15 % each) at the round-1 rolling
# resistance mu_r = 0.4?  Nothing else is varied: the same discretisation,
# dt, restitution and energy ledger as the round-1/2 validation points, and
# the mu = 0.5 row is a reproduction control for rounds 1-2.
MU_SWEEP = (0.5, 0.6, 0.7, 0.8)
SWEEP_ASPECTS = (0.5, 1.0, 1.5, 2.0, 3.0)
CALIBRATION_ASPECT = 0.5
CALIBRATION_MUR = (0.2, 0.3, 0.4)


def wave1_cases() -> list[dict]:
    """Negative control + the mu_rolling calibration grid (cheap, ~3 min total)."""
    cases = [{"id": "control_mur0_a1.0", "phase": "control",
              "note": "no rolling resistance: spheres cannot arrest (negative control)",
              "params": dict(BASE, aspect_ratio=1.0, mu_rolling=0.0)}]
    for mur in CALIBRATION_MUR:
        cases.append({"id": f"calib_mur{mur:g}_a{CALIBRATION_ASPECT:g}",
                      "phase": "calibration",
                      "note": f"rolling resistance mu_r = {mur:g} at a = "
                              f"{CALIBRATION_ASPECT:g} (single fitted parameter)",
                      "params": dict(BASE, aspect_ratio=CALIBRATION_ASPECT,
                                     mu_rolling=mur)})
    return cases


def wave2_cases(mu_rolling: float) -> list[dict]:
    """Validation scan + sensitivity grid at the calibrated mu_rolling."""
    cases = [{"id": f"valid_mur{mu_rolling:g}_a{a:g}", "phase": "validation",
              "note": f"mu_r = {mu_rolling:g} (calibrated at a = 0.5)",
              "params": dict(BASE, aspect_ratio=a, mu_rolling=mu_rolling)}
             for a in VALIDATION_ASPECTS]
    sens = (
        ("sens_dt0.5_a2.0", dict(aspect_ratio=2.0, dt_scale=0.5),
         "time step halved"),
        ("sens_gpr6_a2.0", dict(aspect_ratio=2.0, grains_per_radius=6.0),
         "particle resolution: 6 grains per column radius (coarser)"),
        ("sens_mu0.3_a2.0", dict(aspect_ratio=2.0, mu=0.3),
         "contact friction mu = 0.3"),
        ("sens_gpr11_a2.0", dict(aspect_ratio=2.0, grains_per_radius=11.0),
         "particle resolution: 11 grains per column radius"),
        ("sens_gpr14_a2.0", dict(aspect_ratio=2.0, grains_per_radius=14.0),
         "particle resolution: 14 grains per column radius "
         "(the pre-registered ladder point, see module docstring)"),
        ("sens_mu0.8_a2.0", dict(aspect_ratio=2.0, mu=0.8),
         "contact friction mu = 0.8 (pre-registered axis)"),
    )
    for cid, params, note in sens:
        cases.append({"id": cid + ("" if mu_rolling == BASE["mu_rolling"]
                                   else f"_mur{mu_rolling:g}"),
                      "phase": "sensitivity", "note": note,
                      "params": dict(BASE, **params, mu_rolling=mu_rolling)})
    return cases


def wave3_cases(mu_rolling: float) -> list[dict]:
    """Contact-friction curve grid: ``mu x a`` at the calibrated ``mu_rolling``.

    Pre-registered (see ``MU_SWEEP`` / ``SWEEP_ASPECTS``): the hypothesis under
    test is that one ``mu`` reproduces the reference runout law over the whole
    aspect-ratio range, i.e. curve-level rather than point-level agreement.
    """
    suffix = "" if mu_rolling == BASE["mu_rolling"] else f"_mur{mu_rolling:g}"
    return [{"id": f"sweep_mu{mu:g}_a{a:g}" + suffix, "phase": "sweep_mu",
             "note": f"contact friction mu = {mu:g} at a = {a:g} "
                     f"(mu_r = {mu_rolling:g} fixed)",
             "params": dict(BASE, aspect_ratio=a, mu=mu, mu_rolling=mu_rolling)}
            for mu in MU_SWEEP for a in SWEEP_ASPECTS]


def wave3_control_cases(mu_rolling: float) -> list[dict]:
    """**Post-hoc** controls for round 3 (added *after* inspecting the sweep,
    labelled as such): the rounds-1/2 duration factor ``duration_factor = 8``
    was inherited unchanged, but two sweep points (mu = 0.7, a = 0.5 and
    a = 1.0) terminated on ``stop_reason = time_limit`` instead of ``at_rest``,
    i.e. their runout could in principle be truncated.  Here the duration is
    doubled (16) at unchanged everything else, so a converged runout shows up
    as an unchanged ratio.  These two points do not enter the pre-registered
    verdict; they only test whether that verdict is truncation-limited.
    """
    suffix = "" if mu_rolling == BASE["mu_rolling"] else f"_mur{mu_rolling:g}"
    return [{"id": f"ctrl_dur16_mu{mu:g}_a{a:g}" + suffix, "phase": "sweep_mu_control",
             "note": f"post-hoc truncation control: duration_factor = 16, "
                     f"mu = {mu:g}, a = {a:g} (mu_r = {mu_rolling:g})",
             "params": dict(BASE, aspect_ratio=a, mu=mu, mu_rolling=mu_rolling,
                            duration_factor=16.0)}
            for mu, a in ((0.7, 0.5), (0.7, 1.0))]


# --- round 4 (m16862): joint (mu, mu_r) identifiability --------------------
# Round 3 left the anchor open with a *structural* reading: at the round-1
# rolling resistance (mu_r = 0.4) the linear branch (a <= 1.5) wants
# mu = 0.53-0.57 while the square-root branch (a >= 2) wants mu = 0.80-0.84, so
# no single contact friction covers the curve.  That statement is conditional
# on mu_r = 0.4, which round 1 *fitted* at a = 0.5; mu_r was never varied
# afterwards.  Round 4 therefore asks the joint question, pre-registered before
# any wave-4 run:
#
#   H4: there exists ONE constant pair (mu, mu_r) that reproduces the Lube
#       (2004) runout ratio within 15 % at every aspect ratio
#       a in SWEEP_ASPECTS = {0.5, 1.0, 1.5, 2.0, 3.0}.
#
# Decision rule (fixed here, before the runs): probe only the two branch
# representatives a = 1.0 (linear, a < 1.7) and a = 3.0 (square root, a > 1.7)
# on the WAVE4_MU x WAVE4_MUR grid; for every (branch, mu_r) find the required
# contact friction mu_req by linear interpolation of the *signed* runout error
# through zero.  H4 is refuted iff |mu_req(sqrt, mu_r) - mu_req(linear, mu_r)|
# does not pass through zero on the grid (i.e. no grid interval changes sign);
# otherwise the interpolated crossing (mu, mu_r) is the pre-registered fit and
# is checked on all five aspects (the a in {0.5, 1.5, 2.0} rows are out of
# sample for the fit).  Tolerance (15 %), discretisation, dt, restitution and
# the energy ledger are unchanged from rounds 1-3.
WAVE4_MU = (0.5, 0.6, 0.7, 0.8, 0.9)
WAVE4_MUR = (0.1, 0.2, 0.3, 0.4, 0.5)
PROBE_ASPECTS = (1.0, 3.0)


def wave4_cases() -> list[dict]:
    """Joint ``(mu, mu_r)`` probe grid on the two branch representatives.

    Interleaved order (mu_r outermost) so that ``--shard i/n`` keeps every
    shard balanced and every shard still contains both probe aspects.
    """
    return [{"id": f"probe_mu{mu:g}_mur{mur:g}_a{a:g}", "phase": "probe_mu_mur",
             "note": f"joint (mu, mu_r) probe: mu = {mu:g}, mu_r = {mur:g}, "
                     f"a = {a:g} ({'linear' if a < 1.7 else 'square-root'} branch)",
             "params": dict(BASE, aspect_ratio=a, mu=mu, mu_rolling=mur)}
            for mur in WAVE4_MUR for mu in WAVE4_MU for a in PROBE_ASPECTS]


def wave4_validation_cases(mu: float, mu_rolling: float) -> list[dict]:
    """The pre-registered rule's chosen ``(mu, mu_r)`` checked on every aspect.

    ``a in {0.5, 1.5, 2.0}`` were **not** used to pick the pair, so they are
    out-of-sample confirmations; ``a in {1.0, 3.0}`` reproduce the probe rows
    (determinism control).
    """
    return [{"id": f"wave4_valid_mu{mu:g}_mur{mu_rolling:g}_a{a:g}",
             "phase": "wave4_validation",
             "note": f"curve-level check of the pre-registered pair "
                     f"(mu = {mu:g}, mu_r = {mu_rolling:g}) at a = {a:g}",
             "params": dict(BASE, aspect_ratio=a, mu=mu, mu_rolling=mu_rolling)}
            for a in SWEEP_ASPECTS]


# --- round 5 (m17111): rate-dependent mu(I) as the single closing law -------
# Round 4 left the anchor open with a *rate* reading: at a common rolling
# resistance the linear branch wants mu ~ 0.51 and the square-root branch
# mu ~ 0.80 (docs/case_column_collapse_round4_m16862.md), i.e. the demanded
# friction grows with the aspect ratio, which is what the inertial number of
# the flow grows with too.  Dry granular friction is known to be rate dependent
# through mu(I) = mu_s + (mu_2 - mu_s) I/(I_0 + I) (Jop, Forterre & Pouliquen
# 2006, Nature 441:727-730, doi:10.1038/nature04801; da Cruz et al. 2005 PRE
# 72:011305).  Round 5 therefore asks, pre-registered before the runs:
#
#   H5: with the mu(I_c) closure of tensordem.granular_column.inertial_friction
#       and the Jop et al. (2006) parameters (mu_s, mu_2, I_0) = (0.36, 0.84,
#       0.30) -- no parameter fitted to the runout data -- ONE contact law
#       reproduces the Lube (2004) runout ratio within 15 % at every aspect
#       ratio a in SWEEP_ASPECTS = {0.5, 1.0, 1.5, 2.0, 3.0} at a fixed rolling
#       resistance.
#
# Controlled comparison: the constant-mu arm at the same mu_rolling (mu = 0.5,
# mu_r = 0.5, a wave-4 grid point reproduced here as a determinism control)
# versus the mu(I) arm, everything else identical (discretisation, dt,
# restitution, duration, energy ledger).  Sensitivity axes pre-declared: the
# rolling resistance (mu_r = 0.4, the round-1 calibrated value) and the
# discretisation (12 grains per column radius).  Tolerance 15 % unchanged.
WAVE5_MUR_PRIMARY = 0.5
WAVE5_MUR_SECONDARY = 0.4
WAVE5_RESOLUTION_GPR = 12.0
WAVE5_RESOLUTION_ASPECTS = (1.0, 3.0)


def _law_params() -> dict:
    return (float(JOP_2006["mu_s"]), float(JOP_2006["mu_2"]),
            float(JOP_2006["i_0"]))


def wave5_constant_cases(mu_rolling: float = WAVE5_MUR_PRIMARY) -> list[dict]:
    """Constant-friction control arm at the round-5 rolling resistance."""
    tag = f"mur{mu_rolling:g}"
    return [{"id": f"w5_const_{tag}_a{a:g}", "phase": "wave5_constant",
             "note": f"constant-mu control arm: mu = {BASE['mu']:g}, "
                     f"mu_r = {mu_rolling:g}, a = {a:g}",
             "params": dict(BASE, aspect_ratio=a, mu=BASE["mu"],
                            mu_rolling=mu_rolling)}
            for a in SWEEP_ASPECTS]


def wave5_law_cases(mu_rolling: float = WAVE5_MUR_PRIMARY) -> list[dict]:
    """The pre-registered mu(I_c) arm over the whole aspect-ratio grid."""
    tag = f"mur{mu_rolling:g}"
    return [{"id": f"w5_law_{tag}_a{a:g}", "phase": "wave5_law_mu_i",
             "note": f"mu(I_c) Jop-2006 law {_law_params()}, mu_r = {mu_rolling:g}, "
                     f"a = {a:g}",
             "params": dict(BASE, aspect_ratio=a, mu=float(JOP_2006["mu_s"]),
                            mu_rolling=mu_rolling, mu_inertial=_law_params())}
            for a in SWEEP_ASPECTS]


def wave5_sensitivity_cases() -> list[dict]:
    """Pre-declared sensitivity axes of the law arm (not used to fit anything)."""
    out = []
    mur = WAVE5_MUR_SECONDARY
    for a in (1.0, 3.0):
        out.append({"id": f"w5_law_mur{mur:g}_a{a:g}",
                    "phase": "wave5_law_mu_i_mur_sens",
                    "note": f"mu(I_c) law at the round-1 rolling resistance "
                            f"mu_r = {mur:g} (sensitivity axis), a = {a:g}",
                    "params": dict(BASE, aspect_ratio=a, mu=float(JOP_2006["mu_s"]),
                                   mu_rolling=mur, mu_inertial=_law_params())})
    for a in WAVE5_RESOLUTION_ASPECTS:
        out.append({"id": f"w5_law_gpr{WAVE5_RESOLUTION_GPR:g}_a{a:g}",
                    "phase": "wave5_law_mu_i_resolution",
                    "note": f"mu(I_c) law at {WAVE5_RESOLUTION_GPR:g} grains per "
                            f"column radius (discretisation sensitivity), a = {a:g}",
                    "params": dict(BASE, aspect_ratio=a, mu=float(JOP_2006["mu_s"]),
                                   mu_rolling=WAVE5_MUR_PRIMARY,
                                   grains_per_radius=WAVE5_RESOLUTION_GPR,
                                   mu_inertial=_law_params())})
    return out


def wave5_cases() -> list[dict]:
    """Every round-5 case: control arm + law arm + pre-declared sensitivities."""
    return (wave5_constant_cases() + wave5_law_cases()
            + wave5_sensitivity_cases())


# --- round 5b (m17111): rough-floor boundary condition ----------------------
# The constant-friction arm of wave 5 leaves a *shape* error: runout is short at
# a = 0.5 (-4 %) and long at a = 3 (+18 %), i.e. the simulated curve is too
# steep in the square-root branch.  A friction coefficient alone cannot fix a
# shape error.  The reference experiment (Lube et al. 2004, JFM 508:175-199,
# S2) used a **glued** base, whereas our floor is a plane wall whose friction
# is the grain-grain mu -- effectively a *smoother* base than the experiment,
# and basal slip controls how far the moving layer travels (Lube et al. 2005
# JFM 544:1-14; Lajeunesse et al. 2005 Phys. Fluids 17:103302).  Round 5b
# therefore asks, pre-registered before the runs:
#
#   H6: raising the floor friction to mu_wall in {0.8, 1.0} at fixed
#       (mu, mu_r) = (0.5, 0.5) reduces the runout excess MORE (in percentage
#       points of runout) at the square-root branch (a = 3) than at the linear
#       branch (a = 1), i.e. the floor is the missing *shape* lever, so that one
#       triple (mu, mu_r, mu_wall) brings every aspect ratio within 15 %.
#
# Decision rule: probe (mu_wall = 0.8, 1.0) x (a = 1, 3) first; the hypothesis
# is refuted if the reduction at a = 3 is not larger than at a = 1, or if no
# mu_wall in {0.8, 1.0} brings BOTH branch representatives into tolerance.  Only
# a surviving hypothesis justifies the full-aspect confirmation grid below.
WAVE5B_MU_WALL = (0.8, 1.0)
WAVE5B_PROBE_ASPECTS = (1.0, 3.0)


def wave5b_probe_cases() -> list[dict]:
    """Pre-registered floor-roughness probe on the two branch representatives."""
    return [{"id": f"w5b_wall{muw:g}_a{a:g}", "phase": "wave5_floor_roughness",
             "note": f"rough-floor probe: mu_wall = {muw:g} at fixed "
                     f"mu = {BASE['mu']:g}, mu_r = {WAVE5_MUR_PRIMARY:g}, a = {a:g}",
             "params": dict(BASE, aspect_ratio=a, mu_rolling=WAVE5_MUR_PRIMARY,
                            mu_wall=muw)}
            for muw in WAVE5B_MU_WALL for a in WAVE5B_PROBE_ASPECTS]


def wave5b_confirmation_cases(mu_wall: float) -> list[dict]:
    """Full-aspect confirmation of a surviving ``mu_wall`` (out-of-sample a)."""
    return [{"id": f"w5b_confirm_wall{mu_wall:g}_a{a:g}",
             "phase": "wave5b_confirmation",
             "note": f"rough-floor confirmation: mu_wall = {mu_wall:g}, "
                     f"mu = {BASE['mu']:g}, mu_r = {WAVE5_MUR_PRIMARY:g}, a = {a:g}",
             "params": dict(BASE, aspect_ratio=a, mu_rolling=WAVE5_MUR_PRIMARY,
                            mu_wall=mu_wall)}
            for a in SWEEP_ASPECTS]


# --- round 5c (m17111): convergence controls of the surviving configuration --
# The wave-5 law arm showed a runout change of 15 % between 8 and 12 grains per
# column radius at a = 1, so the discretisation axis is *not* exhausted by the
# historical value and every claim needs it.  These controls are declared
# post-hoc (after wave 5 and 5b, before running them) and are labelled as such;
# they test the winning configuration of the pre-registered wave-5b grid
# (mu = 0.5, mu_r = 0.5, mu_wall = 0.8) along the two axes that can invalidate a
# runout verdict: the grain resolution (8 -> 12 grains per column radius) and
# the time step (dt -> dt/2).
WAVE5C_CONFIRM_MU_WALL = 0.8
WAVE5C_RESOLUTION_GPR = 12.0


def wave5c_cases() -> list[dict]:
    """Post-hoc convergence controls of the surviving (mu, mu_r, mu_wall) triple."""
    params = dict(BASE, mu_rolling=WAVE5_MUR_PRIMARY,
                  mu_wall=WAVE5C_CONFIRM_MU_WALL)
    out = []
    for a in (1.0, 3.0):
        out.append({"id": f"w5c_gpr{WAVE5C_RESOLUTION_GPR:g}_a{a:g}",
                    "phase": "wave5c_sensitivity",
                    "note": f"post-hoc resolution control: {WAVE5C_RESOLUTION_GPR:g} "
                            f"grains per column radius at mu_wall = "
                            f"{WAVE5C_CONFIRM_MU_WALL:g}, a = {a:g}",
                    "params": dict(params, aspect_ratio=a,
                                   grains_per_radius=WAVE5C_RESOLUTION_GPR)})
    out.append({"id": "w5c_dt0.5_a1", "phase": "wave5c_sensitivity",
                "note": "post-hoc time-step control: dt = 0.5 * dt_ref at "
                        f"mu_wall = {WAVE5C_CONFIRM_MU_WALL:g}, a = 1",
                "params": dict(params, aspect_ratio=1.0, dt_scale=0.5,
                               sample_stride=10)})
    out.append({"id": "w5c_dt0.5_a3", "phase": "wave5c_sensitivity",
                "note": "post-hoc time-step control: dt = 0.5 * dt_ref at "
                        f"mu_wall = {WAVE5C_CONFIRM_MU_WALL:g}, a = 3",
                "params": dict(params, aspect_ratio=3.0, dt_scale=0.5,
                               sample_stride=10)})
    return out


def all_cases(mu_rolling: float) -> list[dict]:
    """Every pre-declared case: wave 1 (control + calibration), wave 2
    (validation + sensitivity), wave 3 (mu curve grid)."""
    return (wave1_cases() + wave2_cases(mu_rolling) + wave3_cases(mu_rolling)
            + wave3_control_cases(mu_rolling) + wave4_cases() + wave5_cases()
            + wave5b_probe_cases()
            + wave5b_confirmation_cases(WAVE5B_MU_WALL[0]) + wave5c_cases())


def case_path(case_id: str) -> str:
    return os.path.join(CASE_DIR, f"{case_id}.json")


def evaluate(res: dict, case: AxisymCase) -> dict:
    """Predeclared criteria for one case (targets evaluated from the formulas)."""
    a = case.aspect_ratio
    ref_ratio = lube_2004_runout_ratio(a)
    ref_time = lube_2004_emplacement_time(a, case.column_radius, case.gravity)
    return {
        "runout_ratio": verdict(ref_ratio, res["runout_ratio_p995"], TOLERANCE),
        "runout_ratio_max_metric": verdict(ref_ratio, res["runout_ratio_max"], TOLERANCE),
        "emplacement_time": verdict(ref_time, res["t_reach"], TOLERANCE),
    }


def run_case(spec: dict, progress: bool = False) -> dict:
    t0 = time.perf_counter()
    case = AxisymCase(**spec["params"], label=spec["id"])
    res = simulate_axisymmetric_column(case, progress=progress)
    res["wall_clock_s"] = time.perf_counter() - t0
    res["phase"] = spec["phase"]
    res["note"] = spec.get("note", "")
    res["criteria"] = evaluate(res, case)
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(case_path(spec["id"]), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def _row(res: dict) -> dict:
    c = res["case"]
    crit = res["criteria"]
    return {
        "case": c["label"],
        "phase": res["phase"],
        "aspect_requested": c["aspect_ratio"],
        "aspect_measured": res["aspect_measured"],
        "n_particles": res["n_particles"],
        "dt": res["dt"],
        "grains_per_radius": c["grains_per_radius"],
        "mu": c["mu"],
        "mu_rolling": c["mu_rolling"],
        "runout_ratio_p995": res["runout_ratio_p995"],
        "runout_ratio_max": res["runout_ratio_max"],
        "ref_runout_ratio": crit["runout_ratio"]["reference"],
        "err_runout_p995": crit["runout_ratio"]["relative_error"],
        "pass_runout_p995": crit["runout_ratio"]["pass"],
        "err_runout_max": crit["runout_ratio_max_metric"]["relative_error"],
        "t_reach": res["t_reach"],
        "t90": res.get("t_reach_by_fraction", {}).get("0.9"),
        "t95": res.get("t_reach_by_fraction", {}).get("0.95"),
        "ref_t_reach": crit["emplacement_time"]["reference"],
        "err_emplacement_time": crit["emplacement_time"]["relative_error"],
        "pass_emplacement_time": crit["emplacement_time"]["pass"],
        "stop_reason": res["stop_reason"],
        "mu_law": c.get("mu_law", "constant"),
        "mu_inertial": c.get("mu_inertial"),
        "mu_eff_mean": res.get("inertial", {}).get("mu_eff_mean"),
        "inertial_number_mean": res.get("inertial", {}).get("inertial_number_mean"),
        "energy_residual": res["energy"]["residual_fraction_of_potential"],
        "wall_clock_s": res.get("wall_clock_s"),
    }


def collect() -> dict:
    rows = []
    for path in sorted(glob.glob(os.path.join(CASE_DIR, "*.json"))):
        with open(path) as fh:
            rows.append(_row(json.load(fh)))

    def group(phase):
        return [r for r in rows if r["phase"] == phase]

    def stats(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "pass_runout": sum(1 for r in rs if r["pass_runout_p995"]),
            "pass_time": sum(1 for r in rs if r["pass_emplacement_time"]),
            "max_abs_err_runout": max(abs(r["err_runout_p995"]) for r in rs),
            "max_abs_err_time": max((abs(r["err_emplacement_time"]) for r in rs
                                     if r["err_emplacement_time"] == r["err_emplacement_time"]),
                                    default=None),
        }

    calib = [r for r in group("calibration")
             if r["err_runout_p995"] == r["err_runout_p995"]]
    calibrated = min(calib, key=lambda r: abs(r["err_runout_p995"])) if calib else None
    valid = group("validation")
    out_of_sample = [r for r in valid
                     if abs(r["aspect_requested"] - CALIBRATION_ASPECT) > 1e-9]
    summary = {
        "case": "CASE-COLUMN-COLLAPSE",
        "reference": LUBE2004_SOURCE,
        "doi": LUBE2004_DOI,
        "tolerance": TOLERANCE,
        "calibration": {
            "chosen_mu_rolling": calibrated["mu_rolling"] if calibrated else None,
            "chosen_case": calibrated["case"] if calibrated else None,
            "chosen_err": calibrated["err_runout_p995"] if calibrated else None,
        },
        "phases": {
            "control": stats(group("control")),
            "calibration": stats(group("calibration")),
            "validation": stats(valid),
            "validation_out_of_sample": stats(out_of_sample),
            "sensitivity": stats(group("sensitivity")),
            "sweep_mu": stats(group("sweep_mu")),
            "sweep_mu_control": stats(group("sweep_mu_control")),
            "probe_mu_mur": stats(group("probe_mu_mur")),
            "wave4_validation": stats(group("wave4_validation")),
        },
        "rows": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT_DIR, "cases.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    return summary


def _report_row(r: dict) -> str:
    return (f"{r['case']:>24} a={r['aspect_requested']:.2f} "
            f"ratio={r['runout_ratio_p995']:.3f} ref={r['ref_runout_ratio']:.3f} "
            f"err={r['err_runout_p995'] * 100:+.1f}% "
            f"{'PASS' if r['pass_runout_p995'] else 'FAIL'} | "
            f"t99={r['t_reach']:.3f} (t90={r['t90']:.3f}) ref={r['ref_t_reach']:.3f} "
            f"err={r['err_emplacement_time'] * 100:+.1f}% "
            f"{'PASS' if r['pass_emplacement_time'] else 'FAIL'}")


def _run_specs(specs: list[dict], tag: str, progress: bool) -> None:
    for spec in specs:
        print(f"[{tag}] {spec['id']}", flush=True)
        res = run_case(spec, progress=progress)
        print(f"  n={res['n_particles']} ratio={res['runout_ratio_p995']:.3f} "
              f"ref={res['reference_runout_ratio']:.3f} t99={res['t_reach']:.3f} "
              f"stop={res['stop_reason']} wall={res['wall_clock_s']:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run one wave-1 case id")
    ap.add_argument("--wave1", action="store_true", help="control + calibration cases")
    ap.add_argument("--mu-rolling", type=float, default=BASE["mu_rolling"],
                    help="calibrated rolling resistance for the wave-2 grid")
    ap.add_argument("--shard", help="i/n: run the i-th of n interleaved wave-2 cases")
    ap.add_argument("--wave3", action="store_true",
                    help="run the round-3 mu curve grid (mu x a)")
    ap.add_argument("--wave3-control", action="store_true",
                    help="run the post-hoc truncation controls (doubled duration)")
    ap.add_argument("--wave4", action="store_true",
                    help="run the round-4 joint (mu, mu_r) probe grid")
    ap.add_argument("--wave4-shard",
                    help="i/n: run the i-th of n interleaved wave-4 probe cases")
    ap.add_argument("--wave4-validate", nargs=2, type=float, metavar=("MU", "MU_R"),
                    help="check the pre-registered (mu, mu_r) pair on all aspects")
    ap.add_argument("--wave5", action="store_true",
                    help="run the round-5 mu(I_c) law grid + control arm")
    ap.add_argument("--wave5-shard", help="i/n: run the i-th of n wave-5 cases")
    ap.add_argument("--wave5b", action="store_true",
                    help="run the round-5b rough-floor probe (mu_wall x a)")
    ap.add_argument("--wave5b-confirm", type=float, metavar="MU_WALL",
                    help="confirm a surviving mu_wall on every aspect ratio")
    ap.add_argument("--wave5c", action="store_true",
                    help="run the post-hoc convergence controls of the surviving triple")
    ap.add_argument("--collect", action="store_true", help="aggregate + verdicts")
    ap.add_argument("--case", action="append", default=[],
                    help="run one pre-declared case id (wave-1 or wave-2); repeatable")
    ap.add_argument("--list", action="store_true", help="list wave-1 case ids")
    ap.add_argument("--list-cases", action="store_true",
                    help="list every pre-declared case id (wave-1 + wave-2)")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for spec in wave1_cases():
            print(spec["id"], spec["phase"], spec["params"]["aspect_ratio"])
        return 0
    if args.list_cases:
        for spec in all_cases(args.mu_rolling):
            print(spec["id"], spec["phase"], spec["params"]["aspect_ratio"])
        return 0
    if args.collect:
        s = collect()
        print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
        for r in s["rows"]:
            print(_report_row(r))
        return 0
    if args.case:
        specs = all_cases(args.mu_rolling)
        unknown = [cid for cid in args.case if not any(s["id"] == cid for s in specs)]
        if unknown:
            print(f"unknown case id(s) {unknown}; see --list-cases", file=sys.stderr)
            return 2
        _run_specs([s for cid in args.case for s in specs if s["id"] == cid], "case",
                   args.progress)
        return 0
    if args.run:
        spec = next((s for s in wave1_cases() if s["id"] == args.run), None)
        if spec is None:
            print(f"unknown case id {args.run}", file=sys.stderr)
            return 2
        res = run_case(spec, progress=args.progress)
        print(json.dumps({k: res[k] for k in
                          ("n_particles", "dt", "n_steps", "time_end", "stop_reason",
                           "runout_ratio_p995", "reference_runout_ratio", "t_reach",
                           "reference_emplacement_time")}, indent=1))
        return 0
    if args.wave1:
        _run_specs(wave1_cases(), "wave1", args.progress)
        return 0
    if args.wave3:
        _run_specs(wave3_cases(args.mu_rolling), "wave3", args.progress)
        return 0
    if args.wave3_control:
        _run_specs(wave3_control_cases(args.mu_rolling), "wave3-control", args.progress)
        return 0
    if args.wave4:
        _run_specs(wave4_cases(), "wave4", args.progress)
        return 0
    if args.wave4_shard:
        i, n = (int(x) for x in args.wave4_shard.split("/"))
        mine = [c for k, c in enumerate(wave4_cases()) if k % n == i]
        _run_specs(mine, f"wave4 {i}/{n}", args.progress)
        return 0
    if args.wave4_validate:
        mu, mur = args.wave4_validate
        _run_specs(wave4_validation_cases(mu, mur), "wave4-validate", args.progress)
        return 0
    if args.wave5:
        _run_specs(wave5_cases(), "wave5", args.progress)
        return 0
    if args.wave5_shard:
        i, n = (int(x) for x in args.wave5_shard.split("/"))
        mine = [c for k, c in enumerate(wave5_cases()) if k % n == i]
        _run_specs(mine, f"wave5 {i}/{n}", args.progress)
        return 0
    if args.wave5b:
        _run_specs(wave5b_probe_cases(), "wave5b", args.progress)
        return 0
    if args.wave5c:
        _run_specs(wave5c_cases(), "wave5c", args.progress)
        return 0
    if args.wave5b_confirm is not None:
        _run_specs(wave5b_confirmation_cases(args.wave5b_confirm), "wave5b-confirm",
                   args.progress)
        return 0
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        mine = [c for k, c in enumerate(wave2_cases(args.mu_rolling)) if k % n == i]
        _run_specs(mine, f"wave2 {i}/{n}", args.progress)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
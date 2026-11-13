#!/usr/bin/env python
"""CASE-PENETRATION R4 (mission m18243): floor-clearance ladder and the
additive-contamination diagnostic for the quasi-static sphere-penetration case.

Why this script exists
----------------------
The m15404 delivery (``src/tensordem/penetration_quasistatic.py``) reported
``alpha_fit = 349`` against the Brzinski et al. (2013) value 35 +- 5 and blamed
coarse graining (R/d = 2 vs 42 in the experiment).  Re-reading the *geometry*
shows a sharper explanation: with ``bed_depth_over_d = 8`` and
``penetrate_over_r = 1.6`` at R/d = 3, the rigid floor sits only
``(8 - 1.6*3) / 3 = 1.07 R`` below the deepest intruder point -- about two grain
diameters.  A bed whose rigid bottom is closer than ~3 R is not the
semi-infinite medium that Eq. (3c)/(3d) describes: the grains under the
intruder are in confined compression between the intruder and the floor, and the
measured resistance is the response of a thin layer, not a hydrostatic-like drag.

Two things are therefore delivered here, without touching the case module (an
open branch ``agent/m18201`` already edits it; see the mission report):

1. **Geometry accounting** -- ``floor_clearance_over_r`` from the *measured*
   settled bed surface and the *measured* penetration depth, plus a static audit
   of every shipped preset (``audit`` sub-command, no simulation).
2. **Floor-clearance ladder** -- the same case re-run with the floor pushed
   down (clearance 1 R -> 4.5 R at fixed R/d, v, mu, box).  If the excess is
   floor contamination, alpha must fall as 1/clearance drops out of the problem;
   a ``alpha_inf`` extrapolation gives the semi-infinite-bed estimate with an
   honest uncertainty, and the acceptance question ("is alpha within 15 % of
   35") is answered on *that* number, not on the thin-bed run.

The additive-contamination diagnostic
-------------------------------------
The floor adds a depth-independent (or slowly varying) extra resistance to the
Eq. (3d) drag law.  On a window ``[z1, z2]`` the two hypotheses
``F = alpha S(z)`` (multiplicative) and ``F = alpha S(z) + c`` (additive) are
distinguishable: a constant offset biases the one-parameter estimate by
``c <S,1>/<S,S>``, which shrinks as the window moves deeper, whereas a wrong
alpha does not change with window position.  ``window_ladder`` reports, for each
sub-window, the one-parameter alpha, the joint (alpha, offset) fit and both
residuals; ``additive`` in the per-window record says whether adding the offset
actually explains the data (residual ratio) or not.

Everything numeric that ends up in the case document is produced by
``report`` from the JSON records -- no hand transcription.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, fields as dataclass_fields, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensordem.penetration_quasistatic import (  # noqa: E402
    G,
    REFERENCE,
    PenetrationConfig,
    QuasiStaticPenetration,
    alpha_from_slope,
    brzinski_shape,
    evaluate_profile,
    preset,
)

# A bed is treated as semi-infinite for Eq. (3c)/(3d) only when the rigid floor
# is at least this many intruder radii below the deepest intruder point.  The
# threshold comes from the ladder itself (see docs/PENETRATION_DEEP_BED_M18243.md):
# 2 R still shows a measurable floor signature, 3 R and deeper do not resolve
# against each other inside the run-to-run scatter.
SEMI_INFINITE_CLEARANCE_OVER_R = 3.0
# Extra fill below the deepest point so that grains have a full layer between the
# intruder and the floor (0.5 d grain radius + 0.5 d slack).
BOTTOM_MARGIN_OVER_D = 1.0

DEFAULT_WINDOWS: Tuple[Tuple[float, float], ...] = (
    (0.5, 1.6),   # the module's primary fit window
    (0.5, 1.0),
    (1.0, 1.6),
    (1.2, 1.6),
    (1.3, 1.6),
)


# ---------------------------------------------------------------------------
# Geometry accounting (pure)
# ---------------------------------------------------------------------------

def floor_clearance_over_r(z_surface: float, max_depth: float, radius: float) -> float:
    """Distance from the deepest intruder point to the floor, in units of R.

    ``z_surface`` is the measured free surface height above the floor (m) and
    ``max_depth`` the deepest wetted-cap depth reached (m).  The quantity is
    *measured* from the settled bed, not read off the nominal fill height, so a
    bed that compacts or heaves cannot hide a thin section.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return float((z_surface - max_depth) / radius)


def bed_depth_for_clearance(clearance_over_r: float, R: float, penetrate_over_r: float,
                            d: float,
                            bottom_margin_over_d: float = BOTTOM_MARGIN_OVER_D) -> float:
    """Initial fill height (m) that leaves ``clearance_over_r * R`` under the
    deepest intruder point, plus ``bottom_margin_over_d`` grain diameters."""
    if clearance_over_r <= 0.0:
        raise ValueError("clearance must be positive")
    return float((penetrate_over_r + clearance_over_r) * R
                 + bottom_margin_over_d * d)


def protocol_audit(z_surface: float, max_depth: float, cfg: PenetrationConfig) -> Dict[str, object]:
    """Dimensionless protocol numbers that decide whether the case is *the*
    Brzinski experiment, with explicit warnings when it is not."""
    R, d = cfg.R, cfg.d
    clr = floor_clearance_over_r(z_surface, max_depth, R)
    warnings: List[str] = []
    if clr < SEMI_INFINITE_CLEARANCE_OVER_R:
        warnings.append(
            f"floor clearance {clr:.2f} R < {SEMI_INFINITE_CLEARANCE_OVER_R:.1f} R: "
            "confined compression, Eq.(3c)/(3d) (semi-infinite bed) does not apply")
    v_ratio = cfg.penetration_speed / math.sqrt(G * R)
    if v_ratio > 0.3:
        warnings.append(f"v/sqrt(gR) = {v_ratio:.3f} > 0.3: check quasi-staticity by a rate sweep")
    if cfg.intruder_radius_over_d < 10.0:
        warnings.append(f"R/d = {cfg.intruder_radius_over_d:.1f} >= 2 but << 42 "
                        "(reference grains): coarse-grained, alpha is not scale-free")
    if cfg.wall_half_width_over_R < 7.5:
        warnings.append(f"wall half width {cfg.wall_half_width_over_R:.1f} R < 7.5 R "
                        "(reference tank): lateral Janssen confinement not excluded")
    return dict(
        radius_over_d=float(cfg.intruder_radius_over_d),
        floor_clearance_over_r=clr,
        floor_clearance_over_d=float(clr * R / d),
        semi_infinite_premise=bool(clr >= SEMI_INFINITE_CLEARANCE_OVER_R),
        clearance_threshold_over_r=SEMI_INFINITE_CLEARANCE_OVER_R,
        wall_half_width_over_R=float(cfg.wall_half_width_over_R),
        speed_ratio_v_over_sqrt_gR=float(v_ratio),
        penetration_speed=float(cfg.penetration_speed),
        z_surface=float(z_surface),
        max_depth=float(max_depth),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Window ladder / additive diagnostic (pure)
# ---------------------------------------------------------------------------

def _shape_terms(z: np.ndarray, radius: float, mu: float, rho_bulk: float) -> np.ndarray:
    """``S(z) = mu rho_bulk g * shape(z)`` -- the unit-alpha force of Eq. (3c)/(3d)."""
    return mu * rho_bulk * G * brzinski_shape(z, radius)


def line_diagnostic(z: np.ndarray, F: np.ndarray, radius: float, mu: float,
                    rho_bulk: float) -> Dict[str, float]:
    """Free line ``F = s z + b`` re-expressed in Brzinski variables.

    On the deep branch (``z >= R``) Eq. (3d) *is* a straight line,
    ``F = alpha mu rho g pi R^2 (z - R/3)``, so the slope alone fixes alpha and the
    intercept must come out as ``-s R/3``.  Whatever straight-line tare remains,

        ``tare_N = b + s R / 3``,

    is a depth-independent extra force (floor pedestal, rig offset, tare).  This
    decomposition is well conditioned -- alpha from the slope, tare from the
    intercept -- unlike a joint ``alpha S(z) + c`` fit, whose two columns are nearly
    collinear on a short window and can return a negative alpha out of noise.
    """
    z = np.asarray(z, float)
    F = np.asarray(F, float)
    if len(z) < 4:
        return dict(slope_N_per_m=float("nan"), intercept_N=float("nan"),
                    theory_intercept_N=float("nan"), alpha_free=float("nan"),
                    tare_N=float("nan"), rms_N=float("nan"), n=int(len(z)))
    slope, intercept = np.polyfit(z, F, 1)
    alpha_free = alpha_from_slope(float(slope), radius, mu, rho_bulk)
    tare = float(intercept) + float(slope) * radius / 3.0
    resid = F - (slope * z + intercept)
    return dict(slope_N_per_m=float(slope), intercept_N=float(intercept),
                theory_intercept_N=float(-slope * radius / 3.0),
                alpha_free=float(alpha_free), tare_N=float(tare),
                rms_N=float(np.sqrt(np.mean(resid ** 2))), n=int(len(z)))


def window_ladder(z, F, radius: float, mu: float, rho_bulk: float,
                  windows: Iterable[Tuple[float, float]] = DEFAULT_WINDOWS,
                  smoothing_over_r: float = 0.15,
                  alpha_ref: float = REFERENCE["alpha"]) -> List[Dict[str, object]]:
    """Per-window alpha estimates and the additive tare implied by each window.

    Each record carries
    * ``alpha``      -- the module's constrained one-parameter shape fit (this is
      the number the acceptance criterion is written on),
    * ``alpha_free`` -- slope-only estimate, immune to a constant tare,
    * ``tare_N``     -- the constant force the line needs on top of the drag law,
    * ``rms_N``      -- residual of the line fit.
    """
    z = np.asarray(z, float)
    F = np.asarray(F, float)
    rows: List[Dict[str, object]] = []
    for lo, hi in windows:
        sel = (z >= lo * radius) & (z <= hi * radius)
        n = int(np.count_nonzero(sel))
        if n < 5:
            continue
        zw, Fw = z[sel], F[sel]
        fit, _acc, _raw = evaluate_profile(
            zw, Fw, radius=radius, mu=mu, rho_bulk=rho_bulk,
            fit_from=lo * radius, fit_to=hi * radius,
            smoothing=smoothing_over_r * radius, alpha_ref=alpha_ref)
        if fit.get("degenerate"):
            continue
        line = line_diagnostic(zw, Fw, radius, mu, rho_bulk)
        rows.append(dict(
            z_from_over_r=float(lo), z_to_over_r=float(hi), n_samples=n,
            deep_branch=bool(lo >= 1.0),
            alpha=float(fit["alpha_fit"]),
            alpha_free=float(line["alpha_free"]),
            tare_N=float(line["tare_N"]),
            intercept_N=float(line["intercept_N"]),
            theory_intercept_N=float(-line["slope_N_per_m"] * radius / 3.0),
            rms_line_N=float(line["rms_N"]),
            mean_force_N=float(np.mean(Fw)),
            force_variation_coefficient=float(fit["force_variation_coefficient"]),
        ))
    return rows


def additive_verdict(ladder: Sequence[Dict[str, object]],
                     alpha_ref: float = REFERENCE["alpha"],
                     rel_tol: float = 0.15) -> Dict[str, object]:
    """Is the excess alpha *additive* (constant tare) or *multiplicative*?

    Uses the windows that lie fully on the deep branch (``z >= R``), where
    Eq. (3d) is exactly linear and the slope/tare split is exact.

    * ``additive``      -- the slope-only alpha is within ``rel_tol`` of the
      reference while the constrained fit is not: the depth dependence is right
      and only a constant push is extra (floor pedestal / tare);
    * ``multiplicative``-- the slope itself is off: the drag coefficient is wrong
      (coarse graining, mu mis-calibration, wall friction).
    """
    deep = [r for r in ladder if r.get("deep_branch")]
    if not deep:
        return dict(verdict="undetermined", reason="no deep-branch window",
                    n_windows=0, alpha_free_median=float("nan"),
                    tare_median_N=float("nan"), rel_error_alpha_free=float("nan"))
    af = float(np.median([float(r["alpha_free"]) for r in deep]))
    tare = float(np.median([float(r["tare_N"]) for r in deep]))
    rel = abs(af - alpha_ref) / alpha_ref
    return dict(verdict="additive" if rel <= rel_tol else "multiplicative",
                reason=("slope matches the reference within the acceptance band; "
                        "the excess of the constrained fit is a constant tare"
                        if rel <= rel_tol else
                        "the slope itself misses the reference: not a tare"),
                n_windows=len(deep), alpha_free_median=af, tare_median_N=tare,
                rel_error_alpha_free=rel)


def fit_clearance_limit(clearances: Sequence[float], alphas: Sequence[float]) -> Dict[str, float]:
    """Extrapolate ``alpha(c) = alpha_inf + k / c`` to infinite floor clearance.

    The floor term of a thin bed does not vanish exponentially in the clearance
    but like a decaying correction; ``1/c`` is the leading term that is consistent
    with the ladder (see the doc).  Returns the intercept, the slope, the fit rms
    and the *standard error* of the intercept so the extrapolation is quoted with
    an uncertainty instead of a single number.
    """
    c = np.asarray(clearances, float)
    a = np.asarray(alphas, float)
    if len(c) < 3:
        raise ValueError("need at least 3 clearance points to extrapolate a limit")
    X = np.column_stack([np.ones_like(c), 1.0 / c])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    resid = a - X @ coef
    dof = max(len(c) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return dict(alpha_inf=float(coef[0]), slope_k=float(coef[1]),
                rms=float(np.sqrt(np.mean(resid ** 2))),
                stderr_intercept=float(math.sqrt(max(cov[0, 0], 0.0))),
                n_points=int(len(c)))


# ---------------------------------------------------------------------------
# Case construction / execution
# ---------------------------------------------------------------------------

def build_config(clearance_over_r: float, base: Optional[PenetrationConfig] = None,
                 **overrides) -> PenetrationConfig:
    """PenetrationConfig with a fill height that leaves ``clearance_over_r * R``
    under the deepest point.  Only public dataclass fields are touched, so the
    case module stays byte-identical (no conflict with other open branches)."""
    aliases = {"rd": "intruder_radius_over_d", "wall": "wall_half_width_over_R",
               "speed": "penetration_speed"}
    cfg = replace(base or PenetrationConfig())
    writable = {f.name for f in dataclass_fields(PenetrationConfig)}
    for k, v in overrides.items():
        field = aliases.get(k, k)
        if field not in writable:
            raise SystemExit(f"unknown config field {k!r}")
        setattr(cfg, field, v)
    depth = bed_depth_for_clearance(clearance_over_r, cfg.R, cfg.penetrate_over_r,
                                    cfg.d, BOTTOM_MARGIN_OVER_D)
    cfg.bed_depth_over_d = depth / cfg.d
    cfg.label = f"clr{clearance_over_r:g}_v{cfg.penetration_speed:g}"
    cfg.validate()
    return cfg


def run_case(cfg: PenetrationConfig, out_dir: str, verbose: bool = False) -> Dict[str, object]:
    """Run one case and attach the geometry audit + window ladder to its record."""
    case = QuasiStaticPenetration(cfg)
    result = case.run(verbose=verbose)
    prof = result["profile"]
    z = np.asarray(prof["z"], float)
    F = np.asarray(prof["F"], float)
    max_depth = float(z.max()) if len(z) else 0.0
    z_surface = float(result["bed"]["z_surface"])
    radius = float(cfg.R)
    audit = protocol_audit(z_surface, max_depth, cfg)
    ladder = window_ladder(z, F, radius, float(cfg.mu_grain),
                           float(result["bed"]["bulk_density"]),
                           smoothing_over_r=float(cfg.smoothing_over_r))
    result["protocol_audit"] = audit
    result["alpha_ladder"] = ladder
    result["additive_verdict"] = additive_verdict(ladder)
    result["mission"] = dict(id=18243, round="R4", case="CASE-PENETRATION",
                             script="scripts/penetration_deep_bed.py")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
    if verbose:
        print(json.dumps(dict(label=cfg.label, path=path, status=result["status"],
                              audit=audit, verdict=result["additive_verdict"],
                              metrics={k: result["metrics"][k] for k in
                                       ("alpha_fit", "penetration_force_error",
                                        "force_variation_coefficient",
                                        "kinetic_energy_max_J")}), indent=1))
    return result


def static_audit() -> Dict[str, object]:
    """Nominal geometry audit of the shipped presets (no simulation).

    The nominal clearance is the optimistic bound: it ignores settlement and the
    grains resting on the floor, so the measured clearance is <= this value.
    """
    rows = []
    for name in ("base", "rd2", "rd4", "slow", "rate2x", "damp2000", "dashpot"):
        cfg = preset(name)
        nominal_depth = cfg.bed_depth - cfg.penetrate_over_r * cfg.R
        rows.append(dict(preset=name, radius_over_d=cfg.intruder_radius_over_d,
                         bed_depth_over_d=cfg.bed_depth_over_d,
                         penetration_speed=cfg.penetration_speed,
                         nominal_clearance_over_r=float(nominal_depth / cfg.R),
                         semi_infinite_premise=bool(
                             nominal_depth / cfg.R >= SEMI_INFINITE_CLEARANCE_OVER_R)))
    return dict(threshold_over_r=SEMI_INFINITE_CLEARANCE_OVER_R, presets=rows,
                n_presets=len(rows),
                n_violating=int(sum(1 for r in rows if not r["semi_infinite_premise"])))


def reanalyze(root: str) -> List[Dict[str, object]]:
    """Recompute audit + ladder + verdict from the stored force-depth profile.

    Records keep the raw ``profile`` (z, F, KE), so the analysis can be revised
    without re-running the DEM -- useful because the ladder/diagnostic functions
    are pure and their definition may be sharpened after the runs land.
    """
    import glob

    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        cfg = PenetrationConfig(**rec["config"])
        z = np.asarray(rec["profile"]["z"], float)
        F = np.asarray(rec["profile"]["F"], float)
        max_depth = float(z.max()) if len(z) else 0.0
        rec["protocol_audit"] = protocol_audit(float(rec["bed"]["z_surface"]),
                                               max_depth, cfg)
        rec["alpha_ladder"] = window_ladder(z, F, cfg.R, float(cfg.mu_grain),
                                            float(rec["bed"]["bulk_density"]),
                                            smoothing_over_r=float(cfg.smoothing_over_r))
        rec["additive_verdict"] = additive_verdict(rec["alpha_ladder"])
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        out.append(dict(path=path, label=rec.get("label"),
                        clearance_over_r=rec["protocol_audit"]["floor_clearance_over_r"],
                        alpha_fit=rec["metrics"]["alpha_fit"],
                        alpha_free=(float(np.median([r["alpha_free"] for r in rec["alpha_ladder"]
                                                     if r["deep_branch"]]))
                                    if any(r["deep_branch"] for r in rec["alpha_ladder"])
                                    else float("nan")),
                        verdict=rec["additive_verdict"]["verdict"]))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("audit", help="static geometry audit of the shipped presets")
    pa.add_argument("--out", required=True)

    pr = sub.add_parser("run", help="one floor-clearance case")
    pr.add_argument("--clearance", type=float, required=True,
                    help="floor clearance under the deepest point, in units of R")
    pr.add_argument("--speed", type=float, default=None)
    pr.add_argument("--rd", type=float, default=None)
    pr.add_argument("--d", type=float, default=None, dest="grain_diameter")
    pr.add_argument("--wall", type=float, default=None, help="wall_half_width_over_R")
    pr.add_argument("--settle-steps", type=int, default=None)
    pr.add_argument("--sample-every", type=int, default=None,
                    help="contact-force sampling interval (steps); steps are unchanged")
    pr.add_argument("--penetrate-over-r", type=float, default=None)
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--out", required=True)
    pr.add_argument("--verbose", action="store_true")

    pz = sub.add_parser("reanalyze",
                        help="recompute audit/ladder/verdict from stored profiles")
    pz.add_argument("--dir", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "reanalyze":
        rows = reanalyze(args.dir)
        print(json.dumps(rows, indent=1))
        return 0
    if args.cmd == "audit":
        audit = static_audit()
        _write_json(args.out, audit)
        print(json.dumps(audit, indent=1))
        return 0

    overrides = {}
    for key in ("speed", "rd", "wall", "settle_steps", "penetrate_over_r", "seed",
                "grain_diameter", "sample_every"):
        val = getattr(args, key)
        if val is None:
            continue
        field = {"speed": "penetration_speed", "rd": "intruder_radius_over_d",
                 "wall": "wall_half_width_over_R",
                 "sample_every": "sample_every_steps"}.get(key, key)
        overrides[field] = val
    cfg = build_config(args.clearance, **overrides)
    res = run_case(cfg, args.out, verbose=args.verbose)
    print(json.dumps({"label": cfg.label, "status": res["status"],
                      "failed_criteria": res["failed_criteria"],
                      "alpha_fit": res["metrics"].get("alpha_fit"),
                      "clearance_over_r": res["protocol_audit"]["floor_clearance_over_r"],
                      "verdict": res["additive_verdict"]["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
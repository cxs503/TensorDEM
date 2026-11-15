"""Round 21 (m21528): is the low-I cell "not shearing", or is the abscissa blind?

Motivation
----------
Round 20 landed one point of the ``CASE-DEM-PLANE-SHEAR-MUIDI`` sweep at
``I = 3e-3`` and reported ``q ~ -0.010``, i.e. "the measured bulk shear rate is
30x below the imposed platen rate; the bed is not shearing, the cell is still
in its start-up transient", and from that drew a protocol conclusion ("low-I
points need a longer strain window").  That conclusion rests entirely on one
estimator: ``shear_rate_measured`` is a weighted least-squares slope fitted to
the **central 50 % of the gap** (``z_lo = 0.25 * plate_z``,
``z_hi = 0.75 * plate_z``), with ``profile_r2`` not even reported.

A flat interior profile and a static bed are different physical statements.  A
bed that localises its deformation inside the two wall layers - which is what a
frictional smooth wall at low inertial number does - has a flat interior profile
while both walls still move.  The two are distinguishable only if the wall
layers are measured, which is what ``strain_partition`` (added in this round to
``tensordem.mu_i_rheology``) does: it bins the whole gap and reports a full-gap
least-squares rate, an extreme-bin difference rate (immune to profile
curvature), the two wall velocity ratios and the interior strain fraction.

This script runs the round-20 device with that diagnostic enabled and answers
the question with a pre-registered decision rule instead of a narrative.

Pre-registered verdicts (decided before the run, see ``PREREG``)
---------------------------------------------------------------
``LOCALISED``  q_interior <= 0.2 AND q_adjacent >= 0.6 AND
               interior_strain_fraction <= 0.5:  the bed shears, the legacy
               interior estimator is blind -> round 20's "not shearing" is an
               estimator artefact and the measured-rate abscissa must be
               rebuilt from the wall layers.
``SLIP``       q_full <= 0.2 AND v_top_over_platen <= 0.5:  the platen slides
               over a quasi-static bed; then Coulomb requires
               mu_wall = mu_contact (reported), otherwise the two wall
               quantities are mutually inconsistent.
``STATIC_OK``  q_full <= 0.2 and the platen velocity is transmitted
               (v_top_over_platen >= 0.9):  physically impossible for a
               velocity-driven wall (it would violate no-slip at the platen),
               so it is reported as an INCONSISTENT state, not a result.
``QUOTABLE``   q_adjacent within 20 % of 1:  the point kinematically delivers
               the requested rate, so mu/phi from it may be quoted on the
               measured abscissa.

Usage::

    python low_i_profile_audit.py --I 3e-3 --strain 0.5 --tag i3e3
    python low_i_profile_audit.py --I 1e-2 --strain 0.5 --tag i1e2 --settle 1500
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tensordem.mu_i_rheology import (  # noqa: E402
    ServoShearCase, inertial_number, simulate_servo_shear,
)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "low_i_profile_r21"

# --- device: identical to round 20 (high_n_coverage.py) --------------------
P_DEFAULT = 6.25e4      # Pa
KN = 6.25e4             # N/m
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
R = 0.005
RHO = 2500.0
SEED = 5279
W, D, H = 0.16, 0.08, 0.09      # 32d x 16d x 18d -> N ~ 1000+
SETTLE_STEPS = 3000
SAMPLE_STRIDE = 25

PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": 21,
    "kind": "estimator-defect audit of the low-I measured-rate abscissa",
    "device": "tensordem.mu_i_rheology.simulate_servo_shear",
    "new_diagnostic": "tensordem.mu_i_rheology.strain_partition (whole-gap)",
    "question": ("round 20 read q = -0.010 at I = 3e-3 as 'the bed is not "
                 "shearing'. Is that the material or the central-50 % fit "
                 "window?"),
    "decision_rule": {
        "LOCALISED": ("q_interior <= 0.2", "q_adjacent >= 0.6",
                      "interior_strain_fraction <= 0.5"),
        "SLIP": ("q_full <= 0.2", "v_top_over_platen <= 0.5"),
        "INCONSISTENT": ("q_full <= 0.2", "v_top_over_platen >= 0.9"),
        "QUOTABLE": ("|q_adjacent - 1| <= 0.2",),
    },
    "coulomb_stick_check": ("slip at a frictional wall requires the wall "
                            "traction ratio to sit at mu_contact; a reported "
                            "mu_wall well below mu_contact with a large "
                            "velocity deficit is self-contradictory"),
    "honesty": ("the verdict is emitted from the decision rule, whatever it "
                "says; no threshold is chosen after seeing the numbers"),
}


def build_case(*, I: float, strain: float, p: float, settle: int,
               stride: int) -> ServoShearCase:
    return ServoShearCase(
        inertial_number=I, mu=MU_CONTACT, mu_rolling=MU_ROLLING,
        restitution=E, kn=KN, normal_stress=p, shear_strain=strain,
        settle_steps=settle, min_shear_steps=1, sample_stride=stride,
        packing_gap=0.02, jitter=0.01, seed=SEED, radius=R, density=RHO,
        box_width=W, box_depth=D, fill_height=H,
        bulk_stress=False,          # wall traction is enough here
        profile_detail=True, profile_bins=9,
        max_recover_strain=0.25,    # a strain request is a ceiling (round 17)
    )


def classify(row: dict) -> dict:
    """Apply ``PREREG['decision_rule']`` to a measured row.  Never re-tuned."""
    q_int = row.get("q_interior")
    q_adj = row.get("q_adjacent")
    q_full = row.get("q_full")
    v_top = row.get("v_top_over_platen")
    ivf = row.get("interior_strain_fraction")
    mu_wall = row.get("mu_wall")
    checks = []

    def _f(x):
        return isinstance(x, (int, float)) and math.isfinite(float(x))

    localised = (_f(q_int) and _f(q_adj) and _f(ivf)
                 and q_int <= 0.2 and q_adj >= 0.6 and ivf <= 0.5)
    slip = (_f(q_full) and _f(v_top) and q_full <= 0.2 and v_top <= 0.5)
    inconsistent = (_f(q_full) and _f(v_top) and q_full <= 0.2 and v_top >= 0.9)
    quotable = _f(q_adj) and abs(q_adj - 1.0) <= 0.2
    checks = [
        {"name": "localised_shear", "pass": bool(localised),
         "detail": f"q_interior={q_int}, q_adjacent={q_adj}, ivf={ivf}"},
        {"name": "wall_slip", "pass": bool(slip),
         "detail": f"q_full={q_full}, v_top/v_platen={v_top}, "
                   f"mu_wall={mu_wall}, mu_contact={MU_CONTACT}"},
        {"name": "internally_inconsistent", "pass": bool(inconsistent),
         "detail": "flat full-gap profile while the platen velocity is "
                   "transmitted - violates no-slip at the driven wall"},
        {"name": "abscissa_quotable", "pass": bool(quotable),
         "detail": f"q_adjacent={q_adj} (|q-1| <= 0.2)"},
    ]
    if localised:
        verdict = "LOCALISED"
    elif inconsistent:
        verdict = "INCONSISTENT"
    elif slip:
        verdict = "SLIP"
    elif quotable:
        verdict = "QUOTABLE"
    else:
        verdict = "UNDETERMINED"
    return {"verdict": verdict, "checks": checks}


def _tail_mean(series: list[dict], key: str, frac: float = 0.2):
    if not series:
        return None
    n = max(1, int(frac * len(series)))
    vals = [float(s[key]) for s in series[-n:]
            if key in s and math.isfinite(float(s[key]))]
    return float(sum(vals) / len(vals)) if vals else None


def run_point(*, I: float, strain: float, p: float, settle: int, stride: int,
              tag: str) -> dict:
    case = build_case(I=I, strain=strain, p=p, settle=settle, stride=stride)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case, return_series=True)
    wall = time.perf_counter() - t0
    gap = res.get("gap_kinematics") or {}
    series = res.get("_series") or res.get("series") or []

    # the two abscissae on the same state
    p_final = res.get("normal_stress_measured_pa") or 0.0
    d = 2.0 * R
    row = {
        "tag": tag,
        "I_target": I,
        "strain_requested": strain,
        "n_particles": res.get("n_particles"),
        "wall_clock_s": wall,
        "gamma_dot_imposed": res.get("shear_rate_imposed"),
        "plate_height": res.get("plate_height_steady") or res.get("plate_height"),
        "p_measured_pa": p_final,
        "p_control_error_final": res.get("pressure_control_error_final"),
        "mu_wall": res.get("mu_final") if res.get("mu_final") is not None
                   else res.get("mu_effective"),
        "phi_container": res.get("phi_container_final"),
        "phi_extent": res.get("phi_extent_final"),
        "strain_achieved": res.get("shear_strain_achieved"),
        "strain_budget_respected": res.get("strain_budget_respected"),
        "q_interior": gap.get("gap_q_interior"),
        "q_full": gap.get("gap_q_full"),
        "q_adjacent": gap.get("gap_q_adjacent"),
        "rate_interior": gap.get("gap_shear_rate_interior"),
        "rate_full": gap.get("gap_shear_rate_full"),
        "rate_adjacent": gap.get("gap_shear_rate_adjacent"),
        "v_top_over_platen": gap.get("gap_v_top_over_platen"),
        "v_bot_over_platen": gap.get("gap_v_bot_over_platen"),
        "interior_strain_fraction": gap.get("gap_interior_strain_fraction"),
        "wall_strain_fraction": gap.get("gap_wall_strain_fraction"),
        "gap_n_particles": gap.get("gap_n_particles"),
        "profile_r2_interior": res.get("profile_r2"),
        "I_imposed_abscissa": res.get("inertial_number_imposed_rate"),
        "I_interior_abscissa": res.get("inertial_number_measured_rate_final"),
        "I_full_abscissa": (inertial_number(gap["gap_shear_rate_full"], d,
                                           p_final, RHO)
                            if gap.get("gap_shear_rate_full") and p_final else None),
        "I_adjacent_abscissa": (inertial_number(gap["gap_shear_rate_adjacent"], d,
                                                p_final, RHO)
                                if gap.get("gap_shear_rate_adjacent") and p_final
                                else None),
    }
    row.update(classify(row))

    # time series of the competing estimators (strain-resolved)
    ts = {"strain": [], "q_interior": [], "q_full": [], "q_adjacent": [],
          "ivf": [], "v_top": []}
    for s in series:
        ts["strain"].append(s.get("strain"))
        ts["q_interior"].append(s.get("gap_q_interior"))
        ts["q_full"].append(s.get("gap_q_full"))
        ts["q_adjacent"].append(s.get("gap_q_adjacent"))
        ts["ivf"].append(s.get("gap_interior_strain_fraction"))
        ts["v_top"].append(s.get("gap_v_top_over_platen"))
    row["_timeseries"] = ts
    row["_profile_final"] = res.get("gap_profile_final")
    return row


def _write(point: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"point_{point['tag']}.json"
    out.write_text(json.dumps(point, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--I", type=float, action="append",
                    help="inertial-number target (repeatable)")
    ap.add_argument("--strain", type=float, default=0.5)
    ap.add_argument("--p", type=float, default=P_DEFAULT)
    ap.add_argument("--settle", type=int, default=SETTLE_STEPS)
    ap.add_argument("--stride", type=int, default=SAMPLE_STRIDE)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    targets = args.I or [3e-3]
    rows = []
    for I in targets:
        tag = args.tag or f"i{I:g}".replace("0.", "0p").replace("-", "m")
        if len(targets) > 1:
            tag = f"i{I:g}".replace("0.", "0p").replace("-", "m")
        pt = run_point(I=I, strain=args.strain, p=args.p, settle=args.settle,
                       stride=args.stride, tag=tag)
        path = _write(pt)
        rows.append(pt)
        g = pt["gap_n_particles"]
        print(f"[{tag}] I_target={I:g} N={pt['n_particles']} "
              f"gamma_dot={pt['gamma_dot_imposed']:.4g} "
              f"q_int={pt['q_interior']} q_full={pt['q_full']} "
              f"q_adj={pt['q_adjacent']} v_top/v_pl={pt['v_top_over_platen']} "
              f"ivf={pt['interior_strain_fraction']} (gap n={g}) "
              f"-> {pt['verdict']}  [{pt['wall_clock_s']:.1f}s] -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
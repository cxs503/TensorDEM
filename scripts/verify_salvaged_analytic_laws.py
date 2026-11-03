"""Independent closed-form re-verification of the analytic laws restored in m2536.

The point of this script is *independence*: every reference value below is coded from
the textbook relation, not from the ``tensordem`` helper it is compared against.  If a
salvaged helper were quietly wrong, the relative-error column would show it.

Anchors
-------
H  Hertz normal contact (Johnson, *Contact Mechanics*, eq. 4.2 / 4.5):
      F = (4/3) E* sqrt(R*) d^(3/2),  U = (8/15) E* sqrt(R*) d^(5/2),
      k_s = F/d = (4/3) E* sqrt(R*) d^(1/2),
   1/E* = sum (1-nu^2)/E,  1/R* = sum 1/R  (rigid counter-surface: R* = R, E* = E/(1-nu^2)).
R  Viscous restitution map (linear dashpot, small-damping closure):
      zeta = -ln(e) / sqrt(pi^2 + ln(e)^2),  c_n = 2 zeta sqrt(k m_eff),
   inverted back through the *engine* restitution measure.
T  Cundall-Strack tangential hysteresis: with a_stick = mu F_n / k_t the loop area per cycle is
      W = 4 mu F_n (a - a_stick)  for a > a_stick,  W = 0 otherwise,
   and the time-stepped simulation must reproduce it to O(dt).

Outputs (artifacts/salvage_m2536/):
    analytic_law_verification.json   full table with every measured value and error
    analytic_law_verification.csv    the same table, spreadsheet friendly
    analytic_law_verification.png    three-panel figure, module vs closed form
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensordem.restitution import damping_for_restitution_analytic, restitution_from_damping
from tensordem.contact_tangential import tangential_cycle_dissipated_work
from tensordem.tangential_verification import StickSlipCase, simulate_stick_slip
from tensordem.dem3d import (
    hertz_contact_potential,
    hertz_effective_radius,
    hertz_normal_force,
    hertz_reduced_modulus,
    hertz_secant_stiffness,
)

E_YOUNG = 1.0e9
NU = 0.3
RADIUS = 0.025
STIFFNESS = 2.0e5
MASS = 0.06
MU = 0.3
F_N = 100.0
K_T = 1000.0

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "salvage_m2536"


# ---------------------------------------------------------------------------
# independent closed forms (no tensordem import used here)
# ---------------------------------------------------------------------------
def hertz_reference(overlap: float, *, rigid: bool) -> dict[str, float]:
    e_star = 1.0 / (((1.0 - NU**2) / E_YOUNG) * (1.0 if rigid else 2.0))
    r_star = RADIUS * (1.0 if rigid else 0.5)
    coef = (4.0 / 3.0) * e_star * math.sqrt(r_star)
    force = coef * overlap**1.5
    return {
        "E_star": e_star,
        "R_star": r_star,
        "force": force,
        "potential": (8.0 / 15.0) * e_star * math.sqrt(r_star) * overlap**2.5,
        "secant": coef * math.sqrt(overlap),
    }


def restitution_reference(e_target: float) -> float:
    log_e = math.log(e_target)
    zeta = -log_e / math.sqrt(math.pi**2 + log_e**2)
    return 2.0 * zeta * math.sqrt(STIFFNESS * MASS)


def hysteresis_reference(amplitude: float, mu: float = MU, f_n: float = F_N, k_t: float = K_T) -> float:
    a_stick = mu * f_n / k_t
    return 0.0 if amplitude <= a_stick else 4.0 * mu * f_n * (amplitude - a_stick)


def rel_error(measured: float, reference: float) -> float:
    if reference == 0.0:
        return abs(measured)
    return abs(measured - reference) / abs(reference)


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def hertz_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rigid in (False, True):
        e_star = float(hertz_reduced_modulus(E_YOUNG, NU, rigid_counterpart=rigid))
        r_star = float(hertz_effective_radius(RADIUS, rigid_counterpart=rigid))
        for delta in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            ref = hertz_reference(delta, rigid=rigid)
            force = float(hertz_normal_force(torch.tensor(delta, dtype=torch.float64), e_star, r_star))
            potential = float(hertz_contact_potential(delta, e_star, r_star))
            secant = float(hertz_secant_stiffness(delta, e_star, r_star))
            rows.append(
                {
                    "anchor": "H_hertz",
                    "pair": "rigid_plane" if rigid else "sphere_sphere",
                    "input": delta,
                    "quantity": "F_n [N]",
                    "measured": force,
                    "reference": ref["force"],
                    "rel_error": rel_error(force, ref["force"]),
                }
            )
            rows.append(
                {
                    "anchor": "H_hertz",
                    "pair": "rigid_plane" if rigid else "sphere_sphere",
                    "input": delta,
                    "quantity": "U [J]",
                    "measured": potential,
                    "reference": ref["potential"],
                    "rel_error": rel_error(potential, ref["potential"]),
                }
            )
            rows.append(
                {
                    "anchor": "H_hertz",
                    "pair": "rigid_plane" if rigid else "sphere_sphere",
                    "input": delta,
                    "quantity": "k_s [N/m]",
                    "measured": secant,
                    "reference": ref["secant"],
                    "rel_error": rel_error(secant, ref["secant"]),
                }
            )
    return rows


def restitution_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for e_target in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        damping = float(damping_for_restitution_analytic(STIFFNESS, e_target, MASS))
        reference = restitution_reference(e_target)
        # forward check: run the damping back through the engine restitution measure
        recovered = float(restitution_from_damping(STIFFNESS, damping, MASS))
        rows.append(
            {
                "anchor": "R_restitution",
                "input": e_target,
                "quantity": "c_n [N s/m]",
                "measured": damping,
                "reference": reference,
                "rel_error": rel_error(damping, reference),
            }
        )
        rows.append(
            {
                "anchor": "R_restitution",
                "input": e_target,
                "quantity": "e recovered [-]",
                "measured": recovered,
                "reference": e_target,
                "rel_error": rel_error(recovered, e_target),
            }
        )
    return rows


def hysteresis_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for amplitude in (0.005, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4):
        reference = hysteresis_reference(amplitude)
        closed = float(tangential_cycle_dissipated_work(MU, F_N, K_T, amplitude))
        sim = simulate_stick_slip(StickSlipCase(amplitude=amplitude, steps_per_cycle=2000), clamp_state=True)
        rows.append(
            {
                "anchor": "T_hysteresis",
                "input": amplitude,
                "quantity": "W closed form [J]",
                "measured": closed,
                "reference": reference,
                "rel_error": rel_error(closed, reference),
            }
        )
        rows.append(
            {
                "anchor": "T_hysteresis",
                "input": amplitude,
                "quantity": "W time stepped [J]",
                "measured": float(sim["work_measured"]),
                "reference": reference,
                "rel_error": rel_error(float(sim["work_measured"]), reference),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(hertz_rows, restitution_rows, hysteresis_rows, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    ax = axes[0]
    for pair, style in (("sphere_sphere", "-o"), ("rigid_plane", "-s")):
        sub = [r for r in hertz_rows if r["pair"] == pair and r["quantity"] == "F_n [N]"]
        x = [r["input"] for r in sub]
        ax.loglog(x, [r["reference"] for r in sub], style, mfc="none", label=f"closed form ({pair})")
        ax.loglog(x, [r["measured"] for r in sub], "x", color="crimson", label="tensordem" if pair == "sphere_sphere" else None)
    ax.set_xlabel("overlap $\\delta$ [m]")
    ax.set_ylabel("$F_n$ [N]")
    ax.set_title("H: Hertz normal force, max rel. err %.1e" % max(r["rel_error"] for r in hertz_rows))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    sub = [r for r in restitution_rows if r["quantity"] == "c_n [N s/m]"]
    e = [r["input"] for r in sub]
    ax.plot(e, [r["reference"] for r in sub], "-o", mfc="none", label="closed form $2\\zeta\\sqrt{km}$")
    ax.plot(e, [r["measured"] for r in sub], "x", color="crimson", label="tensordem")
    ax.set_xlabel("target restitution $e$ [-]")
    ax.set_ylabel("$c_n$ [N s/m]")
    ax.set_title("R: dashpot inversion, max rel. err %.1e" % max(r["rel_error"] for r in sub))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    sub = [r for r in hysteresis_rows if r["quantity"] == "W time stepped [J]"]
    a = [r["input"] for r in sub]
    ax.plot(a, [r["reference"] for r in sub], "-o", mfc="none", label="closed form $4\\mu F_n(a-a_{stick})$")
    ax.plot(a, [r["measured"] for r in sub], "x", color="crimson", label="time stepped DEM")
    ax.axvline(MU * F_N / K_T, ls=":", color="gray", label="$a_{stick}$")
    ax.set_xlabel("tangential amplitude $a$ [m]")
    ax.set_ylabel("dissipated work / cycle [J]")
    ax.set_title("T: stick-slip hysteresis, max rel. err %.1e" % max(r["rel_error"] for r in sub))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = hertz_table() + restitution_table() + hysteresis_table()
    hertz_rows = [r for r in rows if r["anchor"] == "H_hertz"]
    restitution_rows = [r for r in rows if r["anchor"] == "R_restitution"]
    hysteresis_rows = [r for r in rows if r["anchor"] == "T_hysteresis"]

    summary: dict[str, object] = {}
    for anchor, sub in (("H_hertz", hertz_rows), ("R_restitution", restitution_rows), ("T_hysteresis", hysteresis_rows)):
        worst = max(sub, key=lambda r: r["rel_error"])
        summary[anchor] = {
            "rows": len(sub),
            "max_rel_error": worst["rel_error"],
            "worst_case": {k: worst[k] for k in ("input", "quantity", "measured", "reference")},
            "below_3pct": worst["rel_error"] < 0.03,
        }

    payload = {
        "protocol": "tensordem-analytic-salvage-verification-v1",
        "constants": {
            "young_modulus_Pa": E_YOUNG,
            "poisson_ratio": NU,
            "radius_m": RADIUS,
            "stiffness_N_per_m": STIFFNESS,
            "mass_kg": MASS,
            "mu": MU,
            "normal_force_N": F_N,
            "tangential_stiffness_N_per_m": K_T,
        },
        "summary": summary,
        "rows": rows,
    }
    (OUT / "analytic_law_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with (OUT / "analytic_law_verification.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["anchor", "pair", "input", "quantity", "measured", "reference", "rel_error"])
        for r in rows:
            writer.writerow([r["anchor"], r.get("pair", ""), r["input"], r["quantity"],
                             repr(r["measured"]), repr(r["reference"]), repr(r["rel_error"])])

    make_figure(hertz_rows, restitution_rows, hysteresis_rows, OUT / "analytic_law_verification.png")
    print(json.dumps(summary, indent=2))
    return payload


if __name__ == "__main__":
    main()
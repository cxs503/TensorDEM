"""Audit trail for the level-ice resistance baselines.

Protocol: ``tensordem-threefield-resistance-audit-v1`` (see
``docs/threefield/FORMULAS.md``).

Emits, from the solver module itself and with no hand-transcribed numbers:

* ``artifacts/threefield/resistance_audit.json`` -- component breakdown, both
  baselines, band width, the reference-speed identities, and the inertness
  report;
* ``artifacts/threefield/figures/resistance_thickness_sweep.png`` -- components
  and both baselines versus ice thickness;
* ``artifacts/threefield/figures/resistance_speed_sweep.png`` -- both baselines
  versus speed with the band shaded.

Everything here is closed-form evaluation at a few hundred points, so the script
is cheap enough to run on a loaded machine.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.threefield.resistance import (  # noqa: E402
    Environment,
    HullGeometry,
    IceProperties,
    bending_resistance,
    crushing_resistance,
    lindqvist_resistance,
    resistance_band,
    riska_resistance,
    submersion_resistance,
)

THICKNESSES = (0.30, 0.45, 0.60)
STUDY_SPEED = 1.0


def component_table(env: Environment, hull: HullGeometry) -> list[dict[str, float]]:
    rows = []
    for h in THICKNESSES:
        ice = IceProperties(thickness_m=h)
        lind = lindqvist_resistance(ice, hull, STUDY_SPEED, env)
        ris = riska_resistance(ice, hull, STUDY_SPEED, env)
        band = resistance_band(ice, hull, STUDY_SPEED, env)
        rows.append(
            {
                "thickness_m": h,
                "crushing_N": crushing_resistance(ice, hull),
                "bending_N": bending_resistance(ice, hull),
                "submersion_N": submersion_resistance(ice, hull, env),
                "added_drag_N": lind["speed_dependent_N"],
                "lindqvist_total_N": lind["total_N"],
                "riska_c1_N": ris["c1_N"],
                "riska_c2_N_per_m_s": ris["c2_N_per_m_s"],
                "riska_total_N": ris["total_N"],
                "band_low_N": band["band_low_N"],
                "band_high_N": band["band_high_N"],
                "band_spread_pct": band["band_spread_pct"],
            }
        )
    return rows


def identity_report(env: Environment, hull: HullGeometry) -> list[dict[str, float]]:
    """Reference-speed identities behind the Riska C2 coefficients."""
    out = []
    g, rho, mu = env.gravity_m_s2, env.water_density_kg_m3, hull.friction_coefficient
    stem = 1.0 + 0.063 * hull.stem_angle_deg
    for h in (0.30, 0.45, 0.60, 1.00):
        v_ref = math.sqrt(g * h)
        friction_only = riska_resistance(
            IceProperties(thickness_m=h, flexural_strength_pa=1e-6), hull, 0.0, env
        )["c2_N_per_m_s"]
        bending_only = riska_resistance(
            IceProperties(thickness_m=h, compressive_strength_pa=1e-6),
            HullGeometry(friction_coefficient=0.0, beam_m=hull.beam_m, length_m=hull.length_m,
                         stem_angle_deg=hull.stem_angle_deg,
                         waterline_entry_angle_deg=hull.waterline_entry_angle_deg),
            0.0,
            env,
        )["c2_N_per_m_s"]
        hydrostatic = mu * rho * g * hull.beam_m * h * h
        r_bend = bending_resistance(IceProperties(thickness_m=h), hull)
        out.append(
            {
                "thickness_m": h,
                "reference_speed_m_s": v_ref,
                "c2_friction_over_hydrostatic": friction_only * v_ref / hydrostatic,
                "c2_friction_expected": 0.5 * stem,
                "c2_bending_over_r_bend": bending_only * v_ref / r_bend,
                "c2_bending_expected": (4.0 / 3.0) * stem,
            }
        )
    return out


def inert_report(env: Environment, hull: HullGeometry) -> dict[str, float]:
    ice, v = IceProperties(), STUDY_SPEED
    base_l = lindqvist_resistance(ice, hull, v, env)["total_N"]
    base_r = riska_resistance(ice, hull, v, env)["total_N"]
    probes = {
        "buttock_angle_deg": (
            lindqvist_resistance(ice, HullGeometry(buttock_angle_deg=35.0), v, env)["total_N"],
            riska_resistance(ice, HullGeometry(buttock_angle_deg=35.0), v, env)["total_N"],
        ),
        "elastic_modulus_pa": (
            lindqvist_resistance(IceProperties(elastic_modulus_pa=9.0e9), hull, v, env)["total_N"],
            riska_resistance(IceProperties(elastic_modulus_pa=9.0e9), hull, v, env)["total_N"],
        ),
        "stem_angle_deg": (
            lindqvist_resistance(ice, HullGeometry(stem_angle_deg=45.0), v, env)["total_N"],
            riska_resistance(ice, HullGeometry(stem_angle_deg=45.0), v, env)["total_N"],
        ),
    }
    return {
        name: {
            "lindqvist_delta_N": pair[0] - base_l,
            "riska_delta_N": pair[1] - base_r,
        }
        for name, pair in probes.items()
    }


def speed_sweep(env: Environment, hull: HullGeometry, thickness: float, n: int = 121):
    speeds = [3.5 * i / (n - 1) for i in range(n)]
    lind, ris, low, high = [], [], [], []
    ice = IceProperties(thickness_m=thickness)
    for v in speeds:
        band = resistance_band(ice, hull, v, env)
        lind.append(band["lindqvist"]["total_N"])
        ris.append(band["riska"]["total_N"])
        low.append(band["band_low_N"])
        high.append(band["band_high_N"])
    return speeds, lind, ris, low, high


def plot_thickness(rows, path: Path) -> None:
    hs = [r["thickness_m"] for r in rows]
    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax_c.plot(hs, [r["crushing_N"] / 1e3 for r in rows], "o-", label="crushing")
    ax_c.plot(hs, [r["submersion_N"] / 1e3 for r in rows], "s-", label="submersion")
    ax_c.plot(hs, [r["bending_N"] / 1e3 for r in rows], "^-", label="bending")
    ax_c.plot(hs, [r["added_drag_N"] / 1e3 for r in rows], "d-", label="added drag (V=1 m/s)")
    ax_c.set_xlabel("ice thickness h [m]")
    ax_c.set_ylabel("component [kN]")
    ax_c.set_title("Lindqvist-style components vs ice thickness")
    ax_c.grid(alpha=0.3)
    ax_c.legend(fontsize=8)

    ax_t.plot(hs, [r["lindqvist_total_N"] / 1e3 for r in rows], "o-", color="tab:blue",
              label="Lindqvist-style")
    ax_t.plot(hs, [r["riska_total_N"] / 1e3 for r in rows], "s-", color="tab:red",
              label="Riska-style (C1 + C2 V)")
    ax_t.fill_between(
        hs,
        [r["band_low_N"] / 1e3 for r in rows],
        [r["band_high_N"] / 1e3 for r in rows],
        color="tab:gray", alpha=0.2, label="reference band",
    )
    for r in rows:
        ax_t.annotate(f"{r['band_spread_pct']:.1f}%", (r["thickness_m"], r["band_high_N"] / 1e3),
                      textcoords="offset points", xytext=(6, 2), fontsize=8)
    ax_t.set_xlabel("ice thickness h [m]")
    ax_t.set_ylabel("total resistance [kN]")
    ax_t.set_title("Both baselines and the band they span (V = 1 m/s)")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8)

    fig.suptitle("TensorDEM three-field level-ice resistance audit "
                 "(protocol tensordem-threefield-resistance-audit-v1)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speed(env, hull, thickness: float, path: Path) -> None:
    speeds, lind, ris, low, high = speed_sweep(env, hull, thickness)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.fill_between([v / 1.0 for v in speeds], [x / 1e3 for x in low], [x / 1e3 for x in high],
                    color="tab:gray", alpha=0.2, label="reference band")
    ax.plot(speeds, [x / 1e3 for x in lind], "-", color="tab:blue", label="Lindqvist-style")
    ax.plot(speeds, [x / 1e3 for x in ris], "-", color="tab:red", label="Riska-style")
    v_ref = math.sqrt(env.gravity_m_s2 * thickness)
    ax.axvline(v_ref, ls=":", color="k", lw=1.0)
    ax.annotate(r"$V=\sqrt{gh}$", (v_ref, ax.get_ylim()[1] * 0.92), fontsize=9,
                xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("ship speed [m/s]")
    ax.set_ylabel("resistance [kN]")
    ax.set_title(f"Resistance vs speed, h = {thickness:.2f} m "
                 "(band = the interval the two baselines span)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="artifacts/threefield")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    env, hull = Environment(), HullGeometry()

    rows = component_table(env, hull)
    payload = {
        "protocol": "tensordem-threefield-resistance-audit-v1",
        "hull": hull.as_dict(),
        "ice": IceProperties().as_dict(),
        "environment": {
            "water_density_kg_m3": env.water_density_kg_m3,
            "gravity_m_s2": env.gravity_m_s2,
        },
        "study_speed_m_s": STUDY_SPEED,
        "components": rows,
        "reference_speed_identities": identity_report(env, hull),
        "inert_parameters": inert_report(env, hull),
        "citation_status": {
            "literature_agreement": "WITHHELD",
            "reason": "Lindqvist (1989) and Riska (1989) not available offline; "
                      "coefficients are implementation conventions pinned by "
                      "tests/test_threefield_resistance.py",
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "resistance_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    plot_thickness(rows, outdir / "figures" / "resistance_thickness_sweep.png")
    plot_speed(env, hull, 0.60, outdir / "figures" / "resistance_speed_sweep.png")

    print(f"wrote {json_path}")
    for row in rows:
        print(
            "h={thickness_m:.2f} crush={crushing_N:10.1f} bend={bending_N:9.1f} "
            "sub={submersion_N:10.1f} drag={added_drag_N:8.1f} "
            "Ln={lindqvist_total_N:10.1f} Rs={riska_total_N:10.1f} "
            "spread={band_spread_pct:5.1f}%".format(**row)
        )
    for ident in payload["reference_speed_identities"]:
        print(
            "h={thickness_m:.2f} c2_fric/hydrostatic={c2_friction_over_hydrostatic:.6f} "
            "(expect {c2_friction_expected:.6f}) c2_bend/R_bend={c2_bending_over_r_bend:.6f} "
            "(expect {c2_bending_expected:.6f})".format(**ident)
        )


if __name__ == "__main__":
    main()
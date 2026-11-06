"""Audit of the ice-sheet plate-on-elastic-foundation solution.

Produces ``docs/artifacts/plate_on_foundation/`` (machine-readable ``report.json``,
``summary.md`` and ``figure.png``).  Every number in the documentation page comes
from this script, so nothing is transcribed by hand.

    PYTHONPATH=src python scripts/audit_plate_on_foundation.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from tensordem.plate_on_foundation import (
    GRAVITY,
    SEA_WATER_DENSITY,
    WESTERGAARD_CONSTANT,
    characteristic_length,
    foundation_modulus,
    ice_sheet_bearing_capacity,
    kelvin_kei,
    patch_centre_moment,
    patch_load_deflection,
    patch_load_moments,
    plate_rigidity,
    point_load_deflection,
    point_load_moments,
    westergaard_max_stress,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "artifacts" / "plate_on_foundation"

YOUNG = 5.0e9
THICK = 1.0
NU = 0.3
D = plate_rigidity(YOUNG, THICK, NU)
K = foundation_modulus()
L = characteristic_length(D, K)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    peak_load = 1.0e5

    # -- 1. exact identities -------------------------------------------------
    w0 = float(point_load_deflection(0.0, load=peak_load, rigidity=D, modulus=K)[0])
    w0_closed = peak_load / (8.0 * math.sqrt(K * D))
    radial_fine = np.linspace(0.0, 60.0 * L, 240001)
    w_fine = point_load_deflection(radial_fine, load=1.0, rigidity=D, modulus=K)
    reaction = float(K * np.trapezoid(w_fine * 2.0 * math.pi * radial_fine, radial_fine))
    identities = {
        "point_centre_deflection_m": w0,
        "point_centre_deflection_closed_form_m": w0_closed,
        "point_centre_deflection_relative_error": abs(w0 - w0_closed) / w0_closed,
        "point_reaction_over_load": reaction,
        "point_reaction_relative_error": abs(reaction - 1.0),
    }

    # -- 2. the two routes ---------------------------------------------------
    routes = []
    for a_over_l in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        radius = a_over_l * L
        closed = patch_centre_moment(load=peak_load, radius=radius, rigidity=D, modulus=K)
        radial, hoop = patch_load_moments(0.0, load=peak_load, radius=radius, rigidity=D, modulus=K)
        w_patch = float(patch_load_deflection(
            0.0, load=peak_load, radius=radius, rigidity=D, modulus=K)[0])
        routes.append({
            "a_over_l": a_over_l,
            "centre_moment_closed_form_Nm": closed,
            "centre_moment_hankel_Nm": float(radial[0]),
            "centre_moment_relative_difference": abs(radial[0] - closed) / abs(closed),
            "hoop_over_radial": float(hoop[0] / radial[0]),
            "patch_over_point_deflection": w_patch / w0,
        })
    max_route_error = max(r["centre_moment_relative_difference"] for r in routes)
    max_equibiaxial_error = max(abs(r["hoop_over_radial"] - 1.0) for r in routes)

    # -- 3. published Westergaard constant ----------------------------------
    asymptote = []
    for eps in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        kei_prime = float(kelvin_kei(np.array([eps]))[1][0])
        asymptote.append({
            "a_over_l": eps,
            "extracted_constant": 2.0 / eps * kei_prime - math.log(1.0 / eps),
            "published_constant": WESTERGAARD_CONSTANT,
        })
    constant_error = abs(asymptote[-1]["extracted_constant"] - WESTERGAARD_CONSTANT)

    # -- 4. exact peak stress vs the published formula -----------------------
    stress = []
    for a_over_l in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        radius = a_over_l * L
        exact = 6.0 * patch_centre_moment(
            load=peak_load, radius=radius, rigidity=D, modulus=K) / THICK**2
        published = westergaard_max_stress(
            peak_load, radius=radius, thickness=THICK, young_modulus=YOUNG,
            poisson_ratio=NU, modulus=K)
        stress.append({
            "a_over_l": a_over_l,
            "exact_peak_stress_Pa": exact,
            "westergaard_peak_stress_Pa": published,
            "published_relative_error": (published - exact) / exact,
        })

    # -- 5. error of the published formula versus a/l (drives the 3 % gate) --
    worst_published_error = max(abs(s["published_relative_error"]) for s in stress)

    # -- 6. bearing capacity family -----------------------------------------
    capacity = [ice_sheet_bearing_capacity(thickness=h) for h in (0.3, 0.5, 1.0, 1.5, 2.0)]
    for row, h in zip(capacity, (0.3, 0.5, 1.0, 1.5, 2.0)):
        row["thickness_m"] = h

    report = {
        "protocol": "tensordem-ice-sheet-plate-on-foundation-v1",
        "material": {
            "young_modulus_Pa": YOUNG,
            "poisson_ratio": NU,
            "thickness_m": THICK,
            "water_density_kg_m3": SEA_WATER_DENSITY,
            "gravity_m_s2": GRAVITY,
            "rigidity_Nm": D,
            "foundation_modulus_N_per_m3": K,
            "characteristic_length_m": L,
        },
        "identities": identities,
        "routes": routes,
        "max_route_relative_difference": max_route_error,
        "max_equibiaxial_relative_difference": max_equibiaxial_error,
        "westergaard_constant_convergence": asymptote,
        "westergaard_constant_absolute_error": constant_error,
        "peak_stress_comparison": stress,
        "worst_published_relative_error": worst_published_error,
        "bearing_capacity": capacity,
    }

    # -- figures -------------------------------------------------------------
    figure_name = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
        r = np.linspace(0.0, 6.0 * L, 241)
        w_point = point_load_deflection(r, load=peak_load, rigidity=D, modulus=K) / w0
        m_point = point_load_moments(r[1:], load=peak_load, rigidity=D, modulus=K)[0]
        axes[0].plot(r / L, w_point, label="point load (Kelvin, exact)")
        rad_patch = np.linspace(0.0, 6.0 * L, 41)
        w_patch = patch_load_deflection(
            rad_patch, load=peak_load, radius=0.05 * L, rigidity=D, modulus=K)
        axes[0].plot(rad_patch / L, w_patch / w0, "--",
                     label="patch a/l = 0.05 (Hankel)")
        axes[0].set_xlabel("r / l")
        axes[0].set_ylabel("w / w(0)")
        axes[0].set_title("deflection profile")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].semilogy(r[1:] / L, m_point / m_point[0], label="point load (exact)")
        axes[1].semilogy([0.05], [patch_centre_moment(
            load=peak_load, radius=0.05 * L, rigidity=D, modulus=K) / m_point[0]], "o",
            label="patch centre a/l = 0.05")
        axes[1].set_xlabel("r / l")
        axes[1].set_ylabel("M_r(r) / M_r(r->0)")
        axes[1].set_title("radial moment decay")
        axes[1].grid(alpha=0.3, which="both")
        axes[1].legend(fontsize=8)

        eps_list = [row["a_over_l"] for row in stress]
        ratios = [row["westergaard_peak_stress_Pa"] / row["exact_peak_stress_Pa"] for row in stress]
        axes[2].semilogx(eps_list, ratios, "o-", label="published / exact")
        axes[2].axhline(1.03, color="r", ls=":", label="+3 % gate")
        axes[2].axhline(0.97, color="r", ls=":")
        axes[2].set_xlabel("a / l")
        axes[2].set_ylabel("sigma published / sigma exact")
        axes[2].set_title("Westergaard asymptote accuracy")
        axes[2].grid(alpha=0.3, which="both")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        figure_name = "figure.png"
        fig.savefig(OUT / figure_name, dpi=140)
        plt.close(fig)
    except ImportError:  # pragma: no cover - viz extra not installed
        figure_name = None

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")

    lines = [
        "# Ice-sheet plate-on-elastic-foundation audit (programmatic output)",
        "",
        f"- rigidity D = {D:.6g} N m, foundation k = {K:.6g} N/m^3, "
        f"characteristic length l = {L:.6g} m",
        f"- exact point-load identity w(0) = P/(8 sqrt(kD)): relative error "
        f"{identities['point_centre_deflection_relative_error']:.2e}",
        f"- point-load reaction / load = {reaction:.9f} "
        f"(relative error {identities['point_reaction_relative_error']:.2e})",
        f"- max centre-moment difference, closed form vs Hankel route: {max_route_error:.2e}",
        f"- max |M_t(0)/M_r(0) - 1| (equibiaxial): {max_equibiaxial_error:.2e}",
        f"- published Westergaard constant {WESTERGAARD_CONSTANT} vs extracted "
        f"{asymptote[-1]['extracted_constant']:.6f} at a/l = 1e-3 "
        f"(absolute error {constant_error:.2e})",
        f"- worst published-vs-exact peak-stress difference over a/l in "
        f"[0.005, 0.2]: {100.0 * worst_published_error:.3f} %",
        "",
        "| h (m) | l (m) | P_allow (kN) | sigma per N (Pa/N) | Westergaard rel. error |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in capacity:
        lines.append(
            f"| {row['thickness_m']:.1f} | {row['characteristic_length_m']:.3f} | "
            f"{row['allowable_load_N'] / 1e3:.1f} | {row['peak_stress_per_newton_Pa']:.6f} | "
            f"{row['westergaard_relative_error']:+.2e} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    report["figure"] = figure_name
    return report


if __name__ == "__main__":
    result = main()
    print(json.dumps({k: result[k] for k in (
        "max_route_relative_difference", "westergaard_constant_absolute_error",
        "worst_published_relative_error", "figure")}, indent=2))
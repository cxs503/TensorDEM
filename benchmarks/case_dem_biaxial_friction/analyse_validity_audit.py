#!/usr/bin/env python
"""Analysis for the anchor validity audit (convergence screen + c = 0 test).

Consumes
    runs/m21262/screen/<tag>.json     (screen_drivers.py)
    runs/m21262/pressure/<tag>.json   (audit_pressure_bonds.py)

Produces
    artifacts/CASE-DEM-BIAXIAL-FRICTION/validity_audit.json
    docs/data/CASE-DEM-BIAXIAL-FRICTION_validity_audit.json   (in-repo copy)
    docs/figures/CASE-DEM-BIAXIAL-FRICTION_validity_audit.png

Convention: the anchor defines the friction angle in the 2-D biaxial form
    sin(phi) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3) = q / (2 p),  p = (s1 + s3)/2
so a cohesive Mohr-Coulomb envelope in the same variables is
    q_peak = 2 c cos(phi) + 2 p_peak sin(phi)
and the fit of q_peak against p_peak gives slope/intercept -> (phi_fit, c).
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCREEN = os.path.join(REPO, "runs", "m21262", "screen")
PRESSURE = os.path.join(REPO, "runs", "m21262", "pressure")
ART = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION")
DOCDATA = os.path.join(REPO, "docs", "data")
FIGS = os.path.join(REPO, "docs", "figures")

PHI_REF_DEG = 38.0
TOL = 0.08

# size sweep order
SIZE_ORDER = ["s08x16", "s12x24", "s16x32", "s24x48"]
RATE_ORDER: list[str] = []
CONS_ORDER: list[str] = []
PRESS_ORDER = ["p050", "p100", "p200"]


def _load(d: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn)) as f:
            rec = json.load(f)
        out[rec["tag"]] = rec
    return out


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x; returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ssr / ss if ss > 0 else float("nan")
    return a, b, r2


def size_section(screen: dict[str, dict]) -> dict:
    rows = []
    for tag in SIZE_ORDER:
        if tag not in screen:
            continue
        r = screen[tag]
        n = r["specimen"].get("n_particles")
        rows.append({
            "tag": tag,
            "columns": r["overrides"]["columns"],
            "rows": r["overrides"]["rows"],
            "n_particles": n,
            "phi_wall_deg": r["phi_wall_deg"],
            "phi_internal_deg": r["phi_internal_deg"],
            "eps_a_peak": r["peak"]["axial_strain"],
            "dilation_deg": r["peak"]["dilation_angle_deg"],
        })
    spread = float("nan")
    if len(rows) >= 2:
        ph = [r["phi_wall_deg"] for r in rows]
        spread = max(ph) - min(ph)
    return {"rows": rows, "phi_spread_deg": spread}


def rate_section(screen: dict[str, dict]) -> dict:
    rows = []
    for tag in RATE_ORDER:
        if tag not in screen:
            continue
        r = screen[tag]
        rows.append({
            "tag": tag,
            "strain_rate": r["overrides"]["strain_rate"],
            "inertial_number": r["loading"].get("inertial_number"),
            "phi_wall_deg": r["phi_wall_deg"],
            "phi_internal_deg": r["phi_internal_deg"],
        })
    d = float("nan")
    if len(rows) >= 2:
        d = abs(rows[1]["phi_wall_deg"] - rows[0]["phi_wall_deg"])
    return {"rows": rows, "delta_phi_deg": d}


def cons_section(screen: dict[str, dict]) -> dict:
    rows = []
    for tag in CONS_ORDER:
        if tag not in screen:
            continue
        r = screen[tag]
        rows.append({
            "tag": tag,
            "consolidation_steps": r["overrides"]["consolidation_steps"],
            "phi_wall_deg": r["phi_wall_deg"],
            "phi_internal_deg": r["phi_internal_deg"],
        })
    d = float("nan")
    if len(rows) >= 2:
        d = abs(rows[1]["phi_wall_deg"] - rows[0]["phi_wall_deg"])
    return {"rows": rows, "delta_phi_deg": d}


def pressure_section(press: dict[str, dict]) -> dict:
    rows = []
    for tag in PRESS_ORDER:
        if tag not in press:
            continue
        r = press[tag]
        rows.append({
            "tag": tag,
            "confining_kPa": r["confining_kPa"],
            "q_kPa": r["q_kPa"],
            "p_kPa": r["p_kPa"],
            "phi_wall_deg": r["phi_wall_deg"],
            "phi_internal_deg": r["phi_internal_deg"],
        })
    fit = {"phi_fit_deg": float("nan"), "cohesion_c_kPa": float("nan"),
           "r2": float("nan"), "slope": float("nan"), "intercept_kPa": float("nan")}
    if len(rows) >= 2:
        a, b, r2 = _linfit([r["p_kPa"] for r in rows], [r["q_kPa"] for r in rows])
        sinphi = b / 2.0
        sinphi = max(-1.0, min(1.0, sinphi))
        phi_fit = math.degrees(math.asin(sinphi))
        c = a / (2.0 * math.cos(math.radians(phi_fit))) if math.isfinite(phi_fit) else float("nan")
        fit = {"phi_fit_deg": phi_fit, "cohesion_c_kPa": c, "r2": r2,
               "slope": b, "intercept_kPa": a}
    spread = float("nan")
    if len(rows) >= 2:
        ph = [r["phi_wall_deg"] for r in rows]
        spread = max(ph) - min(ph)
    bond = None
    if "p100_b0" in press and "p100" in press:
        bond = {
            "phi_bonded_deg": press["p100"]["phi_wall_deg"],
            "phi_unbonded_deg": press["p100_b0"]["phi_wall_deg"],
            "delta_phi_deg": press["p100"]["phi_wall_deg"] - press["p100_b0"]["phi_wall_deg"],
            "phi_unbonded_error_vs_ref": abs(press["p100_b0"]["phi_wall_deg"] - PHI_REF_DEG) / PHI_REF_DEG,
        }
    return {"rows": rows, "fit": fit, "phi_spread_deg": spread, "bond_control": bond}


def make_figure(res: dict, path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable: {exc}")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ps = res["pressure"]
    if ps["rows"]:
        p = [r["confining_kPa"] for r in ps["rows"]]
        q = [r["q_kPa"] for r in ps["rows"]]
        ax.plot(p, q, "o-", label="peak envelope")
        if math.isfinite(ps["fit"]["intercept_kPa"]):
            xx = [min(p), max(p)]
            yy = [ps["fit"]["intercept_kPa"] + ps["fit"]["slope"] * x for x in xx]
            ax.plot(xx, yy, "--", lw=1,
                    label=(f"fit: $\\phi$={ps['fit']['phi_fit_deg']:.1f}$^\\circ$, "
                           f"c={ps['fit']['cohesion_c_kPa']:.1f} kPa"))
        ax.set_xlabel(r"peak mean stress $p$ [kPa]")
        ax.set_ylabel(r"peak deviator $q$ [kPa]")
        ax.set_title("c = 0 test: $q_{peak}$ vs $p_{peak}$")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)

    ax = axes[1]
    if ps["rows"]:
        p = [r["confining_kPa"] for r in ps["rows"]]
        ph = [r["phi_wall_deg"] for r in ps["rows"]]
        ax.plot(p, ph, "s-", label="bonded (f_b = 0.35)")
        if ps["bond_control"]:
            ax.axhline(ps["bond_control"]["phi_unbonded_deg"], color="tab:red", ls="--",
                       label=f"unbonded (f_b = 0) = {ps['bond_control']['phi_unbonded_deg']:.1f}$^\\circ$")
        ax.axhline(PHI_REF_DEG, color="k", ls=":", lw=1, label="ref 38$^\\circ$ (Bolton)")
        ax.fill_between([min(p), max(p)], PHI_REF_DEG * (1 - TOL), PHI_REF_DEG * (1 + TOL),
                        color="k", alpha=.08, label="8 % band")
        ax.set_xlabel(r"confining pressure $\sigma_3$ [kPa]")
        ax.set_ylabel(r"apparent $\phi$ [deg]")
        ax.set_title(r"$\phi$($\sigma_3$): pressure independence?")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)

    ax = axes[2]
    sc = res["size"]
    if sc["rows"]:
        n = [r["n_particles"] for r in sc["rows"]]
        ph = [r["phi_wall_deg"] for r in sc["rows"]]
        phi_i = [r["phi_internal_deg"] for r in sc["rows"]]
        ax.plot(n, ph, "o-", label="wall traction")
        ax.plot(n, phi_i, "^-", label="Love-Weber internal")
        ax.axhline(PHI_REF_DEG, color="k", ls=":", lw=1, label="ref 38$^\\circ$")
        ax.axhspan(PHI_REF_DEG * (1 - TOL), PHI_REF_DEG * (1 + TOL), color="k", alpha=.08)
        ax.set_xlabel("particle count")
        ax.set_ylabel(r"peak $\phi$ [deg]")
        ax.set_title("finite-size sensitivity")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    screen = _load(SCREEN)
    press = _load(PRESSURE)
    res = {
        "case_key": "CASE-DEM-BIAXIAL-FRICTION",
        "phi_ref_deg": PHI_REF_DEG,
        "tolerance": TOL,
        "screen_tags": sorted(screen),
        "pressure_tags": sorted(press),
        "size": size_section(screen),
        "rate": rate_section(screen),
        "consolidation": cons_section(screen),
        "pressure": pressure_section(press),
    }
    res["figure"] = make_figure(res, os.path.join(
        FIGS, "CASE-DEM-BIAXIAL-FRICTION_validity_audit.png"))

    for d in (ART, DOCDATA):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "validity_audit.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)

    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
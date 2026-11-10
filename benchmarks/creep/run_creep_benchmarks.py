"""Driver for the ice-creep benchmark suite (mission P46W3-N02).

Runs the four acceptance gates plus the literature-anchored checks, writes
``artifacts/ice_creep_p46w3n02/results.json`` (every number quoted in the report
comes from here - nothing is transcribed by hand) and six figures.

Usage
-----
    PYTHONPATH=src python benchmarks/creep/run_creep_benchmarks.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensordem.ice_creep import GlenCreepLaw, LatticeCreepParameters, uniaxial_constant_stress  # noqa: E402
from tensordem.ice_creep_andrade import (  # noqa: E402
    ALPHA_ANDRADE,
    AndradeCreepLaw,
    FractionalDemChain,
    fractional_relaxation_reference,
)

from creep.gates import (  # noqa: E402
    YOUNG_ICE,
    andrade_law_for_retardation,
    gate_andrade_curve,
    gate_glen_exponent,
    gate_negative_control,
    gate_relaxation,
    gate_summary,
)
from creep.literature import literature_checks  # noqa: E402

OUT_DIR = REPO_ROOT / "artifacts" / "ice_creep_p46w3n02"
MISSION = "P46W3-N02"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- #
# serialisation helpers
# --------------------------------------------------------------------------- #
def _trim(array, limit: int = 240):
    arr = np.asarray(array, dtype=float).ravel()
    if arr.size <= limit:
        return arr.tolist()
    idx = np.unique(np.linspace(0, arr.size - 1, limit).astype(int))
    return arr[idx].tolist()


def jsonable(obj, limit: int = 240):
    """Recursively convert numpy/scipy output into JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, limit) for v in obj]
    if isinstance(obj, np.ndarray):
        return _trim(obj, limit)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_andrade_creep(gate_b: dict, path: Path) -> dict:
    t = np.array(gate_b["history"]["t"])
    strain = np.array(gate_b["history"]["strain"])
    analytic = np.array(gate_b["history"]["analytic"])
    creep = np.array(gate_b["history"]["creep_strain"])
    fit = gate_b["fit_after_startup"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax1.loglog(t[1:], strain[1:], "o", ms=2.4, label="DEM chain (8 bonds)")
    ax1.loglog(t[1:], analytic[1:], "-", lw=1.2,
               label=r"analytic $\varepsilon_0+\varepsilon_1 t^{1/3}$")
    ax1.loglog(t[1:], fit["eps_0"] + fit["eps_1"] * t[1:] ** fit["alpha"], "--",
               lw=1.0, label=r"fit $\alpha$=%.4f" % fit["alpha"])
    ax1.set_xlabel("time t [s]")
    ax1.set_ylabel("strain [-]")
    ax1.set_title("(b) Andrade primary creep: DEM vs analytic")
    ax1.legend(loc="lower right", fontsize=7)

    rel = np.abs(strain - analytic) / np.abs(analytic)
    ax2.semilogx(t[1:], rel[1:], "-", lw=1.0)
    ax2.axhline(0.05, color="crimson", ls="--", lw=1.0, label="5 % criterion")
    ax2.set_xlabel("time t [s]")
    ax2.set_ylabel("relative error [-]")
    ax2.set_ylim(0, max(0.06, 1.2 * float(rel[1:].max())))
    ax2.set_title("curve error, max = %.2e" % float(rel[1:].max()))
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"file": path.name,
            "caption": "Andrade primary creep of a fractional-Maxwell DEM bond chain "
                       "(8 bonds) against eps_0 + eps_1 t^(1/3); right panel: pointwise "
                       "relative error with the 5 % acceptance line."}


def fig_ml_relaxation(gate_c: dict, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colours = {1.0: "tab:blue", 0.5: "tab:orange", ALPHA_ANDRADE: "tab:green"}
    for alpha, colour in colours.items():
        law = andrade_law_for_retardation(alpha)
        chain = FractionalDemChain(n_bonds=8, law=law)
        run = chain.relax(1.0e-3, 8.0 * chain.retardation_time, 2000)
        norm = run["t"] / run["tau"]
        ax.semilogy(norm, run["force_normalised"], marker="o", ls="none", ms=1.8,
                    color=colour, label=f"DEM alpha = {alpha:.4f}")
        ref = fractional_relaxation_reference(run["t"], run["tau"], alpha)
        ax.semilogy(norm, ref, ls="-", lw=1.0, color=colour, alpha=0.8)
    ax.set_xlabel(r"$t/\tau$,  $\tau=(\eta_\alpha/E)^{1/\alpha}$")
    ax.set_ylabel(r"$F(t)/F_0$")
    ax.set_title("(c) fractional relaxation: DEM (points) vs Mittag-Leffler (lines)")
    ax.legend(fontsize=7)
    ax.set_ylim(1e-3, 1.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"file": path.name,
            "caption": "Stress relaxation at constant elongation for alpha = 1, 1/2, 1/3. "
                       "alpha = 1 is the exponential, alpha = 1/2 the exp(x^2)erfc(x) "
                       "identity, alpha = 1/3 the Andrade element."}


def fig_tau_fluidity(gate_c: dict, path: Path) -> dict:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    rows = gate_c["norton_bond_sweep"]["rows"]
    a = np.array([r["fluidity"] for r in rows])
    tau = np.array([r["tau_eff_s"] for r in rows])
    ax1.loglog(a, tau, "o", label="single Norton bond (implicit Euler)")
    l_rows = gate_c["lattice_punch_sweep"]["rows"]
    la = np.array([r["fluidity"] for r in l_rows])
    lt = np.array([r["tau_eff_s"] for r in l_rows])
    ax1.loglog(la, lt, "s", label="DEM flat-punch block, level 0.9")
    ref = tau[0] * (a / a[0]) ** -1.0
    ax1.loglog(a, ref, "k--", lw=0.9, label=r"$\tau\propto A^{-1}$")
    corr = gate_c["norton_bond_sweep"]["correlation"]
    ax1.set_xlabel("fluidity A [Pa$^{-3}$ s$^{-1}$]")
    ax1.set_ylabel(r"$\tau_{\rm eff}$ [s]")
    ax1.set_title("Norton: slope %.4f, r = %.5f" % (corr["exponent"], corr["pearson_r"]))
    ax1.legend(fontsize=7)

    frows = gate_c["fractional_chain_sweep"]["rows"]
    eta = np.array([r["eta_alpha"] for r in frows])
    ftau = np.array([r["tau_eff_s"] for r in frows])
    ax2.loglog(eta, ftau, "o", label=r"fractional chain, $\alpha=1/3$")
    fcorr = gate_c["fractional_chain_sweep"]["correlation"]
    ax2.loglog(eta, ftau[0] * (eta / eta[0]) ** fcorr["exponent"], "k--", lw=0.9,
               label="fit slope %.3f (theory 3)" % fcorr["exponent"])
    ax2.set_xlabel(r"springpot viscosity $\eta_\alpha$ [Pa s$^\alpha$]")
    ax2.set_ylabel(r"$\tau_{\rm eff}$ [s]")
    ax2.set_title("fractional: r = %.6f" % fcorr["pearson_r"])
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"file": path.name,
            "caption": "Relaxation time versus creep parameter. Left: Norton bond and "
                       "many-body DEM flat punch, both with slope -1 in log-log. Right: "
                       "fractional chain, slope 1/alpha = 3."}


def fig_glen_exponent(gate_a: dict, path: Path) -> dict:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    sigmas = np.array(gate_a["sigmas_Pa"], dtype=float)
    rates = np.array(gate_a["per_window"]["10-40s"]["rates"], dtype=float)
    ax1.loglog(sigmas, rates, "o", label="DEM creep (3x3x3 particles)")
    fit_line = rates[0] * (sigmas / sigmas[0]) ** gate_a["fitted_n"]
    ax1.loglog(sigmas, fit_line, "-", lw=1.0,
               label="fit n = %.4f (r2 %.5f)" % (gate_a["fitted_n"],
                                                 gate_a["per_window"]["10-40s"]["r2"]))
    ax1.loglog(sigmas, rates[0] * (sigmas / sigmas[0]) ** 3.0, "k--", lw=0.9,
               label="literature n = 3")
    mc = gate_a["noise_mc"]
    ax1.set_xlabel(r"applied stress $\sigma$ [Pa]")
    ax1.set_ylabel(r"steady creep rate $\dot\varepsilon$ [s$^{-1}$]")
    ax1.set_title("(a) Glen exponent from DEM creep")
    ax1.legend(fontsize=7)

    res = gate_a["resolution_study"]
    keys = list(res)
    vals = [res[k]["exponent"] for k in keys]
    ax2.plot(range(len(keys)), vals, "o-")
    ax2.axhline(3.0, color="crimson", ls="--", lw=1.0, label="literature n = 3")
    ax2.axhspan(mc["ci95"][0], mc["ci95"][1], color="tab:blue", alpha=0.15,
                label="noise MC 95 % CI")
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels([f"{k}³" for k in keys])
    ax2.set_xlabel("specimen resolution [particles]")
    ax2.set_ylabel("fitted n")
    ax2.set_title("systematic lattice bias (resolution study)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"file": path.name,
            "caption": "Glen exponent recovery. Left: simulated steady creep rate vs "
                       "stress, fitted n = 2.913. Right: the fitted exponent versus "
                       "specimen resolution against the statistical 95 % Monte-Carlo "
                       "band - the bias is systematic, not statistical."}


def fig_negative_control(gate_d: dict, path: Path) -> dict:
    sigma0 = gate_d["sigma0_Pa"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    law = AndradeCreepLaw(alpha=ALPHA_ANDRADE, eta_alpha=4.677e10)
    t = np.linspace(0.0, 1000.0, 400)
    for mult, label in ((1.0, "nominal"), (1.0e2, "eta x 100"), (1.0e4, "eta x 1e4"),
                        (1.0e16 / 4.677e10, "eta -> inf (no creep)")):
        stiff = FractionalDemChain(n_bonds=8, law=AndradeCreepLaw(
            alpha=ALPHA_ANDRADE, eta_alpha=law.eta_alpha * mult))
        run = stiff.creep(sigma0, 1000.0, 600)
        ax.semilogx(np.array(run["t"])[1:], np.array(run["strain"])[1:], lw=1.1,
                    label=f"{label}")
    ax.axhline(sigma0 / YOUNG_ICE, color="k", ls=":", lw=1.0,
               label=r"$\sigma/E$ (elastic limit)")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("strain [-]")
    ax.set_title("(d) negative control: creep collapses with eta")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"file": path.name,
            "caption": "Negative control: as the springpot viscosity grows, the strain "
                       "history collapses onto the instantaneous elastic value sigma/E."}


def fig_literature(lit: dict, gate_a: dict, gate_b: dict, path: Path) -> dict:
    lit_items = lit["default_configuration"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    items = [
        ("A(-10 C)\n[Pa^-3 s^-1]", lit_items["fluidity_minus10C"]["band"],
         lit_items["fluidity_minus10C"]["value"], "A"),
        ("eta_eff(1 MPa)\n[Pa s]", lit_items["eta_eff_1MPa"]["band"],
         lit_items["eta_eff_1MPa"]["value"], None),
        ("edot_min(1 MPa)\n[s^-1]", lit_items["min_creep_rate_1MPa_minus10C"]["band"],
         lit_items["min_creep_rate_1MPa_minus10C"]["value"], None),
    ]
    for i, (label, band, value, _) in enumerate(items):
        low, high = band
        ax.barh(i, math.log10(high / low), left=math.log10(low), height=0.45,
                color="tab:blue", alpha=0.25, label="literature band" if i == 0 else None)
        ax.plot([math.log10(value)], [i], "o", color="tab:blue", ms=6,
                label="literature reference" if i == 0 else None)
        if _ == "A":
            ax.plot([math.log10(3.5e-24)], [i], "D", color="tab:red", ms=6,
                    label="model default (3.5e-24)")
            ax.plot([math.log10(3.5e-25)], [i], "^", color="tab:green", ms=6,
                    label="literature-calibrated (3.5e-25)")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([row[0] for row in items])
    ax.set_xlabel("log10(value)")
    ax.set_title("literature bands and model anchors")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"file": path.name,
            "caption": "Literature bands (blue) with the model anchors. Curve-level "
                       "overlays of Cole (1998) / Schulson (2001) are WITHHELD: no "
                       "digitized data are bundled in the repository."}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] gate (a): Glen exponent ...")
    gate_a = gate_glen_exponent()
    print("      fitted n = %.4f (rel. err %.2f %%), mission criterion %s"
          % (gate_a["fitted_n"], 100 * gate_a["rel_error_vs_literature"],
             "PASS" if gate_a["mission_criterion_pass"] else "FAIL"))

    print("[2/4] gate (b): Andrade creep curve ...")
    gate_b = gate_andrade_curve()
    print("      max curve error %.3e, fitted alpha %.5f"
          % (gate_b["max_rel_error_curve"], gate_b["fit_after_startup"]["alpha"]))

    print("[3/4] gate (c): relaxation and tau(A) ...")
    gate_c = gate_relaxation()
    print("      Norton r = %.6f slope %.6f | fractional r = %.6f | lattice r = %.6f"
          % (gate_c["norton_bond_sweep"]["correlation"]["pearson_r"],
             gate_c["norton_bond_sweep"]["correlation"]["exponent"],
             gate_c["fractional_chain_sweep"]["correlation"]["pearson_r"],
             gate_c["lattice_punch_sweep"]["correlation"]["pearson_r"]))

    print("[4/4] gate (d): negative control ...")
    gate_d = gate_negative_control()
    print("      no-creep collapse: Norton %.2e, fractional %.2e"
          % (gate_d["norton"]["rel_change"],
             gate_d["fractional"]["creep_over_elastic"]))

    summary = gate_summary({"a": gate_a, "b": gate_b, "c": gate_c, "d": gate_d})

    sigma_ref = 1.0e6
    alpha_fit = gate_b["fit_after_startup"]["alpha"]

    def _literature_for(law):
        a_minus10 = law.fluidity(-10.0)
        rate = float(law.uniaxial_rate(sigma_ref, -10.0))
        return a_minus10, sigma_ref / rate, literature_checks({
            "n_fit": gate_a["fitted_n"],
            "a_minus10": a_minus10,
            "q_kj_per_mol": law.q_over_r * 8.314462618 / 1000.0,
            "eta_eff_1mpa": sigma_ref / rate,
            "min_rate_1mpa_minus10": rate,
            "alpha_fit": alpha_fit,
        })

    # the module default (m6910) and the literature-calibrated fluidity are both
    # reported: the default sits at the fast end of the literature spread, which
    # the predeclared +-1 decade band around the Barnes anchor does not accept
    law_default = GlenCreepLaw()
    a_minus10, eta_eff, literature_default = _literature_for(law_default)
    law_calibrated = GlenCreepLaw(a_ref=3.5e-25)
    a_cal, eta_cal, literature_calibrated = _literature_for(law_calibrated)
    literature = {
        "default_configuration": literature_default,
        "literature_calibrated_configuration": literature_calibrated,
        "summary": literature_default["summary"],
        "calibrated_summary": literature_calibrated["summary"],
        "note": "the predeclared band around the Barnes (1971) anchor is +-1 decade; "
                "the module default A(-10 C) = 3.5e-24 Pa^-3 s^-1 lies one decade "
                "above it (fast end of the literature spread) and is reported as a "
                "FAIL of that item, not silently re-banded",
    }

    figures = [
        fig_andrade_creep(gate_b, OUT_DIR / "fig_andrade_creep.png"),
        fig_ml_relaxation(gate_c, OUT_DIR / "fig_ml_relaxation.png"),
        fig_tau_fluidity(gate_c, OUT_DIR / "fig_tau_fluidity.png"),
        fig_glen_exponent(gate_a, OUT_DIR / "fig_glen_exponent.png"),
        fig_negative_control(gate_d, OUT_DIR / "fig_negative_control.png"),
        fig_literature(literature, gate_a, gate_b, OUT_DIR / "fig_literature.png"),
    ]

    results = {
        "mission": MISSION,
        "repository": "TensorDEM",
        "module": "src/tensordem/ice_creep_andrade.py",
        "preexisting_module": "src/tensordem/ice_creep.py (mission m6910, Norton/Glen side)",
        "gates": {"a": gate_a, "b": gate_b, "c": gate_c, "d": gate_d},
        "gate_summary": summary,
        "literature": literature,
        "model_anchors": {
            "A_minus10C": a_minus10,
            "A_minus10C_literature_calibrated": a_cal,
            "eta_eff_1MPa": eta_eff,
            "eta_eff_1MPa_literature_calibrated": eta_cal,
            "q_kJ_per_mol": law_default.q_over_r * 8.314462618 / 1000.0,
            "andrade_alpha": ALPHA_ANDRADE,
        },
        "figures": figures,
    }
    payload = jsonable(results)
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2),
                                          encoding="utf-8")

    print("\ngate summary:", summary)
    print("literature (default A):", literature_default["summary"])
    print("literature (calibrated A):", literature_calibrated["summary"])
    print("written:", OUT_DIR / "results.json")
    return payload


if __name__ == "__main__":
    main()
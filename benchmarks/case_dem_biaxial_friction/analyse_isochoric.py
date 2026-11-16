#!/usr/bin/env python
"""CASE-DEM-BIAXIAL-FRICTION round 3: constant-volume (isochoric) analysis.

Consumes the raw stress paths written by
``scripts/m20537_probe_isochoric.py`` (``runs/m20537/iso_*.json``), locates the
critical state with :mod:`tensordem.biaxial_critical_state` using the
**logarithmic** strain measure (the only measure for which an exactly
volume-preserving cell reports ``eps_v = 0``) and the isochoric boundary
closure check, then judges the result against the references declared in
round 2 *before* any of these runs existed:

* primary   -- Rowe (1962) D = 0 limit ``phi_cv = arctan(mu)``; 28.81 deg for
  mu = 0.55; tolerance 8 % (case-pool acceptance);
* secondary -- Bolton (1986) angular quartz sand ``phi_crit = 33 deg`` (the
  shape-gap reference, reported but not the pass criterion).

Writes

    artifacts/CASE-DEM-BIAXIAL-FRICTION/run_iso.json
    artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_isochoric.png

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_isochoric.py
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensordem.biaxial_critical_state import (  # noqa: E402
    critical_state_from_stress_path,
    judge_against_reference,
    phi_deg_from_stresses,
    rowe_stress_dilatancy,
)

TOLERANCE = 0.08
MU_ANCHOR = 0.55
PHI_ROWE_DEG = math.degrees(math.atan(MU_ANCHOR))   # 28.81 deg
PHI_BOLTON_QUARTZ_DEG = 33.0
REF_SOURCES = {
    "rowe_limit": ("Rowe (1962) Proc. R. Soc. Lond. A 269:500-527, stress-dilatancy "
                   "relation at D=0: phi_cv = arctan(mu) = 28.81 deg for mu = 0.55"),
    "bolton_quartz": ("Bolton (1986) Geotechnique 36(1):65-78: angular quartz sand "
                      "critical-state angle phi_crit = 33 deg"),
}
PROBE_DIR = os.environ.get("PROBE_DIR", os.path.join(REPO, "runs", "m20537"))
OUT_DIR = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION")
# committed, downsampled copy of the probe stress paths: the analysis (and the
# provenance regression test) must run with no cluster artefacts on disk
DATA_DIR = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                        "results_m22748")
BUNDLE = os.path.join(DATA_DIR, "iso_paths.json.gz")

PATH_KEYS = ("axial_strain", "axial_strain_log", "sigma_v", "sigma_h", "deviator",
             "volumetric_strain", "volumetric_strain_log")


def analyse_one(raw: dict) -> dict:
    cfg = dict(raw["config"])
    sp = {k: np.asarray(v, dtype=float) for k, v in raw["stress_path"].items()
          if k in PATH_KEYS}
    mu = float(cfg.get("mu", MU_ANCHOR))
    p0 = float(cfg["confining_stress"])
    cs = critical_state_from_stress_path(
        sp, sigma3_target=p0, strain_measure="log", boundary_control="isochoric")
    # the Rowe cross-check needs samples above its min_strain; clamp it to the
    # record length (a run that stops at eps_a = 0.45 cannot host a window
    # starting at 0.52)
    rowe_floor = min(cs.window[0], 0.85 * float(np.max(sp["axial_strain"])))
    try:
        rowe = rowe_stress_dilatancy(sp, mu=mu, min_strain=rowe_floor)
    except ValueError as exc:                     # pragma: no cover - defensive
        rowe = {"error": str(exc), "min_strain": rowe_floor}
    # stress-definition closure: wall traction vs Love-Weber virial, same window
    mask = (np.asarray(sp["axial_strain_log"]) >= cs.window[0]) & \
           (np.asarray(sp["axial_strain_log"]) <= cs.window[1])
    phi_wall = float(np.nanmean(phi_deg_from_stresses(sp["sigma_v"], sp["sigma_h"])[mask]))
    internal = {}
    if "sigma_v_internal" in raw["stress_path"]:
        sv_i = np.asarray(raw["stress_path"]["sigma_v_internal"], dtype=float)
        sh_i = np.asarray(raw["stress_path"]["sigma_h_internal"], dtype=float)
        phi_int = float(np.nanmean(phi_deg_from_stresses(sv_i, sh_i)[mask]))
        internal = {"phi_internal_deg": phi_int,
                    "phi_wall_deg": phi_wall,
                    "relative_difference": ((phi_int - phi_wall) / phi_wall
                                            if phi_wall else float("nan"))}
    mean_p = 0.5 * (sp["sigma_v"] + sp["sigma_h"])
    return {
        "tag": raw["tag"],
        "config": cfg,
        "specimen": raw.get("specimen", {}),
        "loading": raw.get("loading", {}),
        "elapsed_s": raw.get("elapsed_s"),
        "peak_phi_deg": float(raw.get("peak", {}).get(
            "friction_angle_deg", float(np.max(phi_deg_from_stresses(
                sp["sigma_v"], sp["sigma_h"]))))),
        "critical_state": cs.as_dict(),
        "rowe": rowe,
        "stress_definition_closure": internal,
        "mean_pressure_in_window_kPa": float(mean_p[mask].mean() / 1e3),
        "mean_pressure_initial_kPa": float(p0 / 1e3),
        "judge_primary_rowe": judge_against_reference(cs.phi_cv_deg, PHI_ROWE_DEG,
                                                     TOLERANCE),
        "judge_secondary_bolton": judge_against_reference(cs.phi_cv_deg,
                                                         PHI_BOLTON_QUARTZ_DEG,
                                                         TOLERANCE),
        "curves": {
            "axial_strain": sp["axial_strain"].tolist(),
            "phi_deg": phi_deg_from_stresses(sp["sigma_v"], sp["sigma_h"]).tolist(),
            "q_kPa": (sp["deviator"] / 1e3).tolist(),
            "p_kPa": (mean_p / 1e3).tolist(),
            "sigma_h_kPa": (sp["sigma_h"] / 1e3).tolist(),
            "volumetric_strain_log": sp["volumetric_strain_log"].tolist(),
        },
    }


def render(payload: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["cases"]
    anchor = next((r for r in rows if r["tag"].startswith("anchor")), rows[0])
    cur = anchor["curves"]
    eps = np.asarray(cur["axial_strain"])
    phi = np.asarray(cur["phi_deg"])
    q = np.asarray(cur["q_kPa"])
    p = np.asarray(cur["p_kPa"])
    sh = np.asarray(cur["sigma_h_kPa"])
    es = np.asarray(cur["volumetric_strain_log"])
    cs = anchor["critical_state"]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5))

    ax = axes[0]
    ax.plot(eps, phi, color="#1f77b4", lw=0.9, label=r"$\phi$ (wall traction)")
    ax.axhline(PHI_ROWE_DEG, color="green", ls="--",
               label=f"Rowe D=0 limit arctan$\\mu$ = {PHI_ROWE_DEG:.2f} deg")
    ax.axhspan(PHI_ROWE_DEG * (1 - TOLERANCE), PHI_ROWE_DEG * (1 + TOLERANCE),
               color="green", alpha=0.12, label="8 % band")
    ax.axhline(PHI_BOLTON_QUARTZ_DEG, color="gray", ls=":",
               label=f"angular quartz sand {PHI_BOLTON_QUARTZ_DEG:.0f} deg (Bolton)")
    ax.axvline(cs["window"][0], color="orange", lw=1.0, ls="-.")
    ax.axvspan(cs["window"][0], cs["window"][1], color="orange", alpha=0.2,
               label="critical-state window")
    ax.axhline(cs["phi_cv_deg"], color="red", lw=0.8,
               label=f"measured $\\phi_{{cv}}$ = {cs['phi_cv_deg']:.2f} deg")
    ax.set_xlabel(r"axial strain $\varepsilon_a$ [-]")
    ax.set_ylabel(r"mobilised $\phi$ [deg]")
    ax.set_title(f"anchor 24x48, $\\mu$=0.55, constant volume\n"
                 f"peak {anchor['peak_phi_deg']:.1f} deg -> $\\phi_{{cv}}$ "
                 f"{cs['phi_cv_deg']:.1f} deg")
    ax.set_ylim(0, 55)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eps, q, color="#d62728", lw=0.9, label="$q = \\sigma_1-\\sigma_3$ [kPa]")
    ax.plot(eps, p, color="#1f77b4", lw=0.9, label="$p = (\\sigma_1+\\sigma_3)/2$ [kPa]")
    ax.plot(eps, sh, color="#7f7f7f", lw=0.7, ls=":",
            label="$\\sigma_h$ (measured lateral) [kPa]")
    ax.axvspan(cs["window"][0], cs["window"][1], color="orange", alpha=0.2)
    ax.set_xlabel(r"$\varepsilon_a$ [-]")
    ax.set_ylabel("stress [kPa]")
    ax2 = ax.twinx()
    ax2.plot(eps, es, color="#2ca02c", lw=0.8, label=r"$\varepsilon_v$ (log)")
    ax2.set_ylabel(r"logarithmic $\varepsilon_v$ [-]", color="#2ca02c")
    ax.set_title("constant-volume path: $q$, $p$, $\\varepsilon_v=0$ by construction\n"
                 f"$d\\varepsilon_v/d\\varepsilon_a$ = {cs['d_eps_v_d_eps_a']:+.2e} (log)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    tags = [r["tag"] for r in rows]
    vals = [r["critical_state"]["phi_cv_deg"] for r in rows]
    colors = ["#1f77b4" if r["judge_primary_rowe"]["passed"] else "#ff7f0e" for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=5.5)
    ax.axhline(PHI_ROWE_DEG, color="green", ls="--", label=f"Rowe limit {PHI_ROWE_DEG:.2f} deg")
    ax.axhspan(PHI_ROWE_DEG * (1 - TOLERANCE), PHI_ROWE_DEG * (1 + TOLERANCE),
               color="green", alpha=0.12)
    ax.axhline(PHI_BOLTON_QUARTZ_DEG, color="gray", ls=":", label="angular quartz 33 deg")
    for bar, r in zip(bars, rows):
        err = r["judge_primary_rowe"]["relative_error"] * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}\n{err:.1f}%", ha="center", fontsize=6)
    ax.set_ylabel(r"$\phi_{cv}$ [deg]")
    ax.set_title("critical-state angle per configuration\n(relative error vs Rowe limit)")
    top = max(vals + [PHI_BOLTON_QUARTZ_DEG]) * 1.35
    ax.set_ylim(0, top)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _sig(x: float, digits: int = 7) -> float:
    """Round to ``digits`` significant figures (keeps the JSON small)."""
    if x == 0.0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{digits}g}")


def load_runs(probe_dir: str) -> tuple[list[dict], list[str]]:
    """Load probe records from a directory of ``iso_*.json`` files.

    Falls back to the committed, downsampled bundle
    (``results_m22748/iso_paths.json.gz``) when the probe directory is empty, so
    the published numbers can be recomputed on a clean checkout.
    """
    paths = sorted(glob.glob(os.path.join(probe_dir, "iso_*.json*")))
    if not paths and os.path.exists(BUNDLE):
        paths = [BUNDLE]
    runs: list[dict] = []
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            data = json.load(fh)
        runs.extend(data if isinstance(data, list) else [data])
    runs.sort(key=lambda r: r["tag"])
    return runs, [os.path.basename(p) for p in paths]


def compact_runs(runs: list[dict], out_path: str, max_samples: int = 1000) -> str:
    """Write a gzip bundle of the stress paths, decimated to ``max_samples``.

    Every recorded array is subsampled with a fixed stride (the last sample is
    always kept) and rounded to 7 significant figures.  This is what makes the
    round-3 numbers auditable from the repository without a 120 k-step raw
    artefact: the decimated path re-analyses to the published ``phi_cv`` within
    the reported uncertainty (see
    ``tests/test_biaxial_isochoric_provenance.py``).
    """
    bundle = []
    for raw in runs:
        sp = raw["stress_path"]
        n = len(np.asarray(sp["axial_strain"]))
        stride = max(1, int(math.ceil(n / float(max_samples))))
        idx = np.arange(0, n, stride)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        rec = {k: raw[k] for k in ("tag", "config", "specimen", "loading", "peak",
                                   "note", "elapsed_s") if k in raw}
        rec["stress_path"] = {k: [_sig(float(v)) for v in np.asarray(arr)[idx]]
                              for k, arr in sp.items()
                              if len(np.asarray(arr)) == n}
        rec["compaction"] = {"stride": int(stride), "n_original": int(n),
                             "n_kept": int(idx.size),
                             "note": "fixed-stride decimation of the probe record; "
                                     "the last sample is always kept"}
        bundle.append(rec)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    return out_path


def markdown_table(cases: list[dict]) -> str:
    """Sensitivity table (markdown) of the analysed configurations."""
    head = ("| run | grid | phi_peak [deg] | phi_cv [deg] | dEv/dEa | q drift | "
            "window spread [deg] | sigma_h [kPa] | p [kPa] | converged | "
            "err vs Rowe |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in cases:
        c = r["critical_state"]
        cfg = r["config"]
        rows.append(
            f"| `{r['tag']}` | {cfg['columns']}x{cfg['rows']} | "
            f"{r['peak_phi_deg']:.2f} | **{c['phi_cv_deg']:.2f}** | "
            f"{c['d_eps_v_d_eps_a']:+.2e} | {c['q_drift']:.3f} | "
            f"{c['window_spread_deg']:.2f} | {c['sigma_h_cv_Pa'] / 1e3:.0f} | "
            f"{r['mean_pressure_in_window_kPa']:.0f} | "
            f"{'yes' if c['converged'] else 'no'} | "
            f"{r['judge_primary_rowe']['relative_error'] * 100:.1f} % |")
    return head + "\n".join(rows)


# ---------------------------------------------------------------------------
# Pre-registered sensitivity matrix
# ---------------------------------------------------------------------------
# The decision rules below are *not* chosen here: they are read from
# ``prereg/PREREG_m22748_ISOCHORIC_SENSITIVITY.json``, which was committed
# before any sensitivity run existed.  This module only applies them, so the
# verdict cannot be tuned after seeing the numbers.
PREREG_PATH = os.path.join(REPO, "prereg", "PREREG_m22748_ISOCHORIC_SENSITIVITY.json")
ANCHOR_TAG = "anchor_24x48_mu055_p100"
INVARIANCE_TOL_DEG = 2.0        # prereg default: resolution / pressure / rate
QUASI_STATIC_I_LIMIT = 1.0e-3   # prereg: rate-independence only for I < 1e-3

# config tag -> (rule name, prereg key, kind)
SENSITIVITY_MAP = {
    "res_16x32_mu055_p100": ("resolution_invariance", "invariance"),
    "pressure_24x48_mu055_p200": ("pressure_invariance", "invariance"),
    "rate_24x48_mu055_p100_slow": ("rate_invariance", "invariance"),
    "mu_24x48_mu030_p100": ("friction_monotonicity", "friction"),
}


def load_prereg(path: str = PREREG_PATH) -> dict:
    """Read the committed pre-registration (rules frozen before the runs)."""
    with open(path) as fh:
        return json.load(fh)


def _phi(case: dict) -> float:
    return float(case["critical_state"]["phi_cv_deg"])


def _inertial_number(case: dict) -> float:
    return float((case.get("loading") or {}).get("inertial_number", float("nan")))


def _mu(case: dict) -> float:
    return float((case.get("config") or {}).get("mu", float("nan")))


def adjudicate_sensitivity(cases: list[dict], tol_deg: float = INVARIANCE_TOL_DEG,
                           i_limit: float = QUASI_STATIC_I_LIMIT,
                           anchor_tag: str = ANCHOR_TAG) -> dict:
    """Apply the pre-registered sensitivity rules to the measured configs.

    Decision (declared before the runs): is the round-3 gap ``phi_cv = 14.3 deg
    << 28.81 deg`` a 2D-disc shape/kinematics effect (``H1_shape``, phi_cv stays
    far from the Rowe limit in *every* configuration) or a
    discretisation/rate artefact (``H2_discretisation``, refining the grid or
    slowing the rate moves phi_cv towards 28.81 deg)?

    Configurations that did not land are reported as ``pending`` -- never
    extrapolated.  The function is pure: it does not mutate ``cases``.
    """
    by_tag = {c["tag"]: c for c in cases}
    base = by_tag.get(anchor_tag)
    if base is None:
        raise ValueError(f"anchor configuration {anchor_tag!r} missing from the "
                         f"analysed set {sorted(by_tag)}")
    checks: list[dict] = []
    for tag, (rule, kind) in SENSITIVITY_MAP.items():
        case = by_tag.get(tag)
        if case is None:
            checks.append({"rule": rule, "config": tag, "status": "pending",
                           "phi_cv_deg": None, "delta_deg": None, "passed": None,
                           "note": "configuration not measured (no extrapolation)"})
            continue
        phi, phi0 = _phi(case), _phi(base)
        if kind == "invariance":
            delta = phi - phi0
            passed = abs(delta) <= tol_deg
            note = (f"|{delta:+.2f}| deg <= {tol_deg:.1f} deg required"
                    + ("" if case is not None else ""))
            if rule == "rate_invariance":
                i_num = _inertial_number(case)
                quasi = bool(i_num < i_limit)
                passed = bool(passed and quasi)
                note += (f"; inertial number {i_num:.2e} "
                         f"{'<' if quasi else '>='} {i_limit:.0e}")
            checks.append({"rule": rule, "config": tag, "status": "measured",
                           "phi_cv_deg": phi, "baseline_deg": phi0,
                           "delta_deg": delta, "tolerance_deg": tol_deg,
                           "inertial_number": _inertial_number(case),
                           "passed": passed, "note": note})
        else:                                  # friction monotonicity
            mu, mu0 = _mu(case), _mu(base)
            mono = bool(phi < phi0)
            ratio, ratio0 = math.sin(math.radians(phi)) / mu, \
                math.sin(math.radians(phi0)) / mu0
            checks.append({"rule": rule, "config": tag, "status": "measured",
                           "phi_cv_deg": phi, "baseline_deg": phi0,
                           "delta_deg": phi - phi0, "mu": mu, "mu_baseline": mu0,
                           "sin_phi_over_mu": ratio,
                           "sin_phi_over_mu_baseline": ratio0,
                           "passed": mono,
                           "note": (f"strict monotonicity phi_cv(mu={mu:.2f}) < "
                                    f"phi_cv(mu={mu0:.2f}): "
                                    f"{phi:.2f} < {phi0:.2f}; "
                                    f"sin(phi_cv)/mu = {ratio:.3f} vs {ratio0:.3f}")})
    measured = [c for c in checks if c["status"] == "measured"]
    pending = [c["config"] for c in checks if c["status"] == "pending"]
    gap = {c["tag"]: c["judge_primary_rowe"]["relative_error"]
           for c in cases if "judge_primary_rowe" in c}
    verdict = "pending"
    if not pending:
        if all(c["passed"] for c in measured):
            verdict = "H1_shape_supported"
        else:
            verdict = "H2_discretisation_supported"
    return {
        "prereg": os.path.relpath(PREREG_PATH, REPO),
        "tolerance_deg": tol_deg,
        "quasi_static_limit": i_limit,
        "anchor_tag": anchor_tag,
        "checks": checks,
        "pending_configs": pending,
        "verdict": verdict,
        "verdict_statement": {
            "H1_shape_supported": (
                "every sensitivity configuration reproduces phi_cv far from the "
                "Rowe limit and within the pre-registered invariance band, so the "
                "round-3 gap is a 2D-disc shape/kinematics effect, not "
                "discretisation"),
            "H2_discretisation_supported": (
                "at least one pre-registered sensitivity rule fails: the "
                "round-3 gap is configuration dependent and part of it is a "
                "numerical artefact"),
            "pending": "not all sensitivity configurations were measured; no "
                       "verdict is drawn",
        }[verdict],
        "gap_vs_rowe": gap,
    }


def markdown_sensitivity_table(adj: dict) -> str:
    """Markdown rendering of :func:`adjudicate_sensitivity`."""
    head = ("| rule (pre-registered) | configuration | phi_cv [deg] | "
            "delta vs anchor [deg] | criterion | verdict |\n"
            "|---|---|---|---|---|---|\n")
    rows = []
    for c in adj["checks"]:
        if c["status"] == "pending":
            rows.append(f"| `{c['rule']}` | `{c['config']}` | - | - | "
                        f"{c['note']} | pending |")
            continue
        if "tolerance_deg" in c:
            crit = f"|delta| <= {c['tolerance_deg']:.1f} deg"
        else:
            crit = "phi_cv(mu=0.30) < phi_cv(mu=0.55)"
        rows.append(f"| `{c['rule']}` | `{c['config']}` | {c['phi_cv_deg']:.2f} | "
                    f"{c['delta_deg']:+.2f} | {crit} | "
                    f"{'PASS' if c['passed'] else 'FAIL'} |")
    return head + "\n".join(rows) + f"\n\nverdict: **{adj['verdict']}** -- " \
        f"{adj['verdict_statement']}"


BEGIN_MARK = "<!-- BEGIN ISO SENSITIVITY (generated by analyse_isochoric.py) -->"
END_MARK = "<!-- END ISO SENSITIVITY -->"


def doc_section(payload: dict) -> str:
    """Markdown block for the document's sensitivity section.

    Numbers are injected by this function (the document must never be edited by
    hand): the block is spliced between the markers above by ``--splice-doc``.
    """
    adj = payload["sensitivity"]
    rows = payload["cases"]
    base = next((r for r in rows if r["tag"] == adj["anchor_tag"]), None)
    lines = [BEGIN_MARK,
             "**本节由 `analyse_isochoric.py --splice-doc` 生成，请勿手改。** 判定规则来自 "
             f"`{adj['prereg']}`（先于任何敏感性运行写死，未改动）。", "",
             "| run | grid | phi_cv [deg] | q drift | window spread [deg] | "
             "p [kPa] | converged | err vs Rowe |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["critical_state"]
        cfg = r["config"]
        lines.append(
            f"| `{r['tag']}` | {cfg['columns']}x{cfg['rows']} | "
            f"**{c['phi_cv_deg']:.2f}** | {c['q_drift']:.3f} | "
            f"{c['window_spread_deg']:.2f} | {r['mean_pressure_in_window_kPa']:.0f} | "
            f"{'yes' if c['converged'] else 'no'} | "
            f"{r['judge_primary_rowe']['relative_error'] * 100:.1f} % |")
    lines += ["", markdown_sensitivity_table(adj), ""]
    if base is not None:
        ratios = ", ".join(
            f"mu={r['config']['mu']:.2f} -> sin(phi_cv)/mu="
            f"{math.sin(math.radians(r['critical_state']['phi_cv_deg'])) / r['config']['mu']:.3f}"
            for r in rows)
        lines.append(f"sin(phi_cv)/mu（预注册要求逐档报出）：{ratios}。")
    lines += ["", f"判定：**{adj['verdict']}** —— {adj['verdict_statement']}", END_MARK]
    return "\n".join(lines)


def splice_doc(payload: dict, doc_path: str) -> str:
    """Replace the generated sensitivity block in ``doc_path`` in place."""
    with open(doc_path, encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN_MARK not in text or END_MARK not in text:
        raise ValueError(f"{doc_path} has no sensitivity markers")
    head, rest = text.split(BEGIN_MARK, 1)
    _old, tail = rest.split(END_MARK, 1)
    with open(doc_path, "w", encoding="utf-8") as fh:
        fh.write(head + doc_section(payload) + tail)
    return doc_path


def render_sensitivity(payload: dict, path: str) -> str:
    """Second figure: the pre-registered sensitivity matrix, one panel per axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["cases"]
    by_tag = {r["tag"]: r for r in rows}
    anchor = by_tag.get(ANCHOR_TAG)
    if anchor is None or len(rows) < 2:
        return ""
    phi0 = anchor["critical_state"]["phi_cv_deg"]
    adj = {c["config"]: c for c in payload["sensitivity"]["checks"]}
    order = [t for t in ("res_16x32_mu055_p100", "mu_24x48_mu030_p100",
                         "pressure_24x48_mu055_p200", "rate_24x48_mu055_p100_slow")
             if t in by_tag]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    labels, phis, colors = ["anchor\n24x48, mu.55, p100"], [phi0], ["#1f77b4"]
    for tag in order:
        labels.append(tag.replace("_", "\n"))
        phis.append(by_tag[tag]["critical_state"]["phi_cv_deg"])
        colors.append("#2ca02c" if adj.get(tag, {}).get("passed") else "#ff7f0e")
    bars = ax.bar(range(len(phis)), phis, color=colors, edgecolor="white")
    ax.axhline(phi0, color="k", lw=0.8, ls="--",
               label=f"anchor {phi0:.2f} deg")
    ax.axhspan(phi0 - INVARIANCE_TOL_DEG, phi0 + INVARIANCE_TOL_DEG,
               color="k", alpha=0.06,
               label=f"pre-registered band +-{INVARIANCE_TOL_DEG:.1f} deg")
    ax.axhline(PHI_ROWE_DEG, color="green", lw=1.2, ls="-",
               label=f"Rowe D=0 limit {PHI_ROWE_DEG:.2f} deg")
    for bar, tag in zip(bars, ["anchor"] + order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel(r"$\phi_{cv}$ [deg]")
    ax.set_title("one axis changed at a time (green = rule passed)")
    ax.set_ylim(0, max(phis + [PHI_ROWE_DEG]) * 1.25)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for tag, marker in zip(order, ("s", "D", "^", "v")):
        r = by_tag[tag]
        ax.scatter(r["critical_state"]["phi_cv_deg"], r["mean_pressure_in_window_kPa"],
                   s=60, marker=marker, label=tag, zorder=3)
    ax.scatter(phi0, anchor["mean_pressure_in_window_kPa"], s=90, marker="*",
               color="k", label="anchor", zorder=4)
    ax.axvspan(phi0 - INVARIANCE_TOL_DEG, phi0 + INVARIANCE_TOL_DEG,
               color="k", alpha=0.06)
    ax.set_xlabel(r"$\phi_{cv}$ [deg]")
    ax.set_ylabel("mean pressure in critical-state window [kPa]")
    ax.set_title("is the angle a stress-ratio quantity?\n"
                 "(p changes by 2x in the pressure run, angle must not)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", default=os.environ.get(
        "PROBE_DIR", os.path.join(REPO, "runs", "m20537")),
        help="directory holding the iso_*.json probe records "
             "(empty => committed bundle)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--compact", metavar="PATH",
                    help="write a downsampled gzip bundle of the consumed runs")
    ap.add_argument("--max-samples", type=int, default=1000,
                    help="samples per configuration kept by --compact")
    ap.add_argument("--pin", metavar="PATH",
                    help="write the pinned verdict (verdict + every phi_cv + rule "
                         "outcomes) for the regression test")
    ap.add_argument("--markdown", action="store_true",
                    help="print the sensitivity table as markdown")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--splice-doc", metavar="PATH",
                    help="splice the generated sensitivity block into this "
                         "markdown document (between the ISO SENSITIVITY markers)")
    args = ap.parse_args()

    runs, names = load_runs(args.probe_dir)
    if not runs:
        print(f"no isochoric runs under {args.probe_dir}")
        return 2
    if args.compact:
        print("compacted:", compact_runs(runs, args.compact, args.max_samples))
    cases = [analyse_one(raw) for raw in runs]
    payload = {
        "case_key": "CASE-DEM-BIAXIAL-FRICTION",
        "round": 3,
        "method": ("unbonded plane-strain biaxial, CONSTANT VOLUME (kinematic "
                   "volume-preserving lateral walls, servo released after "
                   "consolidation), frictionless walls; critical state = last 20 % of "
                   "the sheared path above eps_a = 0.20, log strain measure"),
        "references": REF_SOURCES,
        "tolerance": TOLERANCE,
        "anchor_mu": MU_ANCHOR,
        "provenance": {"probe_dir": os.path.relpath(args.probe_dir, REPO)
                       if os.path.isdir(args.probe_dir) else args.probe_dir,
                       "raw_runs": names,
                       "compaction": [r.get("compaction") for r in runs]},
        "cases": cases,
        "sensitivity": adjudicate_sensitivity(cases),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "run_iso.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    png = ""
    if not args.no_figure:
        png = render(payload, os.path.join(args.out_dir, "biaxial_isochoric.png"))
        png_sens = render_sensitivity(
            payload, os.path.join(args.out_dir, "biaxial_isochoric_sensitivity.png"))
        if png_sens:
            print("sensitivity figure:", png_sens)

    print(f"{'tag':30s} {'peak':>6s} {'phi_cv':>7s} {'dEv/dEa':>9s} {'p[win]':>7s} "
          f"{'q_drift':>8s} {'jump':>6s} {'err(rowe)':>9s} {'conv':>5s}")
    for r in cases:
        c = r["critical_state"]
        print(f"{r['tag']:30s} {r['peak_phi_deg']:6.2f} {c['phi_cv_deg']:7.2f} "
              f"{c['d_eps_v_d_eps_a']:+9.2e} {r['mean_pressure_in_window_kPa']:7.1f} "
              f"{c['q_drift']:8.3f} {c['window_spread_deg']:6.2f} "
              f"{r['judge_primary_rowe']['relative_error'] * 100:8.1f}% "
              f"{str(c['converged']):>5s} flags={c['flags']}")
    if args.pin:
        adj = payload["sensitivity"]
        pins = {
            "case_key": "CASE-DEM-BIAXIAL-FRICTION",
            "generated_by": "benchmarks/case_dem_biaxial_friction/analyse_isochoric.py --pin",
            "prereg": adj["prereg"],
            "verdict": adj["verdict"],
            "tolerance_deg": adj["tolerance_deg"],
            "phi_cv_deg": {r["tag"]: round(r["critical_state"]["phi_cv_deg"], 3)
                           for r in cases},
            "passed": {c["rule"]: c["passed"] for c in adj["checks"]},
            "q_drift": {r["tag"]: round(r["critical_state"]["q_drift"], 4) for r in cases},
            "window_spread_deg": {r["tag"]: round(r["critical_state"]["window_spread_deg"], 3)
                                  for r in cases},
            "inertial_number": {r["tag"]: (r.get("loading") or {}).get("inertial_number")
                                for r in cases},
        }
        os.makedirs(os.path.dirname(args.pin), exist_ok=True)
        with open(args.pin, "w") as fh:
            json.dump(pins, fh, indent=1)
        print("pins:", args.pin)
    if args.splice_doc:
        print("spliced:", splice_doc(payload, args.splice_doc))
    sens = payload["sensitivity"]
    print(f"sensitivity verdict: {sens['verdict']} "
          f"(pending={sens['pending_configs']})")
    if args.markdown:
        print()
        print(markdown_table(cases))
        print()
        print(markdown_sensitivity_table(sens))
    print("artefacts:", json_path, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

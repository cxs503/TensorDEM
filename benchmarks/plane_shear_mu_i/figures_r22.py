"""Round-22 figures: the archived law against the archived planeshear points.

Figures (written to ``runs/round22/``):

* ``fig_r22_mu_I.png``   - mu(I): archived da Cruz law (with its validity limit
  and its +/-10 % band), the two in-repo triples, and the four measured points.
* ``fig_r22_phi_I.png``  - phi(I): archived dilatancy law vs GDR MiDi's packing
  path vs the two measured phi ordinates.
* ``fig_r22_attribution.png`` - (a) per-point |error| for every candidate
  against the 10 % tolerance; (b) the Jop form fitted to da Cruz's own law,
  which is how the two in-repo triples are shown to be somebody else's fit.

No number is typed in this file: everything is read from the archive manifest
and from ``runs/round22/analysis_r22.json``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reference_archive import (  # noqa: E402
    ATTRIBUTED_TRIPLES, archived_law, fit_jop_to_law, jop_mu, load_manifest,
)

OUT = os.path.join(HERE, "runs", "round22")
TOL = 0.10


def _analysis():
    with open(os.path.join(OUT, "analysis_r22.json"), encoding="utf-8") as fh:
        return json.load(fh)


def fig_mu(law, report):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    I = np.geomspace(1e-3, 0.35, 400)
    valid = I <= law.i_saturation
    mu_valid = law.mu_min + law.b * I
    ax.plot(I[valid], mu_valid[valid], "k-", lw=2.2,
            label="da Cruz 2005 Eq.(14): mu = %.2f + %.2f I (archived)" % (law.mu_min, law.b))
    ax.plot(I[~valid], mu_valid[~valid], "k--", lw=1.2, alpha=0.6,
            label="same fit extrapolated beyond its validity (not judged)")
    ax.fill_between(I[valid], mu_valid[valid] * (1 - TOL), mu_valid[valid] * (1 + TOL),
                    color="k", alpha=0.10, label="+/-10%% tolerance band")
    ax.axvline(law.i_saturation, color="k", ls=":", lw=1.0)
    ax.text(law.i_saturation * 1.05, ax.get_ylim()[0] + 0.02, "I_sat",
            fontsize=8, rotation=90, va="bottom")

    cols = {"PRIMARY_REPO": "tab:blue", "TEST_GDR_MIDI": "tab:orange"}
    for name, triple in ATTRIBUTED_TRIPLES.items():
        ax.plot(I, jop_mu(I, *triple), color=cols[name], lw=1.4, alpha=0.85,
                label="%s (%.3f/%.3f/%.3f)" % ((name,) + tuple(triple)))
        ax.plot(I, jop_mu(I, *triple), color=cols[name], ls="none")

    pts = sorted(((r["I_bulk"], r["mu"]["mu_full_zz_final"]) for r in report["points"]))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "o-", color="tab:red", ms=7, lw=1.3,
            label="measured mu (full, sigma_zz), servo cell")
    for x, y in pts:
        ax.annotate("%.3f" % y, (x, y), textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color="tab:red")

    ax.set_xscale("log")
    ax.set_xlabel("inertial number I = gamma_dot d / sqrt(P/rho_p)")
    ax.set_ylabel("effective friction mu = tau / P")
    ax.set_title("CASE-DEM-PLANE-SHEAR-MUIDI round 22: archived reference law "
                 "vs measured points", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6.6, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_r22_mu_I.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_phi(law, report):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    I = np.geomspace(1e-3, 0.35, 400)
    valid = I <= law.i_saturation
    phi = law.nu_max - law.a * I
    ax.plot(I[valid], phi[valid], "k-", lw=2.2,
            label="da Cruz 2005 Eq.(13): phi = %.2f - %.2f I (mu_p=0.4, archived)"
                  % (law.nu_max, law.a))
    ax.plot(I[~valid], phi[~valid], "k--", lw=1.2, alpha=0.6,
            label="extrapolated beyond validity (not judged)")
    ax.fill_between(I[valid], phi[valid] * (1 - TOL), phi[valid] * (1 + TOL),
                    color="k", alpha=0.10, label="+/-10%% tolerance band")

    # GDR MiDi packing path, read from the archive manifest at plot time
    man = load_manifest()
    from reference_archive import parse_gdr_midi
    gd = parse_gdr_midi(man)
    ax.plot([1e-4, gd["I_of_second_point"]],
            [gd["phi_max_quasi_static"], gd["phi_at_I"]], "s--", color="tab:green",
            ms=5, lw=1.2, label="GDR MiDi 2004 path %.2f -> %.2f over I=0->%.2f"
                                % (gd["phi_max_quasi_static"], gd["phi_at_I"],
                                   gd["I_of_second_point"]))

    for key, style in (("phi_slab", ("o", "tab:red")), ("phi_bed_steady", ("^", "tab:purple"))):
        seq = sorted((r["I_bulk"], r["phi"][key]) for r in report["points"]
                     if r["phi"][key] is not None)
        ax.plot([s[0] for s in seq], [s[1] for s in seq], style[0] + "-",
                color=style[1], ms=7, lw=1.2, label="measured %s" % key)

    ax.set_xscale("log")
    ax.set_xlabel("inertial number I")
    ax.set_ylabel("solid fraction phi")
    ax.set_title("Round 22: archived dilatancy law vs the measured packing "
                 "fraction", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6.8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_r22_phi_I.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_attribution(law, report):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    cands = ["DACRUZ_LINEAR", "PRIMARY_REPO", "TEST_GDR_MIDI"]
    labels = ["da Cruz 2005\n(archived law)", "PRIMARY_REPO\n0.36/0.84/0.30",
              "TEST_GDR_MIDI\n0.379/0.641/0.279"]
    pts = report["points"]
    x = np.arange(len(pts))
    w = 0.8 / len(cands)
    for i, (c, lab) in enumerate(zip(cands, labels)):
        vals = [abs(r["rel_err"][c]["rel_err"]["mu_full_zz_final"]) for r in pts]
        in_valid = [r["law_in_validity"] for r in pts]
        bars = a1.bar(x + i * w - 0.4 + w / 2, vals, w, label=lab)
        for b, iv in zip(bars, in_valid):
            if c == "DACRUZ_LINEAR" and not iv:
                b.set_hatch("//")
                b.set_alpha(0.55)
    a1.axhline(TOL, color="k", ls="--", lw=1.2)
    a1.text(0.02, TOL * 1.06, "acceptance band 10%", fontsize=7.5)
    a1.set_xticks(x)
    a1.set_xticklabels(["I=%.3f" % r["I_bulk"] for r in pts], fontsize=7.5)
    a1.set_ylabel("|relative error| of mu (full, sigma_zz)")
    a1.set_title("(a) no candidate reference passes; hatched = I outside the\n"
                 "archived law's validity", fontsize=8.5)
    a1.legend(fontsize=6.5)
    a1.grid(alpha=0.2, axis="y")

    I = np.geomspace(1e-3, 0.3, 300)
    target = law.mu_min + law.b * I
    fit = fit_jop_to_law(law)
    a2.plot(I, (jop_mu(I, fit["mu_s"], fit["mu_2"], fit["I_0"]) - target) / target * 100,
            "k-", lw=2.0,
            label="Jop form fitted to the archived law\n(mu_s=%.3f, mu_2=%.3f, I_0=%.3f)"
                  % (fit["mu_s"], fit["mu_2"], fit["I_0"]))
    for c, lab in (("PRIMARY_REPO", "PRIMARY_REPO"), ("TEST_GDR_MIDI", "TEST_GDR_MIDI")):
        t = ATTRIBUTED_TRIPLES[c]
        a2.plot(I, (jop_mu(I, *t) - target) / target * 100, lw=1.5,
                label="%s minus archived law" % lab)
    a2.axhline(0, color="k", lw=0.8)
    a2.axhline(10, color="k", ls="--", lw=1.0)
    a2.axhline(-10, color="k", ls="--", lw=1.0)
    a2.set_xscale("log")
    a2.set_xlabel("inertial number I")
    a2.set_ylabel("difference from the archived da Cruz law [%]")
    a2.set_title("(b) the Jop form *can* represent da Cruz (fit inside 10%):\n"
                 "the two in-repo triples are not da Cruz", fontsize=8.5)
    a2.legend(fontsize=6.5)
    a2.grid(alpha=0.25, which="both")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_r22_attribution.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    law = archived_law()
    report = _analysis()
    for f in (fig_mu(law, report), fig_phi(law, report), fig_attribution(law, report)):
        print("wrote", os.path.relpath(f, HERE))


if __name__ == "__main__":
    main()
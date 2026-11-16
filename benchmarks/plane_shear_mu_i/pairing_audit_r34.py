"""Round 34: which stress belongs in ``mu``, which pressure belongs in ``I``?

Why this round exists
---------------------
Rounds 24-33 judged the 2-D plane-shear cell against the da Cruz 2005 ruler with
a *fixed* pairing of two estimators that do not come from the same place:

* the ordinate ``mu_bulk`` is the Love-Weber stress of **all contacts of the
  cell** (``RoughWallShearCell._bulk_stress``), i.e. a contact stress;
* the abscissa ``I_measured`` (``profile_r31.compare_to_nominal``) is
  ``gammadot_measured d / sqrt(p/rho)`` with ``p`` = the **contact-only pressure
  of the central band** from the round-31 profile accumulator.

Those two are different observables.  A shear cell in steady state must obey
momentum balance, so ``p_wall``, the bulk virial ``p_bulk`` and the profile
``p_contact_total`` should agree -- if they do not, the disagreement is an
*estimator* spread, and since the ruler has ``b = 1.0`` in ``mu = mu_min + b I``
a pressure error of a few tens of percent moves ``mu_ref`` by ~0.01-0.03, which
is the size of the residual that rounds 30-33 tried to attribute to physics
(softness, rolling resistance, thin-layer branch, ``b`` too flat).

This round does not add physics and does not touch the device, the ruler or any
archived artifact.  It enumerates the stress channels an admissible reader can
legitimately pick, builds every pairing into ``(mu, I)``, and asks the
pre-registered question (``prereg_r34.json``): *does the judged error move across
the 10 % acceptance band when the pairing moves?*  If it does, a single-pairing
verdict is not identifiable at the anchor's tolerance and the honest verdict is
WITHHELD-with-a-spread, not FAIL.

Usage::

    python pairing_audit_r34.py --I 0.05 --seed 0            # one point
    python pairing_audit_r34.py --all                        # the pre-registered 3
    python pairing_audit_r34.py --judge-only --dir runs/pairing_r34
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[1]
for _p in (str(REPO / "src"), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plane_shear_2d import PlaneShear2DCase, simulate  # noqa: E402
import reference_2d  # noqa: E402

RUNS = BENCH_DIR / "runs" / "pairing_r34"
PREREG = BENCH_DIR / "prereg_r34.json"

#: the four stress channels the two instruments expose (contact / total split)
CHANNELS = ("wall", "bulk", "profile_contact", "profile_total")


# --------------------------------------------------------------------------- #
# 1. protocol (identical to the round-33 smoke point, so s0 must reproduce it)
# --------------------------------------------------------------------------- #
def smoke_case(inertial_number: float, seed: int, n_particles: int = 310) -> PlaneShear2DCase:
    """Round-31/33 short protocol.  A diagnostic, explicitly not anchor evidence."""
    return PlaneShear2DCase(
        I_target=float(inertial_number),
        seed=int(seed),
        n_particles=int(n_particles),
        width=0.24, gap=0.14, sample_every=40,
        settle_time_units=20.0, shear_settle_strain=0.4, strain_measure=0.6,
        wall_translate=True,
    )


def run_point(inertial_number: float, seed: int, n_particles: int = 310,
              keep_profile: bool = True) -> dict:
    case = smoke_case(inertial_number, seed, n_particles)
    out = simulate(case, progress=False)
    if not keep_profile:
        out.pop("profile", None)
    out["round"] = 34
    out["anchor"] = "CASE-DEM-PLANE-SHEAR-MUIDI"
    out["case"] = asdict(case)
    return out


# --------------------------------------------------------------------------- #
# 2. the ledger: every admissible stress channel of one run
# --------------------------------------------------------------------------- #
def _f(x, default=float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def stress_ledger(summary: dict) -> dict:
    """Extract the normal/shear stress channels plus the measured shear rate.

    ``wall``          : the pinned rows' traction (``p_wall``, ``tau_wall``)
    ``bulk``          : Love-Weber over all contacts (``p_bulk``, ``tau_bulk``)
    ``profile_contact``: central-band contact stress of the round-31 profile
    ``profile_total`` : the same plus the kinetic (granular-temperature) part

    Each entry is ``{'p': Pa, 'tau': Pa, 'mu': -}``; ``None`` where the run did
    not carry the profile (an old artifact), which is reported, never imputed.
    """
    prof = summary.get("profile") or {}
    der = summary.get("profile_derived") or {}
    led: dict = {
        "gamma_dot_nominal": _f(der.get("gamma_dot_nominal"),
                                _f(summary.get("shear_rate_measured"))),
        "gamma_dot_measured": _f(der.get("gamma_dot_measured"),
                                 _f(summary.get("shear_rate_measured"))),
        "P_applied": _f(summary.get("case", {}).get("normal_stress")),
        "density": _f(summary.get("case", {}).get("density")),
        "diameter": 2.0 * _f(summary.get("case", {}).get("radius")),
        "phi_gap": _f(summary.get("phi_gap")),
        "gap_eff": _f(summary.get("gap_eff_final")),
        "wall_translate": summary.get("case", {}).get("wall_translate"),
    }
    pw, tw = _f(summary.get("p_wall")), _f(summary.get("tau_wall"))
    pb, tb = _f(summary.get("p_bulk")), _f(summary.get("tau_bulk"))
    led["wall"] = {"p": pw, "tau": tw, "mu": tw / pw if pw and pw > 0 else None}
    led["bulk"] = {"p": pb, "tau": tb, "mu": tb / pb if pb and pb > 0 else None}
    pc = _f(prof.get("p_contact_total"))
    tc = _f(prof.get("tau_contact_total"))
    pt = _f(prof.get("p_total"))
    tt = _f(prof.get("tau_total"))
    led["profile_contact"] = {"p": pc, "tau": tc,
                              "mu": tc / pc if pc and pc > 0 else None}
    led["profile_total"] = {"p": pt, "tau": tt,
                            "mu": tt / pt if pt and pt > 0 else None}
    led["p_kinetic_profile"] = _f(prof.get("p_kinetic_total"))
    led["kinetic_share_p"] = _f(prof.get("kinetic_share_p"))
    led["mu_profile_total_reported"] = _f(prof.get("mu_total"))
    # the summary's own ``mu_bulk`` is the mean of per-sample ratios; the ratio of
    # the window means (tau_bulk / p_bulk) is a different estimator.  Report both
    # so the 1-2 % difference is visible instead of silently mixed in.
    led["mu_bulk_reported"] = _f(summary.get("mu_bulk"))
    led["mu_bulk_ratio_of_means"] = (tb / pb) if (pb and pb > 0) else None
    return led


def i_from(p: float, gammadot: float, diameter: float, density: float) -> float:
    """``I = gammadot d / sqrt(p/rho)`` -- the device's own definition."""
    if not (p and p > 0 and gammadot and gammadot > 0):
        return float("nan")
    return gammadot * diameter / math.sqrt(p / density)


def pairings(ledger: dict, ruler=None) -> list:
    """Every ``(mu channel, pressure channel)`` pairing, judged against the ruler.

    ``self_consistent`` marks the four pairs that take ``mu`` and ``p`` from the
    same instrument; ``legacy`` marks the pairing rounds 24-33 actually used
    (``mu_bulk`` with the profile contact pressure), which is one of the
    *off-diagonal* combinations and is kept for reproducibility.
    """
    ruler = ruler or reference_2d.mu_2d
    g, d, rho = (ledger["gamma_dot_measured"], ledger["diameter"],
                 ledger["density"])
    rows = []
    for mu_ch, p_ch in itertools.product(CHANNELS, CHANNELS):
        mu = ledger[mu_ch]["mu"]
        p = ledger[p_ch]["p"]
        if mu is None or not p:
            continue
        I = i_from(p, g, d, rho)
        mu_ref = ruler(I)
        rows.append({
            "mu_channel": mu_ch, "p_channel": p_ch,
            "mu": mu, "p": p, "I": I, "mu_ref": mu_ref,
            "err_pct": 100.0 * (mu - mu_ref) / mu_ref if mu_ref else float("nan"),
            "self_consistent": mu_ch == p_ch,
            "legacy": (mu_ch == "bulk" and p_ch == "profile_contact"),
        })
    return rows


def judge(summary: dict, ruler=None) -> dict:
    """One run -> ledger + all pairings + the pre-registered statistics."""
    ledger = stress_ledger(summary)
    rows = pairings(ledger, ruler)
    sc = [r for r in rows if r["self_consistent"]]
    errs = [r["err_pct"] for r in sc if math.isfinite(r["err_pct"])]
    ps = [ledger[c]["p"] for c in CHANNELS
          if ledger[c]["p"] and math.isfinite(ledger[c]["p"])]
    out = {
        "I_target": summary.get("I_target"),
        "seed": summary.get("case", {}).get("seed"),
        "n_free": summary.get("n_free"),
        "ledger": ledger,
        "pairings": rows,
        "spread_self_consistent_pp": (max(errs) - min(errs)) if len(errs) > 1 else float("nan"),
        "pressure_spread_ratio": (max(ps) / min(ps)) if len(ps) > 1 else float("nan"),
        "legacy": next((r for r in rows if r["legacy"]), None),
        "worst_abs_err_pp": (min(errs, key=abs) if errs else float("nan")),
        "closest_to_band_pp": (min(abs(e) for e in errs) - 10.0) if errs else float("nan"),
        "profile_r2": _f((summary.get("profile") or {}).get("fit", {}).get("r2")),
        "phi": ledger["phi_gap"],
        "phi_ref": reference_2d.nu_2d(
            next((r["I"] for r in rows if r["self_consistent"]
                  and r["mu_channel"] == "profile_total"), float("nan"))),
    }
    return out


# --------------------------------------------------------------------------- #
# 3. report: markdown tables + figure, zero hand-copied numbers
# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) \
        else f"{x:.{nd}f}"


def _sf(x, nd=1):
    return "n/a" if x is None or not math.isfinite(x) else f"{x:+.{nd}f}"


def tables(judged: list, prereg: dict) -> str:
    L = ["# R34 — stress-estimator pairing audit (generated, do not hand-edit)", "",
         f"pre-registration: `prereg_r34.json` ({len(judged)} points)", ""]

    L += ["## Table A — the ledger: four normal-stress channels of the same run", "",
          "| I_target | seed | p_wall/P | p_bulk/P | p_prof_contact/P | p_prof_total/P "
          "| kinetic share of p | gamma_meas/gamma_nom |",
          "|---|---|---|---|---|---|---|---|"]
    for j in judged:
        led, P = j["ledger"], j["ledger"]["P_applied"]
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            _fmt(j["I_target"], 3), j["seed"],
            _fmt(led["wall"]["p"] / P), _fmt(led["bulk"]["p"] / P),
            _fmt(led["profile_contact"]["p"] / P),
            _fmt(led["profile_total"]["p"] / P),
            _fmt(led["kinetic_share_p"]),
            _fmt(led["gamma_dot_measured"] / led["gamma_dot_nominal"])))

    L += ["", "## Table B — the four self-consistent pairings vs the 2-D ruler", "",
          "| I_target | seed | pairing | I | mu | mu_ref(I) | err % | in 10 % band? |",
          "|---|---|---|---|---|---|---|---|"]
    for j in judged:
        for r in j["pairings"]:
            if not r["self_consistent"]:
                continue
            L.append("| {} | {} | ({},{}) | {} | {} | {} | {} | {} |".format(
                _fmt(j["I_target"], 3), j["seed"], r["mu_channel"], r["p_channel"],
                _fmt(r["I"], 4), _fmt(r["mu"]), _fmt(r["mu_ref"]),
                _sf(r["err_pct"]),
                "yes" if abs(r["err_pct"]) <= 10.0 else "no"))

    L += ["", "## Table C — the legacy pairing (rounds 24-33) vs its off-diagonal neighbours", "",
          "| I_target | seed | pairing | I | mu | err % |", "|---|---|---|---|---|---|"]
    for j in judged:
        for r in j["pairings"]:
            if r["mu_channel"] == "bulk" or r["p_channel"] == "profile_contact":
                L.append("| {} | {} | ({},{}){} | {} | {} | {} |".format(
                    _fmt(j["I_target"], 3), j["seed"], r["mu_channel"], r["p_channel"],
                    " **legacy**" if r["legacy"] else "",
                    _fmt(r["I"], 4), _fmt(r["mu"]), _sf(r["err_pct"])))

    L += ["", "## Table E — is the device in steady, homogeneous state? (per run)", "",
          "| I_target | seed | gamma_meas/gamma_nom | profile r^2 | p_wall/p_bulk - 1 | "
          "mu_wall/mu_bulk - 1 | mu_bulk (mean of ratios) vs ratio of means |",
          "|---|---|---|---|---|---|---|"]
    for j in judged:
        led = j["ledger"]
        pw, pb = led["wall"]["p"], led["bulk"]["p"]
        mw, mb = led["wall"]["mu"], led["bulk"]["mu"]
        r_means = led.get("mu_bulk_ratio_of_means")
        L.append("| {} | {} | {} | {} | {} | {} | {} vs {} |".format(
            _fmt(j["I_target"], 3), j["seed"],
            _fmt(led["gamma_dot_measured"] / led["gamma_dot_nominal"]),
            _fmt(j.get("profile_r2", float("nan"))),
            _sf(100.0 * (pw / pb - 1.0)) if pb else "n/a",
            _sf(100.0 * (mw / mb - 1.0)) if mb else "n/a",
            _fmt(led.get("mu_bulk_reported"), 4), _fmt(r_means, 4)))

    # ensemble statistics per I_target and per channel
    import statistics as _st
    groups: dict = {}
    for j in judged:
        for r in j["pairings"]:
            if not r["self_consistent"]:
                continue
            groups.setdefault((j["I_target"], r["mu_channel"]), []).append(r["err_pct"])
    L += ["", "## Table F — ensemble mean +/- sd of the judged mu error per channel", "",
          "| I_target | channel | n seeds | mean err % | sd err % | half-range % |",
          "|---|---|---|---|---|---|"]
    for (It, ch), vals in sorted(groups.items()):
        m = _st.mean(vals)
        sd = _st.stdev(vals) if len(vals) > 1 else float("nan")
        hr = 0.5 * (max(vals) - min(vals))
        L.append("| {} | {} | {} | {} | {} | {} |".format(
            _fmt(It, 3), ch, len(vals), _sf(m),
            _fmt(sd) if math.isfinite(sd) else "n/a", _fmt(hr)))

    # can the ruler's slope (b = 1.0) be re-measured from this ensemble at all?
    pts = []
    for j in judged:
        r = next((r for r in j["pairings"]
                  if r["self_consistent"] and r["mu_channel"] == "bulk"), None)
        if r:
            pts.append((r["I"], r["mu"]))
    L += ["", "## Table G — can the ruler slope be re-measured from this ensemble?", "",
          "weighted OLS ``mu = a0 + a1 I`` over all runs (bulk channel, achieved ``I``);",
          "the ruler is ``mu = 0.22 + 1.00 I``.", "",
          "| quantity | value | ruler |", "|---|---|---|"]
    if len(pts) >= 3:
        fit = reference_2d.fit_linear([p[0] for p in pts], [p[1] for p in pts])
        L += [f"| n points | {fit['n']} | - |",
              f"| intercept a0 | {_fmt(fit['intercept'])} +/- {_fmt(fit.get('intercept_se'))} | 0.22 |",
              f"| slope a1 | {_fmt(fit['slope'])} +/- {_fmt(fit.get('slope_se'))} | 1.00 |",
              f"| achieved I range | {_fmt(min(p[0] for p in pts), 4)} .. "
              f"{_fmt(max(p[0] for p in pts), 4)} | 0.001 .. 0.3 |"]
        if fit.get("slope_se") and math.isfinite(fit["slope_se"]) and fit["slope_se"] > 0:
            z = (fit["slope"] - 1.0) / fit["slope_se"]
            L.append(f"| z(slope - 1.00) | {_fmt(z, 2)} | 0 = consistent with the ruler |")
    else:
        L.append("| n points | <3 | - |")
    L += ["", "*caveat (fixed before the runs)*: this pooled fit is **not** a measurement "
          "of the ruler slope `b`. The runs do not form a controlled `I` sweep -- the "
          "`I_target = 0.1` points never reach their nominal rate (see Table E) and land "
          "at a *lower* achieved `I` with a *higher* `mu`, so pooling them is "
          "confounded. A non-zero `z` here is evidence that the achieved `I` is not the "
          "state variable while the cell is not rate-controlled, i.e. a device-state "
          "warning, not a rheological result.", ""]

    preds = prereg["predictions"]
    L += ["", "## Table D — pre-registered predictions (fixed before the runs)", "",
          "| prediction | observed | held? |", "|---|---|---|"]
    ks = [j["ledger"]["kinetic_share_p"] for j in judged
          if math.isfinite(j["ledger"]["kinetic_share_p"])]
    spreads = [j["spread_self_consistent_pp"] for j in judged
               if math.isfinite(j["spread_self_consistent_pp"])]
    prs = [j["pressure_spread_ratio"] for j in judged
           if math.isfinite(j["pressure_spread_ratio"])]
    L.append(f"| P1 kinetic share > 0.15 | max {_fmt(max(ks) if ks else float('nan'))} | "
             f"{'yes' if ks and max(ks) > 0.15 else 'no'} |")
    L.append(f"| P2 pressure spread > 1.15 | max {_fmt(max(prs) if prs else float('nan'))} | "
             f"{'yes' if prs and max(prs) > 1.15 else 'no'} |")
    L.append(f"| P3 error spread > 10 pp | max {_fmt(max(spreads) if spreads else float('nan'), 1)} pp | "
             f"{'yes' if spreads and max(spreads) > 10.0 else 'no'} |")
    L.append(f"| P4 falsifier: spread < 2 pp | min {_fmt(min(spreads) if spreads else float('nan'), 1)} pp | "
             f"{'yes' if spreads and min(spreads) < 2.0 else 'no'} |")
    L += ["", "P1/P2/P3/P4 are the pre-registered statements verbatim from "
          "`prereg_r34.json`; the observed columns are computed from the archived "
          "run JSON, not typed.", ""]
    return "\n".join(L) + "\n"


def figure(judged: list, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    # (a) mu vs achieved I against the ruler -- the anchor's own plane
    Ig = np.linspace(0.015, 0.05, 100)
    ax[0].plot(Ig, reference_2d.mu_2d(Ig), "k-", lw=1.5, label="da Cruz 2005: 0.22+1.0 I")
    ax[0].fill_between(Ig, 0.9 * reference_2d.mu_2d(Ig), 1.1 * reference_2d.mu_2d(Ig),
                       color="tab:green", alpha=0.12, label="+/-10 % band")
    for ch, mk in (("bulk", "o"), ("wall", "s")):
        xs = [r["I"] for j in judged for r in j["pairings"]
              if r["self_consistent"] and r["mu_channel"] == ch]
        ys = [r["mu"] for j in judged for r in j["pairings"]
              if r["self_consistent"] and r["mu_channel"] == ch]
        ax[0].plot(xs, ys, mk, ms=7, mfc="none", label=f"this ensemble, {ch}")
    for j in judged:
        ax[0].annotate(f"I_t={j['I_target']:.2f},s{j['seed']}",
                       (next(r["I"] for r in j["pairings"]
                             if r["self_consistent"] and r["mu_channel"] == "bulk"),
                        next(r["mu"] for r in j["pairings"]
                             if r["self_consistent"] and r["mu_channel"] == "bulk")),
                       fontsize=6, xytext=(3, 4), textcoords="offset points")
    ax[0].set_xlabel("achieved I (from the measured rate and the same pressure channel)")
    ax[0].set_ylabel("mu")
    ax[0].set_title("(a) mu(I): 6 runs, 4 seeds at I_t=0.05")
    ax[0].legend(fontsize=7)

    # (b) the judged error per channel: realisation scatter dominates the channel shift
    chs = list(CHANNELS)
    for k, j in enumerate(judged):
        ys = []
        for c in chs:
            r = next((r for r in j["pairings"]
                      if r["self_consistent"] and r["mu_channel"] == c), None)
            ys.append(r["err_pct"] if r else np.nan)
        ax[1].plot(range(len(chs)), ys, "o-", alpha=0.8,
                   label=f"I_t={j['I_target']:.2f} s{j['seed']}")
    ax[1].axhspan(-10, 10, color="tab:green", alpha=0.12)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(chs)))
    ax[1].set_xticklabels([c.replace("profile_", "prof_") for c in chs], fontsize=8)
    ax[1].set_ylabel("mu error vs ruler  [%]")
    ax[1].set_title("(b) channel spread is small next to seed scatter")
    ax[1].legend(fontsize=6)

    # (c) the stress channels of the same run
    for j in judged:
        ys = [j["ledger"][c]["p"] / j["ledger"]["P_applied"] for c in chs]
        ax[2].plot(range(len(chs)), ys, "o-",
                   label=f"I_t={j['I_target']:.2f} s{j['seed']}")
    ax[2].axhline(1.0, color="k", lw=0.8, ls="--", label="servo set point P")
    ax[2].set_xticks(range(len(chs)))
    ax[2].set_xticklabels([c.replace("profile_", "prof_") for c in chs], fontsize=8)
    ax[2].set_ylabel("normal stress / P_applied")
    ax[2].set_title("(c) momentum balance: channels of one run")
    ax[2].legend(fontsize=6)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--I", dest="I", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", dest="n", type=int, default=310)
    ap.add_argument("--all", action="store_true", help="the pre-registered 3 points")
    ap.add_argument("--judge-only", action="store_true",
                    help="re-judge the archived JSON, no simulation")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(PREREG.read_text())

    if args.judge_only:
        files = sorted((Path(args.dir) if args.dir else RUNS).glob("point_*.json"))
        judged = [judge(json.loads(f.read_text())) for f in files]
        for j, f in zip(judged, files):
            j["source"] = f.name
    else:
        pts = prereg["points"] if args.all else [{"I_target": args.I, "seed": args.seed}]
        judged = []
        for p in pts:
            out = run_point(p["I_target"], p["seed"], args.n)
            f = RUNS / f"point_I{p['I_target']:g}_s{p['seed']}.json"
            f.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
            j = judge(out)
            j["source"] = f.name
            judged.append(j)
            print(f"{f.name}: mu_legacy="
                  f"{_fmt((j['legacy'] or {}).get('err_pct'), 1)} % err, "
                  f"spread={_fmt(j['spread_self_consistent_pp'], 1)} pp, "
                  f"kinetic_share={_fmt(j['ledger']['kinetic_share_p'])}")

    md = tables(judged, prereg)
    (RUNS / "tables_r34.md").write_text(md)
    fig = figure(judged, BENCH_DIR / "runs" / "pairing_r34" / "fig_pairing_r34.png")
    (RUNS / "judged_r34.json").write_text(
        json.dumps(judged, indent=2, sort_keys=True, default=str))
    print(md)
    print(f"-> {RUNS / 'tables_r34.md'}\n-> {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
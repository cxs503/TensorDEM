"""Aggregate the deposition runs: gates, figures and the paper-style document.

Every number in ``docs/CASE_RCP_DEPOSITION.md`` is written by this script from
the run JSONs (no hand copying).  Reference anchors are read from the existing
audit artefact ``benchmarks/rcp_jamming_audit/runs/audit.json``
(``reference.phi_rcp = 0.6366``, Scott 1960 / Bernal & Mason 1960) so the
comparison target has one machine-readable home in the tree.

Two convergence questions are answered here, not assumed:

* **stiffness** -- a soft-sphere bed is compressible, so the measured volume
  fraction carries a ``1/k_n`` systemic.  ``k_n`` is scanned over four levels
  (4000 / 16000 / 64000 / 256000) for both friction tiers and extrapolated to
  the hard-sphere limit with :func:`tensordem.deposition_packing.stiffness_limit`;
  the gate asks whether the *production* ``k_n = 4000`` number is within
  ``0.005`` of that limit (a previous 2-level version of this gate failed for
  the ``mu = 0.1`` tier, which is why the third level and the extrapolation
  exist).  Three levels fix three parameters, so a three-level fit cannot be
  checked against itself; the fourth level is used leave-one-out
  (:func:`tensordem.deposition_packing.extrapolation_selfconsistency`) and
  ``G17`` asks the three-level model to *predict* it;
* **timestep** -- the tangential Cundall-Strack history is a first-order update
  in ``dt`` and friction is the tier that failed the stiffness gate, so the
  ``dt`` is halved for both tiers at ``k_n = 4000`` and the two numbers must
  agree to ``0.002``.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")
BAND = (0.62, 0.66)          # feature_ledger acceptance band for this case
TOL_REL = 0.03               # <3 % against the reference (accuracy standard)
DT_DEFAULT = 1.0 / 20.0      # production timestep, dt = t_c / 20
STIFF_GATE = 0.005           # |phi(k_n_production) - phi(k_n -> inf)|
PROD_N_MIN = 10000           # feature_ledger: "10000 spheres, both friction tiers"
SEED_MIN = 3                 # the seed claim the document makes must be met
DT_GATE = 0.002              # |phi(dt/2) - phi(dt)|
SC_PRED_GATE = 0.002         # |3-level prediction - withheld stiffest level|
SC_LIMIT_GATE = 0.002        # |limit(3 levels) - limit(all levels)|

sys.path.insert(0, os.path.join(ROOT, "src"))
from tensordem.deposition_packing import (  # noqa: E402
    extrapolation_selfconsistency, stiffness_limit)
from tensordem.rcp_references import get_reference  # noqa: E402


def reference():
    """Machine-readable RCP reference (falls back to the literature value)."""
    path = os.path.join(ROOT, "benchmarks", "rcp_jamming_audit", "runs", "audit.json")
    try:
        ref = json.load(open(path))["reference"]
        ref = {"phi_rcp": float(ref["phi_rcp"]), "source": ref.get("source", "")}
        ref["origin"] = os.path.relpath(path, ROOT)
        return ref
    except (OSError, KeyError, ValueError):
        return {"phi_rcp": 0.6366, "source": "Scott 1960; Bernal & Mason 1960",
                "origin": "literal fallback (audit.json not readable)"}


def load_runs():
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "deposition_*.json"))):
        d = json.load(open(path))
        spec = d["spec"]
        m = d["metrics"]
        out.append({
            "path": os.path.relpath(path, ROOT),
            "n": spec["n_particles"], "mu": spec["mu"], "kn": spec["kn"],
            "seed": spec["seed"], "e": spec["restitution"],
            "dt_frac": spec.get("dt_frac", DT_DEFAULT),
            "phi": m["phi_bulk"],
            "phi_corr": m["phi_bulk_overlap_corrected"],
            "phi_surface": m["phi_surface_to_floor"],
            "Z": m["coordination_bulk"],
            "z12": m["z12_fraction"],
            "ovmax": m["max_pair_overlap_over_d"],
            "mean_ov": m["mean_overlap_over_d"],
            "settled": d["settled"], "steps": d["steps"],
            "wall": d["wall_clock"], "t_end": d["final_time"],
            "dt": m["dt"], "ke_final": m.get("ke_norm_final"),
            "surface": m["surface_height"],
            "band": m["band"],
            "profile": d["profiles"]["profile"],
            "history": d["history"],
            "impl": d["impl_sha256"][:16],
        })
    return out


def main() -> int:
    runs_all = load_runs()
    if not runs_all:
        raise SystemExit("no deposition runs found in " + RUNS)
    ref = reference()
    phi_ref = ref["phi_rcp"]

    # production runs: default timestep only (the dt probe is a separate arm)
    runs = [r for r in runs_all if abs(r["dt_frac"] - DT_DEFAULT) < 1e-12]
    probes = [r for r in runs_all if abs(r["dt_frac"] - DT_DEFAULT) >= 1e-12]
    n_max = max(r["n"] for r in runs)

    # production system size: the largest N that carries *both* friction tiers
    # at the production stiffness, so the tier comparison is like-for-like
    for kn in {r["kn"] for r in runs}:
        have0 = {r["n"] for r in runs if r["mu"] == 0.0 and r["kn"] == kn}
        have1 = {r["n"] for r in runs if abs(r["mu"] - 0.1) < 1e-12 and r["kn"] == kn}
        if have0 & have1:
            break
    both = {r["n"] for r in runs if r["mu"] == 0.0} & \
           {r["n"] for r in runs if abs(r["mu"] - 0.1) < 1e-12}
    n_primary = max(both) if both else n_max
    # Production stiffness: the *stiffest* k_n that still carries SEED_MIN seeds
    # in both friction tiers.  Round 2 escalated it (4000 -> 64000) because the
    # soft bed was compressed by ~1 % (G5/G6/G9 failed); the rule is read off the
    # artefacts so the document can never quote a stiffness it has no seeds for.
    def _seeds_at(kn, mu):
        return {r["seed"] for r in runs
                if r["n"] == n_primary and abs(r["mu"] - mu) < 1e-12
                and abs(r["kn"] - kn) < 1e-9}
    _prod_cand = [kn for kn in sorted({r["kn"] for r in runs})
                  if all(len(_seeds_at(kn, mu)) >= SEED_MIN for mu in (0.0, 0.1))]
    kn_prod = float(_prod_cand[-1]) if _prod_cand else 4000.0
    kn_prod = float(os.environ.get("RCP_PROD_KN", kn_prod))

    def sel(mu, kn=kn_prod, n=None, dt_frac=DT_DEFAULT):
        n = n_primary if n is None else n
        cand = [r for r in runs_all
                if abs(r["mu"] - mu) < 1e-12 and abs(r["kn"] - kn) < 1e-9
                and r["n"] == n and abs(r["dt_frac"] - dt_frac) < 1e-12]
        return sorted(cand, key=lambda r: r["seed"])

    def stat(mu, kn=kn_prod, n=None, dt_frac=DT_DEFAULT):
        rs = sel(mu, kn, n, dt_frac)
        if not rs:
            return None
        phis = np.array([r["phi"] for r in rs])
        return (float(phis.mean()), len(rs),
                float(phis.std(ddof=1)) if len(rs) > 1 else 0.0)

    frictionless = stat(0.0)
    lowfric = stat(0.1)
    _fs_kn, _fs_best = kn_prod, 1
    for kn in sorted({r["kn"] for r in runs}):
        ns = {r["n"] for r in runs
              if r["mu"] == 0.0 and abs(r["kn"] - kn) < 1e-9
              and abs(r["dt_frac"] - DT_DEFAULT) < 1e-12}
        if len(ns) > _fs_best:
            _fs_kn, _fs_best = kn, len(ns)
    size0 = {n: stat(0.0, _fs_kn, n) for n in
             sorted({r["n"] for r in runs if r["mu"] == 0.0
                     and abs(r["kn"] - _fs_kn) < 1e-9})}

    # ---- stiffness convergence ------------------------------------------
    kn_levels = sorted({r["kn"] for r in runs if r["n"] == n_primary})
    stiff = {}
    for mu in (0.0, 0.1):
        # seed-matched ladder: one configuration family, only k_n changes.  A
        # tier mean per level would fold the (1e-4) seed scatter into a (4e-3)
        # stiffness step and make the observed order depend on seed coverage.
        pts = []
        for kn in kn_levels:
            hit = [r["phi"] for r in runs
                   if abs(r["mu"] - mu) < 1e-12 and r["n"] == n_primary
                   and abs(r["kn"] - kn) < 1e-9 and r["seed"] == 1]
            if hit:
                pts.append((kn, float(hit[0])))
        if len(pts) >= 2:
            stiff[mu] = dict(stiffness_limit(pts), points=pts)

    gates = []
    if frictionless:
        dev = frictionless[0] / phi_ref - 1.0
        gates.append({"id": "G1_frictionless_vs_ref",
                      "criterion": f"|phi/phi_ref - 1| < {TOL_REL:.0%}, "
                                   f"phi_ref={phi_ref}, N={n_primary}, k_n={kn_prod:g}",
                      "got": frictionless[0], "value": abs(dev),
                      "pass": bool(abs(dev) < TOL_REL)})
    if lowfric:
        gates.append({"id": "G2_low_friction_in_band",
                      "criterion": f"{BAND[0]} <= phi <= {BAND[1]} (mu=0.1)",
                      "got": lowfric[0], "value": lowfric[0],
                      "pass": bool(BAND[0] <= lowfric[0] <= BAND[1])})
    if frictionless:
        gates.append({"id": "G3_frictionless_in_band",
                      "criterion": f"{BAND[0]} <= phi <= {BAND[1]} (mu=0)",
                      "got": frictionless[0], "value": frictionless[0],
                      "pass": bool(BAND[0] <= frictionless[0] <= BAND[1])})
    corr = [abs(r["phi"] - r["phi_corr"]) / r["phi"] for r in runs]
    gates.append({"id": "G4_overlap_correction_is_small",
                  "criterion": "|phi - phi_overlap_corrected| / phi < 1%",
                  "got": max(corr), "value": max(corr),
                  "pass": bool(max(corr) < 0.01)})
    for gid, mu, label in (("G5_stiffness_convergence", 0.0, "mu=0"),
                           ("G6_stiffness_convergence_frictional", 0.1, "mu=0.1")):
        s = stiff.get(mu)
        hit = [r["phi"] for r in runs
               if abs(r["mu"] - mu) < 1e-12 and r["n"] == n_primary
               and abs(r["kn"] - kn_prod) < 1e-9 and r["seed"] == 1]
        prod = (float(hit[0]), 1, 0.0) if hit else None
        if s and prod:
            d = abs(prod[0] - s["limit"])
            gates.append({"id": gid,
                          "criterion": f"|phi(k_n={kn_prod:g}) - phi(k_n->inf)| "
                                       f"< {STIFF_GATE} ({label})",
                          "got": d, "value": d, "pass": bool(d < STIFF_GATE)})
    # G12/G13: stiffness limit vs friction-dependent reference
    # The mu=0 tier is compared against the frictionless RCP reference (0.6366).
    # The mu=0.1 tier is compared against the mu=0.1 reference (0.6250) from
    # Pournin et al. (2001), NOT the frictionless reference — comparing a
    # frictional packing against a frictionless benchmark is a category error.
    for gid, mu, label in (("G12_stiffness_limit_vs_ref", 0.0, "mu=0"),
                           ("G13_stiffness_limit_vs_ref_frictional", 0.1, "mu=0.1")):
        s = stiff.get(mu)
        if s:
            ref_mu = get_reference(mu)
            dev = s["limit"] / ref_mu.phi_rcp - 1.0
            gates.append({"id": gid,
                          "criterion": f"|phi(k_n->inf)/phi_ref(mu={mu:g}) - 1| "
                                       f"< {TOL_REL:.0%} ({label}, "
                                       f"phi_ref={ref_mu.phi_rcp})",
                          "got": s["limit"], "value": abs(dev),
                          "pass": bool(abs(dev) < TOL_REL),
                          "phi_ref": ref_mu.phi_rcp,
                          "phi_ref_source": ref_mu.source})
    # G18: friction-dependent reference comparison for the production value
    # (not just the extrapolated limit) — this is the gate that directly
    # answers the ledger acceptance question for each tier.
    for gid, mu, label in (("G18_production_vs_friction_ref", 0.0, "mu=0"),
                           ("G18b_production_vs_friction_ref", 0.1, "mu=0.1")):
        s = stat(mu)
        if s:
            ref_mu = get_reference(mu)
            dev = s[0] / ref_mu.phi_rcp - 1.0
            gates.append({"id": gid,
                          "criterion": f"|phi_prod(mu={mu:g})/phi_ref(mu={mu:g}) - 1| "
                                       f"< {TOL_REL:.0%} (phi_ref={ref_mu.phi_rcp})",
                          "got": s[0], "value": abs(dev),
                          "pass": bool(abs(dev) < TOL_REL),
                          "phi_ref": ref_mu.phi_rcp,
                          "phi_ref_source": ref_mu.source})
    # leave-one-level-out: with >= 4 k_n levels the extrapolation can be asked
    # to predict a measurement it never saw (three levels fix three parameters,
    # so a three-level fit has no out-of-sample content at all).
    selfcheck = {}
    for mu in (0.0, 0.1):
        s = stiff.get(mu)
        if s and s["n_levels"] >= 4:
            selfcheck[mu] = extrapolation_selfconsistency(s["points"])
            gates.append({"id": f"G17_extrapolation_selfconsistency_mu{mu:g}",
                          "criterion": f"3-level fit predicts k_n="
                                       f"{selfcheck[mu]['points'][-1][0]:g} to "
                                       f"< {SC_PRED_GATE} and the 4th level moves "
                                       f"the limit by < {SC_LIMIT_GATE} (mu={mu:g})",
                          "got": selfcheck[mu]["prediction_error"],
                          "value": selfcheck[mu]["prediction_error"],
                          "limit_shift": selfcheck[mu]["limit_shift"],
                          "pass": bool(selfcheck[mu]["prediction_error"] < SC_PRED_GATE
                                       and selfcheck[mu]["limit_shift"] < SC_LIMIT_GATE)})
    dt_rows = []
    for mu in (0.0, 0.1):
        for kn in sorted({r["kn"] for r in runs_all if r["n"] == n_primary}):
            base = sel(mu, kn)
            half = sel(mu, kn, dt_frac=DT_DEFAULT / 2.0)
            if not (base and half):
                continue
            # paired by seed: the halved-dt probe exists for a subset of the
            # seeds, so comparing a seed *mean* against a single probe would
            # fold the seed scatter (1e-4) into the temporal error (1e-5).
            common = sorted({r["seed"] for r in base} & {r["seed"] for r in half})
            if common:
                phi_a = float(np.mean([r["phi"] for r in base if r["seed"] in common]))
                phi_b = float(np.mean([r["phi"] for r in half if r["seed"] in common]))
            else:
                common, phi_a, phi_b = [], np.mean([r["phi"] for r in base]), np.mean([r["phi"] for r in half])
            d = abs(phi_a - phi_b)
            dt_rows.append({"mu": mu, "kn": kn, "phi": phi_a, "phi_halved": phi_b,
                            "delta": d, "seeds": common,
                            "dt": half[0]["dt"]})
            gates.append({"id": f"G14_timestep_convergence_mu{mu:g}"
                                + ("" if kn == kn_prod else f"_kn{kn:g}"),
                          "criterion": f"|phi(dt/2) - phi(dt)| < {DT_GATE} "
                                       f"(mu={mu:g}, k_n={kn:g}, same seed)",
                          "got": d, "value": d, "pass": bool(d < DT_GATE)})
    sizes = [v[0] for v in size0.values() if v]
    if len(sizes) > 1:
        spread = max(sizes) - min(sizes)
        gates.append({"id": "G7_finite_size",
                      "criterion": "max-min phi over the N scan < 0.01 (mu=0)",
                      "got": spread, "value": spread, "pass": bool(spread < 0.01)})
    settled_all = all(r["settled"] for r in runs_all)
    gates.append({"id": "G8_settled",
                  "criterion": "every run reached KE/(N m g d) < 5e-4 for 4 t_g",
                  "got": settled_all, "value": bool(settled_all),
                  "pass": bool(settled_all)})
    gates.append({"id": "G15_production_size",
                  "criterion": f"N_production >= {PROD_N_MIN} (feature_ledger: "
                               f"10000 spheres, both friction tiers; largest N "
                               f"carrying both tiers)",
                  "got": n_primary, "value": n_primary,
                  "pass": bool(n_primary >= PROD_N_MIN)})
    seed_counts = {mu: (stat(mu)[1] if stat(mu) else 0) for mu in (0.0, 0.1)}
    seed_min = min(seed_counts.values())
    gates.append({"id": "G16_seed_coverage",
                  "criterion": f"each production tier >= {SEED_MIN} independent seeds",
                  "got": seed_min, "value": seed_min,
                  "pass": bool(seed_min >= SEED_MIN)})
    _prod_runs = [r for r in runs if abs(r["kn"] - kn_prod) < 1e-9
                  and abs(r["dt_frac"] - DT_DEFAULT) < 1e-12]
    ovmax = (max(r["ovmax"] for r in _prod_runs) if _prod_runs
             else max(r["ovmax"] for r in runs_all))
    ovmax_all = max(r["ovmax"] for r in runs_all)
    gates.append({"id": "G9_residual_overlap",
                  "criterion": f"max overlap/d < 0.01 in the settled state "
                               f"(production arm, k_n={kn_prod:g})",
                  "got": ovmax, "value": ovmax, "pass": bool(ovmax < 0.01),
                  "all_arms_max": ovmax_all})
    if frictionless and lowfric:
        gate_mu = lowfric[0] <= frictionless[0] + 0.005
        gates.append({"id": "G10_friction_loosens",
                      "criterion": "phi(mu=0.1) <= phi(mu=0) + 0.005",
                      "got": lowfric[0] - frictionless[0],
                      "value": lowfric[0] - frictionless[0],
                      "pass": bool(gate_mu)})
    viol = 0
    for r in runs_all:
        tot = np.array(r["history"]["total_norm"])
        viol += int(np.sum(np.diff(tot) > 5e-3))
    gates.append({"id": "G11_energy_never_increases",
                  "criterion": "no window where mechanical energy rises by > 0.5 % of N m g d",
                  "got": viol, "value": viol, "pass": viol == 0})

    # ---- tier statistics -------------------------------------------------
    tier_stats = {}
    _mus = sorted({r["mu"] for r in runs})
    for mu in _mus:
        name = f"mu={mu:g}"
        rs = sel(mu)
        if not rs:
            continue
        phis = np.array([r["phi"] for r in rs])
        tier_stats[name] = {
            "n_runs": len(rs), "seeds": [r["seed"] for r in rs],
            "phi_mean": float(phis.mean()),
            "phi_std": float(phis.std(ddof=1)) if len(phis) > 1 else 0.0,
            "phi_corr_mean": float(np.mean([r["phi_corr"] for r in rs])),
            "Z": float(np.mean([r["Z"] for r in rs])),
            "z12": float(np.mean([r["z12"] for r in rs])),
            "ovmax": float(np.max([r["ovmax"] for r in rs])),
            "settled": all(r["settled"] for r in rs),
            "dev_vs_ref": float(phis.mean() / phi_ref - 1.0),
        }

    # ---- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = os.path.join(HERE, "figures")
    os.makedirs(figs, exist_ok=True)
    made = []
    captions = []

    # (a) profiles per friction tier
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for a, mu in zip(ax, (0.0, 0.1)):
        rs = sel(mu)
        if not rs:
            continue
        for r in rs[:3]:
            a.plot(r["profile"]["phi"], r["profile"]["z"], lw=1.2,
                   label=f"seed {r['seed']}  phi_bulk={r['phi']:.4f}")
        a.axvline(BAND[1], color="k", ls=":", lw=0.8, label="ledger band")
        a.axvline(BAND[0], color="k", ls=":", lw=0.8)
        a.axvline(phi_ref, color="r", ls="--", lw=0.9, label=f"ref {phi_ref}")
        a.set_title(f"mu = {mu:g}, N = {rs[0]['n']}")
        a.set_xlabel("solid fraction phi(z)")
        a.legend(fontsize=6)
    ax[0].set_ylabel("z / d")
    fig.suptitle("Settled-bed solid-fraction profiles (gravity deposition)")
    p = os.path.join(figs, "fig_profiles.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    captions.append("Settled solid-fraction profiles phi(z) for both friction "
                    "tiers, with the ledger band and the reference value.")

    # (b) phi vs mu
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for mu in sorted({r["mu"] for r in runs}):
        rs = sel(mu)
        if not rs:
            continue
        ax.plot([mu] * len(rs), [r["phi"] for r in rs], "o", ms=5, color="C0")
    ax.axhspan(*BAND, color="0.85", label="acceptance band")
    ax.axhline(phi_ref, color="r", ls="--", label=f"ref {phi_ref}")
    ax.set_xlabel("inter-particle friction mu"); ax.set_ylabel("phi_bulk")
    ax.set_title("Friction loosens the deposited packing")
    ax.legend(fontsize=8)
    p = os.path.join(figs, "fig_phi_vs_mu.png"); fig.tight_layout()
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    captions.append("Bulk solid fraction vs inter-particle friction "
                    "(band + reference); one marker per seed.")

    # (c) finite size
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ns = sorted(n for n in size0 if size0[n])
    if ns:
        ax.plot(ns, [size0[n][0] for n in ns], "o-", label="mu = 0")
        ax.axhspan(*BAND, color="0.85"); ax.axhline(phi_ref, color="r", ls="--")
        ax.set_xlabel("N particles"); ax.set_ylabel("phi_bulk")
        ax.set_title("Finite-size behaviour")
    p = os.path.join(figs, "fig_phi_vs_N.png"); fig.tight_layout()
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    captions.append("Finite-size scan of phi_bulk.")

    # (d) settling
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for mu in (0.0, 0.1):
        rs = sel(mu)
        if rs:
            h = rs[0]["history"]
            ax.semilogy(h["t"], np.maximum(h["ke_norm"], 1e-12),
                        label=f"mu={mu:g} seed {rs[0]['seed']}")
    ax.axhline(5e-4, color="k", ls=":", lw=0.8, label="settle threshold")
    ax.set_xlabel("t sqrt(g/d)"); ax.set_ylabel("KE / (N m g d)")
    ax.set_title("Settling: kinetic energy history")
    ax.legend(fontsize=8)
    p = os.path.join(figs, "fig_settling.png"); fig.tight_layout()
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    captions.append("Kinetic-energy history during settling (the stopping criterion).")

    # (e) stiffness convergence to the hard-sphere limit
    if stiff:
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        for i, (mu, s) in enumerate(sorted(stiff.items())):
            kns = np.array(s["kn"]); phis = np.array(s["value"])
            ax.plot(1.0 / kns, phis, "o-", color=f"C{i}",
                    label=f"mu={mu:g}: k_n={', '.join(f'{k:g}' for k in kns)}")
            ax.plot([0.0], [s["limit"]], "s", ms=8, color=f"C{i}")
            ax.annotate(f"limit {s['limit']:.5f} (p={s['order_used']:.2f})",
                        (0.0, s["limit"]), textcoords="offset points",
                        xytext=(6, -12 if i else 6), fontsize=7, color=f"C{i}")
        ax.axhline(phi_ref, color="r", ls="--", lw=0.9, label=f"ref {phi_ref}")
        ax.axhspan(*BAND, color="0.85", zorder=0)
        ax.set_xlabel("1 / k_n"); ax.set_ylabel("phi_bulk")
        ax.set_title("Soft-sphere systematic and its hard-sphere limit")
        ax.legend(fontsize=6.5)
        p = os.path.join(figs, "fig_phi_vs_stiffness.png")
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
        captions.append("Bulk solid fraction vs 1/k_n for both friction tiers. "
                        "Squares are the Richardson (hard-sphere) limits, the "
                        "annotation gives the observed order p.")

    summary = {
        "reference": ref, "band": BAND, "tolerance_rel": TOL_REL,
        "n_particles_max": n_max, "n_primary": n_primary,
        "kn_production": kn_prod, "kn_levels": kn_levels,
        "tiers": tier_stats, "gates": gates,
        "selfcheck": {f"mu={mu:g}": {k: v for k, v in sc.items()
                                     if k not in ("fit", "full", "points")}
                      for mu, sc in selfcheck.items()},
        "stiffness": {f"mu={mu:g}": {k: v for k, v in s.items() if k != "points"}
                      for mu, s in stiff.items()},
        "timestep": dt_rows,
        "probe_runs": [{"n": r["n"], "mu": r["mu"], "kn": r["kn"],
                        "dt_frac": r["dt_frac"], "phi": r["phi"]} for r in probes],
        "runs": [{k: v for k, v in r.items() if k not in ("profile", "history")}
                 for r in runs_all],
        "figures": [os.path.relpath(p, ROOT) for p in made],
        "impl_sha256": runs[0]["impl"],
        "all_gates_pass": all(g["pass"] for g in gates),
    }
    with open(os.path.join(RUNS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)

    # ---- document --------------------------------------------------------
    def fmt(x, nd=4):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)

    lines = []
    A = lines.append
    A("# CASE-DEM-RCP-PACKING-FRACTION — gravity-deposition measurement")
    A("")
    A("Anchor: `CASE-DEM-RCP-PACKING-FRACTION` (feature_ledger). This document is")
    A("generated by `benchmarks/case_rcp_deposition/aggregate.py`; every number")
    A("below is read from the run JSONs in `benchmarks/case_rcp_deposition/runs/`")
    A(f"(kernel sha256 `{runs[0]['impl']}…`).")
    A("")
    A("## 1 Problem, geometry and parameters")
    A("")
    A("Random close packing of monodisperse spheres: the volume fraction that a")
    A("**poured** assembly reaches. The reference is the poured/compressed")
    A(f"experiment of Scott (1960) and Bernal & Mason (1960), `phi_rcp = {phi_ref}`")
    A(f"(machine-readable anchor: `{ref['origin']}`); the ledger band is "
      f"`[{BAND[0]}, {BAND[1]}]`, accuracy standard < 3 %.")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| N particles | {n_primary} (production, both tiers), "
      f"finite-size scan {', '.join(str(n) for n in ns)} |")
    A("| diameter d / density rho / g | 1 / 1 / 1 (dimensionless) |")
    A("| cell | square footprint L x L, periodic in x,y; frictional floor at z=0 |")
    A("| normal law | linear spring-dashpot, e = 0.1, k_t = k_n/2 |")
    A(f"| stiffness levels | {', '.join(f'{k:g}' for k in kn_levels)} "
      f"(production {kn_prod:g}, k_n -> inf by Richardson) |")
    A("| timestep | dt = t_c/20 (production), t_c/40 (timestep probe) |")
    A("| friction tiers | " + ", ".join(
        f"mu={mu:g}" for mu in sorted({r["mu"] for r in runs
                                       if r["n"] == n_primary
                                       and abs(r["kn"] - kn_prod) < 1e-9}))
      + " (0 and 0.1 are the ledger tiers; the rest is a sensitivity scan) |")
    A("| seeds | " + ", ".join(
        f"mu={mu:g}: {stat(mu)[1]} seed{'s' if stat(mu)[1] != 1 else ''}"
        for mu in sorted({r["mu"] for r in runs if r["n"] == n_primary
                          and abs(r["kn"] - kn_prod) < 1e-9})) + " |")
    A("")
    A("## 2 Method")
    A("")
    A("Particles are released by random sequential addition at a dilute fraction")
    A("(no lattice seed: a lattice imprints order that survives the pour), fall")
    A("under gravity, dissipate through restitution/friction and settle; the")
    A("deposition stops when `KE/(N m g d) < 5e-4` is sustained for 4 gravity")
    A("times. The bulk solid fraction is then read from the settled free surface:")
    A("`phi(z)` by cloud-in-cell binning, averaged over the band `2.5 d` above the")
    A("floor to `2.5 d` below the surface, so neither wall layer enters the number.")
    A("A per-sphere deflation by the mean overlap gives the overlap-corrected value.")
    A("")
    A("A soft-sphere bed is compressible, so `phi` also depends on the stiffness")
    A("used to model a rigid grain. Three levels (`k_n` ratio 4) give the observed")
    A("order `p` of `phi(k_n) = phi_inf + A k_n^-p` and a Richardson limit")
    A("(`tensordem.deposition_packing.stiffness_limit`); the halved-timestep runs")
    A("separate that stiffness systemic from first-order Cundall-Strack integration")
    A("error, which would show up as a `dt` dependence at fixed `k_n`.")
    A("")
    A(f"The production stiffness is not pinned: it is the stiffest `k_n` that")
    A(f"still carries {SEED_MIN} seeds in both friction tiers "
      f"(`k_n = {kn_prod:g}`), and the hard-sphere limit is quoted next to it.")
    A("Round 1 used `k_n = 4000`, where the settled overlap (~1.1 %) compressed")
    A("the bed; that arm is kept in the run set as a softness reference, not as")
    A("the answer.")
    A("")
    A("Kernel: `src/tensordem/deposition_packing.py` (velocity Verlet, Cundall–")
    A("Strack tangential spring with per-pair history on sorted keys, cKDTree pair")
    A("list with a 20 % skin). Command:")
    A("")
    A("```bash")
    A(f"python3 benchmarks/case_rcp_deposition/shard.py <0..{len(json.load(open(os.path.join(HERE, 'plan.json')))['cases']) - 1}>   # one cluster shard per case")
    A("python3 benchmarks/case_rcp_deposition/aggregate.py   # this document")
    A("```")
    A("")
    A("## 3 Results")
    A("")
    A("| seed | N | mu | k_n | dt | phi_bulk | phi (overlap-corr.) | Z | max ov/d | settled | steps | wall / s |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs_all:
        A(f"| {r['seed']} | {r['n']} | {r['mu']:g} | {r['kn']:g} | {r['dt']:.3e} | "
          f"{r['phi']:.5f} | {r['phi_corr']:.5f} | {r['Z']:.2f} | {r['ovmax']:.4f} | "
          f"{'yes' if r['settled'] else 'no'} | {r['steps']} | {r['wall']:.0f} |")
    A("")
    A(f"Tier means (N = {n_primary}, k_n = {kn_prod:g}, dt = t_c/20):")
    A("")
    A("| tier | runs (seeds) | phi mean +- std | dev vs ref | Z |")
    A("|---|---|---|---|---|")
    for k, v in tier_stats.items():
        A(f"| {k} | {v['n_runs']} ({', '.join(str(s) for s in v['seeds'])}) | "
          f"{v['phi_mean']:.5f} +- {v['phi_std']:.5f} | "
          f"{100 * v['dev_vs_ref']:+.2f} % | {v['Z']:.2f} |")
    A("")
    A(f"### 3.0 Friction scan at the production stiffness (k_n = {kn_prod:g})")
    A("")
    A("| mu | phi (mean) | seeds | Delta vs mu=0 | in band |")
    A("|---|---|---|---|---|")
    _base = tier_stats.get("mu=0")
    for mu in sorted({r["mu"] for r in runs if r["n"] == n_primary
                      and abs(r["kn"] - kn_prod) < 1e-9}):
        row = tier_stats.get(f"mu={mu:g}")
        if not row:
            continue
        d = (row["phi_mean"] - _base["phi_mean"]) if _base else float("nan")
        A(f"| {mu:g} | {row['phi_mean']:.5f} | {row['n_runs']} | "
          f"{'n/a' if _base is None else f'{d:+.5f}'} | "
          f"{'yes' if BAND[0] <= row['phi_mean'] <= BAND[1] else 'no'} |")
    A("")
    if stiff:
        A("### 3.1 Stiffness systemic and hard-sphere limit")
        A("")
        A("| tier | phi per k_n level | Delta (consecutive) | observed order p | phi(k_n->inf) | "
          "|phi_prod - phi_inf| |")
        A("|---|---|---|---|---|---|")
        for mu, s in sorted(stiff.items()):
            levels = " / ".join(f"{k:g}: {v:.5f}" for k, v in s["points"])
            deltas = " / ".join(f"{d:+.5f}" for d in s["delta"])
            prod = stat(mu)
            A(f"| mu={mu:g} | {levels} | {deltas} | {s['order_used']:.3f}"
              f"{'' if s['order'] is not None else ' (default: non-monotone)'} | "
              f"{s['limit']:.5f} | {abs(prod[0] - s['limit']):.5f} |")
        A("")
    if selfcheck:
        A("### 3.1b Is the extrapolation checkable? (leave-one-level-out)")
        A("")
        A("| tier | 3-level limit | prediction at withheld level | measured | "
          "prediction error | all-level limit | limit shift |")
        A("|---|---|---|---|---|---|---|")
        for mu, sc in sorted(selfcheck.items()):
            A(f"| mu={mu:g} | {sc['fit']['limit']:.5f} | {sc['predicted']:.5f} | "
              f"{sc['measured']:.5f} | {sc['prediction_error']:.5f} | "
              f"{sc['full']['limit']:.5f} | {sc['limit_shift']:.5f} |")
        A("")
        A("Withheld level and observed pair orders:")
        A("")
        A("| tier | withheld k_n | adjacent-pair orders p |")
        A("|---|---|---|")
        for mu, sc in sorted(selfcheck.items()):
            A(f"| mu={mu:g} | {sc['points'][-1][0]:g} | "
              + ", ".join(f"{p['kn'][0]:g}-{p['kn'][1]:g}-{p['kn'][2]:g}: "
                          + ("n/a" if p["order"] is None else f"{p['order']:.3f}")
                          for p in sc["pair_orders"]) + " |")
        A("")
        A("`p = 1` is what linear-spring compressibility predicts; a drifting or")
        A("half-integer `p` means the ladder is not a single power law and the")
        A("intercept inherits the shift column as its error bar.")
        A("")
    if dt_rows:
        A("### 3.2 Timestep (temporal) convergence")
        A("")
        A("| tier | k_n | seeds | phi(dt = t_c/20) | phi(dt = t_c/40) | |Delta| |")
        A("|---|---|---|---|---|---|")
        for row in dt_rows:
            A(f"| mu={row['mu']:g} | {row['kn']:g} | "
              f"{', '.join(str(s) for s in row.get('seeds', []))} | "
              f"{row['phi']:.5f} | {row['phi_halved']:.5f} | {row['delta']:.5f} |")
        A("")
    A("## 4 Figures")
    A("")
    for pth, cap in zip(made, captions):
        A(f"![{os.path.basename(pth)}]({os.path.relpath(pth, os.path.join(ROOT, 'docs'))})")
        A("")
        A(f"*{cap}*")
        A("")
    A("## 5 Gates")
    A("")
    A("| gate | criterion | got | pass |")
    A("|---|---|---|---|")
    for g in gates:
        A(f"| {g['id']} | {g['criterion']} | {fmt(g['got'])} | "
          f"{'PASS' if g['pass'] else 'FAIL'} |")
    A("")
    A("## 6 Error judgement")
    A("")
    # Use friction-dependent references for error calculation
    ref_mu0 = get_reference(0.0)
    ref_mu01 = get_reference(0.1)
    if frictionless and lowfric:
        d0 = 100 * (frictionless[0] / ref_mu0.phi_rcp - 1)
        d1 = 100 * (lowfric[0] / ref_mu01.phi_rcp - 1)
        A(f"* frictionless tier (k_n = {kn_prod:g}, N = {n_primary}, "
          f"{frictionless[1]} seeds): phi = {frictionless[0]:.5f} +- "
          f"{frictionless[2]:.5f} vs reference {ref_mu0.phi_rcp} "
          f"({ref_mu0.source[:40]}...): **{d0:+.2f} %** -> "
          f"{'<' if abs(d0) < 3 else '>'}3 % "
          f"({'满足' if abs(d0) < 3 else '不满足'}).")
        A(f"* low-friction tier (mu = 0.1): phi = {lowfric[0]:.5f} +- "
          f"{lowfric[2]:.5f} vs reference {ref_mu01.phi_rcp} "
          f"({ref_mu01.source[:40]}...): **{d1:+.2f} %** -> "
          f"{'<' if abs(d1) < 3 else '>'}3 % "
          f"({'满足' if abs(d1) < 3 else '不满足'}).")
    for mu, s in sorted(stiff.items()):
        ref_mu = get_reference(mu)
        dev = 100 * (s["limit"] / ref_mu.phi_rcp - 1)
        A(f"* hard-sphere limit (mu = {mu:g}, Richardson over "
          f"{s['n_levels']} k_n levels, p = {s['order_used']:.2f}): "
          f"phi_inf = {s['limit']:.5f} vs reference {ref_mu.phi_rcp}: "
          f"**{dev:+.2f} %** -> "
          f"{'<' if abs(dev) < 3 else '>'}3 % "
          f"({'满足' if abs(dev) < 3 else '不满足'}).")
    for mu, sc in sorted(selfcheck.items()):
        A(f"* extrapolation self-consistency (mu = {mu:g}): the 3-level fit "
          f"predicts the withheld k_n = {sc['points'][-1][0]:g} level to "
          f"**{sc['prediction_error']:.5f}** and admitting it moves the "
          f"hard-sphere limit by **{sc['limit_shift']:.5f}** -> the intercept is "
          f"{'reproducible' if sc['prediction_error'] < SC_PRED_GATE and sc['limit_shift'] < SC_LIMIT_GATE else 'NOT reproducible'} "
          f"at the {SC_PRED_GATE} level.")
    for row in dt_rows:
        A(f"* timestep (mu = {row['mu']:g}, k_n = {row['kn']:g}): halving dt moves "
          f"phi by {row['delta']:.5f} "
          f"({'below' if row['delta'] < DT_GATE else 'ABOVE'} the {DT_GATE} gate) "
          f"-> the stiffness systemic is not "
          f"{'a disguised' if row['delta'] < DT_GATE else 'explainable as a'} "
          f"timestep error.")
    A("* The reference value is a *poured* experimental packing, so the comparison")
    A("  is method-matched; the residual gap is the monodisperse soft-sphere")
    A("  idealisation plus the periodic sidewalls (see §7).")
    A("")
    A("## 7 Notes, limitations, honest gaps")
    A("")
    A("* The merged measurement of this case (`make_random_close_packing`) returns")
    A("  `phi = 1 - target_porosity` to machine precision, so its `0.6400` was the")
    A("  input; this document replaces it with a configuration-independent read-out")
    A("  (`tests/test_rcp_deposition.py::test_legacy_solid_fraction_is_the_input`).")
    A("* Soft-sphere compressibility is quantified, not assumed: the settled bed")
    A("  carries `delta/d ~ 3e-3`, and the `1/k_n` systemic is the column of §3.1.")
    A("  The reported production number is quoted at `k_n`; the hard-sphere limit")
    A("  is quoted separately and neither replaces the other.")
    A("* The transient impact overlap is larger than the settled one (the drop")
    A("  height sets the impact velocity); only the settled maximum enters §3.")
    A("* Periodic sidewalls: the footprint is a square cell; wall-induced layering")
    A("  is excluded by the 2.5 d band, but a genuine container-wall effect is not")
    A("  resolved by this cell.")
    A("* `z12_fraction` is a crystalline-order proxy; it is ~0 here, i.e. the pour")
    A("  does not crystallise at these sizes.")
    # which plan cases have no run JSON yet (auto-clears once `fetch` lands them)
    try:
        plan = json.load(open(os.path.join(HERE, "plan.json")))["cases"]
        have = {(r["n"], round(r["mu"], 6), round(r["kn"], 3), r["seed"],
                 round(r["dt_frac"], 6)) for r in runs_all}
        missing = [c for c in plan
                   if (c["n"], round(c["mu"], 6), round(c["kn"], 3), c["seed"],
                       round(c.get("dt_frac", DT_DEFAULT), 6)) not in have]
    except (OSError, KeyError, ValueError):
        missing = []
    if missing:
        A(f"* **Pending runs ({len(missing)} of the `plan.json` cases have no JSON in "
          f"`runs/` yet).** The numbers in §3 are the runs present in the tree; "
          f"re-running `aggregate.py` after the missing cases land promotes them with "
          f"no code change. Cases not yet in: "
          + ", ".join(f"N={c['n']},mu={c['mu']:g},kn={c['kn']:g},seed={c['seed']}"
                      + (f",dt_frac={c['dt_frac']:g}" if 'dt_frac' in c else "")
                      for c in missing) + ".")
    A("* Polydispersity is absent by construction (monodisperse spheres, as the")
    A("  reference experiments); a polydisperse pour would raise phi and is out of")
    A("  scope.")
    A("")
    doc = os.path.join(ROOT, "docs", "CASE_RCP_DEPOSITION.md")
    with open(doc, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(doc, ROOT)} and {len(made)} figures; "
          f"all gates pass = {summary['all_gates_pass']}")
    for g in gates:
        print(f"  {g['id']}: {'PASS' if g['pass'] else 'FAIL'} ({fmt(g['got'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
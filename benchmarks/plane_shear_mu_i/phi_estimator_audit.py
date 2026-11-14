"""Round 15 (m20791) -- solid-fraction definition audit + strain convergence.

Why this round exists
---------------------
Round 14 left CASE-DEM-PLANE-SHEAR-MUIDI in honest FAIL and named five root
causes.  Two concern ``phi``:

* the *sweep* arm reported a ``phi`` that is constant with ``I`` (it used the
   settled height, frozen before shear);
* the *full-cell* arm reported ``phi_slab`` that **increases** with ``I``, up to
   0.722 at ``I = 0.3`` -- unphysical for a sheared sphere packing and 26 %
   above the same run's ``phi_bed_steady`` (0.575).

Two estimators of one state cannot both be right, so this round asks which one
is measuring what.  ``tensordem.mu_i_rheology.phi_definitions`` now returns the
three quantities by name, evaluated on the identical sample state:

===================== ==========================================================
``phi_bed``           ``N V_p / (W D plate_z)`` -- container-averaged, solid
                      volume over gap volume.  **The definition the mu(I)/phi(I)
                      literature uses**, hence the only one comparable with the
                      reference law.
``phi_extent``        ``N V_p / (W D (z_max-z_min+2r))`` -- same quantity rebuilt
                      from the particle extent; reference-free cross-check of
                      the platen bookkeeping.
``phi_centres``       centre-counted interior slab density (the archived
                      ``solid_fraction``).  A *bulk local* density: no centre can
                      lie within ``r`` of a wall, so it exceeds the container
                      average by a boundary-layer factor ``plate_z/(plate_z-2r)``
                      that **grows as the bed compacts** -- which by itself
                      manufactures the false ``phi`` rise with ``I``.
===================== ==========================================================

A first hypothesis of this round -- that the slab denominator omitted the
particle shell *inside* the slab -- was tested on homogeneous packings and
**falsified** (the shell-corrected value overshot by 11 %); the boundary-layer
normalisation above is the effect that actually reproduces the measurements.
The audit therefore *measures* ``slab_over_bed`` on every state instead of
assuming a factor.

Second half: **strain convergence**.  The archived protocol shears to
``max(min_shear_steps, gamma_target/(gdot dt))``, so the low-``I`` points only
reached ``gamma ~ 0.4`` while the high-``I`` points ran past ``gamma ~ 10``: the
two ends of the archived ``phi(I)`` curve sample different points of their own
transient.  ``--live`` shears to a requested strain and reports the drift of
``phi`` and ``mu`` between the first and last windows, so "steady state" is
measured rather than assumed.

Usage::

    python benchmarks/plane_shear_mu_i/phi_estimator_audit.py --synthetic
    python benchmarks/plane_shear_mu_i/phi_estimator_audit.py --live --I 0.01 --strain 2.0
    python benchmarks/plane_shear_mu_i/phi_estimator_audit.py --assemble

No literature constant enters the estimator comparison: every check is either an
analytic packing or an internal-consistency requirement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RUNS = HERE / "runs" / "phi_audit"
RUNS.mkdir(parents=True, exist_ok=True)

RADIUS = 0.005
W, D = 0.12, 0.05
MARGIN_FRAC = 0.25
Y_MARGIN_FRAC = 0.20


def _defs():
    from tensordem.mu_i_rheology import phi_definitions
    return phi_definitions


def archived_slab_phi(pos, radius=RADIUS, box_width=W, box_depth=D,
                      margin_frac=MARGIN_FRAC, y_margin_frac=Y_MARGIN_FRAC,
                      plate_z=None):
    """The archived estimator, reproduced verbatim (centre count / centre volume)."""
    if plate_z is None:
        plate_z = float(pos[:, 2].max()) + radius
    margin = margin_frac * plate_z
    z_lo, z_hi = margin, plate_z - margin
    y_lo, y_hi = y_margin_frac * box_depth, (1.0 - y_margin_frac) * box_depth
    sel = ((pos[:, 2] > z_lo) & (pos[:, 2] < z_hi)
           & (pos[:, 1] > y_lo) & (pos[:, 1] < y_hi))
    n_bulk = int(np.count_nonzero(sel))
    v_bulk = box_width * (y_hi - y_lo) * max(z_hi - z_lo, 1e-12)
    volume_particle = 4.0 / 3.0 * math.pi * radius ** 3
    return {"phi": n_bulk * volume_particle / v_bulk, "n": n_bulk,
            "volume": float(v_bulk), "plate_z": float(plate_z),
            "z_lo": float(z_lo), "z_hi": float(z_hi),
            "y_lo": float(y_lo), "y_hi": float(y_hi)}


# --------------------------------------------------------------------------
# synthetic packings with analytically known container-averaged solid fraction
# --------------------------------------------------------------------------
def homogeneous_packing(n: int = 6000, radius=RADIUS, box_width=0.24,
                        box_depth=0.10, height=0.12, seed: int = 7,
                        wall_depleted: bool = True):
    """Uniform random centres; phi_bed = N V_p/(W D H) is exact by construction."""
    rng = np.random.default_rng(seed)
    lo = radius if wall_depleted else 0.0
    pos = np.column_stack([
        rng.uniform(0.0, box_width, n),
        rng.uniform(lo, box_depth - lo if wall_depleted else box_depth, n),
        rng.uniform(lo, height - lo if wall_depleted else height, n)])
    phi = n * (4.0 / 3.0 * math.pi * radius ** 3) / (box_width * box_depth * height)
    return pos, phi, box_width, box_depth, height


def synthetic_audit(verbose: bool = True) -> dict:
    phi_definitions = _defs()
    rows = []
    # n chosen so the container-averaged phi is ~0.52 (a physical dense-packing
    # value); an over-dense random cloud would report phi > 1 and prove nothing.
    cases = [("gap_0.12", dict(n=2860, height=0.12)),
             ("gap_0.06", dict(n=1430, height=0.06)),
             ("gap_0.045", dict(n=1073, height=0.045))]
    for name, kw in cases:
        pos, phi_true, bw, bd, h = homogeneous_packing(**kw)
        leg = archived_slab_phi(pos, box_width=bw, box_depth=bd, plate_z=h)
        d = phi_definitions(pos, RADIUS, bw, bd, h, n_slab=leg["n"],
                            slab_volume=leg["volume"])
        rows.append({
            "packing": name, "n_particles": int(pos.shape[0]),
            "phi_true": float(phi_true),
            "phi_bed": d["phi_bed"], "phi_extent": d["phi_extent"],
            "phi_centres": d["phi_centres"],
            "predicted_boundary_factor": d["boundary_factor"],
            "measured_slab_over_bed": d["slab_over_bed"],
            "bed_err_pct": 100.0 * (d["phi_bed"] / phi_true - 1.0),
            "extent_err_pct": 100.0 * (d["phi_extent"] / phi_true - 1.0),
            "centres_err_pct": 100.0 * (d["phi_centres"] / phi_true - 1.0),
            "factor_err_pct": 100.0 * (d["slab_over_bed"]
                                       / d["boundary_factor"] - 1.0),
        })
        if verbose:
            r = rows[-1]
            print(f"[{name}] phi_true={r['phi_true']:.4f} bed={r['phi_bed']:.4f} "
                  f"({r['bed_err_pct']:+.1f}%) extent={r['phi_extent']:.4f} "
                  f"({r['extent_err_pct']:+.1f}%) centres={r['phi_centres']:.4f} "
                  f"({r['centres_err_pct']:+.1f}%) | measured slab/bed="
                  f"{r['measured_slab_over_bed']:.3f} vs predicted "
                  f"{r['predicted_boundary_factor']:.3f} "
                  f"({r['factor_err_pct']:+.1f}%)")
    # verdict: bed and extent are unbiased, centres are biased high by the
    # boundary-layer factor (that is the definitional defect, quantified).
    ok = (all(abs(r["bed_err_pct"]) < 2.0 for r in rows)
          and all(abs(r["extent_err_pct"]) < 3.0 for r in rows)
          and all(r["centres_err_pct"] > 5.0 for r in rows)
          and all(abs(r["factor_err_pct"]) < 10.0 for r in rows))
    out = {"rows": rows,
           "verdict": ("phi_bed_and_phi_extent_unbiased__phi_centres_biased_"
                       "high_by_boundary_factor") if ok else "unexpected",
           "ok": bool(ok)}
    (RUNS / "synthetic.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# live: strain-convergence run on the shipped servo-shear device
# --------------------------------------------------------------------------
def live_run(I_target: float, strain: float = 2.0, wall: str = "smooth",
             seed: int = 5279, kn: float = 6.25e4, pressure: float = 6.25e4,
             fill_height: float = 0.06) -> dict:
    from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear

    probe = ServoShearCase(radius=RADIUS, kn=kn, normal_stress=pressure,
                           inertial_number=I_target, fill_height=fill_height,
                           wall_mode=wall, seed=seed)
    gamma_dot = probe.inertial_number * math.sqrt(pressure / probe.density) \
        / probe.particle_diameter
    dt = probe.timestep
    steps = int(math.ceil(strain / (gamma_dot * dt))) + 100
    case = ServoShearCase(
        radius=RADIUS, density=2500.0, mu=0.5, mu_rolling=0.05, restitution=0.3,
        kn=kn, kt_ratio=0.5, box_width=W, box_depth=D, fill_height=fill_height,
        normal_stress=pressure, inertial_number=I_target, seed=seed,
        settle_steps=2500, min_shear_steps=steps, shear_strain=strain,
        sample_stride=max(1, steps // 400), bulk_stress=True, wall_mode=wall)
    res = simulate_servo_shear(case, return_series=True)
    series = res.get("series") or []
    if not series:
        raise RuntimeError("no series returned")

    def col(key):
        return np.array([s.get(key, float("nan")) for s in series], dtype=float)

    def win(arr, frac, tail=True):
        k = max(1, int(frac * len(arr)))
        return float(np.nanmean(arr[-k:] if tail else arr[:k]))

    cols = {k: col(k) for k in ("strain", "solid_fraction", "phi_bed_now",
                               "phi_extent", "slab_over_bed", "full_mu", "mu",
                               "plate_z", "normal_stress_pa")}
    drift_abs = {k: win(v, 0.2) - win(v, 0.2, tail=False)
                 for k, v in cols.items() if k in ("phi_bed_now", "phi_extent",
                                                   "solid_fraction", "full_mu",
                                                   "mu")}
    drift_rel = {k: 100.0 * v / abs(win(cols[k], 0.2, tail=False))
                 for k, v in drift_abs.items()}
    row = {
        "I_target": I_target, "wall_mode": wall,
        "strain_requested": strain, "strain_achieved": float(cols["strain"][-1]),
        "n_particles": int(res["n_particles"]), "dt": float(dt),
        "shear_steps_executed": int(res["shear_steps_executed"]),
        "wall_clock_s": float(res["wall_clock_s"]),
        "phi_container_final": win(cols["phi_bed_now"], 0.2),
        "phi_extent_final": win(cols["phi_extent"], 0.2),
        "phi_centres_final": win(cols["solid_fraction"], 0.2),
        "phi_bed_steady": res["phi_bed_steady"],
        "slab_over_bed_final": win(cols["slab_over_bed"], 0.2),
        "boundary_layer_factor": res.get("boundary_layer_factor"),
        "mu_full_final": win(cols["full_mu"], 0.2),
        "mu_wall_final": win(cols["mu"], 0.2),
        "I_imposed": res["inertial_number"],
        "I_bulk_final": res.get("inertial_number_bulk_final"),
        "pressure_control_error_final": res["pressure_control_error_final"],
        "servo_valid": bool(res["servo_valid"]),
        "coordination": res["coordination_number"],
        "drift_abs": drift_abs, "drift_rel_pct": drift_rel,
        "steady_phi": bool(abs(drift_rel["phi_bed_now"]) < 1.0),
        "steady_mu": bool(abs(drift_rel["full_mu"]) < 3.0),
        "series": {k: [float(x) for x in v] for k, v in cols.items()},
    }
    tag = f"I{I_target:g}_{wall}_s{seed}"
    (RUNS / f"{tag}.json").write_text(json.dumps(row, indent=2))
    print(f"I={I_target:g} wall={wall} gamma={row['strain_achieved']:.2f} "
          f"N={row['n_particles']} phi_bed={row['phi_container_final']:.4f} "
          f"phi_extent={row['phi_extent_final']:.4f} "
          f"phi_centres={row['phi_centres_final']:.4f} "
          f"slab/bed={row['slab_over_bed_final']:.3f} "
          f"mu_full={row['mu_full_final']:.4f} "
          f"drift(phi)={drift_rel['phi_bed_now']:+.2f}% "
          f"drift(mu)={drift_rel['full_mu']:+.2f}% {row['wall_clock_s']:.0f}s")
    return row


def assemble() -> dict:
    from reference_law import ref_mu, ref_phi

    rows = [json.loads(f.read_text()) for f in sorted(RUNS.glob("I*.json"))]
    rows.sort(key=lambda r: (r["wall_mode"], r["I_target"]))
    lines = ["# Round 15 - solid-fraction definition audit and strain convergence", "",
             "All estimators are evaluated on the identical final-window state;",
             "`phi_bed` is the container-averaged definition used by the",
             "reference law, `phi_centres` is the archived `solid_fraction`.", "",
             "| I | wall | gamma | N | phi_bed | phi_extent | phi_centres "
             "| slab/bed | bound.fac | phi_ref | mu_full | mu_ref | err_mu[%] "
             "| drift phi[%] | drift mu[%] | steady |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mu_ref = ref_mu(r["I_target"])
        phi_ref = ref_phi(r["I_target"])
        r["phi_ref"] = phi_ref
        r["mu_ref"] = mu_ref
        r["err_mu_pct"] = 100.0 * (r["mu_full_final"] / mu_ref - 1.0)
        r["err_phi_pct"] = 100.0 * (r["phi_container_final"] / phi_ref - 1.0)
        lines.append(
            f"| {r['I_target']:g} | {r['wall_mode']} | {r['strain_achieved']:.2f} "
            f"| {r['n_particles']} | {r['phi_container_final']:.4f} "
            f"| {r['phi_extent_final']:.4f} | {r['phi_centres_final']:.4f} "
            f"| {r['slab_over_bed_final']:.3f} | {r['boundary_layer_factor']:.3f} "
            f"| {phi_ref:.4f} | {r['mu_full_final']:.4f} | {mu_ref:.4f} "
            f"| {r['err_mu_pct']:+.1f} | {r['drift_rel_pct']['phi_bed_now']:+.2f} "
            f"| {r['drift_rel_pct']['full_mu']:+.2f} "
            f"| {'Y' if r['steady_phi'] and r['steady_mu'] else 'N'} |")
    (RUNS / "tables.md").write_text("\n".join(lines) + "\n")
    (RUNS / "result.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    figure(rows)
    return {"rows": rows}


def figure(rows) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from reference_law import ref_mu, ref_phi
    except Exception as exc:                          # pragma: no cover
        print(f"figure skipped: {exc}")
        return
    Is = np.array([r["I_target"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    grid = np.logspace(math.log10(max(Is.min() / 2, 1e-4)),
                       math.log10(Is.max() * 2), 60)
    ax[0].plot(Is, [r["phi_centres_final"] for r in rows], "o--",
               label=r"centre-counted slab (archived $\phi$)")
    ax[0].plot(Is, [r["phi_container_final"] for r in rows], "s-",
               label=r"container-averaged $\phi$ (literature definition)")
    ax[0].plot(Is, [r["phi_extent_final"] for r in rows], "^:",
               label="extent cross-check")
    ax[0].plot(grid, ref_phi(grid), "k:", label="reference band (provenance-unverified)")
    ax[0].axhline(0.7405, color="grey", lw=0.8, ls="-.", label="HCP limit")
    ax[0].set_xscale("log"); ax[0].set_xlabel("I"); ax[0].set_ylabel(r"$\phi$")
    ax[0].legend(fontsize=6.5); ax[0].set_title("three phi definitions")
    ax[1].plot(Is, [r["slab_over_bed_final"] for r in rows], "o-",
               label="measured slab/bed")
    ax[1].plot(Is, [r["boundary_layer_factor"] for r in rows], "k--",
               label=r"$plate_z/(plate_z-2r)$")
    ax[1].set_xscale("log"); ax[1].set_xlabel("I")
    ax[1].set_ylabel("slab / container phi")
    ax[1].legend(fontsize=8); ax[1].set_title("definitional gap, measured")
    ax[2].plot(Is, [r["mu_full_final"] for r in rows], "o-", label="full-cell virial")
    ax[2].plot(grid, ref_mu(grid), "k:", label="mu(I) reference")
    for r in rows:
        ax[2].annotate(f"{r['err_mu_pct']:+.0f}%", (r["I_target"], r["mu_full_final"]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[2].set_xscale("log"); ax[2].set_xlabel("I"); ax[2].set_ylabel(r"$\mu$")
    ax[2].legend(fontsize=8); ax[2].set_title("friction coefficient")
    fig.tight_layout()
    fig.savefig(RUNS / "fig_phi_definitions.png", dpi=130)
    print(f"figure -> {RUNS / 'fig_phi_definitions.png'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--I", type=float, default=0.01)
    ap.add_argument("--strain", type=float, default=2.0)
    ap.add_argument("--wall", default="smooth", choices=["smooth", "rough"])
    ap.add_argument("--seed", type=int, default=5279)
    ap.add_argument("--kn", type=float, default=6.25e4)
    args = ap.parse_args(argv)
    if args.synthetic:
        synthetic_audit()
    if args.live:
        live_run(args.I, strain=args.strain, wall=args.wall, seed=args.seed,
                 kn=args.kn)
    if args.assemble:
        assemble()
    if not (args.synthetic or args.live or args.assemble):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
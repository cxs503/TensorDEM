"""Anchor case ``CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE``.

Plane-strain biaxial cell of a frictional, sparsely freeze-bonded disc specimen
(24 x 48 spacings, confining pressure ``p = 100 kPa``, wall friction ``0.55``,
4000 consolidation steps); the deliverable quantity is the shear-band
inclination ``theta`` measured from the ``sigma_3`` axis and compared with the
literature value for dense Hostun sand.

The case is called *KNOWN-ANGLE* because the measurement operator is validated
**before** it is trusted on the DEM result: a synthetic slip band of an exactly
known inclination is injected into the very lattice of the case and pushed
through the unmodified production measurement chain (affine background removal
-> k-nearest-neighbour deformation gradient -> Radon/projection focus).  Only
then is the same operator applied to the DEM snapshots.  That ordering is the
point of the case: an earlier delivery reported 44.3 deg (26 % error) purely
from a biased estimator (R2 diagnosis, see
``docs/CASE_BIAXIAL_SHEAR_BAND_R2.md``).

Public API
----------
``run_known_angle_benchmark``  reference-vs-recovered table (analytic reference)
``run_anchor_dem``             the anchor DEM run (24x48, 100 kPa, mu_wall 0.55)
``judge_anchor``               pre-declared acceptance arithmetic
``run_case``                   both of the above + environment echo + JSON artefact
``render_figure``              two-panel figure for the report

Canonical reference
-------------------
Desrues & Viggiani (2004), *Int. J. Numer. Anal. Methods Geomech.* **28**:
279-321 -- dense Hostun sand plane-strain biaxial tests localise into a band at
55-60 deg from the ``sigma_3`` axis (midpoint 60 deg, identical to the
Mohr-Coulomb value ``45 + phi/2`` at ``phi = 30 deg``).  Acceptance is the owner
standard: relative error ``< 15 %`` plus a resolution requirement (>= 3 late
snapshots, spread < 10 deg, radon ``z > 3``) so that a non-localised run cannot
be scored as a pass.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np

CASE_KEY = "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE"

THETA_REF_DEG = 60.0
TOLERANCE = 0.15
THETA_BAND_REF_DEG = (55.0, 60.0)

ANCHOR_SOURCE = (
    "Desrues & Viggiani (2004) Int. J. Numer. Anal. Methods Geomech. 28:279-321: "
    "dense Hostun sand plane-strain biaxial, shear band 55-60 deg from sigma_3 "
    "(midpoint 60 deg = Mohr-Coulomb 45 + phi/2 at phi = 30 deg)")

#: The pre-declared anchor configuration (mirrors the case key in the task board).
ANCHOR_CONFIG: dict[str, Any] = {
    "columns": 24,
    "rows": 48,
    "confining_stress": 100.0e3,
    "wall_friction": 0.55,
    "consolidation_steps": 4000,
    "strain_rate": 0.05,
    "strain_target": 0.12,
    "snapshot_strain": 0.01,
    "servo_gain": 3.0e-4,
    "servo_velocity_limit": 2.0,
    "radius": 0.75,
    "freeze_bond_fraction": 0.35,
}

#: Reduced configuration used by the unit test / smoke path (never a result).
QUICK_CONFIG: dict[str, Any] = {
    "columns": 8,
    "rows": 16,
    "confining_stress": 100.0e3,
    "wall_friction": 0.55,
    "consolidation_steps": 300,
    "strain_rate": 0.05,
    "strain_target": 0.03,
    "snapshot_strain": 0.01,
    "servo_gain": 3.0e-4,
    "servo_velocity_limit": 2.0,
    "radius": 0.75,
    "freeze_bond_fraction": 0.35,
}

CASE_LATTICE = dict(spacing=1.5, jitter=0.03, seed=7,
                    wall_margin_diameters=2.0, diameter=1.5)


# ---------------------------------------------------------------------------
# geometry of the case (identical lattice / window as the production cell)
# ---------------------------------------------------------------------------
def case_lattice(columns: int = 24, rows: int = 48, *, spacing: float = 1.5,
                 jitter: float = 0.03, seed: int = 7) -> np.ndarray:
    """Jittered triangular disc lattice exactly as ``_make_specimen`` builds it.

    Kept dependency-free (numpy only) so the known-angle benchmark runs without
    instantiating a torch assembly; the generator/seed/jitter convention matches
    ``tensordem.biaxial_shear_band._make_specimen``.
    """
    if columns < 2 or rows < 2:
        raise ValueError("columns and rows must be >= 2")
    row_spacing = float(spacing) * math.sqrt(3.0) / 2.0
    width = (int(columns) - 1) * float(spacing)
    generator = np.random.default_rng(int(seed))
    points = []
    for row in range(int(rows)):
        offset = 0.5 * spacing if row % 2 else 0.0
        for column in range(int(columns)):
            x = -0.5 * width + column * spacing + offset
            if abs(x) > 0.5 * width + 1e-9:
                continue
            points.append((x, -0.5 * (int(rows) - 1) * row_spacing + row * row_spacing))
    pos = np.asarray(points, dtype=np.float64)
    return pos + (float(jitter) * float(spacing)) * generator.standard_normal(pos.shape)


def measurement_window(pos: np.ndarray, *, wall_margin_diameters: float = 2.0,
                       diameter: float = 1.5) -> tuple[float, float, float, float]:
    """``(xmin, xmax, ymin, ymax)`` of the measurement window.

    The boundary layer within ``wall_margin_diameters`` diameters of every wall is
    excluded: rough walls nucleate their own boundary bands, which are not the
    material shear band.
    """
    pos = np.asarray(pos, dtype=np.float64)
    margin = float(wall_margin_diameters) * float(diameter)
    xmin, xmax = float(pos[:, 0].min()) + margin, float(pos[:, 0].max()) - margin
    ymin, ymax = float(pos[:, 1].min()) + margin, float(pos[:, 1].max()) - margin
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("measurement window is empty; reduce wall_margin_diameters")
    return (xmin, xmax, ymin, ymax)


# ---------------------------------------------------------------------------
# (a) known-angle benchmark: analytic reference vs recovered inclination
# ---------------------------------------------------------------------------
def run_known_angle_benchmark(angles_deg: Sequence[float] = (30.0, 45.0, 60.0, 70.0),
                              *, columns: int = 24, rows: int = 48,
                              slip: float | None = None,
                              band_width: float | None = None,
                              volumetric_strain: float = 0.0,
                              k: int = 8) -> dict[str, Any]:
    """Inject bands of exactly known inclination and recover them.

    The reference value here is *analytic* (the injected angle), so the error
    column of this table is a pure operator error -- no physics, no DEM, no
    literature tolerance.  The pre-R2 estimator is measured on the identical
    strain field as the documented biased baseline.
    """
    from tensordem.biaxial_band_diagnostics import estimator_bias_sweep

    pos = case_lattice(columns, rows)
    window = measurement_window(
        pos, wall_margin_diameters=CASE_LATTICE["wall_margin_diameters"],
        diameter=CASE_LATTICE["diameter"])
    sweep = estimator_bias_sweep(
        pos, angles_deg=[float(a) for a in angles_deg], slip=slip,
        band_width=band_width, volumetric_strain=volumetric_strain,
        measurement_window=window, k=k)
    rows_out = []
    for row in sweep["rows"]:
        recovered = float(row["recovered_deg"])
        injected = float(row["injected_deg"])
        rel = abs(recovered - injected) / injected if np.isfinite(recovered) else math.nan
        rows_out.append({
            "injected_deg": injected,
            "recovered_deg": recovered,
            "abs_error_deg": (abs(recovered - injected)
                              if np.isfinite(recovered) else math.nan),
            "relative_error": rel,
            "within_tolerance": bool(np.isfinite(rel) and rel <= TOLERANCE),
            "eta2": row["eta2"],
            "z_score": row["z_score"],
            "legacy_recovered_deg": row.get("legacy_recovered_deg"),
            "legacy_abs_error_deg": (abs(row["legacy_error_deg"])
                                     if row.get("legacy_error_deg") is not None
                                     and np.isfinite(row["legacy_error_deg"]) else math.nan),
        })
    finite = [r["abs_error_deg"] for r in rows_out if np.isfinite(r["abs_error_deg"])]
    legacy = [r["legacy_abs_error_deg"] for r in rows_out
              if np.isfinite(r["legacy_abs_error_deg"])]
    return {
        "kind": "known_angle",
        "reference": "injected analytic band inclination (no DEM, no literature)",
        "lattice": {"columns": int(columns), "rows": int(rows), **CASE_LATTICE},
        "measurement_window": list(window),
        "aspect_ratio": sweep["aspect_ratio"],
        "slip": sweep["slip"],
        "band_width": sweep["band_width"],
        "volumetric_strain": sweep["volumetric_strain"],
        "rows": rows_out,
        "max_abs_error_deg": float(max(finite)) if finite else math.nan,
        "mean_abs_error_deg": float(np.mean(finite)) if finite else math.nan,
        "legacy_max_abs_error_deg": float(max(legacy)) if legacy else math.nan,
        "passed": bool(finite and max(finite) <= TOLERANCE * min(angles_deg)),
    }


# ---------------------------------------------------------------------------
# (b) acceptance arithmetic
# ---------------------------------------------------------------------------
def judge_anchor(result: dict[str, Any], *, theta_ref: float = THETA_REF_DEG,
                 tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Pre-declared acceptance arithmetic for the DEM anchor run."""
    band = result["band"]
    theta = float(band["angle_deg"])
    representative = band.get("representative") or {}
    cumulative = representative.get("cumulative") or {}
    z_score = float(cumulative.get("z_score", math.nan))
    eta2 = float(cumulative.get("eta2", math.nan))
    rel = abs(theta - theta_ref) / theta_ref if np.isfinite(theta) else math.nan
    resolved = bool(band.get("resolved", False))
    localised = bool(resolved and np.isfinite(z_score) and z_score > 3.0
                     and np.isfinite(eta2) and eta2 > 0.2)
    return {
        "ref_deg": float(theta_ref),
        "got_deg": theta,
        "relative_error": rel,
        "tolerance": float(tolerance),
        "within_tolerance": bool(np.isfinite(rel) and rel <= tolerance),
        "resolved": resolved,
        "late_snapshots": band.get("late_snapshots"),
        "angle_std_deg": float(band.get("angle_std_deg", math.nan)),
        "z_score": z_score,
        "eta2": eta2,
        "localised": localised,
        "passed": bool(np.isfinite(rel) and rel <= tolerance and localised),
    }


# ---------------------------------------------------------------------------
# (c) the anchor DEM run
# ---------------------------------------------------------------------------
def run_anchor_dem(*, quick: bool = False, record_snapshots: bool = True,
                   **overrides: Any) -> dict[str, Any]:
    """Run the pre-declared anchor cell and return the raw result dict."""
    from tensordem.biaxial_shear_band import run_biaxial_shear_band

    config = dict(QUICK_CONFIG if quick else ANCHOR_CONFIG)
    config.update({k: v for k, v in overrides.items() if v is not None})
    freeze_bond = float(config.pop("freeze_bond_fraction", 0.35))
    radius = float(config.pop("radius", 0.75))
    started = time.time()
    result = run_biaxial_shear_band(
        confining_stress=float(config["confining_stress"]),
        columns=int(config["columns"]), rows=int(config["rows"]),
        radius=radius,
        strain_rate=float(config["strain_rate"]),
        strain_target=float(config["strain_target"]),
        consolidation_steps=int(config["consolidation_steps"]),
        snapshot_strain=float(config["snapshot_strain"]),
        servo_gain=float(config["servo_gain"]),
        servo_velocity_limit=float(config["servo_velocity_limit"]),
        wall_friction=float(config["wall_friction"]),
        material_overrides={"freeze_bond_fraction": freeze_bond},
        record_snapshots=record_snapshots)
    result["wall_time_s"] = time.time() - started
    result["config"] = config
    result["config"]["freeze_bond_fraction"] = freeze_bond
    result["config"]["radius"] = radius
    result["config"]["record_snapshots"] = bool(record_snapshots)
    return result


def _environment() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }


def repro_command(out_dir: str = "artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE") -> str:
    """Exact command that regenerates the artefact (thread pin included)."""
    return ("TENSORDEM_THREADS=4 PYTHONPATH=src python "
            "benchmarks/biaxial_shear_band/run_anchor_case.py "
            f"--out-dir {out_dir}")


# ---------------------------------------------------------------------------
# (d) orchestration
# ---------------------------------------------------------------------------
def run_case(*, output_dir: str = "artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE",
             quick: bool = False, run_dem: bool = True,
             angles_deg: Sequence[float] = (30.0, 45.0, 60.0, 70.0),
             figure: bool = True) -> dict[str, Any]:
    """Run both parts of the case and persist ``run.json`` (+ optional figure).

    Returns the payload that was written.  ``run_dem=False`` is the cheap path
    used by unit tests (known-angle part only).
    """
    payload: dict[str, Any] = {
        "case": CASE_KEY,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchor": {
            "quantity": "shear band inclination from sigma_3 axis (deg)",
            "ref_deg": THETA_REF_DEG,
            "ref_range_deg": list(THETA_BAND_REF_DEG),
            "tolerance": TOLERANCE,
            "source": ANCHOR_SOURCE,
        },
        "config": dict(QUICK_CONFIG if quick else ANCHOR_CONFIG),
        "repro_cmd": repro_command(),
        "environment": _environment(),
    }
    t0 = time.time()
    payload["known_angle"] = run_known_angle_benchmark(angles_deg)
    if run_dem:
        dem = run_anchor_dem(quick=quick, record_snapshots=figure)
        verdict = judge_anchor(dem)
        payload["dem"] = {
            "verdict": verdict,
            "specimen": dem["specimen"],
            "loading": dem["loading"],
            "peak": dem["peak"],
            "theory": dem["theory"],
            "band": {"angle_deg": dem["band"]["angle_deg"],
                     "angle_std_deg": dem["band"]["angle_std_deg"],
                     "late_snapshots": dem["band"]["late_snapshots"],
                     # keep the representative Radon block so the verdict can be
                     # re-derived from the artefact alone (no recomputation)
                     "representative": dem["band"].get("representative"),
                     "history": dem["band"]["history"]},
            "broken_bonds": dem["broken_bonds"],
            "wall_time_s": dem["wall_time_s"],
            "config": dem["config"],
        }
        if figure:
            payload["figure_fields"] = _figure_fields(dem, verdict)
    payload["wall_time_s"] = time.time() - t0
    passed_items = [payload["known_angle"]["passed"]]
    if "dem" in payload:
        passed_items.append(payload["dem"]["verdict"]["passed"])
    payload["summary"] = {
        "known_angle_passed": payload["known_angle"]["passed"],
        "dem_passed": payload["dem"]["verdict"]["passed"] if "dem" in payload else None,
        "dem_theta_deg": (payload["dem"]["verdict"]["got_deg"] if "dem" in payload
                          else None),
        "dem_relative_error": (payload["dem"]["verdict"]["relative_error"]
                               if "dem" in payload else None),
        "all_passed": bool(all(passed_items)),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "run.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)
    payload["artifacts"] = {"run_json": out_path}
    if figure and "figure_fields" in payload:
        from tensordem.biaxial_anchor_case import render_figure

        fig_path = os.path.join(output_dir, "shear_band_case.png")
        render_figure(payload, fig_path)
        payload["artifacts"]["figure"] = fig_path
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=1, default=float)
    return payload


def _figure_fields(dem: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    """Last late snapshot's ``gamma`` field + fitted band line, for the figure."""
    from tensordem.biaxial_shear_band import local_shear_strain

    history = [h for h in dem["band"]["history"]
               if np.isfinite(h.get("folded_angle_deg", math.nan))]
    field: dict[str, Any] = {"history": history}
    snapshots = dem.get("snapshots") or []
    if len(snapshots) >= 2 and history:
        # rebuild the field of the last measured window from the last two snapshots
        ref, now = snapshots[-2], snapshots[-1]
        window = ref.get("window") or now.get("window")
        pos = ref["positions"]
        if window is not None:
            xmin, xmax, ymin, ymax = window
            inside = ((pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)
                      & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax))
        else:
            inside = np.ones(pos.shape[0], dtype=bool)
        gamma = local_shear_strain(pos[inside], now["positions"][inside], k=8)
        field.update({"x": pos[inside, 0].tolist(), "y": pos[inside, 1].tolist(),
                      "gamma": np.asarray(gamma, dtype=float).tolist()})
    return field


def render_figure(payload: dict[str, Any], path: str) -> str:
    """Two-panel figure: (a) known-angle ref vs got, (b) DEM band field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    known = payload["known_angle"]
    injected = [r["injected_deg"] for r in known["rows"]]
    recovered = [r["recovered_deg"] for r in known["rows"]]
    legacy = [r["legacy_recovered_deg"] for r in known["rows"]]
    axes[0].plot([0, 90], [0, 90], "k--", lw=1.0, label="exact")
    axes[0].plot(injected, recovered, "o-", color="#c62828",
                 label=f"production estimator (max |err| {known['max_abs_error_deg']:.1f} deg)")
    if all(v is not None for v in legacy):
        axes[0].plot(injected, legacy, "s:", color="#616161",
                     label=f"pre-R2 estimator (max |err| {known['legacy_max_abs_error_deg']:.1f} deg)")
    axes[0].set_xlabel("injected band inclination (deg)")
    axes[0].set_ylabel("recovered inclination (deg)")
    axes[0].set_title("(a) known-angle benchmark, 24x48 case lattice")
    axes[0].set_xlim(20, 80)
    axes[0].set_ylim(20, 80)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")

    fields = payload.get("figure_fields") or {}
    if fields.get("x"):
        x = np.asarray(fields["x"])
        y = np.asarray(fields["y"])
        gamma = np.asarray(fields["gamma"])
        scatter = axes[1].scatter(x, y, c=gamma, s=8, cmap="inferno",
                                  vmin=0.0, vmax=float(np.percentile(gamma, 95)))
        theta = payload["dem"]["verdict"]["got_deg"]
        x0, x1 = float(x.min()), float(x.max())
        ymid = 0.5 * (float(y.min()) + float(y.max()))
        if np.isfinite(theta):
            slope = math.tan(math.radians(float(theta)))
            axes[1].plot([x0, x1], [ymid - slope * (x1 - x0) / 2,
                                    ymid + slope * (x1 - x0) / 2],
                         "c--", lw=1.4,
                         label=f"fitted band {theta:.1f} deg")
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].set_title("(b) local shear strain gamma, late window")
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(fontsize=7, loc="upper right")
        fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.03, label="gamma (-)")
    else:
        known_ref = payload["anchor"]["ref_deg"]
        verdict = payload["dem"]["verdict"]
        axes[1].axhline(known_ref, color="k", ls="--", lw=1.0,
                        label=f"Desrues & Viggiani midpoint {known_ref:.0f} deg")
        axes[1].fill_between([-0.5, 0.5], known_ref * (1 - TOLERANCE),
                             known_ref * (1 + TOLERANCE), color="#90caf9", alpha=0.5,
                             label="+-15 % band")
        axes[1].bar([0.0], [verdict["got_deg"]], width=0.4, color="#c62828")
        axes[1].set_xlim(-0.5, 0.5)
        axes[1].set_xticks([])
        axes[1].set_ylabel("band inclination (deg)")
        axes[1].set_title(f"(b) DEM anchor run, error {100 * verdict['relative_error']:.1f} %")
        axes[1].legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


__all__ = [
    "CASE_KEY", "THETA_REF_DEG", "TOLERANCE", "ANCHOR_SOURCE", "ANCHOR_CONFIG",
    "QUICK_CONFIG", "case_lattice", "measurement_window",
    "run_known_angle_benchmark", "judge_anchor", "run_anchor_dem", "run_case",
    "render_figure", "repro_command",
]
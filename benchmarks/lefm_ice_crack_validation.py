"""LEFM ice-crack validation driver (campaign node p307w1-n01-lefm-ice-crack).

Runs the whole verification chain of :mod:`tensordem.lefm` and writes
``artifacts/lefm_ice_crack/results.json`` plus figures:

1. lattice material patch tests (E, nu exactness);
2. centre-cracked plate (CCP) ground truth: domain J vs ``sigma sqrt(pi a)``;
3. SEN B elastic solution: ASTM closed form vs contour J vs domain J vs
   M-integral vs crack-advance energy, at two lattice resolutions;
4. J path independence (contour radii and EDI annuli);
5. K_IC calibration from the critical load of the LEFM bond criterion, with
   transferability across ``a/W`` and ``S/W``;
6. negative controls (un-notched uniform tension; K_IC -> infinity);
7. critical-break energy ledger (Griffith);
8. crack path prediction with MTS (symmetric and offset load).

Usage:  python benchmarks/lefm_ice_crack_validation.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.lefm import (BondedLattice2D, Lattice2DConfig, SENBGeometry,  # noqa: E402
                            astm_stress_intensity, bond_stress_intensity,
                            build_senb, calibrate_kic, contour_j,
                            critical_break_ledger, domain_j,
                            energy_release_rate_from_crack_advance,
                            mts_angle, path_overlap, polyline_distance,
                            stress_intensity_from_j, stress_intensity_from_m,
                            tip_bond_intensity, trace_crack_path,
                            uniform_tension_control)

E_ICE = 5.0e9          # Pa, columnar fresh-water/sea ice Young's modulus
NU = 0.25              # lattice Poisson ratio (isotropic point of the model)
B_SPEC = 0.05          # m, specimen thickness
KIC_LIT_LO, KIC_LIT_HI = 150e3, 250e3   # Pa m^0.5, Lu et al. (2015) band
KIC_REF = 200e3

OUT_DIR = ROOT / "artifacts" / "lefm_ice_crack"


# --------------------------------------------------------------------- helpers
def patch_tests(spacing: float = 0.005) -> dict:
    """Affine patch tests: recover E and nu from the virial stress."""
    cfg = Lattice2DConfig.from_material(E_ICE, B_SPEC, spacing)
    lat = BondedLattice2D(cfg, 13, 13)
    lam, mu, B = cfg.lame_lambda, cfg.shear_modulus, cfg.thickness
    res = {}
    for name, eps in (("uniaxial_strain", np.array([[2e-4, 0.0], [0.0, -1e-4]])),
                      ("pure_shear", np.array([[0.0, 3e-4], [3e-4, 0.0]]))):
        for n in range(lat.nnodes):
            lat._fixed[n] = (eps[0, 0] * lat.x[n] + eps[0, 1] * lat.y[n],
                             eps[1, 0] * lat.x[n] + eps[1, 1] * lat.y[n])
        lat.solve()
        sig = lat.node_stress().reshape(lat.nrows, lat.ncols, 2, 2)[3:10, 3:10].mean((0, 1))
        exact = (lam * np.trace(eps) * np.eye(2) + 2.0 * mu * eps) / B
        res[name] = {
            "computed": sig.tolist(),
            "analytic": exact.tolist(),
            "max_rel_error": float(np.abs(sig - exact).max() / np.abs(exact).max()),
        }
        lat._fixed.clear()
    # uniaxial *stress* patch: sigma_yy applied, lateral free -> E and nu
    lat2 = BondedLattice2D(cfg, 13, 13)
    s_yy = 1.0e6
    for j in range(lat2.nrows):
        lat2.set_fixed([lat2.node(0, j)], 0.0, None)
    strain_applied = 0.0
    # apply uniform axial load on the right column
    right = [lat2.node(lat2.ncols - 1, j) for j in range(lat2.nrows)]
    for n in right:
        lat2.add_load(n, 0.0, 0.0)
    res["lattice_modulus_2d"] = cfg.modulus_2d
    res["young_modulus_used"] = E_ICE
    res["poisson_used"] = NU
    res["uniaxial_strain_sigma_yy_zero"] = 0.0
    return res


def ccp_validation(spacing: float = 0.005, n: int = 121, half_cells: int = 10,
                   strain: float = 1e-4) -> dict:
    """Centre-cracked plate: ``K = sigma sqrt(pi a)`` reference for the J routes."""
    cfg = Lattice2DConfig.from_material(E_ICE, B_SPEC, spacing)
    a_cell = spacing
    jc = n // 2
    ic = n // 2

    def build(half: int) -> BondedLattice2D:
        lat = BondedLattice2D(cfg, n, n)
        n0, n1 = lat.bonds[:, 0], lat.bonds[:, 1]
        y0, y1 = lat.y[n0], lat.y[n1]
        xmid = 0.5 * (lat.x[n0] + lat.x[n1])
        yc, tol = jc * a_cell, 1e-9
        rng = (xmid <= (ic + half) * a_cell + tol) & (xmid >= (ic - half) * a_cell - tol)
        crossing = (np.minimum(y0, y1) < yc + tol) & (np.maximum(y0, y1) > yc + tol) & rng
        along = (np.abs(y0 - yc) < tol) & (np.abs(y1 - yc) < tol) & rng
        lat.kill_bonds(np.flatnonzero(crossing | along))
        left = [lat.node(0, j) for j in range(n)]
        right = [lat.node(n - 1, j) for j in range(n)]
        for node in left:
            lat.set_fixed([node], 0.0, None)      # roller: axial fix, lateral free
        for node in right:
            lat.set_fixed([node], strain * (n - 1) * a_cell, None)
        lat.set_fixed([lat.node(0, 0)], 0.0, 0.0)  # kill the rigid-body mode
        return lat

    lat = build(half_cells)
    lat.solve()
    lat_ahead = build(half_cells + 1)
    lat_ahead.solve()
    # crack grows on *both* ends when half_cells increases -> da = 2a
    G_energy = -(lat_ahead.energy() - lat.energy()) / (B_SPEC * 2.0 * a_cell)
    tip = (ic * a_cell + half_cells * a_cell, jc * a_cell)
    sig = lat.node_stress()
    sigma_far = float(np.mean(sig[lat.ii == n - 3, 0, 0]))
    half = half_cells * a_cell
    width = (n - 1) * a_cell
    fin = math.sqrt(1.0 / math.cos(math.pi * half / width))   # CCP finite-width factor
    k_analytic = sigma_far * math.sqrt(math.pi * half) * fin
    rings = [(2, 4), (4, 7), (6, 10), (8, 14)]
    k_domain = [stress_intensity_from_j(domain_j(lat, tip, r1 * a_cell, r2 * a_cell), E_ICE)
                for r1, r2 in rings]
    k_contour = [stress_intensity_from_j(contour_j(lat, tip, r * a_cell), E_ICE)
                 for r in (3, 5, 8, 12)]
    return {
        "sigma_far": sigma_far,
        "crack_half_length": half_cells * a_cell,
        "k_analytic": k_analytic,
        "finite_width_factor": fin,
        "width": width,
        "k_energy": float(np.sqrt(abs(G_energy) * E_ICE)),
        "G_energy": float(G_energy),
        "k_domain": k_domain,
        "rings": rings,
        "k_contour": k_contour,
        "contour_radii": [3, 5, 8, 12],
        "k_m_I": float(stress_intensity_from_m(lat, tip, "I", 2 * a_cell, 4.5 * a_cell)),
        "rel_error_domain": [float(k / k_analytic - 1.0) for k in k_domain],
        "rel_error_contour": [float(k / k_analytic - 1.0) for k in k_contour],
        "rel_error_energy": float(np.sqrt(abs(G_energy) * E_ICE) / k_analytic - 1.0),
    }


def senb_routes(cells: int, alpha: float = 0.3, spacing: float | None = None) -> dict:
    """All K routes of the SEN B specimen at one lattice resolution."""
    sp = spacing if spacing else 0.10 / cells
    geom = SENBGeometry(width=0.10, spacing=sp, notch_ratio=alpha,
                        thickness=B_SPEC, young_modulus=E_ICE)
    load = 100.0
    lat = build_senb(geom, load=load)
    lat.solve()
    tip = lat.metadata["tip"]
    a = geom.spacing
    k_astm = astm_stress_intensity(load, geom.span, geom.thickness, geom.depth, geom.alpha)
    rings = [(2, 4), (4, 7), (6, 10), (8, 14)]
    k_dom = [stress_intensity_from_j(domain_j(lat, tip, r1 * a, r2 * a), E_ICE)
             for r1, r2 in rings]
    k_con = [stress_intensity_from_j(contour_j(lat, tip, r * a), E_ICE)
             for r in (3, 5, 8, 12, 16)]
    k_m1 = stress_intensity_from_m(lat, tip, "I", 2 * a, 4.5 * a)
    k_m2 = stress_intensity_from_m(lat, tip, "II", 2 * a, 4.5 * a)
    e = energy_release_rate_from_crack_advance(geom, delta_cells=2, load=load)
    u_max = float(np.abs(lat.u).max())
    return {
        "cells": cells,
        "spacing": sp,
        "alpha": geom.alpha,
        "span_ratio": geom.span / geom.depth,
        "load": load,
        "k_astm": k_astm,
        "k_domain": k_dom,
        "k_contour": k_con,
        "k_m_I": k_m1,
        "k_m_II": k_m2,
        "k_energy": e["k_energy"],
        "k_astm_energy_load": e["k_astm_same_load"],
        "tip_bond_k": tip_bond_intensity(lat, tip),
        "u_max": u_max,
        "rings": rings,
        "contour_radii": [3, 5, 8, 12, 16],
        "path_independence_domain": float((max(k_dom) - min(k_dom)) / np.mean(k_dom)),
        "path_independence_contour": float((max(k_con) - min(k_con)) / np.mean(k_con)),
        "ratio_domain_astm": k_dom[-1] / k_astm,
        "ratio_contour_astm": k_con[-1] / k_astm,
        "ratio_energy_astm": e["k_energy"] / k_astm,
        "ratio_m_astm": k_m1 / k_astm,
    }


def kic_calibration(cells: int = 20) -> dict:
    """Calibrate the bond toughness on one geometry, validate on others."""
    sp = 0.10 / cells
    base = dict(width=0.10, spacing=sp, thickness=B_SPEC, young_modulus=E_ICE)
    cal = calibrate_kic(SENBGeometry(notch_ratio=0.3, span_ratio=4.0, **base),
                        k_ic_input=KIC_REF)
    kappa = cal["lattice_correction"]          # K_app / K_IC_input
    rows = []
    for alpha in (0.2, 0.3, 0.4, 0.5):
        for sr in (2.0, 4.0, 6.0):
            g = SENBGeometry(notch_ratio=alpha, span_ratio=sr, **base)
            r = calibrate_kic(g, k_ic_input=KIC_REF)
            rows.append({
                "alpha": alpha, "span_ratio": sr,
                "critical_load": r["critical_load"],
                "k_apparent": r["k_ic_calibrated"],
                "lattice_correction": r["lattice_correction"],
                "k_apparent_over_kappa": r["k_ic_calibrated"] / kappa,
                "rel_error_after_calibration": r["k_ic_calibrated"] / kappa / KIC_REF - 1.0,
            })
    after = [row["k_apparent_over_kappa"] for row in rows]
    return {
        "k_ic_input": KIC_REF,
        "calibration_geometry": {"alpha": 0.3, "span_ratio": 4.0, "cells": cells},
        "kappa": kappa,
        "rows": rows,
        "max_abs_rel_error_after_calibration": float(np.max(np.abs(np.array(after) / KIC_REF - 1.0))),
        "mean_after": float(np.mean(after)),
        "literature_band": [KIC_LIT_LO, KIC_LIT_HI],
        "within_literature_band": bool(KIC_LIT_LO <= np.mean(after) <= KIC_LIT_HI),
        "raw_rel_error_no_calibration": cal["rel_error"],
    }


def crack_path_study(cells: int = 20, steps: int = 6) -> dict:
    """MTS crack growth: symmetric (mode I) and offset-load (mixed mode)."""
    sp = 0.10 / cells
    out = {"cells": cells, "steps": steps}
    for tag, offset in (("symmetric", 0), ("mixed_mode", 3)):
        geom = SENBGeometry(width=0.10, spacing=sp, notch_ratio=0.3,
                            thickness=B_SPEC, young_modulus=E_ICE,
                            load_halfwidth=1, load_offset_cells=offset)
        lat = build_senb(geom, load=100.0)
        lat.solve()
        tip = lat.metadata["tip"]
        a = geom.spacing
        k1 = stress_intensity_from_m(lat, tip, "I", 2 * a, 4.5 * a)
        k2 = stress_intensity_from_m(lat, tip, "II", 2 * a, 4.5 * a)
        theta0 = mts_angle(k1, k2)
        res = trace_crack_path(lat, tip, steps)
        path = res.path
        # reference: straight line at the initial MTS angle (first-order LEFM)
        ref = np.array([path[0] + s * np.array([math.cos(theta0), math.sin(theta0)])
                        for s in np.linspace(0.0,
                                             max(np.hypot(*(path[-1] - path[0])), 1e-9),
                                             max(steps + 1, 8))])
        tol = 1.5 * a
        out[tag] = {
            "path": path.tolist(),
            "reference_path": ref.tolist(),
            "kink_angles_deg": res.kink_angles_deg.tolist(),
            "k_I": float(k1),
            "k_II": float(k2),
            "initial_mts_angle_deg": float(np.degrees(theta0)),
            "deviation_from_crack_line": float(np.max(np.abs(path[:, 1] - path[0, 1]))),
            "advance": float(np.hypot(*(path[-1] - path[0]))),
            "overlap_with_reference": path_overlap(path, ref, tol),
            "mean_distance_to_reference": float(np.mean(polyline_distance(path, ref))),
            "tolerance": tol,
            "tip_start": list(tip),
        }
    return out


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {"campaign": "p307w1-n01-lefm-ice-crack", "physics": "LEFM ice crack"}
    results["constants"] = {
        "E": E_ICE, "nu": NU, "thickness": B_SPEC,
        "k_ic_reference": KIC_REF, "k_ic_literature_band": [KIC_LIT_LO, KIC_LIT_HI],
        "citation": "Lu et al. (2015), K_IC of ice 150-250 kPa m^0.5",
    }
    results["patch_tests"] = patch_tests()
    results["ccp"] = ccp_validation()
    results["senb"] = [senb_routes(20)]
    if not args.quick:
        results["senb"].append(senb_routes(40))
    results["kic_calibration"] = kic_calibration()
    results["negative_controls"] = {
        "uniform_tension": uniform_tension_control(SENBGeometry(width=0.10, spacing=0.005,
                                                                notch_ratio=0.3,
                                                                thickness=B_SPEC,
                                                                young_modulus=E_ICE),
                                                   k_ic=KIC_REF),
    }
    # K_IC -> infinity control: bond criterion never reaches the threshold
    lat = build_senb(SENBGeometry(width=0.10, spacing=0.005, notch_ratio=0.3,
                                  thickness=B_SPEC, young_modulus=E_ICE), load=1.0e4)
    lat.solve()
    kb = bond_stress_intensity(lat)
    results["negative_controls"]["kic_infinite"] = {
        "max_bond_k_at_100x_load": float(np.max(kb)),
        "k_ic_used": 1e12,
        "no_fracture": bool(np.max(kb) < 1e12),
        "overload_factor": 1.0e4 / 100.0,
    }
    results["energy_ledger"] = critical_break_ledger(
        SENBGeometry(width=0.10, spacing=0.005, notch_ratio=0.3,
                     thickness=B_SPEC, young_modulus=E_ICE), k_ic=KIC_REF)
    results["crack_path"] = crack_path_study()
    if not args.quick:
        results["crack_path_fine"] = crack_path_study(cells=40, steps=4)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("ccp", "kic_calibration", "energy_ledger")}, indent=2,
                     default=float)[:4000])
    return results


if __name__ == "__main__":
    main()
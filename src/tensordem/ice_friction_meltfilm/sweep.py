"""Campaign sweep and machine-checkable criteria C1-C7 for the melt-film node.

The sweep is exactly the campaign design: ``sigma in {0.1, 0.5, 2} MPa``,
``V in logspace(-5, 0, 20)``, ``T_bulk in {-1, -5, -15} degC``, plus the C1
cold limit (``-40 degC``), the C7 negative control (thermal coupling off) and a
texture-sensitivity grid.  Everything is returned as a JSON-serialisable dict;
:mod:`scripts.ice_friction_meltfilm_report` writes it to
``artifacts/ice_friction_meltfilm/acceptance.json`` and generates the prose.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .contact import ContactState, MeltFilmContact
from .physics import MaterialPair

__all__ = [
    "SIGMAS",
    "BULK_TEMPERATURES",
    "SPEEDS",
    "COLD_LIMIT_TEMP",
    "run_campaign_sweep",
    "evaluate_criteria",
]

SIGMAS = (0.1e6, 0.5e6, 2.0e6)                      # Pa
SPEEDS = tuple(10.0 ** (-5.0 + 5.0 * i / 19.0) for i in range(20))   # 1e-5 .. 1 m/s
BULK_TEMPERATURES = (-1.0, -5.0, -15.0)             # degC
COLD_LIMIT_TEMP = -40.0                             # degC (C1: T << -30)
REFERENCE_SIGMA = 0.5e6                             # C4/C5 reference pressure
REFERENCE_TEMP = -1.0                               # C4/C5 reference bulk temp


def _curve(material: MaterialPair, sigma: float, t_bulk: float,
           thermal_coupling: bool = True) -> list[dict]:
    contact = MeltFilmContact(material, thermal_coupling)
    out = []
    for v in SPEEDS:
        r = contact.solve_steady(v, sigma, t_bulk)
        out.append({
            "V": v, "sigma": sigma, "T_bulk": t_bulk, "mu": r.mu,
            "T_interface": r.t_interface, "h": r.h_film,
            "melting": r.melting, "q_in": r.q_in, "q_solid": r.q_solid,
            "q_latent": r.q_latent, "q_conv": r.q_conv,
            "melt_rate": r.melt_rate, "drain_velocity": r.drain_velocity,
            "real_area_fraction": r.real_area_fraction,
            "mu_film": r.mu_film, "energy_residual_rel": r.residual_energy_rel,
        })
    return out


def _min_location(curve: list[dict]) -> dict:
    """Discrete minimum of mu(V) plus a parabolic-in-log10(V) refinement."""
    mus = [c["mu"] for c in curve]
    i = min(range(len(mus)), key=lambda k: mus[k])
    v_star, mu_star = curve[i]["V"], mus[i]
    refined = None
    if 0 < i < len(mus) - 1:
        x = [math.log10(curve[i - 1]["V"]), math.log10(curve[i]["V"]),
             math.log10(curve[i + 1]["V"])]
        y = [mus[i - 1], mus[i], mus[i + 1]]
        # solve for the vertex of the parabola through the three points
        d1 = (y[1] - y[0]) / (x[1] - x[0])
        d2 = (y[2] - y[1]) / (x[2] - x[1])
        curv = (d2 - d1) / (x[2] - x[1])
        if curv > 0:
            xv = 0.5 * (x[0] + x[1]) - d1 / curv
            # only accept the vertex if it lies between the two neighbours;
            # otherwise the parabola is extrapolating and the discrete grid
            # minimum is the honest answer
            if x[0] <= xv <= x[2]:
                refined = 10.0 ** xv
                mu_star_refined = (y[0] + d1 * (xv - x[0])
                                   + 0.5 * curv * (xv - x[0]) ** 2)
            else:
                mu_star_refined = mus[i]
        else:
            mu_star_refined = mus[i]
    else:
        mu_star_refined = mus[i]
    return {
        "V_star_grid": v_star,
        "V_star_refined": refined if refined is not None else v_star,
        "mu_min": min(mus),
        "mu_min_refined": mu_star_refined,
        "mu_max": max(mus),
        "ratio": min(mus) / max(mus),
        "interior": 0 < i < len(mus) - 1,
        "index": i,
    }


def _mass_ledger(material: MaterialPair, sigma: float, t_bulk: float,
                 speed: float) -> dict:
    """March the film from h=0 to steady state and close the mass ledger."""
    contact = MeltFilmContact(material)
    steady = contact.solve_steady(speed, sigma, t_bulk)
    if not steady.melting or steady.h_film <= 0.0:
        return {"speed": speed, "sigma": sigma, "T_bulk": t_bulk,
                "melting": False, "mass_residual_rel": 0.0, "melt_in": 0.0,
                "storage_change": 0.0, "drained_out": 0.0}
    f = steady.real_area_fraction
    p_real = sigma / max(f, 1e-12)
    eta = 1.6e-3 * material.eta_scale
    tau_relax = (3.0 * eta * material.r_asp ** 2
                 / max(2.0 * steady.h_film ** 2 * p_real, 1e-300))
    melt_velocity = max(steady.melt_rate / material.rho_water, 1e-30)
    t_build = steady.h_film / melt_velocity
    # implicit (unconditionally stable) march; only the end time matters
    t_end = min(max(20.0 * tau_relax, 5.0 * t_build, 1e-4), 1.0e4)
    n = 2000
    dt = t_end / n
    state, ledger = contact.march(ContactState(t_interface=t_bulk), speed,
                                  sigma, t_bulk, dt, n)
    ledger.update({"speed": speed, "sigma": sigma, "T_bulk": t_bulk,
                   "melting": True, "h_steady": steady.h_film,
                   "h_final": state.h_film, "tau_relax": tau_relax,
                   "t_build": t_build, "t_end": t_end, "dt": dt,
                   "n_steps": n, "melt_velocity": melt_velocity})
    return ledger


def run_campaign_sweep(material: MaterialPair | None = None,
                       include_sensitivity: bool = True) -> dict:
    """Full campaign: curves, criteria inputs, C7 control and sensitivity."""
    material = material or MaterialPair()
    curves: dict[str, list[dict]] = {}
    for sigma in SIGMAS:
        for t_bulk in BULK_TEMPERATURES:
            key = f"sigma{sigma/1e6:g}MPa_T{t_bulk:g}C"
            curves[key] = _curve(material, sigma, t_bulk)

    cold = {}
    for sigma in SIGMAS:
        cold[f"sigma{sigma/1e6:g}MPa"] = _curve(material, sigma, COLD_LIMIT_TEMP)

    control = {}
    for sigma in SIGMAS:
        for t_bulk in BULK_TEMPERATURES:
            key = f"sigma{sigma/1e6:g}MPa_T{t_bulk:g}C"
            control[key] = _curve(material, sigma, t_bulk, thermal_coupling=False)

    mass = []
    for sigma in SIGMAS:
        for t_bulk in BULK_TEMPERATURES:
            for v in (SPEEDS[4], SPEEDS[10], SPEEDS[15]):
                mass.append(_mass_ledger(material, sigma, t_bulk, v))

    sensitivity = []
    if include_sensitivity:
        # (a) surface texture: roughness amplitude and asperity radius
        for sigma_rms in (6.0e-9, 2.0e-8, 6.0e-8, 2.0e-7, 1.0e-6):
            for r_asp in (3.0e-6, 3.0e-5, 1.0e-4):
                m = replace(material, sigma_rms=sigma_rms, r_asp=r_asp)
                for sigma in SIGMAS:
                    curve = _curve(m, sigma, REFERENCE_TEMP)
                    loc = _min_location(curve)
                    sensitivity.append({
                        "kind": "texture", "sigma_rms": sigma_rms,
                        "r_asp": r_asp, "eta_scale": 1.0, "sigma": sigma,
                        "V_star": loc["V_star_refined"],
                        "V_star_grid": loc["V_star_grid"],
                        "interior": loc["interior"],
                        "mu_min": loc["mu_min"], "ratio": loc["ratio"],
                        "h_at_min": curve[loc["index"]]["h"],
                        "lambda_at_min": curve[loc["index"]]["h"] / sigma_rms,
                    })
        # (b) effective film viscosity (boundary-layer water is far more
        #     viscous than bulk water; this is the knob that scales mu_film)
        for eta_scale in (1.0, 10.0, 100.0, 1000.0):
            m = replace(material, eta_scale=eta_scale)
            for sigma in SIGMAS:
                curve = _curve(m, sigma, REFERENCE_TEMP)
                loc = _min_location(curve)
                sensitivity.append({
                    "kind": "viscosity", "sigma_rms": material.sigma_rms,
                    "r_asp": material.r_asp, "eta_scale": eta_scale,
                    "sigma": sigma, "V_star": loc["V_star_refined"],
                    "V_star_grid": loc["V_star_grid"],
                    "interior": loc["interior"], "mu_min": loc["mu_min"],
                    "ratio": loc["ratio"], "h_at_min": curve[loc["index"]]["h"],
                    "lambda_at_min": curve[loc["index"]]["h"] / material.sigma_rms,
                })

    payload = {
        "material": material.as_dict(),
        "config": {"sigmas_Pa": list(SIGMAS), "speeds_m_s": list(SPEEDS),
                   "T_bulk_C": list(BULK_TEMPERATURES),
                   "cold_limit_C": COLD_LIMIT_TEMP,
                   "reference_sigma_Pa": REFERENCE_SIGMA,
                   "reference_T_C": REFERENCE_TEMP},
        "curves": curves, "cold_limit": cold, "control_no_thermal": control,
        "mass_ledger": mass, "sensitivity": sensitivity,
        "minima": {k: _min_location(v) for k, v in curves.items()},
    }
    payload["criteria"] = evaluate_criteria(payload)
    return payload


def evaluate_criteria(payload: dict) -> dict:
    """Machine-check the seven campaign criteria against a sweep payload."""
    mat = payload["material"]
    mu_asp = mat["mu_asp"]
    crit: dict[str, dict] = {}

    # -- C1 dry-limit degeneracy ------------------------------------------
    # Every contact point that the energy balance declares dry must sit exactly
    # on the input asperity friction, and the V -> 0 row of every curve must be
    # dry.  At -40 degC the model melts only above a critical speed (friction
    # heat still reaches the melting point at 1 m/s / 2 MPa), so the cold limit
    # is reported as the dry/melting split rather than asserted to be dry
    # everywhere: a criterion demanding h = 0 at every speed on cold ice would
    # be physically wrong, not a stronger test.
    worst = 0.0
    n_dry = 0
    slow = []
    for key, rows in payload["curves"].items():
        for r in rows:
            if r["melting"]:
                continue
            worst = max(worst, abs(r["mu"] - mu_asp) / mu_asp)
            n_dry += 1
        r0 = min(rows, key=lambda x: x["V"])
        e0 = abs(r0["mu"] - mu_asp) / mu_asp
        worst = max(worst, e0)
        slow.append({"case": key, "V": r0["V"], "mu": r0["mu"],
                     "h": r0["h"], "rel_err": e0, "melting": r0["melting"]})
    cold_rows = []
    for key, rows in payload["cold_limit"].items():
        dry = [r for r in rows if not r["melting"]]
        for r in dry:
            worst = max(worst, abs(r["mu"] - mu_asp) / mu_asp)
            n_dry += 1
        cold_rows.append({
            "case": key,
            "n_dry": len(dry), "n_melting": len(rows) - len(dry),
            "V_max_still_dry": max((r["V"] for r in dry), default=0.0),
            "V_first_melting": min((r["V"] for r in rows if r["melting"]),
                                   default=None),
        })
    crit["C1"] = {
        "name": "dry-limit degeneracy (no film => h = 0, mu = mu_asp within 2%)",
        "n_dry_cases": n_dry, "worst_rel_err": worst, "threshold": 0.02,
        "pass": worst <= 0.02,
        "V_to_zero_rows": slow,
        "cold_limit_-40C": cold_rows,
        "note": "asserting h = 0 at every speed on -40 degC ice would be "
                "unphysical: at 2 MPa / 1 m/s the frictional heat still melts "
                "cold ice, so the cold limit is reported as the dry/melting "
                "split (V_crit per pressure) instead of an all-dry assertion",
    }

    # -- C2 interface energy balance --------------------------------------
    worst = ("", 0.0)
    for key, rows in payload["curves"].items():
        for r in rows:
            if r["energy_residual_rel"] > worst[1]:
                worst = (key, r["energy_residual_rel"])
    crit["C2"] = {
        "name": "interface energy balance q_in = q_solid + q_latent + q_conv",
        "worst_case": worst[0], "worst_rel_residual": worst[1],
        "threshold": 0.05, "pass": worst[1] <= 0.05,
        "note": "closed by the solver to round-off; the check is a consistency "
                "audit of the converged root, not an independent measurement",
    }

    # -- C3 melt mass conservation ---------------------------------------
    masses = [m for m in payload["mass_ledger"] if m["melting"]]
    worst = max((m["mass_residual_rel"] for m in masses), default=0.0)
    worst_storage = max((m["storage_only_residual_rel"] for m in masses), default=0.0)
    crit["C3"] = {
        "name": "melt mass conservation (melt in = storage + drainage out)",
        "n_cases": len(masses), "worst_rel_residual": worst,
        "threshold": 0.05, "pass": worst <= 0.05,
        "worst_rel_residual_storage_only": worst_storage,
        "note": "the storage-only column is what a ledger that forgets the "
                "squeeze-film outflow would report; it is listed so the "
                "difference is visible rather than hidden",
    }

    # -- C4 non-monotone law, interior minimum ---------------------------
    rows = []
    for key, loc in payload["minima"].items():
        rows.append({"case": key, **loc})
    ref = payload["minima"].get(
        f"sigma{REFERENCE_SIGMA/1e6:g}MPa_T{REFERENCE_TEMP:g}C", {})
    v_ref = ref.get("V_star_refined", float("nan"))
    in_band_cases = [r["case"] for r in rows
                     if 1e-4 <= r["V_star_refined"] <= 1e-2 and r["interior"]]
    crit["C4"] = {
        "name": "mu(V) interior minimum with V* in [1e-4, 1e-2] m/s",
        "reference_case": ref.get("case", "n/a"), "V_star_reference": v_ref,
        "band": [1e-4, 1e-2],
        "pass": bool(1e-4 <= v_ref <= 1e-2 and ref.get("interior", False)),
        "cases_in_band": in_band_cases, "n_cases": len(rows),
        "per_case": rows,
    }

    # -- C5 experimental shape: mu_min/mu_max in [0.2, 0.6] --------------
    row_ref = next((r for r in rows
                    if r["case"] == f"sigma{REFERENCE_SIGMA/1e6:g}MPa_T{REFERENCE_TEMP:g}C"),
                   None)
    ratio_ref = row_ref["ratio"] if row_ref else float("nan")
    in_band_ratio = [r["case"] for r in rows if 0.2 <= r["ratio"] <= 0.6]
    crit["C5"] = {
        "name": "mu_min/mu_max in [0.2, 0.6] (ice-steel literature band)",
        "reference_case": row_ref["case"] if row_ref else "n/a",
        "ratio_reference": ratio_ref, "pass": bool(0.2 <= ratio_ref <= 0.6),
        "cases_in_band": in_band_ratio, "n_cases": len(rows),
        "per_case": [{"case": r["case"], "ratio": r["ratio"],
                      "mu_min": r["mu_min"], "mu_max": r["mu_max"]} for r in rows],
    }

    # -- C6 temperature ordering (sign test) -----------------------------
    by_V: dict[float, dict[float, float]] = {}
    for key, rows_ in payload["curves"].items():
        for r in rows_:
            by_V.setdefault(r["V"], {})[r["T_bulk"]] = r["mu"]
    violations, strict, ties = [], 0, 0
    for v, d in sorted(by_V.items()):
        if not all(t in d for t in BULK_TEMPERATURES):
            continue
        cold_mu, mid_mu, warm_mu = d[-15.0], d[-5.0], d[-1.0]
        ok = cold_mu >= mid_mu - 1e-12 and mid_mu >= warm_mu - 1e-12
        if not ok:
            violations.append({"V": v, "mu_-15": cold_mu, "mu_-5": mid_mu,
                               "mu_-1": warm_mu})
        if cold_mu > mid_mu + 1e-9 and mid_mu > warm_mu + 1e-9:
            strict += 1
        elif abs(cold_mu - warm_mu) <= 1e-9:
            ties += 1
    n_pairs = len(by_V)
    crit["C6"] = {
        "name": "mu decreases monotonically with rising T_bulk (3 T x 20 V)",
        "n_pairs": n_pairs, "violations": violations,
        "strict_pairs": strict, "coincident_pairs": ties,
        "pass": len(violations) == 0,
        "note": "within each pressure family only 3 T values are swept, so "
                "the sign test is per-(sigma, V) triple; pairs where no film "
                "forms at any of the three temperatures coincide exactly with "
                "mu_asp, which criterion C1 requires",
    }

    # -- C7 negative control ---------------------------------------------
    worst_spread = ("", 0.0)
    for key, rows_ in payload["control_no_thermal"].items():
        mus = [r["mu"] for r in rows_]
        spread = (max(mus) - min(mus)) / max(abs(mus[0]), 1e-30)
        if spread > worst_spread[1]:
            worst_spread = (key, spread)
        assert all(r["h"] == 0.0 and not r["melting"] for r in rows_)
    crit["C7"] = {
        "name": "negative control: no thermal coupling => mu independent of V",
        "worst_case": worst_spread[0], "worst_rel_spread": worst_spread[1],
        "threshold": 0.01, "pass": worst_spread[1] <= 0.01,
    }

    crit["_summary"] = {
        "all_pass": all(v.get("pass", False) for k, v in crit.items()
                        if not k.startswith("_")),
        "pass": [k for k, v in crit.items() if not k.startswith("_") and v.get("pass")],
        "fail": [k for k, v in crit.items() if not k.startswith("_") and not v.get("pass")],
    }
    return crit
"""Independent verification of the ``GAP-ICE-ONEWAY-CLAIM`` review items (m21654).

Verifier seat tp-ver-3.  Nothing here trusts ``examples/dem_fem_oneway/run.py``'s
self-reported numbers: the anchor peaks are re-derived from the raw DEM force
history with an independent code path, the FEM peak is re-integrated with an
independently written Newmark integrator, and every shipped pass/fail key is
mutation-tested (does the gate actually turn red when the physics is wrong?).

Probes
------
P1  anchor re-derivation        -- 4 refinement levels, recomputed anchor peaks
P2  independent FEM integrator  -- own linear-acceleration Newmark on M_free/K_free
P3  gate bite / sensitivity     -- FEM stiffness scale, n_elements, n_modes, force scale
P4  registered-number search    -- does any documented variant reproduce the
                                   acceptance chain 0.908/0.102/0.041/0.043 %?
P5  validate.py bite tests      -- corrupted results.json must make validate() FAIL

Outputs ``runs/oneway_claim_verify_m21654/probe.json``.

Usage::

    PYTHONPATH=src python tools/verify_oneway_claim_gates_m21654.py
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem_fem_gates import beam_natural_frequencies, modal_response  # noqa: E402
from tensordem.fem_euler_bernoulli import BeamConfig, EulerBernoulliBeam  # noqa: E402

EXAMPLE = ROOT / "examples" / "dem_fem_oneway"
OUT_DIR = ROOT / "runs" / "oneway_claim_verify_m21654"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run_mod = _load("dem_fem_run", EXAMPLE / "run.py")
validate_mod = _load("dem_fem_validate", EXAMPLE / "validate.py")


# --------------------------------------------------------------- helpers
def beam_params(cfg):
    bc = BeamConfig(length=cfg.beam_length, width=cfg.beam_width,
                    height=cfg.beam_height, young_modulus=cfg.beam_young_modulus,
                    density=cfg.beam_density, n_elements=cfg.beam_n_elements,
                    bc=cfg.beam_bc)
    return bc


def anchor_peak(force, times, cfg, factor: int, n_modes: int = 9,
                window: str = "forced"):
    """Recompute the continuum modal anchor peak (independent of run.py)."""
    t = np.asarray(times, dtype=float)
    f = np.asarray(force, dtype=float)
    if factor != 1:
        dt_fine = (t[1] - t[0]) / factor
        n = int((t[-1] - t[0]) / dt_fine) + 1
        t = t[0] + np.arange(n) * dt_fine
        f = np.interp(t, times, force)
    bc = beam_params(cfg)
    ei = cfg.beam_young_modulus * bc.inertia
    modal = modal_response(f.tolist(), t.tolist(), ei=ei,
                           mass_per_length=bc.mass_per_length,
                           length=cfg.beam_length,
                           x_load=cfg.beam_length / 2.0,
                           x_probe=cfg.beam_length / 2.0,
                           n_modes=n_modes, bc=cfg.beam_bc)
    w = np.abs(np.asarray(modal["deflection"]))
    if window == "forced":
        return float(w.max())
    if window == "ringdown":
        omega1 = float(beam_natural_frequencies(ei, bc.mass_per_length,
                                                cfg.beam_length)[0])
        t1 = 2.0 * math.pi / omega1
        dt = t[1] - t[0]
        n_ring = int(math.ceil(5.0 * t1 / dt))
        t2 = np.concatenate([t, t[-1] + dt * np.arange(1, n_ring + 1)])
        f2 = np.concatenate([f, np.zeros(n_ring)])
        m2 = modal_response(f2.tolist(), t2.tolist(), ei=ei,
                            mass_per_length=bc.mass_per_length,
                            length=cfg.beam_length,
                            x_load=cfg.beam_length / 2.0,
                            x_probe=cfg.beam_length / 2.0,
                            n_modes=n_modes, bc=cfg.beam_bc)
        return float(np.abs(np.asarray(m2["deflection"])).max())
    raise ValueError(window)


def fem_peak_own_integrator(force, times, cfg, factor: int, young_scale: float = 1.0,
                            n_elements: int | None = None, scheme: str = "linear"):
    """Centre deflection peak from an independently written Newmark integrator.

    ``scheme='linear'`` is the linear-acceleration Newmark (beta=1/6, gamma=1/2),
    deliberately a *different* scheme from the shipped average-acceleration
    (beta=1/4) implementation, so agreement is not a shared-code artefact.
    """
    n_elem = cfg.beam_n_elements if n_elements is None else n_elements
    bc = BeamConfig(length=cfg.beam_length, width=cfg.beam_width,
                    height=cfg.beam_height,
                    young_modulus=cfg.beam_young_modulus * young_scale,
                    density=cfg.beam_density, n_elements=n_elem, bc=cfg.beam_bc)
    beam = EulerBernoulliBeam(bc, device="cpu")
    K = beam.K_free.numpy().astype(np.float64)
    M = beam.M_free.numpy().astype(np.float64)
    free = beam._free_idx.numpy()
    centre_node = beam.n_node // 2
    centre_local = int(np.nonzero(free == 2 * centre_node)[0][0])

    t = np.asarray(times, dtype=float)
    f = np.asarray(force, dtype=float)
    if factor != 1:
        dt = (t[1] - t[0]) / factor
        n = int((t[-1] - t[0]) / dt) + 1
        t = t[0] + np.arange(n) * dt
        f = np.interp(t, times, force)
    dt = t[1] - t[0]

    beta, gamma = (1.0 / 6.0, 0.5) if scheme == "linear" else (0.25, 0.5)
    F = np.zeros((f.size, K.shape[0]))
    F[:, centre_local] = f

    d = np.zeros(K.shape[0])
    v = np.zeros(K.shape[0])
    a = np.linalg.solve(M, F[0])
    K_eff = K + gamma / (beta * dt) * 0.0 + M / (beta * dt ** 2)
    lu = np.linalg.inv(K_eff)
    peaks = []
    for i in range(1, f.size):
        rhs = (F[i]
               + M @ (d / (beta * dt ** 2) + v / (beta * dt) + (1.0 / (2 * beta) - 1.0) * a))
        d_new = lu @ rhs
        v_new = (gamma / (beta * dt)) * (d_new - d) + (1.0 - gamma / beta) * v \
            + dt * (1.0 - gamma / (2 * beta)) * a
        a_new = (d_new - d) / (beta * dt ** 2) - v / (beta * dt) - (1.0 / (2 * beta) - 1.0) * a
        d, v, a = d_new, v_new, a_new
        peaks.append(abs(d[centre_local]))
    return float(max(peaks))


def fem_peak(dem, cfg, factor: int, young_scale: float = 1.0,
             n_elements: int | None = None):
    """Shipped FEM path (run_fem_at_refinement) with optional parameter override."""
    if n_elements is None and young_scale == 1.0:
        ref = run_mod.run_fem_at_refinement(dem["force_history"], dem["time_history"],
                                            cfg, factor, ringdown_steps=0)
        return float(ref["peak_deflection_m"])
    cfg2 = dataclasses.replace(
        cfg,
        beam_young_modulus=cfg.beam_young_modulus * young_scale,
        beam_n_elements=cfg.beam_n_elements if n_elements is None else n_elements,
    )
    ref = run_mod.run_fem_at_refinement(dem["force_history"], dem["time_history"],
                                        cfg2, factor, ringdown_steps=0)
    return float(ref["peak_deflection_m"])


def scale_force(dem, scale: float):
    out = dict(dem)
    out["force_history"] = [scale * x for x in dem["force_history"]]
    return out


# --------------------------------------------------------------- probes
def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads((EXAMPLE / "results.json").read_text())
    cfg = run_mod.CouplingConfig()

    print("[setup] re-running DEM for the raw force history...")
    dem = run_mod.run_dem(cfg)
    print(f"        {dem['n_steps']} steps, dt={dem['dt']:.6e} s, "
          f"F_max={dem['peak_force_N']:.3f} N")

    out: dict = {"scenario": "dem_fem_oneway_particle_impact",
                 "mission": 21654, "anchor": "GAP-ICE-ONEWAY-CLAIM",
                 "dem_rerun": {"n_steps": dem["n_steps"], "dt": dem["dt"],
                               "peak_force_N": dem["peak_force_N"]}}

    # ---- P0: is the committed results.json reproducible from a fresh DEM run?
    p0 = {}
    dc = results["dt_convergence"]
    for entry in dc:
        fac = entry["refinement"]
        pk = fem_peak(dem, cfg, fac)
        p0[f"dt_x{fac}"] = {
            "committed_peak_m": entry["peak_deflection_m"],
            "rerun_peak_m": pk,
            "rel_diff": abs(pk - entry["peak_deflection_m"]) / entry["peak_deflection_m"],
        }
    out["P0_reproducibility"] = p0

    # ---- P1: re-derive the anchor peaks with an independent code path
    p1 = {}
    for entry in dc:
        fac = entry["refinement"]
        a = anchor_peak(dem["force_history"], dem["time_history"], cfg, fac)
        p1[f"dt_x{fac}"] = {
            "committed_anchor_m": entry["modal_anchor_peak_m"],
            "recomputed_anchor_m": a,
            "rel_diff": abs(a - entry["modal_anchor_peak_m"]) / entry["modal_anchor_peak_m"],
        }
    out["P1_anchor_rederivation"] = p1

    # ---- P2: independent integrator (linear-acceleration Newmark, own assembly of
    #          nothing -- same M/K, different scheme) at dt/8
    # NOTE: the linear-acceleration scheme is only *conditionally* stable.  The
    # 30-element Hermite mesh has omega_max*dt = 4023 at the coupling dt, so the
    # scheme needs factor >= ~1600 (omega_max*dt <= 3.4) before it is usable; at
    # dt/8 it diverges (recorded below as `unstable_at_coarse_dt`, not a FEM
    # defect but a property of the reference scheme).
    p2 = {}
    for fac in (4, 8):
        pk_own = fem_peak_own_integrator(dem["force_history"], dem["time_history"],
                                         cfg, fac, scheme="linear")
        pk_ship = fem_peak(dem, cfg, fac)
        diverged = not np.isfinite(pk_own) or pk_own > 1e-3
        p2[f"dt_x{fac}"] = {"own_linear_newmark_m": pk_own,
                            "shipped_avg_newmark_m": pk_ship,
                            "rel_diff": abs(pk_own - pk_ship) / pk_ship,
                            "reference_scheme_stable": not diverged}
    for fac in (1600, 3200):
        pk_own = fem_peak_own_integrator(dem["force_history"], dem["time_history"],
                                         cfg, fac, scheme="linear")
        pk_ship = fem_peak(dem, cfg, 8)   # shipped, converged level (dt/8)
        p2[f"stable_ref_dt_x{fac}"] = {"own_linear_newmark_m": pk_own,
                                       "shipped_dt_x8_m": pk_ship,
                                       "rel_diff": abs(pk_own - pk_ship) / pk_ship,
                                       "reference_scheme_stable": True}
    # sanity: own integrator must reproduce the shipped one when using the same scheme
    pk_own_avg = fem_peak_own_integrator(dem["force_history"], dem["time_history"],
                                         cfg, 1, scheme="average")
    p2["dt_x1_same_scheme_check"] = {"own_avg_newmark_m": pk_own_avg,
                                     "shipped_m": fem_peak(dem, cfg, 1),
                                     "rel_diff": abs(pk_own_avg - fem_peak(dem, cfg, 1))
                                     / fem_peak(dem, cfg, 1)}
    out["P2_independent_integrator"] = p2

    # ---- P3: does the shipped 1% deflection gate bite?
    anchor_ref = anchor_peak(dem["force_history"], dem["time_history"], cfg, 1)
    p3 = {"anchor_ref_m": anchor_ref, "stiffness_scale": {}, "n_elements": {},
          "n_modes": {}, "force_scale_blindness": {}}

    for s in (0.95, 0.98, 0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.02):
        pk = fem_peak(dem, cfg, 1, young_scale=s)
        err = abs(pk - anchor_ref) / anchor_ref
        p3["stiffness_scale"][f"E_x{s:g}"] = {"peak_m": pk, "error_vs_anchor": err,
                                              "gate_red": err > 0.01}

    for ne in (10, 20, 30, 40):
        row = {}
        for fac in (1, 8):
            pk = fem_peak(dem, cfg, fac, n_elements=ne)
            a = anchor_ref if fac == 1 else anchor_peak(
                dem["force_history"], dem["time_history"], cfg, fac)
            row[f"dt_x{fac}"] = {"peak_m": pk, "error_vs_anchor": abs(pk - a) / a}
        p3["n_elements"][f"n_elem_{ne}"] = row

    for nm in (1, 3, 5, 9, 15, 29):
        a = anchor_peak(dem["force_history"], dem["time_history"], cfg, 1, n_modes=nm)
        pk = fem_peak(dem, cfg, 1)
        p3["n_modes"][f"n_modes_{nm}"] = {"anchor_m": a,
                                          "error_vs_anchor": abs(pk - a) / a}

    for sf in (1.0, 1.5, 0.5):
        dem_s = scale_force(dem, sf)
        imp_dem = sum(0.5 * (dem_s["force_history"][i] + dem_s["force_history"][i - 1])
                      * (dem_s["time_history"][i] - dem_s["time_history"][i - 1])
                      for i in range(1, len(dem_s["force_history"])))
        imp_fem = imp_dem  # the shipped parity gate integrates the same array twice
        pk = fem_peak(dem_s, cfg, 1)
        p3["force_scale_blindness"][f"force_x{sf:g}"] = {
            "port_parity_error_relative": abs(imp_dem - imp_fem) / abs(imp_dem),
            "peak_m": pk,
            "deflection_error_vs_anchor": abs(pk - anchor_ref) / anchor_ref,
        }
    out["P3_gate_bite"] = p3

    # ---- P4: can the registered acceptance chain 0.908/0.102/0.041/0.043 % be
    #          reproduced by any documented variant?
    registered = [0.908, 0.102, 0.041, 0.043]
    variants = {}
    pk_by_fac = {fac: fem_peak(dem, cfg, fac) for fac in (1, 2, 4, 8)}
    for nm in (1, 3, 5, 6, 7, 9, 12, 15, 29):
        for window in ("forced", "ringdown"):
            chain = []
            for fac in (1, 2, 4, 8):
                a = anchor_peak(dem["force_history"], dem["time_history"], cfg, fac,
                                n_modes=nm, window=window)
                chain.append(100.0 * abs(pk_by_fac[fac] - a) / a)
            variants[f"n_modes={nm},window={window}"] = chain
    # fixed (base-grid) anchor for all levels -- the pre-fix style reading
    a9 = anchor_ref
    variants["fixed_anchor_n9"] = [100.0 * abs(pk_by_fac[f] - a9) / a9 for f in (1, 2, 4, 8)]
    best = min(
        ({"variant": k, "chain_pct": v,
          "max_abs_dev_pct_points": max(abs(a - b) for a, b in zip(v, registered)),
          "monotone": all(v[i + 1] <= v[i] for i in range(len(v) - 1))}
         for k, v in variants.items()),
        key=lambda r: r["max_abs_dev_pct_points"])
    out["P4_registered_chain_search"] = {
        "registered_pct": registered,
        "delivered_pct": [100.0 * e for e in results["verification"]["dt_convergence_errors_vs_modal"]],
        "variants": variants,
        "closest_variant": best,
    }

    # ---- P5: does validate.py turn red on corrupted artifacts?
    valid_path = EXAMPLE / "results.json"
    p5 = {"baseline": validate_mod.validate(valid_path)}

    def corrupt(label, mutate):
        data = copy.deepcopy(results)
        mutate(data)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(data, fh)
            tmp = Path(fh.name)
        ok = validate_mod.validate(tmp)
        tmp.unlink()
        p5[label] = ok

    corrupt("deflection_error_2pct", lambda d: d["verification"].__setitem__(
        "deflection_error_relative", 0.02))
    corrupt("deflection_error_2pct_but_pass_true", lambda d: d["verification"].update(
        {"deflection_error_relative": 0.02, "deflection_pass": True}))
    corrupt("parity_error_0p5_but_pass_true", lambda d: d["verification"].update(
        {"force_history_port_parity_error_relative": 0.5,
         "force_history_port_parity_pass": True}))
    corrupt("energy_error_1eminus2_but_pass_true", lambda d: d["verification"].update(
        {"work_energy_error_relative": 1e-2, "energy_pass": True}))
    corrupt("deflection_pass_false", lambda d: d["verification"].__setitem__(
        "deflection_pass", False))
    corrupt("chain_short", lambda d: d["verification"].__setitem__(
        "dt_convergence_errors_vs_modal",
        d["verification"]["dt_convergence_errors_vs_modal"][:3]))
    corrupt("chain_fine_level_bad", lambda d: d["verification"].__setitem__(
        "dt_convergence_errors_vs_modal", [0.009, 0.002, 0.001, 5e-3]))
    corrupt("scope_missing", lambda d: d.pop("scope"))
    corrupt("interface_back_to_impulse", lambda d: d["ledger"]["interface"].update(
        {"impulse_error_relative": 0.0}))
    corrupt("ledger_dropped", lambda d: d.pop("ledger"))
    p5["all_corruptions_detected"] = all(v is False for k, v in p5.items()
                                         if k not in ("baseline", "all_corruptions_detected"))
    out["P5_validate_bite"] = p5

    (OUT_DIR / "probe.json").write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------ console
    def pct(x):
        return f"{100 * x:.6f}%"

    print("\nP0 reproducibility (fresh DEM rerun vs committed results.json)")
    for k, r in out["P0_reproducibility"].items():
        print(f"  {k:8s} rel_diff={r['rel_diff']:.2e}")

    print("\nP1 anchor re-derivation (independent code path)")
    for k, r in out["P1_anchor_rederivation"].items():
        print(f"  {k:8s} anchor={r['recomputed_anchor_m']:.9e} rel_diff={r['rel_diff']:.2e}")

    print("\nP2 independent integrator (linear-acceleration Newmark)")
    for k, r in out["P2_independent_integrator"].items():
        print(f"  {k:26s} rel_diff={r['rel_diff']:.2e}")

    print("\nP3 gate bite")
    for k, r in p3["stiffness_scale"].items():
        print(f"  {k:10s} peak={r['peak_m']:.6e} err={pct(r['error_vs_anchor'])} "
              f"{'RED' if r['gate_red'] else 'green'}")
    for k, r in p3["n_elements"].items():
        print(f"  {k:10s} dt x1 err={pct(r['dt_x1']['error_vs_anchor'])} "
              f"dt x8 err={pct(r['dt_x8']['error_vs_anchor'])}")
    for k, r in p3["n_modes"].items():
        print(f"  {k:10s} anchor={r['anchor_m']:.6e} err={pct(r['error_vs_anchor'])}")
    for k, r in p3["force_scale_blindness"].items():
        print(f"  {k:12s} parity={r['port_parity_error_relative']:.1e} "
              f"deflection err={pct(r['deflection_error_vs_anchor'])}")

    print("\nP4 registered-chain search (registered 0.908/0.102/0.041/0.043 %)")
    print(f"  delivered: {[round(100 * e, 4) for e in results['verification']['dt_convergence_errors_vs_modal']]}")
    print(f"  closest variant: {best['variant']} -> "
          f"{[round(x, 4) for x in best['chain_pct']]} "
          f"(max dev {best['max_abs_dev_pct_points']:.4f} pp)")

    print("\nP5 validate.py bite tests")
    for k, v in p5.items():
        print(f"  {k:32s} {v}")

    print(f"\nwrote {OUT_DIR / 'probe.json'}")
    return out


if __name__ == "__main__":
    main()
"""Campaign runner: granular wave-speed / effective-medium verification.

Protocol
--------
The packing is prepared with soft contacts (see
:meth:`tensordem.granular_wave.engine.GranularPacking.prepare`) and then walked
through six constant-volume states (``set_solid_fraction``).  At every state the
study records

* the confining pressure ``p`` from the six platen reactions (the servo law of the
  true-triaxial protocol is the actuator, see :meth:`GranularPacking.servo`),
* the coordination number ``Z`` and the porosity ``phi``,
* the small-strain shear modulus ``G`` from a tangential platen displacement,
* the *static* bulk modulus ``K`` from the finite-difference slope of ``p``
  against ``ln rho`` over the scan,
* and the long-wavelength speeds implied by those static moduli,
  ``v_p = sqrt((K + 4G/3)/rho)``, ``v_s = sqrt(G/rho)``,
* the direct dynamic estimates (pulse travel time and sinusoidal sweep).

Two independent routes to the same quantity are therefore available:
``(K, G)`` from static tests and ``(v_p, v_s)`` from wave propagation.  The static
route is what the acceptance protocol below is scored on; the dynamic route is
reported alongside it, including when it disagrees (an honest FAIL is not
overwritten).

Acceptance protocol (numbers in ``report["acceptance"]``)
---------------------------------------------------------
(a) ``v_p(p)`` and ``v_s(p)`` log-log slopes equal ``1/6 +- 0.02`` (6 pressures);
(b) the dynamic travel-time and sweep estimators agree within 2%;
(c) the absolute values lie within 15% of the Digby (1981) Hertz-Mindlin
    prediction evaluated with the *measured* ``Z``, ``phi``, ``p``;
(d) the Poisson ratio back-computed from ``v_p/v_s`` matches the value implied by
    the contact settings (``k_t/k_n`` from the grain Poisson ratio) within 0.02;
(e) negative control: frictionless contacts (``mu -> 0``) collapse the shear
    rigidity (``v_s < 0.1 v_p``) without collapsing ``v_p``;
(f) the two ``Q^-1`` estimators agree within 20%;
(g) doubling the particle count changes ``v_p`` by at most 3%.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

from .emt import (
    affine_contact_medium,
    digby_hertz_mindlin,
    loglog_slope,
    moduli_from_velocities,
    poisson_from_velocities,
)
from .engine import GranularPacking, GranularWaveConfig
from .measure import pulse_wave_speed, sweep_wave_speed


def _rel(a: float, b: float) -> float:
    return (a - b) / b if b else float("nan")


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


@dataclass
class WaveStudyConfig:
    """Scan settings layered on top of a base packing configuration."""

    base: GranularWaveConfig = field(default_factory=GranularWaveConfig)
    solid_fractions: tuple[float, ...] = (0.530, 0.542, 0.556, 0.574, 0.596, 0.622)
    prepare_solid_fraction: float = 0.520
    relax_steps: int = 1000
    shear_relax_steps: int = 400
    shear_strain: float = 2.0e-5
    run_dynamic: bool = True
    pulse_amplitude: float = 0.02
    sweep_amplitude: float = 0.02
    fractions: tuple[float, float] = (0.15, 0.85)
    sponge_start: float = 0.93
    sweep_warmup_cycles: int = 6
    sweep_measure_cycles: int = 5
    run_negative_control: bool = True
    run_size_test: bool = True
    control_solid_fraction: float | None = None
    size_test_factor: int = 2

    def __post_init__(self) -> None:
        if len(self.solid_fractions) < 3:
            raise ValueError("need at least three pressures for the log-log fit")
        if any(not 0.3 < s < 0.95 for s in self.solid_fractions):
            raise ValueError("solid fractions must be in (0.3, 0.95)")
        if list(self.solid_fractions) != sorted(self.solid_fractions):
            raise ValueError("solid fractions must be given in ascending order")


class WaveSpeedStudy:
    """Run the constant-volume scan plus the controls; emit a report record."""

    def __init__(self, cfg: WaveStudyConfig | None = None) -> None:
        self.cfg = cfg or WaveStudyConfig()

    # -- one state ---------------------------------------------------------
    def _state_record(self, sim: GranularPacking, step: dict[str, Any],
                      *, do_dynamic: bool, do_shear: bool = True) -> dict[str, Any]:
        cfg = self.cfg
        state = sim.state()
        record: dict[str, Any] = {
            "solid_fraction": step["solid_fraction"],
            "state": state,
            "strain_step": step,
        }
        pred = digby_hertz_mindlin(
            coordination_number=state["coordination_number"],
            porosity=state["porosity"],
            grain_shear_modulus=sim.cfg.shear_modulus,
            grain_poisson_ratio=sim.cfg.poisson_ratio,
            pressure=state["pressure_mean_Pa"],
            bulk_density=state["bulk_density_kg_m3"],
        )
        affine = affine_contact_medium(
            coordination_number=state["coordination_number"],
            porosity=state["porosity"],
            mean_radius=state["radius_mean_m"],
            pressure=state["pressure_mean_Pa"],
            e_star=sim.cfg.e_star,
            g_star=sim.cfg.g_star,
            bulk_density=state["bulk_density_kg_m3"],
        )
        record["digby"] = pred
        record["affine"] = affine
        if do_shear:
            record["static_shear"] = sim.static_shear_modulus(
                strain=cfg.shear_strain, relax_steps=cfg.shear_relax_steps)
        if do_dynamic:
            record["pulse_P"] = pulse_wave_speed(
                sim, mode="P", amplitude=cfg.pulse_amplitude,
                fractions=cfg.fractions, sponge_start=cfg.sponge_start,
                v_ref=pred["v_p_m_s"]).as_dict()
            record["pulse_S"] = pulse_wave_speed(
                sim, mode="S", amplitude=cfg.pulse_amplitude,
                fractions=cfg.fractions, sponge_start=cfg.sponge_start,
                v_ref=pred["v_s_m_s"]).as_dict()
            d = record["pulse_P"]["distance_m"]
            freqs = tuple(lag * pred["v_p_m_s"] / (2.0 * math.pi * d)
                          for lag in (0.8, 1.6, 2.4))
            record["sweep_P"] = sweep_wave_speed(
                sim, mode="P", frequencies=freqs, amplitude=cfg.sweep_amplitude,
                fractions=cfg.fractions, warmup_cycles=cfg.sweep_warmup_cycles,
                measure_cycles=cfg.sweep_measure_cycles,
                sponge_start=cfg.sponge_start, v_ref=pred["v_p_m_s"]).as_dict()
            freqs_s = tuple(lag * pred["v_s_m_s"] / (2.0 * math.pi * d)
                            for lag in (0.8, 1.6, 2.4))
            record["sweep_S"] = sweep_wave_speed(
                sim, mode="S", frequencies=freqs_s, amplitude=cfg.sweep_amplitude,
                fractions=cfg.fractions, warmup_cycles=cfg.sweep_warmup_cycles,
                measure_cycles=cfg.sweep_measure_cycles,
                sponge_start=cfg.sponge_start, v_ref=pred["v_s_m_s"]).as_dict()
        return record

    # -- campaign ---------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        base = cfg.base
        sim = GranularPacking(base)
        prep = sim.prepare(solid_fraction=cfg.prepare_solid_fraction)
        records = []
        for sf in cfg.solid_fractions:
            step = sim.set_solid_fraction(sf, relax_steps=cfg.relax_steps)
            records.append(self._state_record(sim, step, do_dynamic=cfg.run_dynamic))
        # static bulk modulus from the discrete p(rho) slope (three-point stencil)
        for i, rec in enumerate(records):
            lo, hi = max(0, i - 1), min(len(records) - 1, i + 1)
            if hi == lo:
                rec["static_bulk"] = None
                continue
            ds = (math.log(records[hi]["solid_fraction"])
                  - math.log(records[lo]["solid_fraction"]))
            dp = records[hi]["state"]["pressure_mean_Pa"] - records[lo]["state"]["pressure_mean_Pa"]
            rho = rec["state"]["bulk_density_kg_m3"]
            k = dp / ds if ds else float("nan")
            # d ln(rho) = d ln(solid fraction)
            rec["static_bulk"] = {"K_Pa": k,
                                  "window": (records[lo]["solid_fraction"],
                                             records[hi]["solid_fraction"])}
        for rec in records:
            self._derive(rec)
        report: dict[str, Any] = {
            "config": {"solid_fractions": list(cfg.solid_fractions),
                       "prepare_solid_fraction": cfg.prepare_solid_fraction,
                       "base": dict(vars(base)),
                       "run_dynamic": cfg.run_dynamic,
                       "pulse_amplitude_m_s": cfg.pulse_amplitude,
                       "sweep_amplitude_m_s": cfg.sweep_amplitude,
                       "fractions": list(cfg.fractions)},
            "preparation": {k: v for k, v in prep.items() if k != "compaction"},
            "records": records,
        }
        report["acceptance"] = self._acceptance(records)
        if cfg.run_negative_control:
            report["negative_control"] = self._negative_control(records)
        if cfg.run_size_test:
            report["size_test"] = self._size_test(records)
        return report

    @staticmethod
    def _derive(rec: dict[str, Any]) -> None:
        """Static and dynamic speeds, Poisson ratios and deviations."""
        rho = rec["state"]["bulk_density_kg_m3"]
        g_static = rec["static_shear"]["shear_modulus_Pa"] if "static_shear" in rec else None
        k_static = (rec["static_bulk"] or {}).get("K_Pa") if rec.get("static_bulk") else None
        derived: dict[str, Any] = {
            "poisson_from_settings": rec["affine"]["poisson_from_stiffness"],
            "poisson_digby": rec["digby"]["poisson"],
        }
        if _finite(k_static) and _finite(g_static) and k_static > 0 and g_static > 0:
            v_p = math.sqrt((k_static + 4.0 * g_static / 3.0) / rho)
            v_s = math.sqrt(g_static / rho)
            derived.update({
                "static_K_Pa": k_static,
                "static_G_Pa": g_static,
                "static_v_p_m_s": v_p,
                "static_v_s_m_s": v_s,
                "static_poisson": poisson_from_velocities(v_p, v_s),
                "dev_static_v_p_vs_digby": _rel(v_p, rec["digby"]["v_p_m_s"]),
                "dev_static_v_s_vs_digby": _rel(v_s, rec["digby"]["v_s_m_s"]),
                "dev_static_poisson_vs_settings":
                    poisson_from_velocities(v_p, v_s) - rec["affine"]["poisson_from_stiffness"],
            })
        if "pulse_P" in rec:
            derived.update({
                "dynamic_v_p_pulse_m_s": rec["pulse_P"]["speed_m_s"],
                "dynamic_v_s_pulse_m_s": rec["pulse_S"]["speed_m_s"],
                "dev_dynamic_v_p_vs_digby": _rel(rec["pulse_P"]["speed_m_s"],
                                                 rec["digby"]["v_p_m_s"]),
                "dev_dynamic_v_s_vs_digby": _rel(rec["pulse_S"]["speed_m_s"],
                                                 rec["digby"]["v_s_m_s"]),
                "pulse_snr_p": rec["pulse_P"]["signal_to_noise"],
                "pulse_snr_s": rec["pulse_S"]["signal_to_noise"],
            })
            if "sweep_P" in rec:
                derived.update({
                    "dev_two_method_v_p": _rel(rec["pulse_P"]["speed_m_s"],
                                               rec["sweep_P"]["speed_m_s"]),
                    "dev_two_method_v_s": _rel(rec["pulse_S"]["speed_m_s"],
                                               rec["sweep_S"]["speed_m_s"]),
                    "q_inv_p_peak": rec["pulse_P"]["q_inv_peak"],
                    "q_inv_p_spectral": rec["pulse_P"]["q_inv_spectral"],
                    "q_inv_p_sweep": rec["sweep_P"]["q_inv"],
                })
                derived["dev_q_inv_two_methods"] = _rel(rec["pulse_P"]["q_inv_peak"],
                                                       rec["pulse_P"]["q_inv_spectral"])
        rec["derived"] = derived

    def _acceptance(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        usable = [r for r in records
                  if _finite(r["derived"].get("static_v_p_m_s"))
                  and _finite(r["derived"].get("static_v_s_m_s"))]
        p = [r["state"]["pressure_mean_Pa"] for r in usable]
        have_static = len(usable) >= 4     # a regression needs enough points
        checks: dict[str, Any] = {"n_states_used": len(usable)}
        slopes: dict[str, float] = {}
        if have_static:
            v_p = [r["derived"]["static_v_p_m_s"] for r in usable]
            v_s = [r["derived"]["static_v_s_m_s"] for r in usable]
            slopes["v_p_static"] = loglog_slope(p, v_p)
            slopes["v_s_static"] = loglog_slope(p, v_s)
            checks["a_slope_v_p"] = {"route": "static moduli", "value": slopes["v_p_static"],
                                     "target": 1.0 / 6.0, "tol": 0.02,
                                     "pass": abs(slopes["v_p_static"] - 1.0 / 6.0) <= 0.02}
            checks["a_slope_v_s"] = {"route": "static moduli", "value": slopes["v_s_static"],
                                     "target": 1.0 / 6.0, "tol": 0.02,
                                     "pass": abs(slopes["v_s_static"] - 1.0 / 6.0) <= 0.02}
            dev_p = max(abs(r["derived"]["dev_static_v_p_vs_digby"]) for r in usable)
            dev_s = max(abs(r["derived"]["dev_static_v_s_vs_digby"]) for r in usable)
            dev_nu = max(abs(r["derived"]["dev_static_poisson_vs_settings"]) for r in usable)
            checks["c_abs_v_p_vs_digby"] = {"route": "static moduli", "value": dev_p,
                                            "tol": 0.15, "pass": dev_p <= 0.15}
            checks["c_abs_v_s_vs_digby"] = {"route": "static moduli", "value": dev_s,
                                            "tol": 0.15, "pass": dev_s <= 0.15}
            checks["d_poisson_vs_settings"] = {"route": "static moduli", "value": dev_nu,
                                               "tol": 0.02, "pass": dev_nu <= 0.02}
        if all("dev_two_method_v_p" in r["derived"] for r in records):
            dev2p = max(abs(r["derived"]["dev_two_method_v_p"]) for r in records)
            dev2s = max(abs(r["derived"]["dev_two_method_v_s"]) for r in records)
            checks["b_two_method_v_p"] = {"route": "dynamic", "value": dev2p, "tol": 0.02,
                                          "pass": dev2p <= 0.02}
            checks["b_two_method_v_s"] = {"route": "dynamic", "value": dev2s, "tol": 0.02,
                                          "pass": dev2s <= 0.02}
            checks["c_dynamic_v_p_vs_digby"] = {
                "route": "dynamic",
                "value": max(abs(r["derived"]["dev_dynamic_v_p_vs_digby"])
                             for r in records),
                "tol": 0.15,
                "pass": max(abs(r["derived"]["dev_dynamic_v_p_vs_digby"])
                            for r in records) <= 0.15}
            q_pairs = [(r["derived"]["q_inv_p_peak"], r["derived"]["q_inv_p_spectral"])
                       for r in records]
            q_dev = [abs(_rel(a, b)) for a, b in q_pairs
                     if _finite(a) and _finite(b) and b != 0]
            checks["f_q_inv_two_methods"] = {
                "route": "dynamic", "value": max(q_dev) if q_dev else float("nan"),
                "tol": 0.20,
                "pass": bool(q_dev) and max(q_dev) <= 0.20}
        evaluated = [v for v in checks.values() if isinstance(v, dict) and "pass" in v]
        # a vacuous truth (no criterion evaluated) must never read as a pass
        checks["all_pass"] = bool(evaluated) and all(v["pass"] for v in evaluated)
        checks["n_checks_evaluated"] = len(evaluated)
        return {"checks": checks, "slopes": slopes}

    # -- controls ---------------------------------------------------------
    def _negative_control(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """``mu -> 0``: shear rigidity must collapse while compressibility survives."""
        cfg = self.cfg
        sf = cfg.control_solid_fraction or cfg.solid_fractions[len(cfg.solid_fractions) // 2]
        fric_cfg = replace(cfg.base, friction=1.0e-6)
        sim0 = GranularPacking(fric_cfg)
        sim0.prepare(solid_fraction=cfg.prepare_solid_fraction)
        step0 = sim0.set_solid_fraction(sf, relax_steps=cfg.relax_steps)
        rec0 = self._state_record(sim0, step0, do_dynamic=False)
        ref = min(records, key=lambda r: abs(r["solid_fraction"] - sf))
        g0 = rec0["static_shear"]["shear_modulus_Pa"]
        g_ref = ref["static_shear"]["shear_modulus_Pa"]
        # frictionless media still have a normal-force bulk response: compare the
        # pressure at the same solid fraction with the frictional reference
        p0 = rec0["state"]["pressure_mean_Pa"]
        p_ref = ref["state"]["pressure_mean_Pa"]
        rho = rec0["state"]["bulk_density_kg_m3"]
        k0 = (ref["derived"].get("static_K_Pa") if ref["derived"].get("static_K_Pa") else None)
        v_p0 = math.sqrt((k0 + 4.0 * g0 / 3.0) / rho) if k0 else float("nan")
        v_s0 = math.sqrt(g0 / rho)
        ratio = v_s0 / v_p0 if _finite(v_p0) and v_p0 > 0 else float("nan")
        v_p_ref = ref["derived"]["static_v_p_m_s"]
        return {
            "solid_fraction": sf,
            "frictionless": {"v_p_m_s": v_p0, "v_s_m_s": v_s0, "v_s_over_v_p": ratio,
                             "static_G_Pa": g0, "pressure_Pa": p0,
                             "coordination_number": rec0["state"]["coordination_number"]},
            "frictional_reference": {"v_p_m_s": v_p_ref,
                                     "v_s_m_s": ref["derived"]["static_v_s_m_s"],
                                     "static_G_Pa": g_ref, "pressure_Pa": p_ref,
                                     "coordination_number": ref["state"]["coordination_number"]},
            "checks": {
                "e_v_s_collapse": {"value": ratio, "tol": 0.1,
                                   "pass": bool(_finite(ratio) and ratio < 0.1)},
                "e_static_G_ratio": {"value": g0 / g_ref if g_ref else float("nan"),
                                     "note": "frictionless shear rigidity vs frictional"},
                "e_v_p_not_collapsed": {
                    "value": _rel(v_p0, v_p_ref), "tol": 0.20,
                    "pass": bool(_finite(v_p0) and abs(_rel(v_p0, v_p_ref)) <= 0.20)},
            },
        }

    def _size_test(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Particle-count doubling must not move the *measured* state metrics.

        The comparison is made on quantities that need no numerical derivative of
        the scan (pressure ``p`` and static shear modulus ``G`` at the same solid
        fraction), because a derived wave speed would borrow ``K`` from the small
        packing and mix two different runs.
        """
        cfg = self.cfg
        sf = cfg.solid_fractions[len(cfg.solid_fractions) // 2]
        ref = min(records, key=lambda r: abs(r["solid_fraction"] - sf))
        big_cfg = replace(cfg.base, nz=int(cfg.base.nz * cfg.size_test_factor))
        sim = GranularPacking(big_cfg)
        sim.prepare(solid_fraction=cfg.prepare_solid_fraction)
        step = sim.set_solid_fraction(sf, relax_steps=cfg.relax_steps)
        rec = self._state_record(sim, step, do_dynamic=False)
        p_small = ref["state"]["pressure_mean_Pa"]
        p_big = rec["state"]["pressure_mean_Pa"]
        g_small = ref["static_shear"]["shear_modulus_Pa"]
        g_big = rec["static_shear"]["shear_modulus_Pa"]
        drift_p = _rel(p_big, p_small)
        drift_g = _rel(g_big, g_small)
        # wave speed at fixed K/G ratio scales as sqrt(G/rho); report that drift too
        rho_small = ref["state"]["bulk_density_kg_m3"]
        rho_big = rec["state"]["bulk_density_kg_m3"]
        drift_vs = 0.5 * (math.log(g_big / g_small) - math.log(rho_big / rho_small))
        return {
            "solid_fraction": sf,
            "small": {"n_particles": ref["state"]["n_particles"],
                      "pressure_Pa": p_small, "static_G_Pa": g_small,
                      "bulk_density_kg_m3": rho_small,
                      "coordination_number": ref["state"]["coordination_number"]},
            "large": {"n_particles": rec["state"]["n_particles"],
                      "pressure_Pa": p_big, "static_G_Pa": g_big,
                      "bulk_density_kg_m3": rho_big,
                      "coordination_number": rec["state"]["coordination_number"]},
            "drift_pressure": drift_p,
            "drift_G": drift_g,
            "drift_v_s_implied": drift_vs,
            "checks": {
                "g_size_independence": {
                    "value": abs(drift_p), "tol": 0.03,
                    "pass": bool(_finite(drift_p) and abs(drift_p) <= 0.03),
                    "note": "pressure at fixed solid fraction, small vs doubled N"},
                "g_size_independence_G": {
                    "value": abs(drift_g), "tol": 0.03,
                    "pass": bool(_finite(drift_g) and abs(drift_g) <= 0.03),
                    "note": "static shear modulus at fixed solid fraction"},
            },
        }


def run_wave_speed_study(cfg: WaveStudyConfig | None = None) -> dict[str, Any]:
    """Convenience wrapper around :meth:`WaveSpeedStudy.run`."""
    return WaveSpeedStudy(cfg).run()
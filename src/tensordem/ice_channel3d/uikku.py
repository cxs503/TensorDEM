"""MT Uikku campaign driver: ten level-ice cases, ablations, criteria.

The reference table is ``benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv``
(transcribed from Hu & Zhou 2016, DOI 10.1016/j.ijnaoe.2016.01.004).  The
column ``experimental_reference_table8_kN`` is used, because that is the column
the paper itself compares formulas against; the second experimental column
(``measured_mean_resistance_table3_kN``) is carried in the artifacts and its
discrepancy is documented, never silently reconciled.

Nothing in this module is fitted to the ten cases: hull particulars are the ones
already used by :mod:`tensordem.level_ice.formula` and the closing constants are
inherited from that module's graded ledger.  The campaign therefore measures the
model, it does not calibrate it.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .slab import (IceSheetProps, HullFootprint, RunSummary, SlabChannelSim,
                   SlabConfig)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "benchmarks" / "dem3d" / "ship_ice_validation" / "mt_uikku_level_ice.csv"

#: Criteria of the campaign (P82W2-N05).  Thresholds are the machine-checkable
#: numbers quoted in the mission; they are declared here, before any run.
CRITERIA: Dict[str, Dict[str, object]] = {
    "a_mass_conservation": {"max_frame_relative_error": 1e-6,
                            "definition": "max over frames of |m0+in-out-deleted-m(t)|/m0"},
    "b_flux_closure": {"closure_band": [0.99, 1.01],
                       "definition": "(departure-return)/(m_res(t)-m_res(0))"},
    "c_resolution_independence": {"max_relative_change": 0.05,
                                  "definition": "|R(layers=2)-R(layers=4)|/R and |R(r=2m)-R(r=1.5m)|/R"},
    "d_width_scan": {"monotone": "non-increasing with W/B",
                     "two_d_limit_max_relative_change": 0.10,
                     "definition": "R(W/B) non-increasing; R(W/B=0.3) vs the closed 2-D slice result"},
    "e_resistance_mape": {"target_percent": 15.0,
                          "fallback": "per-case attribution required if exceeded"},
    "f_negative_control": {"min_mape_percent": 100.0,
                           "definition": "2-D slice, closed walls, instant clearance"},
}


def load_reference_cases(path: Optional[Path] = None) -> List[Dict[str, object]]:
    """Read the ten Hu & Zhou level-ice cases (SI units, missing stays missing)."""
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"reference table not found: {csv_path}")
    cases: List[Dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            def num(key: str) -> Optional[float]:
                raw = (row.get(key) or "").strip()
                return None if raw == "" else float(raw)
            cases.append({
                "case_id": row["test_id"].strip(),
                "ice_sheet": row["ice_sheet"].strip(),
                "ice_thickness_m": num("ice_thickness_m"),
                "flexural_strength_kPa": num("flexural_strength_kPa"),
                "compressive_strength_kPa": num("compressive_strength_kPa"),
                "elastic_modulus_MPa": num("ice_elastic_modulus_MPa"),
                "drift_speed_m_s": num("ice_drift_speed_m_s"),
                "measured_table3_kN": num("measured_mean_resistance_table3_kN"),
                "reference_table8_kN": num("experimental_reference_table8_kN"),
                "lindqvist_kN": num("lindqvist_prediction_kN"),
                "riska_kN": num("riska_prediction_kN"),
                "jeong_kN": num("jeong_prediction_kN"),
                "keinonen_kN": num("keinonen_prediction_kN"),
            })
    if len(cases) != 10:
        raise ValueError(f"expected 10 level-ice cases, found {len(cases)}")
    return cases


def ice_props_of(case: Dict[str, object]) -> IceSheetProps:
    return IceSheetProps(
        thickness_m=float(case["ice_thickness_m"]),
        flexural_strength_pa=float(case["flexural_strength_kPa"]) * 1e3,
        compressive_strength_pa=float(case["compressive_strength_kPa"]) * 1e3,
        elastic_modulus_pa=float(case["elastic_modulus_MPa"]) * 1e6,
    )


def run_identifier(case: Dict[str, object], config: SlabConfig,
                   hull: Optional[HullFootprint] = None) -> RunSummary:
    """One case -> one run, with the drift speed of that case."""
    cfg = SlabConfig(**{**config.as_dict(), "drift_speed_m_s": float(case["drift_speed_m_s"])})
    sim = SlabChannelSim(cfg, ice_props_of(case), hull=hull, case_id=str(case["case_id"]))
    return sim.run()


def run_ablation(case: Dict[str, object], base: SlabConfig,
                 hull: Optional[HullFootprint] = None) -> Dict[str, RunSummary]:
    """One-factor-at-a-time ablation around the corrected configuration.

    ``corrected``     3-D slab, open lateral boundary + reservoir, no deletion
    ``no_opening``    same but with reflective lateral walls (dimension kept)
    ``single_layer``  same but one slice through the thickness (2-D in z)
    ``instant_clear`` same but floes in the wake are deleted (legacy clearance)
    ``legacy_2d``     closed walls + one slice + instant clearance at the same
                      channel width: the configuration that produced the
                      16702 % distortion
    """
    corrected = run_identifier(case, base, hull=hull)
    no_opening = run_identifier(case, SlabConfig(**{**base.as_dict(), "lateral_open": False}), hull=hull)
    single_layer = run_identifier(case, SlabConfig(**{**base.as_dict(), "n_layers": 1}), hull=hull)
    instant_clear = run_identifier(case, SlabConfig(**{**base.as_dict(), "instant_removal": True}), hull=hull)
    legacy = run_identifier(case, SlabConfig(**{**base.as_dict(), "lateral_open": False,
                                               "n_layers": 1, "instant_removal": True}),
                            hull=hull)
    return {"corrected": corrected, "no_opening": no_opening,
            "single_layer": single_layer, "instant_clear": instant_clear,
            "legacy_2d": legacy}


def w_over_b_scan(case: Dict[str, object], base: SlabConfig,
                  widths: Sequence[float] = (0.3, 1.0, 1.5, 2.0),
                  hull: Optional[HullFootprint] = None) -> Dict[str, RunSummary]:
    out: Dict[str, RunSummary] = {}
    for w in widths:
        cfg = SlabConfig(**{**base.as_dict(), "channel_width_over_beam": float(w)})
        out[f"{w:g}"] = run_identifier(case, cfg, hull=hull)
    return out


def mape(predictions: Sequence[float], references: Sequence[float]) -> float:
    pred = np.asarray(predictions, dtype=float)
    ref = np.asarray(references, dtype=float)
    if pred.shape != ref.shape:
        raise ValueError("predictions and references must have the same shape")
    return float(np.mean(np.abs(pred - ref) / np.abs(ref)) * 100.0)


@dataclass
class UikkuCampaign:
    """Ten-case resistance comparison for one configuration."""

    config: SlabConfig
    hull: HullFootprint

    def run(self, cases: Optional[Sequence[Dict[str, object]]] = None
            ) -> Dict[str, object]:
        rows = list(cases if cases is not None else load_reference_cases())
        results: List[Dict[str, object]] = []
        for case in rows:
            summary = run_identifier(case, self.config, hull=self.hull)
            ref = float(case["reference_table8_kN"])
            pred = summary.total_with_closures_kN
            results.append({
                "case_id": case["case_id"],
                "ice_thickness_m": case["ice_thickness_m"],
                "drift_speed_m_s": case["drift_speed_m_s"],
                "reference_kN": ref,
                "measured_table3_kN": case["measured_table3_kN"],
                "lindqvist_kN": case["lindqvist_kN"],
                "predicted_kN": pred,
                "predicted_dem_only_kN": summary.resistance_mean_kN,
                "signed_error_percent": (pred - ref) / ref * 100.0,
                "abs_error_percent": abs(pred - ref) / ref * 100.0,
                "break_component_kN": summary.break_component_kN,
                "crush_component_kN": summary.crush_component_kN,
                "submersion_closure_kN": summary.submersion_closure_kN,
                "max_frame_mass_error": summary.max_frame_mass_error,
                "flux_closure_ratio": summary.flux_closure_ratio,
                "deleted_kg": summary.deleted_kg,
                "breaking_events": summary.breaking_events,
                "contacts": summary.contacts_in_contact_set,
            })
        predictions = [r["predicted_kN"] for r in results]
        references = [r["reference_kN"] for r in results]
        return {
            "config": self.config.as_dict(),
            "hull": self.hull.__dict__,
            "cases": results,
            "mape_percent": mape(predictions, references),
            "bias_percent": float(np.mean((np.asarray(predictions) - np.asarray(references))
                                          / np.asarray(references)) * 100.0),
            "rmse_kN": float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(references)) ** 2))),
        }
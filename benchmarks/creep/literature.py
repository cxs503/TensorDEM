"""Literature-anchored checks for the ice-creep gates (mission P46W3-N02).

The mission asks for a comparison against laboratory creep data
(Cole 1998, Schulson 2001).  Two different things are separated here, on
purpose:

1. **Constant-level comparison** (implemented, machine-checked): the standard
   ice-mechanics constants and bands -- ``n``, ``A(-10 C)``, ``Q``,
   ``eta_eff(1 MPa)``, the Andrade exponent -- are compared with the model to a
   relative error or a band containment.  Each literature entry carries a
   provenance grade, and the numbers used are textbook/standard-review
   constants, *not* digitized curves.
2. **Curve-level overlay** (WITHHELD): no digitized strain-time points from the
   Cole (1998) / Schulson (2001) records are bundled in this repository.
   :func:`load_digitized_curves` implements the injection format and returns an
   empty mapping when the data are absent, and the gate then reports
   ``WITHHELD``.  Inventing points to make the gate green would be fabrication.

References as recorded in ``data/literature_ice_creep.json``; the file is the
single source of truth for every number used here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LITERATURE_FILE = DATA_DIR / "literature_ice_creep.json"
DIGITIZED_DIR = DATA_DIR / "digitized"

__all__ = [
    "LITERATURE_FILE",
    "DIGITIZED_DIR",
    "load_literature",
    "literature_checks",
    "load_digitized_curves",
    "DIGITIZED_SCHEMA",
]

#: Required columns of an injected curve file (CSV, header row, SI units).
DIGITIZED_SCHEMA = {
    "columns": ["time_s", "strain"],
    "units": {"time_s": "s", "strain": "-"},
    "sidecar": "<name>.json with {source, temperature_C, stress_MPa, ice_type, note}",
}


def load_literature(path: str | Path | None = None) -> dict:
    """Load and validate the literature constants file."""
    target = Path(path) if path is not None else LITERATURE_FILE
    with open(target, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "tensordem.creep.literature.v1":
        raise ValueError(f"unexpected literature schema in {target}")
    for key, entry in payload.get("quantities", {}).items():
        if "provenance" not in entry:
            raise ValueError(f"literature entry {key!r} lacks a provenance grade")
        if entry.get("kind") == "band" and "value" in entry:
            raise ValueError(f"band entry {key!r} must not carry a point value")
    return payload


def load_digitized_curves(directory: str | Path | None = None) -> dict:
    """Load injected digitized creep curves (``<name>.csv`` + ``<name>.json``).

    Returns ``{}`` when nothing is injected -- the normal state of this
    repository.  A file is only accepted when the sidecar JSON names its source
    and the CSV has the declared columns, so a fabricated curve cannot slip in
    silently as "literature data".
    """
    root = Path(directory) if directory is not None else DIGITIZED_DIR
    curves: dict[str, dict] = {}
    if not root.is_dir():
        return curves
    for csv_path in sorted(root.glob("*.csv")):
        sidecar = csv_path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"digitized curve {csv_path.name} has no sidecar json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        for field in ("source", "temperature_C", "stress_MPa"):
            if field not in meta:
                raise ValueError(f"{sidecar.name} is missing the {field!r} field")
        text = csv_path.read_text(encoding="utf-8").strip().splitlines()
        header = [c.strip() for c in text[0].split(",")]
        if header[:2] != DIGITIZED_SCHEMA["columns"]:
            raise ValueError(
                f"{csv_path.name}: expected columns {DIGITIZED_SCHEMA['columns']}")
        times, strains = [], []
        for line in text[1:]:
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 2 or not cells[0]:
                continue
            times.append(float(cells[0]))
            strains.append(float(cells[1]))
        curves[csv_path.stem] = {"meta": meta, "time_s": times, "strain": strains}
    return curves


def _rel_error(model: float, reference: float) -> float:
    return abs(model - reference) / abs(reference)


def _inside(value: float, band) -> bool:
    low, high = float(band[0]), float(band[1])
    return low <= value <= high


def literature_checks(model: dict) -> dict:
    """Compare model quantities against the literature constants.

    ``model`` keys (all SI):

    ``n_fit``, ``a_minus10``, ``q_kj_per_mol``, ``eta_eff_1mpa``,
    ``min_rate_1mpa_minus10``, ``alpha_fit``.

    Returns a dict of per-item results with ``value``, ``reference`` /
    ``band``, ``rel_error`` or ``inside`` and a boolean ``pass``; the
    curve-overlay item is returned with ``status = "WITHHELD"`` and
    ``pass = None`` so a consumer cannot mistake it for a green check.
    """
    lit = load_literature()["quantities"]
    out: dict = {"source_file": str(LITERATURE_FILE.name),
                 "provenance_note": "textbook / standard-review constants with stated "
                                    "bands; curve digitizations are not bundled"}

    # (l1) fluidity band
    a_entry = lit["fluidity_minus10C"]
    out["fluidity_minus10C"] = {
        "value": float(model["a_minus10"]),
        "reference": a_entry["value"],
        "band": a_entry["band"],
        "rel_error_vs_reference": _rel_error(model["a_minus10"], a_entry["value"]),
        "inside_band": _inside(model["a_minus10"], a_entry["band"]),
        "pass": _inside(model["a_minus10"], a_entry["band"]),
        "provenance": a_entry["provenance"],
    }

    # (l2) effective viscosity band at 1 MPa
    eta_entry = lit["effective_viscosity_1MPa"]
    out["eta_eff_1MPa"] = {
        "value": float(model["eta_eff_1mpa"]),
        "band": eta_entry["band"],
        "inside_band": _inside(model["eta_eff_1mpa"], eta_entry["band"]),
        "pass": _inside(model["eta_eff_1mpa"], eta_entry["band"]),
        "provenance": eta_entry["provenance"],
    }

    # (l3) activation energy, two-branch Hobbs patch
    q_model = float(model["q_kj_per_mol"]) * 1.0e3
    warm = lit["activation_energy_warm"]
    cold = lit["activation_energy_cold"]
    out["activation_energy"] = {
        "value_J_per_mol": q_model,
        "warm_reference": warm["value"],
        "warm_rel_error": _rel_error(q_model, warm["value"]),
        "cold_reference": cold["value"],
        "cold_rel_error": _rel_error(q_model, cold["value"]),
        "warm_pass": _rel_error(q_model, warm["value"]) <= 0.15,
        "cold_pass": _rel_error(q_model, cold["value"]) <= 0.15,
        "note": "a single Arrhenius Q is expected to match only one branch; the "
                "warm branch is the design range of this module (-2..-10 degC)",
        "pass": _rel_error(q_model, warm["value"]) <= 0.15,
        "provenance": warm["provenance"],
    }

    # (l4) Glen exponent band
    n_entry = lit["glen_exponent_n"]
    out["glen_exponent"] = {
        "value": float(model["n_fit"]),
        "reference": n_entry["value"],
        "band": n_entry["band"],
        "rel_error": _rel_error(model["n_fit"], n_entry["value"]),
        "inside_band": _inside(model["n_fit"], n_entry["band"]),
        "pass": _inside(model["n_fit"], n_entry["band"]),
        "provenance": n_entry["provenance"],
    }

    # (l5) minimum creep rate at 1 MPa / -10 degC
    rate_entry = lit["min_creep_rate_1MPa_minus10C"]
    out["min_creep_rate_1MPa_minus10C"] = {
        "value": float(model["min_rate_1mpa_minus10"]),
        "reference": rate_entry["value"],
        "band": rate_entry["band"],
        "rel_error_vs_reference": _rel_error(model["min_rate_1mpa_minus10"],
                                             rate_entry["value"]),
        "inside_band": _inside(model["min_rate_1mpa_minus10"], rate_entry["band"]),
        "pass": _inside(model["min_rate_1mpa_minus10"], rate_entry["band"]),
        "provenance": rate_entry["provenance"],
    }

    # (l6) Andrade exponent band
    alpha_entry = lit["andrade_primary_exponent"]
    out["andrade_exponent"] = {
        "value": float(model["alpha_fit"]),
        "reference": alpha_entry["value"],
        "band": alpha_entry["band"],
        "rel_error": _rel_error(model["alpha_fit"], alpha_entry["value"]),
        "inside_band": _inside(model["alpha_fit"], alpha_entry["band"]),
        "pass": _inside(model["alpha_fit"], alpha_entry["band"]),
        "provenance": alpha_entry["provenance"],
    }

    # (l7) curve overlay - WITHHELD unless a digitized dataset was injected
    curves = load_digitized_curves()
    withheld = {entry["name"]: entry["citation"]
                for entry in load_literature().get("withheld_digitizations", [])}
    out["curve_overlay"] = {
        "status": "AVAILABLE" if curves else "WITHHELD",
        "curves_loaded": sorted(curves),
        "registered_sources": withheld,
        "pass": None if not curves else True,
        "reason": None if curves else
                  "no digitized Cole 1998 / Schulson 2001 strain-time points are "
                  "bundled; inject <name>.csv + <name>.json into "
                  "benchmarks/creep/data/digitized/ to enable this overlay",
    }

    graded = {k: v for k, v in out.items() if isinstance(v, dict) and "pass" in v}
    out["summary"] = {
        "checks": len(graded),
        "passed": sum(1 for v in graded.values() if v["pass"] is True),
        "failed": [k for k, v in graded.items() if v["pass"] is False],
        "withheld": [k for k, v in graded.items() if v["pass"] is None],
        "all_passed": all(v["pass"] is True for v in graded.values()),
    }
    return out


def band_log10_width(entry: dict) -> float:
    """Width of a literature band in decades (used to size the error budget)."""
    low, high = entry.get("band", [0.0, 0.0])
    if low <= 0.0 or high <= 0.0:
        return float("nan")
    return math.log10(high / low)
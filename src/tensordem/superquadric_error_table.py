"""Clump-approximation error table and the API gate that consumes it.

Purpose
-------
The campaign deliverable is a *recomputeable* table that answers, for a given
particle shape and clump budget, whether the cheaper sphere-clump
representation may be used or the true superquadric contact kernel is required.
The table is written as JSON (``artifacts/superquadric/clump_disable_ratio_table.json``)
and read back by :func:`assess_clump_approximation`, which is the gate the rest
of the platform calls.

Decision rule (each criterion is stored in the JSON alongside the tables so the
verdict can be re-derived without this module):

* geometric : ``max_distance_ratio`` (two-sided surface deviation / smallest half
  extent) <= ``max_distance_ratio`` criterion (default 1e-2);
* repose    : ``|theta_true - theta_clump| <= 3 deg`` -- only *measured* entries
  carry this number; entries filled by the surrogate are flagged
  ``evidence = "surrogate"`` and are rejected unless the caller opts in with
  ``allow_surrogate=True``;
* force chain : ``force_chain_deviation_deg <= 5 deg`` (same evidence rule).

Everything the gate returns is JSON-serialisable so it can be logged.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .clump_geometry import fit_clump, geometric_error

__all__ = [
    "TABLE_SCHEMA_VERSION",
    "DEFAULT_CRITERIA",
    "build_geometric_table",
    "ClumpGateDecision",
    "build_disable_ratio_table",
    "write_disable_ratio_table",
    "load_disable_ratio_table",
    "assess_clump_approximation",
]

TABLE_SCHEMA_VERSION = "tensordem.superquadric.disable_ratio_table/1"

DEFAULT_CRITERIA: Dict[str, float] = {
    "max_distance_ratio": 1.0e-2,          # two-sided deviation / smallest half extent
    "max_normal_deviation_deg": 10.0,
    "max_repose_error_deg": 3.0,
    "max_force_chain_deviation_deg": 5.0,
    "max_performance_ratio": 3.0,
}


def _shape_from_aspect(aspect_ratio: float, exponent: float) -> Tuple[float, float, float, float, float]:
    """Body with ``min half extent = 1 mm`` and the requested aspect ratio."""
    base = 1.0e-3
    return (aspect_ratio * base, base, base, exponent, exponent)


def build_geometric_table(
    aspect_ratios: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    exponents: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
    n_spheres: Sequence[int] = (1, 4, 8, 16),
) -> List[Dict[str, object]]:
    """Geometric error rows for the (aspect ratio x exponent x N) grid."""
    rows: List[Dict[str, object]] = []
    for ar in aspect_ratios:
        for n in exponents:
            a, b, c, n1, n2 = _shape_from_aspect(ar, n)
            for ns in n_spheres:
                fit = fit_clump(a, b, c, n1, n2, n_spheres=ns)
                err = geometric_error(a, b, c, n1, n2, fit)
                row = err.to_dict()
                row.update({
                    "missing_max_m": float(fit.missing_max),
                    "protrusion_max_m": float(fit.protrusion_max),
                    "evidence": "geometric",
                })
                rows.append(row)
    return rows


@dataclass
class ClumpGateDecision:
    """Verdict of the clump-approximation gate."""
    use_clump: bool
    evidence: str
    reasons: List[str] = field(default_factory=list)
    row: Dict[str, object] = field(default_factory=dict)
    criteria: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "use_clump": bool(self.use_clump),
            "evidence": self.evidence,
            "reasons": list(self.reasons),
            "row": dict(self.row),
            "criteria": dict(self.criteria),
        }


def build_disable_ratio_table(
    geometric_rows: Optional[List[Dict[str, object]]] = None,
    macroscopic_rows: Optional[Iterable[Dict[str, object]]] = None,
    *,
    criteria: Optional[Dict[str, float]] = None,
    performance_ratio: Optional[float] = None,
    provenance: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Assemble the JSON table (geometric rows + measured macroscopic rows).

    ``macroscopic_rows`` entries must carry
    ``aspect_ratio``, ``shape_exponent``, ``n_spheres``,
    ``repose_error_deg`` and ``force_chain_deviation_deg`` (plus optional
    diagnostics).  They are merged onto the matching geometric row; rows without
    a macroscopic measurement get ``evidence = "geometric_only"``.
    """
    crit = dict(DEFAULT_CRITERIA)
    if criteria:
        crit.update(criteria)
    rows = list(geometric_rows) if geometric_rows is not None else build_geometric_table()
    by_key = {(float(r["aspect_ratio"]), float(r["shape_exponent"]), int(r["n_spheres"])): r
              for r in rows}
    for macro in macroscopic_rows or ():
        key = (float(macro["aspect_ratio"]), float(macro["shape_exponent"]),
               int(macro["n_spheres"]))
        row = by_key.get(key)
        if row is None:
            continue
        row["repose_error_deg"] = float(macro["repose_error_deg"])
        row["force_chain_deviation_deg"] = float(macro["force_chain_deviation_deg"])
        row["evidence"] = "measured"
        if "note" in macro:
            row["note"] = macro["note"]
    for row in rows:
        row.setdefault("evidence", "geometric")
        ok_geom = (float(row["max_distance_ratio"]) <= crit["max_distance_ratio"]
                   and float(row["max_normal_deviation_deg"]) <= crit["max_normal_deviation_deg"])
        ok_macro = True
        if "repose_error_deg" in row:
            ok_macro = (float(row["repose_error_deg"]) <= crit["max_repose_error_deg"]
                        and float(row["force_chain_deviation_deg"])
                        <= crit["max_force_chain_deviation_deg"])
        row["use_clump"] = bool(ok_geom and ok_macro)
        row["verdict_reasons"] = _reasons(row, crit)
    table = {
        "schema": TABLE_SCHEMA_VERSION,
        "criteria": crit,
        "rows": rows,
        "summary": {
            "n_rows": len(rows),
            "n_allowed": int(sum(1 for r in rows if r.get("use_clump"))),
            "performance_ratio_true_vs_clump": performance_ratio,
        },
        "provenance": provenance or {},
    }
    return table


def _reasons(row: Dict[str, object], crit: Dict[str, float]) -> List[str]:
    out: List[str] = []
    if float(row["max_distance_ratio"]) > crit["max_distance_ratio"]:
        out.append(f"max_distance_ratio {row['max_distance_ratio']:.4g} > "
                   f"{crit['max_distance_ratio']:g}")
    if float(row["max_normal_deviation_deg"]) > crit["max_normal_deviation_deg"]:
        out.append(f"max_normal_deviation_deg {row['max_normal_deviation_deg']:.4g} > "
                   f"{crit['max_normal_deviation_deg']:g}")
    if "repose_error_deg" in row and float(row["repose_error_deg"]) > crit["max_repose_error_deg"]:
        out.append(f"repose_error_deg {row['repose_error_deg']:.4g} > "
                   f"{crit['max_repose_error_deg']:g}")
    if ("force_chain_deviation_deg" in row and
            float(row["force_chain_deviation_deg"]) > crit["max_force_chain_deviation_deg"]):
        out.append(f"force_chain_deviation_deg {row['force_chain_deviation_deg']:.4g} > "
                   f"{crit['max_force_chain_deviation_deg']:g}")
    return out


def write_disable_ratio_table(table: Dict[str, object], path: str) -> str:
    """Write the table JSON (creating parent directories) and return the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2, sort_keys=True)
    return path


def load_disable_ratio_table(path: str) -> Dict[str, object]:
    """Load and schema-check a disable-ratio table."""
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    if table.get("schema") != TABLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported table schema {table.get('schema')!r}")
    if "rows" not in table or "criteria" not in table:
        raise ValueError("table is missing 'rows' or 'criteria'")
    return table


def assess_clump_approximation(
    aspect_ratio: float,
    shape_exponent: float,
    n_spheres: int,
    *,
    table_path: Optional[str] = None,
    table: Optional[Dict[str, object]] = None,
    allow_surrogate: bool = False,
) -> ClumpGateDecision:
    """Gate call: may this shape be replaced by an ``n_spheres`` clump?

    Raises ``ValueError`` when neither ``table`` nor ``table_path`` is given, and
    ``KeyError`` when the (aspect ratio, exponent, N) combination is not in the
    table -- the gate never extrapolates silently.
    """
    if table is None:
        if table_path is None:
            raise ValueError("either 'table' or 'table_path' is required")
        table = load_disable_ratio_table(table_path)
    rows = table["rows"]
    crit = table["criteria"]
    match = None
    for row in rows:
        if (abs(float(row["aspect_ratio"]) - float(aspect_ratio)) < 1e-9
                and abs(float(row["shape_exponent"]) - float(shape_exponent)) < 1e-9
                and int(row["n_spheres"]) == int(n_spheres)):
            match = row
            break
    if match is None:
        raise KeyError(f"no table row for aspect_ratio={aspect_ratio}, "
                       f"shape_exponent={shape_exponent}, n_spheres={n_spheres}")
    reasons = list(match.get("verdict_reasons", []))
    evidence = str(match.get("evidence", "geometric_only"))
    use = bool(match.get("use_clump", False))
    if evidence != "measured" and not allow_surrogate:
        use = False
        reasons.append("macroscopic evidence missing (geometric-only verdict is "
                       "rejected unless allow_surrogate=True)")
    return ClumpGateDecision(use_clump=use, evidence=evidence, reasons=reasons,
                             row=dict(match), criteria=dict(crit))
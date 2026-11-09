"""``r_u(T_v)`` interface table: the machine-readable hand-off to liquefaction.

The liquefaction chain (see :mod:`tensordem.cargo_liquefaction_effstress`)
drives the effective-stress reduction with an excess pore-pressure ratio
``r_u``.  Until now that number came from a shear-induced contractancy
closure with no time scale.  This module publishes the *consolidation* side of
the same quantity as a sampled, versioned JSON table so a downstream criterion
script can read it without importing any internals:

``r_u(T_v) = u_excess(T_v) / d_sigma``
    ``r_u_average`` is the depth average (``= 1 - U``, mass-consistent),
    ``r_u_midplane`` is the value at the impermeable boundary (the slowest
    point, the conservative choice for a stability criterion), and
    ``r_u_profile`` samples the whole depth for criteria that need ``z``.

Every table carries its own applicability verdict and its ``C_v`` confidence
band, so a consumer can refuse a table instead of silently extrapolating.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .terzaghi import U_series, u_profile

__all__ = [
    "SCHEMA",
    "build_ru_table",
    "write_ru_table",
    "read_ru_table",
    "r_u_at",
    "r_u_profile_at",
    "table_sha256",
]

SCHEMA = "tensordem.consolidation.ru_table/1"

#: Depth samples (fraction of the drainage path, 0 = drained face).
DEFAULT_DEPTHS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def build_ru_table(
    *,
    Tv_grid: Sequence[float],
    drainage: str,
    drainage_path: float,
    C_v: float,
    C_v_ci: dict | None = None,
    applicability: dict | None = None,
    provenance: dict | None = None,
    profile_Tv: Sequence[float] | None = None,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    validation: dict | None = None,
) -> dict:
    """Assemble the exported table (pure function, no I/O)."""
    Tv = np.asarray(sorted(float(t) for t in Tv_grid), dtype=np.float64)
    if Tv.size < 2 or Tv[0] <= 0.0:
        raise ValueError("Tv_grid needs at least two strictly positive entries")
    if not C_v > 0.0:
        raise ValueError("C_v must be positive")
    if drainage not in ("double", "single"):
        raise ValueError("drainage must be 'double' or 'single'")

    U = np.asarray(U_series(Tv), dtype=np.float64)
    prof_Tv = np.asarray(sorted(float(t) for t in (profile_Tv or Tv[:: max(1, Tv.size // 5)])),
                         dtype=np.float64)
    depth = np.asarray(depths, dtype=np.float64)
    profile = np.asarray(u_profile(depth, prof_Tv), dtype=np.float64)
    if profile.ndim == 1:  # single profile_Tv
        profile = profile[None, :]

    return {
        "schema": SCHEMA,
        "definition": {
            "r_u": "excess pore pressure ratio u_excess / d_sigma",
            "T_v": C_v_definition(),
            "U": "degree of consolidation = settlement / final settlement",
            "drainage": drainage,
            "drainage_path_m": float(drainage_path),
            "anchors": "uniform initial excess pressure, step surcharge, "
                       "saturated, incompressible grains, constant k and E_oed",
        },
        "C_v_m2_s": float(C_v),
        "C_v_confidence": C_v_ci or {},
        "T_v": Tv.tolist(),
        "U_analytic": U.tolist(),
        "r_u_average": (1.0 - U).tolist(),
        "r_u_midplane": np.asarray(u_profile(1.0, Tv), dtype=np.float64).tolist(),
        "r_u_profile": {
            "z_over_H_dr": depth.tolist(),
            "T_v": prof_Tv.tolist(),
            "r_u": profile.tolist(),
        },
        "applicability": applicability or {},
        "validation": validation or {},
        "provenance": provenance or {},
    }


def C_v_definition() -> str:
    return "C_v = k * E_oed / gamma_w ; T_v = C_v * t / H_dr^2"


def table_sha256(table: dict) -> str:
    """Content hash of a table (stable key order, floats repr'd)."""
    blob = json.dumps(table, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_ru_table(path: str | Path, table: dict) -> dict:
    """Write the table and return ``{"path", "sha256", "bytes"}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = table_sha256(table)
    payload = dict(table)
    payload["integrity"] = {"sha256_of_content": digest, "schema": SCHEMA}
    blob = json.dumps(payload, indent=2, sort_keys=False)
    path.write_text(blob + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": digest, "bytes": len(blob)}


def read_ru_table(path: str | Path) -> dict:
    """Read and validate a table; refuses an unknown schema or bad content."""
    path = Path(path)
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema {table.get('schema')!r}, want {SCHEMA}")
    for key in ("T_v", "U_analytic", "r_u_average", "r_u_midplane", "C_v_m2_s"):
        if key not in table:
            raise ValueError(f"table is missing the required field {key!r}")
    Tv = np.asarray(table["T_v"], dtype=np.float64)
    if Tv.size < 2 or np.any(np.diff(Tv) <= 0.0):
        raise ValueError("T_v grid must be strictly increasing")
    ru = np.asarray(table["r_u_average"], dtype=np.float64)
    if np.any(ru < -1e-12) or np.any(ru > 1.0 + 1e-12):
        raise ValueError("r_u must lie in [0, 1]")
    if np.any(np.diff(ru) > 1e-12):
        raise ValueError("r_u must be non-increasing in T_v")
    stored = table.get("integrity", {}).get("sha256_of_content")
    if stored:
        content = {k: v for k, v in table.items() if k != "integrity"}
        if table_sha256(content) != stored:
            raise ValueError("table content hash mismatch (file was edited)")
    return table


def r_u_at(table: dict, Tv: float) -> float:
    """Depth-averaged ``r_u`` interpolated on the exported grid.

    Outside the tabulated window the *conservative* end value is returned:
    ``r_u(T_v -> 0)`` at the low end (no dissipation yet) and the last sample
    at the high end.
    """
    Tv_arr = np.asarray(table["T_v"], dtype=np.float64)
    ru = np.asarray(table["r_u_average"], dtype=np.float64)
    return float(np.interp(float(Tv), Tv_arr, ru))


def r_u_profile_at(table: dict, Tv: float) -> tuple[np.ndarray, np.ndarray]:
    """``(z/H_dr, r_u)`` at the tabulated profile time closest to ``Tv``."""
    block = table["r_u_profile"]
    Tv_tab = np.asarray(block["T_v"], dtype=np.float64)
    j = int(np.argmin(np.abs(Tv_tab - float(Tv))))
    depth = np.asarray(block["z_over_H_dr"], dtype=np.float64)
    return depth, np.asarray(block["r_u"][j], dtype=np.float64)
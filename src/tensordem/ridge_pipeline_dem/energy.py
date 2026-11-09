"""Energy-ledger closure for the ridge-keel / buried-pipeline grounding runs.

The worktree's grounding workflow (:mod:`tensordem.ridge_grounding`) already
accumulates every power channel of the coupled DEM during a run and dumps a
per-sample snapshot into ``history["energy_track"]``.  This module turns that
record into an auditable closure statement with an **explicit partition**,
because the partition is exactly where a closure statement can silently lie.

The statement integrated over a window ``[t0, t1]`` of one run is

.. code-block:: text

    W_platen + W_body                                  (sources)
      = dK + dE_bond + dE_contact                      (particle storage)
      + phi_contact_damping + phi_contact_friction + phi_drag
      + phi_pipe_damping + phi_pipe_friction
      + phi_soil_damping + phi_soil_friction           (DEM dissipation)
      + A_pipe + A_soil + A_plane                      (boundary absorption)
      + E_fracture                                     (bond release)
      + residual

Partition rules (each one is a place where the naive sum double counts):

``A_soil`` is the energy the Mohr-Coulomb subgrade *takes out of the
particles*.  It contains both the reversible part (the ``min(k*delta, q_ult)``
elastic reaction) and the irreversible part (plastic flow plus the
velocity-dependent soil dashpot).  The reversible part is therefore already
inside ``A_soil`` and ``dE_soil`` (the state-based subgrade storage) must **not**
be added a second time.  Adding both - the convention of
``ridge_grounding._close_energy_ledger`` - inflates the sinks by the full
elastic subgrade energy; on the campaign runs that is a ~10-20 % of the source
budget and it is the dominant reason the legacy number reads 30-70 % instead of
a few per cent.

``A_pipe`` is likewise the energy the pipeline takes out of the ice.  Its
internal split (spring storage / damping / kinetic) is reported but, again, the
spring storage is already inside ``A_pipe``.

``E_fracture`` is the elastic energy held by freeze bonds at the instant they
break.  It leaves the particle system at that instant, so it is a sink; the
surviving bonds' storage is ``dE_bond`` and the two do not overlap.

Everything here is a pure function of the recorded track: no run is repeated,
no number is hand-copied, and any window of the record can be audited.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

__all__ = [
    "PARTITIONS",
    "closure_from_track",
    "closure_scan",
    "dt_refinement_table",
    "sample_index_at_time",
]

PARTITIONS = ("corrected", "legacy")
"""``corrected`` = soil storage counted once (this module's default).

``legacy`` reproduces ``ridge_grounding._close_energy_ledger`` term by term
(``dE_soil`` added on top of ``A_soil``); it is kept so the report can show the
before/after numbers of the fix instead of asserting them.
"""


def _get(sample: Mapping, key: str) -> float:
    try:
        return float(sample[key])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"track sample is missing {key!r}") from exc


def sample_index_at_time(track: Sequence[Mapping], time: float) -> int:
    """First sample at or after ``time`` (linear scan, records are short)."""
    for index, sample in enumerate(track):
        if _get(sample, "time") >= time:
            return index
    return len(track) - 1


def closure_from_track(track: Sequence[Mapping], *, start: int = 0,
                       end: int | None = None,
                       partition: str = "corrected",
                       include_fracture: bool = True) -> dict:
    """Close the energy ledger over ``track[start:end+1]``.

    ``partition='corrected'`` counts the subgrade storage exactly once (inside
    ``A_soil``); ``partition='legacy'`` reproduces the double-counting of
    :meth:`tensordem.ridge_grounding._close_energy_ledger` so that the two can be
    compared on the same record.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    if len(track) < 2:
        raise ValueError("an energy track needs at least two samples")
    if end is None:
        end = len(track) - 1
    if not 0 <= start < end < len(track):
        raise ValueError("require 0 <= start < end < len(track)")
    head, tail = track[start], track[end]

    def delta(key: str) -> float:
        return _get(tail, key) - _get(head, key)

    platen = delta("platen_work")
    body = delta("body_work")
    d_kinetic = delta("kinetic")
    d_bond = delta("elastic_bond")
    d_contact = delta("elastic_contact")
    d_soil = delta("elastic_soil")
    dissipated = delta("dissipated")
    absorbed_pipe = -delta("exchanged_pipe")
    absorbed_soil = -delta("exchanged_soil")
    absorbed_plane = -delta("exchanged_plane")
    fracture = max(delta("bond_fracture"), 0.0)
    sources = platen + body
    sinks = (d_kinetic + d_bond + d_contact + dissipated + absorbed_pipe
             + absorbed_soil + absorbed_plane)
    if include_fracture:
        sinks += fracture
    if partition == "legacy":
        sinks += d_soil
    residual = sources - sinks
    scale = max(abs(sources), abs(sinks), 1.0e-12)
    return {
        "t_start": _get(head, "time"),
        "t_end": _get(tail, "time"),
        "samples": end - start + 1,
        "partition": partition,
        "sources": {"platen": platen, "body": body, "total": sources},
        "sinks": {
            "kinetic": d_kinetic,
            "bond_storage": d_bond,
            "contact_storage": d_contact,
            "soil_storage_double_count": d_soil if partition == "legacy" else 0.0,
            "dissipated_dem": dissipated,
            "absorbed_pipe": absorbed_pipe,
            "absorbed_soil": absorbed_soil,
            "absorbed_plane": absorbed_plane,
            "bond_fracture": fracture if include_fracture else 0.0,
            "total": sinks,
        },
        "residual": residual,
        "closure_error": abs(residual) / scale,
        "closed": abs(residual) / scale < 0.05,
    }


def closure_scan(track: Sequence[Mapping], *, fractions: Iterable[float] = (
        0.0, 0.125, 0.25, 0.5), partition: str = "corrected") -> list[dict]:
    """Closure of every suffix window ``[fraction*T, T]`` of the record.

    The suffix scan separates the kinematic start-up transient (the platen is
    switched on instantaneously at ``t=0``, which fires a stress wave through the
    bonded lattice and breaks the first bond population) from the quasi-static
    loading phase that the physical question is about.
    """
    last = len(track) - 1
    rows = []
    for fraction in fractions:
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fractions must lie in [0, 1)")
        start = int(round(fraction * last))
        start = min(start, last - 1)
        row = closure_from_track(track, start=start, partition=partition)
        row["fraction"] = fraction
        rows.append(row)
    return rows


def dt_refinement_table(entries: Sequence[Mapping], *,
                        partition: str = "corrected") -> dict:
    """Closure error vs time step for a set of runs of the same case.

    Each entry is ``{"dt": float, "track": track}`` (plus optional ``label``).
    Returns the table sorted from coarse to fine, the ratio between successive
    errors, and the verdict "the ledger closes in the steady loading phase".
    """
    rows = []
    for entry in entries:
        track = entry["track"]
        rows.append({"label": entry.get("label"), "dt": float(entry["dt"]),
                     "whole_record": closure_from_track(track, partition=partition),
                     "loading_phase": closure_from_track(
                         track, start=len(track) // 4, partition=partition)})
    rows.sort(key=lambda row: row["dt"], reverse=True)
    ratios = []
    for coarse, fine in zip(rows, rows[1:]):
        if fine["whole_record"]["closure_error"] > 0.0:
            ratios.append({"ratio_coarse_over_fine":
                           coarse["whole_record"]["closure_error"]
                           / fine["whole_record"]["closure_error"]})
    return {
        "rows": rows,
        "refinement_ratios": ratios,
        "finest_whole_record": rows[-1]["whole_record"]["closure_error"] if rows else None,
        "finest_loading_phase": rows[-1]["loading_phase"]["closure_error"] if rows else None,
        "monotone_in_dt": all(r["ratio_coarse_over_fine"] >= 1.0 for r in ratios),
        "satisfied_loading_phase": bool(rows and math.isfinite(
            rows[-1]["loading_phase"]["closure_error"])
            and rows[-1]["loading_phase"]["closure_error"] < 0.05),
    }
"""Contact-force time-history export contract (DEM -> fatigue chain).

Why this module exists
----------------------
Ice-zone fatigue conclusions usually start from a *load statistic* (a peak, an
RMS, a rainflow histogram) rather than from the DEM contact-force time history
that produced it.  Between the solver and the S-N curve there was no schema, no
transport and no self-check, so "the load that reached the S-N curve" could
silently differ from the load the DEM computed.  This module defines the
transport object and its admissibility checks.

Columns (one row per sample of one contact)
-------------------------------------------
``t, contact_id, floe_i, floe_j, F_n, F_t, patch_area`` and the optional
physics-complete energy group ``delta, v_n, e_spring, e_diss``.

Discipline enforced by :meth:`ContactForceHistory.validate`
-----------------------------------------------------------
* every column finite, equal length;
* ``t`` strictly increasing **within** a contact, non-decreasing over the table
  (two contacts may share a time stamp);
* ``F_n >= 0``, ``F_t >= 0``, ``patch_area >= 0``, ``delta >= 0``;
* ``contact_id >= 0``, ``floe_i >= 0``, ``floe_j >= 0`` or ``floe_j == HULL``;
* **energy budget** (physics-complete mode only), per contact:

      e_diss(t) = int F_n v_n dt - (e_spring(t) - e_spring(t0))

  with ``e_spring`` itself cross-checked against ``int F_n d(delta)``.  Both
  sides are trapezoidal reductions of the *exported* samples, so the check
  tests the export (rounding, decimation or a corrupted force column cannot
  pass it), not the solver.

Modes
-----
``physics_complete``  : the energy group is present and every check runs.
``statistics_only``   : no energy group (legacy/externally calibrated records);
                        only kinematic and force checks run and the report says
                        exactly which checks were skipped.

Fail-closed
-----------
An empty history raises :class:`MissingForceHistory`; there is no silent
zero-damage path anywhere in :mod:`tensordem.fatigue_chain`.

Sign conventions: ``delta`` is the overlap (``>= 0``), ``v_n`` is the relative
normal velocity of approach (``> 0`` while the pair is compressed), ``F_n`` the
compressive normal force, ``F_t`` the tangential force magnitude,
``patch_area`` the instantaneous contact patch area.  Boundary contacts (hull,
platen, wall) use ``floe_j = HULL = -1``.

This contract certifies that an exported record is *internally consistent*; it
does not certify the solver, which is what :mod:`tensordem.fatigue_chain.cases`
is for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "SCHEMA_ID",
    "CORE_COLUMNS",
    "ENERGY_COLUMNS",
    "HULL",
    "ForceHistoryError",
    "MissingForceHistory",
    "ContactForceHistory",
]

SCHEMA_ID = "tensordem.contact-force-history/1"
CORE_COLUMNS = ("t", "contact_id", "floe_i", "floe_j", "F_n", "F_t", "patch_area")
ENERGY_COLUMNS = ("delta", "v_n", "e_spring", "e_diss")
HULL = -1


class ForceHistoryError(ValueError):
    """The exported contact-force history violates the contract."""


class MissingForceHistory(ForceHistoryError):
    """Fail-closed: the fatigue chain was handed no DEM time history."""


def _trapz_cumulative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral, ``out[0] = 0``."""
    if y.size < 2:
        return np.zeros_like(y, dtype=float)
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.empty(y.size, dtype=float)
    out[0] = 0.0
    np.cumsum(seg, out=out[1:])
    return out



def _state_function_error(delta, energy, n_bins: int = 40) -> float:
    """Max within-bin scatter of ``energy`` after removing the local slope.

    ``e_spring`` must be a function of the overlap alone, so samples that share
    an overlap must share the stored energy.  Inside a ``delta`` bin a *valid*
    law still varies (linearly at leading order: ``dE = k delta d(delta)``),
    so each bin is detrended with a straight line and the largest residual
    spread is returned, in energy units.  A corrupted/rounded energy column
    keeps an O(1) residual; a genuine spring law leaves O(bin^2) curvature.
    """
    delta = np.asarray(delta, dtype=float)
    energy = np.asarray(energy, dtype=float)
    span = float(delta.max() - delta.min()) if delta.size else 0.0
    if span <= 0.0 or delta.size < 4:
        return 0.0
    bins = np.clip(((delta - delta.min()) / span * (n_bins - 1)).astype(int), 0, n_bins - 1)
    worst = 0.0
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < 3:
            continue
        x, y = delta[sel], energy[sel]
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        worst = max(worst, float(resid.max() - resid.min()))
    return worst


class ContactForceHistory:
    """Sparse table of contact-force samples, one row per (time, contact)."""

    def __init__(
        self,
        t,
        contact_id,
        floe_i,
        floe_j,
        F_n,
        F_t,
        patch_area,
        *,
        delta=None,
        v_n=None,
        e_spring=None,
        e_diss=None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.t = np.asarray(t, dtype=float).ravel()
        self.contact_id = np.asarray(contact_id, dtype=np.int64).ravel()
        self.floe_i = np.asarray(floe_i, dtype=np.int64).ravel()
        self.floe_j = np.asarray(floe_j, dtype=np.int64).ravel()
        self.F_n = np.asarray(F_n, dtype=float).ravel()
        self.F_t = np.asarray(F_t, dtype=float).ravel()
        self.patch_area = np.asarray(patch_area, dtype=float).ravel()

        energy = {"delta": delta, "v_n": v_n, "e_spring": e_spring, "e_diss": e_diss}
        present = [name for name, value in energy.items() if value is not None]
        if present and len(present) != len(energy):
            missing = sorted(set(energy) - set(present))
            raise ForceHistoryError(
                "the energy group must be complete or absent; missing "
                + ", ".join(missing)
            )
        self.delta = None if delta is None else np.asarray(delta, dtype=float).ravel()
        self.v_n = None if v_n is None else np.asarray(v_n, dtype=float).ravel()
        self.e_spring = None if e_spring is None else np.asarray(e_spring, dtype=float).ravel()
        self.e_diss = None if e_diss is None else np.asarray(e_diss, dtype=float).ravel()
        self.meta: Dict[str, Any] = dict(meta or {})

        sizes = {self.t.size, self.contact_id.size, self.floe_i.size, self.floe_j.size,
                 self.F_n.size, self.F_t.size, self.patch_area.size}
        for name in ENERGY_COLUMNS:
            value = getattr(self, name)
            if value is not None:
                sizes.add(value.size)
        if len(sizes) != 1:
            raise ForceHistoryError(f"column lengths differ: {sorted(sizes)}")

    # -- basic properties ---------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.t.size)

    @property
    def physics_complete(self) -> bool:
        return self.delta is not None

    @property
    def mode(self) -> str:
        return "physics_complete" if self.physics_complete else "statistics_only"

    def contact_ids(self) -> np.ndarray:
        return np.unique(self.contact_id)

    def n_contacts(self) -> int:
        return int(self.contact_ids().size)

    def contact_series(self, contact_id: int) -> Dict[str, np.ndarray]:
        """All columns of one contact, time-ordered."""
        mask = self.contact_id == contact_id
        order = np.argsort(self.t[mask], kind="stable")
        out = {
            "t": self.t[mask][order],
            "F_n": self.F_n[mask][order],
            "F_t": self.F_t[mask][order],
            "patch_area": self.patch_area[mask][order],
            "floe_i": self.floe_i[mask][order],
            "floe_j": self.floe_j[mask][order],
        }
        for name in ENERGY_COLUMNS:
            value = getattr(self, name)
            if value is not None:
                out[name] = value[mask][order]
        return out

    # -- validation ---------------------------------------------------------
    def validate(self, *, tol_energy: float = 2.0e-2) -> Dict[str, Any]:
        """Contract admissibility report (never raises on bad physics data)."""
        checks: list[Dict[str, Any]] = []

        def add(name: str, passed: bool, detail: Any, **extra: Any) -> None:
            checks.append({"name": name, "passed": bool(passed),
                           "detail": detail, **extra})

        add("non_empty", self.n_samples > 0, {"n_samples": self.n_samples})
        if self.n_samples == 0:
            return self._report(checks, admissible=False, tol_energy=tol_energy)

        arrays = {"t": self.t, "F_n": self.F_n, "F_t": self.F_t,
                  "patch_area": self.patch_area}
        if self.physics_complete:
            arrays.update({"delta": self.delta, "v_n": self.v_n,
                           "e_spring": self.e_spring, "e_diss": self.e_diss})
        bad = {name: int((~np.isfinite(arr)).sum()) for name, arr in arrays.items()}
        add("finite", sum(bad.values()) == 0, bad)

        add("F_n_non_negative", bool(np.all(self.F_n >= 0.0)),
            {"min_F_n": float(self.F_n.min())})
        add("F_t_non_negative", bool(np.all(self.F_t >= 0.0)),
            {"min_F_t": float(self.F_t.min())})
        add("patch_area_non_negative", bool(np.all(self.patch_area >= 0.0)),
            {"min_patch_area": float(self.patch_area.min())})
        add("ids_valid", bool(np.all(self.contact_id >= 0) and np.all(self.floe_i >= 0)
                              and np.all((self.floe_j >= 0) | (self.floe_j == HULL))),
            {"contact_id_min": int(self.contact_id.min()),
             "floe_j_min": int(self.floe_j.min())})

        # global time order (non-decreasing)
        global_dt = np.diff(self.t)
        add("t_global_non_decreasing", bool(np.all(global_dt >= 0.0)),
            {"min_dt": float(global_dt.min()) if global_dt.size else 0.0})

        # per-contact strictly increasing
        min_dt = np.inf
        n_dup = 0
        n_contacts = 0
        for cid in self.contact_ids():
            # the *stored* order is what a consumer sees: sorting here would
            # hide a non-monotone export (caught by the negative test)
            tt = self.t[self.contact_id == cid]
            n_contacts += 1
            if tt.size > 1:
                d = np.diff(tt)
                min_dt = min(min_dt, float(d.min()))
                n_dup += int((d <= 0.0).sum())
        add("t_strictly_increasing_per_contact", n_dup == 0,
            {"n_duplicate_or_reversed": n_dup,
             "min_dt": None if not np.isfinite(min_dt) else min_dt,
             "n_contacts": n_contacts})

        energy_detail: Dict[str, Any] = {"mode": self.mode}
        if self.physics_complete:
            add("delta_non_negative", bool(np.all(self.delta >= 0.0)),
                {"min_delta": float(self.delta.min())})
            # the tolerance is relative to the *record* scale (peak stored
            # contact energy), not to each contact: a grazing contact that
            # stores nano-joules cannot define the admissible error of the
            # export, and would make the check meaningless (measured: 1e-9 J
            # contacts would otherwise report >100 % relative error).
            global_scale = 0.0
            for cid in self.contact_ids():
                tt = self.contact_series(cid)["e_spring"]
                if tt.size:
                    global_scale = max(global_scale, float(np.max(np.abs(tt))))
            global_scale = global_scale if global_scale > 0.0 else 1.0
            negligible = 0
            worst_spring = 0.0
            worst_budget = 0.0
            spring_monotone = True
            for cid in self.contact_ids():
                s = self.contact_series(cid)
                if s["t"].size < 3:
                    continue
                local_scale = float(np.max(np.abs(s["e_spring"])))
                if local_scale < 1e-6 * global_scale:
                    negligible += 1
                    continue
                worst_spring = max(worst_spring,
                                   _state_function_error(s["delta"], s["e_spring"]) / local_scale)
                order = np.argsort(s["delta"], kind="stable")
                if np.any(np.diff(s["e_spring"][order]) < -1e-12 * local_scale):
                    spring_monotone = False
                work = _trapz_cumulative(s["F_n"] * s["v_n"], s["t"])
                diss_pred = work - (s["e_spring"] - s["e_spring"][0])
                diss_rec = s["e_diss"] - s["e_diss"][0]
                worst_budget = max(worst_budget,
                                   float(np.max(np.abs(diss_pred - diss_rec)) / global_scale))
            energy_detail.update({"max_rel_state_function_error": worst_spring,
                                  "max_rel_budget_error": worst_budget,
                                  "spring_monotone_in_delta": spring_monotone,
                                  "record_energy_scale_J": global_scale,
                                  "negligible_contacts_skipped": negligible,
                                  "tolerance": tol_energy})
            add("energy_spring_state_function",
                worst_spring <= tol_energy and spring_monotone,
                {"max_rel_error": worst_spring, "monotone_in_delta": spring_monotone,
                 "tolerance": tol_energy})
            add("energy_budget_consistent", worst_budget <= tol_energy,
                {"max_rel_error": worst_budget, "tolerance": tol_energy})
        else:
            energy_detail["skipped"] = ["delta_non_negative", "energy_spring_state_function",
                                        "energy_budget_consistent"]

        skipped = energy_detail.pop("skipped", [])
        return self._report(checks, admissible=all(c["passed"] for c in checks),
                            tol_energy=tol_energy, energy=energy_detail, skipped=skipped)

    def _report(self, checks, *, admissible: bool, tol_energy: float,
                energy: Optional[Dict[str, Any]] = None,
                skipped: Optional[list[str]] = None) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "mode": self.mode,
            "n_samples": self.n_samples,
            "n_contacts": self.n_contacts(),
            "t_span_s": [float(self.t.min()), float(self.t.max())] if self.n_samples else [0.0, 0.0],
            "admissible": admissible,
            "tolerance_energy": tol_energy,
            "energy": energy or {"mode": self.mode},
            "skipped_checks": list(skipped or []),
            "checks": checks,
            "meta": self.meta,
        }

    # -- aggregation to a uniform load channel ------------------------------
    def to_uniform(self, rate_hz: float, t_start: Optional[float] = None,
                   t_end: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Sum the normal forces of all contacts on a uniform time grid.

        Each contact contributes its own linearly interpolated ``F_n`` inside
        ``[t_first, t_last]``; the channel is the sum.  This is the
        contact-aggregated load channel the structural side consumes, and the
        only place where the sparse per-contact table becomes a time series.
        """
        if self.n_samples == 0:
            raise MissingForceHistory("empty contact-force history cannot be aggregated")
        if not (rate_hz > 0.0):
            raise ForceHistoryError("rate_hz must be positive")
        dt = 1.0 / float(rate_hz)
        t0 = float(self.t.min()) if t_start is None else float(t_start)
        t1 = float(self.t.max()) if t_end is None else float(t_end)
        if t1 <= t0:
            t1 = t0 + dt
        n = int(np.floor((t1 - t0) / dt + 1e-9)) + 1
        grid = t0 + dt * np.arange(n)
        total = np.zeros(n, dtype=float)
        for cid in self.contact_ids():
            s = self.contact_series(cid)
            ti, fi = s["t"], s["F_n"]
            if ti.size == 0:
                continue
            if ti.size == 1:
                j = int(np.argmin(np.abs(grid - ti[0])))
                total[j] += fi[0]
                continue
            lo, hi = max(ti[0], t0), min(ti[-1], t1)
            if hi <= lo:
                continue
            sel = (grid >= lo) & (grid <= hi)
            total[sel] += np.interp(grid[sel], ti, fi)
        return grid, total

    def statistics(self, rate_hz: Optional[float] = None) -> Dict[str, float]:
        """Peak / RMS / P95 / mean of the aggregated normal-force channel."""
        rate = rate_hz
        if rate is None:
            rate = float(self.meta.get("sampling_rate_hz", 1000.0))
        t, f = self.to_uniform(rate)
        return {
            "rate_hz": float(rate),
            "n_samples": int(t.size),
            "peak_N": float(f.max()),
            "rms_N": float(np.sqrt(np.mean(f * f))),
            "p95_N": float(np.percentile(f, 95.0)),
            "mean_N": float(np.mean(f)),
            "t_span_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        }

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        samples = []
        for i in range(self.n_samples):
            row = {
                "t": float(self.t[i]),
                "contact_id": int(self.contact_id[i]),
                "floe_i": int(self.floe_i[i]),
                "floe_j": int(self.floe_j[i]),
                "F_n": float(self.F_n[i]),
                "F_t": float(self.F_t[i]),
                "patch_area": float(self.patch_area[i]),
            }
            if self.physics_complete:
                for name in ENERGY_COLUMNS:
                    row[name] = float(getattr(self, name)[i])
            samples.append(row)
        return {"schema": SCHEMA_ID, "meta": self.meta, "samples": samples}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ContactForceHistory":
        if payload.get("schema") != SCHEMA_ID:
            raise ForceHistoryError(
                f"schema mismatch: expected {SCHEMA_ID!r}, got {payload.get('schema')!r}"
            )
        samples = payload.get("samples") or []
        if not samples:
            raise MissingForceHistory("payload carries no samples")
        cols: Dict[str, list] = {name: [] for name in CORE_COLUMNS}
        for name in ENERGY_COLUMNS:
            cols[name] = []
        for row in samples:
            for name in CORE_COLUMNS:
                cols[name].append(row[name])
            if all(name in row for name in ENERGY_COLUMNS):
                for name in ENERGY_COLUMNS:
                    cols[name].append(row[name])
        has_energy = len(cols["delta"]) == len(samples) and bool(samples)
        kwargs: Dict[str, Any] = {}
        if has_energy:
            for name in ENERGY_COLUMNS:
                kwargs[name] = cols[name]
        return cls(
            cols["t"], cols["contact_id"], cols["floe_i"], cols["floe_j"],
            cols["F_n"], cols["F_t"], cols["patch_area"],
            meta=payload.get("meta") or {}, **kwargs,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ContactForceHistory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
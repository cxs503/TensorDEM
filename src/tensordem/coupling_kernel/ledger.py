"""Energy-ledger hooks for the DEM coupling kernel.

The ledger is intentionally dumb: the kernel hands it *already integrated*
terms (joules) once per record, the ledger sums the mechanical channels, adds
the cumulative dissipative channels and reports the discrete balance residual

    residual = (K + U_g + U_c)(t) - (K + U_g + U_c)(0) + W_diss(t)

Itemised channels (the "三场" split used by the coupling chain):

* DEM bulk field: ``kinetic_J``, ``gravity_potential_J``, ``elastic_J``
  (particle–particle and particle–wall Hertz contact energy),
* contact/dissipation field: ``dem_damping_dissipated_J`` (particle–particle
  dashpots), ``wall_damping_dissipated_J`` (wall dashpots),
* fluid-coupling field: ``drag_dissipated_J`` (work against the drag law, i.e.
  the momentum the DEM leg hands to the flow leg).

External hooks: :meth:`EnergyLedger.attach` registers any callable that will be
handed each row (e.g. a CSV writer, a live monitor, a coupling-chain tap).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = ["EnergyLedger", "LEDGER_FIELDS"]

LEDGER_FIELDS = (
    "step",
    "time_s",
    "kinetic_J",
    "gravity_potential_J",
    "elastic_J",
    "dem_damping_dissipated_J",
    "wall_damping_dissipated_J",
    "drag_dissipated_J",
)

_DISSIPATIVE = (
    "dem_damping_dissipated_J",
    "wall_damping_dissipated_J",
    "drag_dissipated_J",
)


class EnergyLedger:
    """Time series of energy channels with closure diagnostics."""

    def __init__(self, baseline_mechanical_J: float | None = None) -> None:
        self.rows: list[dict[str, float]] = []
        self._hooks: list[Callable[[dict[str, float]], None]] = []
        self._baseline = baseline_mechanical_J

    # -- hooks ------------------------------------------------------------
    def attach(self, hook: Callable[[dict[str, float]], None]) -> None:
        self._hooks.append(hook)

    def detach_all(self) -> None:
        self._hooks.clear()

    # -- recording --------------------------------------------------------
    def record(self, step: int, time_s: float, **terms: float) -> dict[str, float]:
        missing = [k for k in _DISSIPATIVE if k not in terms]
        if missing:
            raise KeyError(f"ledger record missing dissipative channels: {missing}")
        mechanical = (
            float(terms["kinetic_J"])
            + float(terms["gravity_potential_J"])
            + float(terms["elastic_J"])
        )
        if self._baseline is None:
            self._baseline = mechanical
        dissipated = sum(float(terms[k]) for k in _DISSIPATIVE)
        residual = mechanical - self._baseline + dissipated
        row = {
            "step": int(step),
            "time_s": float(time_s),
            "kinetic_J": float(terms["kinetic_J"]),
            "gravity_potential_J": float(terms["gravity_potential_J"]),
            "elastic_J": float(terms["elastic_J"]),
            "dem_damping_dissipated_J": float(terms["dem_damping_dissipated_J"]),
            "wall_damping_dissipated_J": float(terms["wall_damping_dissipated_J"]),
            "drag_dissipated_J": float(terms["drag_dissipated_J"]),
            "mechanical_J": mechanical,
            "dissipated_J": dissipated,
            "balance_residual_J": residual,
            "relative_residual": residual / max(abs(self._baseline), 1e-300)
            if self._baseline
            else 0.0,
        }
        for extra_key, extra_val in terms.items():
            if extra_key not in row:
                row[extra_key] = float(extra_val)
        self.rows.append(row)
        for hook in self._hooks:
            hook(row)
        return row

    # -- outputs ----------------------------------------------------------
    @property
    def baseline_mechanical_J(self) -> float:
        return 0.0 if self._baseline is None else float(self._baseline)

    def latest(self) -> dict[str, float]:
        if not self.rows:
            raise RuntimeError("ledger is empty")
        return self.rows[-1]

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"records": 0, "baseline_mechanical_J": self.baseline_mechanical_J}
        last = self.rows[-1]
        worst = max(self.rows, key=lambda r: abs(r["relative_residual"]))
        return {
            "records": len(self.rows),
            "baseline_mechanical_J": self.baseline_mechanical_J,
            "final_kinetic_J": last["kinetic_J"],
            "final_gravity_potential_J": last["gravity_potential_J"],
            "final_elastic_J": last["elastic_J"],
            "dem_damping_dissipated_J": last["dem_damping_dissipated_J"],
            "wall_damping_dissipated_J": last["wall_damping_dissipated_J"],
            "drag_dissipated_J": last["drag_dissipated_J"],
            "final_balance_residual_J": last["balance_residual_J"],
            "final_relative_residual": last["relative_residual"],
            "max_abs_relative_residual": abs(worst["relative_residual"]),
            "max_abs_relative_residual_step": worst["step"],
        }

    def to_json(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "rows": self.rows}
        if extra:
            payload.update(extra)
        path.write_text(json.dumps(payload, indent=2))
        return path

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(self.rows[0].keys()) if self.rows else list(LEDGER_FIELDS)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        return path

    def __iter__(self) -> Iterable[dict[str, float]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)
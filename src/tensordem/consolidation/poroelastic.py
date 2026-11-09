"""Layered two-field pore-pressure / skeleton column (C_v is *assembled*).

The solver integrates the two constitutive statements of Terzaghi theory
directly:

* **fluid phase** -- Darcy flux between cell faces,
  ``q = -(k/gamma_w) du/dz`` (``u`` = excess pore pressure, gravity drainage);
* **skeleton** -- a quasi-static, linear oedometer law per cell,
  ``eps_i = (d_sigma - u_i) / E_oed,i`` (total stress increment ``d_sigma``
  constant in time); fluid mass balance in a saturated, grain-incompressible
  bed gives ``d eps_i/dt = (q_{i-1/2} - q_{i+1/2})/dz``.

Eliminating ``eps`` recovers ``du/dt = C_v d2u/dz2`` with
``C_v = k E_oed / gamma_w`` -- but ``C_v`` is **never an input of this class**:
it is assembled from ``k`` and ``E_oed`` (each of which has an independent,
micro-mechanical provenance) and is only reported afterwards.  That is the
whole point of the layer: the consolidation curve must be *predicted* by the
two measured micro-parameters, not drawn from the answer.

Discretisation: cell-centred finite volume, harmonic-mean face conductivity,
explicit forward Euler with a stability-driver time step, drained Dirichlet
face (``u = 0``) and impermeable Neumann face.  Two-way drainage is solved on
one half of the bed (``H_dr = H/2``) with a symmetry Neumann face and the
settlement doubled; one-way drainage uses ``H_dr = H``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["ColumnParameters", "PoroelasticColumn", "WATER_UNIT_WEIGHT", "WATER_VISCOSITY"]

WATER_UNIT_WEIGHT = 9810.0     # N/m^3 at 20 degC (rho_w g)
WATER_VISCOSITY = 1.0e-3       # Pa s at 20 degC


@dataclass(frozen=True)
class ColumnParameters:
    """Physical set-up of the column (SI)."""

    height: float                 # full bed height H (m)
    stress_step: float            # step surcharge d_sigma (Pa)
    drainage: str = "double"      # "double" (two-way) or "single"
    gamma_w: float = WATER_UNIT_WEIGHT

    def __post_init__(self) -> None:
        if not (self.height > 0.0 and math.isfinite(self.height)):
            raise ValueError("height must be finite and positive")
        if not (self.stress_step > 0.0 and math.isfinite(self.stress_step)):
            raise ValueError("stress_step must be finite and positive")
        if self.drainage not in ("double", "single"):
            raise ValueError("drainage must be 'double' or 'single'")
        if not (self.gamma_w > 0.0):
            raise ValueError("gamma_w must be positive")

    @property
    def drainage_path(self) -> float:
        return 0.5 * self.height if self.drainage == "double" else self.height

    @property
    def settlement_multiplier(self) -> float:
        """2 for the half-bed symmetry solve, 1 otherwise."""
        return 2.0 if self.drainage == "double" else 1.0


class PoroelasticColumn:
    """Explicit two-field column: ``k`` and ``E_oed`` in, curves out."""

    def __init__(
        self,
        *,
        params: ColumnParameters,
        k_hyd,
        E_oed,
        n_cells: int = 40,
        mu_w: float = WATER_VISCOSITY,
        safety: float = 0.40,
    ) -> None:
        if n_cells < 2:
            raise ValueError("n_cells must be >= 2")
        self.params = params
        self.mu_w = float(mu_w)
        self.n_cells = int(n_cells)
        self.h_dr = params.drainage_path
        self.dz = self.h_dr / self.n_cells
        self.k = self._as_cells(k_hyd, "k_hyd")
        self.E = self._as_cells(E_oed, "E_oed")
        if np.any(self.k <= 0.0) or np.any(self.E <= 0.0):
            raise ValueError("k_hyd and E_oed must be positive")

        # harmonic-mean conductivity between neighbouring cells; the drained
        # face is handled through the ghost value u_ghost = -u_0 (Dirichlet 0)
        # and the impermeable face by a zero flux at the far boundary.
        self.k_face = np.empty(self.n_cells + 1)
        self.k_face[1:-1] = 2.0 * self.k[:-1] * self.k[1:] / (self.k[:-1] + self.k[1:])
        self.k_face[0] = self.k[0]
        self.k_face[-1] = 0.0  # impermeable far face

        self.u = np.full(self.n_cells, params.stress_step)
        self._cv_cell = self.k * self.E / params.gamma_w
        self.dt = safety * self.dz ** 2 * params.gamma_w / float(np.max(self.k * self.E))
        self.t = 0.0
        self.steps = 0
        self._history: dict[str, list[float]] = {
            "t": [], "U": [], "settlement": [], "r_u_average": [],
        }

    # -- helpers ---------------------------------------------------------
    def _as_cells(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        arr = np.broadcast_to(arr, (self.n_cells,)).copy() if arr.ndim == 0 else arr.copy()
        if arr.shape != (self.n_cells,):
            raise ValueError(f"{name} must be scalar or have n_cells entries")
        return arr

    def face_flux(self) -> np.ndarray:
        """Darcy specific discharge (m/s, positive upward) on all faces."""
        du = np.empty(self.n_cells + 1)
        du[1:-1] = self.u[1:] - self.u[:-1]
        du[0] = 2.0 * self.u[0]           # drained face: gradient over dz/2
        du[-1] = 0.0                      # impermeable face
        return -(self.k_face / self.params.gamma_w) * du / self.dz

    def strain(self) -> np.ndarray:
        """Skeleton strain per cell, ``(d_sigma - u)/E_oed`` (compression +)."""
        return (self.params.stress_step - self.u) / self.E

    def settlement(self) -> float:
        """Current surface settlement over the *full* bed height (m)."""
        return float(np.sum(self.strain() * self.dz) * self.params.settlement_multiplier)

    @property
    def settlement_final(self) -> float:
        """Analytic 1-D compression of the same column, ``d_sigma H / E_oed``."""
        return float(np.sum(self.dz * self.params.stress_step / self.E)
                     * self.params.settlement_multiplier)

    def U(self) -> float:
        return self.settlement() / self.settlement_final

    def r_u_average(self) -> float:
        """Depth-averaged excess pore-pressure ratio, ``1 - U``."""
        return 1.0 - self.U()

    def cv_assembled(self) -> float:
        """``C_v = k E_oed / gamma_w`` (diagnostic; not an input)."""
        return float(np.mean(self._cv_cell))

    def Tv(self) -> float:
        return self.t * self.cv_assembled() / self.h_dr ** 2

    def time_of(self, Tv: float) -> float:
        """Physical time at a target ``T_v`` (using the assembled ``C_v``)."""
        return float(Tv) * self.h_dr ** 2 / self.cv_assembled()

    def snapshot(self) -> dict:
        z = (np.arange(self.n_cells) + 0.5) * self.dz
        return {
            "t": self.t,
            "T_v": self.Tv(),
            "U": self.U(),
            "settlement": self.settlement(),
            "z_over_hdr": (z / self.h_dr).tolist(),
            "u_excess": self.u.copy().tolist(),
            "r_u": (self.u / self.params.stress_step).tolist(),
            "strain": self.strain().tolist(),
            "flux": self.face_flux().tolist(),
        }

    # -- integration -----------------------------------------------------
    def step(self) -> None:
        q = self.face_flux()
        du = -self.E * (q[1:] - q[:-1]) / self.dz
        self.u = self.u + du * self.dt
        self.u = np.minimum(self.u, self.params.stress_step)  # never exceed the step
        self.t += self.dt
        self.steps += 1
        h = self._history
        h["t"].append(self.t)
        h["U"].append(self.U())
        h["settlement"].append(self.settlement())
        h["r_u_average"].append(self.r_u_average())

    def advance_to(self, t_end: float, *, max_steps: int = 20_000_000) -> None:
        target = float(t_end)
        guard = 0
        while self.t < target - 1e-15:
            dt_save = self.dt
            if self.t + self.dt > target:
                self.dt = target - self.t
            self.step()
            self.dt = dt_save
            guard += 1
            if guard > max_steps:
                raise RuntimeError("advance_to exceeded max_steps")

    def run(self, Tv_points: Iterable[float]) -> dict:
        """March to every requested ``T_v`` and return the sampled history."""
        points = sorted(float(x) for x in Tv_points)
        snapshots, rows = [], []
        h = self._history
        for Tv in points:
            self.advance_to(self.time_of(Tv))
            snap = self.snapshot()
            snapshots.append(snap)
            rows.append({
                "T_v": snap["T_v"],
                "t": snap["t"],
                "U_simulated": snap["U"],
                "settlement_simulated": snap["settlement"],
                "r_u_average_simulated": 1.0 - snap["U"],
            })
        hist = {
            "t": np.asarray(h["t"]),
            "U": np.asarray(h["U"]),
            "settlement": np.asarray(h["settlement"]),
            "r_u_average": np.asarray(h["r_u_average"]),
        }
        return {
            "parameters": {
                "height": self.params.height,
                "drainage": self.params.drainage,
                "drainage_path": self.h_dr,
                "stress_step": self.params.stress_step,
                "gamma_w": self.params.gamma_w,
                "mu_w": self.mu_w,
                "n_cells": self.n_cells,
                "dz": self.dz,
                "dt": self.dt,
                "k_hyd": self.k.tolist(),
                "E_oed": self.E.tolist(),
            },
            "C_v_assembled": self.cv_assembled(),
            "settlement_final": self.settlement_final,
            "samples": rows,
            "snapshots": snapshots,
            "history": hist,
        }
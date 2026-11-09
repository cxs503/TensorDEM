"""Ice-belt plate-panel check under the DEM-derived ridge load.

The extreme ridge load on the ship's ice belt is transmitted to a plate panel
bounded by frames (transverse), stringers (longitudinal) and, at the panel
edge, by the shell/deck.  This module provides the structural end of the
ridge-interaction chain:

    DEM crushing pressure  ->  patch load on the panel  ->  sigma_max, delta_max
                                                       ->  yield check

Element formulation
-------------------
A non-conforming but C1-continuous rectangular plate element is used: the
bicubic Hermite (Bogner-Fox-Schmit, BFS) element with the 16 element degrees of
freedom ``(w, w_x, w_y, w_xy)`` at each of the four corner nodes.  It is exact
for the Kirchhoff kinematics within a bicubic field, integrates the bending
energy ``0.5 * kappa^T D_b kappa`` with 3x3 Gauss quadrature (exact for the
degree-5 integrands of the element), and reproduces the classical clamped-plate
benchmarks to well under one percent on an 8x8 mesh (see
``tests/test_myi_ridge_panel.py``).

Clamped edges are represented exactly by the Hermite degrees of freedom
(``w = w_x = w_y = 0`` on the edge), so the clamped benchmarks are the primary
accuracy evidence.  Simply supported edges use the standard Hermite "SS1"
restriction ``w = 0, w_n = 0`` and are reported with the deviation from the
series solution.

Scope: linear elastic, isotropic, small deflection, no stiffener torsion, no
in-plane membrane action; the panel is checked against a bending-stress yield
criterion with a material safety factor.  Production structural models belong
to TensorFEM; this module is the verified panel-level adapter used here so the
ridge chain closes inside one reproducible worktree.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "PanelResult",
    "plate_rigidity",
    "solve_panel",
    "required_thickness",
    "check_ice_belt",
    "STEEL_EH36",
]

#: Elastic and strength constants for EH36 hull structural steel.
STEEL_EH36 = {
    "young_modulus": 206.0e9,
    "poisson_ratio": 0.30,
    "yield_strength": 355.0e6,
    "density": 7850.0,
}

_GAUSS_POINTS = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
_GAUSS_WEIGHTS = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def _shift(s: float) -> float:
    """Map the 3-point Gauss abscissa on [-1, 1] to the element [0, 1]."""
    return 0.5 * (s + 1.0)


def _hermite(index: int, s: float) -> tuple[float, float, float]:
    """Value, first and second derivative of the 1-D Hermite basis."""
    if index == 0:
        return 1.0 - 3.0 * s ** 2 + 2.0 * s ** 3, -6.0 * s + 6.0 * s ** 2, -6.0 + 12.0 * s
    if index == 1:
        return s - 2.0 * s ** 2 + s ** 3, 1.0 - 4.0 * s + 3.0 * s ** 2, -4.0 + 6.0 * s
    if index == 2:
        return 3.0 * s ** 2 - 2.0 * s ** 3, 6.0 * s - 6.0 * s ** 2, 6.0 - 12.0 * s
    if index == 3:
        return -s ** 2 + s ** 3, -2.0 * s + 3.0 * s ** 2, -2.0 + 6.0 * s
    raise ValueError("Hermite index must lie in 0..3")


def plate_rigidity(young_modulus: float, thickness: float,
                   poisson_ratio: float) -> float:
    """Plate bending rigidity ``D = E t^3 / (12 (1 - nu^2))`` [N m]."""
    for name, value in (("young_modulus", young_modulus), ("thickness", thickness)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    return young_modulus * thickness ** 3 / (12.0 * (1.0 - poisson_ratio ** 2))


@dataclass
class PanelResult:
    """Result of one linear-elastic plate-panel solve."""

    deflection_max: float
    deflection_centre: float
    stress_x_max: float
    stress_y_max: float
    stress_von_mises_max: float
    moment_x_max: float
    moment_y_max: float
    load_total: float
    nodes_x: int
    nodes_y: int
    thickness: float
    pressure: float

    @property
    def stress_max(self) -> float:
        return max(abs(self.stress_x_max), abs(self.stress_y_max))

    def as_dict(self) -> dict:
        return {
            "deflection_max": self.deflection_max,
            "deflection_centre": self.deflection_centre,
            "stress_x_max": self.stress_x_max,
            "stress_y_max": self.stress_y_max,
            "stress_von_mises_max": self.stress_von_mises_max,
            "moment_x_max": self.moment_x_max,
            "moment_y_max": self.moment_y_max,
            "stress_max": self.stress_max,
            "load_total": self.load_total,
            "nodes_x": self.nodes_x,
            "nodes_y": self.nodes_y,
            "thickness": self.thickness,
            "pressure": self.pressure,
        }


def _element_matrices(ax: float, by: float, db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Element stiffness (16x16) and unit-pressure load vector (16,)."""
    ke = np.zeros((16, 16))
    fe = np.zeros(16)
    for gp, weight_s in zip(_GAUSS_POINTS, _GAUSS_WEIGHTS):
        s = _shift(gp)
        for ht, weight_t in zip(_GAUSS_POINTS, _GAUSS_WEIGHTS):
            t = _shift(ht)
            weight = weight_s * weight_t * 0.25  # d(s) d(t) = 0.5 * 0.5
            values = np.zeros(16)
            curv = np.zeros((3, 16))
            for a in range(4):
                ha, dha, ddha = _hermite(a, s)
                fs = ax if a % 2 else 1.0
                for b in range(4):
                    hb, dhb, ddhb = _hermite(b, t)
                    ft = by if b % 2 else 1.0
                    k = a * 4 + b
                    values[k] = fs * ha * ft * hb
                    curv[0, k] = fs * ddha * ft * hb / ax ** 2
                    curv[1, k] = fs * ha * ft * ddhb / by ** 2
                    curv[2, k] = 2.0 * fs * dha * ft * dhb / (ax * by)
            ke += weight * (curv.T @ db @ curv) * ax * by
            fe += weight * values * ax * by
    return ke, fe


def _dof_index(node_i: int, node_j: int, dof: int, nodes_x: int) -> int:
    return 4 * (node_j * nodes_x + node_i) + dof


def solve_panel(span_x: float, span_y: float, thickness: float,
                pressure: float, *, elements_x: int = 12, elements_y: int = 12,
                boundary: str = "clamped",
                young_modulus: float = STEEL_EH36["young_modulus"],
                poisson_ratio: float = STEEL_EH36["poisson_ratio"],
                patch: tuple[float, float, float, float] | None = None) -> PanelResult:
    """Static Kirchhoff plate solve for a rectangular panel under pressure.

    ``patch`` is an optional ``(x0, y0, x1, y1)`` rectangle loaded by
    ``pressure``; without it the whole panel is loaded.  ``boundary`` is
    ``"clamped"`` or ``"simply_supported"``.
    """
    if boundary not in ("clamped", "simply_supported"):
        raise ValueError("boundary must be 'clamped' or 'simply_supported'")
    for name, value in (("span_x", span_x), ("span_y", span_y)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if not math.isfinite(pressure):
        raise ValueError("pressure must be finite")
    if elements_x < 1 or elements_y < 1:
        raise ValueError("at least one element per direction is required")
    if patch is not None:
        x0, y0, x1, y1 = (float(v) for v in patch)
        if not (0.0 <= x0 < x1 <= span_x and 0.0 <= y0 < y1 <= span_y):
            raise ValueError("patch must lie inside the panel")

    d0 = plate_rigidity(young_modulus, thickness, poisson_ratio)
    db = d0 * np.array([[1.0, poisson_ratio, 0.0],
                        [poisson_ratio, 1.0, 0.0],
                        [0.0, 0.0, 0.5 * (1.0 - poisson_ratio)]])
    nodes_x = elements_x + 1
    nodes_y = elements_y + 1
    ndof = 4 * nodes_x * nodes_y
    stiffness = np.zeros((ndof, ndof))
    load = np.zeros(ndof)
    ax = span_x / elements_x
    by = span_y / elements_y
    ke, fe = _element_matrices(ax, by, db)

    loaded_area = 0.0
    for j in range(elements_y):
        for i in range(elements_x):
            rows = np.array([[a * 4 + b,
                              _dof_index(i + (1 if a >= 2 else 0),
                                         j + (1 if b >= 2 else 0),
                                         (a % 2) + 2 * (b % 2), nodes_x)]
                             for a in range(4) for b in range(4)])
            index = rows[:, 1].astype(int)
            stiffness[np.ix_(index, index)] += ke
            centre_x = (i + 0.5) * ax
            centre_y = (j + 0.5) * by
            inside = True
            if patch is not None:
                x0, y0, x1, y1 = patch
                inside = x0 <= centre_x <= x1 and y0 <= centre_y <= y1
            if inside:
                load[index] += fe * pressure
                loaded_area += ax * by

    edge_nodes = set()
    for k in range(nodes_x):
        edge_nodes.add(0 * nodes_x + k)
        edge_nodes.add((nodes_y - 1) * nodes_x + k)
    for k in range(nodes_y):
        edge_nodes.add(k * nodes_x)
        edge_nodes.add(k * nodes_x + nodes_x - 1)
    fixed = []
    if boundary == "clamped":
        for node in edge_nodes:
            fixed.extend([4 * node + d for d in range(4)])
    else:
        for node in edge_nodes:
            fixed.append(4 * node)  # w = 0 everywhere on the edge
        # a straight edge with w = 0 has zero tangential slope: that is the
        # kinematic part of simple support.  The normal slope stays free, so
        # the edge can rotate and the moment free condition acts naturally.
        for k in range(nodes_x):  # y-edges: tangential = x
            fixed.append(4 * (0 * nodes_x + k) + 1)
            fixed.append(4 * ((nodes_y - 1) * nodes_x + k) + 1)
        for k in range(nodes_y):  # x-edges: tangential = y
            fixed.append(4 * (k * nodes_x) + 2)
            fixed.append(4 * (k * nodes_x + nodes_x - 1) + 2)

    free = np.ones(ndof, dtype=bool)
    free[np.unique(np.asarray(fixed, dtype=int))] = False
    reduced = stiffness[np.ix_(free, free)]
    solution = np.zeros(ndof)
    solution[free] = np.linalg.solve(reduced, load[free])

    # recover curvatures / moments at element nodes and Gauss points
    deflection = np.zeros((nodes_y, nodes_x))
    for j in range(nodes_y):
        for i in range(nodes_x):
            deflection[j, i] = solution[_dof_index(i, j, 0, nodes_x)]
    stress_x = 0.0
    stress_y = 0.0
    vm = 0.0
    moment_x = 0.0
    moment_y = 0.0
    for j in range(elements_y):
        for i in range(elements_x):
            rows = np.array([[a * 4 + b,
                              _dof_index(i + (1 if a >= 2 else 0),
                                         j + (1 if b >= 2 else 0),
                                         (a % 2) + 2 * (b % 2), nodes_x)]
                             for a in range(4) for b in range(4)])
            local = solution[rows[:, 1].astype(int)]
            samples = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
                       (0.5, 0.5)]
            samples += [(_shift(gp), _shift(ht)) for gp in _GAUSS_POINTS
                        for ht in _GAUSS_POINTS]
            for s, t in samples:
                if True:
                    curv = np.zeros(3)
                    for a in range(4):
                        ha, dha, ddha = _hermite(a, s)
                        fs = ax if a % 2 else 1.0
                        for b in range(4):
                            hb, dhb, ddhb = _hermite(b, t)
                            ft = by if b % 2 else 1.0
                            value = local[a * 4 + b]
                            curv[0] += value * fs * ddha * ft * hb / ax ** 2
                            curv[1] += value * fs * ha * ft * ddhb / by ** 2
                            curv[2] += value * 2.0 * fs * dha * ft * dhb / (ax * by)
                    mx = -d0 * (curv[0] + poisson_ratio * curv[1])
                    my = -d0 * (curv[1] + poisson_ratio * curv[0])
                    sx = 6.0 * mx / thickness ** 2
                    sy = 6.0 * my / thickness ** 2
                    if abs(sx) > abs(stress_x):
                        stress_x = sx
                    if abs(sy) > abs(stress_y):
                        stress_y = sy
                    if abs(sx) > abs(moment_x):
                        moment_x = mx
                    if abs(sy) > abs(moment_y):
                        moment_y = my
                    vm = max(vm, abs(sx) ** 2 - sx * sy + sy ** 2)

    centre_i = nodes_x // 2 if nodes_x % 2 else nodes_x // 2
    centre_j = nodes_y // 2 if nodes_y % 2 else nodes_y // 2
    return PanelResult(
        deflection_max=float(np.abs(deflection).max()),
        deflection_centre=float(deflection[centre_j, centre_i]),
        stress_x_max=float(stress_x),
        stress_y_max=float(stress_y),
        stress_von_mises_max=float(math.sqrt(vm)) if vm > 0.0 else 0.0,
        moment_x_max=float(moment_x),
        moment_y_max=float(moment_y),
        load_total=float(pressure * loaded_area),
        nodes_x=nodes_x,
        nodes_y=nodes_y,
        thickness=float(thickness),
        pressure=float(pressure),
    )


def _dof_index_idx(node: int, dof: int) -> int:
    return 4 * int(node) + int(dof)


def required_thickness(span_x: float, span_y: float, pressure: float,
                       *, allowable_stress: float, reference_thickness: float = 0.025,
                       elements_x: int = 12, elements_y: int = 12,
                       boundary: str = "clamped",
                       patch: tuple[float, float, float, float] | None = None,
                       young_modulus: float = STEEL_EH36["young_modulus"],
                       poisson_ratio: float = STEEL_EH36["poisson_ratio"]) -> float:
    """Thickness for which the peak bending stress equals the allowable value.

    For Kirchhoff plates with a fixed geometry and pressure the bending moment
    is independent of the thickness while ``sigma = 6 M / t^2``, so the required
    thickness follows from one reference solve in closed form.  The returned
    value is verified by a second solve in the campaign checks.
    """
    if not math.isfinite(allowable_stress) or allowable_stress <= 0.0:
        raise ValueError("allowable_stress must be positive")
    reference = solve_panel(span_x, span_y, reference_thickness, pressure,
                            elements_x=elements_x, elements_y=elements_y,
                            boundary=boundary, patch=patch,
                            young_modulus=young_modulus,
                            poisson_ratio=poisson_ratio)
    if reference.stress_max <= 0.0:
        raise ValueError("reference solve produced no stress")
    return reference_thickness * math.sqrt(reference.stress_max / allowable_stress)


def check_ice_belt(pressure: float, *, span_x: float, span_y: float,
                   thickness: float, yield_strength: float = STEEL_EH36["yield_strength"],
                   safety_factor: float = 1.15, elements_x: int = 8,
                   elements_y: int = 8, boundary: str = "clamped",
                   patch: tuple[float, float, float, float] | None = None,
                   young_modulus: float = STEEL_EH36["young_modulus"],
                   poisson_ratio: float = STEEL_EH36["poisson_ratio"]) -> dict:
    """Structural acceptance check of the ice-belt panel.

    Acceptance: ``sigma_max < sigma_yield / safety_factor`` (the criterion of
    the ridge-interaction campaign).  Utilisation and the thickness that would
    satisfy the criterion are reported alongside the raw field maxima.
    """
    if not math.isfinite(safety_factor) or safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive")
    result = solve_panel(span_x, span_y, thickness, pressure,
                         elements_x=elements_x, elements_y=elements_y,
                         boundary=boundary, patch=patch,
                         young_modulus=young_modulus, poisson_ratio=poisson_ratio)
    allowable = yield_strength / safety_factor
    utilisation = result.stress_max / allowable
    thickness_required = thickness * math.sqrt(max(utilisation, 1e-30))
    return {
        **result.as_dict(),
        "yield_strength": yield_strength,
        "safety_factor": safety_factor,
        "allowable_stress": allowable,
        "utilisation": utilisation,
        "margin": allowable / result.stress_max if result.stress_max > 0 else math.inf,
        "pass": bool(result.stress_max < allowable),
        "thickness_required": thickness_required,
    }
"""Beam on a Kerr (three-parameter) elastic foundation: analytic + mixed FEM.

Physical model (ice-ridge keel grounding, W5-N03)
-------------------------------------------------
A grounded ridge keel (or the ship bottom girder resting on the grounded keel)
is idealised as a slender elastic beam of bending stiffness ``EI`` on a Kerr
foundation: an upper spring layer ``k_u`` under the beam, a shear layer ``g``,
and a lower spring layer ``k_l`` on a rigid base.  The three-parameter Kerr
model reproduces the shear coupling of a *rubble* foundation, which a single
parameter Winkler bed cannot.  The ship contact is a downward point load ``P``
at ``x_p``; scanning ``x_p`` gives the moving-load envelope.

Governing equations (``w`` positive downward, EI w'''' = q)
----------------------------------------------------------
* beam:      ``EI w'''' + p = q``
* Kerr:      ``alpha p - beta p'' = k_u w - k_u beta w''``
  with ``alpha = 1 + k_u/k_l`` and ``beta = g/k_l`` (derived by eliminating the
  shear-layer displacement ``w_s`` from ``p = k_u (w - w_s)`` and
  ``p = k_l w_s - g w_s''``).
* elimination gives the homogeneous 6th-order ODE
  ``beta EI w'''''' - alpha EI w'''' + k_u beta w'' - k_u w = 0``.

Two independent solution paths are provided:

* :func:`kerr_beam_analytic` -- exact state-space solution: the six roots of the
  characteristic polynomial are found from the cubic in ``mu = lambda^2``; the
  six coefficients are fixed by the symmetry conditions at the load and the free
  end conditions (``w''=w'''=w_s'=0``).
* :func:`kerr_beam_fem` -- mixed finite element from the potential energy
  ``Pi = 1/2∫EI w''^2 + 1/2∫k_u (w-w_s)^2 + 1/2∫k_l w_s^2
  + 1/2∫g w_s'^2 - P w(x_p)``, Hermite ``w`` / linear ``w_s``, Gauss quadrature.
  Both paths solve the *same* variational problem, so their difference is pure
  discretisation error and is measured by :func:`compare_analytic_fem`.

Units: SI, per metre of strip width.  ``k_u``, ``k_l`` in N/m^3, ``g`` in N,
``EI`` in N m^2, ``P`` in N/m, ``w`` in m, ``p`` in N/m^2.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

KERR_SCHEMA = "tensordem.kerr-beam/1"


@dataclass(frozen=True)
class KerrFoundation:
    """Three-parameter Kerr foundation (per metre of strip width)."""

    upper: float
    lower: float
    shear: float = 0.0

    def __post_init__(self) -> None:
        for name in ("upper", "lower"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.shear) or self.shear < 0.0:
            raise ValueError("shear must be finite and nonnegative")

    @property
    def equivalent_winkler(self) -> float:
        """Series-combined Winkler modulus ``k_u k_l / (k_u + k_l)`` [N/m^3]."""
        return self.upper * self.lower / (self.upper + self.lower)

    def as_dict(self) -> dict[str, float]:
        return {
            "upper_N_per_m3": self.upper,
            "lower_N_per_m3": self.lower,
            "shear_N": self.shear,
            "equivalent_winkler_N_per_m3": self.equivalent_winkler,
        }


@dataclass(frozen=True)
class BeamSection:
    """Rectangular unit-width beam section."""

    young_modulus: float
    thickness: float
    width: float = 1.0

    def __post_init__(self) -> None:
        for name in ("young_modulus", "thickness", "width"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def inertia(self) -> float:
        return self.width * self.thickness**3 / 12.0

    @property
    def ei(self) -> float:
        return self.young_modulus * self.inertia

    @property
    def area(self) -> float:
        return self.width * self.thickness


def characteristic_length(ei: float, foundation: KerrFoundation) -> float:
    """Winkler characteristic length ``l = (4 EI / k_eq)^(1/4)`` [m]."""
    if not math.isfinite(ei) or ei <= 0.0:
        raise ValueError("ei must be finite and positive")
    return (4.0 * ei / foundation.equivalent_winkler) ** 0.25


@dataclass(frozen=True)
class KerrBeamSolution:
    """Sampled solution of a beam on a Kerr foundation."""

    x: np.ndarray
    deflection: np.ndarray       # w [m], positive downward
    pressure: np.ndarray         # contact pressure p [N/m^2], positive downward
    shear_layer: np.ndarray      # w_s [m]
    moment: np.ndarray           # M = -EI w'' [N m/m]
    shear: np.ndarray            # V = -EI w''' [N/m]
    load: float
    load_position: float
    length: float
    ei: float
    foundation: KerrFoundation
    method: str

    @property
    def max_deflection(self) -> float:
        return float(np.max(np.abs(self.deflection)))

    @property
    def max_pressure(self) -> float:
        return float(np.max(self.pressure))

    @property
    def total_foundation_force(self) -> float:
        """Integral of the contact pressure along the beam [N/m]."""
        return float(np.trapezoid(self.pressure, self.x)) if hasattr(np, "trapezoid") else float(
            np.trapz(self.pressure, self.x)
        )

    def as_dict(self, with_profiles: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": KERR_SCHEMA,
            "method": self.method,
            "length_m": self.length,
            "ei_Nm2": self.ei,
            "load_N_per_m": self.load,
            "load_position_m": self.load_position,
            "foundation": self.foundation.as_dict(),
            "max_deflection_m": self.max_deflection,
            "max_pressure_Pa": self.max_pressure,
            "total_foundation_force_N_per_m": self.total_foundation_force,
        }
        if with_profiles:
            data["profiles"] = {
                "x_m": self.x.tolist(),
                "deflection_m": self.deflection.tolist(),
                "pressure_Pa": self.pressure.tolist(),
                "shear_layer_m": self.shear_layer.tolist(),
                "moment_Nm_per_m": self.moment.tolist(),
            }
        return data


def kerr_characteristic_roots(ei: float, foundation: KerrFoundation) -> np.ndarray:
    """Roots of the homogeneous characteristic polynomial.

    ``beta EI l^6 - alpha EI l^4 + k_u beta l^2 - k_u = 0`` is even in ``l``, so
    ``mu = l^2`` solves a cubic; each ``mu`` gives the pair ``+/-sqrt(mu)``.
    For ``g = 0`` the fourth-order Winkler problem is solved instead
    (``l^4 = -k_eq / EI``).
    """
    if not math.isfinite(ei) or ei <= 0.0:
        raise ValueError("ei must be finite and positive")
    ku, kl, g = foundation.upper, foundation.lower, foundation.shear
    if g <= 0.0:
        k_eq = foundation.equivalent_winkler
        mag = (k_eq / ei) ** 0.25
        return np.array(
            [
                mag * (1 + 1j) / math.sqrt(2),
                mag * (-1 + 1j) / math.sqrt(2),
                mag * (-1 - 1j) / math.sqrt(2),
                mag * (1 - 1j) / math.sqrt(2),
            ],
            dtype=np.complex128,
        )
    alpha = 1.0 + ku / kl
    beta = g / kl
    coeffs = [1.0, -alpha / beta, ku / ei, -ku / (beta * ei)]
    roots: list[complex] = []
    for mu in np.roots(coeffs):
        s = np.sqrt(complex(mu))
        roots.extend((s, -s))
    return np.array(roots, dtype=np.complex128)


def _derivatives(x: np.ndarray, roots: np.ndarray, order: int) -> np.ndarray:
    return np.exp(np.outer(np.asarray(x, dtype=np.float64), roots)) * (roots[None, :] ** order)


def kerr_beam_analytic(
    length: float,
    ei: float,
    foundation: KerrFoundation,
    load: float,
    load_position: float | None = None,
    n_points: int = 401,
) -> KerrBeamSolution:
    """State-space (semi-analytic) solution, load at mid-span.

    Half-domain ``[0, L/2]`` with the load at ``x = 0`` using the symmetry of the
    problem, then mirrored:

    * ``w'(0) = 0``, ``w_s'(0) = 0`` (symmetry),
    * ``EI w'''(0) = P/2`` (shear jump from the point load),
    * ``w''(L/2) = 0``, ``w'''(L/2) = 0``, ``w_s'(L/2) = 0`` (free beam end with
      a free shear-layer end).

    ``w_s' = w' + EI w''''' / k_u`` because ``p = -EI w''''`` and
    ``w_s = w - p/k_u``.
    """
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not math.isfinite(load) or load <= 0.0:
        raise ValueError("load must be finite and positive")
    if not isinstance(n_points, int) or n_points < 5:
        raise ValueError("n_points must be an integer >= 5")
    xp = length / 2.0 if load_position is None else float(load_position)
    if abs(xp - length / 2.0) > 1e-9 * max(length, 1.0):
        raise ValueError("analytic path requires the load at mid-span")

    roots = kerr_characteristic_roots(ei, foundation)
    half = length / 2.0
    ku = foundation.upper
    zero, end = np.array([0.0]), np.array([half])
    d1_0 = _derivatives(zero, roots, 1)[0]
    d3_0 = _derivatives(zero, roots, 3)[0]
    d1_e = _derivatives(end, roots, 1)[0]
    d2_e = _derivatives(end, roots, 2)[0]
    d3_e = _derivatives(end, roots, 3)[0]

    if roots.size == 6:
        d5_0 = _derivatives(zero, roots, 5)[0]
        d5_e = _derivatives(end, roots, 5)[0]
        matrix = np.vstack([d1_0, d3_0, d1_0 + ei * d5_0 / ku, d2_e, d3_e, d1_e + ei * d5_e / ku])
        rhs = np.array([0.0, load / (2.0 * ei), 0.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    elif roots.size == 4:
        matrix = np.vstack([d1_0, d3_0, d2_e, d3_e])
        rhs = np.array([0.0, load / (2.0 * ei), 0.0, 0.0], dtype=np.complex128)
    else:  # pragma: no cover - defensive
        raise ValueError("unexpected characteristic root count")

    coeffs = np.linalg.solve(matrix, rhs)
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("singular analytic system -- check the foundation parameters")

    xs_half = np.linspace(0.0, half, n_points)
    w = np.real(_derivatives(xs_half, roots, 0) @ coeffs)
    w2 = np.real(_derivatives(xs_half, roots, 2) @ coeffs)
    w3 = np.real(_derivatives(xs_half, roots, 3) @ coeffs)
    if roots.size == 6:
        w4 = np.real(_derivatives(xs_half, roots, 4) @ coeffs)
        p = -ei * w4
    else:
        p = foundation.equivalent_winkler * w
    ws = w - p / ku

    x = np.concatenate([half - xs_half[::-1], half + xs_half[1:]])
    flip = lambda arr: np.concatenate([arr[::-1], arr[1:]])  # noqa: E731
    return KerrBeamSolution(
        x=x,
        deflection=flip(w),
        pressure=flip(np.asarray(p, dtype=np.float64)),
        shear_layer=flip(np.asarray(ws, dtype=np.float64)),
        moment=flip(-ei * w2),
        shear=np.concatenate([-(-ei * w3)[::-1], (-ei * w3)[1:]]),
        load=load,
        load_position=xp,
        length=length,
        ei=ei,
        foundation=foundation,
        method="kerr-analytic",
    )


_GAUSS = (
    (-math.sqrt(3.0 / 5.0), 5.0 / 9.0),
    (0.0, 8.0 / 9.0),
    (math.sqrt(3.0 / 5.0), 5.0 / 9.0),
)


def kerr_beam_fem(
    length: float,
    ei: float,
    foundation: KerrFoundation,
    load: float,
    load_position: float | None = None,
    elements: int | None = None,
    samples_per_element: int = 4,
) -> KerrBeamSolution:
    """Mixed finite element solution (Hermite ``w`` + linear ``w_s``).

    Element matrices are produced by 3-point Gauss quadrature of the energy in
    the module docstring.  The load must sit on a node; when ``elements`` is not
    given an even count is chosen so the mid-span is a node.
    """
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not math.isfinite(load) or load <= 0.0:
        raise ValueError("load must be finite and positive")
    xp = length / 2.0 if load_position is None else float(load_position)
    if not (0.0 < xp < length):
        raise ValueError("load_position must lie strictly inside the beam")
    if elements is None:
        elements = 240
    if not isinstance(elements, int) or elements < 4:
        raise ValueError("elements must be an integer >= 4")
    if not isinstance(samples_per_element, int) or samples_per_element < 1:
        raise ValueError("samples_per_element must be an integer >= 1")

    h = length / elements
    ratio = xp / h
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("load_position must coincide with a node")
    n_node = elements + 1
    ndof = 3 * n_node
    k_global = np.zeros((ndof, ndof))
    f_global = np.zeros(ndof)
    ku, kl, g = foundation.upper, foundation.lower, foundation.shear

    for e in range(elements):
        ke = np.zeros((6, 6))
        for gp, gw in _GAUSS:
            xi = 0.5 * (gp + 1.0)
            weight = gw * h / 2.0
            n_w = np.array([
                1.0 - 3.0 * xi**2 + 2.0 * xi**3,
                h * (xi - 2.0 * xi**2 + xi**3),
                3.0 * xi**2 - 2.0 * xi**3,
                h * (-xi**2 + xi**3),
            ])
            d2_w = np.array([
                (-6.0 + 12.0 * xi) / h**2,
                (-4.0 + 6.0 * xi) / h,
                (6.0 - 12.0 * xi) / h**2,
                (-2.0 + 6.0 * xi) / h,
            ])
            n_s = np.array([1.0 - xi, xi])
            d1_s = np.array([-1.0, 1.0]) / h
            loc_w = np.array([0, 1, 3, 4])
            loc_s = np.array([2, 5])
            ke[np.ix_(loc_w, loc_w)] += ei * weight * np.outer(d2_w, d2_w)
            ke[np.ix_(loc_w, loc_w)] += ku * weight * np.outer(n_w, n_w)
            ke[np.ix_(loc_s, loc_s)] += ku * weight * np.outer(n_s, n_s)
            ke[np.ix_(loc_s, loc_s)] += kl * weight * np.outer(n_s, n_s)
            ke[np.ix_(loc_s, loc_s)] += g * weight * np.outer(d1_s, d1_s)
            ke[np.ix_(loc_w, loc_s)] -= ku * weight * np.outer(n_w, n_s)
            ke[np.ix_(loc_s, loc_w)] -= ku * weight * np.outer(n_s, n_w)
        dofs = np.array([
            3 * e + 0, 3 * e + 1, 3 * e + 2,
            3 * (e + 1) + 0, 3 * (e + 1) + 1, 3 * (e + 1) + 2,
        ])
        k_global[np.ix_(dofs, dofs)] += ke

    node = int(round(ratio))
    f_global[3 * node] += load

    sol = np.linalg.solve(k_global, f_global)
    if not np.all(np.isfinite(sol)):
        raise ValueError("singular FEM system -- check the foundation parameters")
    x_nodes = np.linspace(0.0, length, n_node)
    w_nodes, theta_nodes, ws_nodes = sol[0::3], sol[1::3], sol[2::3]

    xs = np.linspace(0.0, length, elements * samples_per_element + 1)
    idx = np.clip(np.floor(xs / h).astype(int), 0, elements - 1)
    xi = (xs - x_nodes[idx]) / h
    w = _hermite(xi, h, w_nodes[idx], theta_nodes[idx], w_nodes[idx + 1], theta_nodes[idx + 1])
    d2 = _hermite_d2(xi, h, w_nodes[idx], theta_nodes[idx], w_nodes[idx + 1], theta_nodes[idx + 1])
    d3 = _hermite_d3(xi, h, w_nodes[idx], theta_nodes[idx], w_nodes[idx + 1], theta_nodes[idx + 1])
    ws = (1.0 - xi) * ws_nodes[idx] + xi * ws_nodes[idx + 1]
    p = ku * (w - ws)
    return KerrBeamSolution(
        x=xs,
        deflection=w,
        pressure=p,
        shear_layer=ws,
        moment=-ei * d2,
        shear=-ei * d3,
        load=load,
        load_position=xp,
        length=length,
        ei=ei,
        foundation=foundation,
        method="kerr-fem",
    )


def _hermite(xi, h, w0, t0, w1, t1):
    return (w0 * (1 - 3 * xi**2 + 2 * xi**3) + t0 * h * (xi - 2 * xi**2 + xi**3)
            + w1 * (3 * xi**2 - 2 * xi**3) + t1 * h * (-xi**2 + xi**3))


def _hermite_d2(xi, h, w0, t0, w1, t1):
    return (w0 * (-6 + 12 * xi) / h**2 + t0 * (-4 + 6 * xi) / h
            + w1 * (6 - 12 * xi) / h**2 + t1 * (-2 + 6 * xi) / h)


def _hermite_d3(xi, h, w0, t0, w1, t1):
    return (w0 * 12 / h**3 + t0 * 6 / h**2 - w1 * 12 / h**3 + t1 * 6 / h**2)


def compare_analytic_fem(
    length: float,
    ei: float,
    foundation: KerrFoundation,
    load: float,
    elements: int = 240,
    n_points: int = 401,
) -> dict[str, Any]:
    """Relative difference between the analytic and FEM Kerr solutions."""
    ana = kerr_beam_analytic(length, ei, foundation, load, n_points=n_points)
    fem = kerr_beam_fem(length, ei, foundation, load, elements=elements)
    w_fem = np.interp(ana.x, fem.x, fem.deflection)
    scale = max(ana.max_deflection, 1e-300)
    peak_fem = float(np.max(np.abs(fem.deflection)))
    peak_ana = float(ana.max_deflection)
    return {
        "method": "kerr-analytic-vs-fem",
        "length_m": length,
        "ei_Nm2": ei,
        "load_N_per_m": load,
        "elements": elements,
        "characteristic_length_m": characteristic_length(ei, foundation),
        "analytic_peak_deflection_m": peak_ana,
        "fem_peak_deflection_m": peak_fem,
        "deflection_profile_rel_error": float(np.max(np.abs(w_fem - ana.deflection)) / scale),
        "peak_deflection_rel_error": abs(peak_fem - peak_ana) / scale,
        "analytic_peak_pressure_Pa": ana.max_pressure,
        "fem_peak_pressure_Pa": float(fem.max_pressure),
        "pressure_peak_rel_error": abs(float(fem.max_pressure) - ana.max_pressure)
        / max(ana.max_pressure, 1e-300),
    }


def convergence_study(
    length: float,
    ei: float,
    foundation: KerrFoundation,
    load: float,
    element_counts: Sequence[int] = (24, 48, 96, 192, 384),
) -> list[dict[str, Any]]:
    """Analytic-vs-FEM error versus the number of elements (mesh convergence)."""
    rows = []
    for n in element_counts:
        row = compare_analytic_fem(length, ei, foundation, load, elements=int(n))
        rows.append({
            "elements": int(n),
            "element_size_m": length / int(n),
            "deflection_profile_rel_error": row["deflection_profile_rel_error"],
            "peak_deflection_rel_error": row["peak_deflection_rel_error"],
        })
    return rows


def moving_load_scan(
    length: float,
    ei: float,
    foundation: KerrFoundation,
    load: float,
    positions: Sequence[float],
    elements: int = 240,
) -> dict[str, Any]:
    """Moving-load envelope (deflection / contact pressure) along the beam."""
    pos = [float(p) for p in positions]
    if not pos:
        raise ValueError("positions must not be empty")
    grid = np.linspace(0.0, length, 1001)
    env_w = np.zeros_like(grid)
    env_p = np.zeros_like(grid)
    worst_w = worst_p = -math.inf
    worst_pos = pos[0]
    for xp in pos:
        sol = kerr_beam_fem(length, ei, foundation, load, load_position=xp, elements=elements)
        env_w = np.maximum(env_w, np.interp(grid, sol.x, sol.deflection))
        env_p = np.maximum(env_p, np.interp(grid, sol.x, sol.pressure))
        if sol.max_deflection > worst_w:
            worst_w, worst_p, worst_pos = sol.max_deflection, sol.max_pressure, xp
    return {
        "positions_m": pos,
        "x_m": grid.tolist(),
        "deflection_envelope_m": env_w.tolist(),
        "pressure_envelope_Pa": env_p.tolist(),
        "worst_deflection_m": float(worst_w),
        "worst_pressure_Pa": float(worst_p),
        "worst_position_m": worst_pos,
    }


def load_deflection_response(
    length: float,
    ei: float,
    foundation: KerrFoundation,
    loads: Sequence[float],
    elements: int = 240,
) -> dict[str, list[float]]:
    """Peak deflection / pressure / reaction balance for a load ramp."""
    out: dict[str, list[float]] = {
        "load_N_per_m": [],
        "peak_deflection_m": [],
        "peak_pressure_Pa": [],
        "foundation_force_N_per_m": [],
    }
    for p in loads:
        sol = kerr_beam_fem(length, ei, foundation, float(p), elements=elements)
        out["load_N_per_m"].append(float(p))
        out["peak_deflection_m"].append(sol.max_deflection)
        out["peak_pressure_Pa"].append(sol.max_pressure)
        out["foundation_force_N_per_m"].append(sol.total_foundation_force)
    return out
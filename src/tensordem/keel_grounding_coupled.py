"""Coupled keel grounding: a bending structure on a unilateral, nonlinear keel.

W28 · dual convergence study of ice-ridge keel grounding.

The W27 study (``ridge_keel_grounding``, ``keel_foundation_dem``) measured the
keel response with a *rigid* footing pushed into a bonded-granular specimen and
found a particle-count GCI of 39.6 %.  This module supplies the second
discretisation the grounding problem actually has: the *structure* that
transmits the load into the keel (ice sheet / subsea structure), discretised with
beam finite elements of size ``h``.

Physics
-------
Euler-Bernoulli beam ``EI w'''' = q - p`` (``w`` positive downward) resting on a
Winkler-type keel whose pressure law ``p(w)`` is the *measured* DEM
confined-compression curve (monotone, no tension, so ``p = 0`` where the beam
lifts off).  Two driving modes: a rigid platen of width ``b`` settled by ``w0``
(the grounding case) and a distributed patch load (verification mode against the
classical Hetenyi infinite-beam solution).

Both the keel law (particle count) and ``h`` (mesh) enter the *same* three
measured quantities, so a two-dimensional refinement study is well posed:

* ``R``     total keel resistance per unit out-of-plane width [N/m];
* ``A_c``   keel contact area per unit width [m^2], i.e. the contact length [m];
* ``q_max`` peak keel contact pressure [Pa].

Numerics
--------
Hermite cubics for ``w``, consistent foundation matrix per element, secant
modulus from the nonlinear keel law, active-set iteration for the no-tension
switch (a convex energy minimisation because ``p(w)`` is monotone), plus a
fixed-point loop on the secant stiffness.  Contact length and contact pressure
are evaluated on a sub-element sample grid with bisection refinement of the
``w = 0`` crossing, so a mesh-refinement study measures the FE error and not the
output sampling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = [
    "KeelPressureLaw",
    "beam_on_unilateral_keel",
    "winkler_infinite_beam_patch",
    "observed_order",
    "richardson_limit",
    "spearman",
]


# ---------------------------------------------------------------------------
# keel constitutive law (from the DEM confined-compression test)
# ---------------------------------------------------------------------------
@dataclass
class KeelPressureLaw:
    """Monotone pressure-settlement law ``p(w)`` of the keel [Pa, m].

    Built from the DEM confined-compression test points plus the origin.  Beyond
    the measured settlement range the last secant is continued linearly (a
    documented extrapolation; the caller checks whether the coupled solution
    stays inside the DEM range).
    """

    settlements_m: Sequence[float]
    pressures_Pa: Sequence[float]

    def __post_init__(self) -> None:
        w = np.asarray(self.settlements_m, dtype=float)
        p = np.asarray(self.pressures_Pa, dtype=float)
        if w.ndim != 1 or p.ndim != 1 or w.size != p.size or w.size < 2:
            raise ValueError("settlement and pressure arrays must be 1-D of equal length >= 2")
        if np.any(np.diff(w) <= 0.0):
            raise ValueError("settlements must be strictly increasing")
        if np.any(np.diff(p) <= 0.0):
            raise ValueError("pressures must be strictly increasing (monotone keel)")
        if w[0] > 0.0:
            w = np.concatenate(([0.0], w))
            p = np.concatenate(([0.0], p))
        self.w = w
        self.p = p
        self.w_max = float(w[-1])
        self.p_max = float(p[-1])
        self.secant_max = self.p_max / self.w_max
        self.small_strain_secant = float(p[1] / w[1])
        self._f = PchipInterpolator(w, p, extrapolate=False)

    def pressure(self, w: Iterable[float]) -> np.ndarray:
        ww = np.clip(np.asarray(w, dtype=float), 0.0, None)
        out = np.empty_like(ww)
        inside = ww <= self.w_max
        out[inside] = self._f(ww[inside])
        out[~inside] = self.p_max + self.secant_max * (ww[~inside] - self.w_max)
        return out

    def dpressure(self, w: Iterable[float]) -> np.ndarray:
        """Tangent modulus ``dp/dw`` [N/m^3], with the same extrapolation rule."""
        ww = np.clip(np.asarray(w, dtype=float), 0.0, None)
        out = np.empty_like(ww)
        inside = ww <= self.w_max
        d = self._f.derivative()
        out[inside] = d(ww[inside])
        out[~inside] = self.secant_max
        return out

    def secant(self, w: Iterable[float]) -> np.ndarray:
        """Secant modulus ``p(w)/w`` [N/m^3] (small-strain value at ``w -> 0``)."""
        ww = np.clip(np.asarray(w, dtype=float), 0.0, None)
        out = np.full_like(ww, self.small_strain_secant)
        big = ww > 0.0
        out[big] = self.pressure(ww[big]) / ww[big]
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"settlements_m": [float(v) for v in self.w],
                "pressures_Pa": [float(v) for v in self.p],
                "secant_at_w_max_N_per_m3": float(self.secant_max),
                "small_strain_secant_N_per_m3": float(self.small_strain_secant),
                "extrapolation": "linear continuation of the last secant beyond w_max"}


# ---------------------------------------------------------------------------
# beam element machinery
# ---------------------------------------------------------------------------
def _hermite(xi: Any, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.asarray(xi, dtype=float)
    n1 = 1.0 - 3.0 * xi ** 2 + 2.0 * xi ** 3
    n2 = h * (xi - 2.0 * xi ** 2 + xi ** 3)
    n3 = 3.0 * xi ** 2 - 2.0 * xi ** 3
    n4 = h * (-xi ** 2 + xi ** 3)
    return n1, n2, n3, n4


def _ramp_factor(w: Any, delta: float) -> np.ndarray:
    """C1 no-tension ramp: ``1`` for ``w >= delta``, ``0`` for ``w <= 0``.

    ``smoothstep`` gives ``p(0) = p'(0) = 0``, so the pressure law stays C1 at the
    separation point.  A hard switch (``delta = 0``) makes the discrete solution
    jump with the mesh (the contact set is an element-level set), which is why the
    formal refinement study uses the ramp and the hard switch is only a
    diagnostic.
    """
    ww = np.asarray(w, dtype=float)
    if delta <= 0.0:
        return (ww > 0.0).astype(float)
    u = np.clip(ww / delta, 0.0, 1.0)
    return 3.0 * u ** 2 - 2.0 * u ** 3


def _beam_stiffness(ei: float, h: float) -> np.ndarray:
    c = ei / h ** 3
    return c * np.array([[12.0, 6.0 * h, -12.0, 6.0 * h],
                         [6.0 * h, 4.0 * h * h, -6.0 * h, 2.0 * h * h],
                         [-12.0, -6.0 * h, 12.0, -6.0 * h],
                         [6.0 * h, 2.0 * h * h, -6.0 * h, 4.0 * h * h]])


def _foundation_stiffness(k: float, h: float) -> np.ndarray:
    """Consistent Winkler matrix ``k * int N_i N_j dx`` (exact for Hermite cubics)."""
    c = k * h / 420.0
    return c * np.array([[156.0, 22.0 * h, 54.0, -13.0 * h],
                         [22.0 * h, 4.0 * h * h, 13.0 * h, -3.0 * h * h],
                         [54.0, 13.0 * h, 156.0, -22.0 * h],
                         [-13.0 * h, -3.0 * h * h, -22.0 * h, 4.0 * h * h]])


def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a x = b`` with a ridge/lstsq fallback for near-singular tangents."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        diag = np.abs(np.diag(a))
        ridge = 1e-10 * max(float(diag.max()) if diag.size else 1.0, 1e-30)
        try:
            return np.linalg.solve(a + ridge * np.eye(a.shape[0]), b)
        except np.linalg.LinAlgError:  # pragma: no cover
            return np.linalg.lstsq(a, b, rcond=None)[0]


def _sub_sample(x0: float, h: float, w_nodes: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    xi = np.linspace(0.0, 1.0, n + 1)
    n1, n2, n3, n4 = _hermite(xi, h)
    w = n1 * w_nodes[0] + n2 * w_nodes[1] + n3 * w_nodes[2] + n4 * w_nodes[3]
    return x0 + xi * h, w


def _contact_length(x0: float, h: float, w_nodes: np.ndarray, n: int) -> float:
    """Contact length of one element: ``int 1[w > 0] dx`` with root refinement."""
    xs, ws = _sub_sample(x0, h, w_nodes, n)
    total = 0.0
    for i in range(ws.size - 1):
        a, b = ws[i], ws[i + 1]
        if a > 0.0 and b > 0.0:
            total += xs[i + 1] - xs[i]
        elif (a > 0.0 >= b) or (a <= 0.0 < b):
            lo, hi = 0.0, 1.0
            positive_left = a > 0.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                n1, n2, n3, n4 = _hermite(np.array([mid]), h)
                val = float(n1[0] * w_nodes[0] + n2[0] * w_nodes[1]
                            + n3[0] * w_nodes[2] + n4[0] * w_nodes[3])
                if (val > 0.0) == positive_left:
                    lo = mid
                else:
                    hi = mid
            x_root = xs[i] + 0.5 * (lo + hi) * (xs[i + 1] - xs[i])
            total += (x_root - xs[i]) if positive_left else (xs[i + 1] - x_root)
    return total


def graded_nodes(length: float, n_elements: int, *, centres: Sequence[float],
                 sigma: float, amplitude: float) -> np.ndarray:
    """Node positions of a mesh clustered (Gaussian density bump) at ``centres``.

    The density ``rho(x) = 1 + sum_k amplitude * exp(-((x - c_k)/sigma)^2)`` is
    integrated and inverted at equal increments, so every refinement level of the
    same family halves the element size *locally* (a uniform refinement ratio)
    while the element size stays smallest at the knowable singular points
    (platen edges, where the contact pressure is discontinuous).
    """
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be finite and > 0")
    xs = np.linspace(0.0, length, 20001)
    rho = np.ones_like(xs)
    for c in centres:
        rho += float(amplitude) * np.exp(-((xs - float(c)) / sigma) ** 2)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(xs))))
    cdf /= cdf[-1]
    targets = np.linspace(0.0, 1.0, n_elements + 1)
    return np.interp(targets, cdf, xs)


def beam_on_unilateral_keel(*, length: float, ei: float, n_elements: int,
                            platen_centre: float | None = None,
                            platen_width: float | None = None,
                            w0: float | None = None,
                            law: KeelPressureLaw | None = None,
                            k_uniform: float | None = None,
                            load_intensity: float | None = None,
                            load_half_width: float | None = None,
                            samples_per_element: int = 32,
                            end_support: bool = False,
                            ramp_width_m: float = 0.0,
                            separation_offset_m: float = 0.0,
                            n_quad: int = 8, x_nodes: np.ndarray | None = None,
                            tol: float = 1e-10, max_newton: int = 100) -> dict[str, Any]:
    """Beam on a unilateral nonlinear keel foundation, solved by a Newton method.

    Discretisation: Hermite cubics for ``w``; the foundation energy
    ``int P(w) dx`` (``P' = p``) is integrated with ``n_quad`` Gauss points per
    element and the tangent ``int p'(w) N N^T dx`` is assembled consistently, so
    the discrete residual is zero at the solution and the reported platen reaction
    and keel reaction agree to quadrature accuracy.

    Driving modes:

    ``platen`` (``w0`` given)  rigid platen of width ``platen_width`` centred at
        ``platen_centre`` settled by ``w0``; no-tension keel law ``law``.
    ``load`` (``load_intensity`` given)  patch load on a linear Winkler layer
        ``k_uniform`` -- verification against the classical infinite-beam
        solution (no no-tension switch in this mode).

    ``ramp_width_m = delta`` softens the no-tension switch with a C1 smoothstep of
    width ``delta`` (``delta = 0`` keeps the hard switch); ``end_support`` holds
    both beam ends at ``w = 0``; ``separation_offset_m`` starts contact at that
    settlement instead of at zero.
    """
    if n_elements < 4:
        raise ValueError("n_elements must be >= 4")
    for name, val in (("length", length), ("ei", ei)):
        if not (math.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be finite and > 0")
    if x_nodes is not None:
        x = np.asarray(x_nodes, dtype=float)
        if x.ndim != 1 or x.size < 5 or np.any(np.diff(x) <= 0.0) or abs(x[0]) > 1e-12 \
                or abs(x[-1] - length) > 1e-9 * max(1.0, length):
            raise ValueError("x_nodes must be increasing and span [0, length]")
        n_elements = x.size - 1
    else:
        x = np.linspace(0.0, length, n_elements + 1)
    nn = x.size
    ndof = 2 * nn
    h_elem = np.diff(x)
    h = float(length / n_elements)

    if law is None and k_uniform is None:
        raise ValueError("either law or k_uniform must be given")
    unilateral = law is not None or ramp_width_m > 0.0
    delta = float(ramp_width_m)
    offset = float(separation_offset_m)

    fixed = np.zeros(ndof, dtype=bool)
    fixed_values = np.zeros(ndof)
    f_ext = np.zeros(ndof)
    plate = np.zeros(nn, dtype=bool)
    mode = "platen" if w0 is not None else "load"
    if mode == "platen":
        if law is None:
            raise ValueError("platen mode needs a keel pressure law")
        if platen_width is None or platen_centre is None:
            raise ValueError("platen mode needs platen_width and platen_centre")
        plate = np.abs(x - float(platen_centre)) <= 0.5 * float(platen_width) + 1e-12 * length
        if int(np.count_nonzero(plate)) < 1:
            raise ValueError("platen contains no node: increase platen_width or n_elements")
        w_dofs = 2 * np.flatnonzero(plate)
        fixed[w_dofs] = True
        fixed_values[w_dofs] = float(w0)
    else:
        if k_uniform is None or load_intensity is None or load_half_width is None:
            raise ValueError("load mode needs k_uniform, load_intensity and load_half_width")
        xc = float(platen_centre if platen_centre is not None else 0.5 * length)
        a = float(load_half_width)
        q = float(load_intensity)
        xi = np.linspace(0.0, 1.0, 801)
        for e in range(n_elements):
            he = float(h_elem[e])
            n1, n2, n3, n4 = _hermite(xi, he)
            xe = x[e] + xi * he
            inside = (xe >= xc - a) & (xe <= xc + a)
            if not np.any(inside):
                continue
            weight = np.where(inside, 1.0, 0.0) * q * he * np.gradient(xi)
            f_ext[2 * e] += float(np.sum(n1 * weight))
            f_ext[2 * e + 1] += float(np.sum(n2 * weight))
            f_ext[2 * e + 2] += float(np.sum(n3 * weight))
            f_ext[2 * e + 3] += float(np.sum(n4 * weight))
    if end_support:
        for dof in (0, 2 * (nn - 1)):
            fixed[dof] = True
            fixed_values[dof] = 0.0
    free = ~fixed

    def _p_dp(w: float) -> tuple[float, float]:
        we = w - offset
        if law is not None:
            base = float(law.pressure([we])[0])
            dbase = float(law.dpressure([we])[0])
        else:
            base = float(k_uniform) * we
            dbase = float(k_uniform)
        if not unilateral:
            return base, dbase
        if delta > 0.0:
            u = min(max(we / delta, 0.0), 1.0)
            ramp = 3.0 * u * u - 2.0 * u ** 3
            dramp = (6.0 * u - 6.0 * u * u) / delta if 0.0 < we < delta else 0.0
        else:
            ramp = 1.0 if we > 0.0 else 0.0
            dramp = 0.0
        return base * ramp, dbase * ramp + base * dramp

    gx, gw = np.polynomial.legendre.leggauss(int(n_quad))
    xi_g = 0.5 * (gx + 1.0)
    wt_g = 0.5 * gw                                                   # times the element length
    ke_elem = [_beam_stiffness(ei, float(he)) for he in h_elem]
    shapes_elem = [np.array([np.concatenate(_hermite(np.array([s]), float(he))) for s in xi_g])
                   for he in h_elem]                                  # per element: (nq, 4)

    def assemble(u: np.ndarray, tangent: bool) -> tuple[np.ndarray, np.ndarray | None]:
        res = -f_ext.copy()
        kg = np.zeros((ndof, ndof)) if tangent else None
        for e in range(n_elements):
            idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            ue = u[idx]
            res[idx] += ke_elem[e] @ ue
            if tangent:
                kg[np.ix_(idx, idx)] += ke_elem[e]
            he = float(h_elem[e])
            for g in range(xi_g.size):
                N = shapes_elem[e][g]
                w = float(N @ ue)
                p, dp = _p_dp(w)
                J = wt_g[g] * he
                res[idx] += p * N * J
                if tangent:
                    kg[np.ix_(idx, idx)] += dp * np.outer(N, N) * J
        return res, kg

    u = np.zeros(ndof)
    u[fixed] = fixed_values[fixed]
    history: list[dict[str, Any]] = []
    converged = False
    finite = True
    scale = None
    newton_it = 0
    max_step = 4.0 * (abs(float(w0)) if mode == "platen" else 1.0)
    for newton_it in range(1, max_newton + 1):
        res, kg = assemble(u, True)
        rf = res[free]
        if not (np.all(np.isfinite(res)) and np.all(np.isfinite(kg))):
            finite = False
            break
        norm = float(np.max(np.abs(rf))) if rf.size else 0.0
        if scale is None:
            scale = max(norm, 1e-30)
        history.append({"iteration": newton_it, "residual_inf_N_per_m": norm,
                        "residual_rel": norm / scale})
        if norm <= tol * scale:
            converged = True
            break
        du = _solve_linear(kg[np.ix_(free, free)], -rf)
        if not np.all(np.isfinite(du)):
            finite = False
            break
        # step limiter on the displacement dofs (keeps the Newton path physical)
        w_dofs = np.flatnonzero(free)[0::2]
        if w_dofs.size:
            big = float(np.max(np.abs(du[0::2]))) if du.size > 1 else 0.0
            if big > max_step:
                du *= max_step / big
        step = 1.0
        accepted = False
        for _ in range(40):
            u_try = u.copy()
            u_try[free] += step * du
            res_try, _ = assemble(u_try, False)
            if not np.all(np.isfinite(res_try)):
                step *= 0.5
                continue
            norm_try = float(np.max(np.abs(res_try[free]))) if rf.size else 0.0
            if norm_try < norm * (1.0 - 1e-4 * step) or step < 1e-6:
                u = u_try
                accepted = True
                break
            step *= 0.5
        if not accepted:  # pragma: no cover - backtracking exhausted
            break

    # ---------------- global equilibrium and outputs ----------------
    res, kg = assemble(u, True)
    platen_dofs = 2 * np.flatnonzero(plate)
    platen_reaction = float(np.sum(res[platen_dofs] + f_ext[platen_dofs]))

    w_nodes = u[0:2 * nn:2]
    theta = u[1:2 * nn:2]
    contact = 0.0
    resistance = 0.0
    q_max = 0.0
    xs_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for e in range(n_elements):
        he = float(h_elem[e])
        w_e = np.array([w_nodes[e], theta[e], w_nodes[e + 1], theta[e + 1]])
        contact += _contact_length(x[e], he, w_e - offset, samples_per_element)
        xs, ws = _sub_sample(x[e], he, w_e, samples_per_element)
        pe = np.array([_p_dp(v)[0] for v in ws])
        resistance += float(np.trapezoid(pe, xs))
        q_max = max(q_max, float(np.max(pe)))
        xs_all.append(xs)
        p_all.append(pe)
    max_keel_settle = float(np.max(np.clip(w_nodes - offset, 0.0, None)))
    return {
        "mode": mode,
        "n_elements": n_elements, "element_length_m": h,
        "element_length_min_m": float(np.min(h_elem)),
        "element_length_max_m": float(np.max(h_elem)),
        "length_m": float(length),
        "ei_N_m2": float(ei),
        "ramp_width_m": delta, "separation_offset_m": offset,
        "end_support": bool(end_support),
        "x_m": x.tolist(), "w_m": w_nodes.tolist(), "theta_rad": theta.tolist(),
        "resistance_N_per_m": float(resistance),
        "platen_reaction_N_per_m": platen_reaction,
        "equilibrium_rel_error": float(abs(platen_reaction - resistance)
                                       / max(abs(platen_reaction), 1e-300)),
        "contact_area_m2_per_m": float(contact),
        "contact_length_m": float(contact),
        "q_max_Pa": float(q_max),
        "w_max_m": float(np.max(np.abs(w_nodes))),
        "w_min_m": float(np.min(w_nodes)),
        "collected": [state.get("residual_rel") for state in history],
        "newton_iterations": int(newton_it),
        "converged": bool(converged and finite),
        "solution_finite": bool(finite and np.all(np.isfinite(u))),
        "newton_history": history,
        "contact_pressure_profile": {"x_m": np.concatenate(xs_all).tolist(),
                                     "p_Pa": np.concatenate(p_all).tolist()},
        "max_keel_settlement_m": max_keel_settle,
        "keel_law_w_max_m": None if law is None else float(law.w_max),
        "inside_dem_range": (None if law is None else bool(max_keel_settle <= law.w_max)),
    }


# ---------------------------------------------------------------------------
# verification reference: Hetenyi infinite beam on a linear Winkler layer
# ---------------------------------------------------------------------------
def winkler_infinite_beam_patch(*, ei: float, k: float, load_intensity: float,
                                load_half_width: float, x: Iterable[float],
                                n_gauss: int = 600) -> np.ndarray:
    """Deflection of an infinite beam on a Winkler layer under a uniform patch load.

    ``w(x) = (q beta / 2k) int_{-a}^{a} A(beta |x - xi|) dxi`` with the even kernel
    ``A(beta s) = exp(-beta s) (cos beta s + sin beta s)`` and
    ``beta = (k/(4EI))**0.25`` (Hetenyi 1946, beam on elastic foundation; the
    kernel is even in ``s`` because the infinite-beam Green function is symmetric).
    """
    beta = (k / (4.0 * ei)) ** 0.25
    x = np.asarray(x, dtype=float)
    xi, wq = np.polynomial.legendre.leggauss(int(n_gauss))
    a = float(load_half_width)
    out = np.empty_like(x)
    for i, xx in enumerate(x):
        # integrand has a kink at s = 0, i.e. at xi = x: split the patch there
        if -a < xx < a:
            panels = [(-a, xx), (xx, a)]
        else:
            panels = [(-a, a)]
        total = 0.0
        for lo, hi in panels:
            if hi <= lo:
                continue
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            s = np.abs(xx - (mid + half * xi))
            d = np.exp(-beta * s) * (np.cos(beta * s) + np.sin(beta * s))
            total += half * float(np.sum(wq * d))
        out[i] = total * float(load_intensity) * beta / (2.0 * k)
    return out


# ---------------------------------------------------------------------------
# refinement convergence: observed order, GCI, Richardson limit + band
# ---------------------------------------------------------------------------
def observed_order(fine: float, medium: float, coarse: float, *, r: float) -> dict[str, Any]:
    """Observed order ``p`` and the GCI of the fine level (Roache 1994)."""
    f, m, c = float(fine), float(medium), float(coarse)
    out: dict[str, Any] = {"fine": f, "medium": m, "coarse": c, "r": float(r)}
    diff_mc = m - c
    out["monotone"] = bool(abs(f - m) < abs(m - c) and (f - m) * (m - c) > 0.0)
    if diff_mc == 0.0:
        out.update({"observed_order_p": None, "gci_fine_rel": None,
                    "note": "medium == coarse (no convergence signal)"})
        return out
    ratio = (f - m) / diff_mc
    out["convergence_ratio_R"] = float(ratio)
    if ratio <= 0.0 or f == m:
        out.update({"observed_order_p": None, "gci_fine_rel": None,
                    "note": "non-monotone or divergent triad: p undefined"})
        return out
    try:
        p = abs(math.log(ratio) / math.log(float(r)))
    except (ValueError, ZeroDivisionError):  # pragma: no cover
        p = float("nan")
    if not math.isfinite(p) or p <= 1e-12:
        out.update({"observed_order_p": None, "gci_fine_rel": None,
                    "note": "observed order vanishes: no geometric error reduction"})
        return out
    out["observed_order_p"] = float(p)
    out["gci_fine_rel"] = float(1.25 * abs((f - m) / f) / (float(r) ** p - 1.0))
    return out


def richardson_limit(fine: float, medium: float, coarse: float, *, r: float,
                     safety: float = 1.25) -> dict[str, Any]:
    """Observed order + GCI + Richardson-extrapolated limit and uncertainty band."""
    info = observed_order(fine, medium, coarse, r=r)
    f, m = float(fine), float(medium)
    p = info.get("observed_order_p")
    if p is None or not math.isfinite(p) or p <= 1e-12:
        info.update({"richardson_limit": None, "uncertainty_band_abs": None,
                     "uncertainty_band_rel": None,
                     "verdict": "no_extrapolation (p undefined/divergent)"})
        return info
    rp = float(r) ** p
    limit = f + (f - m) / (rp - 1.0)
    band = safety * abs(limit - f)
    info.update({"richardson_limit": float(limit),
                 "uncertainty_band_abs": float(band),
                 "uncertainty_band_rel": float(band / abs(limit)) if limit else None,
                 "safety_factor": float(safety),
                 "verdict": "extrapolated"})
    return info


def spearman(a: Iterable[float], b: Iterable[float]) -> dict[str, Any]:
    """Spearman rank correlation with average ranks and a two-sided t p-value."""
    from scipy import stats

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size != bb.size or aa.size < 3:
        raise ValueError("spearman needs two equal-length arrays of size >= 3")
    rho, pval = stats.spearmanr(aa, bb)
    return {"n": int(aa.size), "rho": float(rho), "p_value": float(pval)}
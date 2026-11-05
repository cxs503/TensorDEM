"""Verify the ridge-grounding Winkler subgrade modulus against Vesic (1961).

The merged geotechnical chain (`src/tensordem/ridge_geotech.py`) models the seabed
under a buried pipeline as a Winkler foundation with

    k_v = 0.65 * E_s / ((1 - nu_s^2) * B)                  [N/m^3]

(`MohrCoulombSoil.subgrade_modulus`).  The ``0.65`` coefficient *is* Vesic's (1961)
coefficient, but Vesic's published beam-on-elastic-half-space relation also carries
a beam rigidity factor:

    k_v = 0.65 * (E_s / (1 - nu_s^2)) * (E_s B^4 / (E_p I_p))^(1/12) / B

That factor is absent from the library, so the foundation stiffness is set by the
soil alone.  This script quantifies the consequence against two independent
references:

1. the Vesic (1961) closed form, and
2. a self-contained 3-D elastic half-space reference in which the beam is coupled
   to a Boussinesq continuum through the exact closed-form settlement kernel

       w(x) = (1 - nu^2) / (pi E) * integral integral p(s) / r ds dt

   discretised with piecewise-constant contact pressure and *exact* rectangle
   integrals of ``1/r`` (:func:`strip_kernel`).  The beam half of the reference is
   validated against the Hetenyi (1946) closed form (the Winkler limit), the kernel
   against 2-D Gauss quadrature and the point-load limit, and the coupled solution
   through length/mesh convergence plus global equilibrium.

Run with ``PYTHONPATH=src python scripts/verify_ridge_winkler_subgrade_modulus.py``.
Writes ``docs/artifacts/ridge_winkler_subgrade/`` (JSON + figure) and injects the
result tables into ``docs/ridge_winkler_subgrade_modulus_verification.md``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensordem.ridge_geotech import MohrCoulombSoil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "artifacts" / "ridge_winkler_subgrade"
DOC = ROOT / "docs" / "ridge_winkler_subgrade_modulus_verification.md"

# ---------------------------------------------------------------- reference case
PIPE_D = 0.914              # m, 36 in line pipe used by the coupled campaign
PIPE_T = 0.0159             # m
STEEL_E = 2.1e11            # Pa
SOIL_E = 10.0e6             # Pa, MohrCoulombSoil default
SOIL_NU = 0.30
PATCH_TOTAL_LOAD = 8.67e5   # N, DEM ridge-keel envelope of the campaign
PATCH_LENGTH = 2.0          # m, ice/pipe contact footprint
BEAM_LENGTH = 160.0         # m, >> 1/beta so the beam ends are inactive
N_ELEM = 800                # 0.2 m elements -> 10 elements under the footprint
VESIC_COEFFICIENT = 0.65


def pipe_inertia(outer_diameter: float, wall_thickness: float) -> float:
    """Second moment of area of a line pipe, m^4 (independent of PipeSection)."""
    inner = outer_diameter - 2.0 * wall_thickness
    return math.pi / 64.0 * (outer_diameter ** 4 - inner ** 4)


# ------------------------------------------------- exact strip settlement kernel
def quadrant_over_r(x, y):
    """``integral_0^x integral_0^y du dt / sqrt(u^2 + t^2)`` for x, y >= 0.

    Exact antiderivative: ``d2/du dt [u asinh(t/u) + t asinh(u/t)] = 1/hypot(u,t)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros(np.broadcast(x, y).shape, dtype=float)
    ok = (x > 0.0) & (y > 0.0)
    xs = np.where(ok, x, 1.0)
    ys = np.where(ok, y, 1.0)
    out[ok] = (xs[ok] * np.arcsinh(ys[ok] / xs[ok])
               + ys[ok] * np.arcsinh(xs[ok] / ys[ok]))
    return out


def strip_kernel(u0, u1, half_width: float):
    """``integral_{u0}^{u1} du integral_{-h}^{h} dt / sqrt(u^2 + t^2)`` (exact).

    The integrand is even in ``u``, so the signed interval is folded onto the
    positive axis: ``integral = sign(u1) S(|u1|) - sign(u0) S(|u0|)`` with
    ``S(v) = 2 * quadrant_over_r(v, h)``.
    """
    def S(v):
        return 2.0 * quadrant_over_r(np.abs(v), half_width)

    u0 = np.asarray(u0, dtype=float)
    u1 = np.asarray(u1, dtype=float)
    return np.sign(u1) * S(u1) - np.sign(u0) * S(u0)


def compliance_matrix(n_elem: int, le: float, width: float, young_modulus: float,
                      poisson_ratio: float) -> np.ndarray:
    """Settlement compliance ``C[i, j]`` [m/Pa]: unit contact pressure on element j
    (piecewise constant over ``le`` x ``width``) -> settlement at element centre i.
    """
    centres = (np.arange(n_elem) + 0.5) * le
    coef = (1.0 - poisson_ratio ** 2) / (math.pi * young_modulus * width)
    u_hi = (centres + 0.5 * le)[None, :] - centres[:, None]
    u_lo = (centres - 0.5 * le)[None, :] - centres[:, None]
    C = coef * strip_kernel(u_lo, u_hi, 0.5 * width)
    return 0.5 * (C + C.T)

DIFF_SPAN = 10.0            # m, reference station for the differential deflection


def _differential(w_mid, le: float, length: float, span: float = DIFF_SPAN) -> float:
    """Deflection measured against the beam response ``span`` metres away.

    A beam on an elastic *half-space* has a settlement field with a logarithmic,
    length-dependent offset, so the absolute deflection is not a well-posed metric;
    the differential value is.  For a Winkler bed the far field is ~0 and this equals
    the maximum deflection to within a couple of per cent.
    """
    centres = (np.arange(len(w_mid)) + 0.5) * le - 0.5 * length
    centre_value = w_mid[int(np.argmin(np.abs(centres)))]
    far_value = w_mid[int(np.argmin(np.abs(np.abs(centres) - span)))]
    return float(centre_value - far_value)


# --------------------------------------------------------------- beam elements
def beam_stiffness(ei: float, le: float) -> np.ndarray:
    """Euler-Bernoulli Hermite element (dof order w1, theta1, w2, theta2)."""
    return ei / le ** 3 * np.array([
        [12.0, 6.0 * le, -12.0, 6.0 * le],
        [6.0 * le, 4.0 * le ** 2, -6.0 * le, 2.0 * le ** 2],
        [-12.0, -6.0 * le, 12.0, -6.0 * le],
        [6.0 * le, 2.0 * le ** 2, -6.0 * le, 4.0 * le ** 2],
    ])


def foundation_stiffness(k_eff: float, le: float) -> np.ndarray:
    """Consistent Hermite matrix of a continuous spring bed ``k_eff`` [N/m^2]."""
    return k_eff * le * np.array([
        [13.0 / 35.0, 11.0 * le / 210.0, 9.0 / 70.0, -13.0 * le / 420.0],
        [11.0 * le / 210.0, le ** 2 / 105.0, 13.0 * le / 420.0, -le ** 2 / 140.0],
        [9.0 / 70.0, 13.0 * le / 420.0, 13.0 / 35.0, -11.0 * le / 210.0],
        [-13.0 * le / 420.0, -le ** 2 / 140.0, -11.0 * le / 210.0, le ** 2 / 105.0],
    ])


def assemble(n_elem: int, ke: np.ndarray) -> np.ndarray:
    ndof = 2 * (n_elem + 1)
    K = np.zeros((ndof, ndof))
    for e in range(n_elem):
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += ke
    return K


def midpoint_map(n_elem: int, le: float) -> np.ndarray:
    """Element midspan deflection from nodal dofs: w_mid = (w1+w2)/2 + le(theta1-theta2)/8."""
    T = np.zeros((n_elem, 2 * (n_elem + 1)))
    for e in range(n_elem):
        T[e, 2 * e] = 0.5
        T[e, 2 * e + 1] = 0.125 * le
        T[e, 2 * e + 2] = 0.5
        T[e, 2 * e + 3] = -0.125 * le
    return T


def patch_load_vector(n_elem: int, le: float, q: float, first: int, count: int) -> np.ndarray:
    """Consistent nodal loads for a uniform line load ``q`` [N/m] over ``count`` elements."""
    f = np.zeros(2 * (n_elem + 1))
    for e in range(first, first + count):
        f[2 * e] += 0.5 * q * le
        f[2 * e + 1] += q * le ** 2 / 12.0
        f[2 * e + 2] += 0.5 * q * le
        f[2 * e + 3] -= q * le ** 2 / 12.0
    return f


def nodal_moments(d: np.ndarray, ei: float, le: float, n_elem: int) -> np.ndarray:
    """Bending moment ``M = EI w''`` at every node, averaged over adjacent elements."""
    acc = np.zeros(n_elem + 1)
    cnt = np.zeros(n_elem + 1)
    for e in range(n_elem):
        w1, t1, w2, t2 = d[2 * e:2 * e + 4]
        acc[e] += ei * (-6.0 * w1 - 4.0 * le * t1 + 6.0 * w2 - 2.0 * le * t2) / le ** 2
        acc[e + 1] += ei * (6.0 * w1 + 2.0 * le * t1 - 6.0 * w2 + 4.0 * le * t2) / le ** 2
        cnt[e] += 1.0
        cnt[e + 1] += 1.0
    return acc / cnt


def _patch_geometry(n_elem: int, patch_length: float, length: float):
    le = length / n_elem
    count = max(1, int(round(patch_length / le)))
    return (n_elem - count) // 2, count


# --------------------------------------------------------------------- solvers
def solve_winkler(length: float, n_elem: int, ei: float, k_eff: float, q: float,
                  patch_length: float) -> dict:
    """Euler-Bernoulli beam on a continuous spring bed ``k_eff`` [N/m^2]."""
    le = length / n_elem
    first, count = _patch_geometry(n_elem, patch_length, length)
    K = assemble(n_elem, beam_stiffness(ei, le) + foundation_stiffness(k_eff, le))
    f = patch_load_vector(n_elem, le, q, first, count)
    d = np.linalg.solve(K, f)
    w_mid = midpoint_map(n_elem, le) @ d
    M = nodal_moments(d, ei, le, n_elem)
    return {
        "deflection_max_m": float(w_mid.max()),
        "differential_deflection_m": _differential(w_mid, le, length),
        "moment_max_Nm_per_m": float(np.abs(M).max()),
        "pressure_max_Pa": abs(k_eff * float(w_mid.max())),
        "n_elem": n_elem,
        "length_m": length,
    }


def solve_halfspace(length: float, n_elem: int, ei: float, width: float,
                    young_modulus: float, poisson_ratio: float, q: float,
                    patch_length: float) -> dict:
    """Beam coupled to a 3-D elastic half-space (Boussinesq kernel, no Winkler)."""
    le = length / n_elem
    first, count = _patch_geometry(n_elem, patch_length, length)
    Kb = assemble(n_elem, beam_stiffness(ei, le))
    T = midpoint_map(n_elem, le)
    C = compliance_matrix(n_elem, le, width, young_modulus, poisson_ratio)
    ndof = 2 * (n_elem + 1)

    Q = np.zeros((ndof, n_elem))
    for e in range(n_elem):
        Q[2 * e, e] = 0.5 * width * le
        Q[2 * e + 1, e] = width * le ** 2 / 12.0
        Q[2 * e + 2, e] = 0.5 * width * le
        Q[2 * e + 3, e] = -width * le ** 2 / 12.0

    f = patch_load_vector(n_elem, le, q, first, count)
    A = np.zeros((ndof + n_elem, ndof + n_elem))
    A[:ndof, :ndof] = Kb
    A[:ndof, ndof:] = Q
    A[ndof:, :ndof] = -T
    A[ndof:, ndof:] = C
    rhs = np.concatenate([f, np.zeros(n_elem)])

    # Row and column equilibration: the raw system mixes 1e12 and 1e-8 magnitudes.
    row_scale = np.concatenate([np.full(ndof, ei / le ** 3), np.full(n_elem, C.max())])
    col_scale = np.concatenate([np.full(ndof, 1.0e-3), np.full(n_elem, 1.0e5)])
    A_scaled = (A / row_scale[:, None]) * col_scale[None, :]
    x_scaled = np.linalg.solve(A_scaled, rhs / row_scale)
    x = x_scaled * col_scale
    d, p = x[:ndof], x[ndof:]

    w_mid = T @ d
    M = nodal_moments(d, ei, le, n_elem)
    applied = float(q * le * count)
    reacted = float(np.sum(p) * width * le)
    return {
        "deflection_max_m": float(w_mid.max()),
        "differential_deflection_m": _differential(w_mid, le, length),
        "uplift_min_m": float(w_mid.min()),
        "moment_max_Nm_per_m": float(np.abs(M).max()),
        "pressure_max_Pa": float(p.max()),
        "contact_length_m": float(np.count_nonzero(p > 0.0) * le),
        "min_pressure_Pa": float(p.min()),
        "equilibrium_rel_error": abs(reacted - applied) / applied,
        "compatibility_residual_m": float(np.abs(T @ d - C @ p).max()),
        "n_elem": n_elem,
        "length_m": length,
        "profile": {"x": ((np.arange(n_elem) + 0.5) * le).tolist(),
                    "deflection": w_mid.tolist(),
                    "pressure": p.tolist()},
    }

# ------------------------------------------------------------ modulus models
def vesic_subgrade_modulus(young_modulus: float, poisson_ratio: float, width: float,
                           beam_ei: float, *, coefficient: float = VESIC_COEFFICIENT) -> float:
    """Vesic (1961) equivalent Winkler modulus of a beam on an elastic half-space.

        k_v = C * (E_s / (1 - nu_s^2)) * (E_s B^4 / (E_p I_p))^(1/12) / B

    Units: N/m^3 (pressure per unit settlement).
    """
    rigidity = young_modulus * width ** 4 / beam_ei
    return (coefficient * young_modulus / (1.0 - poisson_ratio ** 2) / width
            * rigidity ** (1.0 / 12.0))


def library_subgrade_modulus(young_modulus: float, poisson_ratio: float, width: float,
                             *, coefficient: float = VESIC_COEFFICIENT) -> float:
    """The merged `MohrCoulombSoil.subgrade_modulus` (Vesic without the rigidity factor)."""
    soil = MohrCoulombSoil(young_modulus=young_modulus, poisson_ratio=poisson_ratio,
                           subgrade_modulus_factor=coefficient)
    return soil.subgrade_modulus(width)


def equivalent_winkler_modulus(ei: float, target_moment: float, q: float,
                               patch_length: float, length: float, n_elem: int,
                               k_guess: float) -> float:
    """Winkler ``k_eff`` whose beam reproduces ``target_moment`` (bisection on log k)."""
    lo, hi = math.log(k_guess / 20.0), math.log(k_guess * 20.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m = solve_winkler(length, n_elem, ei, math.exp(mid), q, patch_length)["moment_max_Nm_per_m"]
        if m > target_moment:      # stiffer bed -> more concentrated reaction -> larger moment
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def rel_error(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def _round(x: float, nd: int = 4) -> float:
    return float(round(x, nd))


def main() -> dict:
    ei = STEEL_E * pipe_inertia(PIPE_D, PIPE_T)
    q = PATCH_TOTAL_LOAD / PATCH_LENGTH
    half_width = 0.5 * PIPE_D

    # -- 0. reference-solver self-checks -----------------------------------
    kernel_check = {}
    u0, u1 = 3.0, 4.0
    brute = _brute_quadrature(u0, u1, half_width)
    kernel_check["far_field_kernel_exact"] = strip_kernel(u0, u1, half_width)
    kernel_check["far_field_kernel_bruteforce"] = brute
    kernel_check["far_field_rel_error"] = _round(
        abs(kernel_check["far_field_kernel_exact"] - brute) / brute, 10)
    kernel_check["self_patch_kernel_m"] = _round(strip_kernel(-0.1, 0.1, half_width), 8)
    kernel_check["reciprocity_rel_error"] = _round(
        abs(compliance_matrix(60, 0.4, PIPE_D, SOIL_E, SOIL_NU)
            - compliance_matrix(60, 0.4, PIPE_D, SOIL_E, SOIL_NU).T).max()
        / compliance_matrix(60, 0.4, PIPE_D, SOIL_E, SOIL_NU).max(), 12)

    # -- 1. beam FE validated against the Hetenyi closed form (Winkler limit)
    k_lib = library_subgrade_modulus(SOIL_E, SOIL_NU, PIPE_D) * PIPE_D
    k_ves = vesic_subgrade_modulus(SOIL_E, SOIL_NU, PIPE_D, ei) * PIPE_D
    wink_lib = solve_winkler(BEAM_LENGTH, N_ELEM, ei, k_lib, q, PATCH_LENGTH)
    w0_ref, m0_ref = closed_form_patch(BEAM_LENGTH / N_ELEM, k_lib, ei, q, PATCH_LENGTH)
    fe_closed_form = {
        "closed_form_deflection_m": _round(w0_ref, 10),
        "fe_deflection_m": _round(wink_lib["deflection_max_m"], 10),
        "deflection_rel_error_pct": _round(
            rel_error(wink_lib["deflection_max_m"], w0_ref), 4),
        "closed_form_moment_Nm_per_m": _round(m0_ref, 2),
        "fe_moment_Nm_per_m": _round(wink_lib["moment_max_Nm_per_m"], 2),
        "moment_rel_error_pct": _round(
            rel_error(wink_lib["moment_max_Nm_per_m"], m0_ref), 4),
        "winkler_k_eff_N_per_m2": si(k_lib),
    }

    # -- 2. convergence of the coupled half-space reference ----------------
    convergence = []
    for length, n_elem in ((80.0, 400), (160.0, 800), (160.0, 400)):
        sol = solve_halfspace(length, n_elem, ei, PIPE_D, SOIL_E, SOIL_NU, q, PATCH_LENGTH)
        convergence.append({
            "length_m": length, "n_elem": n_elem, "element_m": length / n_elem,
            "differential_deflection_m": _round(sol["differential_deflection_m"], 10),
            "moment_max_Nm_per_m": _round(sol["moment_max_Nm_per_m"], 1),
            "equilibrium_rel_error": float(f"{sol['equilibrium_rel_error']:.2e}"),
            "compatibility_residual_m": float(f"{sol['compatibility_residual_m']:.2e}"),
        })

    # -- 3. reference case: half-space truth vs the two Winkler moduli -----
    half = solve_halfspace(BEAM_LENGTH, N_ELEM, ei, PIPE_D, SOIL_E, SOIL_NU, q, PATCH_LENGTH)
    wink_ves = solve_winkler(BEAM_LENGTH, N_ELEM, ei, k_ves, q, PATCH_LENGTH)
    k_equiv = equivalent_winkler_modulus(ei, half["moment_max_Nm_per_m"], q, PATCH_LENGTH,
                                         BEAM_LENGTH, N_ELEM, k_lib)
    wink_equiv = solve_winkler(BEAM_LENGTH, N_ELEM, ei, k_equiv, q, PATCH_LENGTH)

    reference = {
        "pipe_outer_diameter_m": PIPE_D, "pipe_wall_thickness_m": PIPE_T,
        "pipe_ei_Nm2": si(ei), "soil_E_Pa": SOIL_E, "soil_nu": SOIL_NU,
        "patch_total_load_N": PATCH_TOTAL_LOAD, "patch_length_m": PATCH_LENGTH,
        "patch_line_load_N_per_m": q,
        "rigidity_ratio": _round(SOIL_E * PIPE_D ** 4 / ei, 8),
        "rigidity_factor": _round((SOIL_E * PIPE_D ** 4 / ei) ** (1.0 / 12.0), 6),
        "k_library_N_per_m2": si(k_lib), "k_vesic_N_per_m2": si(k_ves),
        "k_equivalent_from_moment_N_per_m2": si(k_equiv),
        "k_library_over_equivalent": _round(k_lib / k_equiv, 4),
        "k_vesic_over_equivalent": _round(k_ves / k_equiv, 4),
        "table": [
            {"model": "3-D half-space (reference)",
             "deflection_mm": _round(1000 * half["differential_deflection_m"], 4),
             "moment_kNm_per_m": _round(half["moment_max_Nm_per_m"] / 1e3, 3),
             "contact_length_m": _round(half["contact_length_m"], 3),
             "k_eff_N_per_m2": None},
            {"model": "Winkler, merged k (no rigidity factor)",
             "deflection_mm": _round(1000 * wink_lib["differential_deflection_m"], 4),
             "moment_kNm_per_m": _round(wink_lib["moment_max_Nm_per_m"] / 1e3, 3),
             "contact_length_m": None, "k_eff_N_per_m2": si(k_lib)},
            {"model": "Winkler, Vesic (1961) k",
             "deflection_mm": _round(1000 * wink_ves["differential_deflection_m"], 4),
             "moment_kNm_per_m": _round(wink_ves["moment_max_Nm_per_m"] / 1e3, 3),
             "contact_length_m": None, "k_eff_N_per_m2": si(k_ves)},
            {"model": "Winkler, k inverted from the reference moment",
             "deflection_mm": _round(1000 * wink_equiv["differential_deflection_m"], 4),
             "moment_kNm_per_m": _round(wink_equiv["moment_max_Nm_per_m"] / 1e3, 3),
             "contact_length_m": None, "k_eff_N_per_m2": si(k_equiv)},
        ],
        "errors_vs_halfspace": [
            {"model": "merged k", "deflection_pct": _round(rel_error(
                wink_lib["differential_deflection_m"],
                half["differential_deflection_m"]), 2),
             "moment_pct": _round(rel_error(
                 wink_lib["moment_max_Nm_per_m"], half["moment_max_Nm_per_m"]), 2)},
            {"model": "Vesic k", "deflection_pct": _round(rel_error(
                wink_ves["differential_deflection_m"],
                half["differential_deflection_m"]), 2),
             "moment_pct": _round(rel_error(
                 wink_ves["moment_max_Nm_per_m"], half["moment_max_Nm_per_m"]), 2)},
            {"model": "inverted k (floor for any Winkler fit)", "deflection_pct": _round(rel_error(
                wink_equiv["differential_deflection_m"],
                half["differential_deflection_m"]), 2),
             "moment_pct": _round(rel_error(
                 wink_equiv["moment_max_Nm_per_m"], half["moment_max_Nm_per_m"]), 2)},
        ],
        "profiles": {"halfspace": half["profile"],
                     "winkler_merged": _winkler_profile(ei, k_lib, q),
                     "winkler_vesic": _winkler_profile(ei, k_ves, q)},
        "min_pressure_halfspace_Pa": _round(half["min_pressure_Pa"], 1),
    }

    # -- 4. swept soil modulus and pipe size -------------------------------
    soil_sweep = []
    for e_s in (2.0e6, 5.0e6, 10.0e6, 20.0e6, 50.0e6):
        kl = library_subgrade_modulus(e_s, SOIL_NU, PIPE_D) * PIPE_D
        kv = vesic_subgrade_modulus(e_s, SOIL_NU, PIPE_D, ei) * PIPE_D
        hs = solve_halfspace(BEAM_LENGTH, N_ELEM, ei, PIPE_D, e_s, SOIL_NU, q, PATCH_LENGTH)
        wl = solve_winkler(BEAM_LENGTH, N_ELEM, ei, kl, q, PATCH_LENGTH)
        wv = solve_winkler(BEAM_LENGTH, N_ELEM, ei, kv, q, PATCH_LENGTH)
        soil_sweep.append({
            "soil_E_MPa": e_s / 1e6,
            "rigidity_factor": _round((e_s * PIPE_D ** 4 / ei) ** (1.0 / 12.0), 5),
            "halfspace_deflection_mm": _round(1000 * hs["differential_deflection_m"], 3),
            "merged_deflection_error_pct": _round(rel_error(
                wl["differential_deflection_m"], hs["differential_deflection_m"]), 2),
            "vesic_deflection_error_pct": _round(rel_error(
                wv["differential_deflection_m"], hs["differential_deflection_m"]), 2),
            "merged_moment_error_pct": _round(rel_error(
                wl["moment_max_Nm_per_m"], hs["moment_max_Nm_per_m"]), 2),
            "vesic_moment_error_pct": _round(rel_error(
                wv["moment_max_Nm_per_m"], hs["moment_max_Nm_per_m"]), 2),
        })

    pipe_sweep = []
    for label, d, t in (("8 in", 0.2032, 0.0127), ("18 in", 0.4572, 0.0127),
                        ("36 in", PIPE_D, PIPE_T)):
        ei_p = STEEL_E * pipe_inertia(d, t)
        kl = library_subgrade_modulus(SOIL_E, SOIL_NU, d) * d
        kv = vesic_subgrade_modulus(SOIL_E, SOIL_NU, d, ei_p) * d
        pipe_sweep.append({
            "pipe": label, "outer_diameter_m": d, "wall_mm": 1000 * t,
            "rigidity_ratio": float(f"{SOIL_E * d ** 4 / ei_p:.5g}"),
            "rigidity_factor": _round((SOIL_E * d ** 4 / ei_p) ** (1.0 / 12.0), 5),
            "k_library_over_vesic": _round(kl / kv, 4),
            "stiffness_bias_pct": _round(100.0 * (kl / kv - 1.0), 2),
        })

    result = {
        "protocol": "tensordem-ridge-winkler-subgrade-verification-v1",
        "kernel_checks": kernel_check,
        "beam_fe_vs_closed_form": fe_closed_form,
        "halfspace_convergence": convergence,
        "reference_case": reference,
        "soil_modulus_sweep": soil_sweep,
        "pipe_size_sweep": pipe_sweep,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subgrade_modulus_verification.json").write_text(json.dumps(result, indent=2))
    _figure(result)
    _inject_tables(result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("reference_case",)}, indent=2)[:4000])
    print("reference case:", json.dumps(reference["errors_vs_halfspace"], indent=2))
    return result


def si(value: float) -> float:
    return float(f"{value:.6g}")


def closed_form_patch(le: float, k_eff: float, ei: float, q: float,
                      patch_length: float) -> tuple[float, float]:
    """Hetenyi (1946) infinite beam on a Winkler bed, uniform patch 2a (exact).

    Superposing the point-load solution ``w = P beta/(2k) A(beta x)``,
    ``M = P/(4 beta) C(beta x)`` with ``A(u) = e^-u (cos u + sin u)`` and
    ``C(u) = e^-u (cos u - sin u)`` gives, at the patch centre,

        w(0) = (q/k) (1 - e^-ba cos ba),  M(0) = (q/(2 beta^2)) e^-ba sin ba,

    with ``b = beta`` and ``a = patch_length/2``.  Derived here from the
    point-load solution, independent of the finite-element assembly.
    """
    beta = (k_eff / (4.0 * ei)) ** 0.25
    ba = beta * 0.5 * patch_length
    decay = math.exp(-ba)
    w0 = (q / k_eff) * (1.0 - decay * math.cos(ba))
    m0 = (q / (2.0 * beta ** 2)) * decay * math.sin(ba)
    return float(w0), float(m0)


def _brute_quadrature(u0: float, u1: float, half_width: float, order: int = 400) -> float:
    """2-D Gauss-Legendre check of :func:`strip_kernel` (non-singular pairs only)."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    us = 0.5 * (u1 - u0) * nodes + 0.5 * (u1 + u0)
    ts = half_width * nodes
    wu = 0.5 * (u1 - u0) * weights
    wt = half_width * weights
    r = np.hypot(us[:, None], ts[None, :])
    return float(np.sum((1.0 / r) * wu[:, None] * wt[None, :]))


def _winkler_profile(ei: float, k_eff: float, q: float) -> dict:
    le = BEAM_LENGTH / N_ELEM
    first, count = _patch_geometry(N_ELEM, PATCH_LENGTH, BEAM_LENGTH)
    K = assemble(N_ELEM, beam_stiffness(ei, le) + foundation_stiffness(k_eff, le))
    d = np.linalg.solve(K, patch_load_vector(N_ELEM, le, q, first, count))
    return {"x": ((np.arange(N_ELEM) + 0.5) * le).tolist(),
            "deflection": (midpoint_map(N_ELEM, le) @ d).tolist(),
            "pressure": (k_eff * (midpoint_map(N_ELEM, le) @ d)).tolist()}


def _figure(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = result["reference_case"]["profiles"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    ax = axes[0, 0]
    for key, label, style in (("halfspace", "3-D half-space (reference)", "-"),
                              ("winkler_merged", "Winkler, merged k", "--"),
                              ("winkler_vesic", "Winkler, Vesic k", "-.")):
        x = np.asarray(prof[key]["x"]) - 0.5 * BEAM_LENGTH
        w = np.asarray(prof[key]["deflection"])
        w = w - w[int(np.argmin(np.abs(np.abs(x) - DIFF_SPAN)))]
        ax.plot(x, w * 1e3, style, label=label)
    ax.set_xlim(-8, 8)
    ax.set_xlabel("distance along pipe x [m]")
    ax.set_ylabel("differential deflection [mm]")
    ax.set_title("(a) deflection, ridge-keel patch load")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for key, label, style in (("halfspace", "3-D half-space (reference)", "-"),
                              ("winkler_merged", "Winkler, merged k", "--"),
                              ("winkler_vesic", "Winkler, Vesic k", "-.")):
        x = np.asarray(prof[key]["x"]) - 0.5 * BEAM_LENGTH
        ax.plot(x, np.asarray(prof[key]["pressure"]) / 1e3, style, label=label)
    ax.set_xlim(-8, 8)
    ax.set_xlabel("distance along pipe x [m]")
    ax.set_ylabel("contact pressure [kPa]")
    ax.set_title("(b) contact pressure along the pipe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    sweep = result["soil_modulus_sweep"]
    es = [row["soil_E_MPa"] for row in sweep]
    ax.plot(es, [row["merged_deflection_error_pct"] for row in sweep], "o--",
            label="merged k: deflection error")
    ax.plot(es, [row["vesic_deflection_error_pct"] for row in sweep], "s-",
            label="Vesic k: deflection error")
    ax.plot(es, [row["merged_moment_error_pct"] for row in sweep], "o--",
            label="merged k: moment error")
    ax.plot(es, [row["vesic_moment_error_pct"] for row in sweep], "s-",
            label="Vesic k: moment error")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(15.0, color="r", lw=0.6, ls=":")
    ax.axhline(-15.0, color="r", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("soil Young's modulus E_s [MPa]")
    ax.set_ylabel("error vs half-space [%]")
    ax.set_title("(c) error of both moduli over the soil range (36 in pipe)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    pipes = result["pipe_size_sweep"]
    labels = [f"{row['pipe']}\nD={row['outer_diameter_m']:.3f} m" for row in pipes]
    vals = [row["stiffness_bias_pct"] for row in pipes]
    ax.bar(range(len(vals)), vals, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("merged k vs Vesic k [% too stiff]")
    ax.set_title("(d) omitted rigidity factor by pipe size (E_s = 10 MPa)")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.1f}%", (i, v), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "subgrade_modulus_verification.png", dpi=150)
    plt.close(fig)


def _inject_tables(result: dict) -> None:
    ref = result["reference_case"]
    rows = ["| model | max deflection [mm] | max moment [kNm/m] | k_eff [MN/m^2] |",
            "|---|---|---|---|"]
    for row in ref["table"]:
        k = "—" if row["k_eff_N_per_m2"] is None else f"{row['k_eff_N_per_m2'] / 1e6:.3f}"
        rows.append(f"| {row['model']} | {row['deflection_mm']:.4f} | "
                    f"{row['moment_kNm_per_m']:.3f} | {k} |")
    rows += ["", "| model | deflection error vs reference | moment error vs reference |",
             "|---|---|---|"]
    for row in ref["errors_vs_halfspace"]:
        rows.append(f"| {row['model']} | {row['deflection_pct']:+.2f} % | "
                    f"{row['moment_pct']:+.2f} % |")
    rows += ["", "| soil E_s [MPa] | rigidity factor | merged k deflection err | "
                 "Vesic k deflection err | merged k moment err | Vesic k moment err |",
             "|---|---|---|---|---|---|"]
    for row in result["soil_modulus_sweep"]:
        rows.append(f"| {row['soil_E_MPa']:.0f} | {row['rigidity_factor']:.4f} | "
                    f"{row['merged_deflection_error_pct']:+.1f} % | "
                    f"{row['vesic_deflection_error_pct']:+.1f} % | "
                    f"{row['merged_moment_error_pct']:+.1f} % | "
                    f"{row['vesic_moment_error_pct']:+.1f} % |")
    rows += ["", "| pipe | D [m] | t [mm] | rigidity factor | merged k / Vesic k |",
             "|---|---|---|---|---|"]
    for row in result["pipe_size_sweep"]:
        rows.append(f"| {row['pipe']} | {row['outer_diameter_m']:.3f} | "
                    f"{row['wall_mm']:.1f} | {row['rigidity_factor']:.4f} | "
                    f"{row['k_library_over_vesic']:.3f} |")
    rows += ["", "| half-space convergence | L [m] | n_elem | element [m] | "
                 "w_max [mm] | M_max [kNm/m] | equilibrium err |",
             "|---|---|---|---|---|---|---|"]
    for row in result["halfspace_convergence"]:
        rows.append(f"| | {row['length_m']:.0f} | {row['n_elem']} | "
                    f"{row['element_m']:.2f} | {1000 * row['differential_deflection_m']:.4f} | "
                    f"{row['moment_max_Nm_per_m'] / 1e3:.3f} | "
                    f"{row['equilibrium_rel_error']:.1e} |")
    text = "\n".join(rows)
    if not DOC.exists():
        return
    doc = DOC.read_text()
    start = "<!-- AUTOGEN:RESULT_TABLES -->"
    end = "<!-- /AUTOGEN -->"
    if start in doc and end in doc:
        head, rest = doc.split(start, 1)
        _, tail = rest.split(end, 1)
        DOC.write_text(f"{head}{start}\n{text}\n{end}{tail}")


if __name__ == "__main__":
    main()
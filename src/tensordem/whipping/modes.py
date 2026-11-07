"""Dry and wet vertical-bending modes of a ship hull girder.

The hull girder is modelled as a free-free uniform Euler-Bernoulli beam
discretised with two-node Hermite cubic elements (2 DOF per node, consistent
mass matrix).  The elastic modes are recovered from the generalised
eigenproblem ``K phi = omega^2 M phi`` and are mass-normalised so that the
reduced mass of the structure is the identity.

Wet modes are obtained by a Ritz reduction of the added-mass operator onto the
dry basis,

    (I + Phi^T M_a Phi) a = (1 / omega_wet^2) diag(omega_dry^2) a ,

where ``M_a`` is assembled from the section added mass per unit length
``m_a(x)`` weighted by the mode shapes.  ``m_a(x)`` comes from the 2-D boundary
element section solver (:mod:`tensordem.whipping.sections`) at the section's
local high-frequency limit, optionally multiplied by a longitudinal
three-dimensional relief factor (``relief``) that accounts for the finite
length of the hull: a 2-D strip model is a rigorous *upper bound* for the
entrained water, because along a real hull the flow can escape past the fine
fore/aft bodies.  ``relief = 1`` therefore is the strip upper bound and
``relief < 1`` a 3-D-relieved estimate; the sensitivity of the wet frequencies
to ``relief`` is reported with the benchmark rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "DryModes",
    "WetModes",
    "beam_fem_matrices",
    "free_free_analytic_frequencies",
    "dry_modes",
    "modal_added_mass",
    "wet_modes",
    "taper_breadth",
    "taper_draft",
]

# first four roots of cosh(beta L) cos(beta L) = 1 (free-free beam)
_FREE_FREE_BETA_L = (4.730040744862704, 7.853204624095838,
                     10.995607838001671, 14.137165491257464,
                     17.278759657399481)


def beam_fem_matrices(EI: float, m_bar: float, L: float, n_elements: int):
    """Assemble the free-free Euler-Bernoulli beam stiffness and mass matrices."""
    for name, value in (("EI", EI), ("m_bar", m_bar), ("L", L)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if n_elements < 4:
        raise ValueError("n_elements must be >= 4")
    le = L / n_elements
    n_dof = 2 * (n_elements + 1)
    k_local = EI / le**3 * np.array(
        [[12.0, 6 * le, -12.0, 6 * le],
         [6 * le, 4 * le**2, -6 * le, 2 * le**2],
         [-12.0, -6 * le, 12.0, -6 * le],
         [6 * le, 2 * le**2, -6 * le, 4 * le**2]])
    m_local = m_bar * le / 420.0 * np.array(
        [[156.0, 22 * le, 54.0, -13 * le],
         [22 * le, 4 * le**2, 13 * le, -3 * le**2],
         [54.0, 13 * le, 156.0, -22 * le],
         [-13 * le, -3 * le**2, -22 * le, 4 * le**2]])
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    for e in range(n_elements):
        idx = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[np.ix_(idx, idx)] += k_local
        M[np.ix_(idx, idx)] += m_local
    x_nodes = np.linspace(0.0, L, n_elements + 1)
    return K, M, x_nodes


def free_free_analytic_frequencies(EI: float, m_bar: float, L: float,
                                   n_modes: int = 4) -> np.ndarray:
    """Closed-form free-free Euler-Bernoulli natural frequencies (Hz)."""
    out = []
    for i in range(n_modes):
        if i < len(_FREE_FREE_BETA_L):
            b = _FREE_FREE_BETA_L[i]
        else:  # asymptotic root of cos(beta L) = 0
            b = (i + 1.5) * math.pi
        out.append(b * b / (2.0 * math.pi * L * L) * math.sqrt(EI / m_bar))
    return np.array(out)


def _hermite(x_nodes: np.ndarray, w: np.ndarray, theta: np.ndarray,
             xq: np.ndarray, deriv: int = 0) -> np.ndarray:
    """Evaluate a Hermite cubic field (or its 1st/2nd derivative) at ``xq``."""
    i = np.clip(np.searchsorted(x_nodes, xq) - 1, 0, len(x_nodes) - 2)
    le = x_nodes[1] - x_nodes[0]
    s = (xq - x_nodes[i]) / le
    w0, t0, w1, t1 = w[i], theta[i], w[i + 1], theta[i + 1]
    if deriv == 0:
        n = (1 - 3 * s**2 + 2 * s**3, le * (s - 2 * s**2 + s**3),
             3 * s**2 - 2 * s**3, le * (-s**2 + s**3))
        return n[0] * w0 + n[1] * t0 + n[2] * w1 + n[3] * t1
    if deriv == 1:
        n = ((-6 * s + 6 * s**2) / le, 1 - 4 * s + 3 * s**2,
             (6 * s - 6 * s**2) / le, -2 * s + 3 * s**2)
        return n[0] * w0 + n[1] * t0 + n[2] * w1 + n[3] * t1
    n = (-6 + 12 * s) / le**2, (-4 + 6 * s) / le, (6 - 12 * s) / le**2, (-2 + 6 * s) / le
    return n[0] * w0 + n[1] * t0 + n[2] * w1 + n[3] * t1


@dataclass(frozen=True)
class DryModes:
    """Mass-normalised dry modes of the free-free hull girder."""

    length: float
    x_nodes: np.ndarray
    w: np.ndarray          # (n_modes, n_nodes) displacements
    theta: np.ndarray      # (n_modes, n_nodes) slopes
    freqs: np.ndarray      # Hz

    def shape(self, xq, mode: int, deriv: int = 0) -> np.ndarray:
        return _hermite(self.x_nodes, self.w[mode], self.theta[mode],
                        np.atleast_1d(np.asarray(xq, dtype=float)), deriv)


def dry_modes(EI: float, m_bar: float, L: float, n_elements: int = 60,
              n_modes: int = 4) -> DryModes:
    """Elastic dry modes (rigid translation/rotation removed)."""
    K, M, x_nodes = beam_fem_matrices(EI, m_bar, L, n_elements)
    # symmetric reduction: M = L L^T, A = L^-1 K L^-T is symmetric
    chol = np.linalg.cholesky(M)
    a_mat = np.linalg.solve(chol, np.linalg.solve(chol, K).T)
    a_mat = 0.5 * (a_mat + a_mat.T)
    lam, vec = np.linalg.eigh(a_mat)
    order = np.argsort(lam)
    lam, vec = lam[order], vec[:, order]
    # exactly two rigid-body modes (free translation + free rotation)
    if lam[1] > 1e-10 * lam.max():
        raise ValueError("rigid-body modes not resolved as the lowest two eigenvalues")
    lam = lam[2:]
    # phi = L^-T y brings the eigenvectors back to physical DOFs
    vec = np.linalg.solve(chol.T, vec[:, 2:])
    w = np.zeros((n_modes, x_nodes.size))
    theta = np.zeros((n_modes, x_nodes.size))
    freqs = np.zeros(n_modes)
    for j in range(n_modes):
        phi = vec[:, j]
        phi = phi / math.sqrt(float(phi @ M @ phi))
        w[j] = phi[0::2]
        theta[j] = phi[1::2]
        freqs[j] = math.sqrt(max(lam[j], 0.0)) / (2.0 * math.pi)
    return DryModes(length=L, x_nodes=x_nodes, w=w, theta=theta, freqs=freqs)


def taper_breadth(x: np.ndarray, L: float, B_max: float, exponent: float = 0.55) -> np.ndarray:
    """Waterline breadth distribution: ``B_max (1 - (2x/L - 1)^2)^exponent``."""
    s = 2.0 * np.asarray(x, dtype=float) / L - 1.0
    return B_max * np.clip(1.0 - s * s, 0.0, None) ** exponent


def taper_draft(x: np.ndarray, L: float, T_max: float, exponent: float = 0.45) -> np.ndarray:
    """Draft distribution with the same parabolic taper family."""
    s = 2.0 * np.asarray(x, dtype=float) / L - 1.0
    return T_max * np.clip(1.0 - s * s, 0.0, None) ** exponent


def modal_added_mass(dry: DryModes, m_a_of_x, n_query: int = 2001) -> np.ndarray:
    """Reduce the section added mass per unit length onto the dry basis.

    ``m_a_of_x`` is a callable returning kg/m for an array of longitudinal
    positions.  The integral ``int m_a(x) phi_i(x) phi_j(x) dx`` is evaluated
    with composite Simpson on a uniform grid.
    """
    xq = np.linspace(0.0, dry.length, n_query)
    m_a = np.asarray(m_a_of_x(xq), dtype=float)
    n_modes = dry.freqs.size
    phi = np.array([dry.shape(xq, i) for i in range(n_modes)])
    weighted = m_a[None, :] * phi
    out = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(i, n_modes):
            out[i, j] = out[j, i] = _simpson(weighted[i] * phi[j], xq)
    return out


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    n = y.size
    if n % 2 == 0:
        return float(np.trapezoid(y, x))
    h = x[1] - x[0]
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-2:2].sum()))


@dataclass(frozen=True)
class WetModes:
    """Wet modes in the dry basis."""

    freqs: np.ndarray
    freqs_dry: np.ndarray
    reduction: np.ndarray      # 1 - f_wet / f_dry
    added_mass: np.ndarray     # reduced added-mass matrix
    coefficients: np.ndarray   # wet shapes in the dry basis (n_modes x n_modes)
    dry: DryModes

    def shape(self, xq, mode: int, deriv: int = 0) -> np.ndarray:
        acc = np.zeros_like(np.atleast_1d(np.asarray(xq, dtype=float)))
        for k in range(self.dry.freqs.size):
            acc = acc + self.coefficients[k, mode] * self.dry.shape(xq, k, deriv)
        return acc


def wet_modes(dry: DryModes, m_a_of_x) -> WetModes:
    """Solve the wet eigenproblem in the dry modal basis."""
    m_add = modal_added_mass(dry, m_a_of_x)
    omega_dry2 = (2.0 * math.pi * dry.freqs) ** 2
    reduced = np.eye(dry.freqs.size) + m_add
    lam, vec = np.linalg.eigh(np.linalg.solve(reduced, np.diag(omega_dry2)))
    order = np.argsort(lam)
    lam, vec = lam[order], vec[:, order]
    freqs = np.sqrt(np.maximum(lam, 0.0)) / (2.0 * math.pi)
    reduction = 1.0 - freqs / dry.freqs
    return WetModes(freqs=freqs, freqs_dry=dry.freqs, reduction=reduction,
                    added_mass=m_add, coefficients=vec, dry=dry)
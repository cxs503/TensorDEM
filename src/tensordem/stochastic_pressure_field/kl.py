"""Karhunen-Loeve expansion of the fitted pressure-field covariance.

For a separable covariance ``C(dx, dy) = sigma^2 Rx(dx) Ry(dy)`` the eigenproblem
on a tensor grid factorises exactly: with ``Rx = Ux Lx Ux^T`` and
``Ry = Uy Ly Uy^T``

.. math::

    C = \\sigma^2 (Rx \\otimes Ry) = \\sigma^2 (Ux \\otimes Uy)
        (Lx \\otimes Ly) (Ux \\otimes Uy)^T

so the 2D eigenpairs are ``lambda_ij = sigma^2 lx_i ly_j`` and
``phi_ij = ux_i \\otimes uy_j`` - no approximation beyond the discretisation of
the 1D kernels, and no dense ``(nx*ny)^2`` decomposition.  That is what makes
the truncation auditable: the modes are exactly the eigenfunctions of the
fitted model, and the retained-energy fraction is the exact fraction of the
discrete variance they carry.

The truncation keeps the smallest number of modes whose cumulative eigenvalue
sum reaches ``energy_target`` (and never more than ``n_modes_max``).  If the
target cannot be reached inside the cap, the expansion reports the energy it
actually carries instead of silently truncating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .correlation import matern_1d

__all__ = ["KLExpansion2D", "kl_expansion_separable"]


@dataclass(frozen=True)
class KLExpansion2D:
    """Truncated KL expansion of a 2D random field on a tensor grid."""

    x: np.ndarray                 # (nx,)
    y: np.ndarray                 # (ny,)
    mean_field: np.ndarray        # (ny, nx)
    eigenvalues: np.ndarray       # (M,)
    modes: np.ndarray             # (M, ny, nx)
    variance: float
    energy_fraction: float        # retained / total discrete variance
    energy_target: float
    n_modes_available: int
    n_modes: int
    length_x: float
    length_y: float
    nu: float
    model: str
    periodic: bool = True

    # -- sampling ---------------------------------------------------------
    def sample(self, n_samples: int, *, rng: Optional[np.random.Generator] = None,
               seed: int = 20271109, include_mean: bool = True,
               moment_match: bool = False) -> np.ndarray:
        """Draw ``n_samples`` realisations, shape ``(n_samples, ny, nx)``.

        ``moment_match`` centres and scales each coefficient column to zero
        sample mean and unit sample standard deviation.  That is a variance
        reduction used only for the *statistics verification* ensembles: it
        removes the O(1/sqrt(N)) Monte Carlo noise from the first two moments,
        so what remains in the check is the model (its truncation) rather than
        the luck of the draw.  Production sampling leaves it off.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        gen = rng if rng is not None else np.random.default_rng(seed)
        xi = gen.standard_normal((int(n_samples), self.n_modes))
        if moment_match and n_samples > 1:
            xi = xi - xi.mean(axis=0, keepdims=True)
            sd = xi.std(axis=0, keepdims=True)
            xi = xi / np.where(sd > 1e-12, sd, 1.0)
        flat = (xi * np.sqrt(self.eigenvalues)[None, :]) @ \
            self.modes.reshape(self.n_modes, -1)
        fields = flat.reshape(int(n_samples), self.y.size, self.x.size)
        if include_mean:
            fields = fields + self.mean_field[None, :, :]
        return fields

    def target_covariance(self) -> np.ndarray:
        """Discrete covariance implied by the fitted model (for verification)."""
        dx = np.abs(self.x[:, None] - self.x[None, :])
        dy = np.abs(self.y[:, None] - self.y[None, :])
        if self.periodic:
            dx = np.minimum(dx, (self.x[-1] - self.x[0]) + (self.x[1] - self.x[0]) - dx)
            dy = np.minimum(dy, (self.y[-1] - self.y[0]) + (self.y[1] - self.y[0]) - dy)
        if self.model == "exponential" or self.nu == 0.5:
            cx = np.exp(-dx / self.length_x)
            cy = np.exp(-dy / self.length_y)
        else:
            cx = matern_1d(dx, self.length_x, self.nu)
            cy = matern_1d(dy, self.length_y, self.nu)
        return self.variance * np.kron(cx, cy)

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_modes": int(self.n_modes),
            "n_modes_available": int(self.n_modes_available),
            "energy_fraction": float(self.energy_fraction),
            "energy_target": float(self.energy_target),
            "variance": float(self.variance),
            "length_x": float(self.length_x),
            "length_y": float(self.length_y),
            "nu": float(self.nu),
            "model": self.model,
            "eigenvalues": [float(v) for v in self.eigenvalues],
        }


def kl_expansion_separable(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model: str = "matern",
    length_x: float,
    length_y: float,
    nu: float = 0.5,
    variance: float,
    mean_field: Optional[np.ndarray] = None,
    energy_target: float = 0.95,
    n_modes_max: int = 50,
    periodic: bool = True,
) -> KLExpansion2D:
    """Build the truncated KL expansion of the fitted separable covariance.

    ``mean_field`` (``(ny, nx)``) is carried through unchanged, so a generated
    sample is a full pressure field and not only its fluctuation.

    ``periodic`` (default) uses the minimum-image distance, i.e. the covariance
    of a stationary field on a torus.  That is the correct choice here for two
    reasons: the DEM cell itself is periodic, so the measured correlation is a
    circulant statistic; and a KL expansion on a finite non-periodic grid has
    eigenfunctions that vanish at the edges, which makes the realised field
    non-stationary (its variance tapers within one correlation length of the
    boundary) and biases the correlation estimate at large lags.  With
    ``periodic=True`` the realised covariance is exactly the (circulant) model
    covariance, so the generated ensemble can be verified against the target
    without an edge-effect correction.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or y.size < 2:
        raise ValueError("need at least two grid points per direction")
    if variance <= 0.0:
        raise ValueError("variance must be positive")
    if not (0.0 < energy_target <= 1.0):
        raise ValueError("energy_target must be in (0, 1]")

    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    if periodic:
        span_x = float(x[-1] - x[0]) + (x[1] - x[0])
        span_y = float(y[-1] - y[0]) + (y[1] - y[0])
        dx = np.minimum(dx, span_x - dx)
        dy = np.minimum(dy, span_y - dy)
    if model == "exponential" or nu == 0.5:
        cx = np.exp(-dx / max(length_x, 1e-12))
        cy = np.exp(-dy / max(length_y, 1e-12))
        nu_used = 0.5
    elif model in ("matern", "matern_radial"):
        cx = matern_1d(dx, length_x, nu)
        cy = matern_1d(dy, length_y, nu)
        nu_used = float(nu)
    else:
        raise ValueError(f"unknown correlation model: {model}")

    lx, ux = np.linalg.eigh(cx)
    ly, uy = np.linalg.eigh(cy)
    lx = np.clip(lx, 0.0, None)
    ly = np.clip(ly, 0.0, None)
    # descending order
    ix = np.argsort(lx)[::-1]
    iy = np.argsort(ly)[::-1]
    lx, ux = lx[ix], ux[:, ix]
    ly, uy = ly[iy], uy[:, iy]

    eig = np.outer(lx, ly) * variance                  # (nx, ny), descending in each axis
    order = np.argsort(eig, axis=None)[::-1]
    eig_sorted = eig.ravel()[order]
    total = float(np.sum(eig_sorted))
    if total <= 0.0:
        raise ValueError("degenerate covariance")

    cum = np.cumsum(eig_sorted) / total
    n_keep = int(np.searchsorted(cum, energy_target) + 1)
    n_keep = min(n_keep, int(n_modes_max), eig_sorted.size)
    energy = float(np.sum(eig_sorted[:n_keep]) / total)

    modes = np.empty((n_keep, y.size, x.size))
    for k, flat in enumerate(order[:n_keep]):
        i_x, i_y = np.unravel_index(flat, (x.size, y.size))
        modes[k] = np.outer(uy[:, i_y], ux[:, i_x])

    if mean_field is None:
        mean = np.zeros((y.size, x.size))
    else:
        mean = np.asarray(mean_field, dtype=float)
        if mean.shape != (y.size, x.size):
            raise ValueError("mean_field must have shape (ny, nx)")

    return KLExpansion2D(
        x=x, y=y, mean_field=mean,
        eigenvalues=np.clip(eig_sorted[:n_keep], 0.0, None),
        modes=modes,
        variance=float(variance),
        energy_fraction=energy,
        energy_target=float(energy_target),
        n_modes_available=int(eig_sorted.size),
        n_modes=int(n_keep),
        length_x=float(length_x),
        length_y=float(length_y),
        nu=float(nu_used),
        model=model,
        periodic=bool(periodic),
    )
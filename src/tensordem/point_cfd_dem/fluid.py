"""Triply periodic pseudo-spectral incompressible fluid with particle sources.

Why a spectral solver
---------------------
The three verification faces of a point-particle CFD-DEM model (unbounded
Stokes terminal velocity, inviscid added-mass oscillation, momentum-exchange
conservation) are all defined in an *unbounded* fluid or in a *periodic* cell
without walls.  A Fourier pseudo-spectral discretisation is the natural fit:

* viscosity is applied exactly in Fourier space (integrating factor), so the
  Stokes limit has no time-step error in the diffusion operator;
* the Leray projection only touches ``k != 0`` modes, therefore the
  volume-averaged (``k = 0``) momentum is preserved **exactly**.  This is what
  makes the ``<= 1e-6`` momentum-exchange gate a structural property of the
  scheme rather than a tolerance tuning exercise;
* the trilinear particle-grid operators are built as an exact adjoint pair
  (interpolation ``I`` and spreading ``S = I^T``), so the interphase momentum
  transfer summed over the grid equals the particle force to round-off.

Fields are collocated (node centred) on a uniform grid of spacing ``h`` with
``n`` nodes per direction; the domain length is ``L = n h`` and the grid is
periodic in all three directions.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SpectralPeriodicFluid", "trilinear_weights"]

#: unit-cube corner offsets of the trilinear stencil
_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
     [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=int
)


def trilinear_weights(position, shape, h):
    """Periodic trilinear stencil: corner indices and weights.

    Parameters
    ----------
    position : (N, 3) array of particle positions (domain box ``[0, n*h)``)
    shape : (nx, ny, nz) grid node counts
    h : grid spacing

    Returns
    -------
    idx : (8, N, 3) int array of node indices
    w : (8, N) float array of weights, ``sum(w, axis=0) == 1`` exactly
    """
    pos = np.atleast_2d(np.asarray(position, dtype=float))
    shape = np.asarray(shape, dtype=int)
    s = pos / float(h)
    floor = np.floor(s)
    frac = s - floor
    base = floor.astype(int) % shape
    idx = np.empty((8, pos.shape[0], 3), dtype=int)
    w = np.empty((8, pos.shape[0]), dtype=float)
    for k, c in enumerate(_CORNERS):
        c = np.asarray(c)
        idx[k] = (base + c) % shape
        w[k] = np.prod(np.where(c == 1, frac, 1.0 - frac), axis=1)
    return idx, w


class SpectralPeriodicFluid:
    """Incompressible Newtonian fluid, triply periodic, pseudo-spectral.

    Parameters
    ----------
    shape : (nx, ny, nz) int
    h : grid spacing
    rho : fluid density
    mu : dynamic viscosity (``mu = 0`` gives the inviscid / Euler limit)
    advect : include the nonlinear term (``False`` for pure Stokes problems)
    dealias : apply the 2/3 rule to the advective products
    """

    def __init__(self, shape, h, rho=1.0, mu=0.0, advect=True, dealias=True):
        self.shape = tuple(int(n) for n in shape)
        self.h = float(h)
        self.rho = float(rho)
        self.mu = float(mu)
        self.nu = self.mu / self.rho
        self.advect = bool(advect)
        self.dealias = bool(dealias)

        n = np.asarray(self.shape)
        self.extent = n * self.h
        self.cell_volume = self.h ** 3

        freq = [np.fft.fftfreq(ni, d=self.h) for ni in self.shape]
        self.k = [2.0 * np.pi * np.fft.fftfreq(ni, d=self.h) for ni in self.shape]
        kx, ky, kz = np.meshgrid(*self.k, indexing="ij")
        self.kgrid = (kx, ky, kz)
        self.k2 = kx ** 2 + ky ** 2 + kz ** 2
        self._k2_safe = self.k2.copy()
        self._k2_safe[0, 0, 0] = 1.0
        self._k2_zero = self.k2 < 1e-30
        #: Even grids have a self-paired Nyquist plane where the odd symmetry
        #: ``k(-k) = -k(k)`` fails, which would make the Leray projection
        #: non-Hermitian (and leak a spurious divergence).  Those modes are the
        #: worst resolved ones; they are removed from the velocity field before
        #: projecting (a standard pseudo-spectral safeguard).
        nyq = np.ones(self.shape, dtype=bool)
        for ax, ni in enumerate(self.shape):
            kk = np.abs(np.fft.fftfreq(ni))
            sl = [slice(None)] * 3
            sl[ax] = kk >= 0.5 - 1e-12
            nyq[tuple(sl)] = False
        self._non_nyquist = nyq

        if self.dealias:
            mask = np.ones(self.shape, dtype=bool)
            for ax, ni in enumerate(self.shape):
                kk = np.abs(np.fft.fftfreq(ni))
                thr = 1.0 / 3.0
                sl = [slice(None)] * 3
                sl[ax] = kk > thr
                mask[tuple(sl)] = False
            self._dealias_mask = mask
        else:
            self._dealias_mask = np.ones(self.shape, dtype=bool)

        self.freq = freq

    # ---------------------------------------------------------------- helpers
    def zeros(self):
        """Zero velocity field ``(3, nx, ny, nz)``."""
        return np.zeros((3,) + self.shape, dtype=float)

    def fft(self, a):
        return np.fft.fftn(a)

    def ifft(self, ah):
        return np.fft.ifftn(ah).real

    def project(self, u):
        """Leray projection onto the divergence-free subspace (spectral, exact).

        The ``k = 0`` component is untouched, hence the mean momentum of the
        field is preserved bit-for-bit apart from floating-point round-off.
        """
        uh = [self.fft(u[i]) * self._non_nyquist for i in range(3)]
        kdot = uh[0] * self.kgrid[0] + uh[1] * self.kgrid[1] + uh[2] * self.kgrid[2]
        corr = kdot / self._k2_safe
        corr[0, 0, 0] = 0.0
        out = np.empty_like(u)
        for i in range(3):
            out[i] = self.ifft(uh[i] - self.kgrid[i] * corr)
        return out

    def divergence(self, u):
        """Spectral divergence of a vector field ``(3, ...)``."""
        div = np.zeros(self.shape, dtype=complex)
        for i in range(3):
            div = div + 1j * self.kgrid[i] * self.fft(u[i])
        return self.ifft(div)

    def gradient(self, p):
        """Spectral gradient of a scalar grid field -> ``(3, nx, ny, nz)``."""
        ph = self.fft(p)
        out = np.empty((3,) + self.shape, dtype=float)
        for i in range(3):
            out[i] = self.ifft(1j * self.kgrid[i] * ph)
        return out

    # ------------------------------------------------------- nonlinear term
    def _nonlinear(self, u):
        """``-div(u u)`` projected on the divergence-free subspace, dealiased."""
        if not self.advect:
            return np.zeros_like(u)
        uh = [self.fft(u[i]) for i in range(3)]
        ph = [self.ifft(uh[i]) for i in range(3)]
        out = np.zeros_like(u)
        for i in range(3):
            acc = np.zeros(self.shape, dtype=complex)
            for j in range(3):
                prod = ph[i] * ph[j]
                phj = self.fft(prod) * self._dealias_mask
                acc = acc + 1j * self.kgrid[j] * phj
            out[i] = self.ifft(-acc)
        return self.project(out)

    def diffuse(self, u, dt):
        """Exact viscous integrating factor ``exp(-nu |k|^2 dt)``."""
        if self.nu == 0.0:
            return u
        factor = np.exp(-self.nu * self.k2 * dt)
        out = np.empty_like(u)
        for i in range(3):
            out[i] = self.ifft(self.fft(u[i]) * factor)
        return out

    # ------------------------------------------------- particle-grid transfer
    def interpolate(self, field, positions):
        """Trilinear interpolation of a grid field at particle positions.

        ``field`` has shape ``(n,) + shape`` (any number of leading components).
        Returns an array of shape ``field.shape[:1] + (N,)``.
        """
        idx, w = trilinear_weights(positions, self.shape, self.h)
        ii, jj, kk = idx[..., 0], idx[..., 1], idx[..., 2]
        values = field[..., ii, jj, kk]  # (ncomp, 8, N)
        if values.ndim == 2:  # scalar field
            return np.einsum("kn,kn->n", w, values)
        return np.einsum("kn,ckn->cn", w, values)

    def spread(self, values, positions, mode="adjoint"):
        """Spread per-particle quantities onto the grid.

        ``values`` : shape ``(3, N)`` for a force-per-particle, or ``(N,)`` for a
        scalar (e.g. particle volume).  Returns a grid field with
        ``values.shape[0]`` leading components.

        With ``mode="adjoint"`` the spreading operator is exactly the transpose
        of the trilinear interpolation, so
        ``sum(spread(F)) == sum(F)`` to round-off and
        ``interpolate(spread(F)) == F * sum(w**2)`` (the self-coupling factor is
        therefore known analytically, which the coupling driver uses to remove
        the particle's own induced disturbance).
        """
        values = np.asarray(values, dtype=float)
        scalar = values.ndim == 1
        vals = values[None, :] if scalar else values
        idx, w = trilinear_weights(positions, self.shape, self.h)
        ii, jj, kk = idx[..., 0], idx[..., 1], idx[..., 2]
        if mode == "adjoint":
            weights = w
        elif mode == "nearest":
            # deliberately non-adjoint: nearest-node spreading, used only to
            # demonstrate that the conservation gate actually discriminates.
            weights = np.zeros_like(w)
            nearest = np.argmax(w, axis=0)
            weights[nearest, np.arange(w.shape[1])] = 1.0
        else:
            raise ValueError(f"unknown spread mode {mode!r}")
        out = np.zeros((vals.shape[0],) + self.shape, dtype=float)
        for c in range(vals.shape[0]):
            np.add.at(out[c], (ii, jj, kk), weights * vals[c][None, :])
        return out[0] if scalar else out

    def filter_field(self, field, sigma):
        """Isotropic Gaussian low-pass filter (spectral), ``sigma`` in length units.

        The ``k = 0`` mode is untouched, so the volume-averaged (interstitial)
        velocity of a suspension survives filtering; only structures at the
        particle-support scale are removed.
        """
        if sigma is None or sigma <= 0.0:
            return field
        kernel = np.exp(-0.5 * (sigma ** 2) * self.k2)
        out = np.empty_like(field)
        for i in range(field.shape[0]):
            out[i] = self.ifft(self.fft(field[i]) * kernel)
        return out

    def self_response(self, positions, dt, elapsed=None):
        """Discrete self-response ``R`` (velocity per unit force on the fluid).

        For the spreading/interpolation adjoint pair and the integrator used by
        :meth:`step`, the steady velocity induced at a particle by *its own*
        force is ``u_self = R * F_on_fluid`` with

        ::

            R_i = (2/3) sum_{k != 0} |W_i(k)|^2 / (mu_eff(|k|) V_cell)
            mu_eff(|k|) = rho (exp(nu |k|^2 dt) - 1) / dt
            |W_i(k)|^2 = prod_j [1 - 2 f_j (1-f_j) (1 - cos(k_j h))]

        ``f`` are the particle fractional grid coordinates and the factor
        ``2/3`` is the isotropic average of the Leray projector
        (``trace(I - kk/k^2) = 2``).  The expression is the exact fixed point of
        the discrete scheme for a steady point force, which is what makes the
        implicit self-consistent drag of
        :func:`tensordem.point_cfd_dem.particles.evaluate_forces` cancel the
        self-induced velocity to the accuracy of the projection trace.
        """
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        n = pos.shape[0]
        if self.mu == 0.0 or self.nu == 0.0:
            return np.zeros(n)
        frac = pos / self.h - np.floor(pos / self.h)
        # per-particle |W(k)|^2 is separable in the three directions
        one_minus = 2.0 * frac * (1.0 - frac)          # (n, 3)
        cosfac = 1.0 - one_minus
        sinfac = one_minus
        if elapsed is None:
            development = np.ones(self.shape)
        else:
            elapsed = max(float(elapsed), dt)
            development = 1.0 - np.exp(-self.nu * self.k2 * elapsed)
            development[self._k2_zero] = 1.0
        mu_eff = self.rho * (np.exp(self.nu * self.k2 * dt) - 1.0) / dt
        inv = np.zeros(self.shape, dtype=float)
        mask = ~self._k2_zero
        inv[mask] = 1.0 / mu_eff[mask]
        # per-direction |W_j|^2 = cosfac_j + sinfac_j * cos(k_j h), separable
        kernel = np.ones((n,) + self.shape, dtype=float)
        for j in range(3):
            shape = [1, 1, 1]
            shape[j] = self.shape[j]
            c = np.cos(self.k[j] * self.h).reshape(shape)
            kernel = kernel * (cosfac[:, j].reshape((n,) + (1, 1, 1))
                               + sinfac[:, j].reshape((n,) + (1, 1, 1)) * c)
        weight = kernel * inv[None, :, :, :] * development[None, :, :, :]
        domain_volume = self.cell_volume * int(np.prod(self.shape))
        return (2.0 / 3.0) * np.sum(weight.reshape(n, -1), axis=1) / domain_volume

    def self_weight_sum(self, positions):
        """``sum_k w_k^2`` of the trilinear stencil for each particle."""
        _, w = trilinear_weights(positions, self.shape, self.h)
        return np.sum(w ** 2, axis=0)

    # ------------------------------------------------------------- time step
    def step(self, u, dt, force_density=None, project=True):
        """Advance the velocity field by ``dt``.

        Parameters
        ----------
        u : (3, nx, ny, nz) velocity
        dt : time step
        force_density : optional (3, nx, ny, nz) body force **per unit volume**
            (already summed over particles).  It enters as a momentum impulse
            ``dt * f / rho`` so that the momentum actually handed to the fluid
            equals ``dt * sum(f) * dV`` exactly.
        project : enforce the divergence-free constraint at the end (default)

        Scheme: IFRK2 (midpoint) for advection + source, exact viscous
        integrating factor, final Leray projection.
        """
        src = None
        if force_density is not None:
            src = np.asarray(force_density, dtype=float) / self.rho
        else:
            src = np.zeros_like(u)

        n1 = self._nonlinear(u) + src
        u_half = self.diffuse(u + 0.5 * dt * n1, 0.5 * dt)
        n2 = self._nonlinear(u_half) + src
        u_new = self.diffuse(u + dt * n2, dt)
        if project:
            u_new = self.project(u_new)
        return u_new

    # -------------------------------------------------------------- measures
    def mean_velocity(self, u):
        """Volume-averaged (``k = 0``) velocity, shape ``(3,)``."""
        return np.asarray([u[i].mean() for i in range(3)])

    def momentum(self, u):
        """Total linear momentum ``rho * sum(u) * dV``, shape ``(3,)``."""
        return self.rho * self.mean_velocity(u) * (
            self.cell_volume * np.prod(self.shape)
        )

    def kinetic_energy(self, u):
        return 0.5 * self.rho * self.cell_volume * float(np.sum(u * u))

    def max_divergence(self, u):
        return float(np.max(np.abs(self.divergence(u))))
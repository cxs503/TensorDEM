"""Bonded beam lattice for an ice-sheet slice (2D corotational frame elements).

The sheet is discretised as ``nx x nz`` particles (rigid discs of out-of-plane
width ``t``) connected by **frame bonds** carrying an axial force ``N``, end
moments ``M`` and transverse shear.  The element is a standard 2-node
Euler-Bernoulli frame element written in *corotational* form, so rigid-body
rotation of a fragment produces no spurious force while small elastic strains
are represented exactly:

* axial elongation  ``u = L - L0``            -> ``N  = EA u / L0``
* nodal rotations   ``th_a = (th_a - beta) - (th_a0 - beta0)`` -> ``M_a, M_b``

with ``beta`` the current chord angle and ``beta0`` the reference chord angle.
``v = 0`` in the corotational local frame by construction (the chord absorbs
transverse motion), so the local nodal force vector is ``(N, F_va, M_a)`` with

    ``F_va = (6EI/L0^2)(th_a + th_b)``,
    ``M_a  = (4EI/L0) th_a + (2EI/L0) th_b``,  ``M_b = (2EI/L0) th_a + (4EI/L0) th_b``.

Bond properties are the physical beam properties of a strip of width ``dz``
(horizontal bonds) or ``dx`` (vertical bonds) and out-of-plane thickness ``t``:

    ``EA = E t dz``,  ``EI = E t dz^3 / 12``,  ``L0 = dx``   (horizontal)

so the sheet's macroscopic bending stiffness ``EI_sheet`` emerges from fibre
stretching, as in the continuum.

Fracture
--------
Fiber stresses at a bond are ``sigma = N/A +- |M|/Z`` with ``Z = t dz^2/6``.
Two failure modes are tracked separately:

``bending``   tensile/flexural fiber stress exceeds ``sigma_t`` (flexural strength)
``crushing``  compressive fiber stress exceeds ``sigma_c``

A bond breaks when either criterion is met (the larger utilisation decides the
mode).  The elastic energy stored in the bond at the instant of failure is
returned as ``crack_energy`` and booked in the energy ledger; released energy is
consumed (not re-injected), which is the conservative DEM convention — it is
documented in the verification report as a modelling choice.

All quantities SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["IceMaterial", "IceSheetLattice"]


@dataclass(frozen=True)
class IceMaterial:
    """Isotropic ice properties (SI)."""

    density_kg_m3: float = 917.0
    young_modulus_pa: float = 9.0e9
    poisson_ratio: float = 0.33
    compressive_strength_pa: float = 1.5e6
    flexural_strength_pa: float = 0.5e6


class IceSheetLattice:
    """Bonded frame-bonded lattice representing a longitudinal ice-sheet slice."""

    def __init__(
        self,
        *,
        length_m: float,
        thickness_m: float,
        out_of_plane_width_m: float,
        material: IceMaterial,
        nx: int = 90,
        nz: int = 5,
        water_density_kg_m3: float = 1025.0,
        gravity_m_s2: float = 9.80665,
        clamp_first_column: bool = False,
    ) -> None:
        self.material = material
        self.nx = int(nx)
        self.nz = int(nz)
        self.length_m = float(length_m)
        self.thickness_m = float(thickness_m)
        self.width_t = float(out_of_plane_width_m)  # out-of-plane slice thickness
        self.dx = self.length_m / self.nx
        self.dz = self.thickness_m / self.nz
        self.water_density = float(water_density_kg_m3)
        self.gravity = float(gravity_m_s2)

        n = self.nx * self.nz
        self.n_particles = n
        submergence = min(1.0, material.density_kg_m3 / self.water_density)
        freeboard = self.thickness_m * (1.0 - submergence)
        # particle centres: x in [0, L], z in [-h + dz/2, 0 + ... ] with the
        # sheet floating at its hydrostatic equilibrium (freeboard above z=0).
        ix, iz = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing="ij")
        ix = ix.ravel().astype(float)
        iz = iz.ravel().astype(float)
        self.ix = ix.astype(int)
        self.iz = iz.astype(int)
        self.pos = np.stack(
            [
                (ix + 0.5) * self.dx,
                -self.thickness_m + freeboard + (iz + 0.5) * self.dz,
            ],
            axis=1,
        )
        self.pos0 = self.pos.copy()
        self.vel = np.zeros((n, 2))
        self.ang = np.zeros(n)
        self.omega = np.zeros(n)
        self.radius = 0.5 * self.dz  # in-plane particle radius (circle diameter = dz)

        volume = self.dx * self.dz * self.width_t
        self.mass = np.full(n, material.density_kg_m3 * volume)
        self.inertia = self.mass * (self.dx**2 + self.dz**2) / 12.0

        # reference state for the axial force: the lattice is built stress-free
        self.bonds = self._build_bonds()

        self.fixed = np.zeros(n, dtype=bool)
        if clamp_first_column:
            self.fixed[self.ix == 0] = bool(clamp_first_column)
        # far column clamped: continuity with the unbroken ice field (modelled
        # by fixing the column at x = L, see the verification report)
        self.fixed[self.ix == self.nx - 1] = True

        # bond force cache (for ledger / diagnostics)
        self.bond_axial = np.zeros(self.bonds["n"])
        self.bond_moment = np.zeros(self.bonds["n"])

    # ------------------------------------------------------------------ build
    def _build_bonds(self) -> dict[str, np.ndarray]:
        nx, nz = self.nx, self.nz
        idx = lambda i, j: i * nz + j  # noqa: E731
        ia, ib, kind = [], [], []
        for i in range(nx):
            for j in range(nz):
                if i + 1 < nx:
                    ia.append(idx(i, j))
                    ib.append(idx(i + 1, j))
                    kind.append(1)  # horizontal
                if j + 1 < nz:
                    ia.append(idx(i, j))
                    ib.append(idx(i, j + 1))
                    kind.append(0)  # vertical
        ia = np.array(ia)
        ib = np.array(ib)
        kind = np.array(kind)
        d = self.pos0[ib] - self.pos0[ia]
        L0 = np.linalg.norm(d, axis=1)
        beta0 = np.arctan2(d[:, 1], d[:, 0])
        mat = self.material
        e = mat.young_modulus_pa
        t = self.width_t
        depth = np.where(kind == 1, self.dz, self.dx)  # in-plane beam depth
        area = t * depth
        inertia = t * depth**3 / 12.0
        section_z = t * depth**2 / 6.0
        return {
            "n": len(ia),
            "ia": ia,
            "ib": ib,
            "kind": kind,
            "L0": L0,
            "beta0": beta0,
            "EA": e * area,
            "EI": e * inertia,
            "area": area,
            "Z": section_z,
            "active": np.ones(len(ia), dtype=bool),
            "mode": np.zeros(len(ia), dtype=int),  # 0 intact, 1 bending, 2 crushing
        }

    # --------------------------------------------------------------- geometry
    @property
    def n_bonds(self) -> int:
        return int(self.bonds["n"])

    def bond_energy(self) -> float:
        """Total elastic energy stored in the active bonds [J]."""
        b = self.bonds
        act = b["active"]
        L = np.linalg.norm(self.pos[b["ib"]] - self.pos[b["ia"]], axis=1)
        u = L - b["L0"]
        N = b["EA"] * u / b["L0"]
        ta, tb = self._relative_rotations()
        Ma = 4.0 * b["EI"] / b["L0"] * ta + 2.0 * b["EI"] / b["L0"] * tb
        Mb = 2.0 * b["EI"] / b["L0"] * ta + 4.0 * b["EI"] / b["L0"] * tb
        energy = 0.5 * (u * N + ta * Ma + tb * Mb)
        return float(np.sum(np.where(act, energy, 0.0)))

    def _relative_rotations(self) -> tuple[np.ndarray, np.ndarray]:
        b = self.bonds
        d = self.pos[b["ib"]] - self.pos[b["ia"]]
        beta = np.arctan2(d[:, 1], d[:, 0])
        ta = self.ang[b["ia"]] - beta - (0.0 - b["beta0"])
        tb = self.ang[b["ib"]] - beta - (0.0 - b["beta0"])
        return ta, tb

    # ----------------------------------------------------------------- forces
    def bond_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(force, torque)`` arrays [N], [N m] from all active bonds."""
        b = self.bonds
        ia, ib = b["ia"], b["ib"]
        act = b["active"]
        d = self.pos[ib] - self.pos[ia]
        L = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        inv = 1.0 / np.maximum(L, 1e-12)
        ex = d[:, 0] * inv
        ez = d[:, 1] * inv
        N = b["EA"] * (L - b["L0"]) / b["L0"]
        beta = np.arctan2(d[:, 1], d[:, 0])
        b0 = b["beta0"]
        ta = self.ang[ia] - beta + b0
        tb = self.ang[ib] - beta + b0
        EI = b["EI"]
        L0 = b["L0"]
        k1 = 6.0 * EI / (L0 * L0)
        k4 = 4.0 * EI / L0
        k2 = 2.0 * EI / L0
        s_ab = ta + tb
        # internal (restoring) forces on the nodes are the NEGATIVE of the
        # external nodal forces K q of the standard beam element:
        Fva = -k1 * s_ab
        Fvb = -Fva
        Ma = -(k4 * ta + k2 * tb)
        Mb = -(k2 * ta + k4 * tb)
        actf = act.view(np.int8)
        N = N * actf
        Fva = Fva * actf
        Ma = Ma * actf
        Mb = Mb * actf
        fx = N * ex - Fva * ez
        fz = N * ez + Fva * ex
        n = self.n_particles
        force = np.empty((n, 2))
        force[:, 0] = np.bincount(ia, weights=fx, minlength=n) - np.bincount(ib, weights=fx, minlength=n)
        force[:, 1] = np.bincount(ia, weights=fz, minlength=n) - np.bincount(ib, weights=fz, minlength=n)
        torque = np.bincount(ia, weights=Ma, minlength=n) + np.bincount(ib, weights=Mb, minlength=n)
        self.bond_axial = N
        self.bond_moment = np.maximum(np.abs(Ma), np.abs(Mb))
        return force, torque

    # ---------------------------------------------------------------- failure
    def check_failure(
        self, sigma_t: float, sigma_c: float
    ) -> dict[str, object]:
        """Break bonds whose fiber stress exceeds the ice strengths.

        Returns a dict with the number of new breaks, the per-mode counts and
        the elastic energy released (booked as crack energy in the ledger).
        """
        b = self.bonds
        act = b["active"]
        if not np.any(act):
            return {"broken": 0, "bending": 0, "crushing": 0, "crack_energy": 0.0}
        d = self.pos[b["ib"]] - self.pos[b["ia"]]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        N = b["EA"] * (L - b["L0"]) / b["L0"]
        M = self.bond_moment
        sigma_axial = N / b["area"]
        sigma_bend = M / b["Z"]
        tension = sigma_axial + sigma_bend          # max tensile fiber
        compression = -sigma_axial + sigma_bend     # max compressive fiber
        util_bending = tension / max(sigma_t, 1e-30)
        util_crushing = compression / max(sigma_c, 1e-30)
        util = np.maximum(util_bending, util_crushing)
        fail = act & (util > 1.0)
        n_fail = int(np.count_nonzero(fail))
        if n_fail == 0:
            return {"broken": 0, "bending": 0, "crushing": 0, "crack_energy": 0.0}
        mode = np.where(util_bending >= util_crushing, 1, 2)
        # elastic energy stored in the failing bonds (computed before removal)
        ta, tb = self._relative_rotations()
        Ma = 4.0 * b["EI"] / b["L0"] * ta + 2.0 * b["EI"] / b["L0"] * tb
        Mb = 2.0 * b["EI"] / b["L0"] * ta + 4.0 * b["EI"] / b["L0"] * tb
        u = L - b["L0"]
        energy = 0.5 * (u * N + ta * Ma + tb * Mb)
        crack_energy = float(np.sum(np.where(fail, np.maximum(energy, 0.0), 0.0)))
        bending = int(np.count_nonzero(fail & (mode == 1)))
        crushing = int(np.count_nonzero(fail & (mode == 2)))
        b["active"][fail] = False
        b["mode"][fail] = mode[fail]
        return {
            "broken": n_fail,
            "bending": bending,
            "crushing": crushing,
            "crack_energy": crack_energy,
        }

    # --------------------------------------------------------------- buoyancy
    def body_forces(
        self,
        water_density: float | None = None,
        gravity: float | None = None,
    ) -> np.ndarray:
        """Gravity + buoyancy per particle [N] (waterline at ``z = 0``)."""
        rho_w = self.water_density if water_density is None else water_density
        g = self.gravity if gravity is None else gravity
        volume = self.dx * self.dz * self.width_t
        bottom = self.pos[:, 1] - 0.5 * self.dz
        top = self.pos[:, 1] + 0.5 * self.dz
        submerged = np.clip((0.0 - bottom) / self.dz, 0.0, 1.0)
        submerged = np.where(top <= 0.0, 1.0, submerged)
        fz = self.mass * (-g) + rho_w * g * volume * submerged
        force = np.zeros((self.n_particles, 2))
        force[:, 1] = fz
        return force

    def submerged_mask(self) -> np.ndarray:
        return (self.pos[:, 1] - 0.5 * self.dz) < 0.0
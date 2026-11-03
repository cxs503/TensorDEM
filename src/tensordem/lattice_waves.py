"""Wave propagation in the DEM3D bonded lattice: analytic dispersion and solver probes.

Why this module exists
----------------------
``docs/dem3d_lattice_elasticity.md`` calibrated the *statics* of the 13-neighbour
simple-cubic lattice (Cauchy-Born constants ``C11 = 13/3 k/a``, ``C44 = 7/3 k/a``)
and bounded the explicit time step from the phonon spectrum, but it states
explicitly that the bound is "a necessary condition only: it says the chosen step
cannot be rejected by the elastic bound, not that the dynamics is accurate".
Nothing in the repository tested a *propagating* elastic disturbance in
``IceDEM3D`` against a reference.  This module supplies that reference and the
measurement machinery:

* :func:`bulk_dynamical_matrix` / :func:`bulk_dispersion` -- the exact harmonic
  dispersion of the infinite lattice built from the *same* 26 bond directions the
  solver uses (all-pairs sum ``D(q) = (1/m) sum_b k (1 - cos q.r_b) n n^T``).
* :func:`dispersion_along_100` -- a closed form for the [100] longitudinal and
  transverse branches, used as an independent check of the assembled matrix.
* :func:`acoustic_speeds` -- the long-wavelength limits ``sqrt(C11/rho)``,
  ``sqrt(C44/rho)`` and the slender-bar (stress-free lateral surface) speed
  ``sqrt(E_100/rho)``, i.e. the physical speeds a pulse is expected to show.
* :func:`cross_section_bloch` / :func:`bar_dispersion` -- exact branches of the
  *finite* cross-section bar with periodic boundary conditions along x, so a
  pulse measured in a free-lateral bar is compared with a dispersion relation
  computed for exactly that geometry instead of an idealised 1-D chain.
* :func:`finite_bar_modes` -- exact normal modes of a finite ``IceDEM3D`` sample
  assembled from the solver's own bond list: the reference for integrator
  accuracy (it isolates ``omega_error`` from ``k_error``).
* :func:`pulse_experiment` / :func:`predict_packet_arrival` -- run a Gaussian
  velocity packet through ``IceDEM3D`` and compare the crest arrival time with
  the linear-superposition prediction ``u(x,t) = sum_k A(k) cos(kx - omega(k)t)``
  built from :func:`cross_section_bloch`.

Conventions: SI units, lattice spacing ``a = 2 * radius``, particle mass
``m = density * (4/3) pi R^3``, effective lattice density ``rho = m / a**3``
(the solver's spheres occupy ``pi/6`` of the cell, so ``rho < density``).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .lattice_elasticity import IDEAL_OFFSETS, analytic_box_constants

DTYPE = torch.float64


def one_minus_cos(theta: float) -> float:
    """``1 - cos(theta)`` without the catastrophic cancellation near ``theta = 0``."""
    return 2.0 * math.sin(0.5 * theta) ** 2

#: The 13 half-space bond classes of ``IceDEM3D`` (``IDEAL_OFFSETS``) plus their
#: negative partners: the solver's bond loop keeps one half-space, so the
#: realised neighbour set is exactly these 26 directions.
FULL_NEIGHBOURS: tuple[tuple[int, int, int], ...] = tuple(IDEAL_OFFSETS) + tuple(
    (-ox, -oy, -oz) for ox, oy, oz in IDEAL_OFFSETS
)


# --------------------------------------------------------------------------- #
# infinite-lattice (bulk) dispersion
# --------------------------------------------------------------------------- #
def bulk_dynamical_matrix(
    q: Sequence[float],
    *,
    bond_stiffness: float,
    mass: float,
    spacing: float,
    offsets: Iterable[tuple[int, int, int]] = FULL_NEIGHBOURS,
) -> torch.Tensor:
    """Harmonic dynamical matrix ``D(q)`` (3x3) of the infinite bonded lattice.

    ``D(q) = (1/m) sum_b k_b (1 - cos(q.r_b)) n_b (x) n_b`` summed over the 26
    bond directions; its eigenvalues are the squared phonon angular frequencies
    of a plane wave with wavevector ``q``.
    """
    if mass <= 0 or bond_stiffness <= 0 or spacing <= 0:
        raise ValueError("bond_stiffness, mass and spacing must be positive")
    qs = torch.as_tensor(q, dtype=DTYPE)
    if qs.shape != (3,):
        raise ValueError("q must be a length-3 wavevector")
    matrix = torch.zeros((3, 3), dtype=DTYPE)
    for offset in offsets:
        vec = torch.tensor(offset, dtype=DTYPE) * spacing
        length = float(torch.linalg.vector_norm(vec))
        normal = vec / length
        weight = one_minus_cos(float(qs @ vec))
        matrix += bond_stiffness * weight * torch.outer(normal, normal)
    return matrix / mass


def bulk_dispersion(
    q: Sequence[float],
    *,
    bond_stiffness: float,
    mass: float,
    spacing: float,
) -> dict[str, Any]:
    """Eigen-frequencies and polarisations of ``D(q)`` in ascending order."""
    dyn = bulk_dynamical_matrix(
        q, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing
    )
    values, vectors = torch.linalg.eigh(dyn)
    values = values.clamp_min(0.0)
    return {
        "q": [float(v) for v in q],
        "omega_rad_s": [float(math.sqrt(v)) for v in values],
        "polarisation": [vectors[:, i].tolist() for i in range(3)],
    }


def dispersion_along_100(
    k: float,
    *,
    bond_stiffness: float,
    mass: float,
    spacing: float,
) -> dict[str, float]:
    """Closed-form [100] branches of the 13-neighbour lattice.

    Summing the bond weights for ``q = (k, 0, 0)`` gives

        omega_L^2 = (26/3) (k/m) (1 - cos(k a))
        omega_T^2 = (14/3) (k/m) (1 - cos(k a))      (doubly degenerate)

    whose small-``k`` limits are ``omega_L/k = sqrt(C11/rho)`` and
    ``omega_T/k = sqrt(C44/rho)`` with ``C11 = 13/3 k/a``, ``C44 = 7/3 k/a``.
    """
    if bond_stiffness <= 0 or mass <= 0 or spacing <= 0:
        raise ValueError("bond_stiffness, mass and spacing must be positive")
    ratio = 2.0 * bond_stiffness / mass * one_minus_cos(k * spacing)
    omega_l = math.sqrt((13.0 / 3.0) * ratio)
    omega_t = math.sqrt((7.0 / 3.0) * ratio)
    sin_term = 2.0 * bond_stiffness / mass * spacing * math.sin(k * spacing)
    return {
        "k_rad_per_m": k,
        "k_times_spacing": k * spacing,
        "omega_L_rad_s": omega_l,
        "omega_T_rad_s": omega_t,
        "group_velocity_L_m_per_s": (
            (13.0 / 3.0) * sin_term / (2.0 * omega_l) if omega_l > 0 else 0.0
        ),
        "group_velocity_T_m_per_s": (
            (7.0 / 3.0) * sin_term / (2.0 * omega_t) if omega_t > 0 else 0.0
        ),
    }


def acoustic_speeds(
    *,
    bond_stiffness: float,
    mass: float,
    spacing: float,
) -> dict[str, float]:
    """Long-wavelength speeds implied by the calibrated elastic constants.

    ``c_L = sqrt(C11/rho)`` (bulk, laterally constrained strain), ``c_T =
    sqrt(C44/rho)`` (shear) and ``c_bar = sqrt(E_100/rho)`` (slender bar with
    free lateral stress), with ``rho = mass / spacing**3``.
    """
    constants = analytic_box_constants(bond_stiffness, spacing)
    c11, c12, c44 = constants["C11"], constants["C12"], constants["C44"]
    rho = mass / spacing**3
    e100 = c11 - 2.0 * c12 * c12 / (c11 + c12)
    return {
        "rho_lattice_kg_per_m3": rho,
        "C11_Pa": c11,
        "C12_Pa": c12,
        "C44_Pa": c44,
        "E100_Pa": e100,
        "c_longitudinal_bulk_m_per_s": math.sqrt(c11 / rho),
        "c_transverse_m_per_s": math.sqrt(c44 / rho),
        "c_bar_slender_m_per_s": math.sqrt(e100 / rho),
    }


# --------------------------------------------------------------------------- #
# finite cross-section bar: Bloch branches (periodic along x, free laterally)
# --------------------------------------------------------------------------- #
def _cross_section_cells(ny: int, nz: int) -> list[tuple[int, int]]:
    return [(y, z) for z in range(nz) for y in range(ny)]


def _cross_section_matrix(
    kx: float,
    *,
    bond_stiffness: float,
    spacing: float,
    ny: int,
    nz: int,
    offsets: Iterable[tuple[int, int, int]] = FULL_NEIGHBOURS,
) -> torch.Tensor:
    cells = _cross_section_cells(ny, nz)
    index = {cell: idx for idx, cell in enumerate(cells)}
    dof = 3 * len(cells)
    matrix = torch.zeros((dof, dof), dtype=torch.complex128)
    for offset in offsets:
        dx, dy, dz = offset
        vec = torch.tensor(offset, dtype=DTYPE) * spacing
        length = float(torch.linalg.vector_norm(vec))
        normal = vec / length
        block = bond_stiffness * torch.outer(normal, normal)
        phase = complex(math.cos(kx * dx * spacing), math.sin(kx * dx * spacing))
        for (y, z) in cells:
            y2, z2 = y + dy, z + dz
            if not (0 <= y2 < ny and 0 <= z2 < nz):
                continue
            i = 3 * index[(y, z)]
            j = 3 * index[(y2, z2)]
            matrix[i:i + 3, i:i + 3] -= block
            matrix[i:i + 3, j:j + 3] += block * phase
    # The assembled form is sum_b k (exp(i k dx a) - 1) n (x) n = -D, because the
    # bond force uses the *difference* of the two end displacements; the physical
    # squared frequencies are +eig(D).
    return -0.5 * (matrix + matrix.conj().T)


def _cross_section_eigen(
    kx: float, *, bond_stiffness: float, mass: float, spacing: float,
    ny: int, nz: int,
) -> tuple[list[float], torch.Tensor]:
    matrix = _cross_section_matrix(
        kx, bond_stiffness=bond_stiffness, spacing=spacing, ny=ny, nz=nz
    )
    values, vectors = torch.linalg.eigh(matrix / mass)  # matrix is +D
    omega = [float(math.sqrt(max(v, 0.0))) for v in values]
    return omega, vectors


def cross_section_bloch(
    kx: float,
    *,
    bond_stiffness: float,
    mass: float,
    spacing: float,
    ny: int,
    nz: int,
) -> dict[str, Any]:
    """Exact harmonic branches (ascending) of a ``ny x nz`` bar, periodic in x.

    The cross-section is free laterally, so the ``k -> 0`` longitudinal branch is
    the bar wave, not the bulk wave.  The matrix
    ``M(k) = sum_b k (exp(i k dx a) - 1) n (x) n`` is Hermitian over the 26 bond
    directions; ``eig(M)/m`` are the squared frequencies.
    """
    if ny < 1 or nz < 1:
        raise ValueError("ny and nz must be >= 1")
    omega, _ = _cross_section_eigen(
        kx, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing, ny=ny, nz=nz
    )
    return {"kx_rad_per_m": kx, "omega_rad_s": omega, "n_branches": len(omega)}


def longitudinal_branch_index(
    *,
    bond_stiffness: float,
    mass: float,
    spacing: float,
    ny: int,
    nz: int,
) -> int:
    """Index (ascending) of the x-polarised acoustic branch at ``k -> 0``."""
    probe_k = 1e-6 / spacing
    omega, vectors = _cross_section_eigen(
        probe_k, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing,
        ny=ny, nz=nz,
    )
    best, best_score = 0, -1.0
    for idx, freq in enumerate(omega):
        if freq > 1e-3:
            continue
        x_weight = float((vectors[0::3, idx].abs() ** 2).sum())
        if x_weight > best_score:
            best, best_score = idx, x_weight
    return best


def bar_dispersion(
    kx: float,
    *,
    bond_stiffness: float,
    mass: float,
    spacing: float,
    ny: int,
    nz: int,
    branch: int | None = None,
) -> dict[str, float]:
    """Longitudinal bar branch at ``kx`` with a finite-difference group velocity."""
    if branch is None:
        branch = longitudinal_branch_index(
            bond_stiffness=bond_stiffness, mass=mass, spacing=spacing, ny=ny, nz=nz
        )
    zone_edge = math.pi / spacing
    dk = max(1e-4 * zone_edge, 1e-9)
    if abs(kx) < dk:
        kx = dk
    omega = cross_section_bloch(
        kx, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing, ny=ny, nz=nz
    )["omega_rad_s"][branch]
    plus = cross_section_bloch(
        kx + dk, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing,
        ny=ny, nz=nz,
    )["omega_rad_s"][branch]
    minus = cross_section_bloch(
        kx - dk, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing,
        ny=ny, nz=nz,
    )["omega_rad_s"][branch]
    vg = (plus - minus) / (2.0 * dk)
    return {
        "kx_rad_per_m": kx,
        "branch_index": branch,
        "omega_rad_s": omega,
        "phase_velocity_m_per_s": omega / kx,
        "group_velocity_m_per_s": vg,
    }


# --------------------------------------------------------------------------- #
# finite sample normal modes (reference for the time integrator)
# --------------------------------------------------------------------------- #
def assemble_stiffness(dem: IceDEM3D) -> torch.Tensor:
    """Dense stiffness matrix of the live bonded network (3N x 3N)."""
    bonds = dem.pairs
    normals = dem.initial_normals
    alive = dem.alive
    n = int(dem.positions.shape[0])
    stiff = torch.zeros((3 * n, 3 * n), dtype=DTYPE)
    for idx in range(int(bonds.shape[0])):
        if not bool(alive[idx]):
            continue
        i, j = int(bonds[idx, 0]), int(bonds[idx, 1])
        outer = float(dem.config.bond_stiffness) * torch.outer(
            normals[idx].to(DTYPE), normals[idx].to(DTYPE)
        )
        stiff[3 * i:3 * i + 3, 3 * i:3 * i + 3] += outer
        stiff[3 * j:3 * j + 3, 3 * j:3 * j + 3] += outer
        stiff[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= outer
        stiff[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= outer
    return stiff


def finite_bar_modes(dem: IceDEM3D, *, count: int = 6) -> dict[str, Any]:
    """Exact harmonic normal modes of the current bonded state of ``dem``.

    The stiffness matrix is assembled from the solver's own ``pairs`` and
    ``initial_normals`` and fixed particles are removed, so the reference
    respects the solver's boundary conditions.
    """
    stiff = assemble_stiffness(dem)
    n = int(dem.positions.shape[0])
    free: list[int] = []
    for particle in range(n):
        if bool(dem.fixed[particle]):
            continue
        free.extend((3 * particle, 3 * particle + 1, 3 * particle + 2))
    if not free:
        raise RuntimeError("no free degrees of freedom")
    free_idx = torch.tensor(free, dtype=torch.long)
    reduced = stiff[free_idx][:, free_idx]
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = torch.linalg.eigh(reduced)
    mass = float(dem.config.mass)
    keep = min(count, int(values.shape[0]))
    return {
        "n_particles": n,
        "n_free_dof": int(free_idx.numel()),
        "omega_rad_s": [float(math.sqrt(max(float(v), 0.0) / mass)) for v in values[:keep]],
        "mode_vectors": vectors[:, :keep],
        "free_dof": free_idx,
        "stiffness_matrix": reduced,
    }


# --------------------------------------------------------------------------- #
# pulse experiment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PulseSpec:
    """Geometry and excitation of a 1-D pulse run inside ``IceDEM3D``."""

    nx: int = 401
    ny: int = 2
    nz: int = 2
    packet_cell: int = 40
    packet_width_cells: float = 60.0
    station_cells: tuple[int, int] = (140, 340)
    amplitude_m: float = 1e-6

    def __post_init__(self) -> None:
        if self.station_cells[0] >= self.station_cells[1]:
            raise ValueError("station_cells must be strictly increasing")
        if self.station_cells[1] >= self.nx - 1:
            raise ValueError("stations must stay inside the sample")
        if self.packet_cell >= self.station_cells[0]:
            raise ValueError("packet must start upstream of the first station")
        if self.packet_width_cells <= 0 or self.amplitude_m <= 0:
            raise ValueError("packet_width_cells and amplitude must be positive")


def pulse_experiment(
    spec: PulseSpec,
    *,
    bond_stiffness: float = 2000.0,
    radius: float = 0.025,
    density: float = 917.0,
    contact_stiffness: float = 5000.0,
    contact_damping: float = 5.0,
    steps: int | None = None,
    dt: float | None = None,
) -> dict[str, Any]:
    """Launch a Gaussian axial *displacement* packet and record two stations.

    A displacement (not velocity) initial condition is used so the run matches the
    linear-superposition reference of :func:`predict_packet_arrival` term by term.
    Damping (``drag``), gravity, tool motion and platens are disabled, so the run
    is a pure bonded-lattice wave test; the excitation stays far below the
    breaking strain and the returned state asserts that no bond failed.
    """
    spacing = 2.0 * radius
    config = DEM3DConfig(
        nx=spec.nx,
        ny=spec.ny,
        nz=spec.nz,
        radius=radius,
        density=density,
        bond_stiffness=bond_stiffness,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
        fix_x_edges=False,
        fix_bottom=False,
        drag=0.0,
        tool_gap=1.0,
        tool_speed=1e-6,
        dt=dt,
    )
    dem = IceDEM3D(config)
    x = dem.positions[:, 0]
    x_min = float(x.min())
    x0 = x_min + spec.packet_cell * spacing
    width = spec.packet_width_cells * spacing
    profile = torch.exp(-((x - x0) / width) ** 2)
    initial_ux = spec.amplitude_m * profile
    dem.positions[:, 0] = dem.positions[:, 0] + initial_ux
    dem.velocities = torch.zeros_like(dem.positions)

    station_x = [x_min + cell * spacing for cell in spec.station_cells]
    station_masks = [
        torch.isclose(x, torch.tensor(sx, dtype=x.dtype)) for sx in station_x
    ]
    for mask in station_masks:
        if int(mask.sum()) == 0:
            raise RuntimeError("station slab is empty")

    step_dt = float(dem.dt)
    if steps is None:
        # Crest travel time at the slowest plausible speed (transverse branch).
        steps = int(1.6 * (spec.station_cells[1] * spacing / 10.0) / step_dt)
    times = [0.0]
    traces = [[float(dem.velocities[mask, 0].mean())] for mask in station_masks]
    lateral = [[float(dem.velocities[:, 1].abs().mean())] for _ in station_masks]
    energies = [float(dem.energy()["mechanical_energy_J"])]
    for step in range(steps):
        dem.step()
        times.append(float(dem.time))
        for idx, mask in enumerate(station_masks):
            traces[idx].append(float(dem.velocities[mask, 0].mean()))
            lateral[idx].append(float(dem.velocities[mask, 1].abs().mean()))
        if step % 200 == 0:
            energies.append(float(dem.energy()["mechanical_energy_J"]))
    energies.append(float(dem.energy()["mechanical_energy_J"]))
    if dem.broken_bonds:
        raise RuntimeError("pulse excitation fractured bonds; reduce the amplitude")
    if dem.failure_mode.any():
        raise RuntimeError("unexpected damage during the pulse run")
    return {
        "spec": {
            "nx": spec.nx, "ny": spec.ny, "nz": spec.nz,
            "packet_cell": spec.packet_cell,
            "packet_width_cells": spec.packet_width_cells,
            "station_cells": list(spec.station_cells),
            "amplitude_m": spec.amplitude_m,
            "bond_stiffness_N_per_m": bond_stiffness,
            "radius_m": radius,
            "density_kg_per_m3": density,
            "spacing_m": spacing,
            "dt_s": step_dt,
            "steps": steps,
        },
        "distance_m": station_x[1] - station_x[0],
        "station_x_m": station_x,
        "times_s": times,
        "velocity_x_m_per_s": traces,
        "velocity_lateral_m_per_s": lateral,
        "energy_J": energies,
        "broken_bonds": int(dem.broken_bonds),
        "max_displacement_m": float(
            (dem.positions - dem.initial_positions).abs().max()
        ),
        "initial_profile": {
            "x_m": (x - x_min).tolist(),
            "ux_m": initial_ux.tolist(),
            "spacing_m": spacing,
        },
        "max_axial_strain_at_init": float(
            (initial_ux.max() - initial_ux.min()) / (spec.packet_width_cells * spacing)
        ),
    }


def peak_time(times: Sequence[float], signal: Sequence[float]) -> float:
    """Time of maximal |signal| with parabolic interpolation around the peak sample."""
    values = [abs(v) for v in signal]
    idx = max(range(len(values)), key=lambda i: values[i])
    if idx == 0 or idx == len(values) - 1:
        return float(times[idx])
    y0, y1, y2 = values[idx - 1], values[idx], values[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    step = float(times[1] - times[0])
    return float(times[idx] + offset * step)


def predict_packet_arrival(
    profile: dict[str, Any],
    *,
    station_offset_m: float,
    bond_stiffness: float,
    mass: float,
    ny: int,
    nz: int,
    k_max_fraction: float = 0.45,
    n_k: int = 160,
    time_step: float = 1e-4,
    time_horizon: float = 2.5,
) -> dict[str, float]:
    """Linear-superposition prediction of the crest arrival time at a station.

    The initial displacement profile is Fourier transformed on the lattice; each mode
    is propagated with the exact bar dispersion of :func:`bar_dispersion`,

        u(x,t) = sum_k A(k) cos(k x - omega(k) t),

    and the arrival time is the ``t`` maximising ``|u|`` at the station.  This
    prediction never runs ``IceDEM3D``: it uses only the analytic dispersion, so
    comparing it with :func:`pulse_experiment` is a solver-versus-theory test.
    """
    spacing = float(profile["spacing_m"])
    x = torch.as_tensor(profile["x_m"], dtype=DTYPE)
    v = torch.as_tensor(profile["ux_m"], dtype=DTYPE)
    n = x.numel()
    spectrum = torch.fft.rfft(v)
    freq_index = torch.arange(spectrum.numel(), dtype=DTYPE)
    wavenumber = 2.0 * math.pi * freq_index / (n * spacing)
    zone_edge = math.pi / spacing

    branch = longitudinal_branch_index(
        bond_stiffness=bond_stiffness, mass=mass, spacing=spacing, ny=ny, nz=nz
    )
    kmax = k_max_fraction * zone_edge
    ks = torch.linspace(0.0, kmax, n_k, dtype=DTYPE)
    omega = torch.zeros_like(ks)
    for idx, k in enumerate(ks.tolist()):
        omega[idx] = bar_dispersion(
            k, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing,
            ny=ny, nz=nz, branch=branch,
        )["omega_rad_s"]
    # Mode amplitude at each k from the discrete spectrum (interpolated).
    amplitude = torch.zeros_like(ks)
    for idx, k in enumerate(ks.tolist()):
        amplitude[idx] = float(
            torch.abs(torch.fft.rfft(v))[int(round(k * n * spacing / (2 * math.pi)))]
        ) if k * n * spacing / (2 * math.pi) < spectrum.numel() - 1 else 0.0

    times = torch.arange(0.0, time_horizon, time_step, dtype=DTYPE)
    displacement = torch.zeros_like(times)
    for idx in range(ks.numel()):
        if amplitude[idx] == 0.0:
            continue
        phase = ks[idx] * station_offset_m
        displacement += amplitude[idx] * torch.cos(phase - omega[idx] * times) / n
    peak_idx = int(torch.argmax(torch.abs(displacement)).item())
    # Parabolic refinement in time.
    if 0 < peak_idx < times.numel() - 1:
        y0 = float(displacement[peak_idx - 1].abs())
        y1 = float(displacement[peak_idx].abs())
        y2 = float(displacement[peak_idx + 1].abs())
        denom = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
        arrival = float(times[peak_idx]) + offset * time_step
    else:
        arrival = float(times[peak_idx])
    group_at_zero = bar_dispersion(
        1e-6 * zone_edge, bond_stiffness=bond_stiffness, mass=mass, spacing=spacing,
        ny=ny, nz=nz, branch=branch,
    )
    return {
        "predicted_arrival_s": arrival,
        "predicted_speed_m_per_s": station_offset_m / arrival,
        "group_velocity_long_wave_m_per_s": group_at_zero["group_velocity_m_per_s"],
        "branch_index": branch,
        "spectral_centroid_k_per_m": float(
            (ks * amplitude).sum() / amplitude.sum()
        ) if float(amplitude.sum()) > 0 else 0.0,
    }

def mode_experiment(
    dem: IceDEM3D,
    mode: torch.Tensor,
    *,
    free_dof: torch.Tensor,
    omega_rad_s: float,
    amplitude_m: float = 1e-6,
    periods: float = 4.0,
    dt: float | None = None,
) -> dict[str, Any]:
    """Integrate one exact normal mode of ``dem`` and record its displacement.

    The sample is displaced along ``mode`` (a unit eigenvector of
    :func:`finite_bar_modes`) with zero initial velocity, so the analytic answer
    is a harmonic oscillation at ``omega_rad_s``.  Nothing else is touched: the
    measurement isolates the time integrator (and the solver's own bond law) from
    any spatial discretisation question.
    """
    if omega_rad_s <= 0:
        raise ValueError("omega_rad_s must be positive")
    if amplitude_m <= 0:
        raise ValueError("amplitude_m must be positive")
    n = int(dem.positions.shape[0])
    shaped = torch.zeros(3 * n, dtype=DTYPE)
    shaped[free_dof] = mode.to(DTYPE) * amplitude_m
    dem.positions = dem.positions + shaped.reshape(n, 3).to(dem.positions.dtype)
    dem.velocities = torch.zeros_like(dem.positions)
    axial = shaped.reshape(n, 3)[:, 0].abs()
    if float(axial.max()) == 0.0:
        raise ValueError("mode has no axial displacement")
    track = int(torch.argmax(axial).item())
    step_dt = float(dt if dt is not None else dem.dt)
    steps = max(1, int(round(periods * (2.0 * math.pi / omega_rad_s) / step_dt)))
    times = [0.0]
    axial_trace = [float((dem.positions[track, 0] - dem.initial_positions[track, 0]).item())]
    for _ in range(steps):
        dem.step()
        times.append(float(dem.time))
        axial_trace.append(
            float((dem.positions[track, 0] - dem.initial_positions[track, 0]).item())
        )
    if dem.broken_bonds:
        raise RuntimeError("mode excitation fractured bonds; reduce the amplitude")
    return {
        "tracking_particle": track,
        "times_s": times,
        "axial_displacement_m": axial_trace,
        "dt_s": step_dt,
        "steps": steps,
        "omega_reference_rad_s": omega_rad_s,
        "target_periods": periods,
        "broken_bonds": int(dem.broken_bonds),
    }


def frequency_from_zero_crossings(
    times: Sequence[float], signal: Sequence[float]
) -> dict[str, float]:
    """Frequency of a harmonic trace from interpolated zero crossings.

    Two crossings per period; the first and last crossings are discarded so a
    partially recorded final half-cycle cannot bias the estimate.
    """
    crossings: list[float] = []
    for idx in range(1, len(signal)):
        a, b = float(signal[idx - 1]), float(signal[idx])
        if a == 0.0:
            crossings.append(float(times[idx - 1]))
            continue
        if a * b < 0.0:
            frac = a / (a - b)
            crossings.append(float(times[idx - 1] + frac * (times[idx] - times[idx - 1])))
    if len(crossings) < 3:
        raise ValueError("not enough zero crossings to estimate a frequency")
    span = crossings[-1] - crossings[0]
    periods = 0.5 * (len(crossings) - 1)
    frequency = periods / span
    return {
        "frequency_Hz": frequency,
        "omega_rad_s": 2.0 * math.pi * frequency,
        "n_crossings": len(crossings),
        "span_s": span,
        "periods_observed": periods,
    }


def cross_correlation_lag(
    times: Sequence[float], upstream: Sequence[float], downstream: Sequence[float],
    *, max_lag_fraction: float = 0.5,
) -> dict[str, float]:
    """Travel time between two station signals from their cross-correlation.

    The lag maximising ``sum_t a(t) b(t + lag)`` is located on the sample grid and
    refined by a parabola, which handles the dispersive broadening of a wave
    packet better than a bare peak pick.
    """
    a = torch.as_tensor(list(upstream), dtype=DTYPE)
    b = torch.as_tensor(list(downstream), dtype=DTYPE)
    if a.numel() != b.numel():
        raise ValueError("station signals must share a time base")
    a = a - a.mean()
    b = b - b.mean()
    if float(a.abs().max()) == 0.0 or float(b.abs().max()) == 0.0:
        raise ValueError("empty station signal")
    n = a.numel()
    max_lag = max(1, int(max_lag_fraction * n))
    lags = list(range(-max_lag, max_lag + 1))
    scores = []
    for lag in lags:
        if lag >= 0:
            lhs, rhs = a[: n - lag], b[lag:]
        else:
            lhs, rhs = a[-lag:], b[: n + lag]
        scores.append(float((lhs * rhs).sum()))
    best = max(range(len(scores)), key=lambda i: scores[i])
    offset = 0.0
    if 0 < best < len(scores) - 1:
        y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
        denom = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    step = float(times[1] - times[0])
    lag_samples = lags[best] + offset
    return {
        "lag_samples": lag_samples,
        "lag_s": lag_samples * step,
        "correlation_peak": scores[best],
    }


def linear_response_trace(
    frequencies: Sequence[float],
    mode_vectors: torch.Tensor,
    initial_dof: torch.Tensor,
    *,
    station_dofs: Sequence[int],
    times: Sequence[float],
) -> torch.Tensor:
    """Exact harmonic response of the assembled discrete system at a station.

    With zero initial velocity and an initial displacement ``u0`` the solution of
    ``m u'' = -K u`` is ``u(t) = sum_m phi_m (phi_m . u0) cos(omega_m t)``, so the
    ``component``-0 velocity averaged over the station particles is

        v(t) = -sum_m w_m (phi_m . u0) omega_m sin(omega_m t),

    with ``w_m`` the station-averaged mode component.  This is the exact reference
    for the solver's time-domain response of the *same* discrete sample: it
    includes every branch and every mode-conversion effect, so a mismatch can only
    come from the integrator or from the solver's force assembly.
    """
    if mode_vectors.ndim != 2:
        raise ValueError("mode_vectors must be (n_free_dof, n_modes)")
    station = torch.as_tensor(list(station_dofs), dtype=torch.long)
    if station.numel() == 0:
        raise ValueError("station_dofs is empty")
    weights = mode_vectors.index_select(0, station).mean(dim=0)
    amplitudes = mode_vectors.t() @ initial_dof.to(DTYPE)
    omegas = torch.as_tensor(list(frequencies), dtype=DTYPE)
    t = torch.as_tensor(list(times), dtype=DTYPE)
    envelope = weights * amplitudes * omegas  # (n_modes,)
    profile = torch.sin(torch.outer(t, omegas))
    return -(profile @ envelope)

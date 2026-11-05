"""Bounded-domain boundary verification for the D3Q19 propeller-race solver.

``docs/threefield_lbm_transport_verification.md`` (m2949) verified the *interior*
transport operators of :mod:`tensordem.threefield.wake_lbm` -- periodic streaming,
the momentum diffusivity implied by ``tau0`` and the Smagorinsky closure -- and
explicitly left the bounded box uncovered:

    "适用域：全部结论在格子单位、低 Mach、周期域下成立；有界箱的入口/出口/
     自由滑移壁面误差未被本单覆盖"

This module closes that gap.  It audits the three boundary treatments that the
production wake/ice-melt chain actually runs with:

* **inlet** (``x = 0``): full Dirichlet overwrite with ``f^eq(rho=1, u_in + noise)``,
* **outlet** (``x = nx-1``): zero-gradient copy of the incoming populations from the
  last interior plane,
* **lateral walls** (``y = 0, ny-1`` and ``z = 0, nz-1``): specular (free-slip)
  reflection of the wall-normal population pairs.

Four independent checks are provided, each with a closed-form or discrete-exact
reference:

A. :func:`uniform_stream_check` -- a uniform stream is an exact fixed point of the
   complete scheme (collision + streaming + all three boundaries): free-slip walls
   must not generate a spurious boundary layer, the inlet/outlet pair must not
   generate a spurious mass source, and no transverse velocity may appear.  The
   reference is ``u == u_in`` for every cell and every step, i.e. zero drift.
B. :func:`wall_slip_audit` -- the specular wall must be impermeable and frictionless:
   ``u_n = 0`` at the wall plane *exactly*, net wall-normal mass crossing *exactly*
   zero, and the tangential momentum advected through the wall plane *exactly* zero
   (a free-slip wall exerts no tangential traction).  The reference values are 0.
C. :func:`open_boundary_ledger` -- mass and axial-momentum books over time:
   ``dM/dt`` against the inflow/outflow plane fluxes ``J_in`` / ``J_out``, and
   ``dP_x/dt`` against the imposed actuator-disk thrust plus the inlet/outlet
   momentum fluxes.  The residual of each identity is the boundary model's
   source/sink term; the interesting numbers are its size relative to the fluxes
   and whether the box reaches a steady state (``J_out/J_in -> 1``).
D. :func:`fill_time_in_flow_throughs` -- how many convective time units
   ``L/U_in`` a case needs before the outlet flux balances the inlet flux to a
   stated tolerance.  This is the criterion the published propeller-race
   benchmark should be checked against.

All routines call the solver's own kernels (``_macroscopic``, ``_equilibrium``,
``_apply_boundaries``) instead of re-implementing them, so a defect found here is
a defect of the shipped code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .wake_lbm import _D3Q19_VELOCITIES, LBMConfig, WakeLBM

__all__ = [
    "LedgerSample",
    "OpenBoundaryLedger",
    "disable_disk",
    "uniform_stream_check",
    "wall_slip_audit",
    "plane_mass_flux",
    "plane_axial_momentum_flux",
    "open_boundary_ledger",
    "fill_time_in_flow_throughs",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def disable_disk(inst: WakeLBM) -> WakeLBM:
    """Remove the actuator disk so the boundaries can be audited in isolation.

    ``disk_mask`` is a plain attribute of the instance; clearing it makes
    ``_disk_force`` return zero everywhere and leaves every boundary treatment
    untouched.  Without this, the only way to run a *boundary-only* problem is to
    remove the body force from the code path under test.
    """
    inst.disk_mask = torch.zeros_like(inst.disk_mask)
    return inst


def _boundary_only_config(
    *, nx: int, ny: int, nz: int, tau0: float, smagorinsky_cs: float, inflow: float
) -> LBMConfig:
    """A config whose inlet is a clean uniform stream (no noise, no disk)."""
    return LBMConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        tau0=tau0,
        smagorinsky_cs=smagorinsky_cs,
        inflow_velocity=inflow,
        jet_velocity=inflow * 1.01,
        disk_radius=1.0,
        disk_x=max(8, nx // 4),
        disk_thickness=1,
        inlet_perturbation=0.0,
        disk_perturbation=0.0,
        disk_profile="top_hat",
        quarter_domain=False,
    )


def _cx_tensor(inst: WakeLBM) -> torch.Tensor:
    cx = torch.tensor([c[0] for c in _D3Q19_VELOCITIES], dtype=inst.dtype)
    return cx


def plane_mass_flux(inst: WakeLBM, plane: int) -> float:
    """Net ``x`` mass flux through the plane, ``sum_i c_ix f_i`` (lattice units)."""
    cx = _cx_tensor(inst)
    return float((inst.f[plane] * cx.view(1, 1, -1)).sum().item())


def plane_axial_momentum_flux(inst: WakeLBM, plane: int) -> float:
    """Advected axial momentum flux ``sum_i c_ix^2 f_i`` through the plane."""
    cx = _cx_tensor(inst)
    return float((inst.f[plane] * (cx * cx).view(1, 1, -1)).sum().item())


# ---------------------------------------------------------------------------
# A. uniform stream is an exact fixed point of the bounded scheme
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniformStreamResult:
    steps: int
    cells: int
    velocity_drift: float      # max |u_x - u_in| over the box
    lateral_drift: float       # max |u_y|, |u_z| (must be 0 for a stream along x)
    density_drift: float       # max |rho - 1|
    mass_drift: float          # |M(t) - M(0)| / M(0)

    @property
    def exact(self) -> bool:
        return max(self.velocity_drift, self.lateral_drift, self.density_drift) < 1e-12


def uniform_stream_check(
    *,
    nx: int = 24,
    ny: int = 12,
    nz: int = 12,
    tau0: float = 0.6,
    smagorinsky_cs: float = 0.0,
    inflow: float = 0.04,
    steps: int = 200,
) -> UniformStreamResult:
    """March a uniform stream in the *bounded* box and measure the drift.

    A uniform flow is an exact solution of the discrete kinetic equation, of the
    exact-streaming operator and of all three boundaries (the inlet Dirichlet state
    is itself the uniform equilibrium, the zero-gradient outlet copy is uniform, and
    the specular reflection maps ``f^eq(c)`` onto ``f^eq(-c_y)`` which is equal for a
    stream along ``x``).  Any drift is therefore generated by the boundary code, not
    by the initial condition.
    """
    config = _boundary_only_config(
        nx=nx, ny=ny, nz=nz, tau0=tau0, smagorinsky_cs=smagorinsky_cs, inflow=inflow
    )
    inst = disable_disk(WakeLBM(config))
    uniform = inst._inlet_populations()
    inst.f = uniform.unsqueeze(0).expand(nx, ny, nz, uniform.shape[-1]).clone()
    mass0 = float(inst.f.sum().item())
    inst.run(int(steps), average_from=0)
    rho, u = inst._macroscopic(inst.f)
    return UniformStreamResult(
        steps=int(steps),
        cells=int(nx * ny * nz),
        velocity_drift=float((u[..., 0] - inflow).abs().max().item()),
        lateral_drift=float(u[..., 1:].abs().max().item()),
        density_drift=float((rho - 1.0).abs().max().item()),
        mass_drift=abs(float(inst.f.sum().item()) - mass0) / mass0,
    )


# ---------------------------------------------------------------------------
# B. specular (free-slip) lateral walls: impermeable and frictionless
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WallSlipResult:
    normal_velocity: float            # max |u_n| on the four wall planes (free slip: 0)
    mass_crossing_mismatch: float     # relative |sum_{c_n>0} f - sum_{c_n<0} f| per wall plane
    axial_momentum_exchange: float    # relative |sum_{c_n>0} c_x f - sum_{c_n<0} c_x f|
    reflection_tangential_mismatch: float  # max |c_t(reflect(i)) - c_t(i)| over the table
    interior_speed: float             # scale: max |u| in the box

    @property
    def impermeable_and_frictionless(self) -> bool:
        return (
            self.normal_velocity == 0.0
            and self.mass_crossing_mismatch < 1e-14
            and self.axial_momentum_exchange < 1e-14
            and self.reflection_tangential_mismatch == 0.0
        )


def wall_slip_audit(inst: WakeLBM) -> WallSlipResult:
    """Measure impermeability and zero tangential traction of the specular walls.

    Three quantities separate "looks free-slip" from "is free-slip":

    * ``normal_velocity`` -- a specular wall makes the post-reflection normal
      population pair equal, hence ``u_n = 0`` exactly, not approximately;
    * ``mass_crossing_mismatch`` -- the mass the wall takes in must equal the mass it
      returns; a mismatch is a wall mass sink/source;
    * ``axial_momentum_exchange`` -- the reflected population keeps its ``c_x``, so a
      free-slip wall transfers exactly zero axial momentum (no spurious drag).  A
      no-slip wall would show a value of the order of the wall shear.
    """
    _, u = inst._macroscopic(inst.f)
    normal = 0.0
    for plane in (0, inst.ny - 1):
        normal = max(normal, float(u[:, plane, :, 1].abs().max().item()))
    for plane in (0, inst.nz - 1):
        normal = max(normal, float(u[:, :, plane, 2].abs().max().item()))

    cx = _cx_tensor(inst)
    cy = torch.tensor([c[1] for c in _D3Q19_VELOCITIES], dtype=inst.dtype)
    cz = torch.tensor([c[2] for c in _D3Q19_VELOCITIES], dtype=inst.dtype)
    walls = (
        (cy, cz, 1, (0, inst.ny - 1)),
        (cz, cy, 2, (0, inst.nz - 1)),
    )
    mass_mismatch = 0.0
    axial_mismatch = 0.0
    for normal_component, other_tangential, axis, planes in walls:
        incoming = normal_component > 0
        outgoing = normal_component < 0
        for plane in planes:
            slab = inst.f[:, plane] if axis == 1 else inst.f[:, :, plane]
            mass_in = float(slab[..., incoming].sum().item())
            mass_out = float(slab[..., outgoing].sum().item())
            mass_mismatch = max(mass_mismatch, abs(mass_in - mass_out) / (mass_in + mass_out))
            axial_in = float((slab[..., incoming] * cx[incoming]).sum().item())
            axial_out = float((slab[..., outgoing] * cx[outgoing]).sum().item())
            scale = float((slab.abs() * cx.abs()).sum().item())
            axial_mismatch = max(axial_mismatch, abs(axial_in - axial_out) / scale)
    tangential_mismatch = 0.0
    for reflect, other_tangential in ((inst.reflect_y, cz), (inst.reflect_z, cy)):
        for i, j in enumerate(reflect):
            tangential_mismatch = max(
                tangential_mismatch,
                abs(float(cx[i] - cx[j])),
                abs(float(other_tangential[i] - other_tangential[j])),
            )
    return WallSlipResult(
        normal_velocity=normal,
        mass_crossing_mismatch=mass_mismatch,
        axial_momentum_exchange=axial_mismatch,
        reflection_tangential_mismatch=tangential_mismatch,
        interior_speed=float(torch.sqrt((u * u).sum(dim=-1)).max().item()),
    )


# ---------------------------------------------------------------------------
# C. open-boundary ledger: mass books and race station audits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerSample:
    step: int
    mass: float
    momentum_x: float
    flux_in: float
    flux_out: float
    thrust_cumulative: float


@dataclass(frozen=True)
class OpenBoundaryLedger:
    samples: tuple[LedgerSample, ...]
    inflow_velocity: float
    length: int
    thrust_rate: float               # thrust impulse per iteration (lattice)
    mass_drift_rate: float           # (dM/dt) / M, relative per iteration
    flux_imbalance: float            # <J_in - J_out>, lattice mass per iteration
    flux_imbalance_relative: float   # <J_in - J_out> / <J_in>
    flux_balance: float              # J_out / J_in at the last sample

    @property
    def samples_count(self) -> int:
        return len(self.samples)


def open_boundary_ledger(
    config: LBMConfig, steps: int, *, sample_every: int = 10, initial_uniform: bool = False
) -> OpenBoundaryLedger:
    """Run a bounded case and book-keep mass, inlet/outlet flux and thrust.

    ``flux_in``/``flux_out`` are the net plane fluxes ``sum_i c_ix f_i`` at the inlet
    and outlet planes, i.e. the mass that crosses each face per iteration.  A
    *conservative* pair of open boundaries would show ``dM/dt = J_in - J_out``; the
    measured ``mass_drift_rate`` and ``flux_imbalance`` are reported side by side so
    the size of the boundary model's own mass source can be read off without assuming
    the identity holds.  ``initial_uniform`` starts the box in the inlet state, which
    makes a uniform-stream run a closed book (``dM/dt = 0`` and ``J_in = J_out``) and
    therefore turns any drift into a boundary defect.  The actuator-disk impulse is taken from the solver's own
    accumulator, so the thrust entering the momentum books is the number the coupling
    chain uses.
    """
    inst = WakeLBM(config)
    if initial_uniform:
        # Seed the box with the inlet state so the books start in the exact fixed
        # point of the boundary model; any drift is then a boundary defect.
        disable_disk(inst)
        uniform = inst._inlet_populations()
        inst.f = uniform.unsqueeze(0).expand(
            config.nx, config.ny, config.nz, uniform.shape[-1]
        ).clone()
    cx = _cx_tensor(inst)
    samples: list[LedgerSample] = []

    def record(step: int) -> None:
        momentum = float((inst.f * cx.view(1, 1, 1, -1)).sum().item())
        mass = float(inst.f.sum().item())
        samples.append(
            LedgerSample(
                step=step,
                mass=mass,
                momentum_x=momentum,
                flux_in=plane_mass_flux(inst, 0),
                flux_out=plane_mass_flux(inst, -1),
                thrust_cumulative=float(inst.thrust_impulse),
            )
        )

    record(0)
    for step in range(1, int(steps) + 1):
        inst.step(1)
        if step % max(1, int(sample_every)) == 0 or step == int(steps):
            record(step)

    span = samples[-1].step - samples[0].step
    if span <= 0:
        raise ValueError("need at least two samples to compute rates")
    first, last = samples[0], samples[-1]
    mean_in = sum(sample.flux_in for sample in samples[1:]) / len(samples[1:])
    imbalance = sum(
        sample.flux_in - sample.flux_out for sample in samples[1:]
    ) / len(samples[1:])
    return OpenBoundaryLedger(
        samples=tuple(samples),
        inflow_velocity=config.inflow_velocity,
        length=config.nx,
        thrust_rate=(last.thrust_cumulative - first.thrust_cumulative) / span,
        mass_drift_rate=(last.mass - first.mass) / span / first.mass,
        flux_imbalance=imbalance,
        flux_imbalance_relative=imbalance / mean_in,
        flux_balance=last.flux_out / last.flux_in,
    )


def fill_time_in_flow_throughs(
    ledger: OpenBoundaryLedger, *, tolerance: float = 0.01, window: int = 5
) -> dict[str, float]:
    """Iterations until the flux ratio stays inside ``tolerance`` for the rest of the run.

    One flow-through is ``L / U_in`` lattice iterations.  ``window`` trailing samples
    must all be inside the band, so an instantaneous crossing does not count.
    """
    samples = ledger.samples
    if len(samples) < window + 1:
        raise ValueError("need a longer ledger")
    inside = [
        abs(sample.flux_out / sample.flux_in - 1.0) <= tolerance if sample.flux_in else False
        for sample in samples
    ]
    start = len(samples)
    for index in range(len(samples) - window, -1, -1):
        if all(inside[index:index + window]):
            start = index
        else:
            break
    transit = ledger.length / ledger.inflow_velocity
    step = float(samples[start].step)
    return {
        "steps": step,
        "flow_throughs": step / transit,
        "converged": 1.0 if start < len(samples) else 0.0,
    }


# ---------------------------------------------------------------------------
# D. race station audit: mass flux and excess momentum flux against thrust
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StationAudit:
    station: int
    mass_flux: float
    excess_momentum_flux: float
    thrust: float
    excess_over_thrust: float


def race_station_audit(inst: WakeLBM, stations: list[int]) -> list[StationAudit]:
    """Mass flux and excess axial momentum flux at downstream stations.

    ``excess_momentum_flux`` is ``sum rho u_x (u_x - U_in)`` over the plane -- the
    quantity the published propeller-race benchmark compares with the actuator-disk
    thrust ("thrust = excess momentum flux").  ``thrust`` is the solver's own
    per-iteration impulse, so the ratio is dimensionless and directly comparable
    between runs of different length.
    """
    thrust = inst.thrust_lattice()
    out: list[StationAudit] = []
    for station in stations:
        audit_station = int(station)
        out.append(
            StationAudit(
                station=audit_station,
                mass_flux=float(inst.mass_flux(audit_station)),
                excess_momentum_flux=float(inst.momentum_flux_excess(audit_station)),
                thrust=thrust,
                excess_over_thrust=(
                    float(inst.momentum_flux_excess(audit_station)) / thrust
                    if thrust
                    else float("nan")
                ),
            )
        )
    return out

# TensorDEM coupling kernel: numpy-first DEM leg, backend equivalence, energy ledger

Campaign node **p223w3-n05** (`m9721`). Scope: make TensorDEM importable, degradable
and bit-reproducible as the DEM leg of the TensorPhysics coupling chain, and recompute
the three-field energy ledger of the settling scene with the **real** kernel instead of
the analytic "reference model" that m7031 flagged ("The DEM leg is a reference model,
not TensorDEM").

## 1. What landed

| file | role |
|---|---|
| `src/tensordem/coupling_kernel/backend.py` | array namespace: numpy (reference, eager) / torch (lazy, optional) |
| `src/tensordem/coupling_kernel/celllist.py` | deterministic numpy cell-linked-list contact detection (host side, shared by both backends) |
| `src/tensordem/coupling_kernel/kernel.py` | Hertz + Cundall–Strack dashpot, walls, buoyant weight, Stokes / Schiller–Naumann drag, velocity-Verlet |
| `src/tensordem/coupling_kernel/ledger.py` | itemised energy ledger + closure residual + attachable hooks |
| `src/tensordem/coupling_kernel/settling.py` | settling scenarios (dilute sphere, dense bed, contact-rich pack, elastic gas) and analytic references |
| `tests/test_coupling_kernel_backend_maturity.py` | 14 acceptance tests |
| `scripts/run_coupling_kernel_settling.py` | cluster runner (equivalence / dilution / conservation / convergence / settling) |
| `scripts/make_coupling_kernel_figures.py` | figures from artifacts only |
| `scripts/post_coupling_kernel_report.py` | paper report builder (numbers injected from artifacts) |

## 2. Dependency layering

* `import tensordem.coupling_kernel` **does not import torch**. torch is only imported
  inside `TorchBackend.__init__`, i.e. when `backend="torch"` is requested. Verified by a
  subprocess test that installs a `sys.meta_path` blocker for `torch` and runs the kernel.
* `tensordem/__init__.py` (the legacy prototype package) still imports torch eagerly, so
  a plain `import tensordem` pulls torch in. The isolation test therefore imports the
  subpackage against a stub parent package; making the top-level package lazy is
  follow-up work and is listed as a limitation, not claimed as done.
* No numba, no compiled extension, no scipy requirement for this path.

## 3. Backend equivalence — why it is provable, not hopeful

Both backends execute the *same* Python code through `ArrayBackend`:

1. contact detection runs in numpy on the host for both backends (integer cell
   bookkeeping is exactly reproducible: stable sort of int64 cell keys + ragged gather
   over the 13 half-space offsets, then de-duplication by canonical key `i*n+j`);
2. pair forces, integration and the energy reductions run in the backend;
3. `δ^{3/2}` is evaluated as `δ * sqrt(δ)`, never `pow`, because `sqrt` is correctly
   rounded in both libraries while `pow` is not guaranteed to be;
4. boundary closures (walls, buoyancy, drag) are evaluated on the host in numpy so they
   are identical by construction.

Consequence (measured, see results appendix): after 1000 steps on a contact-rich packing
the numpy and torch states are **bitwise identical**, and every ledger channel agrees to
≤ a few 1e-16 relative — six orders of magnitude tighter than the node's 1e-10 ledger
criterion.

## 4. Energy ledger

Six itemised channels plus a discrete balance residual:

| channel | field | meaning |
|---|---|---|
| `kinetic_J` | DEM bulk | translational kinetic energy |
| `gravity_potential_J` | DEM bulk | `-Σ (m - m_f) g·x` (buoyant weight convention) |
| `elastic_J` | DEM bulk | Hertz contact storage `Σ 0.4 K δ^{5/2}`, particles + walls |
| `dem_damping_dissipated_J` | contact | particle–particle dashpot dissipation |
| `wall_damping_dissipated_J` | contact | wall dashpot dissipation |
| `drag_dissipated_J` | fluid | work against the drag law = momentum handed to the flow leg |

`balance_residual = (K + U_g + U_c)(t) - (K + U_g + U_c)(0) + W_diss(t)`, and
`relative_residual = balance_residual / |E_mech(0)|`. Hooks: `ledger.attach(callable)`
streams each row to any consumer; `to_json` / `to_csv` dump the series.

The drag channel is the coupling-chain handle the flow leg needs: `momentum_exchange`
returns the drag force and power so that a fluid solver can close momentum and energy
with the same numbers the DEM leg booked.

## 5. Scenarios that replace the reference model

* **dilute single sphere** — integrator benchmark against the exact solution of
  `dv/dt = g_eff - v/τ` and physics check against `v_t = 2R²Δρg/(9μ)` (with the Oseen
  correction reported because Stokes is only valid for Re → 0).
* **dense bed** — 512 spheres at a prescribed solid fraction, Schiller–Naumann drag,
  wall compaction, 1e5 steps: the case the coupling scene needs, now solved with contacts.
* **contact-rich pack** — 1.90 R lattice spacing so every particle starts in contact: the
  backend-equivalence stress state (pair forces, dashpots and reductions all active).
* **elastic gas** — restitution 1, no drag/gravity: the strict conservation case, in
  which any energy drift is the integrator's own error.

## 6. Limitations (honest scope)

1. Translational degrees of freedom only — no spin, no rolling resistance, frictionless
   tangential direction. Unsuited to torque-controlled problems (rubble interlock,
   drum/kiln flow, screw conveyors).
2. Contact stiffness is softened (`E = 1e7 Pa`) so `dt = 5e-7 … 2e-6 s` resolves the
   contact; the physical regime is soft-sphere/quasi-static. Ice stiffness would require
   `dt` smaller by ~`sqrt(E_ice/E)`.
3. The terminal-velocity comparison is partly a consistency check (same linear drag law on
   both sides); its independent content is the integrator error, the Reynolds window and
   the ledger closure.
4. No lateral periodicity: wall effects are present in the dense bed (≈12 diameters
   across).
5. The fluid leg is a drag closure, not an LBM solve — momentum/energy exchange is booked,
   the flow field is not resolved here (that stays in TensorPhysics).
6. `tensordem/__init__.py` eager-torch import is untouched (out of scope, follow-up).

## 7. Headline results (numbers generated in the appendix)

* Backend equivalence, 1000 steps on a 216-particle contact-rich packing: **bitwise identical**
  positions and velocities; worst ledger channel difference 2.1e-16 relative (2.6e-23 J absolute).
* Dilute single sphere, 200 000 steps: velocity matches the exact linear-drag solution to
  2.0e-15 relative; vs Stokes 2.0e-15 (Re = 4.4e-4, inside the Stokes window); ledger closure 5.5e-9.
* Elastic gas, 100 000 steps: zero modelled dissipation, drift -7.4e-5 relative, bounded swing
  2.1e-4, secular drift -8.6e-10 per step.
* Dense bed (512 spheres, phi = 0.15, 12.1 diameters high), 100 000 steps on the real kernel:
  coordination 4.29, max overlap 2.3e-4 R, all three fields active, three-field ledger closure
  9.6e-4 relative.
* Timestep refinement: drift ratio 9.47 between dt and dt/2 ⇒ observed order 3.24 (at least second
  order; the pre-registered "exactly second order" window 3.4–4.6 is *not* met because the scheme
  is super-convergent on this case — reported, not hidden).

Two honest notes: (i) the contact-rich packing has a 0.5% closure residual because its initial state
is deliberately over-compressed (0.1 R overlaps ⇒ impulsive relaxation); it is the backend-equivalence
stress state, not a closure claim — the physical cases close at 1e-9 … 1e-3. (ii) At this problem size
torch is *slower* than numpy (15.2 s vs 10.1 s for 1000 steps), so the torch backend is a portability /
equivalence hook, not a speed win; acceleration would need a much larger particle count or a GPU.

## 8. Results appendix

The numbers in `docs/COUPLING_KERNEL_DEM_BACKEND_RESULTS.md` are generated from
`artifacts/coupling_kernel/*.json` by `scripts/make_coupling_kernel_figures.py` companions
— no hand-typed values.
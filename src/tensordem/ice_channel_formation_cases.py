"""Experiment matrix for the W25 ``formation-ice-channel`` rework round.

Single source of truth for *which* DEM runs the paper experiment performs.
Imported by the runner (``w25_formation_runner.py``), the collector
(``w25_formation_collect.py``) and the tests, so the matrix cannot drift.

Design of the matrix (pre-registered in
``prereg/w25_formation_ice_channel_prereg.json``):

``single``   7 runs, one hull, speeds 0.5 ... 2.0 m/s -- the degenerate case for
             the Lindqvist reconciliation.
``formation`` 35 runs, three hulls in column, spacing 1L ... 5L x the 7 speeds --
             the saving-vs-spacing surface.
``gci``      4 runs, ``N x 2`` and ``N x 4`` particle refinement of the base
             single-ship and 2L-spacing cases, at 1.0 m/s.
``numerics`` 1 + 3 runs: one timestep-halving control, three cross-channel
             drift sensitivity controls.

Particle refinement keeps the ice concentration fixed and halves the floe
radius per factor 2 in floe count (``r -> r/sqrt(2)``), and the timestep is
scaled with the floe radius (``dt -> dt/sqrt(2)``, i.e. ``dt`` per floe-count
factor 2) so that the contact-frequency CFL number stays constant: the
anchored contact stiffness is radius independent while the floe mass scales as
``r^2``, hence ``omega = sqrt(k/m) ~ 1/r``.
"""
from __future__ import annotations

from .ice_channel_formation import CASE_MEAN_RADIUS_M, CASE_SEED, FormationCase

SPEEDS_M_S = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SPACINGS_L = (1.0, 2.0, 3.0, 4.0, 5.0)
BASE_DT_S = 0.008
GCI_SPEED_M_S = 1.0
GCI_SPACING_L = 1.0
#: formations are run as a 3-member ensemble; the single-ship reference (the
#: denominator of every saving) gets 5 members because its seed spread is the
#: dominant error term of the saving
ENSEMBLE_SEED_OFFSETS = (0, 1, 2)
SINGLE_SEED_OFFSETS = (0, 1, 2, 3, 4)
#: formations that also get an ensemble member (base case + 2 extra seeds)
DRIFT_BAND_M_S = (0.03, 0.05, 0.08)


def _dt_for(mean_radius_m: float) -> float:
    """Timestep that keeps ``omega*dt`` constant as the floe radius shrinks."""
    return BASE_DT_S * mean_radius_m / CASE_MEAN_RADIUS_M


def case(
    label: str,
    *,
    role: str,
    n_ships: int = 1,
    spacing_L: float | None = None,
    speed_m_s: float = GCI_SPEED_M_S,
    mesh_level: int = 1,
    drift_m_s: float = 0.05,
    dt: float | None = None,
    seed_offset: int = 0,
    notes: str = "",
) -> FormationCase:
    mean_radius = CASE_MEAN_RADIUS_M / (mesh_level ** 0.5)
    return FormationCase(
        label=label,
        n_ships=n_ships,
        spacing_L=spacing_L,
        speed_m_s=speed_m_s,
        mean_radius_m=CASE_MEAN_RADIUS_M,
        drift_m_s=drift_m_s,
        dt=_dt_for(mean_radius) if dt is None else dt,
        seed=CASE_SEED + 1000 * seed_offset,
        mesh_level=mesh_level,
        role=role,
        notes=notes,
    )


def speed_label(speed: float) -> str:
    return f"{speed:.2f}".replace(".", "p")


def build_matrix() -> list[FormationCase]:
    """Return the full, ordered case list of the experiment.

    Every physical case is run as a **seed ensemble** (:data:`ENSEMBLE_SEED_OFFSETS`):
    the measured mean ice load of one 8/10 broken-ice realisation fluctuates by
    several tens of percent between seeds (single-floe contacts dominate the
    time mean), so a single realisation cannot support a saving curve.  The
    ensemble mean is the reported value and the seed spread is its error bar.
    """
    out: list[FormationCase] = []
    for offset in SINGLE_SEED_OFFSETS:
        suffix = "" if offset == 0 else f"_e{offset}"
        # degenerate single-ship cases: the denominator of every saving
        for speed in SPEEDS_M_S:
            out.append(case(f"single_V{speed_label(speed)}{suffix}", role="single",
                            speed_m_s=speed, seed_offset=offset,
                            notes="degenerate single-ship case"))
    for offset in ENSEMBLE_SEED_OFFSETS:
        suffix = "" if offset == 0 else f"_e{offset}"
        # three-ship column formation, spacing x speed surface
        for speed in SPEEDS_M_S:
            for spacing in SPACINGS_L:
                out.append(case(
                    f"form_V{speed_label(speed)}_S{spacing:g}L{suffix}",
                    role="formation", n_ships=3, spacing_L=spacing,
                    speed_m_s=speed, seed_offset=offset,
                    notes="three-ship column formation"))
    # --- particle refinement family (GCI): base seed, nested realisations ---
    for level in (2, 4):
        out.append(case(f"single_V{speed_label(GCI_SPEED_M_S)}_Nx{level}", role="gci",
                        speed_m_s=GCI_SPEED_M_S, mesh_level=level,
                        notes="grid-refinement family, single ship"))
        out.append(case(f"form_V{speed_label(GCI_SPEED_M_S)}_S{GCI_SPACING_L:g}L_Nx{level}",
                        role="gci", n_ships=3, spacing_L=GCI_SPACING_L,
                        speed_m_s=GCI_SPEED_M_S, mesh_level=level,
                        notes="grid-refinement family, 3-ship column"))
    # --- timestep control ---------------------------------------------------
    out.append(case(f"single_V{speed_label(GCI_SPEED_M_S)}_dt_half", role="numerics",
                    speed_m_s=GCI_SPEED_M_S, dt=0.5 * _dt_for(CASE_MEAN_RADIUS_M),
                    notes="timestep halving at base mesh"))
    # --- cross-channel drift sensitivity ------------------------------------
    for drift in (DRIFT_BAND_M_S[0], DRIFT_BAND_M_S[2]):
        out.append(case(f"single_V{speed_label(GCI_SPEED_M_S)}_drift{drift:g}".replace(".", "p"),
                        role="numerics", speed_m_s=GCI_SPEED_M_S, drift_m_s=drift,
                        notes="drift sensitivity, single ship"))
    out.append(case("form_V0p50_S5L_drift0p08", role="numerics", n_ships=3,
                    spacing_L=5.0, speed_m_s=0.5, drift_m_s=DRIFT_BAND_M_S[2],
                    notes="drift sensitivity at the largest spacing"))
    return out


def cost_proxy(item: FormationCase) -> float:
    """Relative wall-cost estimate, used only to balance shards.

    Steps scale as ``travel/(V dt)`` and the per-step cost as ``N^2`` with
    ``N`` proportional to the field area (fixed width, so ``~L_field``).
    """
    steps = item.field_length_m / (item.speed_m_s * item.dt)
    n_floes = item.field_length_m * item.field_width_m / (
        3.141592653589793 * item.mean_radius_refined_m ** 2 / item.concentration)
    return steps * n_floes ** 2 / 1.0e6


def labels() -> list[str]:
    return [item.label for item in build_matrix()]


if __name__ == "__main__":  # pragma: no cover - manual inspection
    for item in build_matrix():
        print(f"{item.label:34s} role={item.role:10s} n={item.n_ships} "
              f"S={item.spacing_m:6.1f}m V={item.speed_m_s:4.2f} "
              f"L_field={item.field_length_m:6.1f}m r={item.mean_radius_refined_m:4.2f}m "
              f"dt={item.dt:.4f}")
    print(f"total cases: {len(build_matrix())}")
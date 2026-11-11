"""Per-impact erosion bookkeeping that a host solver can call in its own loop.

This module is the *inbound* half of the verification face: a main solver
(:class:`tensordem.slurry_transport.pipe_dem.PipeSectionDEMCFD`, or any other
driver that knows the impact kinematics) reports the impacts it detects, and the
recorder turns them into the quantities the criteria are written on --
cumulative erosion ratio ``E`` [kg/kg], removed material mass, wall recession
and a ledger that closes against the crater geometry.

It deliberately contains no solver state and no time loop, so that the same
recorder can be driven by:

* a DEM-CFD slurry pipe (integration caller, see
  ``benchmarks/erosion_pipe_coupled.py`` and
  ``tests/test_erosion_pipe_integration.py``),
* a ballistic sand-jet stream (``erosion_dem.stream``),
* a hand-written list of impacts from an experiment.

Physics comes from :mod:`tensordem.erosion_dem.crater` (one impact -> spherical
cap) exactly as in the standalone face; nothing is re-derived here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from tensordem.erosion.materials import Material

from .crater import CraterGeometry, OkaAnchoredLaw, single_impact_crater

__all__ = ["ImpactErosionRecorder", "SECONDS_PER_YEAR"]

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass
class ImpactErosionRecorder:
    """Accumulate single impacts into a closed erosion ledger.

    Parameters
    ----------
    material :
        Target wall material (density drives mass <-> volume, hardness enters the
        law through ``law``).
    law :
        Erosion law of the wall.  ``None`` falls back to Finnie's closed-form
        cutting ratio (uncalibrated lower bound of the ductile family).
    wall_area_m2 :
        Area of the eroding wall patch, used for the mean recession depth and the
        mm/year extrapolation.  ``None`` disables those two outputs.
    keep_records :
        Store the per-impact records (speed, angle, removed volume).  Needed to
        replay the same impacts through an independent reference model, and to
        make the benchmark auditable; off by default to keep long runs cheap.
    """

    material: Material
    law: OkaAnchoredLaw | None = None
    wall_area_m2: float | None = None
    keep_records: bool = True

    n_impacts: int = field(default=0, init=False)
    n_rejected: int = field(default=0, init=False)
    contact_mass_kg: float = field(default=0.0, init=False)
    removed_volume_m3: float = field(default=0.0, init=False)
    removed_mass_kg: float = field(default=0.0, init=False)
    max_depth_m: float = field(default=0.0, init=False)
    _speed_sum: float = field(default=0.0, init=False)
    _angle_sum: float = field(default=0.0, init=False)
    records: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.wall_area_m2 is not None and not self.wall_area_m2 > 0.0:
            raise ValueError("wall_area_m2 must be > 0 when given")

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def record(
        self,
        speed_m_s: float,
        alpha_deg: float,
        particle_diameter_m: float,
        particle_density_kg_m3: float,
    ) -> dict | None:
        """Book one impact; returns its record, or ``None`` if nothing happens.

        ``speed_m_s`` is the impact speed magnitude, ``alpha_deg`` the angle
        between the velocity and the wall surface (90 deg = normal impact).  A
        zero-speed impact removes nothing at all (exactly 0, not merely small),
        which is the negative control the criteria ask for.
        """
        v = float(speed_m_s)
        a = float(alpha_deg)
        d = float(particle_diameter_m)
        rho = float(particle_density_kg_m3)
        m_p = rho * (1.0 / 6.0) * 3.141592653589793 * d**3
        if v <= 0.0:
            return None
        try:
            crater: CraterGeometry = single_impact_crater(v, a, d, rho,
                                                          self.material, law=self.law)
        except ValueError:
            # the spherical-cap inversion refuses a removed volume that does not
            # fit inside the particle radius: that is perforation, not erosion
            self.n_rejected += 1
            return None
        if not crater.physically_admissible:
            # penetration deeper than the particle radius is perforation, not
            # erosion: refuse to book it rather than extrapolate (same guard as
            # the standalone face).
            self.n_rejected += 1
            return None
        self.n_impacts += 1
        self.contact_mass_kg += m_p
        self.removed_volume_m3 += crater.removed_volume_m3
        self.removed_mass_kg += crater.removed_mass_kg
        self.max_depth_m = max(self.max_depth_m, crater.depth_m)
        self._speed_sum += v
        self._angle_sum += a
        rec = {
            "speed_m_s": v,
            "alpha_deg": a,
            "particle_diameter_m": d,
            "particle_mass_kg": m_p,
            "removed_volume_m3": crater.removed_volume_m3,
            "removed_mass_kg": crater.removed_mass_kg,
            "depth_m": crater.depth_m,
            "crater_radius_m": crater.radius_m,
        }
        if self.keep_records:
            self.records.append(rec)
        return rec

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    @property
    def erosion_ratio_kg_kg(self) -> float:
        """Removed wall mass per unit erodent mass that actually struck."""
        if self.contact_mass_kg <= 0.0:
            return 0.0
        return self.removed_mass_kg / self.contact_mass_kg

    @property
    def ledger_residual(self) -> float:
        """Relative closure residual between the mass and the volume books."""
        if self.removed_mass_kg <= 0.0:
            return 0.0
        by_volume = self.removed_volume_m3 * self.material.density_kg_m3
        return abs(self.removed_mass_kg - by_volume) / self.removed_mass_kg

    def recession_depth_m(self) -> float:
        """Mean wall recession over the patch (removed volume / patch area)."""
        if self.wall_area_m2 is None:
            raise ValueError("wall_area_m2 was not supplied")
        return self.removed_volume_m3 / self.wall_area_m2

    def wall_loss_mm_per_year(
        self,
        solids_mass_rate_kg_s: float,
        *,
        area_m2: float | None = None,
    ) -> float:
        """Extrapolate the measured ratio to a thickness-loss rate.

        ``loss = E * mdot_solid / (rho_t A)`` -- the same conversion DNV-RP-O501
        uses, so the number is directly comparable with a design value.  The rate
        is linearly extrapolated from the sampled window; that is an
        extrapolation and is flagged as such in the report.
        """
        if not solids_mass_rate_kg_s >= 0.0:
            raise ValueError("solids_mass_rate_kg_s must be >= 0")
        area = self.wall_area_m2 if area_m2 is None else float(area_m2)
        if area is None or not area > 0.0:
            raise ValueError("a positive wall area is required")
        mass_rate = self.erosion_ratio_kg_kg * solids_mass_rate_kg_s
        thickness_rate = mass_rate / (self.material.density_kg_m3 * area)
        return thickness_rate * SECONDS_PER_YEAR * 1000.0

    def state(self) -> dict:
        """Scalar summary of the ledger (all keys JSON-safe)."""
        mean_v = self._speed_sum / self.n_impacts if self.n_impacts else 0.0
        mean_a = self._angle_sum / self.n_impacts if self.n_impacts else 0.0
        out = {
            "material": self.material.name,
            "law": "Finnie (uncalibrated)" if self.law is None else "OkaAnchoredLaw",
            "n_impacts": self.n_impacts,
            "n_rejected": self.n_rejected,
            "contact_mass_kg": self.contact_mass_kg,
            "removed_volume_m3": self.removed_volume_m3,
            "removed_mass_kg": self.removed_mass_kg,
            "erosion_ratio_kg_kg": self.erosion_ratio_kg_kg,
            "ledger_residual": self.ledger_residual,
            "max_crater_depth_m": self.max_depth_m,
            "mean_impact_speed_m_s": mean_v,
            "mean_impact_angle_deg": mean_a,
            "masses_non_negative": bool(
                self.contact_mass_kg >= 0.0
                and self.removed_mass_kg >= 0.0
                and self.removed_volume_m3 >= 0.0
            ),
            "removed_mass_le_erodent_mass": bool(
                self.removed_mass_kg <= self.contact_mass_kg * 2.0
            ),
        }
        if self.wall_area_m2 is not None:
            out["wall_area_m2"] = self.wall_area_m2
            out["mean_recession_m"] = self.recession_depth_m()
        return out

    def replay(
        self,
        rate_fn,
        *,
        keys: Mapping[str, str] | None = None,
    ) -> dict:
        """Replay the stored impacts through an independent rate model.

        ``rate_fn(speed_m_s, alpha_deg) -> ratio_kg_kg`` is any reference model
        (e.g. :func:`tensordem.erosion.models.dnv_ductile_erosion_ratio` bound to
        a material).  Returns the reference ratio aggregated over the *same*
        impact records, plus the per-impact relative deviations, so a benchmark
        compares like with like (identical impacts, different physics).
        """
        if not self.records:
            raise ValueError("no impact records stored (keep_records=False?)")
        k = keys or {}
        ref_mass = 0.0
        devs = []
        for rec in self.records:
            ratio = float(rate_fn(rec[k.get("speed", "speed_m_s")],
                                  rec[k.get("alpha", "alpha_deg")]))
            ref_removed = ratio * rec["particle_mass_kg"]
            ref_mass += ref_removed
            if rec["removed_mass_kg"] > 0.0:
                devs.append(abs(rec["removed_mass_kg"] - ref_removed)
                            / rec["removed_mass_kg"])
        ref_ratio = ref_mass / self.contact_mass_kg if self.contact_mass_kg else 0.0
        return {
            "reference_erosion_ratio_kg_kg": ref_ratio,
            "model_erosion_ratio_kg_kg": self.erosion_ratio_kg_kg,
            "relative_deviation_of_ratio": (
                abs(self.erosion_ratio_kg_kg - ref_ratio) / ref_ratio
                if ref_ratio > 0.0 else float("inf")
            ),
            "per_impact_max_relative_deviation": max(devs) if devs else 0.0,
            "per_impact_mean_relative_deviation": (
                sum(devs) / len(devs) if devs else 0.0
            ),
            "n_replayed": len(self.records),
        }
"""End-to-end verification of the columnar-ice anisotropy model.

This module is the machine-readable half of the node report: every acceptance
criterion is computed here from the model objects, with the number, the
threshold and the pass/fail *together*, so a reader (or the campaign gate) can
re-derive the verdict without re-reading prose.

Criterion map
-------------
(a) three-direction strength ratio   ``sigma_par/sigma_perp`` inside 3-5
(b) 45 deg prediction error          continuum vs slip-system reference
(c) isotropic degeneration control   uniform ODF -> ratio 1 (<= 5 %)
(d) fabric ledger                    record / query / round-trip / figure
(e) pytest coverage                  counted by the test suite (reported, not
                                     computed here)
(f) feature-ledger linkage           checked against the JSON link record

The DEM block (``dem``) is optional because it steps the engine; it supplies the
isotropic negative control *on the discrete side* and the measured lattice
saturation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .fabric import FabricLedger, FabricODF
from .slip_systems import SlipSystemModel, load_direction
from .yield_surface import (
    COLUMNAR_RATIO_RANGE,
    HillTI,
    calibrated_slip_reference,
    strength_surface_from_fabric,
)

__all__ = [
    "AnisotropyStudy",
    "default_angles",
    "run_verification",
]

#: Direction sweep used by the report figures (0 deg = load along the columns).
DEFAULT_ANGLES: tuple[float, ...] = (0.0, 10.0, 15.0, 20.0, 30.0, 40.0, 45.0,
                                     50.0, 60.0, 70.0, 80.0, 90.0)


def default_angles() -> tuple[float, ...]:
    return DEFAULT_ANGLES


@dataclass
class AnisotropyStudy:
    """Container for the model objects and the derived curves."""

    odf_columnar: FabricODF
    odf_uniform: FabricODF
    sigma_perp: float
    ratio_columnar: float
    convention: str
    slip: SlipSystemModel
    surface: HillTI
    surface_isotropic: HillTI
    angles: tuple[float, ...]
    odf_uniform_dense: FabricODF | None = None

    # -- curves -----------------------------------------------------------
    def slip_curve(self, odf: FabricODF | None = None,
                   convention: str | None = None) -> list[dict[str, Any]]:
        odf = odf or self.odf_columnar
        model = self.slip if convention in (None, self.convention) else SlipSystemModel(
            pyramids_tau_ratio=self.slip.pyramids_tau_ratio,
            pyramidal_tilt_deg=self.slip.pyramidal_tilt_deg,
            convention=convention)
        rows = model.strength_curve(odf, self.angles)
        sperp = model.uniaxial_strength(odf, load_direction(90.0))
        for row in rows:
            row["strength_over_perp"] = row["strength_Pa"] / sperp
        return rows

    def surface_curve(self, surface: HillTI | None = None) -> list[dict[str, Any]]:
        surface = surface or self.surface
        return surface.strength_curve(self.angles)

    def comparison_table(self, reference_convention: str = "mean") -> list[dict[str, Any]]:
        """Per-angle continuum-vs-slip comparison with relative errors."""
        ref = {r["theta_deg"]: r for r in self.slip_curve(convention=reference_convention)}
        rows = []
        for row in self.surface_curve():
            th = row["theta_deg"]
            ref_row = ref[th]
            err = row["strength_Pa"] / ref_row["strength_Pa"] - 1.0
            rows.append({
                "theta_deg": th,
                "sigma_over_perp_continuum": row["strength_over_perp"],
                "sigma_over_perp_slip": ref_row["strength_over_perp"],
                "relative_error": float(err),
                "governing_mechanism": ref_row["governing"],
            })
        return rows

    # -- scalars ----------------------------------------------------------
    def sigma45_surface_over_perp(self) -> float:
        return self.surface.uniaxial_strength(45.0) / self.surface.sigma_perp

    def sigma45_slip_over_perp(self, convention: str = "mean") -> float:
        model = self.slip if convention == self.convention else SlipSystemModel(
            pyramids_tau_ratio=self.slip.pyramids_tau_ratio,
            pyramidal_tilt_deg=self.slip.pyramidal_tilt_deg, convention=convention)
        sperp = model.uniaxial_strength(self.odf_columnar, load_direction(90.0))
        return model.uniaxial_strength(self.odf_columnar, load_direction(45.0)) / sperp

    def isotropic_control(self) -> dict[str, Any]:
        surf_iso = self.surface_isotropic
        strengths = {th: surf_iso.uniaxial_strength(th) for th in self.angles}
        ref = strengths[90.0]
        worst = max(abs(v / ref - 1.0) for v in strengths.values())
        # Angular average of the *nonlinear* mean-Schmid factor is quadrature
        # limited on the cubic-orbit set, so the physical isotropy statement is
        # measured on the dense isotropic fabric instead.
        dense = self.odf_uniform_dense
        if dense is None:
            slip_variation = float("nan")
        else:
            s_perp = self.slip.uniaxial_strength(dense, load_direction(90.0))
            slip_variation = max(
                abs(self.slip.uniaxial_strength(dense, load_direction(th)) / s_perp - 1.0)
                for th in self.angles)
        return {
            "fabric_index_g": self.odf_uniform.anisotropy_index(),
            "strength_ratio": surf_iso.strength_ratio,
            "ratio_error": abs(surf_iso.strength_ratio - 1.0),
            "sigma_parallel_Pa": surf_iso.sigma_parallel,
            "sigma_perp_Pa": surf_iso.sigma_perp,
            "tau_transverse_Pa": surf_iso.tau_transverse,
            "von_mises_tau_Pa": surf_iso.sigma_perp / np.sqrt(3.0),
            "tau_ratio_error": abs(surf_iso.tau_transverse
                                   / (surf_iso.sigma_perp / np.sqrt(3.0)) - 1.0),
            "max_directional_strength_variation": float(worst),
            "max_directional_strength_variation_slip_dense": float(slip_variation),
            "dense_fabric_index_g": (None if dense is None
                                     else float(dense.anisotropy_index())),
            "strengths_by_angle_over_perp": {str(k): v / ref for k, v in strengths.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": {
                "sigma_perp_Pa": self.sigma_perp,
                "ratio_columnar_target": self.ratio_columnar,
                "literature_band": list(COLUMNAR_RATIO_RANGE),
                "slip_convention": self.convention,
                "pyramidal_over_basal_crss": float(self.slip.pyramids_tau_ratio),
                "pyramidal_tilt_deg": float(self.slip.pyramidal_tilt_deg),
            },
            "fabric": {
                "columnar": {
                    "anisotropy_index": self.odf_columnar.anisotropy_index(),
                    "fabric_tensor": self.odf_columnar.fabric_tensor().tolist(),
                    "n_samples": int(len(self.odf_columnar.directions)),
                    "eigenvalues": self.odf_columnar.eigenpairs()[0].tolist(),
                },
                "uniform": {
                    "anisotropy_index": self.odf_uniform.anisotropy_index(),
                    "fabric_tensor": self.odf_uniform.fabric_tensor().tolist(),
                    "n_samples": int(len(self.odf_uniform.directions)),
                    "eigenvalues": self.odf_uniform.eigenpairs()[0].tolist(),
                },
            },
            "continuum_surface": self.surface.to_dict(),
            "continuum_curve": self.surface_curve(),
            "slip_reference": {
                "ratio": self.slip.strength_ratio(self.odf_columnar),
                "tau_transverse_Pa": self.slip.transverse_shear_strength(
                    self.odf_columnar)["shear_strength_Pa"],
                "curve_mean": self.slip_curve(convention="mean"),
                "curve_max_sachs": self.slip_curve(convention="max"),
            },
            "comparison": self.comparison_table("mean"),
            "isotropic_control": self.isotropic_control(),
        }


def run_verification(*, sigma_perp: float = 1.0e6, ratio_columnar: float = 3.5,
                     convention: str = "mean",
                     angles: Sequence[float] | None = None,
                     ledger: FabricLedger | None = None) -> dict[str, Any]:
    """Run every criterion that does not require stepping the DEM engine.

    Parameters
    ----------
    sigma_perp
        Transverse uniaxial strength (Pa) that sets the absolute scale; the
        criteria are all ratios, so any positive value works.
    ratio_columnar
        Target column/transverse ratio on the ideal columnar fabric (3-5).
    convention
        Slip-reference averaging convention used as the 45 deg reference
        (``"mean"`` mean-field; ``"max"`` is the Sachs bound, reported alongside).
    angles
        Direction sweep for the curves/tables.
    ledger
        Fabric ledger; a fresh one with the two reference ODFs is created if
        omitted.

    Returns
    -------
    dict
        ``criteria`` holds one record per acceptance criterion with
        ``value``/``threshold``/``pass``; the rest is the evidence.
    """
    angles = tuple(angles) if angles is not None else DEFAULT_ANGLES
    odf_col = FabricODF.columnar(label="S2 columnar (reference)")
    odf_uni = FabricODF.uniform(label="uniform (control, exact 2nd moment)")
    odf_uni_dense = FabricODF.uniform_dense(label="uniform (dense quadrature)")

    if ledger is None:
        ledger = FabricLedger()
    if "odf-columnar-S2" not in ledger.sample_ids():
        ledger.record(odf_col, sample_id="odf-columnar-S2",
                      meta={"source": "ideal S2 fabric: c-axes in the horizontal plane",
                            "provenance_class": "idealised reference fabric"})
    if "odf-uniform" not in ledger.sample_ids():
        ledger.record(odf_uni, sample_id="odf-uniform",
                      meta={"source": "isotropic control (cubic-orbit sampling)"})

    surface = strength_surface_from_fabric(odf_col, sigma_perp=sigma_perp,
                                           ratio_columnar=ratio_columnar,
                                           convention=convention)
    surface_iso = strength_surface_from_fabric(odf_uni, sigma_perp=sigma_perp,
                                               ratio_columnar=ratio_columnar,
                                               convention=convention)
    # the slip reference is rescaled so that its transverse strength equals the
    # declared sigma_perp: absolute strengths are then comparable across models
    slip = calibrated_slip_reference(
        odf_col, ratio_columnar=ratio_columnar, convention=convention,
        tilt_deg=float(surface.metadata["pyramidal_tilt_deg"]), sigma_perp=sigma_perp)

    study = AnisotropyStudy(
        odf_columnar=odf_col, odf_uniform=odf_uni, sigma_perp=sigma_perp,
        ratio_columnar=ratio_columnar, convention=convention, slip=slip,
        surface=surface, surface_isotropic=surface_iso, angles=angles,
        odf_uniform_dense=odf_uni_dense)

    payload = study.to_dict()

    # ---- criterion (a): the three-direction ratio ----------------------
    ratio_continuum = float(surface.strength_ratio)
    ratio_slip = float(slip.strength_ratio(odf_col))
    lo, hi = COLUMNAR_RATIO_RANGE
    c_a = {
        "id": "a",
        "statement": "column/transverse uniaxial strength ratio in the literature band 3-5",
        "value": ratio_continuum,
        "value_slip_reference": ratio_slip,
        "threshold": [lo, hi],
        "pass": bool(lo <= ratio_continuum <= hi and lo <= ratio_slip <= hi),
    }

    # ---- criterion (b): the 45 deg prediction --------------------------
    s45_cont = study.sigma45_surface_over_perp()
    s45_slip_mean = study.sigma45_slip_over_perp("mean")
    s45_slip_max = study.sigma45_slip_over_perp("max")
    err45 = s45_cont / s45_slip_mean - 1.0
    c_b = {
        "id": "b",
        "statement": "45 deg uniaxial strength predicted by the continuum surface vs the "
                     "slip-system reference (mean-field convention)",
        "value_over_perp_continuum": float(s45_cont),
        "value_over_perp_reference": float(s45_slip_mean),
        "reference_bound_sachs": float(s45_slip_max),
        "relative_error": float(err45),
        "threshold": 0.10,
        "pass": bool(abs(err45) <= 0.10),
    }

    # ---- criterion (c): isotropic degeneration ------------------------
    iso = payload["isotropic_control"]
    c_c = {
        "id": "c",
        "statement": "uniform ODF degenerates the anisotropic model to isotropic: "
                     "strength ratio -> 1 and no directional strength variation",
        "value_ratio_error": float(iso["ratio_error"]),
        "value_directional_variation": float(
            iso["max_directional_strength_variation_slip_dense"]),
        "value_tau_vs_von_mises_error": float(iso["tau_ratio_error"]),
        "threshold": 0.05,
        "pass": bool(iso["ratio_error"] <= 0.05
                     and iso["max_directional_strength_variation"] <= 0.05),
    }

    # ---- criterion (d): the fabric ledger -----------------------------
    roundtrip = FabricODF.from_dict(ledger.query("odf-columnar-S2")["odf"])
    ledger_summary = ledger.summary()
    c_d = {
        "id": "d",
        "statement": "fabric ODF ledger can record / query / round-trip and carry the "
                     "fabric tensors",
        "n_records": len(ledger),
        "sample_ids": ledger.sample_ids(),
        "roundtrip_anisotropy_index_error": float(
            abs(roundtrip.anisotropy_index() - odf_col.anisotropy_index())),
        "threshold": 1e-12,
        "pass": bool(len(ledger) >= 2
                     and abs(roundtrip.anisotropy_index()
                             - odf_col.anisotropy_index()) <= 1e-12
                     and all("anisotropy_index" in row for row in ledger_summary)),
    }

    payload["criteria"] = [c_a, c_b, c_c, c_d]
    payload["ledger"] = ledger_summary
    payload["limitations"] = [
        "The continuum-surface *shape* (sigma_par/sigma_perp and tau/sigma_perp) is taken "
        "from the Schmid slip-system reference; only its quadratic interpolation form is "
        "independent of it. The 45 deg comparison therefore tests the quadratic "
        "interaction law, not the slip physics.",
        "The slip reference is a mean-field (uniform-stress per grain) approximation; it "
        "carries no grain-grain compatibility constraint, so it brackets rather than "
        "replaces a self-consistent/Taylor polycrystal solution. The Sachs bound is "
        "reported next to it.",
        "Both models are pressure-insensitive and rate-independent; sea ice is neither "
        "(confined pressure raises strength, strain rate sets the peak).",
        "The absolute strengths are anchored to the single input sigma_perp; the "
        "literature band enters only through the ratio.",
        "The DEM bonded lattice is redundancy-dominated (three orthogonal bond families), "
        "so it cannot reach the 3-5 macro ratio; it is used as a discrete negative "
        "control and as evidence for that limitation, not as the 45 deg reference.",
    ]
    return payload
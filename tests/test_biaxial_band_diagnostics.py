"""Tests for the CASE-BIAXIAL-SHEAR-BAND R2 diagnostics.

Regression guard for the estimator defect found in R2: the pre-R2 Radon
estimator min-max rescaled ``x`` and ``y`` *independently*, which is an
anisotropic map, so an elongated specimen reported every band inclination
squashed towards 45 deg.  On the case lattice (24 x 48 spacings, W/H = 0.54) a
60 deg band was reported as 43-44 deg.  These tests pin both sides of the fix:
the production estimator must be unbiased on the case lattice, the archived
pre-R2 behaviour must remain reproducible as a documented baseline, and the
band-local dilation estimator must recover a known isochoric/dilatant band.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.biaxial_band_diagnostics import (
    aspect_ratio_distortion_deg,
    band_local_dilation,
    band_mask,
    estimator_bias_sweep,
    legacy_radon_band_angle,
    synthetic_band_configuration,
)
from tensordem.biaxial_shear_band import (
    local_shear_strain,
    measure_band,
    radon_band_angle,
)

torch.set_num_threads(1)

SPACING = 1.5
JITTER = 0.03


def lattice(columns: int = 24, rows: int = 48, *, seed: int = 7) -> np.ndarray:
    row_spacing = SPACING * math.sqrt(3.0) / 2.0
    width = (columns - 1) * SPACING
    generator = torch.Generator().manual_seed(seed)
    points = []
    for row in range(rows):
        offset = 0.5 * SPACING if row % 2 else 0.0
        for column in range(columns):
            x = -0.5 * width + column * SPACING + offset
            if abs(x) > 0.5 * width + 1e-9:
                continue
            points.append((x, -0.5 * (rows - 1) * row_spacing + row * row_spacing))
    pos = torch.tensor(points, dtype=torch.float64)
    return (pos + (JITTER * SPACING) * torch.randn(pos.shape, dtype=torch.float64,
                                                   generator=generator)).numpy()


def case_window(pos: np.ndarray) -> tuple[float, float, float, float]:
    margin = 2.0 * 2.0 * 0.75            # wall_margin_diameters * diameter
    return (float(pos[:, 0].min()) + margin, float(pos[:, 0].max()) - margin,
            float(pos[:, 1].min()) + margin, float(pos[:, 1].max()) - margin)


@pytest.mark.parametrize("angle", [30.0, 45.0, 55.0, 60.0, 70.0])
def test_production_estimator_is_unbiased_on_the_case_lattice(angle):
    """R2 regression: the reported angle must be the physical angle."""
    pos = lattice()
    window = case_window(pos)
    moved = synthetic_band_configuration(pos, angle_deg=angle, slip=2.45,
                                        band_width=3.0 * SPACING)
    out = measure_band(pos, moved, bounds=window)
    assert abs(out["folded_angle_deg"] - angle) <= 1.0
    assert out["z_score"] > 5.0
    assert out["binning"] == "equal_count"


def test_pre_r2_estimator_bias_is_reproducible_and_analytic():
    """The archived 44.5 deg reading is explained by tan(w) = (W/H) tan(true)."""
    pos = lattice()
    window = case_window(pos)
    inside = ((pos[:, 0] >= window[0]) & (pos[:, 0] <= window[1])
              & (pos[:, 1] >= window[2]) & (pos[:, 1] <= window[3]))
    window_points = pos[inside]
    width = float(np.ptp(window_points[:, 0]))
    height = float(np.ptp(window_points[:, 1]))
    moved = synthetic_band_configuration(pos, angle_deg=60.0, slip=2.45,
                                        band_width=3.0 * SPACING)
    gamma = local_shear_strain(window_points, moved[inside], k=8)
    legacy = legacy_radon_band_angle(window_points[:, 0], window_points[:, 1], gamma)
    measured = float(np.asarray(legacy["folded_angle_deg"]).reshape(-1)[0])
    assert measured == pytest.approx(44.0, abs=2.0)
    predicted = float(np.asarray(aspect_ratio_distortion_deg(
        60.0, width=width, height=height)).reshape(-1)[0])
    assert measured == pytest.approx(predicted, abs=2.0)
    assert predicted == pytest.approx(44.0, abs=2.0)


def test_aspect_ratio_distortion_identity():
    assert float(aspect_ratio_distortion_deg(45.0, width=2.0, height=1.0)[0]) == pytest.approx(
        math.degrees(math.atan(2.0)), abs=1e-9)
    # a square point set is not distorted at all
    assert float(aspect_ratio_distortion_deg(60.0, width=1.0, height=1.0)[0]) == pytest.approx(
        60.0, abs=1e-9)


def test_bias_sweep_reports_before_and_after():
    pos = lattice()
    window = case_window(pos)
    sweep = estimator_bias_sweep(pos, angles_deg=[45.0, 60.0, 70.0],
                                 measurement_window=window)
    assert sweep["max_abs_error_deg"] <= 1.0
    assert sweep["legacy_max_abs_error_deg"] >= 10.0
    assert sweep["aspect_ratio"] < 0.7


def test_band_mask_selects_a_strip():
    pos = lattice()
    mask = band_mask(pos, angle_deg=60.0, half_width=2.0)
    tangent = np.array([math.cos(math.radians(60.0)), math.sin(math.radians(60.0))])
    normal = np.array([-tangent[1], tangent[0]])
    distance = np.abs((pos - pos.mean(axis=0)) @ normal)
    assert mask.sum() > 30
    assert np.all(distance[mask] <= 2.0 + 1e-9)
    assert np.all(distance[~mask] > 2.0)


def _synthetic_history(angle: float, *, opening_per_step: float,
                       slip_per_step: float, steps: int = 6,
                       d_strain: float = 2.0e-3, width: float = 9.0) -> list[dict]:
    pos = lattice(rows=48)
    return [{"strain": i * d_strain,
             "positions": synthetic_band_configuration(
                 pos, angle_deg=angle, slip=i * slip_per_step, band_width=width,
                 volumetric_strain=i * opening_per_step / width)}
            for i in range(steps)]


def test_band_local_dilation_detects_isochoric_band():
    history = _synthetic_history(60.0, opening_per_step=0.0, slip_per_step=0.20)
    out = band_local_dilation(history, angle_deg=60.0, half_width=6.0)
    assert abs(out["psi_band_deg"]) <= 5.0
    assert abs(out["psi_background_deg"]) <= 5.0


def test_band_local_dilation_detects_dilatant_band():
    d_strain = 2.0e-3
    # ``opening_per_step`` is the *total* across-band opening; the local
    # volumetric strain increment inside a band of width 9 m is opening / 9, so
    # this injects d eps_v / d eps_a = 0.6  ->  psi = 36.9 deg.
    history = _synthetic_history(60.0, opening_per_step=0.6 * d_strain * 9.0,
                                slip_per_step=0.0)
    out = band_local_dilation(history, angle_deg=60.0, half_width=6.0)
    injected = math.degrees(math.asin(0.6))
    ratio = out["psi_band_deg"] / injected
    print(f"psi_band={out['psi_band_deg']:.1f} deg vs injected {injected:.1f} deg "
          f"(ratio {ratio:.2f})")
    # KNOWN LIMITATION (reported in docs/CASE_BIAXIAL_SHEAR_BAND_R2.md): the k=8
    # kNN fit spans ~2 particle spacings inside a finite-width band, so the
    # recovered magnitude is attenuated (measured ~0.44 of the injected value).
    # The test pins sign and order of magnitude, not the absolute value.
    assert 0.2 < ratio < 0.9
    assert out["psi_band_deg"] > out["psi_background_deg"] + 10.0


def test_band_local_dilation_requires_snapshots():
    from tensordem.biaxial_shear_band import BandMeasurementError
    with pytest.raises(BandMeasurementError):
        band_local_dilation([{"strain": 0.0, "positions": np.zeros((10, 2))}],
                            angle_deg=60.0, half_width=2.0)


def test_record_snapshots_is_opt_in():
    from tensordem.biaxial_shear_band import run_biaxial_shear_band
    common = dict(confining_stress=100.0e3, columns=6, rows=10, strain_target=0.004,
                  consolidation_steps=30, snapshot_strain=0.002,
                  material_overrides={"freeze_bond_fraction": 0.35})
    quiet = run_biaxial_shear_band(**common)
    assert quiet["snapshots"] == []
    loud = run_biaxial_shear_band(**common, record_snapshots=True)
    assert len(loud["snapshots"]) >= 2
    assert loud["snapshots"][0]["positions"].shape == (
        quiet["specimen"]["particles"], 2)
    assert quiet["band"]["angle_deg"] == pytest.approx(loud["band"]["angle_deg"], nan_ok=True)
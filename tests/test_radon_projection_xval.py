"""Cross-validation of the Radon-projection band measurement (known angles).

Anchor ``XVAL-DEM-RADON-PROJECTION-MEASUREMENT``: the production band-inclination
operator is a Radon/projection variance focus on the local shear-strain field
(:func:`tensordem.biaxial_shear_band.measure_band`).  Before its DEM output can
be compared with the 60 deg dense-sand anchor, the *instrument* has to be
validated against bands of exactly known inclination, and its operating envelope
(how much incremental-displacement noise it survives) has to be stated.

The heavy factor design (angles x lattices x widths x noise) is run on the
cluster by ``benchmarks/biaxial_shear_band/run_radon_xval_measurement.py``; the
tests here pin the fast, decisive invariants: recovery of a known angle on an
elongated lattice, the documented bias of the pre-fix baseline, fold
equivalence, and the correctness of the shard split / summary logic that the
cluster run depends on.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from tensordem.biaxial_band_diagnostics import (
    legacy_radon_band_angle,
    synthetic_band_configuration,
)
from tensordem.biaxial_shear_band import (
    BandMeasurementError,
    fold_angle_deg,
    local_shear_strain,
    measure_band,
    radon_band_angle,
)

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "benchmarks", "biaxial_shear_band")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

SPACING = 1.5          # 2 x particle radius of the case cell
SLIP = 0.02            # metres, late-window cumulative displacement


def _lattice(columns: int = 16, rows: int = 32, *, seed: int = 7) -> np.ndarray:
    """Jittered triangular disc lattice, aspect 0.5 (elongated like the case)."""
    row_spacing = SPACING * math.sqrt(3.0) / 2.0
    rng = np.random.default_rng(seed)
    pts = []
    for j in range(rows):
        for i in range(columns):
            x = i * SPACING + (0.5 * SPACING if j % 2 else 0.0)
            pts.append((x, j * row_spacing))
    pts = np.asarray(pts, dtype=float)
    pts += rng.uniform(-0.03, 0.03, pts.shape) * SPACING
    return pts


def _band_snapshot(points: np.ndarray, angle_deg: float, *,
                   width_d: float = 1.5,
                   slip: float = SLIP) -> np.ndarray:
    return synthetic_band_configuration(points, angle_deg=angle_deg, slip=slip,
                                        band_width=width_d * SPACING,
                                        background_axial_strain=0.02)


@pytest.mark.parametrize("angle", [30.0, 45.0, 60.0, 75.0])
def test_radon_projection_recovers_known_angle_on_elongated_lattice(angle):
    """Production operator on a 16 x 32 (aspect 0.5) lattice, noise-free."""
    ref = _lattice()
    now = _band_snapshot(ref, angle)
    got = float(measure_band(ref, now)["folded_angle_deg"])
    true = fold_angle_deg(angle)
    assert abs(got - true) <= 0.15 * true, (got, true)
    # numerical fidelity is in fact much tighter than the 15 % acceptance band
    assert abs(got - true) <= 2.0, (got, true)


def test_radon_projection_is_fold_invariant():
    """A band at 120 deg and one at 60 deg share the folded inclination."""
    ref = _lattice()
    low = float(measure_band(ref, _band_snapshot(ref, 60.0))["folded_angle_deg"])
    high = float(measure_band(ref, _band_snapshot(ref, 120.0))["folded_angle_deg"])
    assert abs(low - high) <= 1.0
    assert abs(low - 60.0) <= 0.15 * 60.0


def test_pre_fix_per_axis_rescaling_is_biased_on_elongated_lattice():
    """The pre-R2 estimator maps 60 deg to ~45 deg: the R1 FAIL was an artefact.

    This is the baseline the R2 fix replaced; the audit keeps it alive so the
    before/after contrast stays reproducible.  The bias must be **outside** the
    15 % acceptance band, otherwise the fix would have been cosmetic.
    """
    ref = _lattice()
    now = _band_snapshot(ref, 60.0)
    gamma = local_shear_strain(ref, now)
    legacy = float(legacy_radon_band_angle(ref[:, 0], ref[:, 1], gamma)
                   ["folded_angle_deg"])
    production = float(radon_band_angle(ref[:, 0], ref[:, 1], gamma)
                       ["folded_angle_deg"])
    assert abs(legacy - 60.0) > 0.15 * 60.0, legacy
    assert abs(production - 60.0) <= 0.15 * 60.0, production


@pytest.mark.parametrize("angle", [45.0, 60.0])
def test_bare_projection_chain_matches_measure_band_and_is_deterministic(angle):
    """`radon_band_angle` + `local_shear_strain` == `measure_band`; repeats identical."""
    ref = _lattice()
    now = _band_snapshot(ref, angle)
    gamma = local_shear_strain(ref, now)
    bare = float(radon_band_angle(ref[:, 0], ref[:, 1], gamma)["folded_angle_deg"])
    wrapped = float(measure_band(ref, now)["folded_angle_deg"])
    again = float(radon_band_angle(ref[:, 0], ref[:, 1],
                                   local_shear_strain(ref, now))["folded_angle_deg"])
    assert bare == pytest.approx(wrapped, abs=1e-12)
    assert bare == pytest.approx(again, abs=1e-12)


def test_measurement_rejects_degenerate_and_tiny_inputs():
    ref = _lattice()
    with pytest.raises(BandMeasurementError):
        measure_band(ref, ref)                      # zero displacement field
    with pytest.raises(BandMeasurementError):
        measure_band(ref[:20], ref[:20] + 1e-3)     # too few particles


def test_cluster_shard_split_covers_design_once():
    """The cluster split must be an exact partition of the factor design."""
    from run_radon_xval_measurement import build_design

    cells = build_design()
    key = lambda c: (c["lattice"], c["width_d"], c["alpha_true_deg"],
                     c["noise_fraction"], c["repeat"])
    keys = [key(c) for c in cells]
    assert len(keys) == len(set(keys)), "design contains duplicate cells"
    shards = 7                                       # small prime: catches off-by-one
    parts = [[c for i, c in enumerate(cells) if i % shards == s]
             for s in range(shards)]
    assert sum(len(p) for p in parts) == len(cells)
    assert set(k for p in parts for k in map(key, p)) == set(keys)


def test_summary_and_verdict_logic():
    """Fabricated rows: the envelope scan and the pass rule must be mechanical."""
    from run_radon_xval_measurement import OPERATING_NOISE, summarise, verdict

    def row(noise, error, true_angle=60.0, name="radon_production",
            lattice="24x48", width=1.5):
        return {"lattice": lattice, "width_d": width, "noise_fraction": noise,
                "alpha_true_deg": true_angle,
                "alpha_true_folded_deg": true_angle, "repeat": 0,
                name: true_angle + error, name + "_error_deg": error,
                "radon_legacy_peraxis": true_angle - 15.0,
                "radon_legacy_peraxis_error_deg": -15.0}

    clean = [row(0.0, 0.0), row(0.005, 1.0), row(0.01, 2.0), row(0.02, 20.0)]
    s = summarise(clean)
    assert s["radon_production"]["noise_envelope_fraction"] == pytest.approx(
        OPERATING_NOISE)
    assert verdict(s)["pass_precision_15pct_within_envelope"] is True
    assert verdict(s)["pass"] is True

    broken = [row(0.0, 0.0), row(0.005, 30.0), row(0.01, 40.0)]
    s2 = summarise(broken)
    assert s2["radon_production"]["noise_envelope_fraction"] == pytest.approx(0.0)
    assert verdict(s2)["pass_precision_15pct_within_envelope"] is False
    assert verdict(s2)["pass"] is False
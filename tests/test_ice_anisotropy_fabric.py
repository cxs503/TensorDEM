"""Tests for the c-axis ODF / fabric tensor ledger (ice_anisotropy.fabric)."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_anisotropy.fabric import (
    FabricLedger,
    FabricODF,
    cubic_orbit_directions,
    fibonacci_sphere,
)

ISOTROPIC_4TH = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for m in range(3):
                ISOTROPIC_4TH[i, j, k, m] = (
                    (i == j) * (k == m) + (i == k) * (j == m) + (i == m) * (j == k)) / 15.0


def test_fibonacci_sphere_is_unit_and_deterministic():
    a = fibonacci_sphere(64)
    b = fibonacci_sphere(64)
    assert a.shape == (64, 3)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12)
    assert np.array_equal(a, b)


def test_uniform_fabric_is_isotropic_exactly():
    """g == 0 to round-off: the isotropic control is an identity, not a sample mean."""
    odf = FabricODF.uniform()
    assert odf.anisotropy_index() == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(odf.fabric_tensor(), np.eye(3) / 3.0, atol=1e-12)
    assert odf.mean_c_axis() is None


def test_uniform_fabric_fourth_order_tensor_is_isotropic():
    odf = FabricODF.uniform()
    a4 = odf.fourth_order_tensor()
    assert np.allclose(a4, ISOTROPIC_4TH, atol=1.5e-2)  # cubic residual


def test_cubic_orbit_gives_48_distinct_images():
    orbit = cubic_orbit_directions([(0.3, 0.5, 0.8)])
    assert len(orbit) == 48
    assert np.allclose(np.linalg.norm(orbit, axis=1),
                       np.linalg.norm([0.3, 0.5, 0.8]))


def test_columnar_fabric_index_and_tensor():
    odf = FabricODF.columnar(n_azimuth=180)
    assert odf.anisotropy_index() == pytest.approx(1.0, abs=1e-12)
    a = odf.fabric_tensor()
    assert np.allclose(np.diag(a), [0.5, 0.5, 0.0], atol=1e-12)
    assert np.trace(a) == pytest.approx(1.0, abs=1e-12)
    # c-axes are perpendicular to the column axis
    assert np.allclose(np.abs(odf.directions @ np.array([0.0, 0.0, 1.0])), 0.0, atol=1e-12)


def test_columnar_fabric_about_x_axis():
    odf = FabricODF.columnar(axis=(1.0, 0.0, 0.0), n_azimuth=90)
    assert np.allclose(np.diag(odf.fabric_tensor()), [0.0, 0.5, 0.5], atol=1e-12)


def test_columnar_spread_blends_towards_isotropic():
    g = [FabricODF.columnar(spread_deg=spread, n_tilt=9, n_azimuth=72).anisotropy_index()
         for spread in (0.0, 15.0, 45.0, 90.0)]
    assert g[0] > g[1] > g[2] > g[3]
    assert g[-1] < 0.05           # (near) isotropic at a uniform tilt distribution
    assert g[0] == pytest.approx(1.0, abs=1e-12)


def test_misorientation_angles_of_columnar_fabric():
    odf = FabricODF.columnar(n_azimuth=180)
    assert np.allclose(odf.misorientation_from_axis_deg(), 90.0, atol=1e-9)


def test_measured_odf_carries_tensors_and_provenance():
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    odf = FabricODF(dirs, weights=[1.0, 1.0, 2.0], kind="measured",
                    label="thin section 1", provenance={"method": "Rigsby stage"})
    a = odf.fabric_tensor()
    assert a[0, 0] == pytest.approx(0.75) and a[1, 1] == pytest.approx(0.25)
    payload = odf.to_dict()
    assert payload["provenance"]["method"] == "Rigsby stage"
    back = FabricODF.from_dict(payload)
    assert np.allclose(back.directions, odf.directions)
    assert back.anisotropy_index() == pytest.approx(odf.anisotropy_index(), abs=1e-15)


def test_ledger_record_query_roundtrip_and_persistence(tmp_path):
    ledger = FabricLedger()
    col = FabricODF.columnar(label="S2")
    uni = FabricODF.uniform(label="iso")
    sid_col = ledger.record(col, sample_id="s2", meta={"core": "A"})
    ledger.record(uni, meta={"core": "B"})
    assert sid_col == "s2"
    assert len(ledger) == 2
    assert ledger.query("s2")["meta"]["core"] == "A"
    assert ledger.odf("s2").anisotropy_index() == pytest.approx(1.0, abs=1e-12)
    assert ledger.select("uniform") == ["odf-0002"]
    assert {row["sample_id"] for row in ledger.summary()} == {"s2", "odf-0002"}

    path = ledger.save(tmp_path / "ledger.json")
    restored = FabricLedger.load(path)
    assert restored.sample_ids() == ledger.sample_ids()
    assert restored.odf("s2").fabric_tensor() == pytest.approx(col.fabric_tensor())

    with pytest.raises(KeyError):
        ledger.record(uni, sample_id="s2")
    with pytest.raises(KeyError):
        ledger.query("missing")


def test_invalid_odf_inputs_rejected():
    with pytest.raises(ValueError):
        FabricODF(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        FabricODF([[1.0, 0.0, 0.0]], weights=[0.0, 1.0])
    with pytest.raises(ValueError):
        FabricODF([[1.0, 0.0, 0.0]], weights=[-1.0])
    with pytest.raises(ValueError):
        FabricODF.columnar(spread_deg=120.0)
"""IAPWS-anchored verification of the water viscosity temperature model.

The ``viscosity_temperature`` correlation in ``tensordem.ice_slurry_rheology.closures``
is a quadratic fit to log(μ) vs T.  This test anchors it against the IAPWS
industrial standard (IAPWS R12-2008) at 9 reference temperatures and verifies
that the relative error stays below 0.5 % across the full operating range
[0, 40] °C relevant to ice-slurry rheology.

Reference: IAPWS Release on the Viscosity of Ordinary Water Substance (2008).
"""
import math

import pytest

from tensordem.ice_slurry_rheology.closures import viscosity_temperature

# IAPWS reference values (Pa·s) for ordinary water at atmospheric pressure.
# Source: IAPWS R12-2008, Table 1 (interpolated).
_IAPWS_VISCOSITY = {
    0:  1.7870e-3,
    5:  1.5190e-3,
    10: 1.3070e-3,
    15: 1.1380e-3,
    20: 1.0020e-3,
    25: 0.8900e-3,
    30: 0.7980e-3,
    35: 0.7200e-3,
    40: 0.6530e-3,
}

_MAX_REL_ERR = 0.005  # 0.5 %


class TestIAPWSViscosityAnchor:
    """Pointwise IAPWS anchor at 9 temperatures."""

    @pytest.mark.parametrize("T,mu_ref", _IAPWS_VISCOSITY.items())
    def test_iapws_point(self, T, mu_ref):
        mu = viscosity_temperature(float(T))
        rel_err = abs(mu - mu_ref) / mu_ref
        assert rel_err < _MAX_REL_ERR, (
            f"T={T} °C: mu={mu:.6e}, IAPWS={mu_ref:.6e}, rel_err={rel_err*100:.3f}%"
        )

    def test_max_error_over_range(self):
        """The worst-case relative error across all anchor points."""
        errors = []
        for T, mu_ref in _IAPWS_VISCOSITY.items():
            mu = viscosity_temperature(float(T))
            errors.append(abs(mu - mu_ref) / mu_ref)
        max_err = max(errors)
        assert max_err < _MAX_REL_ERR, f"max rel_err = {max_err*100:.3f}%"

    def test_monotonic_decrease(self):
        """Viscosity must decrease monotonically with temperature."""
        temps = sorted(_IAPWS_VISCOSITY.keys())
        mus = [viscosity_temperature(float(T)) for T in temps]
        for i in range(len(mus) - 1):
            assert mus[i] > mus[i + 1], (
                f"Non-monotonic at T={temps[i]}→{temps[i+1]}: "
                f"{mus[i]:.6e} vs {mus[i+1]:.6e}"
            )

    def test_0c_anchor(self):
        """0 °C is the most critical point for ice-slurry work."""
        mu = viscosity_temperature(0.0)
        mu_ref = _IAPWS_VISCOSITY[0]
        rel_err = abs(mu - mu_ref) / mu_ref
        assert rel_err < 0.002, f"0 °C rel_err = {rel_err*100:.3f}%"  # tighter at 0 °C

    def test_20c_anchor(self):
        """20 °C is a common laboratory reference."""
        mu = viscosity_temperature(20.0)
        mu_ref = _IAPWS_VISCOSITY[20]
        rel_err = abs(mu - mu_ref) / mu_ref
        assert rel_err < 0.002, f"20 °C rel_err = {rel_err*100:.3f}%"

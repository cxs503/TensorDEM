"""Verification-debt closure: Hertz pair-contact restitution (m3499).

The ``contact_model="hertz"`` pair branch derives its normal dashpot from
``restitution_coefficient`` through the **linear Kelvin-Voigt** map
``zeta = |ln e| / sqrt(pi**2 + ln(e)**2)`` with an overlap-dependent damper
``gamma(d) = 2 zeta sqrt(k_h sqrt(d) m_eff)``.  Before this module that mapping
had no test and no document, and the merged Hertz document still claimed that
Hertz damping was the absolute ``c_n`` with restitution "not yet controllable".

Three things are pinned here:

1. **Implementation closure** - IceDEM3D reproduces the exact single-contact
   ODE given by its own law (independent RK4 reference, ``benchmarks/
   hertz_pair_restitution_verification.py``) to better than 0.5 %, so the
   engine is not the problem.
2. **Calibration honesty** - the realised restitution is *not* the declared one:
   the linear map under-damps a Hertzian impact.  The test pins the direction
   and the size of the bias, so a future fix cannot silently regress it.
3. **Documented limitation** - the platen branch keeps the absolute
   ``platen_damping`` and ignores ``restitution_coefficient`` entirely.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "benchmarks", ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hertz_pair_restitution_verification as H  # noqa: E402


class ReferenceIntegratorTests(unittest.TestCase):
    """The independent ODE reference must be trustworthy before it judges the engine."""

    def test_reference_energy_balance(self) -> None:
        zeta = H.zeta_from_restitution(0.5)
        res = H.reference_restitution(zeta, H.V_REL, steps=8000)
        self.assertLess(abs(res["energy_balance_rel"]), 5e-3)
        self.assertGreater(res["damper_work_J"], 0.0)

    def test_reference_converges_with_step_refinement(self) -> None:
        zeta = H.zeta_from_restitution(0.5)
        coarse = H.reference_restitution(zeta, H.V_REL, steps=2000)["restitution"]
        fine = H.reference_restitution(zeta, H.V_REL, steps=8000)["restitution"]
        self.assertLess(abs(fine - coarse), 5e-4)

    def test_restitution_is_speed_independent_by_construction(self) -> None:
        """gamma ~ d**(1/4) keeps the local damping ratio constant: e(v) = const."""
        zeta = H.zeta_from_restitution(0.5)
        slow = H.reference_restitution(zeta, 0.5, steps=4000)["restitution"]
        fast = H.reference_restitution(zeta, 2.0, steps=4000)["restitution"]
        self.assertLess(abs(fast - slow), 1e-6)


class EngineClosureTests(unittest.TestCase):
    """IceDEM3D versus the reference solution of its own contact law."""

    def test_engine_matches_reference_law(self) -> None:
        zeta = H.zeta_from_restitution(0.5)
        ref = H.reference_restitution(zeta, H.V_REL, steps=8000)["restitution"]
        engine = H.measure_pair(0.5, dt_scale=0.25)["restitution"]
        self.assertLess(abs(engine / ref - 1.0), 5e-3)

    def test_elastic_limit_is_recovered(self) -> None:
        engine = H.measure_pair(1.0, dt_scale=0.25)["restitution"]
        self.assertLess(abs(engine - 1.0), 2e-3)


class CalibrationBiasTests(unittest.TestCase):
    """The declared coefficient is not the realised restitution - pinned as debt."""

    def test_declared_0p5_under_damps_by_more_than_5_percent(self) -> None:
        engine = H.measure_pair(0.5, dt_scale=0.25)["restitution"]
        bias = engine / 0.5 - 1.0
        self.assertGreater(bias, 0.05)
        self.assertLess(bias, 0.30)

    def test_bias_shrinks_towards_the_elastic_limit(self) -> None:
        bias_high = H.measure_pair(0.9, dt_scale=0.25)["restitution"] / 0.9 - 1.0
        bias_low = H.measure_pair(0.5, dt_scale=0.25)["restitution"] / 0.5 - 1.0
        self.assertLess(abs(bias_high), abs(bias_low))

    def test_calibration_inverse_recovers_the_target(self) -> None:
        inv = H.calibration_inverse(0.5)
        self.assertTrue(math.isfinite(inv["zeta"]))
        realised = H.reference_restitution(inv["zeta"], H.V_REL, steps=4000)["restitution"]
        self.assertLess(abs(realised - 0.5), 2e-3)

    def test_zeta_correction_direction(self) -> None:
        """A Hertzian impact needs *more* zeta than the linear map prescribes.

        The d**(1/4) damper is clipped to zero at contact onset (its force
        exceeds the vanishing spring force) and again early in restitution, so
        part of the damping the linear theory assumes never acts.
        """
        zeta_used = H.zeta_from_restitution(0.5)
        zeta_req = H.calibration_inverse(0.5)["zeta"]
        self.assertGreater(zeta_req, zeta_used)
        self.assertLess(zeta_req, 2.0 * zeta_used)


class PlatenBranchLimitationTests(unittest.TestCase):
    """The platen branch keeps the absolute dashpot: restitution is ignored."""

    def test_platen_restitution_is_elastic_at_default_platen_damping(self) -> None:
        low = H.measure_platen(0.3)["restitution"]
        high = H.measure_platen(0.9)["restitution"]
        self.assertLess(abs(low - 1.0), 5e-3)
        self.assertLess(abs(high - 1.0), 5e-3)
        self.assertLess(abs(low - high), 2e-3)


if __name__ == "__main__":
    unittest.main()
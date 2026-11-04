"""Failure-mode guards for the pair-contact damping convention and API drift.

Both defects were found by a full-suite audit (mission m3441) and are silent by
construction, i.e. they do not raise:

1. ``DEM3DConfig.contact_damping_model`` defaults to ``"restitution"``, a model
   that derives the pair normal damping from ``restitution_coefficient`` and
   *ignores* ``contact_damping``.  The closed-loop harness
   :func:`tensordem.contact_restitution.measure_dem3d_impact` passed
   ``contact_damping=gamma_n`` without selecting the absolute convention, so it
   measured an undamped impact (``e_measured`` ~ 1.0000 for every ``gamma_n``)
   and four solver-side restitution checks failed with 9 %-67 % errors.

2. ``IceDEM3D.forces`` gained an ``update_history`` keyword that
   ``IceDEM3D.diagnostics`` forwards; the overrides in the moving-wedge and
   rudder-ice-load subclasses were not updated, so ``diagnostics()`` raised
   ``TypeError`` on those classes.
"""
import ast
import inspect
import math
import pathlib
import unittest

import torch

from tensordem.contact_restitution import (
    classic_restitution_ratio,
    measure_dem3d_impact,
)

RADIUS = 0.025
DENSITY = 917.0
K_N = 5_000.0
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tensordem"


class ContactDampingConventionGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def impact(self, gamma_n, **kwargs):
        return measure_dem3d_impact(gamma_n, k_n=K_N, radius=RADIUS,
                                    density=DENSITY, **kwargs)

    def test_zero_damping_impact_is_elastic(self):
        self.assertAlmostEqual(self.impact(0.0)["e_measured"], 1.0, places=3)

    def test_damping_is_actually_applied(self):
        # Before the fix every gamma_n returned e = 1.0000.
        weak = self.impact(1.0)["e_measured"]
        strong = self.impact(10.0)["e_measured"]
        self.assertLess(weak, 0.98, msg=f"e(gamma=1) = {weak:.4f}: damping ignored")
        self.assertLess(strong, weak - 0.05)
        self.assertGreater(strong, 0.2)

    def test_harness_matches_classic_weak_damping_limit(self):
        zeta = 0.058
        gamma = 2.0 * zeta * math.sqrt(K_N * MASS)
        measured = self.impact(gamma)["e_measured"]
        classic = classic_restitution_ratio(zeta)
        self.assertLess(abs(measured - classic) / measured, 0.03,
                        msg=f"measured {measured:.4f} vs classic {classic:.4f}")

    def test_absolute_convention_is_the_documented_pair_law(self):
        self.assertIn('contact_damping_model="absolute"',
                      inspect.getsource(measure_dem3d_impact),
                      "measure_dem3d_impact must select the absolute damping "
                      "convention, otherwise contact_damping is inert")


class SubclassForcesSignatureGuard(unittest.TestCase):
    """Every ``IceDEM3D`` subclass must accept the base ``forces`` keywords."""

    REQUIRED = ("update_fracture", "update_history")

    @staticmethod
    def _overrides():
        """Static scan (AST) of the package: no import side effects."""
        found = {}
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not any(isinstance(b, ast.Name) and b.id == "IceDEM3D"
                           for b in node.bases):
                    continue
                forces = next((n for n in node.body
                               if isinstance(n, ast.FunctionDef) and n.name == "forces"),
                              None)
                if forces is None:
                    continue
                names = {a.arg for a in forces.args.args + forces.args.kwonlyargs}
                found[f"{path.name}:{node.name}"] = (names, forces.args.kwarg is not None)
        return found

    def test_forces_overrides_accept_base_keywords(self):
        overrides = self._overrides()
        self.assertGreaterEqual(len(overrides), 3,
                                f"expected >=3 IceDEM3D subclasses, found {sorted(overrides)}")
        for label, (names, var_kw) in overrides.items():
            if var_kw:
                continue
            missing = [kw for kw in self.REQUIRED if kw not in names]
            self.assertFalse(missing, msg=f"{label}.forces is missing {missing}")

    def test_diagnostics_runs_on_the_fixed_subclasses(self):
        from tensordem.dem3d import DEM3DConfig
        from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D
        from tensordem.rudder_ice_load import RudderIceLoadDEM

        for cls in (MovingWedgeBowIceDEM3D, RudderIceLoadDEM):
            sim = cls(DEM3DConfig(nx=3, ny=2, nz=2))
            diag = sim.diagnostics()
            self.assertTrue(diag)
            self.assertTrue(all(math.isfinite(float(v)) for v in diag.values()
                                if isinstance(v, (int, float))))


if __name__ == "__main__":
    unittest.main()

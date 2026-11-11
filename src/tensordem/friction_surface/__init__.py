"""Friction-contact verification surface (node P152W1-N04).

Machine-adjudicable verification of TensorDEM's frictional contact capability,
in three anchors plus the two declared contact laws:

* ``laws``     — ``F_t = min(mu F_n, k_t delta_t) + c_t v_t`` and the rolling
                 resistance couple, anchored on closed forms (onset at
                 ``arctan(mu)`` / ``arctan(mu_r)``, cone saturation, dissipative
                 dashpot);
* ``repose``   — angle of repose: mu = 0 negative control, glass-bead
                 calibration of the single shape parameter, out-of-sample sand
                 prediction;
* ``shear``    — periodic planar shear cell, peak/residual strength vs the
                 Mohr-Coulomb criterion;
* ``packing``  — random close packing vs the 0.64 limit (two independent routes).

``study`` runs the anchors and adjudicates them against the pre-registered
thresholds in ``prereg/P152W1_N04_friction_contact_surface_prereg.json``.
"""
from . import benchmark, laws, kernel, packing, repose, shear  # noqa: F401

__all__ = ["benchmark", "laws", "kernel", "packing", "repose", "shear"]

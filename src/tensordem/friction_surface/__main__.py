"""Entry point: ``python -m tensordem.friction_surface``.

Runs the P152W1-N04 repose benchmark (anchor A negative control, anchor B
glass-bead calibration + out-of-sample sand prediction) and writes the
machine-adjudicable artifact.  Same CLI as
``python -m tensordem.friction_surface.benchmark``; see
:mod:`tensordem.friction_surface.benchmark` for the protocol and the rules.
"""
from .benchmark import main

if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
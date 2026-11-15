"""Rebuild ``runs/provenance_archive/`` from the two cited papers (needs net).

Run from the repository root::

    python -m benchmarks.plane_shear_mu_i.build_reference_archive [pdf_cache_dir]

The committed artefacts (``sources.json``, per-source excerpt txt and decimal
inventory) are all this build produces; everything else in round 22 reads them
offline.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_archive import (  # noqa: E402
    build_archive, parse_dacruz, parse_gdr_midi, fit_jop_to_law, archived_law,
    ATTRIBUTED_TRIPLES,
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cache = argv[0] if argv else None
    m = build_archive(pdf_cache=cache)
    print("\nparsed from archived excerpts")
    dc = parse_dacruz(m)
    for k, v in dc.items():
        if k != "locators":
            print("  daCruz %-24s %s" % (k, v))
    for k, v in parse_gdr_midi(m).items():
        print("  GDR    %-24s %s" % (k, v))
    law = archived_law(m)
    print("\n  archived law mu(0.01)=%.4f  mu(0.1)=%.4f  phi(0.01)=%.4f"
          % (law.mu(0.01), law.mu(0.1), law.phi(0.01)))
    print("\nattributed-triple numbers found in the texts")
    for src in m["sources"]:
        print("  %-14s %s" % (src["key"], src["attributed_triple_numbers_present"]))
    print("\nJop-form best fit to the archived linear law (I=1e-3..0.2)")
    print("  ", {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in fit_jop_to_law(law).items()
                  if k in ("mu_s", "mu_2", "I_0", "max_abs_rel_err")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

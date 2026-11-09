"""Mixed first-year / multi-year (FY/MY) ice fields: material state, FSD and
resistance.

Public surface
--------------
``aging``      age -> sea-ice material state (brine volume, flexural and
               compressive strength, elastic modulus, fracture toughness,
               density).  Every coefficient carries an evidence grade and the
               ledger is hashed into every payload.
``field``      bimodal FY/MY floe-thickness distribution (FSD): Monte-Carlo
               sampler *and* a deterministic quadrature of the same field, so
               mixed-field expectations carry no sampling noise and the sampler
               can be verified against the quadrature.
``resistance`` mixed-field resistance: per-floe engineering anchors
               (Lindqvist, Riska) integrated over the FSD, one-at-a-time
               sensitivity of the field parameters, and the DEM
               strength-sensitivity fit used to locate where the bonded-particle
               section model agrees with the engineering anchors.
``campaign``   the campaign: MY-fraction sweep 0-80 %, FSD sensitivity
               envelope, DEM 2^3 strength/E factorial, reference-case check,
               machine-checkable verdicts, artifact and figures.

Evidence grades used here are those of :mod:`tensordem.level_ice.formula`:
``DERIVED`` (re-derived from a stated assumption), ``CONVENTION`` (pinned
order-one choice), ``CITED`` (a published correlation quoted from literature
that cannot be re-read offline and is therefore *not* independent evidence), and
``WITHHELD`` (no literature agreement is claimed).

Absolute ship-scale error is **not** improved by this package: it inherits the
anchor's documented ~28 % MAPE against the MT Uikku model tests.  What this
package adds is a well-posed *relative* question -- how much does resistance
change when the ice field is partly multi-year, and how sensitive is that change
to the ice-thickness distribution -- with every number reproducible.
"""
from .aging import (COEFFICIENT_LEDGER, IceAgeState, aging_ledger_sha256,
                    brine_volume_fraction, compressive_strength_pa,
                    elastic_modulus_pa, first_year_state, fracture_toughness_pa_sqrt_m,
                    material_state)
from .field import (FloeDistribution, MixedFieldSpec, FieldSample,
                    quadrature_field, sample_field)
from .resistance import (MixedResistance, mixed_resistance,
                         resistance_sensitivity)

__all__ = [
    "COEFFICIENT_LEDGER", "IceAgeState", "aging_ledger_sha256",
    "brine_volume_fraction", "compressive_strength_pa", "elastic_modulus_pa",
    "first_year_state", "fracture_toughness_pa_sqrt_m", "material_state",
    "FloeDistribution", "MixedFieldSpec", "FieldSample",
    "quadrature_field", "sample_field",
    "MixedResistance", "mixed_resistance", "resistance_sensitivity",
]
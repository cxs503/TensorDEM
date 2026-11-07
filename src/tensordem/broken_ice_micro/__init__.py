"""Broken-ice microstructure: FSD evolution, arching/clogging and force chains.

This sub-package assembles the *full-spectrum* validation chain for fragmented
ice covers (campaign IN34-W3-N01).  Three physical layers, one shared
vocabulary:

``fragmentation``
    A bonded-triangular-lattice fragmentation engine.  A level ice plate of
    elastic bonds with Weibull-distributed failure strains is crushed between a
    fixed and a moving platen; the surviving bond network is decomposed into
    connected components, which are the fragments (floes).  The engine returns
    the fragment census and its dispersion/conservation diagnostics.

``fsd``
    Floe-size-distribution (FSD) estimation.  Number distribution
    ``n(D) ~ D**-alpha``; the exponent is estimated by the Clauset-Shalizi-Newman
    maximum-likelihood estimator with an objective (KS-minimising) lower cutoff,
    cross-checked by a log-binned least-squares fit reported with ``R**2`` and a
    Kolmogorov-Smirnov distance.

``arching``
    Janssen silo/hopper mechanics: the analytic 2D Janssen stress profile
    ``sigma_zz(z) = rho_b g lambda (1 - exp(-z/lambda))``, ``lambda = W/(2 K mu_w)``,
    a fit of ``(K, mu_w)`` to the DEM wall-pressure profile, and a gravity-driven
    channel-discharge experiment on the :mod:`tensordem.granular2d` engine that
    measures the clogging probability ``P_clog(W/h)`` and the critical width
    ratio ``(W/h)_crit``.

``chain``
    Contact-network statistics of the same broken-ice medium: contact-force
    distribution ``P(f)`` with an exponential tail fit, coordination number
    ``Z`` and porosity ``n``.

``study``
    The driver that runs the case matrix and emits the machine-checkable
    criteria ledger (``artifacts/broken_ice_micro_study/study.json``).

Units are SI throughout (m, kg, s, N).  Two-dimensional plate quantities carry
per-unit-thickness stress (N/m).
"""
from __future__ import annotations

from .fragmentation import (
    FragmentationConfig,
    FragmentCensus,
    QuasiStaticFragmentation,
    fragment_census,
    triangular_lattice,
)
from .fsd import (
    FSD_FIT_ALPHA_RANGE,
    PowerLawFit,
    fit_power_law,
    log_binned_histogram,
)
from .arching import (
    JanssenFit,
    JanssenProfile,
    SiloConfig,
    SiloRun,
    clogging_curve,
    critical_width_ratio,
    fit_janssen,
    janssen_profile,
    run_silo,
)
from .chain import (
    ForceChainFit,
    MeasuredPowerLawFSD,
    contact_network_stats,
    fit_exponential_tail,
    three_dimensional_rubble,
)

__all__ = [
    "QuasiStaticFragmentation",
    "FragmentationConfig",
    "FragmentCensus",
    "fragment_census",
    "triangular_lattice",
    "FSD_FIT_ALPHA_RANGE",
    "PowerLawFit",
    "fit_power_law",
    "log_binned_histogram",
    "JanssenFit",
    "JanssenProfile",
    "SiloConfig",
    "SiloRun",
    "clogging_curve",
    "critical_width_ratio",
    "fit_janssen",
    "janssen_profile",
    "run_silo",
    "ForceChainFit",
    "MeasuredPowerLawFSD",
    "contact_network_stats",
    "fit_exponential_tail",
    "three_dimensional_rubble",
]
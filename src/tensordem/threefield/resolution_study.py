"""Resolution ladder for the D3Q19 shear-wave transport check (m2949 honest FAIL).

Why this module exists
----------------------
``lbm_transport.py`` (m2949) measured the momentum diffusivity implied by
``tau0`` by fitting the decay of ``u_x = A cos(k y)`` in a periodic box and
reported the rate against the analytic Navier-Stokes value ``nu k^2``,
``nu = (tau0 - 1/2)/3``.  Five of six relaxation times landed inside a 3 % band,
but ``tau0 = 1.4`` (``nu = 0.3``) came out at **-3.18 %** and was booked as an
*honest FAIL* with the stated remediation "use a finer lattice when
``tau0 > 1``".  That remediation was never executed and the FAIL was never
either reproduced at a second resolution or closed -- this module closes it,
without touching the solver.

What is measured
----------------
1. :func:`fit_decay_rate` re-fits a decay curve from the population history that
   :func:`tensordem.threefield.lbm_transport.shear_wave_decay` already returns
   (``ShearWaveResult.history`` / ``.times``).  The window is a free parameter,
   so the *same* trajectory can be booked over a 120-step and a 200-step window;
   comparing the two isolates residual start-up (relaxation) transient from the
   asymptotic diffusive rate instead of assuming it away.
2. :func:`decay_ladder` runs the shipped measurement at ``N = 16/32/64`` for a
   set of relaxation times, projecting onto the same first mode (``k = 2 pi/N``),
   and reports ``nu_measured`` and its error against the analytic rate.  The
   cell wavenumber ``k h = 2 pi/N`` shrinks with the lattice, so the ladder is a
   *truncation-error* ladder: each level solves a slightly different physical
   problem, and the dimensionless error is the quantity that must collapse.
3. :func:`richardson_extrapolate` fits the observed error order ``p`` from three
   lattice levels (``error ~ (k h)^p``) and extrapolates ``k h -> 0``.  The
   extrapolated value is the resolution-free ("truth") estimate of the decay
   rate, i.e. the number the honest FAIL should have been compared against.
4. :func:`applicability_envelope` turns the ladder into the two engineering
   statements the campaign asks for: the largest ``tau0`` that keeps the rate
   error inside a stated band at a given lattice, and the largest ``k h`` that
   does the same.  Both come from measured points (never from a fitted curve
   alone); the fit is reported next to them so the two can be cross-checked.

Conventions: lattice units, ``dx = dt = 1``, ``c_s^2 = 1/3``, incompressible
low-Mach.  ``nu`` here is the *measured* momentum diffusivity of the BGK kernel,
i.e. a property of the solver, not of a closure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .lbm_transport import shear_wave_decay, shear_wave_rate

__all__ = [
    "DecaySample",
    "LadderPoint",
    "RichardsonResult",
    "fit_decay_rate",
    "decay_ladder",
    "richardson_extrapolate",
    "applicability_envelope",
]


# ---------------------------------------------------------------------------
# fitting a decay curve over an explicit window
# ---------------------------------------------------------------------------
def fit_decay_rate(
    times: Sequence[float], amplitudes: Sequence[float], *, start: int, stop: int
) -> float:
    """Least-squares slope of ``ln A`` against ``t`` over ``[start, stop)``.

    ``start`` / ``stop`` are *time values* in steps (``start`` inclusive,
    ``stop`` exclusive), not list indices: the population history of
    :func:`~tensordem.threefield.lbm_transport.shear_wave_decay` starts at the
    first step after ``fit_from``, so index arithmetic silently shifts the
    window by that offset (it did, in the first draft of this module).  The
    normal equations are written out explicitly for the same reason as in
    :mod:`lbm_transport`: ``torch.linalg.lstsq`` returned a slope two orders of
    magnitude too large on this well-conditioned one-parameter problem.  Only
    strictly positive amplitudes enter the log, so a mode that has decayed into
    the rounding floor cannot drag the slope.
    """
    if stop <= start:
        raise ValueError("stop must exceed start")
    if len(times) != len(amplitudes):
        raise ValueError("times and amplitudes must have the same length")
    picked = [
        (float(t), float(a))
        for t, a in zip(times, amplitudes)
        if start <= float(t) < stop and float(a) > 0.0
    ]
    if len(picked) < 3:
        raise ValueError("fewer than three strictly positive samples in the window")
    n = float(len(picked))
    t_bar = sum(t for t, _ in picked) / n
    y_bar = sum(math.log(a) for _, a in picked) / n
    num = sum((t - t_bar) * (math.log(a) - y_bar) for t, a in picked)
    den = sum((t - t_bar) ** 2 for t, _ in picked)
    if den <= 0.0:
        raise ValueError("degenerate time window")
    return -num / den


# ---------------------------------------------------------------------------
# one measurement book
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecaySample:
    """Decay rate of one shear mode fitted over one time window."""

    tau0: float
    size: int
    mode: int
    start: int
    stop: int
    rate: float
    nu_nominal: float
    rate_nominal: float
    rate_discrete: float
    amplitude_final: float
    #: nominal window length that was requested (``stop - start`` may be one
    #: sample shorter: the shipped history starts at ``fit_from + 1``)
    window: int = 0

    @property
    def k(self) -> float:
        return 2.0 * math.pi * self.mode / self.size

    @property
    def kh(self) -> float:
        """Cell wavenumber ``k h`` with ``h = 1``: the resolution parameter."""
        return self.k

    @property
    def steps(self) -> int:
        return self.stop - self.start

    @property
    def nu_measured(self) -> float:
        return self.rate / (self.k * self.k)

    @property
    def error_nominal(self) -> float:
        """``nu_measured / nu_nominal - 1`` against the analytic ``nu = (tau0-1/2)/3``."""
        return self.nu_measured / self.nu_nominal - 1.0

    @property
    def error_discrete(self) -> float:
        """Error against ``nu * 2 (1 - cos k)`` (discrete Laplacian reference)."""
        return self.rate / self.rate_discrete - 1.0


@dataclass(frozen=True)
class LadderPoint:
    """All windows of one ``(tau0, N, mode)`` trajectory."""

    tau0: float
    size: int
    mode: int
    samples: tuple[DecaySample, ...] = ()

    def window(self, steps: int) -> DecaySample:
        for sample in self.samples:
            if sample.window == steps:
                return sample
        raise KeyError(f"no {steps}-step window in this point")

    @property
    def window_sensitivity(self) -> float:
        """Largest spread of ``nu_measured`` over the booked windows (relative).

        A value much larger than the fit noise means the start-up transient has
        not left the short window; it is the direct measure of how much the
        published number depends on the arbitrary fit window.
        """
        values = [sample.nu_measured for sample in self.samples]
        mean = sum(values) / len(values)
        return max(abs(value - mean) for value in values) / mean


def _book(
    times: Sequence[float],
    history: Sequence[float],
    *,
    tau0: float,
    size: int,
    mode: int,
    start: int,
    stop: int,
    window: int,
    nominal: float,
    rate_nominal: float,
    rate_discrete: float,
) -> DecaySample:
    """Fit one window and report the window that the trajectory actually covers."""
    available = [float(t) for t in times if start <= float(t) < stop]
    if len(available) < 20:
        raise ValueError(
            f"window [{start}, {stop}) holds only {len(available)} samples "
            f"(requested {window}); the trajectory decayed out of range"
        )
    rate = fit_decay_rate(times, history, start=start, stop=stop)
    picked = [
        (float(t), float(a)) for t, a in zip(times, history) if start <= float(t) < stop
    ]
    return DecaySample(
        tau0=tau0,
        size=size,
        mode=mode,
        start=int(available[0]),
        stop=int(available[-1]) + 1,
        rate=rate,
        nu_nominal=nominal,
        rate_nominal=rate_nominal,
        rate_discrete=rate_discrete,
        amplitude_final=float(picked[-1][1]),
        window=window,
    )


def decay_ladder(
    tau0s: Iterable[float],
    sizes: Iterable[int],
    *,
    modes: Iterable[int] = (1,),
    steps: int = 220,
    windows: Sequence[int] = (120, 200),
    start: int = 20,
) -> dict[tuple[float, int, int], LadderPoint]:
    """Run the shipped shear-wave measurement over a ``(tau0, N, mode)`` ladder.

    One trajectory is marched to ``steps`` steps and then *re-fitted* over each
    requested window length, so the window comparison is free: it re-uses the
    same population history and cannot be confounded by a different random
    start.  ``start`` skips the relaxation transient, exactly as the m2949
    measurement did, so only the window *length* varies between the windows.
    """
    long_window = max(windows)
    if start + long_window > steps:
        raise ValueError("steps must cover start + the longest window")
    out: dict[tuple[float, int, int], LadderPoint] = {}
    for tau0 in tau0s:
        for size in sizes:
            for mode in modes:
                result = shear_wave_decay(
                    tau0=float(tau0), size=int(size), mode=int(mode), steps=int(steps)
                )
                times = list(result.times)
                history = list(result.history)
                nominal = (float(tau0) - 0.5) / 3.0
                k = 2.0 * math.pi * int(mode) / int(size)
                rate_nominal, rate_discrete = shear_wave_rate(float(tau0), k)
                samples = tuple(
                    _book(times, history, tau0=float(tau0), size=int(size), mode=int(mode),
                          start=int(start), stop=int(start + window), window=int(window),
                          nominal=nominal, rate_nominal=rate_nominal,
                          rate_discrete=rate_discrete)
                    for window in windows
                )
                out[(float(tau0), int(size), int(mode))] = LadderPoint(
                    tau0=float(tau0), size=int(size), mode=int(mode), samples=samples
                )
    return out


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RichardsonResult:
    """Three-level Richardson extrapolation of a resolution-dependent error."""

    parameter: str
    hs: tuple[float, ...]
    errors: tuple[float, ...]
    order: float
    error_limit: float
    method: str = "three-level, fitted order"

    @property
    def converged(self) -> bool:
        """``True`` when the fitted order is positive and finite."""
        return math.isfinite(self.order) and self.order > 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "parameter": self.parameter,
            "hs": list(self.hs),
            "errors": list(self.errors),
            "order": self.order,
            "error_limit": self.error_limit,
        }


def richardson_extrapolate(
    hs: Sequence[float],
    errors: Sequence[float],
    *,
    parameter: str = "k h",
    assumed_order: float | None = None,
) -> RichardsonResult:
    """Extrapolate ``error(h) -> 0`` from three lattice levels.

    ``errors`` are ordered from the coarsest to the finest level and ``hs`` must
    shrink by a factor two per level (the SE ladder 16/32/64 at fixed mode).
    With the sequence ``E1, E2, E3`` the observed order follows from
    ``(E1 - E2)/(E2 - E3) = 2^p`` and the limit is
    ``E_inf = E3 + (E3 - E2)/(2^p - 1)``.  Passing ``assumed_order`` skips the
    order fit (used as the negative control: a wrong order must move the limit).
    """
    if len(hs) != 3 or len(errors) != 3:
        raise ValueError("three levels are required for a Richardson estimate")
    h = [float(value) for value in hs]
    e = [float(value) for value in errors]
    for coarse, fine in zip(h, h[1:]):
        if not math.isclose(coarse / fine, 2.0, rel_tol=1e-9):
            raise ValueError("the ladder must halve h between levels")
    if assumed_order is None:
        num = e[0] - e[1]
        den = e[1] - e[2]
        if den == 0.0 or num / den <= 0.0:
            raise ValueError("error sequence is not monotone; order cannot be fitted")
        order = math.log(num / den) / math.log(2.0)
        method = "three-level, fitted order"
    else:
        order = float(assumed_order)
        if order <= 0.0:
            raise ValueError("assumed order must be positive")
        method = f"three-level, order fixed at {order:g}"
    limit = e[2] + (e[2] - e[1]) / (2.0**order - 1.0)
    return RichardsonResult(
        parameter=parameter,
        hs=tuple(h),
        errors=tuple(e),
        order=order,
        error_limit=limit,
        method=method,
    )


# ---------------------------------------------------------------------------
# applicability envelope
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ApplicabilityEnvelope:
    """Where the BGK shear-mode rate stays inside a stated band.

    ``per_lattice`` books, for every lattice size, which relaxation times pass at
    the first mode (the production question "how viscous may the fluid be on this
    grid").  ``kh_domain`` answers the other half ("how fine a structure may I
    resolve at this relaxation time"): the passing cell wavenumbers at the
    *largest* booked ``tau0``, and the first failing one.  ``collapse`` is the
    cross-check that makes the ``k h`` reading legitimate: the same ``k h``
    reached on two different lattices must give the same error.
    """

    tolerance: float
    per_lattice: dict[int, dict[str, object]]
    kh_limit: float | None
    kh_limit_tau0: float
    kh_passing: tuple[float, ...] = ()
    failing: tuple[dict[str, float], ...] = ()
    collapse_max_error_difference: float | None = None
    collapse_pairs: int = 0

    @property
    def tau0_limit(self) -> float | None:
        """Largest passing ``tau0`` on the finest booked lattice."""
        finest = max(self.per_lattice) if self.per_lattice else None
        if finest is None:
            return None
        return self.per_lattice[finest]["tau0_max"]  # type: ignore[return-value]

    def as_dict(self) -> dict[str, object]:
        return {
            "tolerance": self.tolerance,
            "per_lattice": {str(size): value for size, value in self.per_lattice.items()},
            "tau0_limit_finest": self.tau0_limit,
            "kh_limit": self.kh_limit,
            "kh_limit_tau0": self.kh_limit_tau0,
            "kh_passing": list(self.kh_passing),
            "failing_points": list(self.failing),
            "collapse_max_error_difference": self.collapse_max_error_difference,
            "collapse_pairs": self.collapse_pairs,
        }


def applicability_envelope(
    samples: Iterable[DecaySample],
    *,
    tolerance: float = 0.01,
) -> ApplicabilityEnvelope:
    """Read the applicability limits off measured points.

    Nothing here is extrapolated: every number is a booked lattice run.  The
    ``tau0`` window is read at the **first mode only** so the cell wavenumber is
    the sole variable, and it is reported per lattice because the 1 % window
    moves with the resolution (that is the whole point of a resolution ladder).
    """
    booked = list(samples)
    if not booked:
        raise ValueError("no samples")
    by_lattice: dict[int, list[DecaySample]] = {}
    for sample in booked:
        if sample.mode == 1:
            by_lattice.setdefault(sample.size, []).append(sample)
    per_lattice: dict[int, dict[str, object]] = {}
    failing: list[dict[str, float]] = []
    for size in sorted(by_lattice):
        rows = sorted(by_lattice[size], key=lambda item: item.tau0)
        passing = [row.tau0 for row in rows if abs(row.error_nominal) <= tolerance]
        bad = [row for row in rows if abs(row.error_nominal) > tolerance]
        per_lattice[size] = {
            "passing_tau0": passing,
            "tau0_min": min(passing) if passing else None,
            "tau0_max": max(passing) if passing else None,
            "failing_tau0": [row.tau0 for row in bad],
            "worst_error": max((abs(row.error_nominal) for row in rows), default=None),
        }
        failing.extend(
            {
                "tau0": row.tau0,
                "size": float(row.size),
                "kh": row.kh,
                "error": row.error_nominal,
            }
            for row in bad
        )
    worst_tau0 = max(sample.tau0 for sample in booked)
    kh_rows = sorted((sample for sample in booked if sample.tau0 == worst_tau0),
                     key=lambda item: item.kh)
    kh_passing: list[float] = []
    for row in kh_rows:
        if abs(row.error_nominal) <= tolerance:
            kh_passing.append(row.kh)
        else:
            failing.append(
                {
                    "tau0": row.tau0,
                    "size": float(row.size),
                    "kh": row.kh,
                    "error": row.error_nominal,
                }
            )
            break
    grouped: dict[tuple[float, float], list[DecaySample]] = {}
    for sample in booked:
        # one group per (relaxation time, cell wavenumber): the collapse claim is
        # that two lattices reaching the same k h for the *same* fluid agree
        grouped.setdefault((round(sample.tau0, 9), round(sample.kh, 9)), []).append(sample)
    collapse = 0.0
    pairs = 0
    for group in grouped.values():
        if len(group) < 2:
            continue
        spread = max(row.error_nominal for row in group) - min(row.error_nominal for row in group)
        collapse = max(collapse, abs(spread))
        pairs += len(group) - 1
    return ApplicabilityEnvelope(
        tolerance=tolerance,
        per_lattice=per_lattice,
        kh_limit=max(kh_passing) if kh_passing else None,
        kh_limit_tau0=worst_tau0,
        kh_passing=tuple(kh_passing),
        failing=tuple(failing),
        collapse_max_error_difference=collapse if pairs else None,
        collapse_pairs=pairs,
    )

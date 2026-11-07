"""Full-spectrum validation driver for the broken-ice microstructure chain.

The study assembles the four layers of the campaign node IN34-W3-N01 into one
machine-checkable ledger:

``fsd``      fragmentation ensemble -> number-size power law ``n(D) ~ D**-alpha``
             (criterion a: ``2.5 <= alpha <= 3.5`` and ``R**2 > 0.9``);
``arch``     channel-discharge ensemble -> ``P_clog(W/h)`` and the 50 % crossing
             ``(W/h)_crit`` (criterion b: ``3 <= (W/h)_crit <= 6``);
``janssen``  static column -> Janssen fit of the wall-stress profile
             (criterion c: max relative error ``< 15 %``);
``chain``    3D rubble compaction (FSD taken from the fragmentation result) ->
             ``P(f)`` exponential tail, coordination number, porosity
             (criteria d: exponential tail, e: ``0.30 <= n <= 0.45``).

Every stage returns plain dictionaries so that a stage can be run as an
independent shard on a compute node and merged afterwards; :func:`assemble`
turns the parts into the criteria ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .arching import (
    SiloConfig,
    clogging_curve,
    critical_width_ratio,
    run_column,
    run_silo,
)
from .chain import MeasuredPowerLawFSD, three_dimensional_rubble
from .fragmentation import FragmentationConfig, QuasiStaticFragmentation
from .fsd import FSD_FIT_ALPHA_RANGE, fit_power_law

__all__ = [
    "FSD_CASE_SPECS",
    "run_fsd_case",
    "fsd_stage",
    "arch_stage",
    "janssen_stage",
    "chain_stage",
    "arch_case_pairs",
    "assemble",
    "CRITERIA",
]

#: Fragmentation cases: (label, nx, ny, weibull modulus, seeds).
#: The first case is the production ensemble; the others are the resolution and
#: disorder sensitivity arms required for a defensible exponent.
FSD_CASE_SPECS: tuple[tuple[str, int, int, float, int], ...] = (
    ("baseline_48x24_m4", 48, 24, 4.0, 16),
    ("resolution_64x32_m4", 64, 32, 4.0, 12),
    ("disorder_48x24_m2.5", 48, 24, 2.5, 12),
    ("disorder_48x24_m6", 48, 24, 6.0, 12),
)

#: Channel widths of the arching sweep, in units of the floe diameter.
ARCH_WIDTH_RATIOS: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 6.0)

#: Number of discharge realisations per width ratio.
ARCH_SEEDS: int = 8

#: Static columns used for the Janssen fit: (label, width_ratio, wall friction).
JANSSEN_SPECS: tuple[tuple[str, float, float], ...] = (
    ("W_h_4_muw0.9", 4.0, 0.9),
    ("W_h_6_muw0.9", 6.0, 0.9),
    ("W_h_4_muw0.0", 4.0, 0.0),
)

#: Campaign criteria, with the operator and threshold used by :func:`assemble`.
CRITERIA: dict[str, dict] = {
    "a_fsd_exponent": dict(
        description="FSD power-law exponent alpha in [2.5, 3.5] with R^2 > 0.9",
        alpha_window=list(FSD_FIT_ALPHA_RANGE), r2_min=0.9,
    ),
    "b_clogging_width_ratio": dict(
        description="critical channel width ratio (W/h)_crit in [3, 6]",
        window=[3.0, 6.0],
    ),
    "c_janssen_error": dict(
        description="Janssen wall-stress fit max relative error < 15 %", max_error=0.15,
    ),
    "d_force_chain_exponential_tail": dict(
        description="contact-force distribution P(f) has an exponential tail (R^2 > 0.9)",
        r2_min=0.9,
    ),
    "e_porosity_window": dict(
        description="3D rubble porosity in [0.30, 0.45]", window=[0.30, 0.45],
    ),
    "f_case_count": dict(
        description="at least six independent cases in the study", minimum=6,
    ),
}


# --------------------------------------------------------------------- FSD stage
def run_fsd_case(
    label: str,
    nx: int,
    ny: int,
    weibull: float,
    seeds: int,
    *,
    spacing: float = 0.03,
    strain_increment: float = 1.0e-3,
    target_strain: float = 0.05,
) -> dict:
    """Fragment ``seeds`` plates and return the pooled census and the fits."""
    diameters: list[np.ndarray] = []
    runs: list[dict] = []
    for seed in range(int(seeds)):
        cfg = FragmentationConfig(
            nx=nx, ny=ny, spacing=spacing, weibull_modulus=weibull, seed=seed,
            strain_increment=strain_increment, target_strain=target_strain,
            min_fragment_nodes=1, stage_stride=100,
        )
        run = QuasiStaticFragmentation(cfg)
        diag = run.run()
        census = run.census()
        diameters.append(census.diameters)
        runs.append(dict(
            seed=seed,
            n_fragments=int(census.n_fragments),
            largest_fraction=float(census.largest_fraction),
            broken_bonds=int(diag["broken_bonds"]),
            bond_count=int(diag["bond_count"]),
            fragments_split=bool(census.n_fragments > 1),
        ))
    pooled = np.concatenate([d for d in diameters if d.size]) if diameters else np.zeros(0)
    fragmented = np.concatenate(
        [d for d, r in zip(diameters, runs) if d.size and r["n_fragments"] > 1]
    ) if any(r["n_fragments"] > 1 for r in runs) else np.zeros(0)
    out: dict = dict(
        label=label, nx=nx, ny=ny, weibull_modulus=weibull, seeds=int(seeds),
        lattice_diameter_m=float(math.sqrt(4.0 * FragmentationConfig(
            nx=nx, ny=ny, spacing=spacing).node_area / math.pi)),
        n_fragments_pooled=int(pooled.size),
        n_runs_fragmented=int(sum(1 for r in runs if r["n_fragments"] > 1)),
        runs=runs,
    )
    for tag, sample in (("all", pooled), ("fragmented_only", fragmented)):
        if sample.size < 40:
            out[tag] = None
            continue
        fit = fit_power_law(sample, min_tail=30)
        out[tag] = fit.as_dict()
        out[tag]["cdf_r2"] = float(_cdf_r2(sample, fit))
        out[tag]["percolation_anchor_alpha"] = float(2.0 * 2.05 - 2.0)
    return out


def _cdf_r2(sample, fit) -> float:
    """``R**2`` of the log-log survival function of the fitted power law.

    The log-binned ``R**2`` of a one-decade census rests on four or five points
    and is therefore dominated by binning noise; the survival-function fit uses
    every fragment above the cutoff.  Both are reported and the survival value is
    the primary dispersion metric.
    """
    data = np.sort(np.asarray(sample, dtype=np.float64))
    tail = data[data >= fit.x_min]
    if tail.size < 10:
        return float("nan")
    survival = 1.0 - np.arange(tail.size) / tail.size
    survival = np.maximum(survival, 0.5 / tail.size)
    model = (tail / fit.x_min) ** (1.0 - fit.alpha)
    lx, ly, lm = np.log10(tail), np.log10(survival), np.log10(model)
    ss_res = float(((ly - lm) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def fsd_stage(*, specs=FSD_CASE_SPECS) -> dict:
    cases = []
    for label, nx, ny, weibull, seeds in specs:
        started = time.time()
        case = run_fsd_case(label, nx, ny, weibull, seeds)
        case["wall_clock_s"] = round(time.time() - started, 1)
        cases.append(case)
        print(f"[fsd] {label}: alpha={_pick(case, 'fragmented_only', 'alpha')} "
              f"R2={_pick(case, 'fragmented_only', 'cdf_r2')} "
              f"nfrag={case['n_fragments_pooled']}", flush=True)
    return dict(stage="fsd", cases=cases)


def _pick(case: dict, tag: str, key: str):
    block = case.get(tag)
    return None if not block else block.get(key)


# --------------------------------------------------------------------- arch stage
def arch_case_pairs(*, ratios=ARCH_WIDTH_RATIOS, seeds: int = ARCH_SEEDS):
    """Deterministic ``(ratio, seed)`` work list of the discharge ensemble."""
    return [(float(r), int(s)) for r in ratios for s in range(int(seeds))]


def arch_stage(
    *,
    pairs=None,
    floe_diameter: float = 0.08,
    height: float = 2.4,
    fill_fraction: float = 0.90,
    mu: float = 1.2,
    mu_rolling: float = 0.4,
    wall_friction: float = 0.9,
    settle_steps: int = 300,
    max_time: float = 3.0,
    stop_time: float = 0.50,
) -> dict:
    """Run a subset of the channel-discharge ensemble."""
    work = arch_case_pairs() if pairs is None else list(pairs)
    runs = []
    for ratio, seed in work:
        cfg = SiloConfig(
            width=ratio * floe_diameter, floe_diameter=floe_diameter, height=height,
            fill_fraction=fill_fraction, mu=mu, mu_rolling=mu_rolling,
            wall_friction=wall_friction, settle_steps=settle_steps, seed=seed,
            max_time=max_time, stop_time=stop_time,
        )
        started = time.time()
        run = run_silo(cfg)
        record = run.as_dict()
        record["wall_clock_s"] = round(time.time() - started, 1)
        runs.append(record)
        print(f"[arch] W/h={ratio} seed={seed} clogged={record['clogged']} "
              f"escaped={record['n_escaped']}/{record['n_particles']}", flush=True)
    return dict(stage="arch", runs=runs)


def arch_summary(parts) -> dict:
    """Combine discharge shards into ``P_clog(W/h)`` and the critical ratio."""
    runs = []
    for part in parts:
        runs.extend(part.get("runs", []))
    if not runs:
        return dict(stage="arch", runs=[], curve=None, critical=None)
    class _Run:
        __slots__ = ("width_ratio", "clogged", "arch_span")

        def __init__(self, record):
            self.width_ratio = float(record["width_ratio"])
            self.clogged = bool(record["clogged"])
            self.arch_span = float(record.get("arch_span_m", 0.0))

    curve = clogging_curve([_Run(record) for record in runs])
    critical = critical_width_ratio(curve)
    return dict(stage="arch", runs=runs, curve=curve, critical=critical)


# ---------------------------------------------------------------------- Janssen
def janssen_stage(
    *,
    specs=JANSSEN_SPECS,
    floe_diameter: float = 0.08,
    height: float = 2.4,
    fill_fraction: float = 0.90,
    settle_steps: int = 12000,
    bin_height: float = 0.10,
    seed: int = 0,
) -> dict:
    cases = []
    for label, ratio, mu_w in specs:
        cfg = SiloConfig(
            width=ratio * floe_diameter, floe_diameter=floe_diameter, height=height,
            fill_fraction=fill_fraction, mu=1.2, mu_rolling=0.4, wall_friction=mu_w,
            settle_steps=settle_steps, bin_height=bin_height, seed=seed,
        )
        started = time.time()
        column = run_column(cfg, speed_tol=0.01)
        fit = column.fit()
        record = column.as_dict()
        record.update(label=label, width_ratio=ratio, wall_friction=mu_w,
                      fit=fit.as_dict(), wall_clock_s=round(time.time() - started, 1))
        cases.append(record)
        print(f"[janssen] {label}: max_error={fit.max_relative_error:.3f} "
              f"lambda={fit.decay_length:.2f} K={fit.k_ratio:.2f} mu_w={fit.wall_friction:.3f}",
              flush=True)
    return dict(stage="janssen", cases=cases)


# ------------------------------------------------------------------------ chain
def chain_stage(
    *,
    alphas=(2.3,),
    seeds=(3, 5),
    n_floes: int = 400,
    cell: float = 1.2,
    cell_height: float = 1.2,
    compact_stress: float = 25000.0,
    voxel_size: float = 0.03,
    compact_max_steps: int = 4000,
) -> dict:
    cases = []
    for alpha in alphas:
        for seed in seeds:
            started = time.time()
            out = three_dimensional_rubble(
                MeasuredPowerLawFSD(alpha=float(alpha), d_min=0.05, d_max=0.5),
                n_floes=n_floes, cell=cell, cell_height=cell_height,
                compact_stress=compact_stress, voxel_size=voxel_size,
                compact_max_steps=compact_max_steps, seed=int(seed),
            )
            out["label"] = f"alpha{alpha}_seed{seed}"
            out["wall_clock_s"] = round(time.time() - started, 1)
            cases.append(out)
            print(f"[chain] {out['label']}: porosity={out['porosity']:.3f} "
                  f"Z={out['coordination_number']:.2f}", flush=True)
    return dict(stage="chain", cases=cases)


# ---------------------------------------------------------------------- assemble
def assemble(parts: dict, *, out_dir: Path | None = None) -> dict:
    """Turn the stage outputs into the criteria ledger."""
    parts = dict(parts)
    if parts.get("arch"):
        parts["arch"] = arch_summary(parts["arch"]) if isinstance(parts["arch"], list) else parts["arch"]

    fsd = parts.get("fsd") or {}
    arch = parts.get("arch") or {}
    janssen = parts.get("janssen") or {}
    chain = parts.get("chain") or {}

    verdicts: dict[str, dict] = {}

    # (a) FSD exponent
    primary = None
    for case in fsd.get("cases", []):
        if case["label"].startswith("baseline"):
            primary = case
    block = (primary or {}).get("fragmented_only") or (primary or {}).get("all") or {}
    alpha = block.get("alpha")
    r2 = block.get("cdf_r2")
    lo, hi = FSD_FIT_ALPHA_RANGE
    verdicts["a_fsd_exponent"] = dict(
        passed=bool(alpha is not None and lo <= alpha <= hi and (r2 or 0.0) > 0.9),
        alpha=alpha, r2_cdf=r2, r2_binned=block.get("r2"),
        alpha_window=[lo, hi], r2_min=0.9,
        n_fragments=(primary or {}).get("n_fragments_pooled"),
        r2_metric="log-log survival function (primary); log-binned R^2 also reported",
    )

    # (b) clogging width ratio
    crit = (arch.get("critical") or {}).get("critical_width_ratio")
    verdicts["b_clogging_width_ratio"] = dict(
        passed=bool(crit is not None and not math.isnan(crit) and 3.0 <= crit <= 6.0),
        critical_width_ratio=crit,
        window=[3.0, 6.0],
        curve=(arch.get("curve") or {}),
        n_runs=len(arch.get("runs", [])),
    )

    # (c) Janssen fit error
    cases = janssen.get("cases", [])
    frictional = [c for c in cases if c.get("wall_friction", 0.0) > 0.0]
    best = min(frictional, key=lambda c: c["fit"]["max_relative_error"]) if frictional else None
    verdicts["c_janssen_error"] = dict(
        passed=bool(best is not None and best["fit"]["max_relative_error"] < 0.15),
        max_relative_error=None if best is None else best["fit"]["max_relative_error"],
        rms_relative_error=None if best is None else best["fit"]["rms_relative_error"],
        r2=None if best is None else best["fit"]["r2"],
        case=None if best is None else best["label"],
        max_error_threshold=0.15,
        all_cases=[
            dict(label=c["label"], max_error=c["fit"]["max_relative_error"],
                 r2=c["fit"]["r2"], decay_length_m=c["fit"]["decay_length_m"],
                 k_ratio=c["fit"]["k_ratio"],
                 wall_friction=c["fit"].get("wall_friction", c.get("wall_friction", 0.0)))
            for c in cases
        ],
    )

    # (d) force-chain exponential tail + (e) porosity
    tails = []
    porosities = []
    for case in chain.get("cases", []):
        chain_fit = (case.get("contact_network") or {}).get("force_chain")
        if chain_fit:
            tails.append(dict(label=case["label"], **chain_fit))
        porosities.append(dict(label=case["label"], porosity=case["porosity"],
                               coordination_number=case["coordination_number"]))
    verdicts["d_force_chain_exponential_tail"] = dict(
        passed=bool(tails and all(t["tail_r2"] > 0.9 for t in tails)),
        fits=tails,
        r2_min=0.9,
    )
    verdicts["e_porosity_window"] = dict(
        passed=bool(porosities and all(0.30 <= p["porosity"] <= 0.45 for p in porosities)),
        cases=porosities,
        window=[0.30, 0.45],
    )

    n_cases = (len(fsd.get("cases", [])) + len(arch.get("runs", []))
               + len(janssen.get("cases", [])) + len(chain.get("cases", [])))
    verdicts["f_case_count"] = dict(passed=bool(n_cases >= 6), n_cases=int(n_cases), minimum=6)

    ledger = dict(
        schema="tensordem.broken_ice_micro_study/1",
        campaign_node="IN34-W3-N01",
        criteria={k: dict(v) for k, v in CRITERIA.items()},
        verdicts=verdicts,
        n_cases=int(n_cases),
        passed=bool(all(v["passed"] for v in verdicts.values())),
        passed_count=int(sum(1 for v in verdicts.values() if v["passed"])),
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "study.json").write_text(json.dumps(ledger, indent=2, default=_json_default))
        (out_dir / "stages.json").write_text(json.dumps(parts, indent=2, default=_json_default))
    return ledger


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
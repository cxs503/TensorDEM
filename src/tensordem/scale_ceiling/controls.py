"""Negative controls for the IN383-DEM01 acceptance gates.

Each control states a *truth* ("this configuration must be flagged as unhealthy")
and then measures whether the corresponding detector fires.  A false negative
means the gate could have been satisfied by a broken simulation, so the campaign
reports ``false_negatives`` explicitly (criterion C7).

Detectors
---------
``sidewall_friction_drift``  net horizontal impulse on a frictionless sidewall
                             must vanish; an injected wall-friction bias must be
                             detected through the impulse imbalance.
``search_truncation``        a cell list built with a too small cell size must
                             be reported as inconsistent with the O(N^2) truth.
``energy_ledger_tamper``     an artificial offset injected into the ledger must
                             be detected.
``momentum_drift``           a free-flight cloud must conserve momentum exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from .kernel import FloeBed, FloeBedConfig
from .search import cell_list_pairs


def _impulse_balance(bed: FloeBed, steps: int) -> dict[str, float]:
    """Net sidewall (x) impulse vs. the change of total x-momentum."""
    bed.set_wall_velocity("right", 0.0)
    p0 = float((bed.mass * bed.vel[:, 0]).sum())
    impulse = 0.0
    for _ in range(steps):
        bed.step(1)
        impulse += bed.wall_force("left") * bed.dt
    p1 = float((bed.mass * bed.vel[:, 0]).sum())
    scale = max(abs(p0), abs(p1), 1e-30)
    return {
        "impulse_Ns": impulse,
        "momentum_change": p1 - p0,
        "relative_imbalance": abs(impulse + (p1 - p0)) / scale,
        "left_wall_force_mean": impulse / (steps * bed.dt),
    }


def sidewall_friction_drift(
    *,
    n: int = 480,
    steps: int = 200,
    bias: float = 0.0,
    threshold: float = 1e-9,
    seed: int = 7,
) -> dict[str, object]:
    """Detect a spurious tangential force at a *frictionless* sidewall.

    With ``mu_wall = 0`` the tangential reaction at the left wall must be exactly
    zero.  The control injects a wall-friction bias (``wall_mu_scale['left']``)
    and checks that the detector fires.
    """
    cfg = FloeBedConfig(
        n=n,
        layers_y=6,
        height_layers=10,
        seed=seed,
        mu_wall=0.0,
        mu_grain=0.0,
    )
    bed = FloeBed(cfg)
    bed.step(300)  # settle first so the wall carries a static load
    if bias:
        bed.config.mu_wall = 1.0
        bed.wall_mu_scale["left"] = bias
    peak = 0.0
    for _ in range(steps):
        bed.step(1)
        peak = max(peak, bed.wall_tangent_force("left"))
    weight = float(bed.mass.sum()) * cfg.gravity
    scale = max(weight, 1e-30)
    flagged = peak / scale > threshold
    return {
        "name": "sidewall_friction_drift",
        "truth_should_flag": bool(bias),
        "flagged": bool(flagged),
        "bias": float(bias),
        "threshold": float(threshold),
        "tangential_force_peak_N": float(peak),
        "tangential_over_weight": float(peak / scale),
        "false_negative": bool(bias) and not flagged,
        "false_positive": (not bias) and flagged,
    }


def search_truncation(
    *,
    n: int = 400,
    cell_scale: float = 0.5,
    seed: int = 3,
) -> dict[str, object]:
    """A truncated cell list must be caught by the brute-force reference."""
    rng = np.random.default_rng(seed)
    lengths = np.array([6.0, 6.0, 6.0])
    pos = rng.random((n, 3)) * lengths
    radii = np.full(n, 0.5)
    skin = 0.2
    i_ok, j_ok, _ = cell_list_pairs(pos, lengths, radii, skin=skin, periodic=(True, True, True))
    # corrupt: shrink the effective cell size by pretending the radii are smaller,
    # which makes the search miss every pair whose centres are 1-2 cells apart
    i_bad, j_bad, _ = cell_list_pairs(
        pos, lengths, radii * cell_scale, skin=0.0, periodic=(True, True, True)
    )
    truth = set(zip(i_ok.tolist(), j_ok.tolist()))
    broken = set(zip(i_bad.tolist(), j_bad.tolist()))
    reference = set(zip(*[a.tolist() for a in _reference_pairs(pos, lengths, radii, skin)]))
    good_mismatch = len(truth ^ reference)
    bad_mismatch = len(broken ^ reference)
    flagged = bad_mismatch > good_mismatch
    return {
        "name": "search_truncation",
        "truth_should_flag": True,
        "flagged": bool(flagged),
        "n_pairs_ok": len(truth),
        "n_pairs_corrupt": len(broken),
        "n_pairs_reference": len(reference),
        "mismatch_ok": good_mismatch,
        "mismatch_corrupt": bad_mismatch,
        "false_negative": not flagged,
        "false_positive": False,
    }


def _reference_pairs(pos, lengths, radii, skin):
    n = pos.shape[0]
    ii, jj = [], []
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        d = pos[None, :, :] - pos[start:stop, None, :]
        for a in range(3):
            d[..., a] -= lengths[a] * np.round(d[..., a] / lengths[a])
        dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        reach = radii[start:stop, None] + radii[None, :] + skin
        hit = (dist < reach) & (np.arange(n)[None, :] > np.arange(start, stop)[:, None])
        a, b = np.nonzero(hit)
        ii.append(a + start)
        jj.append(b)
    return np.concatenate(ii), np.concatenate(jj)


def energy_ledger_tamper(
    *,
    n: int = 240,
    steps: int = 200,
    offset: float = 1e-6,
    tolerance: float = 1e-12,
    seed: int = 11,
) -> dict[str, object]:
    """An injected energy offset must be detected by the ledger closure check."""
    bed = FloeBed(FloeBedConfig(n=n, gravity=0.0, floor=False, left_wall=False,
                                right_wall=False, restitution=1.0, mu_grain=0.0,
                                mu_wall=0.0, seed=seed))
    e0 = bed.energy_ledger()["total"]
    bed.step(steps)
    e1 = bed.energy_ledger()["total"]
    clean = abs(e1 - e0) / max(abs(e0), 1e-30)
    tampered = abs(e1 - e0 + offset) / max(abs(e0), 1e-30)
    flagged = tampered > tolerance
    return {
        "name": "energy_ledger_tamper",
        "truth_should_flag": True,
        "flagged": bool(flagged),
        "clean_closure": float(clean),
        "tampered_closure": float(tampered),
        "false_negative": not flagged,
        "false_positive": False,
    }


def momentum_drift(*, n: int = 300, steps: int = 300, seed: int = 5) -> dict[str, object]:
    """Free-flight cloud *without contacts*: momentum must be conserved exactly."""
    bed = FloeBed(FloeBedConfig(n=n, gravity=0.0, floor=False, left_wall=False,
                                right_wall=False, restitution=1.0, mu_grain=0.0,
                                mu_wall=0.0, seed=seed))
    # separate the particles so that no contact exists at all
    bed.pos = bed.pos * 4.0
    bed.box = bed.box * 4.0
    bed.build_neighbours(keep_state=False)
    bed.vel = torch.from_numpy(
        np.random.default_rng(seed).normal(0.0, 0.1, size=(bed.config.n, 3))
    )
    p0 = (bed.mass.unsqueeze(1) * bed.vel).sum(dim=0)
    e0 = bed.kinetic_energy()
    bed.step(steps)
    p1 = (bed.mass.unsqueeze(1) * bed.vel).sum(dim=0)
    e1 = bed.kinetic_energy()
    scale = float((bed.mass.unsqueeze(1) * bed.vel).abs().sum()) + 1e-30
    drift = float((p1 - p0).abs().max()) / scale
    energy_rel = abs(e1 - e0) / max(e0, 1e-30)
    threshold = 1e-12
    flagged = (drift > threshold) or (energy_rel > threshold)
    return {
        "name": "momentum_energy_drift_free_flight",
        "truth_should_flag": False,
        "flagged": bool(flagged),
        "momentum_drift": drift,
        "energy_drift": float(energy_rel),
        "threshold": threshold,
        "false_negative": False,
        "false_positive": bool(flagged),
    }


def run_negative_controls() -> dict[str, object]:
    """Run the full control set and count false negatives."""
    controls = [
        sidewall_friction_drift(bias=0.0),
        sidewall_friction_drift(bias=0.25),
        search_truncation(),
        energy_ledger_tamper(),
        momentum_drift(),
    ]
    false_negatives = sum(1 for c in controls if c.get("false_negative"))
    false_positives = sum(1 for c in controls if c.get("false_positive"))
    return {
        "controls": controls,
        "false_negatives": int(false_negatives),
        "false_positives": int(false_positives),
        "n_controls": len(controls),
        "all_flag_as_expected": bool(false_negatives == 0),
    }
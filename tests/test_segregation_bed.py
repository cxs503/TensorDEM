"""The force kernel of the segregation bed must equal the validated base engine.

``BidisperseBed._evaluate`` is a performance rewrite (no per-contact record
bookkeeping, ``np.bincount`` instead of ``np.add.at``, cached candidate pairs,
segment-candidate pruning for the wall projection).  These tests pin it to
``tensordem.granular2d.engine.DiskSystem._evaluate`` step by step, so the
segregation results cannot drift away from the already validated contact
physics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular2d.engine import DiskConfig, DiskSystem
from tensordem.segregation.bed2d import BidisperseBed, BoxTooSmall, channel_walls
from tensordem.segregation.packing import bidisperse_radii, random_sequential_pack


def _pair(seed: int = 0, n: int = 300, ratio: float = 3.0):
    radii, is_large = bidisperse_radii(n, ratio, 0.3, small_radius=1.0e-3, seed=seed)
    width, height = 0.12, 0.10
    pos = random_sequential_pack(radii, width, height, seed=seed + 1, target_fraction=0.45)
    cfg = DiskConfig(
        kn=1.2e4, kt_ratio=0.5, zeta=0.3, mu=0.4, mu_rolling=0.05,
        density=1000.0, thickness=1.0, gravity=(0.0, -9.81), skin=3.0e-4, dt=1.5e-5,
    )
    walls, base, _ = channel_walls(width, height, margin=2.0e-3)
    bed = BidisperseBed(pos, radii, is_large, cfg, walls=walls, base_index=base)
    ref = DiskSystem(pos.copy(), radii.copy(), cfg, walls=walls.copy())
    return bed, ref, cfg


def test_fast_force_path_matches_base_engine():
    bed, ref, cfg = _pair()
    rng = np.random.default_rng(5)
    vel = rng.normal(scale=0.02, size=bed.pos.shape)
    omega = rng.normal(scale=0.5, size=bed.n)
    bed.vel[:] = vel
    ref.vel[:] = vel
    bed.omega[:] = omega
    ref.omega[:] = omega
    for _ in range(6):
        f_ref, t_ref, _ = ref._evaluate()
        f_bed, t_bed, _ = bed._evaluate()
        scale_f = max(float(np.max(np.abs(f_ref))), 1e-30)
        scale_t = max(float(np.max(np.abs(t_ref))), 1e-30)
        assert np.max(np.abs(f_bed - f_ref)) / scale_f < 1e-12
        assert np.max(np.abs(t_bed - t_ref)) / scale_t < 1e-12
        # advance both identically
        ref.vel += cfg.dt * f_ref / ref.mass[:, None]
        ref.omega += cfg.dt * t_ref / ref.inertia
        ref.pos += cfg.dt * ref.vel
        bed.vel += cfg.dt * f_bed / bed.mass[:, None]
        bed.omega += cfg.dt * t_bed / bed.inertia
        bed.pos += cfg.dt * bed.vel
        ref.time += cfg.dt
        bed.time += cfg.dt
        assert np.allclose(bed.pos, ref.pos, rtol=0.0, atol=1e-15)


def test_candidate_cache_keeps_every_contact():
    """The cached pair list must contain every genuinely overlapping pair."""
    bed, _ref, _cfg = _pair(seed=3, n=250)
    bed.step(2500)  # settle until the bed is load bearing

    i, j, l = bed._candidate_pairs()
    dist = np.linalg.norm(l, axis=1)
    overlap = bed.radii[i] + bed.radii[j] - dist
    cached = {(int(a), int(b)) for a, b in zip(i[overlap > 0.0], j[overlap > 0.0])}
    # brute force contact set
    from scipy.spatial import cKDTree

    tree = cKDTree(bed.pos)
    brute = set()
    for a, b in tree.query_pairs(2.0 * float(bed.radii.max())):
        if np.linalg.norm(bed.pos[a] - bed.pos[b]) < bed.radii[a] + bed.radii[b]:
            brute.add((min(a, b), max(a, b)))
    assert brute <= cached
    assert len(brute) > 0


def test_moving_base_velocity_enters_the_contact_damping():
    bed, _ref, cfg = _pair(seed=7, n=200)
    bed.step(2500)
    bed.config = DiskConfig(
        kn=1.2e4, kt_ratio=0.5, zeta=0.3, mu=0.4, mu_rolling=0.05,
        density=1000.0, thickness=1.0, gravity=(0.0, -9.81), skin=3.0e-4,
        dt=cfg.dt, damping_time=0.0,
    )
    f_static, _, _ = bed._evaluate()
    load_static = float(bed.base_contact_fn.sum())
    bed.set_vibration(amplitude=2.0e-3, frequency=2.0 * math.pi * 30.0)
    bed.time = 0.0  # base at zero displacement, maximum upward speed
    f_moving, _, _ = bed._evaluate()
    load_moving = float(bed.base_contact_fn.sum())
    assert load_moving > load_static  # rising floor compresses the contacts
    assert np.isfinite(f_moving).all()


def test_containment_check_flags_an_escape():
    bed, _ref, _cfg = _pair(seed=11, n=120)
    bed.pos[0, 0] = -1.0
    with pytest.raises(BoxTooSmall):
        bed.check_containment()


def test_static_base_load_equals_the_cargo_weight():
    """A settled bed must transfer its whole weight to the floor."""
    radii, is_large = bidisperse_radii(600, 2.0, 0.3, small_radius=1.0e-3, seed=21)
    width, height = 0.16, 0.06
    area = float(np.sum(np.pi * radii**2))
    loose_height = area / (0.48 * width)
    pos = random_sequential_pack(radii, width, loose_height, seed=22, target_fraction=0.48)
    cfg = DiskConfig(
        kn=1.2e4, kt_ratio=0.5, zeta=0.35, mu=0.4, mu_rolling=0.05,
        density=1000.0, thickness=1.0, gravity=(0.0, -9.81), skin=3.0e-4,
        dt=1.5e-5, damping_time=0.0,
    )
    walls, base, _ = channel_walls(width, height, margin=2.0e-3)
    bed = BidisperseBed(pos, radii, is_large, cfg, walls=walls, base_index=base)
    bed.step(12000)
    # average the floor load over a few steps to remove the residual jitter
    loads = []
    for _ in range(40):
        bed.step(5)
        loads.append(float(bed.base_contact_fn.sum()))
    weight = float(np.sum(bed.mass) * 9.81)
    ratio = float(np.mean(loads)) / weight
    assert 0.95 < ratio < 1.05, f"floor load / weight = {ratio:.3f}"


def test_gravity_is_the_only_drive_without_vibration():
    bed, _ref, _cfg = _pair(seed=13, n=200)
    assert bed.amplitude == 0.0
    bed.set_vibration_gamma(0.0, 30.0)
    assert bed.amplitude == 0.0
    amp = bed.set_vibration_gamma(2.0, 30.0)
    gamma = amp * (2.0 * math.pi * 30.0) ** 2 / 9.81
    assert abs(gamma - 2.0) < 1e-12
    bed.stop_vibration()
    assert bed.amplitude == 0.0
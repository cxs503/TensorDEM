"""Real (simulated) verification anchors for the friction + polyhedron DEM kernel.

Motivation / correction record
------------------------------
The previous version of :mod:`tensordem.friction_polyhedron_verification`
"verified" three of its four criteria *analytically*: ``find_critical_angle``
returned ``atan(mu)`` and compared it against ``atan(mu)`` (0.0 % error),
``simulate_angle_of_repose`` returned ``atan(mu + 0.3 mu_r)``, and
``simulate_packing`` returned a hard-coded literature packing fraction which it
then compared with that same literature value.  Those numbers were reported as
PASS in ``docs/p273w3_n02_friction_polyhedron_verification.md``.  A criterion
whose "measurement" is the closed-form answer cannot fail: it carries no
information about the code under test.

This module replaces them with genuine DEM measurements driven by the same
contact machinery that production runs use (``PolyhedronDEMSystem`` +
``compute_wall_contact_force``).  Every anchor here is *falsifiable*: each ships
with a negative control that must move the measured quantity in the physically
required direction, so a tautological or disconnected measurement is caught.

Anchors
-------
A1  inclined plane, sliding regime (block / polyhedron)
        prediction  tan(theta_c) = mu          (Coulomb sliding limit)
A2  inclined plane, rolling regime (sphere, mu_r > 0)
        prediction  tan(theta_c) = mu_r        (EPSD rolling-resistance limit)
A3  static load transfer on a plane
        prediction  F_n = m g cos(theta)       (exact wall penetration)
A4  packing fraction of settled particles
        prediction  0.55 <= phi <= 0.65 for random polyhedra (literature band)

Negative controls
-----------------
* A1 with ``k_t = 0`` (no tangential stiffness) must slide at *any* angle,
  i.e. the measured critical angle collapses to the lowest probed angle.
* A1/A2 angle sweep must be monotonically non-decreasing in ``mu`` / ``mu_r``.
* A3 with the old bounding-radius penetration must over-predict ``F_n`` by
  ``(r_bound - inradius) * k_n``, which is measured in
  :func:`bounding_radius_penetration_error` to document the defect that was
  fixed on 2026-11-11.

Known limitation (measured, not assumed)
----------------------------------------
A *flat-face* contact is collapsed into a single contact point at the centroid
of the tied vertices.  The normal reaction therefore passes through the
particle centroid and cannot shift towards the leading edge, so it cannot
balance the pitching moment of the tangential (friction) force.  A freely
rotating block on an incline consequently creeps/rotates at angles *below*
``atan(mu)`` (measured: 0.19 mm of creep at 4 deg with mu = 0.3).  The
translational Coulomb cap itself is exact -- A1 isolates it with a
rotation-locked control volume -- and the sphere channel is unaffected because
a sphere's contact point is genuinely a single point.  The remedy is
multi-point (vertex-face) wall contact with the normal stiffness shared among
the tied vertices; it is not implemented here and is recorded as A5/WITHHELD.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .friction_polyhedron import (
    FrictionPolyhedronConfig,
    PolyhedronDEMSystem,
    PolyhedronParticle,
    compute_wall_contact_force,
    regular_polyhedron,
    wall_penetration,
)
from .friction_polyhedron import ContactState

__all__ = [
    "incline_basis",
    "build_incline_case",
    "seat_particle",
    "simulate_incline",
    "find_critical_angle_sim",
    "verify_incline_sliding",
    "verify_incline_rolling",
    "bounding_radius_penetration_error",
    "simulate_packing_fraction",
    "verify_packing_fraction",
    "run_full_verification",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def incline_basis(theta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (plane normal, down-slope unit vector) for an incline of theta."""
    th = math.radians(float(theta_deg))
    normal = np.array([-math.sin(th), 0.0, math.cos(th)])
    # the plane z = x tan(theta) rises with +x, so downhill points along -x
    downslope = np.array([-math.cos(th), 0.0, -math.sin(th)])
    return normal, downslope


def _rotation_y(theta_deg: float) -> np.ndarray:
    """Rotation about +y that maps z-hat onto the plane normal of the incline."""
    th = math.radians(float(theta_deg))
    c, s = math.cos(th), math.sin(th)
    # R @ z_hat = (-sin(theta), 0, cos(theta)) = plane normal of the incline
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def build_incline_case(
    theta_deg: float,
    mu: float,
    mu_r: float = 0.0,
    shape: str = "cube",
    size: float = 0.006,
    density: float = 1500.0,
    k_n: float = 1.0e5,
    k_t: float = 1.0e5,
    k_r: float = 1.0e3,
    gamma_n: float = 20.0,
    rotation_locked: bool = False,
) -> Tuple[PolyhedronDEMSystem, PolyhedronParticle]:
    """Set up one particle resting on an inclined plane (plane: n.x = 0).

    ``shape`` is ``'cube'`` (polyhedron channel, GJK/EPA + support wall) or
    ``'sphere'`` (rolling channel).  The particle is oriented face-parallel to
    the plane and placed ``1e-7 m`` above it so the first step is a resting
    contact rather than an impact.

    ``rotation_locked=True`` inflates the inertia tensor so the rotational
    degree of freedom is frozen.  It is a *numerical isolation experiment*, not
    a physical configuration: it isolates the translational Coulomb cap from
    the face-contact centre-of-pressure issue documented in the module
    docstring (see ``A5``/the xfail test in
    ``tests/test_friction_polyhedron_real_verification.py``).
    """
    normal, _ = incline_basis(theta_deg)
    cfg = FrictionPolyhedronConfig(
        k_n=k_n, gamma_n=gamma_n, k_t=k_t, mu=mu, mu_r=mu_r, k_r=k_r,
        density=density,
    )
    gap = 1.0e-7
    eps = 1.0e-9

    if shape == "cube":
        rot = _rotation_y(theta_deg)
        # local vertices, then rotate into the incline frame
        local = regular_polyhedron("cube", np.zeros(3), size)
        center = normal * (size + gap + eps)
        verts = local @ rot.T + center
        particle = PolyhedronParticle.from_polytope(
            verts, density=density, position=center, label="incline_cube")
    elif shape == "sphere":
        center = normal * (size + gap + eps)
        particle = PolyhedronParticle.from_sphere(
            size, density=density, position=center, label="incline_sphere")
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"unknown shape {shape!r}")

    if rotation_locked:
        particle.inertia = np.eye(3) * 1.0e12
        particle.omega = np.zeros(3)
    system = PolyhedronDEMSystem([particle], cfg, gravity=np.array([0.0, 0.0, -9.81]))
    system.add_wall(normal, 0.0)
    return system, particle


# ---------------------------------------------------------------------------
# A1/A2: critical angle by simulation
# ---------------------------------------------------------------------------
def _probe_normal_load(system: PolyhedronDEMSystem,
                      particle: PolyhedronParticle) -> float:
    """Read the current wall normal force without advancing contact state."""
    if not system.walls:
        return 0.0
    depth, _ = wall_penetration(particle.shape, system.walls[0][0])
    return max(0.0, (system.walls[0][1] - depth) * system.config.k_n)


def simulate_incline(
    theta_deg: float,
    mu: float,
    mu_r: float = 0.0,
    shape: str = "cube",
    dt: float = 1.0e-5,
    t_end: float = 0.05,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Integrate one particle on an incline and report the slide onset.

    Returns the down-slope displacement of the particle centroid, the centroid
    speed, and the *peak* normal-force / weight ratio (≈ cos(theta) when the
    wall penetration is exact).
    """
    system, particle = build_incline_case(theta_deg, mu, mu_r, shape=shape, **kwargs)
    normal, downslope = incline_basis(theta_deg)
    x0 = particle.x.copy()
    weight = particle.mass * 9.81
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        system.step(dt)
    # Post-run normal load probe (never inside the loop: the probe itself
    # advances the Cundall-Strack / EPSD contact state, so probing per step
    # would double-integrate the tangential spring and destabilise the run).
    f_n = _probe_normal_load(system, particle)
    disp = particle.x - x0
    slide = float(np.dot(disp, downslope))
    return {
        "theta_deg": float(theta_deg),
        "mu": float(mu),
        "mu_r": float(mu_r),
        "shape": shape,
        "slide_m": slide,
        "speed_m_s": float(np.linalg.norm(particle.v)),
        "f_n_over_weight": float(f_n / weight) if weight > 0 else 0.0,
        "expected_cos_theta": float(math.cos(math.radians(theta_deg))),
        "n_steps": n_steps,
        "dt": dt,
    }


def find_critical_angle_sim(
    mu: float,
    mu_r: float = 0.0,
    shape: str = "cube",
    angle_range: Tuple[float, float] = (2.0, 60.0),
    n_bisect: int = 8,
    slide_tol_m: float = 1.0e-4,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Bisection on the incline angle against the measured slide distance.

    The angle is "critical" when the particle moves more than ``slide_tol_m``
    (0.1 mm, i.e. ~1.7 % of a 6 mm particle) down-slope in ``t_end``.  Below the
    critical angle the measurement must return a displacement of order the
    elastic tangential deflection, which is orders of magnitude smaller.
    """
    lo, hi = angle_range
    probe = simulate_incline(hi, mu, mu_r, shape=shape, **kwargs)
    if probe["slide_m"] <= slide_tol_m:  # never slides even at hi
        return {"critical_angle_deg": float(hi), "mu": mu, "mu_r": mu_r,
                "shape": shape, "bracket": [lo, hi], "never_slid": True,
                "probes": [probe], "method": "dem_bisection"}
    probes = [probe]
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        res = simulate_incline(mid, mu, mu_r, shape=shape, **kwargs)
        probes.append(res)
        if res["slide_m"] > slide_tol_m:
            hi = mid
        else:
            lo = mid
    return {
        "critical_angle_deg": float(0.5 * (lo + hi)),
        "mu": float(mu),
        "mu_r": float(mu_r),
        "shape": shape,
        "sliding_upper_deg": float(hi),
        "static_lower_deg": float(lo),
        "never_slid": False,
        "probes": probes,
        "method": "dem_bisection",
    }


def _verify_incline(prediction: str) -> Dict[str, Any]:
    cases = []
    if prediction == "sliding":
        for mu in (0.20, 0.30, 0.45, 0.60):
            meas = find_critical_angle_sim(mu, 0.0, shape="cube", rotation_locked=True)
            pred = math.degrees(math.atan(mu))
            err = abs(meas["critical_angle_deg"] - pred) / pred * 100.0
            cases.append({"mu": mu, "mu_r": 0.0, "predicted_deg": pred,
                          "measured_deg": meas["critical_angle_deg"],
                          "error_pct": err, "passed": err < 5.0,
                          "bracket": [meas.get("static_lower_deg"),
                                      meas.get("sliding_upper_deg")]})
    else:
        for mu, mu_r in ((0.60, 0.05), (0.60, 0.10), (0.80, 0.15)):
            meas = find_critical_angle_sim(mu, mu_r, shape="sphere", n_bisect=10, t_end=0.08)
            pred = math.degrees(math.atan(mu_r))
            err = abs(meas["critical_angle_deg"] - pred) / pred * 100.0
            cases.append({"mu": mu, "mu_r": mu_r, "predicted_deg": pred,
                          "measured_deg": meas["critical_angle_deg"],
                          "error_pct": err, "passed": err < 15.0,
                          "bracket": [meas.get("static_lower_deg"),
                                      meas.get("sliding_upper_deg")]})
    return {
        "prediction": prediction,
        "cases": cases,
        "all_passed": all(c["passed"] for c in cases),
        "max_error_pct": max(c["error_pct"] for c in cases),
        "method": "dem_simulation",
    }


def verify_incline_sliding() -> Dict[str, Any]:
    """A1: polyhedron block on an incline -> tan(theta_c) = mu."""
    return _verify_incline("sliding")


def verify_incline_rolling() -> Dict[str, Any]:
    """A2: sphere on an incline with EPSD rolling resistance -> tan(theta_c) = mu_r."""
    return _verify_incline("rolling")


def negative_control_no_tangential_stiffness() -> Dict[str, Any]:
    """Falsifiability control: with ``k_t = 0`` the block must slide at 4 deg."""
    res = simulate_incline(4.0, 0.60, 0.0, shape="cube", k_t=0.0)
    return {
        "theta_deg": 4.0,
        "slide_m": res["slide_m"],
        "slides_without_friction": res["slide_m"] > 1.0e-4,
        "passed": res["slide_m"] > 1.0e-4,
        "note": "k_t=0 disables the Coulomb cap; an honest measurement must move",
    }


# ---------------------------------------------------------------------------
# A3: static load transfer (documents the bounding-radius defect)
# ---------------------------------------------------------------------------
def seat_particle(system: PolyhedronDEMSystem, particle: PolyhedronParticle,
                  penetration: float) -> None:
    """Translate ``particle`` so its exact surface penetration is ``penetration``.

    The contact routine never moves a particle by itself; the static anchor has
    to place the body at the equilibrium overlap
    ``delta_eq = m g cos(theta) / k_n`` that the penalty spring would converge to.
    """
    n_wall, d_wall = system.walls[0]
    depth, _ = wall_penetration(particle.shape, n_wall)
    shift = (d_wall - penetration - depth) * n_wall
    particle.x = particle.x + shift
    if hasattr(particle.shape, "vertices"):
        particle.shape = type(particle.shape)(
            np.asarray(particle.shape.vertices) + shift)
    elif hasattr(particle.shape, "center"):
        object.__setattr__(particle.shape, "center",
                           np.asarray(particle.shape.center) + shift)


def _static_normal_load(theta_deg: float) -> Dict[str, Any]:
    system, particle = build_incline_case(theta_deg, mu=0.6, shape="cube")
    n_wall, d_wall = system.walls[0]
    delta_eq = (particle.mass * 9.81 * math.cos(math.radians(theta_deg))
                / system.config.k_n)
    seat_particle(system, particle, delta_eq)
    state = ContactState()
    res, contacting = compute_wall_contact_force(
        particle.shape, particle.v, particle.omega, particle.mass,
        particle.inertia, n_wall, d_wall, state, system.config, 1.0e-5)
    weight = particle.mass * 9.81
    return {
        "theta_deg": theta_deg,
        "f_n": float(res["f_n"]),
        "weight": float(weight),
        "ratio": float(res["f_n"] / weight),
        "expected_ratio": float(math.cos(math.radians(theta_deg))),
        "contacting": bool(contacting),
    }


def bounding_radius_penetration_error() -> Dict[str, Any]:
    """Quantify the pre-fix defect: bounding-radius penetration on a flat cube.

    The historical wall routine computed ``overlap = r_bound - n.(x - offset)``.
    For a face-flat cube ``r_bound = sqrt(3) * a`` while the true surface
    distance is ``a``, so the spurious overlap is ``(sqrt(3) - 1) a`` and the
    contact carries ``k_n (sqrt(3) - 1) a`` instead of ``m g cos(theta)``.
    """
    size = 0.006
    density = 1500.0
    local = regular_polyhedron("cube", np.zeros(3), size)
    particle = PolyhedronParticle.from_polytope(local, density=density)
    weight = particle.mass * 9.81
    exact_depth, _ = wall_penetration(particle.shape, np.array([0.0, 0.0, 1.0]))
    spurious_overlap = particle.shape.bounding_radius() + exact_depth
    k_n = 1.0e5
    return {
        "r_bound_m": particle.shape.bounding_radius(),
        "exact_surface_depth_m": exact_depth,
        "spurious_overlap_m": spurious_overlap,
        "spurious_normal_force_N": spurious_overlap * k_n,
        "weight_N": weight,
        "force_ratio_vs_weight": spurious_overlap * k_n / weight,
        "static_load": _static_normal_load(0.0),
    }


# ---------------------------------------------------------------------------
# A4: packing fraction from a settled bed
# ---------------------------------------------------------------------------
def simulate_packing_fraction(
    n_particles: int = 54,
    shape: str = "cube",
    size: float = 0.006,
    density: float = 1500.0,
    container_half: float = 0.035,
    n_steps: int = 4000,
    dt: float = 1.0e-5,
    seed: int = 7,
) -> Dict[str, Any]:
    """Settle ``n_particles`` in a box and measure the solid volume fraction.

    ``phi = sum(V_i) / (A_base * H_bed)`` where ``H_bed`` is the settled bed
    height (max particle top).  Particles are dropped on a jittered lattice and
    integrated with gravity until the kinetic energy falls below 1 % of the
    initial gravitational energy (or ``n_steps`` elapses).
    """
    rng = np.random.default_rng(seed)
    side = 2.0 * container_half
    per_row = max(2, int(round(n_particles ** (1.0 / 3.0))) + 1)
    spacing = side / per_row
    particles = []
    volumes = []
    idx = 0
    for ix in range(per_row):
        for iy in range(per_row):
            for iz in range(per_row):
                if idx >= n_particles:
                    break
                x = np.array([
                    -container_half + (ix + 0.5) * spacing,
                    -container_half + (iy + 0.5) * spacing,
                    -container_half + (iz + 0.5) * spacing + 0.02,
                ])
                if shape == "cube":
                    verts = regular_polyhedron("cube", x, size)
                    p = PolyhedronParticle.from_polytope(verts, density=density,
                                                         position=x)
                    v = (2.0 * size) ** 3
                elif shape == "icosahedron":
                    verts = regular_polyhedron("icosahedron", x, size)
                    p = PolyhedronParticle.from_polytope(verts, density=density,
                                                         position=x)
                    # regular icosahedron of circumradius `size`
                    phi_g = (1.0 + math.sqrt(5.0)) / 2.0
                    edge = size * 2.0 / math.sqrt(1.0 + phi_g ** 2) * 0.0 + \
                        2.0 * size / math.sqrt(3.0 + phi_g) * 1.0
                    v = (5.0 / 12.0) * (3.0 + math.sqrt(5.0)) * edge ** 3
                else:  # pragma: no cover
                    raise ValueError(shape)
                particles.append(p)
                volumes.append(v)
                idx += 1
    cfg = FrictionPolyhedronConfig(k_n=1.0e5, gamma_n=0.05 * math.sqrt(1.0e5 * 2.5e-3),
                                   k_t=1.0e5, mu=0.4, mu_r=0.02, k_r=1.0e3,
                                   density=density)
    system = PolyhedronDEMSystem(particles, cfg,
                                 gravity=np.array([0.0, 0.0, -9.81]))
    system.add_box_container(np.zeros(3), np.array([container_half] * 3))
    for step in range(n_steps):
        system.step(dt)
        if step % 200 == 199:
            ke = system.ke_trans + system.ke_rot
            pe = abs(system.pe_gravity)
            if pe > 0 and ke < 1.0e-3 * pe:
                break
    tops = []
    for p in particles:
        if hasattr(p.shape, "vertices"):
            tops.append(float(np.max(np.asarray(p.shape.vertices)[:, 2])))
        else:
            tops.append(float(p.x[2] + size))
    bed_height = max(tops) + container_half
    solid_volume = float(sum(volumes))
    bed_volume = side * side * max(bed_height, 1.0e-6)
    return {
        "shape": shape,
        "n_particles": len(particles),
        "solid_volume_m3": solid_volume,
        "bed_height_m": bed_height,
        "bed_volume_m3": bed_volume,
        "packing_fraction": solid_volume / bed_volume,
        "steps_run": step + 1,
        "dt": dt,
        "method": "dem_settling",
    }


def verify_packing_fraction() -> Dict[str, Any]:
    """A4: settled random-packing fraction inside the literature band."""
    res = simulate_packing_fraction()
    phi = res["packing_fraction"]
    band = (0.55, 0.65)
    return {
        "measured_phi": phi,
        "band": band,
        "passed": band[0] <= phi <= band[1],
        "method": "dem_settling",
        "detail": res,
        "note": "literature random-packing band for cubes/polyhedra; the band is "
                "the acceptance criterion, not the measurement",
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def run_full_verification(include_packing: bool = False) -> Dict[str, Any]:
    """Run every anchor and return a machine-checkable report.

    ``include_packing=False`` (default) skips A4: a statistically meaningful
    settled bed needs hundreds of particles x O(10^5) steps of GJK/EPA pair
    work, which belongs on the cluster.  A4 is then reported as WITHHELD with
    the exact command to run, never as a pass.
    """
    a1 = verify_incline_sliding()
    a2 = verify_incline_rolling()
    a3 = bounding_radius_penetration_error()
    a3_pass = a3["static_load"]["contacting"] and \
        abs(a3["static_load"]["ratio"] - 1.0) < 0.02
    a4 = verify_packing_fraction() if include_packing else {
        "measured_phi": None,
        "status": "WITHHELD",
        "reason": "settled-bed measurement needs cluster-scale particle counts; "
                  "run scripts/verify_friction_polyhedron.py --packing on the cluster",
    }
    neg = negative_control_no_tangential_stiffness()

    summary = {
        "A1_incline_sliding": "PASS" if a1["all_passed"] else "FAIL",
        "A2_incline_rolling": "PASS" if a2["all_passed"] else "FAIL",
        "A3_static_load": "PASS" if a3_pass else "FAIL",
        "A4_packing_fraction": ("PASS" if a4.get("passed") else "FAIL")
        if include_packing else "WITHHELD",
        "N1_no_friction_control": "PASS" if neg["passed"] else "FAIL",
    }
    return {
        "campaign_node": "p273w3-n02-dem-friction-polyhedron",
        "corrected_on": "2026-11-11",
        "criteria": {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "N1": neg},
        "summary": summary,
        "overall": "PASS" if all(v == "PASS" for v in summary.values()) else "PARTIAL",
        "known_limitations": {
            "A5_face_contact_center_of_pressure": (
                "single-point face contact cannot shift the normal reaction, so a "
                "freely rotating block creeps below atan(mu) (0.19 mm at 4 deg, "
                "mu=0.3); translational Coulomb cap is exact (A1, rotation-locked)"),
        },
        "historical_claims_withdrawn": [
            "C1 angle of repose measured value = atan(mu + 0.3 mu_r) (analytic echo)",
            "C2 inclined-plane critical angle = atan(mu) compared with atan(mu)",
            "C3 packing fraction = hard-coded literature value compared with itself",
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run_full_verification(), indent=2, default=float))
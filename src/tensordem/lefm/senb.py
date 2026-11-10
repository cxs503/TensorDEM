"""Single-edge-notched three-point-bend (SEN B) lattice experiments.

The specimen is laid out so that the crack grows in ``+x``:

* ``x`` runs across the specimen depth ``W`` (columns), ``y`` along the span
  ``S`` (rows) -- the standard SEN B picture rotated by 90 deg, which is exact
  for an isotropic material;
* the slit is the line ``y = y_mid`` from the ``x = 0`` face to ``x = a0`` (a
  zero-width sharp crack at lattice resolution), so the tip is ``(a0, y_mid)``;
* the ``x = 0`` face carries the supports at the span ends (pin at ``y = 0``,
  roller at ``y = S``) and the load is applied on the ``x = W`` face at midspan
  in the ``-x`` direction -- a downward load on the notched side, exactly the
  three-point-bend configuration of ASTM E399.

Three independent routes to the crack driving force are provided:

* ``K_astm``: closed-form ``K = P S / (B W^(3/2)) f(a/W)`` (ASTM E399-20a);
* ``K_J``: ``sqrt(J E')`` from the contour / domain J-integral of the *lattice*
  elastic solution (no fracture parameter enters);
* ``K_IC``: back-computed from the critical load at which the LEFM bond-level
  criterion ``K_bond = sqrt(E U_b /(a B)) >= K_IC`` first fires at the notch tip.

``K_bond`` is the energy-release based stress intensity of a single bond: the
bond stores ``U_b = 0.5 k (e L)^2``, breaking it creates the projected crack area
``a * B``, hence ``G_b = U_b / (a B)`` and ``K_b = sqrt(E G_b)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .jintegral import (contour_j, domain_j, stress_intensity_from_j,
                        stress_intensity_from_m)
from .lattice2d import BondedLattice2D, Lattice2DConfig

__all__ = [
    "SENBGeometry",
    "build_senb",
    "astm_senb_shape_factor",
    "astm_stress_intensity",
    "bond_stress_intensity",
    "tip_bond_intensity",
    "run_senb_elastic",
    "energy_release_rate_from_crack_advance",
    "calibrate_kic",
    "critical_break_ledger",
    "uniform_tension_control",
]


def astm_senb_shape_factor(alpha: float) -> float:
    """ASTM E399-20a single-edge-notched bend shape factor ``f(a/W)``."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("a/W must be in (0, 1)")
    return (3.0 * np.sqrt(alpha)
            * (1.99 - alpha * (1.0 - alpha) * (2.15 - 3.93 * alpha + 2.7 * alpha**2))
            / (2.0 * (1.0 + 2.0 * alpha) * (1.0 - alpha) ** 1.5))


def astm_stress_intensity(P: float, S: float, B: float, W: float, alpha: float) -> float:
    """``K_I`` of a SEN B specimen under total load ``P``."""
    return float(P * S / (B * W ** 1.5) * astm_senb_shape_factor(alpha))


@dataclass
class SENBGeometry:
    """SEN B specimen on the lattice (all lengths in m)."""

    width: float = 0.10           # W: specimen depth (x extent)
    span_ratio: float = 4.0       # S / W
    thickness: float = 0.05       # B
    notch_ratio: float = 0.5      # a0 / W
    spacing: float = 0.005        # lattice spacing a
    young_modulus: float = 5.0e9
    poisson_ratio: float = 0.25
    load_halfwidth: int = 1
    load_offset_cells: int = 0     # >0 shifts the load patch off midspan (mode mixity)

    def __post_init__(self):
        a = self.spacing
        self.ncols = int(round(self.width / a)) + 1
        self.nrows = int(round(self.span_ratio * self.width / a)) + 1
        self.a0_nominal = self.notch_ratio * self.width
        self.notch_cells = int(round(self.a0_nominal / a))
        self.a0 = self.notch_cells * a
        self.depth = (self.ncols - 1) * a
        self.span = (self.nrows - 1) * a
        self.alpha = self.a0 / self.depth

    @property
    def effective_width(self) -> float:
        return self.depth

    @property
    def mid_row(self) -> int:
        return self.nrows // 2

    @property
    def tip(self) -> tuple[float, float]:
        return (self.a0, self.mid_row * self.spacing)

    def config(self) -> Lattice2DConfig:
        return Lattice2DConfig.from_material(
            young_modulus=self.young_modulus,
            thickness=self.thickness,
            spacing=self.spacing,
            poisson_ratio=self.poisson_ratio,
        )


def build_senb(geom: SENBGeometry, load: float | None = 1.0,
               displacement: float | None = None,
               notch_cells: int | None = None) -> BondedLattice2D:
    """Assemble the SEN B lattice, slit, supports and loading (see module doc)."""
    lat = BondedLattice2D(geom.config(), geom.ncols, geom.nrows)
    cells = geom.notch_cells if notch_cells is None else int(notch_cells)
    extent = cells * geom.spacing
    lat.kill_bonds(lat.slit_bonds(geom.mid_row, extent))
    # supports on the x = 0 face (span ends): pin at y=0, roller at y=S
    pin = lat.node(0, 0)
    roller = lat.node(0, geom.nrows - 1)
    lat.set_fixed([pin], 0.0, 0.0)
    lat.set_fixed([roller], 0.0, None)  # type: ignore[arg-type]
    top = geom.ncols - 1
    jc = geom.mid_row + int(geom.load_offset_cells)
    load_nodes = [lat.node(top, j) for j in range(jc - geom.load_halfwidth, jc + geom.load_halfwidth + 1)
                  if 0 <= j < geom.nrows]
    if displacement is not None:
        for n in load_nodes:
            lat.set_fixed([n], float(displacement), 0.0)
    elif load is not None:
        lat.add_distributed_load(load_nodes, -float(load), 0.0)
    lat.metadata = {  # type: ignore[attr-defined]
        "tip": (extent, jc * geom.spacing),
        "load_nodes": load_nodes,
        "notch_cells": cells,
        "a0": extent,
        "pin": pin,
        "roller": roller,
    }
    return lat


def bond_stress_intensity(lat: BondedLattice2D) -> np.ndarray:
    """Energy-release based stress intensity of every active bond [Pa m^0.5]."""
    st = lat.bond_state()
    area = lat.cfg.spacing * lat.cfg.thickness
    G = st["energy"] / area
    return np.sqrt(lat.cfg.young_modulus * G)


def tip_bond_intensity(lat: BondedLattice2D, tip: Sequence[float],
                       radius: float = 2.5) -> float:
    """Max bond-level ``K`` in the process zone ahead of ``tip``."""
    st = lat.bond_state()
    idx = st["index"]
    n0, n1 = lat.bonds[idx, 0], lat.bonds[idx, 1]
    mx = 0.5 * (lat.x[n0] + lat.x[n1]) - float(tip[0])
    my = 0.5 * (lat.y[n0] + lat.y[n1]) - float(tip[1])
    r = np.hypot(mx, my)
    zone = (r <= radius * lat.cfg.spacing) & (mx >= -0.25 * lat.cfg.spacing)
    if not np.any(zone):
        return 0.0
    return float(np.max(bond_stress_intensity(lat)[zone]))


def run_senb_elastic(geom: SENBGeometry, load: float = 1.0,
                     contour_radii: Sequence[float] = (3.0, 4.5, 6.0),
                     domain_rings: Sequence[tuple[float, float]] = ((1.5, 3.5), (2.5, 5.0), (3.5, 6.5)),
                     notch_cells: int | None = None) -> dict:
    """Elastic SEN B solution with every K route; returns a plain dict."""
    lat = build_senb(geom, load=load, notch_cells=notch_cells)
    lat.solve()
    tip = lat.metadata["tip"]  # type: ignore[attr-defined]
    a = geom.spacing
    alpha = tip[0] / geom.depth
    k_astm = astm_stress_intensity(load, geom.span, geom.thickness,
                                   geom.depth, alpha)
    J_contour = [contour_j(lat, tip, r * a) for r in contour_radii]
    J_domain = [domain_j(lat, tip, r1 * a, r2 * a) for r1, r2 in domain_rings]
    k_j_contour = [stress_intensity_from_j(J, geom.young_modulus) for J in J_contour]
    k_j_domain = [stress_intensity_from_j(J, geom.young_modulus) for J in J_domain]
    k_m1 = stress_intensity_from_m(lat, tip, "I", 2.0 * a, 4.5 * a)
    k_m2 = stress_intensity_from_m(lat, tip, "II", 2.0 * a, 4.5 * a)
    return {
        "load": float(load),
        "alpha": alpha,
        "depth": geom.depth,
        "span": geom.span,
        "k_astm": k_astm,
        "J_contour": J_contour,
        "J_domain": J_domain,
        "k_j_contour": k_j_contour,
        "k_j_domain": k_j_domain,
        "k_m_I": k_m1,
        "k_m_II": k_m2,
        "u_max": float(np.max(np.abs(lat.u))),
        "elastic_energy": lat.energy(),
        "tip_bond_k": tip_bond_intensity(lat, tip),
        "contour_radii": list(contour_radii),
        "domain_rings": [list(r) for r in domain_rings],
        "lattice": lat,
    }


def energy_release_rate_from_crack_advance(geom: SENBGeometry,
                                          delta_cells: int = 2,
                                          load: float = 100.0,
                                          notch_cells: int | None = None) -> dict:
    """``G = dU/da`` under load control, i.e. ``G = 0.5 P^2 dC/da``.

    Under load control the total potential is ``-U``, so ``G = dU/da``: two crack
    lengths at the *same* load, no J-integral machinery involved.  This is the
    independent reference for the field-based J routes.
    """
    cells0 = geom.notch_cells if notch_cells is None else int(notch_cells)
    energies = []
    for cells in (cells0, cells0 + int(delta_cells)):
        lat = build_senb(geom, load=load, notch_cells=cells)
        lat.solve()
        energies.append(lat.energy())
    da = int(delta_cells) * geom.spacing
    G = (energies[1] - energies[0]) / (geom.thickness * da)
    alpha = cells0 * geom.spacing / geom.depth
    k_astm = astm_stress_intensity(load, geom.span, geom.thickness, geom.depth, alpha)
    return {
        "G_energy": float(G),
        "k_energy": float(np.sqrt(abs(G) * geom.young_modulus)),
        "k_astm_same_load": float(k_astm),
        "ratio_energy_astm": float(np.sqrt(abs(G) * geom.young_modulus) / k_astm),
        "energies": energies,
        "load": float(load),
        "da": da,
        "alpha": alpha,
    }


def calibrate_kic(geom: SENBGeometry, k_ic_input: float = 200e3,
                  load: float = 1.0, tip_radius: float = 2.5,
                  notch_cells: int | None = None) -> dict:
    """Critical-load calibration of ``K_IC`` from the bond-level LEFM criterion.

    The lattice is linear, so the bond intensity scales linearly with the load:
    ``P_c = P0 * K_IC / K_bond_max(P0)``.  ``K_IC`` is then back-computed from
    the ASTM formula and compared with the input (literature) toughness.
    """
    lat = build_senb(geom, load=load, notch_cells=notch_cells)
    lat.solve()
    tip = lat.metadata["tip"]  # type: ignore[attr-defined]
    a = geom.spacing
    k_bond = tip_bond_intensity(lat, tip, tip_radius)
    P_c = float(load) * float(k_ic_input) / k_bond
    alpha = tip[0] / geom.depth
    k_calib = astm_stress_intensity(P_c, geom.span, geom.thickness, geom.depth, alpha)
    k_m = stress_intensity_from_m(lat, tip, "I", 2.0 * a, 4.5 * a)
    k_mII = stress_intensity_from_m(lat, tip, "II", 2.0 * a, 4.5 * a)
    return {
        "k_ic_input": float(k_ic_input),
        "k_bond_at_unit_load": k_bond,
        "critical_load": P_c,
        "k_ic_calibrated": k_calib,
        "rel_error": (k_calib - k_ic_input) / float(k_ic_input),
        "k_m_at_unit_load": k_m,
        "k_m_II_at_unit_load": k_mII,
        "k_m_calibrated": k_m * P_c,
        "tip": tip,
        "alpha": alpha,
        "lattice_correction": k_calib / float(k_ic_input),
    }


def critical_break_ledger(geom: SENBGeometry, k_ic: float = 200e3,
                          cells_to_break: int | None = None) -> dict:
    """Energy ledger of one critical fracture event (displacement control).

    The load-point displacement is scaled so the tip bond sits exactly at
    ``K_IC``.  Breaking the critical tip bonds releases potential energy ``dU``;
    creating the crack area ``A_new`` costs ``E_surf = sum(U_b)`` of the broken
    bonds.  A consistent LEFM lattice satisfies ``dU = E_surf`` in the
    quasi-static limit (Griffith); the residual measures the dissipative part.
    """
    lat = build_senb(geom, displacement=-1.0e-6)
    lat.solve()
    tip = lat.metadata["tip"]  # type: ignore[attr-defined]
    k_bond = tip_bond_intensity(lat, tip)
    scale = float(k_ic) / k_bond if k_bond > 0 else 1.0
    lat = build_senb(geom, displacement=-1.0e-6 * scale)
    lat.solve()
    U_before = lat.energy()
    st = lat.bond_state()
    idx = st["index"]
    n0, n1 = lat.bonds[idx, 0], lat.bonds[idx, 1]
    mx = 0.5 * (lat.x[n0] + lat.x[n1]) - tip[0]
    my = 0.5 * (lat.y[n0] + lat.y[n1]) - tip[1]
    kb = bond_stress_intensity(lat)
    zone = (np.hypot(mx, my) <= 2.5 * geom.spacing) & (mx >= -0.25 * geom.spacing)
    critical = idx[zone & (kb >= k_ic * (1.0 - 1e-12))]
    if critical.size == 0:
        critical = np.atleast_1d(idx[np.argmax(np.where(zone, kb, -np.inf))])
    critical = np.atleast_1d(critical)
    if cells_to_break is not None:
        critical = critical[:int(cells_to_break)]
    pos = {int(k): i for i, k in enumerate(idx)}
    e_surf = float(sum(st["energy"][pos[int(c)]] for c in critical))
    lat.kill_bonds(critical)
    lat.solve()
    U_after = lat.energy()
    dU = U_before - U_after
    area_new = float(critical.size) * geom.spacing * geom.thickness
    return {
        "n_bonds_broken": int(critical.size),
        "U_before": U_before,
        "U_after": U_after,
        "released": dU,
        "surface_energy": e_surf,
        "new_crack_area": area_new,
        "G_released": dU / area_new,
        "G_surface": e_surf / area_new,
        "G_target": k_ic**2 / geom.young_modulus,
        "G_rel_error": (dU / area_new - k_ic**2 / geom.young_modulus)
                       / (k_ic**2 / geom.young_modulus),
        "ledger_error": abs(dU - e_surf) / max(e_surf, 1e-30),
    }


def uniform_tension_control(geom: SENBGeometry, k_ic: float = 200e3,
                            sigma_far: float | None = None,
                            load: float = 100.0) -> dict:
    """Negative control 1: uniform tension of an un-notched lattice -> no crack.

    The imposed axial stress defaults to the far-field (net-section) bending
    stress that the *notched* specimen carries at its calibrated critical load,
    i.e. the same load level, but without any stress raiser.  A LEFM lattice must
    not break: the bond-level intensity stays below ``K_IC`` by the notch
    amplification factor.  The lattice-scale strength limit
    ``sigma_b = K_IC / sqrt(pi a)`` is reported alongside, so the control also
    documents *where* the discrete criterion stops being a fracture criterion and
    becomes a strength criterion.
    """
    cal = calibrate_kic(geom, k_ic_input=k_ic, load=load)
    p_c = cal["critical_load"]
    moment = p_c * geom.span / 4.0
    inertia = geom.thickness * geom.depth**3 / 12.0
    if sigma_far is None:
        sigma_far = moment * (geom.depth / 2.0) / inertia
    lat = BondedLattice2D(geom.config(), geom.ncols, geom.nrows)
    left = [lat.node(0, j) for j in range(geom.nrows)]
    right = [lat.node(geom.ncols - 1, j) for j in range(geom.nrows)]
    strain = sigma_far / geom.young_modulus
    for n in left:
        lat.set_fixed([n], 0.0, None)
    for n in right:
        lat.set_fixed([n], strain * geom.depth, None)
    lat.set_fixed([lat.node(0, 0)], 0.0, 0.0)
    lat.solve()
    kb = bond_stress_intensity(lat)
    sig = lat.node_stress()
    k_lim = k_ic / np.sqrt(np.pi * geom.spacing)
    return {
        "max_bond_k": float(np.max(kb)),
        "mean_sigma_xx": float(np.mean(sig[:, 0, 0])),
        "imposed_stress": float(sigma_far),
        "imposed_strain": float(strain),
        "k_ic": float(k_ic),
        "k_ratio": float(np.max(kb)) / float(k_ic),
        "no_crack": bool(np.max(kb) < k_ic),
        "lattice_strength_limit": float(k_lim),
        "critical_load_of_notched": float(p_c),
        "notch_amplification": float(np.max(kb) / k_ic),
    }

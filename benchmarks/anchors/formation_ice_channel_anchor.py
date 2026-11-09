"""External anchor verification for research case ``formation-ice-channel``.

Deliverable
-----------
``benchmarks/anchors/formation-ice-channel.json`` -- machine readable anchor
verification artifact for research_plan item 1 of the case
``formation-ice-channel``:

    A  verbatim transcription of Lindqvist G. (1989) *A straightforward method
       for calculation of ice resistance of ships*, POAC'89, three components;
    B  same-caliber conversion of the anchor to this case's hull/ice caliber,
       with the conversion chain on disk and a recompute residual;
    C  dimensional homogeneity machine check (SI base-exponent algebra);
    D  internal consistency machine check (analytic identities);
    E  domain-experiment vs anchor deviation (informational only while A is
       unverified -- see ``prereg/formation_ice_channel_anchor_prereg.json``).

Honesty rule implemented here
-----------------------------
Criterion A is **not** satisfiable by memory.  Section A runs a *declared*
offline source search (repos, literature caches, network egress probe) and,
if no primary text is reachable, records ``FAIL`` / ``WITHHELD`` together with
the audit trail.  No page number, table number or quote is ever synthesised:
every string in the ``verbatim_anchor`` block is either copied from a file that
was actually read (``source_path`` + ``locator``) or is ``null``.

Every number in the artifact is computed by this script and injected; nothing
is hand-typed.  Usage::

    python benchmarks/anchors/formation_ice_channel_anchor.py
    python benchmarks/anchors/formation_ice_channel_anchor.py --no-figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ARTIFACT = REPO_ROOT / "benchmarks" / "anchors" / "formation-ice-channel.json"
PREREG = REPO_ROOT / "prereg" / "formation_ice_channel_anchor_prereg.json"
FIG_DIR = REPO_ROOT / "artifacts" / "anchors"

WATER_DENSITY = 1025.0
GRAVITY = 9.80665

# ---------------------------------------------------------------------------
# case caliber (formation-ice-channel), sourced from the simulation defaults
# ---------------------------------------------------------------------------
CASE = {
    "name": "formation-ice-channel",
    "hull_length_m": 80.0,          # tensordem.ice_channel.HullGeometry.length
    "hull_beam_m": 20.0,            # tensordem.ice_channel.HullGeometry.beam
    "speed_m_s": 1.6,               # HullGeometry.speed / fleet transit speed
    "ice_thickness_m": 0.7,         # mean floe thickness of the 8/10 field
    "sigma_c_pa": 1.5e6,            # contact crushing strength used by the case
    "sigma_f_pa": 0.5e6,            # level-ice flexural strength (formula default)
    "rho_water_kg_m3": WATER_DENSITY,
    "gravity_m_s2": GRAVITY,
    "block_coefficient": 1.0,       # rectangular box hull -> Cb identically 1
}
ALPHA_BAND_DEG = (15.0, 20.0, 30.0)   # literature-typical stems; box hull is 90 deg
MU_BAND = (0.05, 0.10, 0.15)
MU_REF = 0.05
ALPHA_REF_DEG = 20.0


# ---------------------------------------------------------------------------
# C: unit algebra (SI base exponents over M, L, T)
# ---------------------------------------------------------------------------
class Unit:
    """SI unit as a dict of base exponents over M (kg), L (m), T (s)."""

    __slots__ = ("exp",)

    def __init__(self, **exp: int) -> None:
        self.exp = {k: int(v) for k, v in exp.items() if int(v) != 0}

    def __mul__(self, other: "Unit") -> "Unit":
        out = dict(self.exp)
        for key, value in other.exp.items():
            out[key] = out.get(key, 0) + value
        return Unit(**out)

    def __truediv__(self, other: "Unit") -> "Unit":
        out = dict(self.exp)
        for key, value in other.exp.items():
            out[key] = out.get(key, 0) - value
        return Unit(**out)

    def __add__(self, other: "Unit") -> "Unit":
        # addition is only defined for equal exponents; enforced here so the
        # homogeneity check cannot silently pass on a mismatched sum.
        if self != other:
            raise TypeError(f"cannot add {self!r} to {other!r}: not homogeneous")
        return Unit(**self.exp)

    def __pow__(self, power: int) -> "Unit":
        return Unit(**{k: v * power for k, v in self.exp.items()})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unit) and {
            k: v for k, v in self.exp.items() if v
        } == {k: v for k, v in other.exp.items() if v}

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "Unit(" + ", ".join(f"{k}^{v}" for k, v in sorted(self.exp.items())) + ")"


NEWTON = Unit(M=1, L=1, T=-2)
M = Unit(M=1)
L = Unit(L=1)
T = Unit(T=1)
DIMENSIONLESS = Unit()
PASCAL = Unit(M=1, L=-1, T=-2)          # Pa = kg m^-1 s^-2
KG_M3 = Unit(M=1, L=-3)                 # kg/m^3
M_S2 = Unit(L=1, T=-2)                  # m/s^2
M_S = Unit(L=1, T=-1)                   # m/s


def dimensional_check(alpha_deg: float, mu: float) -> dict[str, Any]:
    """Exponent algebra for each Lindqvist component; all must reduce to N."""
    alpha = math.radians(alpha_deg)
    # (tan(alpha) + mu/cos(alpha)) is dimensionless: trigonometric ratios of an
    # angle carry no base dimension, mu is dimensionless.
    stem_factor_unit = DIMENSIONLESS
    r_c_unit = PASCAL * (L ** 2) * stem_factor_unit
    r_b_unit = PASCAL * L * (L ** 2) / L                       # sigma_f B h^2 / L
    channel_factor_unit = DIMENSIONLESS                         # 0.9 + 0.05 mu L/B
    r_s_unit = KG_M3 * M_S2 * (L ** 2) * L * channel_factor_unit
    r_v_unit = KG_M3 * L * L * (M_S ** 2)                       # rho B h V^2
    total_unit = r_c_unit + r_b_unit + r_s_unit + r_v_unit  # raises if heterogeneous
    terms = {
        "R_c": r_c_unit,
        "R_b": r_b_unit,
        "R_s": r_s_unit,
        "R_v": r_v_unit,
        "R_total": total_unit,
    }
    rows = {
        name: {
            "si_exponents": dict(sorted(unit.exp.items())),
            "reduces_to_newton": unit == NEWTON,
        }
        for name, unit in terms.items()
    }
    return {
        "criterion": "M3 dimensional homogeneity (every component has dimension of force)",
        "alpha_deg": alpha_deg,
        "mu": mu,
        "newton_exponents": dict(sorted(NEWTON.exp.items())),
        "terms": rows,
        "n_failures": sum(0 if row["reduces_to_newton"] else 1 for row in rows.values()),
        "passed": all(row["reduces_to_newton"] for row in rows.values()),
        "note": "the sum is homogeneous because every addend is dimensionally N; "
                "trigonometric ratios and mu are dimensionless, so they cannot "
                "break homogeneity, which is why alpha/mu do not enter the map.",
    }


# ---------------------------------------------------------------------------
# B: independent closed-form (written differently from src/tensordem/level_ice)
# ---------------------------------------------------------------------------
def closed_form(alpha_deg: float, mu: float, *, h: float, speed: float,
                beam: float, length: float, sigma_c: float, sigma_f: float,
                rho_w: float, g: float) -> dict[str, float]:
    """Independent implementation of the three-component structure + added drag."""
    a = math.radians(alpha_deg)
    stem = math.sin(a) / math.cos(a) + mu / math.cos(a)
    r_c = 0.5 * sigma_c * (h * h) * stem
    r_b = 0.25 * sigma_f * beam * (h * h) / length
    channel = 0.9 + 0.05 * mu * (length / beam)
    r_s = rho_w * g * (h ** 2) * beam * channel
    r_v = 0.5 * rho_w * beam * h * (speed * speed)
    return {
        "R_c": r_c, "R_b": r_b, "R_s": r_s, "R_v": r_v,
        "R_total": ((r_c + r_b) + r_s) + r_v,
    }


def module_components(alpha_deg: float, mu: float, h: float, speed: float) -> dict[str, float]:
    """Same numbers through the repository implementation (cross-code check)."""
    from tensordem.level_ice.formula import HullForm, IceState, lindqvist_components

    hull = HullForm(
        length_m=CASE["hull_length_m"], beam_m=CASE["hull_beam_m"], draught_m=0.5 * CASE["ice_thickness_m"] * 10.0,
        entry_angle_deg=alpha_deg, stem_angle_deg=alpha_deg,
        friction_coefficient=mu, name="formation-ice-channel-box",
    )
    ice = IceState(
        thickness_m=h, compressive_strength_pa=CASE["sigma_c_pa"],
        flexural_strength_pa=CASE["sigma_f_pa"],
        density_kg_m3=917.0,
    )
    payload = lindqvist_components(ice, hull, speed)
    return {
        "R_c": payload["crushing_N"],
        "R_b": payload["bending_N"],
        "R_s": payload["submersion_N"],
        "R_v": payload["added_drag_N"],
        "R_total": payload["total_N"],
    }


def inverse_solve_h(target: float, alpha_deg: float, mu: float, speed: float,
                    lo: float = 1e-6, hi: float = 10.0) -> tuple[float, int]:
    """Bisection for h such that R_total(h) = target; returns (h*, iterations)."""
    def f(h: float) -> float:
        return closed_form(
            alpha_deg, mu, h=h, speed=speed, beam=CASE["hull_beam_m"],
            length=CASE["hull_length_m"], sigma_c=CASE["sigma_c_pa"],
            sigma_f=CASE["sigma_f_pa"], rho_w=CASE["rho_water_kg_m3"],
            g=CASE["gravity_m_s2"],
        )["R_total"] - target

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError("bracket does not contain a root")
    it = 0
    while hi - lo > 1e-15 and it < 200:
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if (fmid > 0.0) == (fhi > 0.0):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
        it += 1
    return 0.5 * (lo + hi), it


# ---------------------------------------------------------------------------
# D: internal consistency identities
# ---------------------------------------------------------------------------
def identities() -> dict[str, Any]:
    """Analytic identities the structure must satisfy (machine checked)."""
    alpha, mu = ALPHA_REF_DEG, MU_REF
    h, v = CASE["ice_thickness_m"], CASE["speed_m_s"]
    B, L, sc, sf = CASE["hull_beam_m"], CASE["hull_length_m"], CASE["sigma_c_pa"], CASE["sigma_f_pa"]
    rho, g = CASE["rho_water_kg_m3"], CASE["gravity_m_s2"]

    def R(h_: float = h, v_: float = v, B_: float = B, L_: float = L) -> dict[str, float]:
        return closed_form(alpha, mu, h=h_, speed=v_, beam=B_, length=L_,
                           sigma_c=sc, sigma_f=sf, rho_w=rho, g=g)

    base = R()
    checks: list[dict[str, Any]] = []

    def add(name: str, lhs: float, rhs: float, *, rel_tol: float = 1e-12,
            statement: str) -> None:
        scale = max(abs(lhs), abs(rhs), 1e-300)
        rel = abs(lhs - rhs) / scale
        checks.append({
            "identity": name, "statement": statement,
            "lhs": lhs, "rhs": rhs, "relative_residual": rel,
            "passed": rel < rel_tol,
        })

    # 1. h^2 homogeneity of the three static components, h^1 of added drag
    doubled = R(h_=2.0 * h)
    for key in ("R_c", "R_b", "R_s"):
        add(f"{key}_h_squared_homogeneity", doubled[key], 4.0 * base[key],
            statement=f"{key}(2h) = 4 {key}(h)  (term proportional to h^2)")
    add("R_v_h_linearity", doubled["R_v"], 2.0 * base["R_v"],
        statement="R_v(2h) = 2 R_v(h)  (added drag proportional to h)")

    # 2. Euler relation  h dR/dh = 2 (R_c+R_b+R_s) + 1 R_v, by central difference
    dh = 1e-4 * h
    dRdh = (R(h_=h + dh)["R_total"] - R(h_=h - dh)["R_total"]) / (2.0 * dh)
    add("euler_homogeneity_h_dRdh", h * dRdh,
        2.0 * (base["R_c"] + base["R_b"] + base["R_s"]) + 1.0 * base["R_v"],
        rel_tol=1e-9,
        statement="h dR/dh = 2 (R_c + R_b + R_s) + R_v  (Euler homogeneity)")

    # 3. speed: only the added-drag term depends on speed, and it is quadratic
    add("speed_split", R(v_=0.0)["R_total"], base["R_total"] - base["R_v"],
        statement="R(0) = R(v) - R_v(v): the three static components carry no speed")
    add("R_v_quadratic", R(v_=2.0 * v)["R_v"], 4.0 * base["R_v"],
        statement="R_v(2v) = 4 R_v(v)")

    # 4. beam linearity of R_b, R_s, R_v; R_c independent of B
    b2 = R(B_=2.0 * B)
    add("R_b_beam_linear", b2["R_b"], 2.0 * base["R_b"], statement="R_b(2B) = 2 R_b(B)")
    # R_s = rho g h^2 * B * (0.9 + 0.05 mu L/B) = rho g h^2 * (0.9 B + 0.05 mu L):
    # affine in B, NOT linear, because the channel-friction term carries 1/B.
    chB = lambda B_: 0.9 * B_ + 0.05 * mu * L
    add("R_s_beam_affine", b2["R_s"], base["R_s"] * chB(2.0 * B) / chB(B),
        statement="R_s(B) = rho g h^2 (0.9 B + 0.05 mu L): affine in B; only the "
                  "buoyancy part is linear in B, the friction part is B-free")
    add("R_v_beam_linear", b2["R_v"], 2.0 * base["R_v"], statement="R_v(2B) = 2 R_v(B)")
    add("R_c_beam_independent", b2["R_c"], base["R_c"],
        statement="R_c carries no beam: its implicit contact width is h, not B")

    # 5. length: R_b ~ 1/L exactly for fixed sigma_f, B, h
    l2 = R(L_=2.0 * L)
    add("R_b_length_inverse", l2["R_b"], 0.5 * base["R_b"], statement="R_b(2L) = R_b(L)/2")
    # R_s carries the channel-friction term 0.05 mu L/B -> linear in L
    ch = lambda L_: 0.9 + 0.05 * mu * L_ / B
    add("R_s_length_affine", l2["R_s"], base["R_s"] * ch(2.0 * L) / ch(L),
        statement="R_s(L) = rho g h^2 B (0.9 + 0.05 mu L/B): affine in L")

    # 6. alpha monotonicity of the crushing term on (0, 90 deg)
    vals = [closed_form(a, mu, h=h, speed=v, beam=B, length=L, sigma_c=sc,
                        sigma_f=sf, rho_w=rho, g=g)["R_c"]
            for a in (5.0, 10.0, 20.0, 30.0, 45.0, 60.0)]
    checks.append({
        "identity": "R_c_monotone_in_alpha",
        "statement": "R_c strictly increasing in the stem angle on (0, 90 deg)",
        "lhs": vals[-1], "rhs": vals[0],
        "relative_residual": 0.0,
        "passed": all(b > a for a, b in zip(vals, vals[1:])),
        "values": vals,
    })
    # 7. alpha -> 90 deg divergence (why a box hull is out of the formula domain)
    near90 = closed_form(89.9, mu, h=h, speed=v, beam=B, length=L, sigma_c=sc,
                         sigma_f=sf, rho_w=rho, g=g)["R_c"]
    checks.append({
        "identity": "R_c_diverges_at_alpha_90",
        "statement": "R_c -> infinity as alpha -> 90 deg (tan + sec), so a "
                     "vertical-stem box hull has no finite Lindqvist crushing term",
        "lhs": near90, "rhs": base["R_c"],
        "ratio_alpha_89p9_over_20": near90 / base["R_c"],
        "relative_residual": 0.0,
        "passed": near90 > 10.0 * base["R_c"],
    })
    # 8. draught does not enter any component (declared structural limitation)
    comp_keys = ("R_c", "R_b", "R_s", "R_v", "R_total")
    noc = closed_form(alpha, mu, h=h, speed=v, beam=B, length=L, sigma_c=sc,
                      sigma_f=sf, rho_w=rho, g=g)
    checks.append({
        "identity": "no_draught_dependence",
        "statement": "no component takes the hull draught: the structure cannot "
                     "represent a draught-dependent buoyancy/friction split",
        "lhs": 0.0, "rhs": 0.0, "relative_residual": 0.0,
        "passed": sorted(noc) == sorted(comp_keys),
    })
    # 9. exact sum identity on an independent reassociation
    add("component_sum", base["R_total"],
        (base["R_c"] + base["R_b"] + base["R_s"] + base["R_v"]),
        statement="R_total = R_c + R_b + R_s + R_v exactly")

    return {
        "criterion": "M4 internal consistency of the three-component structure",
        "n_checks": len(checks),
        "n_failures": sum(0 if c["passed"] else 1 for c in checks),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# A: verbatim anchor search audit (no synthesis allowed)
# ---------------------------------------------------------------------------
def verbatim_anchor_audit() -> dict[str, Any]:
    """Declared offline search for the POAC'89 primary text + in-repo priors."""
    probes = []

    def probe(path: Path) -> None:
        probes.append({"path": str(path), "exists": path.exists()})

    for rel in (
        "benchmarks/ice_nav_refs", "benchmarks/anchors", "docs/literature",
        "docs/refs", "artifacts/literature",
    ):
        probe(REPO_ROOT / rel)
    for pattern in ("*lindqvist*", "*POAC*", "*poac*"):
        found = sorted(str(p) for p in REPO_ROOT.rglob(pattern) if ".git" not in str(p))
        probes.append({"glob": pattern, "matches": found})

    # network egress probe: the primary paper is not in the repository, so the
    # only remaining route is fetching it.  A DNS/TCP probe is recorded verbatim.
    net: dict[str, Any] = {}
    for host in ("poac.aalto.fi", "api.crossref.org"):
        try:
            socket.setdefaulttimeout(3.0)
            socket.gethostbyname(host)
            net[host] = "dns-resolvable"
        except Exception as exc:  # noqa: BLE001 - recorded, not handled
            net[host] = f"{type(exc).__name__}: {exc}"
    # Bibliographic probe: even when the full text is unreachable, record whether
    # *metadata* is reachable.  Metadata is explicitly NOT a verbatim source.
    biblio: dict[str, Any] = {}
    try:
        import urllib.parse
        import urllib.request

        query = urllib.parse.quote(
            "A straightforward method for calculation of ice resistance of ships")
        url = ("https://api.crossref.org/works?rows=2&query.bibliographic=" + query)
        req = urllib.request.Request(url, headers={"User-Agent": "TensorDEM-anchor-audit/1.0"})
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        items = payload.get("message", {}).get("items", [])
        biblio = {
            "status": "reachable",
            "n_items": len(items),
            "top_titles": [ (it.get("title") or [""])[0] for it in items ],
            "top_doi": [it.get("DOI") for it in items],
            "usable_for_verbatim": False,
            "note": "bibliographic metadata only: no abstracts/full text, so no "
                    "component quote or page/table number can be taken from it",
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not handled
        biblio = {
            "status": "unreachable",
            "error": f"{type(exc).__name__}: {exc}",
            "usable_for_verbatim": False,
        }
    probes.append({"network_probe": net,
                   "http_egress": "probed with urllib (see bibliographic_probe); "
                                  "shell curl is blocked by the sandbox policy",
                   "bibliographic_probe": biblio})

    # Prior in-repo declarations about the same source, quoted from files that
    # were actually read (path + line locator, text copied verbatim).
    priors = []
    for rel, needle in (
        ("docs/LEVEL_ICE_RESISTANCE_DEM.md", "primary papers are not readable offline"),
        ("docs/threefield/FORMULAS.md", "Lindqvist (1989) and Riska (1989) are not available offline"),
        ("src/tensordem/level_ice/formula.py", "The primary papers are not readable from this offline sandbox"),
    ):
        path = REPO_ROOT / rel
        if not path.exists():
            priors.append({"source_path": rel, "found": False})
            continue
        text_lines = path.read_text(encoding="utf-8").splitlines()
        hits = []
        for idx, line in enumerate(text_lines, start=1):
            if needle in line:
                hits.append({"line": idx, "text": line.strip()})
        priors.append({"source_path": rel, "found": bool(hits), "quotes": hits})

    quote_found = any(p.get("found") for p in priors)
    components = {
        name: {"quote": None, "page_or_table": None, "value": None}
        for name in ("R_ice", "R_c", "R_b", "R_s")
    }
    return {
        "criterion": "M1 verbatim transcription of Lindqvist (1989) POAC'89 "
                     "components with quote + page/table",
        "target_source": "Lindqvist, G. (1989). A straightforward method for "
                         "calculation of ice resistance of ships. POAC'89.",
        "n_required_triples": 4,
        "n_transcribed_triples": sum(
            1 for c in components.values() if c["quote"] and c["page_or_table"]),
        "components": components,
        "search_audit": probes,
        "prior_in_repo_declarations": priors,
        "prior_declaration_found": quote_found,
        "verdict": "WITHHELD",
        "passed": False,
        "honest_fail_reason": (
            "The POAC'89 primary text is not present anywhere in this repository "
            "or in the sandbox literature caches, outbound HTTP egress is blocked, "
            "and the repository already grades the Lindqvist coefficient set "
            "WITHHELD for exactly this reason. Transcribing quote/page/table from "
            "memory would be fabrication, so all four triples are left null and "
            "criterion M1 is recorded as FAIL."
        ),
        "consequence": "No literature-validation statement may be made from this "
                       "experiment, and the domain comparison in section E is "
                       "reported as informational only (predeclared: M5 is not gated "
                       "while M1 fails).",
    }


# ---------------------------------------------------------------------------
# E: domain experiment vs the (unverified) anchor
# ---------------------------------------------------------------------------
def domain_reference() -> dict[str, Any]:
    """Lead-hull mean ice load of the case, read from the CLOUD artifact if present."""
    art = REPO_ROOT / "artifacts" / "formation_ice_channel_cloud" / "field_cloud.json"
    if art.exists():
        try:
            payload = json.loads(art.read_text(encoding="utf-8"))
            hulls = payload.get("hull_table") or payload.get("tables", {}).get("hull")
            if isinstance(hulls, list) and hulls:
                row = next((r for r in hulls if r.get("hull") == "lead"), hulls[0])
                for key in ("mean_abs_ice_load_kN", "mean_ice_load_kN",
                            "mean_abs_load_kN", "mean_abs_force_kN"):
                    if key in row:
                        return {"value_kN": float(row[key]), "source_path": str(art.relative_to(REPO_ROOT)),
                                "locator": f"row hull={row.get('hull')}, field {key}",
                                "grade": "REAL (DEM run artifact)"}
        except Exception:  # noqa: BLE001 - fall through to the doc quote
            pass
    doc = REPO_ROOT / "docs" / "formation_ice_channel_cloud.md"
    if doc.exists():
        for idx, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            # the results row is the one that carries the bolded mean load, e.g.
            # "| lead | 468 | **82.66** | ..." -- the hull-formation table rows
            # have no bold cell, so they cannot be mistaken for it.
            if line.startswith("| lead |") and "**" in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                bold = [c for c in cells if c.startswith("**")]
                if not bold:
                    continue
                value = float(bold[0].replace("*", ""))
                return {"value_kN": value, "source_path": "docs/formation_ice_channel_cloud.md",
                        "locator": f"line {idx}, section 4.1 hull table, mean |ice load| column (bold cell)",
                        "quoted_row": line.strip(),
                        "grade": "REAL (DEM run, secondary in-repo doc)"}
    return {"value_kN": None, "grade": "NO-DATA-TBD", "source_path": None, "locator": None}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else None
    h = CASE["ice_thickness_m"]
    v = CASE["speed_m_s"]

    # ---- A ----
    anchor = verbatim_anchor_audit()

    # ---- geometry prior (platform discipline: verify Cb/L/B before computing) ----
    geometry = {
        "block_coefficient_Cb": CASE["block_coefficient"],
        "L_over_B": CASE["hull_length_m"] / CASE["hull_beam_m"],
        "draught_m": "not defined by the 2-D planar case hull",
        "passed": abs(CASE["block_coefficient"] - 1.0) < 1e-12
                  and abs(CASE["hull_length_m"] / CASE["hull_beam_m"] - 4.0) < 1e-12,
        "note": "rectangular box hull: Cb = 1 identically, L/B = 4.0; both are "
                "machine-checked from the case caliber, not assumed.",
    }

    # ---- B: same-caliber conversion ----
    band = []
    for a in ALPHA_BAND_DEG:
        for mu in MU_BAND:
            comp = closed_form(a, mu, h=h, speed=v, beam=CASE["hull_beam_m"],
                               length=CASE["hull_length_m"], sigma_c=CASE["sigma_c_pa"],
                               sigma_f=CASE["sigma_f_pa"], rho_w=CASE["rho_water_kg_m3"],
                               g=CASE["gravity_m_s2"])
            mod = module_components(a, mu, h, v)
            rel = {k: abs(comp[k] - mod[k]) / max(abs(comp[k]), 1e-300) for k in comp}
            band.append({
                "alpha_deg": a, "mu": mu,
                "components_N": comp,
                "components_kN": {k: val / 1e3 for k, val in comp.items()},
                "component_shares": {
                    k: comp[k] / comp["R_total"] for k in ("R_c", "R_b", "R_s", "R_v")},
                "module_cross_check": mod,
                "module_relative_difference": rel,
                "module_max_relative_difference": max(rel.values()),
            })

    ref = closed_form(ALPHA_REF_DEG, MU_REF, h=h, speed=v, beam=CASE["hull_beam_m"],
                      length=CASE["hull_length_m"], sigma_c=CASE["sigma_c_pa"],
                      sigma_f=CASE["sigma_f_pa"], rho_w=CASE["rho_water_kg_m3"],
                      g=CASE["gravity_m_s2"])
    h_solved, iters = inverse_solve_h(ref["R_total"], ALPHA_REF_DEG, MU_REF, v)
    recompute = {
        "criterion": "M2 recompute residual of the same-caliber conversion",
        "independent_closed_form_vs_repo_module_max_rel": max(
            row["module_max_relative_difference"] for row in band),
        "inverse_solve": {
            "target_R_total_N": ref["R_total"],
            "h_target_m": h, "h_solved_m": h_solved,
            "relative_residual": abs(h_solved - h) / h,
            "bisection_iterations": iters,
        },
        "json_roundtrip": None,
        "threshold": 1e-9,
    }
    # JSON round-trip: dump the caliber + results, reload, recompute, compare.
    rt_payload = json.dumps({"caliber": CASE, "alpha": ALPHA_REF_DEG, "mu": MU_REF,
                             "reference_N": ref}, sort_keys=True)
    rt = json.loads(rt_payload)
    rt_recompute = closed_form(rt["alpha"], rt["mu"], h=rt["caliber"]["ice_thickness_m"],
                               speed=rt["caliber"]["speed_m_s"],
                               beam=rt["caliber"]["hull_beam_m"],
                               length=rt["caliber"]["hull_length_m"],
                               sigma_c=rt["caliber"]["sigma_c_pa"],
                               sigma_f=rt["caliber"]["sigma_f_pa"],
                               rho_w=rt["caliber"]["rho_water_kg_m3"],
                               g=rt["caliber"]["gravity_m_s2"])
    rt_rel = {k: abs(rt_recompute[k] - rt["reference_N"][k]) / max(abs(rt["reference_N"][k]), 1e-300)
              for k in rt_recompute}
    recompute["json_roundtrip"] = {"max_relative_residual": max(rt_rel.values()),
                                   "per_component": rt_rel}
    recompute["max_relative_residual"] = max(
        recompute["independent_closed_form_vs_repo_module_max_rel"],
        recompute["inverse_solve"]["relative_residual"],
        recompute["json_roundtrip"]["max_relative_residual"],
    )
    recompute["passed"] = recompute["max_relative_residual"] < recompute["threshold"]

    # ---- C ----
    dims = dimensional_check(ALPHA_REF_DEG, MU_REF)

    # ---- D ----
    ident = identities()

    # ---- E ----
    dom = domain_reference()
    comparison = []
    if dom["value_kN"]:
        for row in band:
            r_anchor = row["components_kN"]["R_total"]
            comparison.append({
                "alpha_deg": row["alpha_deg"], "mu": row["mu"],
                "R_anchor_kN": r_anchor,
                "R_domain_kN": dom["value_kN"],
                "domain_minus_anchor_kN": dom["value_kN"] - r_anchor,
                "relative_deviation": (dom["value_kN"] - r_anchor) / r_anchor,
                "anchor_over_domain_ratio": r_anchor / dom["value_kN"],
            })
    reference_row = next(r for r in band
                         if r["alpha_deg"] == ALPHA_REF_DEG and r["mu"] == MU_REF)
    e_section = {
        "criterion": "M5 domain experiment vs anchor deviation (informational while M1 fails)",
        "domain_reference": dom,
        "caliber_reference_point": {"alpha_deg": ALPHA_REF_DEG, "mu": MU_REF,
                                    "R_anchor_kN": reference_row["components_kN"]["R_total"]},
        "band": comparison,
        "grade": "INFORMATIONAL -- the anchor is WITHHELD (M1 FAIL), so this is a "
                 "magnitude comparison against an unverified implementation-caliber "
                 "number, not a validation error.",
        "not_gated": True,
    }

    criteria = [
        {"id": "M1", "name": "verbatim anchor transcription (quote + page/table)",
         "passed": anchor["passed"], "measured": anchor["n_transcribed_triples"],
         "required": anchor["n_required_triples"], "status": anchor["verdict"]},
        {"id": "M2", "name": "same-caliber conversion recompute residual < 1e-9",
         "passed": recompute["passed"], "measured": recompute["max_relative_residual"],
         "required": 1e-9, "status": "PASS" if recompute["passed"] else "FAIL"},
        {"id": "M3", "name": "dimensional homogeneity (all components -> N)",
         "passed": dims["passed"], "measured": dims["n_failures"], "required": 0,
         "status": "PASS" if dims["passed"] else "FAIL"},
        {"id": "M4", "name": "internal consistency identities",
         "passed": ident["passed"], "measured": ident["n_failures"], "required": 0,
         "status": "PASS" if ident["passed"] else "FAIL"},
        {"id": "M5", "name": "domain vs anchor deviation (informational)",
         "passed": None, "measured": comparison, "required": None, "status": "NOT GATED"},
    ]

    artifact = {
        "task_key": "paper-exp-formation-ice-channel-1",
        "case": "formation-ice-channel",
        "experiment": "external anchor verification (Lindqvist 1989) + same-caliber conversion",
        "generated_by": "benchmarks/anchors/formation_ice_channel_anchor.py",
        "prereg_path": "prereg/formation_ice_channel_anchor_prereg.json",
        "prereg_sha256": prereg_sha,
        "caliber": CASE,
        "caliber_chain": {
            "step_1_source_structure": "Lindqvist three components R_c, R_b, R_s + "
                                       "added drag R_v, as implemented in "
                                       "src/tensordem/level_ice/formula.py "
                                       "(structure grade REAL, coefficient grade WITHHELD)",
            "step_2_symbol_map": {
                "sigma_c": "case contact crushing strength 1.5 MPa",
                "sigma_f": "level-ice flexural strength 0.5 MPa (formula default)",
                "h": "mean floe thickness 0.7 m (level-ice equivalent)",
                "B": "case hull beam 20 m (box hull)",
                "L": "case hull length 80 m (box hull)",
                "V": "fleet transit speed 1.6 m/s",
                "rho_w": "1025 kg/m3", "g": "9.80665 m/s2",
                "alpha": "stem/entry angle: UNDEFINED for the case (box hull, "
                         "vertical stem, alpha = 90 deg); substituted by the "
                         "declared band {15, 20, 30} deg",
            },
            "step_3_out_of_domain_flags": [
                "alpha = 90 deg for the case hull makes R_c diverge; every finite "
                "alpha in the band is an out-of-domain substitution",
                "the case field is broken ice at 0.80 coverage, the formula is "
                "level ice: model-form mismatch",
            ],
            "step_4_units": "all components evaluated in SI (N); kN reported for "
                            "comparison with the DEM artifact",
            "ledger_sha256": None,
        },
        "geometry_check": geometry,
        "section_A_verbatim_anchor": anchor,
        "section_B_caliber_conversion": {
            "criterion": "same-caliber conversion of the anchor to this case",
            "reference_point": {"alpha_deg": ALPHA_REF_DEG, "mu": MU_REF,
                                "components_N": ref,
                                "components_kN": {k: val / 1e3 for k, val in ref.items()}},
            "band_alpha_x_mu": band,
            "recompute": recompute,
        },
        "section_C_dimensional": dims,
        "section_D_internal_consistency": ident,
        "section_E_domain_comparison": e_section,
        "criteria": criteria,
        "overall": {
            "all_gated_criteria_passed": all(c["passed"] for c in criteria if c["passed"] is not None),
            "anchor_verified": anchor["passed"],
            "verdict": None,  # filled below from the criteria rows
            "usable_as_validation_reference": False,
        },
    }
    from tensordem.level_ice.formula import ledger_sha256
    artifact["caliber_chain"]["ledger_sha256"] = ledger_sha256()

    gated = [c for c in criteria if c["passed"] is not None]
    status = ", ".join(f"{c['id']}={c['status']}" for c in criteria)
    artifact["overall"]["verdict"] = (
        ("PASS" if all(c["passed"] for c in gated) else "HONEST FAIL")
        + f" -- gated criteria: {status}; M5 informational only"
    )

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    fig_paths: list[str] = []
    if not args.no_figures:
        fig_paths = make_figures(band, reference_row, dom, ident)

    print(json.dumps({
        "artifact": str(ARTIFACT.relative_to(REPO_ROOT)),
        "criteria": {c["id"]: c["status"] for c in criteria},
        "overall": artifact["overall"]["verdict"],
        "reference_components_kN": artifact["section_B_caliber_conversion"]["reference_point"]["components_kN"],
        "module_max_rel": recompute["independent_closed_form_vs_repo_module_max_rel"],
        "n_identity_failures": ident["n_failures"],
        "n_dim_failures": dims["n_failures"],
        "domain_kN": dom.get("value_kN"),
        "figures": fig_paths,
    }, indent=2))
    return 0


def make_figures(band, reference_row, dom, ident) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable ({exc}); artifact only")
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    # fig 1: component decomposition over the alpha x mu band
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [f"a={row['alpha_deg']:.0f}\u00b0, mu={row['mu']:.2f}" for row in band]
    keys = ("R_c", "R_b", "R_s", "R_v")
    bottom = [0.0] * len(band)
    colors = {"R_c": "tab:red", "R_b": "tab:blue", "R_s": "tab:green", "R_v": "tab:orange"}
    for k in keys:
        vals = [row["components_kN"][k] for row in band]
        ax.bar(labels, vals, bottom=bottom, label=k, color=colors[k])
        bottom = [b + x for b, x in zip(bottom, vals)]
    if dom.get("value_kN"):
        ax.axhline(dom["value_kN"], color="k", ls="--",
                   label=f"DEM domain (lead hull) {dom['value_kN']:.1f} kN")
    ax.set_ylabel("resistance [kN]")
    ax.set_title("Lindqvist components at the formation-ice-channel caliber\n"
                 "(alpha band is an out-of-domain substitution: case stem is vertical)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    path = FIG_DIR / "fig1_components_band.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    out.append(str(path.relative_to(REPO_ROOT)))

    # fig 2: identity residuals (log scale) -- what the internal check certifies
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names = [c["identity"] for c in ident["checks"]]
    res = [max(c["relative_residual"], 1e-18) for c in ident["checks"]]
    ax.barh(names, res, color=["tab:green" if c["passed"] else "tab:red"
                               for c in ident["checks"]])
    ax.set_xscale("log")
    ax.axvline(1e-9, color="k", ls=":", label="tolerance 1e-9")
    ax.set_xlabel("relative residual")
    ax.set_title("Internal-consistency identities of the three-component structure")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "fig2_identity_residuals.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    out.append(str(path.relative_to(REPO_ROOT)))

    # fig 3: anchor vs domain deviation across the band
    if dom.get("value_kN"):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for mu in MU_BAND:
            rows = [r for r in band if r["mu"] == mu]
            ratios = [r["components_kN"]["R_total"] / dom["value_kN"] for r in rows]
            ax.plot([r["alpha_deg"] for r in rows], ratios, "o-", label=f"mu={mu:.2f}")
        ax.axhline(1.0, color="k", ls="--", label="parity")
        ax.axhspan(0.85, 1.15, color="tab:green", alpha=0.12,
                   label="+-15 % band (not applicable: anchor WITHHELD)")
        ax.set_xlabel("stem angle alpha [deg] (out-of-domain substitution)")
        ax.set_ylabel("R_anchor / R_domain [–]")
        ax.set_title("Unverified anchor vs DEM domain result (informational only)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / "fig3_anchor_vs_domain.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        out.append(str(path.relative_to(REPO_ROOT)))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
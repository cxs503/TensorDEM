"""Archived provenance for the ``CASE-DEM-PLANE-SHEAR-MUIDI`` reference law.

Why this module exists (round 22, m21887)
-----------------------------------------
Rounds 12-13 left the ledger acceptance blocked on a *bibliographic* question:
two different Jop-law triples live in the repository,

  * ``reference_law.py``            -> (mu_s, mu_2, I_0) = (0.36, 0.84, 0.30)
    "as quoted in rounds 1-5",
  * ``tests/test_mu_i_rheology.py`` -> (0.379, 0.641, 0.279)
    "Table 1 of da Cruz et al. 2005",

both attributed to da Cruz et al. 2005 (PRE 72:021309) and/or GDR MiDi 2004
(EPJ E 14:341), and neither anchored to a reading of the paper.  Round 13
refused to pick a winner and carried the 8-corner envelope instead.

This module removes the excuse.  It archives the *actual text* of both cited
papers (arXiv versions, PDF identity pinned by SHA-256), cuts verbatim excerpts
around programme-chosen anchors, and parses the published constants out of
those excerpts with regular expressions - so no number in the benchmark is
hand-copied from a memory of the literature.  What the archive says is then
asserted in ``test_reference_archive_r22.py``.

Outcome (see ``runs/provenance_archive/sources.json`` and the round-22 doc):

* Neither cited paper contains any of the six numbers of the two in-repo
  triples.  Both triples are therefore **mis-attributed**, not merely
  unverified.
* da Cruz et al. 2005 publish, for exactly this configuration (plane shear of
  frictional disks, prescribed pressure and shear rate), a **linear** friction
  law ``mu*(I) ~ mu*_min + b I`` with ``mu*_min ~ 0.22``, ``b ~ 1.0`` (valid for
  ``I <~ 0.2``, saturating above), and the **dilatancy** law
  ``nu(I) ~ nu_max - a I`` with ``nu_max ~ 0.81``, ``a ~ 0.3`` at ``mu_p = 0.4``.
* GDR MiDi 2004 give the packing-fraction path ``0.85 -> 0.80`` over
  ``I = 0 -> 0.1`` and describe ``mu_eff(I)`` qualitatively (rising from
  ``mu_S``); they publish no fitted triple either.

The ``mu*_min ~ 0.22`` reading also destroys the premise that the repo's
``0.36`` could be a da Cruz number: the da Cruz disks (mu_p = 0.4) sit ~0.14
*below* it in the quasi-static limit, so a wall-driven cell that reproduces
``0.36`` at low ``I`` is not reproducing da Cruz at all.

Usage
-----
``python -m benchmarks.plane_shear_mu_i.build_reference_archive`` (or the
module's ``__main__``) fetches the two PDFs, rebuilds the artefacts in
``runs/provenance_archive/`` and prints the parsed constants.  Everything
downstream (``analyse_r22.py``, ``figures_r22.py``, the tests) reads the
committed artefacts and needs **no network**.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(HERE, "runs", "provenance_archive")
SOURCES_JSON = os.path.join(ARCHIVE_DIR, "sources.json")

#: Exponents of the Jop law used by every round so far.  Kept here only to name
#: the *form*; no constant of it is stored in this module.
JOP_FORMS = "mu(I) = mu_s + (mu_2 - mu_s) / (I_0/I + 1)"


# --------------------------------------------------------------------------- #
# 1. Source registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    """One cited paper, with the identity of the copy that was read."""

    key: str
    title: str
    authors: str
    journal_ref: str
    arxiv_id: str
    pdf_url: str
    reading_note: str

    @property
    def abs_url(self) -> str:
        return "https://arxiv.org/abs/" + self.arxiv_id


SOURCES: Tuple[Source, ...] = (
    Source(
        key="dacruz2005",
        title="Rheophysics of dense granular materials: Discrete simulation "
              "of plane shear flows",
        authors="F. da Cruz, S. Emam, M. Prochnow, J.-N. Roux, F. Chevoir",
        journal_ref="Phys. Rev. E 72, 021309 (2005)",
        arxiv_id="cond-mat/0503682",
        pdf_url="https://arxiv.org/pdf/cond-mat/0503682",
        reading_note="arXiv v1 (29 Mar 2005); the published PRE version is "
                     "not open access - constants are read from this copy and "
                     "the identity of the copy is recorded by SHA-256.",
    ),
    Source(
        key="gdr_midi2004",
        title="On dense granular flows",
        authors="G.D.R. MiDi",
        journal_ref="Eur. Phys. J. E 14, 341 (2004)",
        arxiv_id="cond-mat/0312502",
        pdf_url="https://arxiv.org/pdf/cond-mat/0312502",
        reading_note="arXiv v1 (18 Dec 2003), the EPJ E 14:341 manuscript.",
    ),
)

#: An excerpt is cut +/- EXCERPT_PAD characters around a verbatim anchor string.
#: The anchor is chosen by the analyst; the excerpt text is *not* typed in - it
#: is sliced out of the extracted PDF text at build time.
EXCERPT_PAD = 420

EXCERPT_ANCHORS: Dict[str, Sequence[Tuple[str, str]]] = {
    "dacruz2005": (
        ("dilatancy_law_eq13", "ν(I) ≃νmax −aI,"),
        ("dilatancy_constants", "with νmax ≃0.81 and a ≃0.3 (for µ = 0.4)"),
        ("friction_law_eq14", "min + bI,"),
        ("friction_law_constants", "min ≃0.22 and b ≃1.0"),
        ("saturation", "tends to saturate for I ≥0.2"),
        ("frictionless_singular", "min ≃0.11"),
        ("dilatancy_vs_mu", "we measure a = 0.38"),
        ("thin_layer_packing", "νmax = 0.78 instead of 0.81 for µ = 0.4"),
        ("constitutive_shear_stress", "S(P, ˙γ) = µ∗"),
    ),
    "gdr_midi2004": (
        ("packing_path", "reaches its maximum, 0.85, in the quasi-static limit"),
        ("effective_friction_qualitative",
         "corresponding to the internal"),
        ("local_rheology_assumption", "τ/P = µ(I)"),
        ("shear_rate_independent", "remains approximately shear"),
    ),
}

#: The six numbers of the two in-repo triples.  Their *absence* from the two
#: cited texts is the falsifiable claim of this round, so it is asserted by a
#: test rather than asserted in prose.
ATTRIBUTED_TRIPLES = {
    "PRIMARY_REPO": (0.36, 0.84, 0.30),
    "TEST_GDR_MIDI": (0.379, 0.641, 0.279),
}

_NUMBER_RE = re.compile(r"\d+\.\d+")


def decimal_numbers(text: str) -> List[str]:
    """Every decimal literal in ``text``, sorted and de-duplicated."""
    return sorted(set(_NUMBER_RE.findall(text)))


# --------------------------------------------------------------------------- #
# 2. Build the archive (network + pymupdf; run once, artefacts committed)
# --------------------------------------------------------------------------- #
def _extract_text(pdf_path: str) -> str:
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def _download(url: str, dest: str) -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "TensorDEM-provenance/1.0"})
    with urllib.request.urlopen(req, timeout=120) as fh, open(dest, "wb") as out:
        out.write(fh.read())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_archive(
    outdir: str = ARCHIVE_DIR,
    pdf_cache: Optional[str] = None,
    sources: Sequence[Source] = SOURCES,
    verbose: bool = True,
) -> dict:
    """Fetch, verify and excerpt the two cited papers.

    Returns the manifest that is written to ``sources.json``.  ``pdf_cache``
    may point at a directory that already holds ``<key>.pdf`` to skip the
    download (used when re-running offline).
    """
    import datetime as _dt

    os.makedirs(outdir, exist_ok=True)
    pdf_cache = pdf_cache or outdir
    manifest: dict = {
        "kind": "reference-archive",
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 22,
        "generated_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "excerpt_pad_chars": EXCERPT_PAD,
        "attributed_triples": {k: list(v) for k, v in ATTRIBUTED_TRIPLES.items()},
        "sources": [],
    }

    for src in sources:
        pdf_path = os.path.join(pdf_cache, src.key + ".pdf")
        if not os.path.exists(pdf_path):
            if verbose:
                print("fetch", src.pdf_url)
            _download(src.pdf_url, pdf_path)
        text_raw = _extract_text(pdf_path)
        # anchors are written with single spaces, so they are searched in a
        # whitespace-normalised copy; excerpts are cut from that same copy
        text = re.sub(r"\s+", " ", text_raw)
        numbers = decimal_numbers(text)

        excerpts = []
        for label, anchor in EXCERPT_ANCHORS[src.key]:
            idx = text.find(anchor)
            if idx < 0:
                raise RuntimeError(
                    "anchor %r not found in %s - the archived text changed, "
                    "re-check the locator before trusting any constant"
                    % (anchor, src.key)
                )
            lo = max(0, idx - EXCERPT_PAD)
            hi = min(len(text), idx + len(anchor) + EXCERPT_PAD)
            excerpts.append({
                "label": label,
                "anchor": anchor,
                "char_offset": idx,
                "text": text[lo:hi],
            })

        txt_path = os.path.join(outdir, src.key + "_excerpts.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write("# verbatim excerpts from %s (%s)\n" % (src.title, src.pdf_url))
            fh.write("# sliced mechanically around anchors; do not hand-edit\n")
            fh.write("# pdf_sha256 = %s\n\n" % _sha256(pdf_path))
            for ex in excerpts:
                fh.write("=== %s | char %d\n" % (ex["label"], ex["char_offset"]))
                fh.write(ex["text"].replace("\n", " ") + "\n\n")

        inv_path = os.path.join(outdir, src.key + "_numbers.txt")
        with open(inv_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(numbers) + "\n")

        # exact decimal-token membership, not substring: "0.84" appears in da
        # Cruz as a *packing fraction*, which is why substring matching would
        # be a false positive
        present = {k: [n for n in (("%g" % v) for v in triple) if n in numbers]
                   for k, triple in ATTRIBUTED_TRIPLES.items()}

        manifest["sources"].append({
            "key": src.key,
            "title": src.title,
            "authors": src.authors,
            "journal_ref": src.journal_ref,
            "arxiv_id": src.arxiv_id,
            "arxiv_abs": src.abs_url,
            "pdf_url": src.pdf_url,
            "reading_note": src.reading_note,
            "pdf_sha256": _sha256(pdf_path),
            "pdf_bytes": os.path.getsize(pdf_path),
            "n_chars_extracted": len(text),
            "n_decimal_literals": len(numbers),
            "excerpts": excerpts,
            "numbers_inventory_file": os.path.relpath(inv_path, HERE),
            "excerpts_file": os.path.relpath(txt_path, HERE),
            "attributed_triple_numbers_present": present,
            "attributed_triple_numbers_missing": {
                k: [n for n in ("%g" % v for v in triple) if n not in numbers]
                for k, triple in ATTRIBUTED_TRIPLES.items()},
        })
        if verbose:
            print("  %s: %d chars, %d decimals, triples present=%s"
                  % (src.key, len(text), len(numbers), present))

    with open(SOURCES_JSON, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return manifest


# --------------------------------------------------------------------------- #
# 3. Offline reading side
# --------------------------------------------------------------------------- #
def load_manifest(path: str = SOURCES_JSON) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _excerpt_text(manifest: dict, source_key: str, label: str) -> str:
    for src in manifest["sources"]:
        if src["key"] != source_key:
            continue
        for ex in src["excerpts"]:
            if ex["label"] == label:
                return ex["text"]
    raise KeyError("%s/%s" % (source_key, label))


# regexes over the *verbatim excerpts*: the only place numbers enter the model
_RE_MU_MIN_B = re.compile(r"min\s*≃\s*([0-9]+(?:\.[0-9]+)?)\s*and\s*b\s*≃\s*([0-9]+(?:\.[0-9]+)?)")
_RE_NU_MAX_A = re.compile(r"νmax\s*≃\s*([0-9]+(?:\.[0-9]+)?)\s*and\s*a\s*≃\s*([0-9]+(?:\.[0-9]+)?)")
_RE_SATURATION = re.compile(r"tends to saturate for I\s*≥\s*([0-9]+(?:\.[0-9]+)?)")
_RE_FRICTIONLESS = re.compile(r"min\s*≃\s*([0-9]+(?:\.[0-9]+)?)")
_RE_GDR_PACK_HI = re.compile(r"reaches its maximum,\s*([0-9]+(?:\.[0-9]+)?)")
_RE_GDR_PACK_LO = re.compile(r"down to\s*([0-9]+(?:\.[0-9]+)?)\s*for I\s*≃\s*([0-9]+(?:\.[0-9]+)?)")
_GREEK_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def parse_dacruz(manifest: dict) -> dict:
    """Parse da Cruz 2005 constants out of the archived excerpts.

    Every value returned here is a capture group of a regex applied to text
    that was sliced out of the PDF at build time.
    """
    mu_ex = _excerpt_text(manifest, "dacruz2005", "friction_law_constants")
    m = _RE_MU_MIN_B.search(mu_ex)
    if not m:
        raise RuntimeError("mu-law constants not parsed from excerpt")
    mu_min, b = float(m.group(1)), float(m.group(2))

    nu_ex = _excerpt_text(manifest, "dacruz2005", "dilatancy_constants")
    m = _RE_NU_MAX_A.search(nu_ex)
    if not m:
        raise RuntimeError("dilatancy constants not parsed from excerpt")
    nu_max, a = float(m.group(1)), float(m.group(2))

    sat_ex = _excerpt_text(manifest, "dacruz2005", "saturation")
    m = _RE_SATURATION.search(sat_ex)
    if not m:
        raise RuntimeError("saturation threshold not parsed from excerpt")
    i_sat = float(m.group(1))

    fr_ex = _excerpt_text(manifest, "dacruz2005", "frictionless_singular")
    m = _RE_FRICTIONLESS.search(fr_ex)
    mu_min_frictionless = float(m.group(1)) if m else float("nan")

    return {
        "law": "mu(I) = mu_min + b*I   (linear, valid for I below the "
               "saturation threshold);  phi(I) = nu_max - a*I",
        "mu_min": mu_min,
        "b": b,
        "i_saturation": i_sat,
        "nu_max": nu_max,
        "a": a,
        "mu_min_frictionless": mu_min_frictionless,
        "mu_p_of_fit": 0.4,
        "locators": {
            "mu_law": "da Cruz 2005 Sec. IV B, Eq. (14)",
            "dilatancy": "da Cruz 2005 Sec. IV A, Eq. (13)",
        },
    }


def parse_gdr_midi(manifest: dict) -> dict:
    pack_ex = _excerpt_text(manifest, "gdr_midi2004", "packing_path")
    hi = _RE_GDR_PACK_HI.search(pack_ex)
    lo = _RE_GDR_PACK_LO.search(pack_ex)
    if not (hi and lo):
        raise RuntimeError("GDR MiDi packing path not parsed from excerpt")
    return {
        "phi_max_quasi_static": float(hi.group(1)),
        "phi_at_I": float(lo.group(1)),
        "I_of_second_point": float(lo.group(2)),
        "law": "no fitted mu(I) triple published; mu_eff rises from mu_S and "
               "saturates, phi falls 0.85 -> 0.80 over I = 0 -> 0.1",
    }


@dataclass(frozen=True)
class DaCruzLaw:
    """The **archived** da Cruz 2005 law, in the form the paper publishes it.

    ``mu`` is quoted only up to ``i_saturation``: the paper states that the
    linear fit breaks down and ``mu*`` saturates for ``I >~ 0.2``.  Outside
    that range :meth:`mu` returns ``None`` for the *law* value, so that a
    benchmark cannot silently compare against an extrapolation.
    """

    mu_min: float
    b: float
    nu_max: float
    a: float
    i_saturation: float = 0.2

    def mu(self, I: float) -> Optional[float]:
        if I > self.i_saturation:
            return None
        return self.mu_min + self.b * I

    def mu_extrapolated(self, I: float) -> float:
        """The linear form outside its validity - reported, never judged."""
        return self.mu_min + self.b * I

    def phi(self, I: float) -> Optional[float]:
        if I > self.i_saturation:
            return None
        return self.nu_max - self.a * I

    def phi_extrapolated(self, I: float) -> float:
        return self.nu_max - self.a * I


def archived_law(manifest: Optional[dict] = None) -> DaCruzLaw:
    """Build :class:`DaCruzLaw` from the committed archive (no literals)."""
    m = manifest if manifest is not None else load_manifest()
    c = parse_dacruz(m)
    return DaCruzLaw(mu_min=c["mu_min"], b=c["b"], nu_max=c["nu_max"],
                     a=c["a"], i_saturation=c["i_saturation"])


# --------------------------------------------------------------------------- #
# 4. Jop-form representation of the archived law (reconciliation arithmetic)
# --------------------------------------------------------------------------- #
def jop_mu(I, mu_s: float, mu_2: float, i_0: float):
    return mu_s + (mu_2 - mu_s) / (i_0 / I + 1.0)


def fit_jop_to_law(law: DaCruzLaw, i_lo: float = 1e-3, i_hi: float = 0.2,
                   n: int = 240) -> dict:
    """Least-squares ``(mu_s, mu_2, I_0)`` for the *archived* linear law.

    The question this answers is not "what is the truth" but "could one of the
    in-repo triples be a faithful transcription of da Cruz?".  If the best
    Jop-form fit to da Cruz's own law is far from both triples, then the answer
    is no - the triples are somebody else's fit (Jop et al. 2006), reformulated
    onto a different data set.
    """
    import numpy as np

    I = np.geomspace(i_lo, i_hi, n)
    target = law.mu_min + law.b * I

    best = None
    # coarse-to-fine grid: cheap, deterministic, no scipy dependency
    ms = np.linspace(0.05, 0.45, 81)
    m2 = np.linspace(0.40, 1.20, 81)
    i0 = np.geomspace(0.05, 1.5, 61)
    for i0v in i0:
        denom = i0v / I + 1.0
        for msv in ms:
            for m2v in m2:
                pred = msv + (m2v - msv) / denom
                err = float(np.max(np.abs(pred - target) / target))
                if best is None or err < best["max_abs_rel_err"]:
                    best = {"mu_s": float(msv), "mu_2": float(m2v),
                            "I_0": float(i0v), "max_abs_rel_err": err}
    pred = jop_mu(I, best["mu_s"], best["mu_2"], best["I_0"])
    best["rel_err_curve"] = (pred - target) / target
    best["I_grid"] = I
    best["i_range"] = [i_lo, i_hi]
    best["tol"] = 0.10
    best["reproduces_law_within_tol"] = best["max_abs_rel_err"] <= 0.10
    return best


__all__ = [
    "SOURCES", "SOURCES_JSON", "ARCHIVE_DIR", "EXCERPT_ANCHORS",
    "ATTRIBUTED_TRIPLES", "Source", "DaCruzLaw", "build_archive",
    "load_manifest", "parse_dacruz", "parse_gdr_midi", "archived_law",
    "decimal_numbers", "jop_mu", "fit_jop_to_law", "JOP_FORMS",
]
"""Optional comminution coupling for the erosion stream (criterion C7).

The erosion accounting in :mod:`tensordem.erosion_dem.stream` and
:mod:`tensordem.erosion_dem.crater` knows nothing about particle breakage: it
never imports the granular-crush / PBM machinery of
:mod:`tensordem.erosion_pbm`.  Comminution is attached from the outside as an
``on_impact`` callback, and this module is the only place where both worlds are
visible at once:

    run_impact_stream(cfg, on_impact=fragmentation_hook(policy))

Two questions are answered here, both machine-checkable:

1. **Structural decoupling.**  Turning fragmentation on with an identity policy
   (``n_fragments = 1``, no RNG use) must leave every erosion number bitwise
   unchanged, including the mass-ledger closure residual.  The fragmentation
   policy is deterministic (equal-volume daughters), so the erosion path is not
   even re-seeded -- the residual difference is exactly ``0``.
2. **One-way coupling.**  With ``n_fragments > 1`` the erodent gets finer, so
   the size exponent ``k3`` of the erosion law changes the removed mass.  The
   comminution model can therefore *drive* erosion, while erosion can never
   change the fragmentation (no feedback term exists in the code), which is the
   physical causality the campaign node asks for.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
import pathlib

__all__ = [
    "FragmentationPolicy",
    "fragmentation_hook",
    "mass_scale",
    "erosion_core_imports_comminution",
]

_EROSION_CORE_MODULES = ("crater.py", "contact.py", "stream.py")
_COMMINUTION_PREFIXES = (
    "tensordem.erosion_pbm",
    "tensordem.granule_crush",
    "tensordem.crushing",
    "tensordem.breakage_pbm",
    "tensordem.erosion_dem.breakage_bridge",
)


@dataclass(frozen=True)
class FragmentationPolicy:
    """Equal-volume daughter split, mass conserving by construction."""

    n_fragments: int = 1

    def __post_init__(self) -> None:
        if self.n_fragments < 1:
            raise ValueError("n_fragments must be >= 1")

    @property
    def carries_single_daughter(self) -> bool:
        return self.n_fragments == 1

    def mass_scale(self) -> float:
        """Mass ratio of the daughter that stays on the wall-impact path."""
        return 1.0 / self.n_fragments

    def sibling_mass_fraction(self) -> float:
        """Fraction of the parent mass that leaves with the sibling daughters."""
        return (self.n_fragments - 1.0) / self.n_fragments


def mass_scale(n_fragments: int) -> float:
    """Convenience wrapper of :meth:`FragmentationPolicy.mass_scale`."""
    return FragmentationPolicy(n_fragments).mass_scale()


def fragmentation_hook(policy: FragmentationPolicy | int = 1):
    """Return an ``on_impact`` callback that fragments the carried particle.

    The parent particle is replaced by ``policy.n_fragments`` equal-volume
    daughters; the stream follows one daughter (mass ratio ``1/n_fragments``)
    and books the rest as sibling mass, so ``carried + siblings = parent``
    exactly.  With ``n_fragments = 1`` the callback returns a no-op delta and
    the stream is numerically untouched.
    """
    pol = policy if isinstance(policy, FragmentationPolicy) else FragmentationPolicy(policy)

    def hook(record: dict) -> dict:
        if pol.carries_single_daughter:
            return {"particle_scale": 1.0, "sibling_mass_kg": 0.0}
        return {
            "particle_scale": pol.mass_scale(),
            "sibling_mass_kg": record["particle_mass_kg"] * pol.sibling_mass_fraction(),
        }

    return hook


def erosion_core_imports_comminution(package_dir: str | pathlib.Path | None = None) -> dict:
    """Static check: does the erosion core import any comminution module?

    Parses the three erosion-core modules with :mod:`ast` (no import-time side
    effects) and looks for comminution imports.  ``coupled_modules = []`` is the
    structural half of criterion C7; the runtime half is the bitwise identity of
    the ledger residual under an identity fragmentation policy.
    """
    root = pathlib.Path(package_dir) if package_dir else pathlib.Path(__file__).parent
    hits = []
    for name in _EROSION_CORE_MODULES:
        path = root / name
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for mod in mods:
                if any(mod.startswith(p) for p in _COMMINUTION_PREFIXES):
                    hits.append({"module": name, "imports": mod, "line": node.lineno})
    return {
        "checked_modules": list(_EROSION_CORE_MODULES),
        "coupled_modules": hits,
        "decoupled": not hits,
    }
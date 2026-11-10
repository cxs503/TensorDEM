"""Contact-force -> macroscopic ice-resistance mapping with mechanism decomposition.

The task this module answers is a *bookkeeping* question that sits between a DEM
solution and an engineering resistance formula: given the per-contact force
vectors that a rigid hull exchanges with a bonded/fragmented ice field, how much
of the resulting advance-direction force belongs to **crushing**, how much to
**bending** of the intact sheet, and how much to **submersion** of already
detached ice, and does the decomposition close exactly?

Closure (the identity that is *verifiable*) is deliberately separated from
physical grading (what is only *argued*):

VERIFIED HERE (closure / bookkeeping, exact)
    For every contact ``i`` the resistance contribution is ``c_i = +(F_i . e)``
    with ``e`` the hull advance direction and ``F_i`` the force the hull exerts on
    the ice: the ice reacts on the hull with ``-F_i``, whose component along the
    retarding direction ``-e`` is ``+F_i . e``.  The heading reaction of the hull
    is ``R = -e . sum_i F_i = sum_i c_i``.  The mechanism decomposition assigns every
    contact to exactly one of three classes, so
    ``R = R_crushing + R_bending + R_submersion`` holds to floating-point
    round-off; the residual is measured and returned, never assumed.

ARGUED, NOT VERIFIED HERE (mechanism taxonomy)
    * ``submersion`` -- the contact acts on ice that has **detached** from the
      intact field (particle not connected to the largest surviving bond
      cluster) **and** presses it down (negative vertical force).  Pressing
      detached ice down costs work against buoyancy and drags it along the hull:
      that is the load path Lindqvist represents by the displaced channel plus
      hull friction.  Detached ice squeezed *forward* stays in the crushing
      class instead.
    * ``crushing`` -- an intact-field contact whose normal force has reached the
      **plastic crushing cap** of the contact law (``p_crush * A_contact``).  The
      cap is the DEM representation of compressive failure at the contact.
    * ``bending`` -- every remaining intact-field contact, i.e. points of the
      sheet that carry load in flexure without having reached the crushing cap
      at the sampled instant.

    These are *mechanism proxies*, not a decomposition derived from continuum
    energy partitions.  A contact can be elastic at one instant and capped at the
    next, so a component share is a time-averaged statement about load path, not
    an energy partition.  Grading against published full-scale data is WITHHELD:
    no independent full-scale data set is reachable from this sandbox.

The taxonomy matches the additive split already used by the level-ice section
study (``docs/LEVEL_ICE_RESISTANCE_DEM.md``: capped normal = crushing, uncapped
normal = flexure, tangential = friction) extended with the detached-ice class so
that the three Lindqvist mechanisms, not two, are populated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

#: Mechanism order used by every label tensor and component mapping.
MECHANISMS: tuple[str, str, str] = ("crushing", "bending", "submersion")
CRUSHING, BENDING, SUBMERSION = 0, 1, 2

#: Cap-saturation tolerance: a contact counts as capped when the uncapped penalty
#: force exceeds ``(1 - tol) * cap``.  Pure round-off guard, not a physics knob.
SATURATION_TOL = 1e-9


def _real_tensor(name: str, value: Any, shape: Sequence[int], *, device=None) -> torch.Tensor:
    """Coerce to float64 and check the shape (``-1`` in ``shape`` is a wildcard)."""
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got shape {tuple(tensor.shape)}")
    for actual, expected in zip(tensor.shape, shape):
        if expected >= 0 and int(actual) != int(expected):
            raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


@dataclass(frozen=True)
class HullContactSet:
    """Per-contact state of a hull/ice contact set, engine agnostic.

    All quantities are in SI and per contact (``n`` contacts):

    ``positions``            (n, 3) contact-point positions
    ``normals``              (n, 3) unit hull outward normal (hull -> ice)
    ``normal_force``         (n,)   normal force magnitude, >= 0
    ``friction_force``       (n, 3) tangential (Coulomb) force on the ice
    ``penetration``          (n,)   normal overlap, >= 0
    ``cap_force``            (n,)   plastic crushing cap magnitude (``inf`` = uncapped)
    ``detached``             (n,)   bool, ice particle detached from the intact field
    ``submerged_fraction``   (n,)   submerged volume fraction of the ice particle
    """

    positions: torch.Tensor
    normals: torch.Tensor
    normal_force: torch.Tensor
    friction_force: torch.Tensor
    penetration: torch.Tensor
    cap_force: torch.Tensor
    detached: torch.Tensor
    submerged_fraction: torch.Tensor

    def __post_init__(self) -> None:
        positions = _real_tensor("positions", self.positions, (-1, 3))
        count = positions.shape[0]
        object.__setattr__(self, "positions", positions)
        normals = _real_tensor("normals", self.normals, (count, 3))
        norms = torch.linalg.vector_norm(normals, dim=1)
        if count and float(torch.abs(norms - 1.0).max()) > 1e-9:
            raise ValueError("normals must be unit vectors")
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "friction_force", _real_tensor("friction_force", self.friction_force, (count, 3)))
        for name in ("normal_force", "penetration", "submerged_fraction"):
            tensor = _real_tensor(name, getattr(self, name), (count,))
            if count and float(tensor.min()) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, tensor)
        cap = torch.as_tensor(self.cap_force, dtype=torch.float64)
        if tuple(cap.shape) != (count,):
            raise ValueError(f"cap_force must have shape ({count},), got {tuple(cap.shape)}")
        if bool(torch.isnan(cap).any()) or bool(torch.isneginf(cap).any()):
            raise ValueError("cap_force must be >= 0 (use +inf for an uncapped contact)")
        if count and float(cap.min()) < 0.0:
            raise ValueError("cap_force must be nonnegative (use inf for uncapped)")
        object.__setattr__(self, "cap_force", cap)
        detached = torch.as_tensor(self.detached)
        if tuple(detached.shape) != (count,):
            raise ValueError(f"detached must have shape ({count},)")
        object.__setattr__(self, "detached", detached.to(torch.bool))
        if count and float(self.submerged_fraction.max()) > 1.0 + 1e-12:
            raise ValueError("submerged_fraction must be in [0, 1]")

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def force_on_particle(self) -> torch.Tensor:
        """(n, 3) force the hull exerts on the ice at each contact."""
        return self.normal_force[:, None] * self.normals + self.friction_force

    def saturated(self, tol: float = SATURATION_TOL) -> torch.Tensor:
        """(n,) bool: contact normal force has reached the plastic crushing cap."""
        if not self.count:
            return torch.zeros(0, dtype=torch.bool)
        finite = torch.isfinite(self.cap_force)
        threshold = torch.where(finite, self.cap_force * (1.0 - tol), torch.full_like(self.cap_force, math.inf))
        return self.normal_force >= threshold

    def labels(self, tol: float = SATURATION_TOL) -> torch.Tensor:
        """(n,) int64 mechanism label, one of :data:`MECHANISMS`.

        Priority: **submersion** > crushing > bending.

        * submersion: the ice is detached from the intact field **and** the hull
          is pressing it down (negative vertical force component).  That is the
          load path that costs work against buoyancy, i.e. Lindqvist's
          submersion term.  A detached fragment that is being squeezed *forward*
          is not being submerged -- it is being crushed (next class), which is why
          the vertical-sign guard is part of the definition and not a detail.
        * crushing: the contact has reached the plastic crushing cap (compressive
          failure at the contact patch).
        * bending: everything else -- load transmitted into an intact sheet
          without local crushing, i.e. the flexural load path.
        """
        if not self.count:
            return torch.zeros(0, dtype=torch.int64)
        labels = torch.full((self.count,), BENDING, dtype=torch.int64)
        labels[self.saturated(tol)] = CRUSHING
        pressing_down = self.force_on_particle[:, 2] < 0.0
        labels[self.detached & pressing_down] = SUBMERSION
        return labels


@dataclass(frozen=True)
class MechanismDecomposition:
    """Resistance split of one contact snapshot (or of a time-averaged history)."""

    r_total: float
    reaction_vector: tuple[float, float, float]
    components: dict[str, float]
    residual: float
    relative_residual: float
    contact_counts: dict[str, int]
    contact_count: int
    capped_count: int
    detached_count: int

    @property
    def shares(self) -> dict[str, float]:
        scale = abs(self.r_total)
        if scale <= 0.0:
            return {name: 0.0 for name in MECHANISMS}
        return {name: self.components[name] / scale for name in MECHANISMS}

    @property
    def lateral_reaction(self) -> float:
        return self.reaction_vector[1]

    @property
    def vertical_reaction(self) -> float:
        return self.reaction_vector[2]

    def as_dict(self) -> dict[str, Any]:
        return {
            "r_total": self.r_total,
            "reaction_vector": list(self.reaction_vector),
            "components": dict(self.components),
            "shares": self.shares,
            "residual": self.residual,
            "relative_residual": self.relative_residual,
            "contact_counts": dict(self.contact_counts),
            "contact_count": self.contact_count,
            "capped_count": self.capped_count,
            "detached_count": self.detached_count,
        }


def unit_heading(heading: Iterable[float]) -> torch.Tensor:
    """Normalized (3,) advance direction; rejects a zero vector."""
    vector = _real_tensor("heading", list(heading), (3,))
    norm = float(torch.linalg.vector_norm(vector))
    if norm <= 0.0:
        raise ValueError("heading must be nonzero")
    return vector / norm


def decompose(contacts: HullContactSet, *, heading: Iterable[float] = (1.0, 0.0, 0.0),
              tol: float = SATURATION_TOL) -> MechanismDecomposition:
    """Split the heading reaction of a contact set into the three mechanisms.

    Returns a :class:`MechanismDecomposition` whose ``residual`` is the measured
    closure error ``R_total - (R_crush + R_bend + R_sub)``.  The residual is
    round-off: every contact is assigned to exactly one class, so no force can be
    lost or double counted.  A zero-resistance snapshot (no contacts, or a
    no-ice case) returns an exactly zero decomposition.
    """
    e = unit_heading(heading)
    if not contacts.count:
        zero = {name: 0.0 for name in MECHANISMS}
        counts = {name: 0 for name in MECHANISMS}
        return MechanismDecomposition(0.0, (0.0, 0.0, 0.0), zero, 0.0, 0.0, counts, 0, 0, 0)
    force = contacts.force_on_particle
    # c_i > 0 means the contact resists advance: the ice pushes the hull back
    # along -e with magnitude F_i . e (F_i = force hull -> ice).
    contributions = force @ e
    reaction = -force.sum(dim=0)
    labels = contacts.labels(tol)
    components: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index, name in enumerate(MECHANISMS):
        mask = labels == index
        components[name] = float(contributions[mask].sum())
        counts[name] = int(mask.sum())
    r_total = float(-(reaction @ e))
    residual = r_total - sum(components.values())
    scale = abs(r_total)
    relative = abs(residual) / scale if scale > 0.0 else abs(residual)
    return MechanismDecomposition(
        r_total=r_total,
        reaction_vector=tuple(float(value) for value in reaction.tolist()),
        components=components,
        residual=float(residual),
        relative_residual=float(relative),
        contact_counts=counts,
        contact_count=contacts.count,
        capped_count=int(contacts.saturated(tol).sum()),
        detached_count=int(contacts.detached.sum()),
    )


def average_decompositions(snapshots: Sequence[MechanismDecomposition]) -> MechanismDecomposition:
    """Time-average a history of snapshots (each snapshot stays exactly closed)."""
    if not snapshots:
        raise ValueError("snapshots must not be empty")
    count = float(len(snapshots))
    components = {
        name: sum(snapshot.components[name] for snapshot in snapshots) / count
        for name in MECHANISMS
    }
    r_total = sum(snapshot.r_total for snapshot in snapshots) / count
    reaction = tuple(
        sum(snapshot.reaction_vector[axis] for snapshot in snapshots) / count for axis in range(3)
    )
    residual = r_total - sum(components.values())
    scale = abs(r_total)
    relative = abs(residual) / scale if scale > 0.0 else abs(residual)
    counts = {
        name: int(round(sum(snapshot.contact_counts[name] for snapshot in snapshots) / count))
        for name in MECHANISMS
    }
    return MechanismDecomposition(
        r_total=float(r_total),
        reaction_vector=reaction,
        components=components,
        residual=float(residual),
        relative_residual=float(relative),
        contact_counts=counts,
        contact_count=int(round(sum(s.contact_count for s in snapshots) / count)),
        capped_count=int(round(sum(s.capped_count for s in snapshots) / count)),
        detached_count=int(round(sum(s.detached_count for s in snapshots) / count)),
    )


# ---------------------------------------------------------------------------
# Flexural-loss screening (diagnostic, not part of the closure identity)
# ---------------------------------------------------------------------------
def plate_characteristic_length(elastic_modulus_pa: float, thickness_m: float,
                                water_density_kg_m3: float = 1025.0,
                                gravity_m_s2: float = 9.80665,
                                poisson_ratio: float = 0.3) -> float:
    """Characteristic length ``l = (D / (rho_w g))^(1/4)`` of a floating ice plate.

    ``D = E h^3 / (12 (1 - nu^2))`` is the plate rigidity.  Note this is the
    length over which a floating sheet transmits load to buoyancy, i.e. the
    physical bending length the DEM sheet must resolve; it is used below to
    screen whether the modelled sheet is long enough to bend like a sheet.
    """
    for name, value, positive in (
        ("elastic_modulus_pa", elastic_modulus_pa, True),
        ("thickness_m", thickness_m, True),
        ("water_density_kg_m3", water_density_kg_m3, True),
        ("gravity_m_s2", gravity_m_s2, True),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or (positive and float(value) <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 <= poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    rigidity = elastic_modulus_pa * thickness_m**3 / (12.0 * (1.0 - poisson_ratio**2))
    return (rigidity / (water_density_kg_m3 * gravity_m_s2)) ** 0.25


def flexural_moment_capacity(flexural_strength_pa: float, thickness_m: float) -> float:
    """Root bending moment capacity per unit width ``sigma_f h^2 / 6`` [N m]. """
    if not (flexural_strength_pa > 0.0 and thickness_m > 0.0):
        raise ValueError("strengths and thickness must be positive")
    return flexural_strength_pa * thickness_m**2 / 6.0


def flexural_demand_ratio(vertical_load_per_width_n_m: float, characteristic_length_m: float,
                          flexural_strength_pa: float, thickness_m: float) -> float:
    """Screening ratio of the DEM bending demand to the sheet capacity.

    The peak moment of a line load ``P`` per unit width on a beam on an elastic
    (buoyancy) foundation is ``M_max = P l / 4`` with ``l`` the characteristic
    length; the ratio ``M_max / (sigma_f h^2 / 6)`` therefore estimates whether
    flexural failure is expected at the modelled load level.  Diagnostic only:
    the sheet is fractured in the DEM, so the elastic formula is an upper bound
    after the first crack.
    """
    capacity = flexural_moment_capacity(flexural_strength_pa, thickness_m)
    moment = abs(vertical_load_per_width_n_m) * characteristic_length_m / 4.0
    return moment / capacity
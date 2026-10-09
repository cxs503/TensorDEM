"""Opt-in per-bond central-spring dynamics; legacy IceDEM is unchanged."""
import hashlib
import json
import struct

import torch

from .dem import DEMConfig, IceDEM


class CalibratedIceDEM(IceDEM):
    """Preserves actual disk mass/contact/failure and uses per-bond N/m.

    The scalar config stiffness must upper-bound every coefficient, so inherited
    conservative timestep bounds remain valid. This is not a continuum-mass or
    calibrated-fracture backend; particle translation and central forces only.
    """

    def __init__(self, config=DEMConfig(), *, bond_stiffness=None, **kwargs):
        super().__init__(config, **kwargs)
        if bond_stiffness is None:
            raise ValueError("explicit per-bond stiffness required")
        self.coefficients = torch.as_tensor(
            bond_stiffness, dtype=torch.float64, device=self.device
        ).clone()
        if (
            self.coefficients.shape != (len(self.bond_indices),)
            or not bool(torch.isfinite(self.coefficients).all())
            or bool((self.coefficients <= 0).any())
            or bool((self.coefficients > config.bond_stiffness).any())
        ):
            raise ValueError(
                "per-bond stiffness shape/finite/positive/conservative upper bound mismatch"
            )
        self.coefficient_hash = self._hash()
        self.pair_coefficients = torch.zeros(
            len(self.pairs), dtype=torch.float64, device=self.device
        )
        self.pair_coefficients[self.bonded] = self.coefficients

    def _hash(self):
        # Avoid Tensor.numpy(): hashing must work in minimal PyTorch installs,
        # including environments where PyTorch's optional NumPy bridge is absent.
        values = self.coefficients.detach().to(device="cpu", dtype=torch.float64).tolist()
        payload = struct.pack(f"<{len(values)}d", *values)
        return hashlib.sha256(payload).hexdigest()

    def _check(self):
        if (
            self._hash() != self.coefficient_hash
            or not torch.equal(
                self.pair_coefficients[self.bonded], self.coefficients
            )
            or bool((self.pair_coefficients[~self.bonded] != 0).any())
        ):
            raise ValueError("per-bond stiffness mutated after construction")

    @torch.no_grad()
    def forces(self, external_forces=None):
        self._check()
        previous = self.alive.clone()
        force, reaction = super().forces(external_forces)
        delta = self.positions[self.pairs[:, 1]] - self.positions[self.pairs[:, 0]]
        distance = torch.linalg.vector_norm(delta, dim=1)
        normal = delta / distance.clamp_min(torch.finfo(delta.dtype).eps)[:, None]
        normal = torch.where((distance > 0)[:, None], normal, self.initial_normals)
        extension = distance - self.rest_lengths
        difference = self.pair_coefficients - self.config.bond_stiffness
        correction = (difference * extension * self.alive)[:, None] * normal
        force.index_add_(0, self.pairs[:, 0], correction)
        force.index_add_(0, self.pairs[:, 1], -correction)
        newly_broken = previous & ~self.alive
        self.accounting["fracture_release_J"] += float(
            0.5
            * (difference[newly_broken] * extension[newly_broken].square()).sum()
        )
        return force, reaction

    def mechanical_energy(self):
        self._check()
        record = super().mechanical_energy()
        delta = self.positions[self.pairs[:, 1]] - self.positions[self.pairs[:, 0]]
        extension = torch.linalg.vector_norm(delta, dim=1) - self.rest_lengths
        corrected = float(
            0.5
            * (
                self.pair_coefficients[self.alive]
                * extension[self.alive].square()
            ).sum()
        )
        record["mechanical_J"] += corrected - record["bond_J"]
        record["bond_J"] = corrected
        return record

    def snapshot(self):
        self._check()
        return dict(
            schema="tensordem.calibrated-restart/1",
            base=IceDEM.snapshot(self),
            bond_stiffness_N_per_m=self.coefficients.tolist(),
            coefficient_sha256=self.coefficient_hash,
        )

    @classmethod
    def from_snapshot(cls, state, *, device=None):
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "schema",
                "base",
                "bond_stiffness_N_per_m",
                "coefficient_sha256",
            }
            or state["schema"] != "tensordem.calibrated-restart/1"
        ):
            raise ValueError("invalid calibrated restart schema/fields")
        json.dumps(state, allow_nan=False)
        base = IceDEM.from_snapshot(state["base"], device=device)
        result = cls(base.config, bond_stiffness=state["bond_stiffness_N_per_m"])
        if result.coefficient_hash != state["coefficient_sha256"]:
            raise ValueError("calibrated restart coefficient hash mismatch")
        coefficients = result.coefficients
        pair_coefficients = result.pair_coefficients
        digest = result.coefficient_hash
        result.__dict__.update(base.__dict__)
        result.coefficients = coefficients
        result.pair_coefficients = pair_coefficients
        result.coefficient_hash = digest
        return result

"""RBF/MLS and momentum-exchange counterparts of the 2D/3D mortar operator.

Purpose
-------
Mission m5207 established the conservation/order contract for three interface
operators on a **1D** interface.  This module lifts the RBF and momentum-exchange
operators to interface *curves in 2D* and interface *surfaces in 3D* so that the
three-operator comparison table (mortar / RBF / ME) is dimension-consistent, and
so that the measure-normalisation trap can be exercised in 2D and 3D.

Both operators work on the local tangent coordinates of an
:class:`~tensordem.interface_operators.mortar.InterfaceSide`, i.e. on the
interface manifold only; the ambient embedding is handled by the side's basis.

Conservation contract
---------------------
* ``RBFOperator`` with ``conservative=True``: Gaussian RBF with local linear
  (MLS) augmentation followed by the **measure-proportional** deficit backfill
  ``dv_j = (sent-recv) |T_j| / sum|T|^2``.  A compact kernel alone leaks mass on
  non-matching and boundary cells, so the backfill is what closes the budget.
* ``MomentumExchangeOperator``: gather normalises the kernel **per donor** (each
  donor cell hands out 100 % of its measure-weighted momentum) and spread
  normalises **per acceptor** (each acceptor hands back 100 % of what it holds),
  so the two sides close the momentum budget exactly.  This is the 2D/3D lift of
  the m5207 LBM-DEM ME convention; the exchange measure plays the role of the
  particle exchange volume ``V_p``.

Negative controls (``broken=``)
-------------------------------
``"backfill_linear"``  wrong backfill normalisation (the m5207 section 5.2 trap)
``"no_fix"``           RBF gather without any conservative correction
``"per_node_norm"``    ME gather normalised per acceptor: silently drops momentum
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mortar import (InterfaceSide, integral, l2_error, measure_proportional_backfill)

__all__ = ["RBFOperator", "MomentumExchangeOperator", "characteristic_length"]


def characteristic_length(side: InterfaceSide) -> float:
    """Kernel length scale of a side: mean cell length, or sqrt(mean cell area)."""
    m = float(np.mean(side.measures))
    return m if side.nb == 1 else float(np.sqrt(m))


def _kernel(src: InterfaceSide, tgt: InterfaceSide, eps: float) -> np.ndarray:
    """Gaussian kernel ``(nt, ns)`` from acceptor to donor centroids."""
    x = src.local_centroids
    y = tgt.local_centroids
    d = y[:, None, :] - x[None, :, :]                 # (nt, ns, nb)
    r2 = np.einsum("ijk,ijk->ij", d, d)
    return np.exp(-r2 / (eps * eps)), d


@dataclass
class RBFOperator:
    """RBF/MLS scatter between two discretisations (2D curve or 3D surface).

    Parameters
    ----------
    epsilon : float
        Kernel width in units of :func:`characteristic_length`.
    linear_augment : bool
        Local linear (moving-least-squares) augmentation, which makes the operator
        second order; without it the kernel is zeroth order.
    conservative : bool
        Apply the measure-proportional deficit backfill (``False`` reproduces the
        ``no_fix`` negative control through a cleaner flag).
    broken : {None, "no_fix", "backfill_linear"}
    """

    epsilon: float = 1.8
    linear_augment: bool = True
    conservative: bool = True
    broken: str | None = None
    name: str = "rbf-mls-2d3d"

    def __post_init__(self) -> None:
        if self.broken not in (None, "no_fix", "backfill_linear"):
            raise ValueError(f"unknown broken mode {self.broken!r}")

    def transfer(self, src: InterfaceSide, src_values: np.ndarray,
                 tgt: InterfaceSide) -> np.ndarray:
        u = np.asarray(src_values, dtype=float)
        h = characteristic_length(src)
        eps = self.epsilon * h
        w, d = _kernel(src, tgt, eps)                     # (nt, ns), (nt, ns, nb)
        nb = src.nb
        # augmented least squares: v_j = a_j + G_j . (x_i - y_j)
        S0 = w.sum(axis=1)                                # (nt,)
        R0 = w @ u
        if self.linear_augment and nb >= 1:
            S1 = np.einsum("ji,jin->jn", w, d)            # (nt, nb)
            S2 = np.einsum("ji,jin,jim->jnm", w, d, d)    # (nt, nb, nb)
            R1 = np.einsum("ji,jin,i->jn", w, d, u)       # (nt, nb)
            M = np.zeros((len(S0), nb + 1, nb + 1))
            M[:, 0, 0] = S0
            M[:, 0, 1:] = S1
            M[:, 1:, 0] = S1
            M[:, 1:, 1:] = S2
            rhs = np.concatenate([R0[:, None], R1], axis=1)
            reg = 1e-12 * np.maximum(np.trace(S2, axis1=1, axis2=2), 1e-300)
            for j in range(len(S0)):
                M[j] += reg[j] * np.eye(nb + 1)
            sol = np.empty_like(rhs)
            ok = np.abs(np.linalg.det(M)) > 0.0
            for j in range(len(S0)):
                try:
                    sol[j] = np.linalg.solve(M[j], rhs[j])
                except np.linalg.LinAlgError:            # pragma: no cover
                    sol[j] = np.linalg.pinv(M[j]) @ rhs[j]
            v = np.where(ok, sol[:, 0], R0 / np.maximum(S0, 1e-300))
        else:
            v = R0 / np.maximum(S0, 1e-300)
        if self.conservative and self.broken != "no_fix":
            sent = integral(u, src.measures)
            recv = integral(v, tgt.measures)
            v = measure_proportional_backfill(
                v, sent, recv, tgt.measures,
                broken="backfill_linear" if self.broken == "backfill_linear" else None)
        return v


@dataclass
class MomentumExchangeOperator:
    """Momentum-exchange (ME) transfer on an interface manifold.

    ``gather`` returns the momentum *per acceptor cell* (an integral, not a mean)::

        q_j = sum_i (lambda_ij / Lambda_i) u_i |S_i|,   Lambda_i = sum_j lambda_ij

    so ``sum_j q_j = sum_i u_i |S_i|`` exactly: every donor cell hands out exactly
    what it holds.  ``spread`` returns the reaction handed back to the donors::

        w_i = - sum_j (lambda_ij / Lambda'_j) q_j,     Lambda'_j = sum_i lambda_ij

    so ``sum_i w_i = -sum_j q_j = -sent`` and the momentum budget closes to
    round-off.  ``broken="per_node_norm"`` normalises gather per acceptor instead,
    which silently drops part of the donor momentum (the m5207 ME trap).
    """

    epsilon: float = 1.2
    broken: str | None = None
    name: str = "momentum-exchange-2d3d"

    def __post_init__(self) -> None:
        if self.broken not in (None, "per_node_norm"):
            raise ValueError(f"unknown broken mode {self.broken!r}")

    def _weights(self, src: InterfaceSide, tgt: InterfaceSide) -> np.ndarray:
        eps = self.epsilon * characteristic_length(src)
        w, _ = _kernel(src, tgt, eps)
        return np.maximum(w, 1e-300)

    def gather(self, src: InterfaceSide, src_values: np.ndarray,
               tgt: InterfaceSide) -> np.ndarray:
        """Momentum handed to each acceptor cell (integral form).

        Correct: per **donor** normalisation ``Lambda_i = sum_j lambda_ij``, so
        ``sum_j q_j = sum_i u_i |S_i|`` exactly.  ``broken="per_node_norm"``
        divides by the per-acceptor sum instead -- the m5207 ME trap.
        """
        u = np.asarray(src_values, dtype=float)
        w = self._weights(src, tgt)                       # (nt, ns)
        fed = u * src.measures                            # donor momentum
        if self.broken == "per_node_norm":                # trap
            lam = w.sum(axis=1)
            return (w * fed[None, :]).sum(axis=1) / lam
        lam = w.sum(axis=0)                               # per donor (correct)
        return (w / lam[None, :]) @ fed

    def transfer(self, src: InterfaceSide, src_values: np.ndarray,
                 tgt: InterfaceSide) -> np.ndarray:
        """Acceptor cell means ``v_j = q_j / |T_j|``."""
        return self.gather(src, src_values, tgt) / tgt.measures

    def spread(self, src: InterfaceSide, tgt: InterfaceSide,
               q_tgt: np.ndarray) -> np.ndarray:
        """Reaction handed back to the donor cells (integral form).

        Correct: per **acceptor** normalisation ``Lambda'_j = sum_i lambda_ij``, so
        every acceptor hands back exactly the momentum it holds and
        ``sum_i w_i = -sum_j q_j``.  The ``per_node_norm`` control divides by the
        per-donor sum instead, which leaks momentum.
        """
        q = np.asarray(q_tgt, dtype=float)
        w = self._weights(src, tgt)                       # (nt, ns)
        if self.broken == "per_node_norm":                # trap
            lam = w.sum(axis=0)
            return -((w * q[:, None]).sum(axis=0) / lam)
        lam = w.sum(axis=1)                               # per acceptor (correct)
        return -((q / lam) @ w)

    def momentum_budget(self, src: InterfaceSide, src_values: np.ndarray,
                        tgt: InterfaceSide) -> dict:
        """Closed momentum budget: ``sent + returned`` must vanish."""
        u = np.asarray(src_values, dtype=float)
        sent = integral(u, src.measures)
        q = self.gather(src, u, tgt)
        recv = float(q.sum())
        returned = float(self.spread(src, tgt, q).sum())
        denom = abs(sent) if abs(sent) > 0.0 else 1.0
        return {
            "sent": sent,
            "gathered": recv,
            "returned": returned,
            "gather_residual": abs(recv - sent) / denom,
            "budget_residual": abs(sent + returned) / denom,
        }


def transfer_report(src: InterfaceSide, u: np.ndarray, tgt: InterfaceSide,
                    v: np.ndarray, exact: np.ndarray) -> dict:
    """Conservation residual and L2 error of one transfer."""
    sent = integral(u, src.measures)
    recv = integral(v, tgt.measures)
    denom = abs(sent) if abs(sent) > 0.0 else 1.0
    return {
        "sent": sent,
        "recv": recv,
        "conservation_residual": abs(recv - sent) / denom,
        "l2_error": l2_error(v, exact, tgt.measures),
    }
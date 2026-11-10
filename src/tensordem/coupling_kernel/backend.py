"""Array-namespace abstraction for the DEM coupling kernel.

Design contract (see ``docs/COUPLING_KERNEL_DEM_BACKEND.md``):

* **numpy is the reference backend.**  It is imported eagerly and is the only
  hard requirement of this package.
* **torch is an optional accelerator.**  It is imported lazily inside
  :func:`get_backend` and only when explicitly requested, so
  ``import tensordem.coupling_kernel`` never pulls torch in.

The kernel implementation is written once against this tiny operator set, which
is what makes the numpy and torch backends bit-comparable: both execute the
*same* sequence of element-wise IEEE-754 double operations in the same order.
Only whole-array reductions (``sum``) may differ, because numpy uses pairwise
summation while torch uses a tree reduction; that difference is ~1e-16 relative
and is measured explicitly by the backend-equivalence test.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ArrayBackend", "NumpyBackend", "TorchBackend", "get_backend", "available_backends"]


class ArrayBackend:
    """Common interface; concrete subclasses only wrap a ``xp`` module."""

    name = "base"

    # -- construction -----------------------------------------------------
    def asarray(self, values):
        raise NotImplementedError

    def zeros(self, n: int, dtype=None):
        raise NotImplementedError

    def index(self, values):
        """Convert a host (numpy) integer index array into a backend array."""
        raise NotImplementedError

    # -- host interop -----------------------------------------------------
    def to_numpy(self, a) -> np.ndarray:
        raise NotImplementedError

    def scalar(self, a) -> float:
        return float(self.to_numpy(a).reshape(-1)[0] if np.ndim(self.to_numpy(a)) else self.to_numpy(a))

    # -- element-wise ops -------------------------------------------------
    def sqrt(self, a):
        raise NotImplementedError

    def abs(self, a):
        raise NotImplementedError

    def maximum(self, a, b):
        raise NotImplementedError

    def minimum(self, a, b):
        raise NotImplementedError

    def power(self, a, p: float):
        raise NotImplementedError

    def where(self, cond, a, b):
        raise NotImplementedError

    def sum(self, a):
        raise NotImplementedError

    def rowdot(self, a, b):
        """Row-wise dot product of two (n, 3) arrays -> (n,)."""
        raise NotImplementedError

    def rowsquare_norm(self, a):
        return self.rowdot(a, a)

    def gather(self, a, idx):
        raise NotImplementedError

    def accumulate(self, n: int, idx, values):
        """Sum ``values`` into a length-``n`` array at ``idx`` (duplicates add)."""
        raise NotImplementedError

    def stack(self, items):
        raise NotImplementedError

    def assemble(self, cols):
        """Stack three (n,) components into an (n, 3) array."""
        raise NotImplementedError

    # -- introspection ----------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<ArrayBackend {self.name}>"


class NumpyBackend(ArrayBackend):
    name = "numpy"

    def __init__(self) -> None:
        self.xp = np
        self.dtype = np.float64

    def asarray(self, values):
        return np.asarray(values, dtype=np.float64)

    def zeros(self, n):
        return np.zeros(n, dtype=np.float64)

    def index(self, values):
        return np.asarray(values, dtype=np.int64)

    def to_numpy(self, a):
        return np.asarray(a)

    def sqrt(self, a):
        return np.sqrt(a)

    def abs(self, a):
        return np.abs(a)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def power(self, a, p):
        return np.power(a, p)

    def where(self, cond, a, b):
        return np.where(cond, a, b)

    def sum(self, a):
        return np.sum(a)

    def rowdot(self, a, b):
        return np.sum(a * b, axis=1) if np.ndim(a) > 1 else np.sum(a * b)

    def gather(self, a, idx):
        return a[idx]

    def accumulate(self, n, idx, values):
        return np.bincount(np.asarray(idx), weights=np.asarray(values), minlength=n)

    def stack(self, items):
        return np.stack(items, axis=0)

    def assemble(self, cols):
        return np.stack([np.asarray(c) for c in cols], axis=1)


class TorchBackend(ArrayBackend):
    name = "torch"

    def __init__(self, device: str = "cpu") -> None:
        import torch  # lazy: optional dependency

        self.torch = torch
        self.xp = torch
        self.device = torch.device(device)
        self.dtype = torch.float64

    def asarray(self, values):
        if isinstance(values, self.torch.Tensor):
            return values.to(device=self.device, dtype=self.dtype)
        return self.torch.as_tensor(np.asarray(values), dtype=self.dtype, device=self.device)

    def zeros(self, n):
        return self.torch.zeros(n, dtype=self.dtype, device=self.device)

    def index(self, values):
        return self.torch.as_tensor(np.asarray(values, dtype=np.int64), device=self.device)

    def to_numpy(self, a):
        return a.detach().cpu().numpy()

    def sqrt(self, a):
        return self.torch.sqrt(a)

    def abs(self, a):
        return self.torch.abs(a)

    def maximum(self, a, b):
        return self.torch.clamp_min(a, b) if isinstance(b, float) else self.torch.maximum(a, b)

    def minimum(self, a, b):
        return self.torch.clamp_max(a, b) if isinstance(b, float) else self.torch.minimum(a, b)

    def power(self, a, p):
        return self.torch.pow(a, p)

    def where(self, cond, a, b):
        return self.torch.where(cond, a, b)

    def sum(self, a):
        return self.torch.sum(a)

    def rowdot(self, a, b):
        if a.dim() > 1:
            return self.torch.sum(a * b, dim=1)
        return self.torch.sum(a * b)

    def gather(self, a, idx):
        return a[idx]

    def accumulate(self, n, idx, values):
        out = self.zeros(n)
        out.index_add_(0, idx, values)
        return out

    def stack(self, items):
        return self.torch.stack(list(items), dim=0)

    def assemble(self, cols):
        return self.torch.stack([c.reshape(-1) for c in cols], dim=1)


def get_backend(name: str = "numpy", device: str = "cpu") -> ArrayBackend:
    """Return an :class:`ArrayBackend`; ``torch`` is imported only on demand."""

    key = (name or "numpy").lower()
    if key in ("numpy", "np"):
        return NumpyBackend()
    if key in ("torch", "pytorch"):
        return TorchBackend(device=device)
    raise ValueError(f"unknown backend {name!r}; expected 'numpy' or 'torch'")


def available_backends() -> list[str]:
    have = ["numpy"]
    try:  # pragma: no cover - depends on environment
        import torch  # noqa: F401
    except Exception:
        return have
    have.append("torch")
    return have
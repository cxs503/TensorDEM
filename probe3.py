import time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
N, C = 100_000, 600_000
rng = np.random.default_rng(0)
pos = torch.from_numpy(rng.random((N, 3)) * 50)
i = torch.from_numpy(np.sort(rng.integers(0, N, C)))
j = torch.from_numpy(rng.integers(0, N, C))
kn = torch.from_numpy(rng.random(C) * 1e3)
rad = torch.full((N,), 0.5)


def elementwise():
    d = pos[j] - pos[i]
    dist = d.pow(2).sum(1).sqrt() + 1e-12
    ov = (rad[i] + rad[j] - dist).clamp(min=0)
    fn = kn * ov * ov.sqrt()
    return d * (fn / dist).unsqueeze(1)


f_pair = elementwise()


def with_index_add():
    out = torch.zeros(N, 3)
    out.index_add_(0, i, f_pair)
    return out


def elementwise_only():
    return elementwise()


cnt = torch.bincount(i, minlength=N)
print("has segment_reduce:", hasattr(torch, "segment_reduce"), "pairs", C)
for label, fn in [("elementwise", elementwise_only), ("+index_add", with_index_add)]:
    for p in [1, 8, 16, 32, 64, 128]:
        torch.set_num_threads(p)
        fn()
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        print(f"{label:12s} threads={p:4d} min={min(ts)*1e3:8.2f} ms  med={np.median(ts)*1e3:8.2f} ms")
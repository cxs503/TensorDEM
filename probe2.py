import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

N = 100_000
C = 600_000
rng = np.random.default_rng(0)
pos = rng.random((N, 3)) * 50
i = np.sort(rng.integers(0, N, C))
j = rng.integers(0, N, C)
kn = rng.random(C) * 1e3
prev = rng.random(C)
rad = np.full(N, 0.5)


def work(a, b):
    ii = i[a:b]
    jj = j[a:b]
    d = pos[jj] - pos[ii]
    dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]) + 1e-12
    ov = np.maximum(rad[ii] + rad[jj] - dist, 0.0)
    fn = kn[a:b] * ov * np.sqrt(ov) * (1.0 + 0.1 * prev[a:b])
    sc = fn / dist
    return d * sc[:, None]


def run(chunks, pool):
    edges = np.linspace(0, C, chunks + 1).astype(int)
    futs = [pool.submit(work, edges[k], edges[k + 1]) for k in range(chunks)]
    return [f.result() for f in futs]


for p in [1, 8, 16, 32, 64, 128]:
    pool = ThreadPoolExecutor(max_workers=max(1, p))
    run(p, pool)
    t = time.perf_counter()
    for _ in range(10):
        run(p, pool)
    dt = (time.perf_counter() - t) / 10
    print(f"np threads={p:4d} chunks={p:4d} per_call={dt * 1e3:8.2f} ms")
    pool.shutdown()
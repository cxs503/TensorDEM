"""Raw torch throughput microbenchmark (is the machine just loaded?)."""
import sys, time
import torch

for nt in (1, 8, 32, 128):
    torch.set_num_threads(nt)
    n = 2_000_000
    x = torch.randn(n, 3, dtype=torch.float64)
    y = torch.randn(n, 3, dtype=torch.float64)
    idx = torch.randint(0, 160_000, (n,))
    t0 = time.time()
    for _ in range(3):
        d = y - x
        dist = torch.linalg.vector_norm(d, dim=1)
        f = (d / dist.clamp_min(1e-30)[:, None]) * (dist - 1.0)[:, None]
        out = torch.zeros(160_000, 3, dtype=torch.float64)
        out.index_add_(0, idx, f)
        out.index_add_(0, idx, -f)
    t1 = time.time()
    print(f"threads={nt}: {(t1-t0)/3*1e3:.1f} ms per bond-pass")

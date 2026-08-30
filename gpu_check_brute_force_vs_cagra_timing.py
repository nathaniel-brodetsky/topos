import numpy as np
import cupy as cp
from cuvs.neighbors import cagra, brute_force
from benchmarks.timers import GPUTimer

rng = np.random.default_rng(0)
data = rng.normal(0, 1, (300, 5)).astype(np.float32)
data_gpu = cp.asarray(data)

cagra_index = cagra.build(cagra.IndexParams(), data_gpu)
bf_index = brute_force.build(data_gpu)

single_query = cp.asarray(data[0:1])

for _ in range(5):
    cagra.search(cagra.SearchParams(), cagra_index, single_query, k=2)
    brute_force.search(bf_index, single_query, k=2)
cp.cuda.Stream.null.synchronize()

n_trials = 200

cagra_times = []
for _ in range(n_trials):
    t = GPUTimer()
    with t.measure():
        cagra.search(cagra.SearchParams(), cagra_index, single_query, k=2)
    cagra_times.append(t.elapsed_ms)

bf_times = []
for _ in range(n_trials):
    t = GPUTimer()
    with t.measure():
        brute_force.search(bf_index, single_query, k=2)
    bf_times.append(t.elapsed_ms)

cagra_times = np.array(cagra_times)
bf_times = np.array(bf_times)

print(f"CAGRA:       mean={cagra_times.mean():.4f}ms  p50={np.percentile(cagra_times,50):.4f}ms  p99={np.percentile(cagra_times,99):.4f}ms")
print(f"brute_force: mean={bf_times.mean():.4f}ms  p50={np.percentile(bf_times,50):.4f}ms  p99={np.percentile(bf_times,99):.4f}ms")
print(f"speedup (CAGRA time / brute_force time): {cagra_times.mean() / bf_times.mean():.2f}x")

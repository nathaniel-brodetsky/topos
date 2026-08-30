import numpy as np
import cupy as cp
from cuvs.neighbors import cagra, brute_force
from benchmarks.timers import GPUTimer

def bench_brute_force(data_gpu, query, n_trials=200):
    index = brute_force.build(data_gpu)
    for _ in range(5):
        brute_force.search(index, query, k=2)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_trials):
        t = GPUTimer()
        with t.measure():
            brute_force.search(index, query, k=2)
        times.append(t.elapsed_ms)
    return np.array(times)


def bench_cagra(data_gpu, query, n_trials=200):
    build_params = cagra.IndexParams()
    index = cagra.build(build_params, data_gpu)
    search_params = cagra.SearchParams()
    for _ in range(5):
        cagra.search(search_params, index, query, k=2)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_trials):
        t = GPUTimer()
        with t.measure():
            cagra.search(search_params, index, query, k=2)
        times.append(t.elapsed_ms)
    return np.array(times)


rng = np.random.default_rng(0)
scales = [300, 1000, 5000, 10000, 50000, 100000, 500000]

print(f"{'n_points':>10} {'brute_force_us':>16} {'cagra_us':>12} {'winner':>12}")

results = []
for n in scales:
    data = rng.normal(0, 1, (n, 5)).astype(np.float32)
    data_gpu = cp.asarray(data)
    query = data_gpu[0:1]

    bf_times = bench_brute_force(data_gpu, query) * 1000.0
    cagra_times = bench_cagra(data_gpu, query) * 1000.0

    bf_mean = bf_times.mean()
    cagra_mean = cagra_times.mean()
    winner = "brute_force" if bf_mean < cagra_mean else "CAGRA"

    print(f"{n:>10} {bf_mean:>16.2f} {cagra_mean:>12.2f} {winner:>12}")

    results.append({
        "n_points": n,
        "brute_force_mean_us": float(bf_mean),
        "cagra_mean_us": float(cagra_mean),
        "winner": winner,
    })

    del data_gpu, query
    cp.get_default_memory_pool().free_all_blocks()

import json
with open("benchmarks/cagra_crossover.json", "w") as f:
    json.dump({"results": results}, f, indent=2)

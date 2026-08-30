import numpy as np
import cupy as cp
from cuvs.neighbors import cagra, brute_force
from benchmarks.timers import GPUTimer
import json

def bench_brute_force(data_gpu, query, n_trials=100):
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


def bench_cagra(data_gpu, query, n_trials=100):
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
n_points = 500000
query_batch_sizes = [1, 10, 100, 1000, 10000]

data = rng.normal(0, 1, (n_points, 5)).astype(np.float32)
data_gpu = cp.asarray(data)

print(f"{'query_batch':>12} {'brute_force_us':>16} {'cagra_us':>12} {'winner':>12}")

results = []
for qbs in query_batch_sizes:
    query = data_gpu[:qbs]

    bf_times = bench_brute_force(data_gpu, query) * 1000.0
    cagra_times = bench_cagra(data_gpu, query) * 1000.0

    bf_mean = bf_times.mean()
    cagra_mean = cagra_times.mean()
    winner = "brute_force" if bf_mean < cagra_mean else "CAGRA"

    print(f"{qbs:>12} {bf_mean:>16.2f} {cagra_mean:>12.2f} {winner:>12}")

    results.append({
        "query_batch_size": qbs,
        "brute_force_mean_us": float(bf_mean),
        "cagra_mean_us": float(cagra_mean),
        "winner": winner,
    })

with open("benchmarks/cagra_crossover_batched_query.json", "w") as f:
    json.dump({"n_points": n_points, "results": results}, f, indent=2)

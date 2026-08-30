import numpy as np
import cupy as cp
from cuvs.neighbors import brute_force
from sklearn.neighbors import NearestNeighbors
import time

rng = np.random.default_rng(0)
n_points = 500000
data = rng.normal(0, 1, (n_points, 5)).astype(np.float32)

nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
nn.fit(data)
query_cpu = data[0:1]

for _ in range(5):
    nn.kneighbors(query_cpu)

n_trials = 100
cpu_times = []
for _ in range(n_trials):
    start = time.perf_counter()
    nn.kneighbors(query_cpu)
    end = time.perf_counter()
    cpu_times.append((end - start) * 1e6)

cpu_times = np.array(cpu_times)
print(f"CPU (sklearn ball_tree) single-query: mean={cpu_times.mean():.2f}us  p50={np.percentile(cpu_times,50):.2f}us")

import numpy as np
import cupy as cp
from cuvs.neighbors import brute_force

rng = np.random.default_rng(0)
data = rng.normal(0, 1, (300, 5)).astype(np.float32)
data_gpu = cp.asarray(data)

index = brute_force.build(data_gpu)
print("brute_force build works, index type:", type(index))

query = cp.asarray(data[:5])

distances, indices = brute_force.search(index, query, k=2)
print("search works")
print("distances type:", type(distances))

distances_cp = cp.asarray(distances)
indices_cp = cp.asarray(indices)

print("sample distances (host):", cp.asnumpy(distances_cp)[:3])
print("sample indices (host):", cp.asnumpy(indices_cp)[:3])

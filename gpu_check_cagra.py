import numpy as np
import cupy as cp
from cuvs.neighbors import cagra

rng = np.random.default_rng(0)
data = rng.normal(0, 1, (300, 5)).astype(np.float32)
data_gpu = cp.asarray(data)

build_params = cagra.IndexParams()
index = cagra.build(build_params, data_gpu)
print("build works, index type:", type(index))

query = cp.asarray(data[:5])

search_params = cagra.SearchParams()
distances, indices = cagra.search(search_params, index, query, k=2)
print("search works")
print("raw distances type:", type(distances))

distances_cp = cp.asarray(distances)
indices_cp = cp.asarray(indices)

print("distances_cp type:", type(distances_cp))
print("distances_cp shape:", distances_cp.shape)
print("sample distances (host):", cp.asnumpy(distances_cp)[:3])
print("sample indices (host):", cp.asnumpy(indices_cp)[:3])
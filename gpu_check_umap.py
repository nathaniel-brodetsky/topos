import numpy as np
import cupy as cp
from cuml.manifold import UMAP


def make_calibration_vectors(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 11))
    cluster_b = rng.normal(loc=10.0, scale=0.5, size=(n // 2, 11))
    return np.vstack([cluster_a, cluster_b]).astype(np.float32)


vectors = make_calibration_vectors()
vectors_gpu = cp.asarray(vectors)

model = UMAP(n_components=5, random_state=42)
embedding = model.fit_transform(vectors_gpu)

print("embedding type:", type(embedding))
print("embedding shape:", embedding.shape)

new_vectors = make_calibration_vectors(n=20, seed=1)
new_vectors_gpu = cp.asarray(new_vectors)
transformed = model.transform(new_vectors_gpu)

print("transformed type:", type(transformed))
print("transformed shape:", transformed.shape)
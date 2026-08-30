import numpy as np
import cupy as cp
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from cuml.cluster.hdbscan import approximate_predict


def make_calibration_vectors(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 11))
    cluster_b = rng.normal(loc=10.0, scale=0.5, size=(n // 2, 11))
    return np.vstack([cluster_a, cluster_b]).astype(np.float32)


vectors = make_calibration_vectors()
vectors_gpu = cp.asarray(vectors)

umap_model = UMAP(n_components=5, random_state=42)
embedding = umap_model.fit_transform(vectors_gpu)

clusterer = HDBSCAN(min_cluster_size=15, prediction_data=True)
clusterer.fit(embedding)

labels_host = cp.asnumpy(clusterer.labels_)
print("unique labels:", np.unique(labels_host, return_counts=True))

new_vectors = make_calibration_vectors(n=20, seed=1)
new_vectors_gpu = cp.asarray(new_vectors)
new_embedding = umap_model.transform(new_vectors_gpu)

try:
    predicted_labels, strengths = approximate_predict(clusterer, new_embedding)
    print("approximate_predict works, output type:", type(predicted_labels))
    predicted_host = cp.asnumpy(predicted_labels) if hasattr(predicted_labels, "get") else np.asarray(predicted_labels)
    print("predicted labels:", predicted_host)
except Exception as e:
    print("approximate_predict FAILED:", type(e).__name__, str(e))
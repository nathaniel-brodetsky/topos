import cupy as cp
import numpy as np
from benchmarks.timers import GPUTimer

a = cp.random.normal(size=(20, 20)).astype(cp.float32)
b = cp.random.normal(size=(20, 20)).astype(cp.float32)

def eager_compute(a, b):
    da = a - b
    f = a @ da - da @ a
    return f

for _ in range(5):
    eager_compute(a, b)
cp.cuda.Stream.null.synchronize()

n_trials = 500
eager_times = []
for _ in range(n_trials):
    t = GPUTimer()
    with t.measure():
        eager_compute(a, b)
    eager_times.append(t.elapsed_ms)

eager_times = np.array(eager_times)
print(f"eager: mean={eager_times.mean():.5f}ms  p50={np.percentile(eager_times,50):.5f}ms")

try:
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        f = eager_compute(a, b)
        graph = stream.end_capture()
    print("graph capture succeeded, graph type:", type(graph))

    for _ in range(5):
        graph.launch(stream=stream)
    stream.synchronize()

    graph_times = []
    for _ in range(n_trials):
        t = GPUTimer()
        with t.measure():
            graph.launch(stream=stream)
        stream.synchronize()
        graph_times.append(t.elapsed_ms)

    graph_times = np.array(graph_times)
    print(f"graph: mean={graph_times.mean():.5f}ms  p50={np.percentile(graph_times,50):.5f}ms")
    print(f"speedup: {eager_times.mean() / graph_times.mean():.2f}x")

except Exception as e:
    print("GRAPH CAPTURE FAILED:", type(e).__name__, str(e))

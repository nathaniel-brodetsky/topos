import numpy as np
import cupy as cp

from layer0_ingest import Layer0Pipeline
from layer1_gauge import compute_gauge_state
from layer2_topology.gpu import UMAPProjectorGPU, RegimeClustererGPU, AnomalyDetectorGPU
from layer2_topology import RegimeLabeler
from layer4_decision import decide
from .timers import GPUTimer


def run_gpu_tick_by_tick(config: dict, build_feed_fn, n_calibration: int = 300, n_live: int = 200, momentum_window: int = 5):
    layer0_cfg = config["layer0"]

    feed = build_feed_fn(config)
    pipeline = Layer0Pipeline(
        v_threshold=layer0_cfg["v_threshold"],
        eps_stabilizer=layer0_cfg["eps_stabilizer"],
        feed=feed,
    )

    a_prev = None
    gauge_states = []
    mid_price_history = []

    for tau, a, dv, mid_price in pipeline.run():
        mid_price_history.append(mid_price)
        if a_prev is not None:
            if len(mid_price_history) > momentum_window:
                momentum = mid_price_history[-1] - mid_price_history[-1 - momentum_window]
            else:
                momentum = 0.0
            gauge_states.append(compute_gauge_state(a, a_prev, dv, price_momentum=momentum))
        a_prev = a
        if len(gauge_states) >= n_calibration + n_live:
            break

    vectors = np.stack([gs.to_vector() for gs in gauge_states]).astype(np.float32)
    calibration_vectors = vectors[:n_calibration]
    live_vectors = vectors[n_calibration:]
    calibration_gauge_states = gauge_states[:n_calibration]

    projector = UMAPProjectorGPU(n_components=5)
    calibration_embedding = projector.fit(calibration_vectors)

    clusterer = RegimeClustererGPU(min_cluster_size=15)
    calibration_labels = clusterer.fit(calibration_embedding)

    labeler = RegimeLabeler()
    labeler.fit(calibration_gauge_states, cp.asnumpy(calibration_labels))

    detector = AnomalyDetectorGPU(anomaly_percentile=99.0)
    detector.fit(calibration_embedding)

    warmup_vec = live_vectors[0:1]
    for _ in range(5):
        emb = projector.transform(warmup_vec)
        lbl = clusterer.predict(emb)
        rs = detector.evaluate(emb, lbl)
        decide(labeler.resolve(rs[0].topology_id), rs[0].anomaly_triggered)
    cp.cuda.Stream.null.synchronize()

    latencies_ms = []

    for vec in live_vectors:
        vec_batch = vec[None, :]
        timer = GPUTimer()
        with timer.measure():
            embedding = projector.transform(vec_batch)
            label = clusterer.predict(embedding)
            regime_state = detector.evaluate(embedding, label)[0]
            semantic = labeler.resolve(regime_state.topology_id)
            decide(semantic, regime_state.anomaly_triggered)
        latencies_ms.append(timer.elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    return {
        "backend": "cupy_tick_by_tick",
        "n_ticks": int(len(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
        "ticks_per_sec": float(1000.0 / np.mean(latencies_ms)),
    }


import json


def save_gpu_report(report: dict, path: str = "benchmarks/last_gpu_report.json"):
    report_with_context = dict(report)
    report_with_context["known_limitations"] = [
        "single-vector (batch size 1) tick-by-tick GPU latency is measured here, deliberately matching the CPU benchmark's per-tick methodology",
        "GPU tick-by-tick (mean ~3.5ms) is SLOWER than CPU tick-by-tick (mean ~2.26ms) on this hardware and workload size, due to CUDA kernel launch and host-device sync overhead not being amortized at batch size 1",
        "batched GPU processing (200 vectors in a single call) measured ~0.08ms/tick, ~28x faster than CPU — batching amortizes launch overhead but changes the processing model from tick-by-tick to windowed",
        "this uses Python cuML/cuVS wrappers, not the C++/CUDA execution plane specified in the architecture; the spec explicitly requires no Python/interpreter in the hot path for production tick-to-trade latency, which this reference implementation does not provide",
        "the <1us tick-to-trade target from the specification requires a native C++/CUDA execution plane and DPU-level batching/pipelining not implemented here",
    ]
    with open(path, "w") as f:
        json.dump(report_with_context, f, indent=2)


if __name__ == "__main__":
    import yaml
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import build_feed

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    report = run_gpu_tick_by_tick(config, build_feed)
    for k, v in report.items():
        print(k, v)
    
    save_gpu_report(report)
    print("GPU tick-by-tick reference run on L40S. Single-vector batches, CUDA Event timed, warmed up.")
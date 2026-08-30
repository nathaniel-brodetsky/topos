import numpy as np

from layer0_ingest import Layer0Pipeline, MockFeed
from layer1_gauge import compute_gauge_state
from layer2_topology import UMAPProjector, RegimeClusterer, AnomalyDetector, RegimeLabeler
from layer4_decision import decide
from .timers import make_timer
import json


def save_report(report: dict, path: str = "benchmarks/last_cpu_report.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

def run(backend: str = "numpy", n_calibration: int = 300, n_live: int = 2000, v_threshold: float = 1000.0):
    feed = MockFeed(seed=42, mid_price_start=100.0, tick_size=0.01, base_volume=50.0, volume_jitter=20.0)
    pipeline = Layer0Pipeline(v_threshold=v_threshold, eps_stabilizer=0.01, feed=feed)

    a_prev = None
    calibration_gauge_states = []
    live_ticks = []

    for tau, a, dv, mid_price in pipeline.run():
        if a_prev is not None:
            if len(calibration_gauge_states) < n_calibration:
                calibration_gauge_states.append(compute_gauge_state(a, a_prev, dv))
            else:
                live_ticks.append((a_prev.copy(), a.copy(), dv.copy()))
        a_prev = a
        if len(live_ticks) >= n_live:
            break

    calibration_vectors = np.stack([gs.to_vector() for gs in calibration_gauge_states])

    projector = UMAPProjector(n_components=5)
    calibration_embedding = projector.fit(calibration_vectors)

    clusterer = RegimeClusterer(min_cluster_size=15)
    calibration_labels = clusterer.fit(calibration_embedding)

    labeler = RegimeLabeler()
    labeler.fit(calibration_gauge_states, calibration_labels)

    detector = AnomalyDetector(anomaly_percentile=99.0)
    detector.fit(calibration_embedding)

    latencies_ms = []

    for a_prev_tick, a_tick, dv in live_ticks:
        timer = make_timer(backend)
        with timer.measure():
            gs = compute_gauge_state(a_tick, a_prev_tick, dv)
            vec = gs.to_vector()[None, :]
            embedding = projector.transform(vec)
            label = clusterer.predict(embedding)
            regime_state = detector.evaluate(embedding, label)[0]
            semantic = labeler.resolve(regime_state.topology_id)
            decide(semantic, regime_state.anomaly_triggered)
        latencies_ms.append(timer.elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    return {
        "backend": backend,
        "n_ticks": int(len(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
        "ticks_per_sec": float(1000.0 / np.mean(latencies_ms)),
    }


if __name__ == "__main__":
    report = run()
    for k, v in report.items():
        print(k, v)
    print("CPU reference run. Not representative of target DPU/GPU hardware.")
    save_report(report)
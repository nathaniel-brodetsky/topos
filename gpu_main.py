import numpy as np
import cupy as cp
import yaml

from layer0_ingest import Layer0Pipeline, MockFeed, RegimeMockFeed, default_regimes
from layer1_gauge import compute_gauge_state
from layer2_topology.gpu import UMAPProjectorGPU, RegimeClustererGPU, AnomalyDetectorGPU
from layer2_topology import RegimeLabeler, TOXIC_VORTEX
from layer4_decision import decide, ValueModel
from layer4_decision.value_model import compute_forward_returns
from validation import (
    check_cluster_degeneracy,
    check_calibration_leakage,
    summarize_decisions,
    check_value_model_predictions,
)
from benchmarks.timers import GPUTimer


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_feed(config: dict):
    regime_cfg = config.get("regime_mock_feed", {})
    if regime_cfg.get("enabled", False):
        return RegimeMockFeed(
            seed=regime_cfg["seed"],
            mid_price_start=regime_cfg["mid_price_start"],
            tick_size=regime_cfg["tick_size"],
            regimes=default_regimes(),
        )
    mock_cfg = config["mock_feed"]
    return MockFeed(
        seed=mock_cfg["seed"],
        mid_price_start=mock_cfg["mid_price_start"],
        tick_size=mock_cfg["tick_size"],
        base_volume=mock_cfg["base_volume"],
        volume_jitter=mock_cfg["volume_jitter"],
    )


def run_gpu_pipeline(config: dict, n_calibration: int, n_live: int, momentum_window: int = 5):
    layer0_cfg = config["layer0"]

    feed = build_feed(config)
    pipeline = Layer0Pipeline(
        v_threshold=layer0_cfg["v_threshold"],
        eps_stabilizer=layer0_cfg["eps_stabilizer"],
        feed=feed,
    )

    a_prev = None
    gauge_states = []
    mid_prices = []
    mid_price_history = []

    for tau, a, dv, mid_price in pipeline.run():
        mid_price_history.append(mid_price)
        if a_prev is not None:
            if len(mid_price_history) > momentum_window:
                momentum = mid_price_history[-1] - mid_price_history[-1 - momentum_window]
            else:
                momentum = 0.0
            gauge_states.append(compute_gauge_state(a, a_prev, dv, price_momentum=momentum))
            mid_prices.append(mid_price)
        a_prev = a
        if len(gauge_states) >= n_calibration + n_live:
            break

    vectors = np.stack([gs.to_vector() for gs in gauge_states]).astype(np.float32)
    mid_prices = np.array(mid_prices)

    calibration_vectors = vectors[:n_calibration]
    live_vectors = vectors[n_calibration:]
    calibration_gauge_states = gauge_states[:n_calibration]

    calibration_mid_prices = mid_prices[:n_calibration]
    live_mid_prices = mid_prices[n_calibration:]

    projector = UMAPProjectorGPU(n_components=5)
    calibration_embedding = projector.fit(calibration_vectors)

    clusterer = RegimeClustererGPU(min_cluster_size=15)
    calibration_labels = clusterer.fit(calibration_embedding)

    calibration_labels_host = cp.asnumpy(calibration_labels)
    labeler = RegimeLabeler()
    labeler.fit(calibration_gauge_states, calibration_labels_host)

    detector = AnomalyDetectorGPU(anomaly_percentile=99.0)
    epsilon = detector.fit(calibration_embedding)

    search_params_distances, _ = cp.asarray(calibration_embedding), None
    from cuvs.neighbors import cagra
    search_params = cagra.SearchParams()
    calib_distances_raw, _ = cagra.search(search_params, detector._index, cp.asarray(calibration_embedding, dtype=cp.float32), k=2)
    calibration_self_distances = cp.asnumpy(cp.asarray(calib_distances_raw))[:, 1]

    timer = GPUTimer()
    with timer.measure():
        live_embedding = projector.transform(live_vectors)
        live_labels = clusterer.predict(live_embedding)
        regime_states = detector.evaluate(live_embedding, live_labels)
    gpu_transform_ms = timer.elapsed_ms

    live_distances = np.array([rs.distance for rs in regime_states])

    value_model = ValueModel(target_horizon_ticks=2)
    value_model.train(calibration_vectors, calibration_mid_prices)
    live_value_predictions = value_model.predict(live_vectors)
    live_actual_returns = compute_forward_returns(live_mid_prices, value_model.target_horizon_ticks)
    value_check = check_value_model_predictions(live_value_predictions, live_actual_returns)

    decisions = []
    for rs in regime_states:
        semantic = labeler.resolve(rs.topology_id)
        decisions.append(decide(semantic, rs.anomaly_triggered))

    degeneracy = check_cluster_degeneracy(calibration_labels_host)
    leakage = check_calibration_leakage(epsilon, calibration_self_distances, live_distances)
    backtest = summarize_decisions(decisions)

    return {
        "degeneracy": degeneracy,
        "leakage": leakage,
        "backtest": backtest,
        "value_check": value_check,
        "epsilon_regime": epsilon,
        "gpu_transform_ms_for_full_live_batch": gpu_transform_ms,
        "n_live": n_live,
    }


if __name__ == "__main__":
    config = load_config()
    results = run_gpu_pipeline(config, n_calibration=300, n_live=200)

    print(results["degeneracy"])
    print(results["leakage"])
    print(results["backtest"])
    print(results["value_check"])
    print("GPU transform+predict+evaluate time for full live batch (ms):", results["gpu_transform_ms_for_full_live_batch"])
    print("per-tick avg (ms):", results["gpu_transform_ms_for_full_live_batch"] / results["n_live"])
import argparse
import numpy as np
import yaml

from layer0_ingest import Layer0Pipeline, MockFeed, RegimeMockFeed, default_regimes
from layer1_gauge import compute_gauge_state
from layer2_topology import (
    RegimeLabeler,
    build_initial_snapshot,
    AttractorMap,
)
from layer4_decision import decide, ValueModel
from layer4_decision.value_model import compute_forward_returns
from validation import (
    check_cluster_degeneracy,
    check_calibration_leakage,
    summarize_decisions,
    check_value_model_predictions,
)


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


def run_pipeline(config: dict, n_calibration: int, n_live: int, momentum_window: int = 5):
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

    vectors = np.stack([gs.to_vector() for gs in gauge_states])
    mid_prices = np.array(mid_prices)

    calibration_vectors = vectors[:n_calibration]
    live_vectors = vectors[n_calibration:]
    calibration_gauge_states = gauge_states[:n_calibration]

    calibration_mid_prices = mid_prices[:n_calibration]
    live_mid_prices = mid_prices[n_calibration:]

    initial_snapshot = build_initial_snapshot(
        calibration_vectors,
        n_components=5,
        min_cluster_size=15,
        anomaly_percentile=99.0,
    )
    attractor_map = AttractorMap(initial_snapshot)

    labeler = RegimeLabeler()
    labeler.fit(calibration_gauge_states, initial_snapshot.clusterer._clusterer.labels_)

    calibration_embedding = initial_snapshot.projector._model.embedding_
    calibration_self_distances, _ = initial_snapshot.detector._nn.kneighbors(calibration_embedding, n_neighbors=2)
    calibration_self_distances = calibration_self_distances[:, 1]

    live_distances = []
    regime_states = []
    decisions = []
    refit_triggers = 0
    snapshot_versions_seen = set()

    rolling_window = list(calibration_vectors)
    max_window_size = n_calibration

    for i, vec in enumerate(live_vectors):
        snapshot = attractor_map.current()
        snapshot_versions_seen.add(snapshot.version)

        embedding = snapshot.projector.transform(vec[None, :])
        distance, _ = snapshot.detector._nn.kneighbors(embedding, n_neighbors=1)
        distance = float(distance[0, 0])
        live_distances.append(distance)

        label = snapshot.clusterer.predict(embedding)
        rs = snapshot.detector.evaluate(embedding, label)[0]
        regime_states.append(rs)

        semantic = labeler.resolve(rs.topology_id)
        decision = decide(semantic, rs.anomaly_triggered)
        decisions.append(decision)

        rolling_window.append(vec)
        if len(rolling_window) > max_window_size:
            rolling_window.pop(0)

        if rs.anomaly_triggered:
            thread = attractor_map.trigger_async_refit(
                np.stack(rolling_window),
                min_cluster_size=15,
                anomaly_percentile=99.0,
            )
            if thread is not None:
                refit_triggers += 1

    live_distances = np.array(live_distances)

    value_model = ValueModel(target_horizon_ticks=2)
    value_model.train(calibration_vectors, calibration_mid_prices)
    live_value_predictions = value_model.predict(live_vectors)
    live_actual_returns = compute_forward_returns(live_mid_prices, value_model.target_horizon_ticks)
    value_check = check_value_model_predictions(live_value_predictions, live_actual_returns)

    degeneracy = check_cluster_degeneracy(initial_snapshot.clusterer._clusterer.labels_)
    leakage = check_calibration_leakage(
        initial_snapshot.detector.epsilon_regime,
        calibration_self_distances,
        live_distances,
    )
    backtest = summarize_decisions(decisions)

    return {
        "degeneracy": degeneracy,
        "leakage": leakage,
        "backtest": backtest,
        "epsilon_regime": initial_snapshot.detector.epsilon_regime,
        "n_calibration": n_calibration,
        "n_live": n_live,
        "mid_prices": mid_prices,
        "vectors": vectors,
        "live_value_predictions": live_value_predictions,
        "value_check": value_check,
        "refit_triggers": refit_triggers,
        "final_snapshot_version": attractor_map.current().version,
        "snapshot_versions_seen_during_live": sorted(snapshot_versions_seen),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-calibration", type=int, default=300)
    parser.add_argument("--n-live", type=int, default=200)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_pipeline(config, args.n_calibration, args.n_live)

    print(results["degeneracy"])
    print(results["leakage"])
    print(results["backtest"])
    print(results["value_check"])
    print("refit_triggers:", results["refit_triggers"])
    print("final_snapshot_version:", results["final_snapshot_version"])
    print("snapshot_versions_seen_during_live:", results["snapshot_versions_seen_during_live"])

    if args.report:
        from reporting import build_report, plot_decision_distribution
        report = build_report(results)
        plot_decision_distribution(report["decision_backtest"]["decision_counts"])


if __name__ == "__main__":
    main()
import argparse
import numpy as np
import yaml

from layer0_ingest import Layer0Pipeline, MockFeed
from layer1_gauge import compute_gauge_state
from layer2_topology import UMAPProjector, RegimeClusterer, AnomalyDetector, RegimeLabeler
from layer4_decision import decide
from validation import check_cluster_degeneracy, check_calibration_leakage, summarize_decisions


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict, n_calibration: int, n_live: int):
    mock_cfg = config["mock_feed"]
    layer0_cfg = config["layer0"]

    feed = MockFeed(
        seed=mock_cfg["seed"],
        mid_price_start=mock_cfg["mid_price_start"],
        tick_size=mock_cfg["tick_size"],
        base_volume=mock_cfg["base_volume"],
        volume_jitter=mock_cfg["volume_jitter"],
    )
    pipeline = Layer0Pipeline(
        v_threshold=layer0_cfg["v_threshold"],
        eps_stabilizer=layer0_cfg["eps_stabilizer"],
        feed=feed,
    )

    rng = np.random.default_rng(mock_cfg["seed"])
    a_prev = None
    gauge_states = []

    for tau, a in pipeline.run():
        if a_prev is not None:
            dv = rng.normal(0, 1, 20)
            gauge_states.append(compute_gauge_state(a, a_prev, dv))
        a_prev = a
        if len(gauge_states) >= n_calibration + n_live:
            break

    vectors = np.stack([gs.to_vector() for gs in gauge_states])
    calibration_vectors = vectors[:n_calibration]
    live_vectors = vectors[n_calibration:]
    calibration_gauge_states = gauge_states[:n_calibration]

    projector = UMAPProjector(n_components=5)
    calibration_embedding = projector.fit(calibration_vectors)

    clusterer = RegimeClusterer(min_cluster_size=15)
    calibration_labels = clusterer.fit(calibration_embedding)

    labeler = RegimeLabeler()
    labeler.fit(calibration_gauge_states, calibration_labels)

    detector = AnomalyDetector(anomaly_percentile=99.0)
    epsilon = detector.fit(calibration_embedding)

    calibration_self_distances, _ = detector._nn.kneighbors(calibration_embedding, n_neighbors=2)
    calibration_self_distances = calibration_self_distances[:, 1]

    live_embedding = projector.transform(live_vectors)
    live_distances, _ = detector._nn.kneighbors(live_embedding, n_neighbors=1)
    live_distances = live_distances[:, 0]

    live_labels = clusterer.predict(live_embedding)
    regime_states = detector.evaluate(live_embedding, live_labels)

    decisions = []
    for rs in regime_states:
        semantic = labeler.resolve(rs.topology_id)
        decisions.append(decide(semantic, rs.anomaly_triggered))

    degeneracy = check_cluster_degeneracy(calibration_labels)
    leakage = check_calibration_leakage(epsilon, calibration_self_distances, live_distances)
    backtest = summarize_decisions(decisions)

    return {
        "degeneracy": degeneracy,
        "leakage": leakage,
        "backtest": backtest,
        "epsilon_regime": epsilon,
        "n_calibration": n_calibration,
        "n_live": n_live,
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

    if args.report:
        from reporting import build_report
        build_report(results)


if __name__ == "__main__":
    main()
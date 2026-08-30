import numpy as np

from validation import (
    check_cluster_degeneracy,
    check_calibration_leakage,
    summarize_decisions,
    check_value_model_predictions,
)
from layer4_decision import Decision


def test_check_cluster_degeneracy_flags_all_noise():
    labels = np.full(100, -1)
    report = check_cluster_degeneracy(labels)
    assert report.is_degenerate is True
    assert report.n_unique_clusters == 0


def test_check_cluster_degeneracy_passes_on_healthy_clusters():
    labels = np.array([0] * 40 + [1] * 40 + [-1] * 20)
    report = check_cluster_degeneracy(labels)
    assert report.is_degenerate is False
    assert report.n_unique_clusters == 2


def test_check_calibration_leakage_no_leak_normal_case():
    rng = np.random.default_rng(0)
    calibration_distances = rng.uniform(0, 1, 300)
    live_distances = rng.uniform(0, 1, 200)
    epsilon = float(np.percentile(calibration_distances, 99))

    report = check_calibration_leakage(epsilon, calibration_distances, live_distances)
    assert report.leakage_suspected is False


def test_summarize_decisions_detects_degenerate_all_same():
    decisions = [Decision.CANCEL_ALL] * 100
    summary = summarize_decisions(decisions)
    assert summary.is_decision_table_degenerate is True


def test_summarize_decisions_healthy_distribution():
    decisions = [Decision.CANCEL_ALL] * 25 + [Decision.MARKET_TAKER] * 25 + \
                [Decision.LIQUIDITY_PROVISION] * 25 + [Decision.HOLD] * 25
    summary = summarize_decisions(decisions)
    assert summary.is_decision_table_degenerate is False


def test_check_value_model_predictions_degenerate_on_constant_output():
    predictions = np.ones(100) * 0.5
    actual = np.random.default_rng(0).normal(0, 1, 100)
    report = check_value_model_predictions(predictions, actual)
    assert report.is_degenerate is True


def test_check_value_model_predictions_healthy_correlation():
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 1, 200)
    predictions = actual * 2.0 + rng.normal(0, 0.1, 200)
    report = check_value_model_predictions(predictions, actual)
    assert report.is_degenerate is False
    assert report.correlation_with_actual > 0.9
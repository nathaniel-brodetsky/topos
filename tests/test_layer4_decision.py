import numpy as np

from layer4_decision import decide, Decision
from layer4_decision.value_model import ValueModel, compute_forward_returns
from layer2_topology import TOXIC_VORTEX


def test_decide_cancel_all_on_toxic_vortex():
    assert decide(TOXIC_VORTEX, anomaly_triggered=True) == Decision.CANCEL_ALL


def test_decide_cancel_all_on_anomaly_regardless_of_topology_id():
    assert decide("0", anomaly_triggered=True) == Decision.CANCEL_ALL


def test_decide_market_taker_on_impulse():
    assert decide("IMPULSE", anomaly_triggered=False) == Decision.MARKET_TAKER


def test_decide_liquidity_provision_on_equilibrium():
    assert decide("EQUILIBRIUM", anomaly_triggered=False) == Decision.LIQUIDITY_PROVISION


def test_decide_hold_on_unknown():
    assert decide("UNKNOWN", anomaly_triggered=False) == Decision.HOLD


def test_compute_forward_returns_shape_and_nan_tail():
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    returns = compute_forward_returns(prices, horizon_ticks=2)
    assert returns.shape == (5,)
    assert np.isnan(returns[-1])
    assert np.isnan(returns[-2])
    assert returns[0] == 2.0


def test_value_model_train_predict_roundtrip():
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (100, 11))
    prices = np.cumsum(rng.normal(0, 0.01, 100)) + 100.0

    model = ValueModel(target_horizon_ticks=2)
    model.train(features, prices)
    predictions = model.predict(features)

    assert predictions.shape == (100,)
    assert np.all(np.isfinite(predictions))


def test_value_model_predict_before_train_raises():
    import pytest
    model = ValueModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 11)))
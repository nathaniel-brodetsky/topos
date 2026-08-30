import numpy as np
import pytest

from layer0_ingest import VolumeClock, build_adjacency_matrix, normalize_and_pack_fp16
from layer0_ingest import MockFeed, Layer0Pipeline
from layer0_ingest.regime_mock_feed import RegimeMockFeed, default_regimes


def test_volume_clock_fires_at_threshold():
    clock = VolumeClock(v_threshold=100.0)
    assert clock.accumulate(50.0) is False
    assert clock.tau == 0
    assert clock.accumulate(50.0) is True
    assert clock.tau == 1


def test_volume_clock_resets_after_fire():
    clock = VolumeClock(v_threshold=100.0)
    clock.accumulate(150.0)
    assert clock.tau == 1
    assert clock.accumulate(50.0) is False
    assert clock.accumulate(50.0) is True
    assert clock.tau == 2


def test_adjacency_matrix_shape_and_antisymmetry():
    dv = np.random.default_rng(0).normal(0, 1, 20)
    prices = np.arange(20, dtype=np.float64)
    a = build_adjacency_matrix(dv, prices, eps=0.01)
    assert a.shape == (20, 20)
    assert np.allclose(a, -a.T, atol=1e-5)
    assert np.allclose(np.diag(a), 0.0)


def test_adjacency_matrix_rejects_wrong_shape():
    with pytest.raises(ValueError):
        build_adjacency_matrix(np.zeros(19), np.zeros(20), eps=0.01)


def test_normalize_and_pack_fp16_bounds():
    a = np.array([[0.0, 5.0], [-5.0, 0.0]], dtype=np.float32)
    packed = normalize_and_pack_fp16(a)
    assert packed.dtype == np.float16
    assert np.max(np.abs(packed)) <= 1.0 + 1e-3


def test_normalize_and_pack_fp16_handles_all_zero():
    a = np.zeros((20, 20), dtype=np.float32)
    packed = normalize_and_pack_fp16(a)
    assert np.all(packed == 0.0)


def test_layer0_pipeline_yields_correct_shapes():
    feed = MockFeed(seed=1, mid_price_start=100.0, tick_size=0.01, base_volume=50.0, volume_jitter=10.0)
    pipeline = Layer0Pipeline(v_threshold=500.0, eps_stabilizer=0.01, feed=feed)

    count = 0
    for tau, a, dv, mid_price in pipeline.run():
        assert a.shape == (20, 20)
        assert a.dtype == np.float16
        assert dv.shape == (20,)
        assert isinstance(mid_price, float)
        count += 1
        if count >= 5:
            break
    assert count == 5


def test_regime_mock_feed_transitions_between_regimes():
    feed = RegimeMockFeed(seed=1, mid_price_start=100.0, tick_size=0.01, regimes=default_regimes())
    seen_indices = set()
    stream = feed.stream()
    for _ in range(500):
        next(stream)
        seen_indices.add(feed._current_idx)
    assert len(seen_indices) > 1
import numpy as np

from layer1_gauge import compute_commutator, commutator_norm, curl_component, compute_directed_pressure
from layer1_gauge import compute_gauge_state


def test_commutator_is_zero_for_identical_matrices():
    a = np.random.default_rng(0).normal(0, 1, (20, 20)).astype(np.float16)
    f = compute_commutator(a, a)
    assert np.allclose(f.astype(np.float32), 0.0, atol=1e-2)


def test_commutator_norm_nonzero_for_different_matrices():
    rng = np.random.default_rng(0)
    a1 = rng.normal(0, 1, (20, 20)).astype(np.float16)
    a2 = rng.normal(0, 1, (20, 20)).astype(np.float16)
    f = compute_commutator(a1, a2)
    assert commutator_norm(f) > 0.0


def test_gradient_component_is_structurally_zero_for_antisymmetric_input():
    rng = np.random.default_rng(0)
    a1 = rng.normal(0, 1, (20, 20)).astype(np.float32)
    a1 = (a1 - a1.T) / 2
    a2 = rng.normal(0, 1, (20, 20)).astype(np.float32)
    a2 = (a2 - a2.T) / 2

    da = a1 - a2
    gradient_component = (da + da.T) / 2
    assert np.allclose(gradient_component, 0.0, atol=1e-5)


def test_directed_pressure_sign_reflects_ask_vs_bid():
    dv_ask_heavy = np.concatenate([np.zeros(10), np.ones(10) * 5.0])
    dv_bid_heavy = np.concatenate([np.ones(10) * 5.0, np.zeros(10)])
    assert compute_directed_pressure(dv_ask_heavy) > 0
    assert compute_directed_pressure(dv_bid_heavy) < 0


def test_directed_pressure_rejects_wrong_shape():
    import pytest
    with pytest.raises(ValueError):
        compute_directed_pressure(np.zeros(19))


def test_gauge_state_vector_shape_and_finiteness():
    rng = np.random.default_rng(0)
    a_prev = rng.normal(0, 1, (20, 20)).astype(np.float16)
    a = rng.normal(0, 1, (20, 20)).astype(np.float16)
    dv = rng.normal(0, 1, 20)

    state = compute_gauge_state(a, a_prev, dv, price_momentum=0.5, top_k=5)
    vec = state.to_vector()
    assert vec.shape == (11,)
    assert np.all(np.isfinite(vec))
    assert vec[5] == 0.5
"""Tests for the beta-binomial extension and the likelihood weights of the (mu, rho) grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vafsoma.model import (
    log_betabinom_pmf,
    log_binom_pmf,
    layer_loglik,
    softmax_grid_weights,
)


def test_betabinom_reduces_to_binomial_at_rho_zero():
    y = np.array([2, 5, 10])
    d = np.array([20, 20, 20])
    mu = 0.3
    lo_binom = log_binom_pmf(y, d, mu)
    lo_bb_tiny_rho = log_betabinom_pmf(y, d, mu, 1e-8)
    assert np.allclose(lo_binom, lo_bb_tiny_rho, atol=1e-4)


def test_betabinom_is_more_dispersed_than_binomial():
    # A beta-binomial with rho > 0 should put MORE mass at the extremes (y near 0
    # or near d) than a binomial with the same mean, for a moderate d.
    d = 40
    mu = 0.3
    y_extreme = np.array([0, d])
    y_mid = np.array([int(round(mu * d))])
    lo_binom_extreme = log_binom_pmf(y_extreme, np.full(2, d), mu)
    lo_bb_extreme = np.array([log_betabinom_pmf(np.array([yy]), np.array([d]), mu, 0.05)[0] for yy in y_extreme])
    assert (lo_bb_extreme > lo_binom_extreme).all(), "beta-binomial should assign more mass to extreme counts"


def test_layer_loglik_dispatches_on_rho():
    y = np.array([3])
    d = np.array([20])
    mu = 0.25
    assert np.isclose(layer_loglik(y, d, mu, rho=0.0), layer_loglik(y, d, mu, rho=None if False else 0.0))
    # rho=0 must exactly match the plain-binomial conditional-positive likelihood.
    from vafsoma.model import log_positive_mass_binom

    expected = log_binom_pmf(y, d, mu) - log_positive_mass_binom(d, mu)
    assert np.allclose(layer_loglik(y, d, mu, rho=0.0), expected)


def test_softmax_grid_weights_concentrates_on_best_point():
    logliks = [-100.0, -100.0, -50.0]  # third point is overwhelmingly better
    w = softmax_grid_weights(logliks)
    assert np.isclose(w.sum(), 1.0)
    assert w[2] > 0.999


def test_softmax_grid_weights_uniform_when_tied():
    logliks = [-50.0, -50.0, -50.0]
    w = softmax_grid_weights(logliks)
    assert np.allclose(w, 1.0 / 3.0, atol=1e-9)

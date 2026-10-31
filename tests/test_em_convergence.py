"""Regression tests guarding against premature EM termination.

These tests exist because `em_fit_mixture` previously initialized its
log-likelihood tracker (`prev`) to `-np.inf`. On the first iteration this
made `abs(loglik - prev)` and the convergence threshold both evaluate to
`+inf`, and `inf <= inf` is True under IEEE-754 float comparison -- so the
loop always returned after exactly one M-step, regardless of `tol` or
`max_iter`, and regardless of how far `pi` still was from its fixed point.
See model.py's `em_fit_mixture` for the fix (an `it > 0` guard).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from vafsoma.model import em_fit_mixture, em_fit_mixture_multistart


def _binom_pmf(y: np.ndarray, d: np.ndarray, p: float) -> np.ndarray:
    log_coeff = gammaln(d + 1) - gammaln(y + 1) - gammaln(d - y + 1)
    return np.exp(log_coeff + y * np.log(p) + (d - y) * np.log1p(-p))


def _make_synthetic_mixture(seed: int = 0):
    """A well-separated 4-component binomial mixture with known pi, d fixed at 100."""
    rng = np.random.default_rng(seed)
    true_pi = np.array([0.55, 0.15, 0.05, 0.25])  # E, L1, L2, L3
    true_p = np.array([0.03, 0.10, 0.36, 0.04])  # background is near-zero VAF
    d = 100
    n_draws = 20000
    comp = rng.choice(4, size=n_draws, p=true_pi)
    y = rng.binomial(d, true_p[comp])

    # Group into (y, d) bins with counts, and build the 4-column log-component matrix.
    ys, counts = np.unique(y, return_counts=True)
    dd = np.full_like(ys, d)
    log_comp = np.column_stack(
        [np.log(np.clip(_binom_pmf(ys, dd, p), 1e-300, None)) for p in true_p]
    )
    return log_comp, counts.astype(float), true_pi


def test_em_takes_more_than_one_iteration_from_a_bad_start():
    """A deliberately wrong starting point must not 'converge' after a single M-step.

    This is the direct regression test for the -inf/-inf convergence-check bug:
    started far from its fixed point, EM needs several iterations to get close.
    A one-iteration return here means the convergence check is broken again.
    """
    log_comp, counts, true_pi = _make_synthetic_mixture(seed=1)
    active_mask = np.ones(4, dtype=bool)

    bad_start = np.array([0.94, 0.02, 0.02, 0.02])  # far from true_pi
    pi_hat, loglik, n_iter = em_fit_mixture(
        log_comp, counts, pi_init=bad_start, active_mask=active_mask, tol=1e-8, max_iter=1000
    )

    assert n_iter > 1, (
        f"EM reported convergence after only {n_iter} iteration(s) from a bad start; "
        "this is the signature of the -inf/-inf premature-stopping bug."
    )
    assert np.abs(pi_hat - true_pi).max() < 0.03, (
        f"Recovered pi {pi_hat} is far from the true generating pi {true_pi}; "
        "EM did not actually converge to (near) the MLE."
    )


def test_em_multistart_is_insensitive_to_starting_point():
    """Once genuinely converged, independent starts should agree with each other.

    Before the fix, different starts each stopped after 1 EM step and reported
    wildly different pi estimates and log-likelihoods (see the walkthrough this
    test accompanies). After the fix, starts should agree to within a small
    tolerance because they all reach the same stationary point of the likelihood.
    """
    log_comp, counts, true_pi = _make_synthetic_mixture(seed=2)
    active_mask = np.ones(4, dtype=bool)

    starts = [
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.90, 0.03, 0.03, 0.04]),
        np.array([0.10, 0.60, 0.10, 0.20]),
        np.array([0.40, 0.10, 0.40, 0.10]),
    ]
    results = [
        em_fit_mixture(log_comp, counts, pi_init=s, active_mask=active_mask, tol=1e-8, max_iter=2000)
        for s in starts
    ]
    pis = np.array([r[0] for r in results])
    logliks = np.array([r[1] for r in results])
    n_iters = np.array([r[2] for r in results])

    assert (n_iters > 1).all(), f"Some starts converged in a single iteration: {n_iters}"
    assert pis.max(axis=0).sum() - pis.min(axis=0).sum() < 0.02, (
        f"pi estimates disagree across starting points, suggesting non-convergence: {pis}"
    )
    assert logliks.max() - logliks.min() < 1e-3, (
        f"log-likelihoods disagree across starting points, suggesting non-convergence: {logliks}"
    )


def test_em_multistart_recovers_known_pi():
    log_comp, counts, true_pi = _make_synthetic_mixture(seed=3)
    active_mask = np.ones(4, dtype=bool)
    pi_hat, loglik, n_iter = em_fit_mixture_multistart(
        log_comp, counts, active_mask=active_mask, n_starts=8, seed=3
    )
    assert np.abs(pi_hat - true_pi).max() < 0.03

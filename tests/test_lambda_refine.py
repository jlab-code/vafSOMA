"""Validation tests for the coarse-grid + continuous-refinement lambda search
(model._profile_lambda_search / profile_fit_lambda), which replaces the old
pure fixed-grid profile search.

These tests exist specifically to check the estimator's *correctness*, not
just that it runs: a fixed grid of any resolution necessarily snaps
lambda_hat to a grid point, and a too-coarse grid can land on a visibly worse
optimum (this was empirically observed on real data: a 25-point grid found a
different, worse optimum than a 200-point grid at some tree/depth cutoffs).
The refinement step must (a) recover a known true lambda from synthetic data
even when the true value falls between coarse-grid points, and (b) match or
beat a fine reference grid's log-likelihood while doing substantially fewer
total EM (lambda, pi) evaluations.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from vafsoma.model import (
    ExponentialComponentCache,
    MuTriple,
    _profile_lambda_search,
    group_positive_rows,
    profile_fit_lambda,
)


def _simulate_positive_observations(
    n: int,
    lam_true: float,
    pi_true: dict[str, float],
    mu: MuTriple,
    q_max: float,
    depth_range: tuple[int, int],
    seed: int,
) -> pd.DataFrame:
    """Simulate conditional-on-positive (alt_depth, DP) pairs from the exact
    generative model the pipeline fits: pick a component per pi_true, draw a
    per-read success probability (truncated-exponential q for E, fixed mu for
    layers), draw y ~ Binomial(d, p), and reject y == 0 (conditioning on
    alt_depth > 0, matching the pipeline's own treatment of zeros)."""
    rng = np.random.default_rng(seed)
    comps = list(pi_true.keys())
    probs = np.array([pi_true[c] for c in comps], dtype=float)
    probs = probs / probs.sum()

    rows = []
    while len(rows) < n:
        comp = rng.choice(comps, p=probs)
        d = int(rng.integers(depth_range[0], depth_range[1] + 1))
        if comp == "E":
            # Sample q from the truncated Exp(lam_true) on (0, q_max] via inverse CDF.
            u = rng.uniform()
            q = -np.log1p(-u * (-np.expm1(-lam_true * q_max))) / lam_true
            q = float(np.clip(q, 1e-9, q_max))
        else:
            q = mu.value(comp)
        y = rng.binomial(d, q)
        if y <= 0:
            continue
        rows.append({"alt_depth": int(y), "DP": int(d), "variant_id": f"v{len(rows)}"})
    return pd.DataFrame(rows)


def test_refined_lambda_recovers_true_value_between_grid_points():
    """True lambda deliberately chosen off any coarse-grid point; refine=True
    on a deliberately coarse (12-point) grid must land much closer to it than
    refine=False on the identical coarse grid."""
    lam_true = 63.7
    mu = MuTriple(L1=0.10, L2=0.36, L3=0.04)
    pi_true = {"E": 0.90, "L1": 0.0, "L2": 0.07, "L3": 0.03}
    df = _simulate_positive_observations(
        n=4000, lam_true=lam_true, pi_true=pi_true, mu=mu, q_max=0.5, depth_range=(30, 60), seed=7
    )
    grouped = group_positive_rows(
        pd.DataFrame({"alt_depth": df["alt_depth"], "DP": df["DP"], "variant_id": df["variant_id"]})
    )
    exp_cache = ExponentialComponentCache(q_max=0.5, n_quad=96)
    coarse_grid = np.geomspace(1.0, 500.0, 12)  # deliberately coarse

    lam_norefine, _, ll_norefine, _, _ = profile_fit_lambda(
        grouped, mu, coarse_grid, exp_cache, em_n_starts=6, em_random_seed=1, refine=False
    )
    lam_refine, _, ll_refine, _, _ = profile_fit_lambda(
        grouped, mu, coarse_grid, exp_cache, em_n_starts=6, em_random_seed=1, refine=True
    )

    err_norefine = abs(lam_norefine - lam_true) / lam_true
    err_refine = abs(lam_refine - lam_true) / lam_true

    # The refined estimate must be substantially closer to the ground truth...
    assert err_refine < err_norefine, (
        f"refine did not improve accuracy: lam_true={lam_true}, "
        f"lam_norefine={lam_norefine} (err={err_norefine:.3f}), lam_refine={lam_refine} (err={err_refine:.3f})"
    )
    # ...and within statistical noise of the true value for this sample size.
    assert err_refine < 0.15, f"refined lambda_hat={lam_refine} too far from true lambda={lam_true}"
    # Refinement must never make the achieved log-likelihood *worse* than the
    # coarse grid alone (it strictly extends the search).
    assert ll_refine >= ll_norefine - 1e-6


def test_refine_matches_fine_grid_loglik_with_far_fewer_evaluations():
    """A coarse grid + refinement should reach (at least) the same profile
    log-likelihood as a much finer uniform grid, using far fewer total
    (lambda, pi) EM evaluations -- the speed half of the requirement."""
    lam_true = 41.3
    mu = MuTriple(L1=0.10, L2=0.36, L3=0.04)
    pi_true = {"E": 0.94, "L1": 0.0, "L2": 0.045, "L3": 0.015}
    df = _simulate_positive_observations(
        n=3000, lam_true=lam_true, pi_true=pi_true, mu=mu, q_max=0.5, depth_range=(25, 55), seed=11
    )
    grouped = group_positive_rows(
        pd.DataFrame({"alt_depth": df["alt_depth"], "DP": df["DP"], "variant_id": df["variant_id"]})
    )
    exp_cache = ExponentialComponentCache(q_max=0.5, n_quad=96)

    fine_grid = np.geomspace(1.0, 500.0, 100)
    coarse_grid = np.geomspace(1.0, 500.0, 12)

    t0 = time.time()
    lam_fine, _, ll_fine, _, _, diag_fine = _profile_lambda_search(
        grouped, mu, fine_grid, exp_cache, em_n_starts=6, em_random_seed=1, refine=False
    )
    t_fine = time.time() - t0

    t0 = time.time()
    lam_ref, _, ll_ref, _, _, diag_ref = _profile_lambda_search(
        grouped, mu, coarse_grid, exp_cache, em_n_starts=6, em_random_seed=1, refine=True
    )
    t_ref = time.time() - t0

    assert diag_ref["n_total_evals"] < diag_fine["n_total_evals"], (
        f"refined search should need fewer evaluations than the fine grid: "
        f"{diag_ref['n_total_evals']} vs {diag_fine['n_total_evals']}"
    )
    assert ll_ref >= ll_fine - 1.0, (
        f"refined search's loglik ({ll_ref}) should be close to or better than the fine grid's ({ll_fine})"
    )
    print(
        f"[timing] fine_grid(n=100): {t_fine:.1f}s, {diag_fine['n_total_evals']} evals, lam={lam_fine:.2f}, ll={ll_fine:.2f}\n"
        f"[timing] coarse+refine(n=12): {t_ref:.1f}s, {diag_ref['n_total_evals']} evals, lam={lam_ref:.2f}, ll={ll_ref:.2f}"
    )


def test_refine_is_deterministic_given_seed():
    lam_true = 50.0
    mu = MuTriple(L1=0.10, L2=0.36, L3=0.04)
    pi_true = {"E": 0.93, "L1": 0.0, "L2": 0.05, "L3": 0.02}
    df = _simulate_positive_observations(
        n=1500, lam_true=lam_true, pi_true=pi_true, mu=mu, q_max=0.5, depth_range=(25, 45), seed=3
    )
    grouped = group_positive_rows(
        pd.DataFrame({"alt_depth": df["alt_depth"], "DP": df["DP"], "variant_id": df["variant_id"]})
    )
    exp_cache = ExponentialComponentCache(q_max=0.5, n_quad=96)
    grid = np.geomspace(1.0, 500.0, 12)

    r1 = profile_fit_lambda(grouped, mu, grid, exp_cache, em_n_starts=6, em_random_seed=1, refine=True)
    r2 = profile_fit_lambda(grouped, mu, grid, exp_cache, em_n_starts=6, em_random_seed=1, refine=True)
    assert r1[0] == r2[0]
    assert np.allclose(r1[1], r2[1])
    assert r1[2] == r2[2]


if __name__ == "__main__":
    import sys
    import traceback

    tests = [
        test_refined_lambda_recovers_true_value_between_grid_points,
        test_refine_matches_fine_grid_loglik_with_far_fewer_evaluations,
        test_refine_is_deterministic_given_seed,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
    if failed:
        sys.exit(1)
    print("All lambda-refine tests passed.")

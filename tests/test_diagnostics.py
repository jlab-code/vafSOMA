"""vafSOMA 0.1.6 post-fit diagnostics: layer resolvability and VAF fit check."""
import numpy as np
import pandas as pd

from vafsoma.diagnostics import layer_recall, vaf_fit_check
from vafsoma.model import ExponentialComponentCache, FitResult, MuTriple


def _fit(lam=60.0, pi=(0.97, 0.01, 0.015, 0.005), mu=(0.16, 0.30, 0.0375)):
    return FitResult(mu=MuTriple(L1=mu[0], L2=mu[1], L3=mu[2]), lambda_hat=lam, pi=np.array(pi), loglik=0.0,
                     n_positive_rows=0, n_unique_positive_variants=0, grouped=pd.DataFrame())


def test_layer_inside_background_is_unresolvable_and_separated_layer_is_resolvable():
    cache = ExponentialComponentCache(q_max=0.5, n_quad=96)
    rec = layer_recall(_fit(), np.full(500, 120), cache)
    assert rec["L3"] < 0.1          # mu_L3 = 0.0375 lies inside an exponential background with mean ~0.017
    assert rec["L2"] > 0.9          # mu_L2 = 0.30 is far from the background


def test_fit_check_is_quiet_when_data_follow_the_fitted_model_and_flags_a_heavier_background():
    rng = np.random.default_rng(1)
    cache = ExponentialComponentCache(q_max=0.5, n_quad=96)
    fit = _fit(pi=(1.0, 0.0, 0.0, 0.0))
    n, d = 20000, 100
    q = np.minimum(rng.exponential(1 / fit.lambda_hat, n), 0.5)
    y = rng.binomial(d, q)
    obs = pd.DataFrame({"alt_depth": y, "DP": d})
    chk = vaf_fit_check(fit, obs, cache)
    big = chk["expected"] >= 10
    assert (chk.loc[big, "pearson_residual"].abs() < 4).all()
    q2 = np.minimum(rng.gamma(0.5, (1 / fit.lambda_hat) / 0.5, n), 0.5)   # same mean, heavier tail
    obs2 = pd.DataFrame({"alt_depth": rng.binomial(d, q2), "DP": d})
    chk2 = vaf_fit_check(fit, obs2, cache)
    assert (chk2.loc[chk2["expected"] >= 10, "pearson_residual"].abs() >= 4).any()

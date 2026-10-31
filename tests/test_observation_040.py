"""Tests for the 0.4.0 observation model (per-read sequencing error e; allele-fraction overdispersion from the calibration)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vafsoma import config as vcfg  # noqa: E402
from vafsoma.diagnostics import simulate_reads  # noqa: E402
from vafsoma.model import (  # noqa: E402
    ExponentialComponentCache,
    MuTriple,
    layer_loglik,
    log_binom_pmf,
    make_component_loglik,
    read_params,
)
from vafsoma.replicates import ReplicateLikelihood, build_units  # noqa: E402

MU = MuTriple(0.10, 0.3625, 0.0375)
ACTIVE = ("E", "L1", "L2", "L3")


def test_read_params_identity_without_error():
    assert read_params(0.4, 0.0, 0.0) == (0.4, 0.0)
    assert read_params(0.4, 0.007, 0.0) == (0.4, 0.007)


def test_read_params_moments_match_simulation():
    """Mean and variance of the read probability p = v (1 - e) + (1 - v) e, v ~ Beta(mean mu, rho)."""
    mu, rho, e = 0.42, 1 / 141, 1e-3
    m, r = read_params(mu, rho, e)
    rng = np.random.default_rng(3)
    s = (1 - rho) / rho
    v = rng.beta(mu * s, (1 - mu) * s, 400_000)
    p = v * (1 - e) + (1 - v) * e
    assert m == pytest.approx(p.mean(), abs=2e-4)
    assert m * (1 - m) * r == pytest.approx(p.var(), rel=0.02)


def test_layer_binomial_with_error_equals_binomial_at_shifted_p():
    y = np.arange(1, 81)
    d = np.full_like(y, 80)
    e, mu = 3e-4, 0.36
    p = e + (1 - 2 * e) * mu
    expect = log_binom_pmf(y, d, p) - np.log(-np.expm1(80 * np.log1p(-p)))
    assert np.allclose(layer_loglik(y, d, mu, rho=0.0, read_error=e), expect)


@pytest.mark.parametrize("e", [0.0, 3e-4, 1e-3])
@pytest.mark.parametrize("lam", [5.0, 60.0, 300.0])
def test_background_conditional_density_sums_to_one(e, lam):
    d = 120
    y = np.arange(1, d + 1)
    cache = ExponentialComponentCache(q_max=0.5, n_quad=96, read_error=e)
    logs = make_component_loglik(y, np.full_like(y, d), MU, lam, cache, active_components=ACTIVE, rho=1 / 141)
    total = np.exp(logs).sum(axis=0)
    assert np.allclose(total, 1.0, atol=1e-6)


def test_replicate_singletons_reduce_to_base_model_with_error():
    rng = np.random.default_rng(0)
    rows = []
    for v in range(200):
        vaf = rng.choice([0.0, 0.01, 0.10, 0.3625])
        for s in "ABC":
            dd = int(40 + rng.poisson(10))
            rows.append(dict(variant_id=f"v{v}", chrom="c1", pos=v, ref="A", alt="T", sample=s,
                             alt_depth=int(rng.binomial(dd, vaf * (1 - 2e-3) + 1e-3)), DP=dd))
    df = pd.DataFrame(rows)
    units = build_units(df, {s: s for s in "ABC"})
    for e, rho in [(1e-3, 0.0), (3e-4, 1 / 141)]:
        cache = ExponentialComponentCache(read_error=e)
        lik = ReplicateLikelihood(units, cache)
        for lam in (10.0, 100.0):
            lc = lik.comp_loglik_units(lam, MU, ACTIVE, rho)
            base = make_component_loglik(units["Y"][:, 0], units["D"][:, 0], MU, lam, cache, ACTIVE, rho=rho)
            assert np.allclose(lc[:, 0], base[:, 0], atol=1e-8)
            assert np.allclose(lc[:, 1], base[:, 0], atol=1e-8)
            assert np.allclose(lc[:, 2:5], base[:, 1:], atol=1e-8)


def test_simulate_reads_unchanged_without_error_or_dispersion():
    D = np.array([[30, 40, 50]] * 20)
    a = simulate_reads(np.random.default_rng(5), D, 0.36)
    b = np.random.default_rng(5).binomial(D, 0.36)
    assert np.array_equal(a, b)


def _write_cfg(tmp_path, observation, layer_dispersion=None):
    for f in ("v.tsv", "g.tsv", "r.tsv"):
        (tmp_path / f).write_text("x\n")
    user = {"input": {"variants": "v.tsv", "effective_genome_size": "g.tsv", "root2tip": "r.tsv"},
            "observation": observation}
    if layer_dispersion:
        user["model"] = {"layer_dispersion": layer_dispersion}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(user))
    return vcfg.load_config(p)


def test_config_concentration_sets_rho_and_metadata(tmp_path):
    cfg, base = _write_cfg(tmp_path, {"read_error": 3e-4, "read_concentration": 140.0, "preset": "example_preset",
                                      "config_sha256": "f86c0167", "source_version": "1.1.1"})
    ec = vcfg.to_estimate_config(cfg, base)
    assert ec.read_error == 3e-4
    assert ec.layer_rho == pytest.approx(1 / 141)
    assert ec.obs_preset == "example_preset" and ec.obs_version == "1.1.1"


def test_config_old_provenance_key_still_accepted(tmp_path):
    """0.4.3 renamed observation.plantsoma_obs_version to observation.source_version; the old name still works."""
    cfg, base = _write_cfg(tmp_path, {"read_error": 3e-4, "plantsoma_obs_version": "1.1.1"})
    ec = vcfg.to_estimate_config(cfg, base)
    assert ec.obs_version == "1.1.1"


def test_config_rejects_conflicting_dispersion(tmp_path):
    cfg, base = _write_cfg(tmp_path, {"read_concentration": 140.0}, {"mode": "grid", "rho": 0.0, "rho_grid": None})
    with pytest.raises(vcfg.ConfigError):
        vcfg.to_estimate_config(cfg, base)


def test_config_rejects_bad_error(tmp_path):
    cfg, base = _write_cfg(tmp_path, {"read_error": 0.7})
    with pytest.raises(vcfg.ConfigError):
        vcfg.to_estimate_config(cfg, base)


def test_template_defaults_reproduce_030():
    t = vcfg._strip_comments(copy.deepcopy(vcfg.TEMPLATE))
    assert t["observation"]["read_error"] == 0.0 and t["observation"]["read_concentration"] is None


def test_error_component_follows_read_error_and_is_normalised():
    """E_error exists exactly when e > 0; its conditional density over the reads of one replicate sums to 1."""
    df = pd.DataFrame([dict(variant_id="v", chrom="c", pos=1, ref="A", alt="T", sample="A", alt_depth=1, DP=60)])
    units = build_units(df, {"A": "g"})
    lc0 = ReplicateLikelihood(units, ExponentialComponentCache(read_error=0.0)).comp_loglik(50.0, MU, ACTIVE)
    assert lc0.shape[1] == 6 and np.isneginf(lc0[:, 5]).all()
    d = 60
    rows = [dict(variant_id=f"v{y}", chrom="c", pos=y, ref="A", alt="T", sample="A", alt_depth=y, DP=d) for y in range(1, d + 1)]
    units = build_units(pd.DataFrame(rows), {"A": "g"})
    lik = ReplicateLikelihood(units, ExponentialComponentCache(read_error=1e-3))
    lc = lik.comp_loglik_units(50.0, MU, ACTIVE)
    assert np.exp(lc[:, 5]).sum() == pytest.approx(1.0, abs=1e-9)


def test_base_model_background_with_error_part_is_normalised():
    d = 100
    y = np.arange(1, d + 1)
    cache = ExponentialComponentCache(read_error=1e-3)
    for w in (0.0, 0.3, 0.9):
        logs = make_component_loglik(y, np.full_like(y, d), MU, 40.0, cache, ACTIVE, rho=1 / 141, error_frac=w)
        assert np.exp(logs[:, 0]).sum() == pytest.approx(1.0, abs=1e-6)
    five = make_component_loglik(y, np.full_like(y, d), MU, 40.0, cache, ACTIVE, split_error=True)
    assert five.shape[1] == 5 and np.exp(five[:, 4]).sum() == pytest.approx(1.0, abs=1e-9)


def _toy_with_error_units(seed=2, n=1500, e=1e-3, reps=("A1", "A2", "A3")):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n):
        kind = rng.choice(["bg", "err", "L2"], p=[0.45, 0.5, 0.05])
        q = {"bg": min(rng.exponential(1 / 40), 0.45), "err": 0.0, "L2": 0.3625}[kind]
        for s in reps:
            d = int(rng.poisson(150))
            rows.append(dict(variant_id=f"v{v}", chrom="c", pos=v, ref="A", alt="T", sample=s,
                             alt_depth=int(rng.binomial(d, q * (1 - 2 * e) + e)), DP=d, kind=kind))
    return pd.DataFrame(rows)


def test_group_model_error_part_keeps_background_rate():
    """Error-only units go to E_error and the background rate stays near the simulated value (40)."""
    from vafsoma.replicates import fit_replicate_depth
    df = _toy_with_error_units()
    smap = {"A1": "A", "A2": "A", "A3": "A"}
    grid = np.exp(np.linspace(np.log(1), np.log(500), 15))
    fit, ex, *_ = fit_replicate_depth(df, smap, MU, grid, ExponentialComponentCache(read_error=1e-3), ACTIVE, em_n_starts=4)
    assert ex["pi_E_error"] > 0.2 and fit.pi_E_error == pytest.approx(ex["pi_E_error"])
    assert fit.lambda_hat < 80


def test_base_model_error_part_keeps_background_rate():
    from vafsoma.model import fit_one_depth
    df = _toy_with_error_units(reps=("A1",))
    grid = np.exp(np.linspace(np.log(1), np.log(500), 15))
    fit = fit_one_depth(df, MU, grid, ExponentialComponentCache(read_error=1e-3), active_components=ACTIVE, em_n_starts=4)
    assert fit.pi_E_error > 0.1
    assert len(fit.pi) == 4 and fit.pi.sum() == pytest.approx(1.0)
    assert fit.lambda_hat < 80

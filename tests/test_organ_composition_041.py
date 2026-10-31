"""Tests for organ-specific layer compositions (0.4.1)."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vafsoma import config as vcfg  # noqa: E402
from vafsoma.diagnostics import layer_recall, layer_recall_by_organ, vaf_fit_check, vaf_fit_check_by_organ  # noqa: E402
from vafsoma.estimate import EstimateConfig, build_model_design, organ_scale_for_tree, run_estimate  # noqa: E402
from vafsoma.model import (  # noqa: E402
    ExponentialComponentCache,
    MuTriple,
    fit_one_depth,
    layer_loglik,
    make_component_loglik,
    read_params,
)
from vafsoma.replicates import ReplicateLikelihood, build_units, fit_replicate_depth  # noqa: E402

ACTIVE = ("E", "L1", "L2")
MU = MuTriple(0.10, 0.36, None)
GRID = np.exp(np.linspace(np.log(2), np.log(400), 12))


def _sim(seed=0, n=1500, organs=("A", "B", "C"), c2=None, depth=80, e=3e-4):
    """Toy per-organ data: background (exp, mean 1/40) and L2-fixed variants at 0.5 * c2[organ]."""
    rng = np.random.default_rng(seed)
    c2 = c2 or {o: 0.72 for o in organs}
    rows = []
    for v in range(n):
        kind = rng.choice(["bg", "L2"], p=[0.85, 0.15])
        q = min(rng.exponential(1 / 40), 0.45)
        for o in organs:
            mu = 0.5 * c2[o] if kind == "L2" else q
            d = int(rng.poisson(depth))
            rows.append(dict(variant_id=f"v{v}", chrom="c1", pos=v, ref="A", alt="T", sample=o,
                             alt_depth=int(rng.binomial(d, mu * (1 - 2 * e) + e)), DP=d, kind=kind))
    return pd.DataFrame(rows)


def test_read_params_vectorised():
    mus = np.array([0.05, 0.2, 0.45])
    m, r = read_params(mus, 1 / 141, 3e-4)
    for i, x in enumerate(mus):
        mi, ri = read_params(float(x), 1 / 141, 3e-4)
        assert m[i] == pytest.approx(mi) and r[i] == pytest.approx(ri)


def test_values_and_for_sample():
    mu = dataclasses.replace(MU, organ_scale=(("B", (1.0, 0.5, 1.0)),))
    v = mu.values("L2", np.array(["A", "B", "Z"], dtype=object))
    assert np.allclose(v, [0.36, 0.18, 0.36])
    assert mu.for_sample("B").L2 == pytest.approx(0.18) and mu.for_sample("B").organ_scale is None
    assert mu.values("L2") == 0.36


def test_unit_scale_reproduces_common_model():
    df = _sim()
    cache = ExponentialComponentCache(read_error=3e-4)
    f0 = fit_one_depth(df, MU, GRID, cache, active_components=ACTIVE, em_n_starts=4, rho=1 / 141)
    one = dataclasses.replace(MU, organ_scale=tuple((o, (1.0, 1.0, 1.0)) for o in "ABC"))
    f1 = fit_one_depth(df, one, GRID, cache, active_components=ACTIVE, em_n_starts=4, rho=1 / 141)
    assert f1.loglik == pytest.approx(f0.loglik, rel=1e-9)
    assert f1.lambda_hat == pytest.approx(f0.lambda_hat, rel=1e-6)
    assert np.allclose(f1.pi, f0.pi, atol=1e-7)


def test_row_likelihood_equals_organ_model():
    df = _sim(n=200)
    pos = df[df.alt_depth > 0]
    cache = ExponentialComponentCache(read_error=3e-4)
    sc = (("A", (1.0, 1.0, 1.0)), ("B", (1.0, 0.8, 1.0)), ("C", (1.0, 1.25, 1.0)))
    mu = dataclasses.replace(MU, organ_scale=sc)
    L = make_component_loglik(pos.alt_depth.to_numpy(), pos.DP.to_numpy(), mu, 40.0, cache, ACTIVE, rho=1 / 141,
                              samples=pos["sample"].to_numpy())
    for o, s in sc:
        x = pos["sample"].to_numpy() == o
        ref = layer_loglik(pos.alt_depth.to_numpy()[x], pos.DP.to_numpy()[x], 0.36 * s[1], rho=1 / 141, read_error=3e-4)
        assert np.allclose(L[x, 2], ref)


def test_oracle_composition_improves_l2_count():
    c2 = {"A": 0.50, "B": 0.72, "C": 0.90}
    df = _sim(seed=3, n=2500, c2=c2, depth=120)
    cache = ExponentialComponentCache(read_error=3e-4)
    true_l2 = df[(df.kind == "L2") & (df.alt_depth > 0)].groupby("sample").size().sum()
    res = {}
    for name, mu in [("common", MU), ("oracle", dataclasses.replace(
            MU, organ_scale=tuple((o, (1.0, c / 0.72, 1.0)) for o, c in c2.items())))]:
        f = fit_one_depth(df, mu, GRID, cache, active_components=ACTIVE, em_n_starts=4)
        res[name] = abs(f.pi[2] * f.n_positive_rows / true_l2 - 1)
    assert res["oracle"] < res["common"]


def test_replicate_model_unit_scale_and_rows():
    df = _sim(n=600, organs=("A1", "A2", "B1", "B2"))
    smap = {"A1": "A", "A2": "A", "B1": "B", "B2": "B"}
    cache = ExponentialComponentCache(read_error=3e-4)
    fit0, ex0, *_ = fit_replicate_depth(df, smap, MU, GRID, cache, ACTIVE, em_n_starts=3)
    one = dataclasses.replace(MU, organ_scale=tuple((o, (1.0, 1.0, 1.0)) for o in smap))
    fit1, ex1, *_ = fit_replicate_depth(df, smap, one, GRID, cache, ACTIVE, em_n_starts=3)
    assert fit1.loglik == pytest.approx(fit0.loglik, rel=1e-9)
    units = build_units(df, smap)
    sc = dataclasses.replace(MU, organ_scale=(("A2", (1.0, 0.5, 1.0)),))
    lc = ReplicateLikelihood(units, cache, by_sample=True).comp_loglik_units(40.0, sc, ACTIVE)
    # recompute the L2 column by hand (binomial, rho = 0)
    Y, D, M, S = units["Y"], units["D"], units["M"], units["S"]
    m = np.where(S == "A2", 0.18, 0.36) * (1 - 2 * 3e-4) + 3e-4
    from scipy.stats import binom
    lf = np.where(M, binom.logpmf(Y, D, m), 0).sum(1)
    lf0 = np.where(M, D * np.log1p(-m), 0).sum(1)
    assert np.allclose(lc[:, 3], lf - np.log(-np.expm1(lf0)))


def test_diagnostics_by_organ_reduce_to_common():
    df = _sim(n=400)
    cache = ExponentialComponentCache(read_error=3e-4)
    f0 = fit_one_depth(df, MU, GRID, cache, active_components=ACTIVE, em_n_starts=3)
    f1 = dataclasses.replace(f0, mu=dataclasses.replace(MU, organ_scale=tuple((o, (1.0, 1.0, 1.0)) for o in "ABC")))
    r0 = layer_recall(f0, df.DP.to_numpy(), cache); r1 = layer_recall_by_organ(f1, df, cache)
    assert r1["L2"] == pytest.approx(r0["L2"], abs=0.02)
    c0 = vaf_fit_check(f0, df, cache); c1 = vaf_fit_check_by_organ(f1, df, cache)
    assert np.allclose(c0.observed, c1.observed) and np.allclose(c0.expected, c1.expected, rtol=0.02, atol=0.5)


def test_design_scales_and_errors(tmp_path):
    t = tmp_path / "olp.tsv"
    t.write_text("sample\tL1\tL2\tL3\nA\t0.2\t0.7\t0.1\nB\t0.1\t0.8\t0.1\n")
    cfg = EstimateConfig(variants="x", effective_genome_size="x", root2tip="x", layer_proportions="L1=0.15,L2=0.75,L3=0.1",
                         organ_layer_proportions=str(t))
    d = build_model_design(cfg)
    sc = dict(organ_scale_for_tree(d, "t1", ["A", "B", "C"]))
    assert sc["A"] == pytest.approx((0.2 / 0.15, 0.7 / 0.75, 1.0)) and "C" not in sc
    with pytest.raises(ValueError):
        build_model_design(dataclasses.replace(cfg, layer_specific_seq="yes", active_layer="L2", layer_proportions=None))
    bad = dataclasses.replace(cfg, organ_layer_proportions={("t1", "Q"): {"L1": 0.1, "L2": 0.8, "L3": 0.1}})
    with pytest.raises(ValueError):
        organ_scale_for_tree(build_model_design(bad), "t1", ["A", "B"])


def test_config_key(tmp_path):
    for f in ("v.tsv", "g.tsv", "r.tsv"):
        (tmp_path / f).write_text("x\n")
    (tmp_path / "olp.tsv").write_text("sample\tL1\tL2\tL3\nA\t0.2\t0.7\t0.1\n")
    user = {"input": {"variants": "v.tsv", "effective_genome_size": "g.tsv", "root2tip": "r.tsv"},
            "model": {"organ_layer_proportions": "olp.tsv"}}
    p = tmp_path / "c.json"; p.write_text(json.dumps(user))
    cfg, base = vcfg.load_config(p)
    ec = vcfg.to_estimate_config(cfg, base)
    assert Path(ec.organ_layer_proportions).name == "olp.tsv"
    user["model"]["organ_layer_proportions"] = "missing.tsv"; p.write_text(json.dumps(user))
    cfg, base = vcfg.load_config(p)
    with pytest.raises(vcfg.ConfigError):
        vcfg.to_estimate_config(cfg, base)


def _write_inputs(tmp_path, c2, samples=("S1", "S2", "S3"), depths=(30, 31, 32, 33), seed=5):
    rng = np.random.default_rng(seed)
    var_dir = tmp_path / "variants"; var_dir.mkdir()
    for dmin in depths:
        rows = []; pos = 1000
        for kind, nvar in (("bg", 300), ("L2", 40)):
            for _ in range(nvar):
                pos += 1; q = min(rng.exponential(0.02), 0.45)
                for s in samples:
                    d = int(rng.integers(dmin + 30, dmin + 60)); p = 0.5 * c2[s] if kind == "L2" else q
                    y = int(rng.binomial(d, p))
                    rows.append(dict(sample=s, chrom="chr1", pos=pos, ref="A", alt="T", ref_depth=d - y, alt_depth=y, DP=d,
                                     VAF=y / d))
        pd.DataFrame(rows).to_csv(var_dir / f"tree1_dp_{dmin}_vaf.csv", index=False)
    g = tmp_path / "genome.csv"
    pd.DataFrame({"tree": ["tree1"] * len(depths), "depth_filter": list(depths), "effective_genome_size": [1_000_000] * len(depths)}).to_csv(g, index=False)
    r = tmp_path / "root.csv"
    pd.DataFrame({"sample": list(samples), "root2tip_length": [10, 11, 12]}).to_csv(r, index=False)
    return var_dir, g, r


def test_run_estimate_with_organ_proportions(tmp_path):
    c2 = {"S1": 0.55, "S2": 0.75, "S3": 0.90}
    var_dir, g, r = _write_inputs(tmp_path, c2)
    olp = {s: {"L1": round(0.925 - c, 3), "L2": c, "L3": 0.075} for s, c in c2.items()}
    cfg = EstimateConfig(variants=str(var_dir), effective_genome_size=str(g), root2tip=str(r), dataset_label="olp",
                         outdir=str(tmp_path / "out"), lambda_grid="10,30,60,120", write_plots=False,
                         layer_proportions="L1=0.175,L2=0.75", organ_layer_proportions=olp, em_n_starts=3,
                         selected_dp_window="30:33", read_error=3e-4)
    out = run_estimate(cfg)
    od = tmp_path / "out"
    fp = pd.read_csv(od / "02_fit_parameters_by_tree_depth.csv")
    assert "organ_layer_scale" in fp.columns
    sc = json.loads(fp.organ_layer_scale.iloc[0])
    assert sc["S1"][1] == pytest.approx(0.55 / 0.75)
    meta = json.loads((od / "00_run_metadata.json").read_text())
    assert meta["organ.layer.scale"]["*"]["S3"][1] == pytest.approx(0.90 / 0.75)
    rv = pd.read_csv(Path(out["layer_resolvability"]))
    assert set(rv.layer) >= {"L2"}

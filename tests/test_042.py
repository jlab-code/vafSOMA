"""Tests for vafSOMA 0.4.2: background-quantile condition of the resolvability flag and the all-organs screen."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vafsoma import config as vcfg  # noqa: E402
from vafsoma.diagnostics import background_vaf_quantile  # noqa: E402
from vafsoma.estimate import EstimateConfig, run_estimate  # noqa: E402
from vafsoma.screen import all_organs_screen  # noqa: E402


def test_background_quantile_formula():
    lam, qmax, p = 25.0, 0.5, 0.95
    q = background_vaf_quantile(lam, qmax, p)
    cdf = (1 - np.exp(-lam * q)) / (1 - np.exp(-lam * qmax))          # truncated exponential CDF
    assert cdf == pytest.approx(p, abs=1e-12)
    assert background_vaf_quantile(400.0, 0.5, 0.95) == pytest.approx(-np.log(0.05) / 400.0, rel=1e-9)
    assert background_vaf_quantile(1e-9, 0.5, 0.95) == pytest.approx(0.475, rel=1e-6)   # flat limit
    assert background_vaf_quantile(0.0, 0.5, 0.95) == pytest.approx(0.475)


def _tab(rows):
    df = pd.DataFrame(rows, columns=["tree", "depth_filter", "sample", "chrom", "pos", "ref", "alt", "alt_depth"])
    df["variant_id"] = df.chrom + ":" + df.pos.astype(str) + ":" + df.ref + ">" + df.alt
    return df


def test_all_organs_screen_rule():
    rows = []
    for d in (10, 11):
        for s, alts in (("A", (3, 2, 1, 0)), ("B", (1, 4, 2, 0)), ("C", (2, 0, 1, 0)), ("N", (0, 0, 5, 0))):
            for i, a in enumerate(alts):
                if s == "C" and i == 3:      # C has no row for v4 (counts as 0)
                    continue
                rows.append(("t1", d, s, "c", 100 + i, "A", "T", a if d == 10 else a + (5 if i == 1 else 0)))
    v = _tab(rows)
    out, summ, sites = all_organs_screen(v, {"t1": {"N"}}, min_alt=1)
    # v1 (alt in A, B, C) and v3 (alt in A, B, C and in the normal) are screened; v2 (C = 0 at the lowest tier,
    # although > 0 at tier 11) and v4 are kept
    assert set(sites.pos) == {100, 102}
    assert set(out.pos) == {101, 103}
    assert not (out.pos == 100).any() and (out.depth_filter == 11).any()
    s = summ.iloc[0]
    assert s.depth_filter_used == 10 and s.n_noncontrol_samples == 3 and s.n_screened == 2 and s.n_variants_lowest_tier == 4
    out2, summ2, _ = all_organs_screen(v, {"t1": {"N"}}, min_alt=2)       # B has 1 read at v1, A at v3 -> both kept
    assert set(out2.pos) == {100, 101, 102, 103}


def test_all_organs_screen_design_checks():
    v = _tab([("t1", 10, s, "c", 1, "A", "T", 1) for s in ("A", "B", "N")])
    with pytest.raises(ValueError, match="at least 3"):
        all_organs_screen(v, {"t1": {"N"}})
    v = _tab([("t1", 10, s, "c", 1, "A", "T", 1) for s in ("A", "B", "C")])
    with pytest.raises(ValueError, match="no control organ"):
        all_organs_screen(v, {})
    with pytest.raises(ValueError, match="min_alt"):
        all_organs_screen(v, {"t1": {"N"}}, min_alt=0)


def _write(tmp_path, n_bg=400, n_l2=60, n_art=40, c1=0.06, samples=("S1", "S2", "S3", "S4", "N"), depths=(30, 31, 32, 33),
           bg_rate=60.0, seed=3):
    """Toy tree: sporadic background (exponential, one organ per site), L2-fixed variants branch-specific (one organ
    each), and recurrent artefacts present in every organ except the normal N at VAF ~0.3."""
    rng = np.random.default_rng(seed)
    var_dir = tmp_path / "variants"; var_dir.mkdir()
    a2 = 0.925 - c1
    for dmin in depths:
        rows = []; pos = 1000
        r2 = np.random.default_rng(seed + 1)         # same variants in every tier
        for kind, nvar in (("bg", n_bg), ("L2", n_l2), ("art", n_art)):
            for k in range(nvar):
                pos += 1; q = min(r2.exponential(1 / bg_rate), 0.45); owner = samples[k % 4]
                for s in samples:
                    d = int(rng.integers(dmin + 40, dmin + 70))
                    if kind == "bg":
                        p = q if s == owner else 0.0
                    elif kind == "L2":
                        p = 0.5 * a2 if s == owner else 0.0
                    else:
                        p = 0.3 if s != "N" else 0.0
                    y = int(rng.binomial(d, p * (1 - 2e-3) + 1e-3))
                    rows.append(dict(sample=s, chrom="chr1", pos=pos, ref="A", alt="T", ref_depth=d - y, alt_depth=y, DP=d, VAF=y / d))
        pd.DataFrame(rows).to_csv(var_dir / f"tree1_dp_{dmin}_vaf.csv", index=False)
    g = tmp_path / "genome.csv"
    pd.DataFrame({"tree": ["tree1"] * len(depths), "depth_filter": list(depths), "effective_genome_size": [1_000_000] * len(depths)}).to_csv(g, index=False)
    r = tmp_path / "root.csv"
    pd.DataFrame({"sample": list(samples), "root2tip_length": [10, 11, 12, 13, 5]}).to_csv(r, index=False)
    return var_dir, g, r


def _cfg(tmp_path, var_dir, g, r, **kw):
    base = dict(variants=str(var_dir), effective_genome_size=str(g), root2tip=str(r), dataset_label="t042",
                outdir=str(tmp_path / kw.pop("out", "out")), lambda_grid="10,30,60,120", write_plots=False,
                layer_proportions="L1=0.06,L2=0.865", em_n_starts=3, selected_dp_window="30:33",
                control_organs={"tree1": "N"})
    base.update(kw)
    return EstimateConfig(**base)


def test_run_with_all_organs_screen(tmp_path):
    var_dir, g, r = _write(tmp_path)
    out = run_estimate(_cfg(tmp_path, var_dir, g, r, all_organs_screen=True))
    od = tmp_path / "out"
    summ = pd.read_csv(od / "18_all_organs_screen.csv"); sites = pd.read_csv(od / "18_all_organs_screened_sites.csv")
    assert summ.n_screened.iloc[0] >= 38                      # nearly all 40 artefacts (positions 1461-1500)
    assert sites.pos.between(1461, 1500).mean() > 0.9         # and few others
    meta = json.loads((od / "00_run_metadata.json").read_text())
    assert meta["screen.all_organs"] is True and meta["screen.all_organs.min_alt"] == 1
    vc = pd.read_csv(od / "07_variant_calls_selected_window.csv")
    assert not vc.pos.isin(sites.pos).any()
    assert "all_organs_screen" in out


def test_screen_off_writes_nothing(tmp_path):
    var_dir, g, r = _write(tmp_path, n_bg=150, n_l2=20, n_art=10)
    out = run_estimate(_cfg(tmp_path, var_dir, g, r))
    assert not (tmp_path / "out" / "18_all_organs_screen.csv").exists() and "all_organs_screen" not in out
    meta = json.loads((tmp_path / "out" / "00_run_metadata.json").read_text())
    assert meta["screen.all_organs"] is False and meta["diagnostics.resolvability.background_quantile"] == 0.95


def test_background_condition_in_resolvability(tmp_path):
    # L1 share 0.06 (mu_L1 0.03) inside a background with mean 1/60: L1 lies below the background's 95% quantile
    var_dir, g, r = _write(tmp_path, n_art=0)
    run_estimate(_cfg(tmp_path, var_dir, g, r, out="gate"))
    rv = pd.read_csv(tmp_path / "gate" / "14_layer_resolvability.csv").set_index("layer")
    assert {"background_quantile", "mean_background_quantile_vaf", "above_background"} <= set(rv.columns)
    assert not bool(rv.loc["L1", "above_background"]) and not bool(rv.loc["L1", "resolvable"])
    assert bool(rv.loc["L2", "above_background"])
    q = float(rv.loc["L1", "mean_background_quantile_vaf"])
    assert 0.03 < q < 0.2
    # without the condition (<= 0.4.1 behaviour) the flag is the recall rule alone
    run_estimate(_cfg(tmp_path, var_dir, g, r, out="norule", resolvability_background_quantile=None))
    rv0 = pd.read_csv(tmp_path / "norule" / "14_layer_resolvability.csv").set_index("layer")
    for L in rv0.index:
        assert bool(rv0.loc[L, "resolvable"]) == (rv0.loc[L, "mean_model_implied_recall"] >= rv0.loc[L, "min_recall"])
    assert rv0.above_background.isna().all()


def test_config_keys(tmp_path):
    for f in ("v.tsv", "g.tsv", "r.tsv"):
        (tmp_path / f).write_text("x\n")
    raw = {"input": {"variants": "v.tsv", "effective_genome_size": "g.tsv", "root2tip": "r.tsv",
                                        "control_organs": {"t": "N"}},
           "screen": {"all_organs": True, "min_alt": 2}, "diagnostics": {"resolvability_background_quantile": None}}
    p = tmp_path / "c.json"; p.write_text(json.dumps(raw))
    cfg, base = vcfg.load_config(p)
    est = vcfg.to_estimate_config(cfg, base)
    assert est.all_organs_screen is True and est.all_organs_screen_min_alt == 2
    assert est.resolvability_background_quantile is None

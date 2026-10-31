from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from vafsoma.estimate import EstimateConfig, run_estimate


def test_smoke_fixed_mode(tmp_path: Path):
    rng = np.random.default_rng(3)
    samples = ["S1", "S2"]
    var_dir = tmp_path / "variants"
    var_dir.mkdir()
    for depth in [12, 13, 14, 15]:
        rows = []
        pos = 1000
        for p, nvar in [(0.02, 20), (0.36, 8)]:
            for _ in range(nvar):
                pos += 1
                for sample in samples:
                    d = int(rng.integers(depth + 10, depth + 30))
                    y = int(rng.binomial(d, p))
                    rows.append(
                        {
                            "sample": sample,
                            "chrom": "chr1",
                            "pos": pos,
                            "ref": "A",
                            "alt": "T",
                            "ref_depth": d - y,
                            "alt_depth": y,
                            "DP": d,
                            "VAF": y / d,
                        }
                    )
        pd.DataFrame(rows).to_csv(var_dir / f"tree1_dp_{depth}_vaf.csv", index=False)

    variant_zip = tmp_path / "variants.zip"
    with zipfile.ZipFile(variant_zip, "w") as zf:
        for f in var_dir.glob("*.csv"):
            zf.write(f, f.name)

    genome = tmp_path / "genome.csv"
    pd.DataFrame(
        {"tree": ["tree1"] * 4, "depth_filter": [12, 13, 14, 15], "effective_genome_size": [1_000_000] * 4}
    ).to_csv(genome, index=False)
    root = tmp_path / "root.csv"
    pd.DataFrame({"sample": samples, "root2tip_length": [10, 11]}).to_csv(root, index=False)

    outdir = tmp_path / "out"
    outputs = run_estimate(
        EstimateConfig(
            variants=str(variant_zip),
            effective_genome_size=str(genome),
            root2tip=str(root),
            dataset_label="test",
            outdir=str(outdir),
            lambda_grid="10,50",
            write_plots=False,
        )
    )
    assert outputs["fixed_counts_by_tier"].exists()
    assert outputs["selected_window_summary"].exists()
    tiers = pd.read_csv(outputs["fixed_counts_by_tier"])
    assert len(tiers) == 4
    assert (tiers["effective_genome_size"] > 0).all()
    assert (tiers["n_units"] > 0).all()
    assert (tiers["fixed_call_density"] >= 0).all()
    assert not any(c.startswith("rate_") for c in tiers.columns)
    # window summary: counts and diagnostics flags, no rates
    summ = pd.read_csv(outputs["selected_window_summary"])
    for layer in ("L1", "L2", "L3"):
        for col in (f"mean_posterior_weighted_{layer}_count", f"mean_MAP_{layer}_count",
                    f"{layer}_resolvable", f"{layer}_status", f"{layer}_model_implied_recall"):
            assert col in summ.columns, col
        assert set(summ[f"{layer}_status"]) <= {"reportable", "unresolvable", "no_diagnostics"}
    assert {"fixation_criterion", "unit_type", "n_units", "fit_warning"} <= set(summ.columns)
    assert not any("rate" in c for c in summ.columns)
    # per-unit counts (16) and posteriors (17): consistent with each other and with 03
    units = pd.read_csv(outputs["fixed_counts_by_unit_tier"])
    post = pd.read_csv(outputs["unit_posteriors_by_tier"])
    assert set(units["fixation_criterion"]) == {"single_organ"} and set(units["unit_type"]) == {"organ"}
    assert {"root2tip_length", "effective_genome_size", "is_control", "members"} <= set(units.columns)
    s = post.groupby(["depth_filter", "unit"])["P_L2"].sum()
    for r in units.itertuples(index=False):
        assert np.isclose(r.posterior_weighted_L2_count, s.get((r.depth_filter, r.unit), 0.0), rtol=1e-12, atol=1e-12)
    tot = units[~units["is_control"]].groupby("depth_filter")["MAP_fixed_count"].sum()
    assert np.allclose(tiers.set_index("depth_filter")["sum_MAP_fixed_count"], tot)


def test_layer_design_validation_rules():
    from vafsoma.estimate import EstimateConfig, build_model_design

    try:
        build_model_design(EstimateConfig(variants="v", effective_genome_size="g", root2tip="r", active_layer="L2"))
    except ValueError as e:
        assert "active-layer can only be used" in str(e)
    else:
        raise AssertionError("Expected active-layer to be forbidden for bulk/unlayered data")

    try:
        build_model_design(EstimateConfig(variants="v", effective_genome_size="g", root2tip="r", layer_specific_seq="yes"))
    except ValueError as e:
        assert "active-layer is required" in str(e)
    else:
        raise AssertionError("Expected active-layer to be required for layer-specific data")

    d = build_model_design(
        EstimateConfig(
            variants="v",
            effective_genome_size="g",
            root2tip="r",
            layer_specific_seq="no",
            vaf_phasing="unphased",
            layer_proportions="L1=0.20,L2=0.72,L3=0.08",
        )
    )
    assert d.derived_mu.L1 == 0.10
    assert d.derived_mu.L2 == 0.36
    assert d.derived_mu.L3 == 0.04

    d = build_model_design(
        EstimateConfig(
            variants="v",
            effective_genome_size="g",
            root2tip="r",
            layer_specific_seq="yes",
            vaf_phasing="phased",
            active_layer="L2",
        )
    )
    assert d.active_components == ("E", "L2")
    assert d.derived_mu.L1 is None
    assert d.derived_mu.L2 == 1.0
    assert d.derived_mu.L3 is None


def _smoke_inputs(tmp_path: Path, name_prefix: str = "tree1"):
    rng = np.random.default_rng(3)
    samples = ["S1", "S2"]
    var_dir = tmp_path / "variants"
    var_dir.mkdir()
    for depth in [12, 13, 14, 15]:
        rows, pos = [], 1000
        for p, nvar in [(0.02, 20), (0.36, 8)]:
            for _ in range(nvar):
                pos += 1
                for sample in samples:
                    d = int(rng.integers(depth + 10, depth + 30)); y = int(rng.binomial(d, p))
                    rows.append({"sample": sample, "chrom": "chr1", "pos": pos, "ref": "A", "alt": "T",
                                 "ref_depth": d - y, "alt_depth": y, "DP": d, "VAF": y / d})
        pd.DataFrame(rows).to_csv(var_dir / f"{name_prefix}_dp_{depth}_vaf.csv", index=False)
    egs = tmp_path / "genome.csv"
    pd.DataFrame({"depth_filter": [12, 13, 14, 15], "effective_genome_size": [1_000_000] * 4}).to_csv(egs, index=False)
    root = tmp_path / "root.csv"
    pd.DataFrame({"sample": samples, "root2tip_length": [10, 11]}).to_csv(root, index=False)
    return var_dir, egs, root


def test_json_config_run_equals_estimate(tmp_path: Path):
    """`vafsoma run CONFIG` gives the same results as the estimator called directly; the tree label comes from
    input.tree because neither the file names nor the tables carry one."""
    import json
    from vafsoma.cli import main
    var_dir, egs, root = _smoke_inputs(tmp_path, name_prefix="sample")
    cfg = {"experiment": "smoke", "outdir": "res",
           "input": {"variants": "variants", "effective_genome_size": "genome.csv", "root2tip": "root.csv",
                     "tree": "treeX"},
           "model": {"layer_proportions": {"L1": 0.20, "L2": 0.72, "L3": 0.08}},
           "background": {"lambda_grid": [10, 50]}, "compute": {"plots": False}}
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    assert main(["check", str(tmp_path / "run.json")]) == 0
    assert main(["run", str(tmp_path / "run.json")]) == 0
    out = tmp_path / "res" / "smoke"
    assert (out / "00_config_resolved.json").exists()
    a = pd.read_csv(next(out.glob("03_fixed_counts*.csv")))
    ref = run_estimate(EstimateConfig(variants=str(var_dir), effective_genome_size=str(egs), root2tip=str(root),
                                      dataset_label="smoke", outdir=str(tmp_path / "ref"), tree_label="treeX",
                                      lambda_grid="10,50", write_plots=False))
    b = pd.read_csv(ref["fixed_counts_by_tier"])
    assert set(a["tree"]) == {"treeX"}
    pd.testing.assert_frame_equal(a, b)


def test_config_errors_and_template(tmp_path: Path):
    import json
    from vafsoma import config as vconf
    from vafsoma.cli import main
    t = json.loads(vconf.template_text())
    assert t["model"]["layer_proportions"] == {"L1": 0.13, "L2": 0.84, "L3": 0.03}
    (tmp_path / "bad.json").write_text(json.dumps({"experiment": "x", "input": {"variants": "missing.zip"},
                                                   "model": {"layer_proportions": {"L1": 0.5, "L2": 0.6}}}))
    cfg, base = vconf.load_config(tmp_path / "bad.json")
    problems = vconf.validate(cfg, base)
    assert any("sum to 1" in p for p in problems) and any("input.variants not found" in p for p in problems)
    assert main(["check", str(tmp_path / "bad.json")]) == 2
    (tmp_path / "typo.json").write_text(json.dumps({"experimnt": "x"}))
    try:
        vconf.load_config(tmp_path / "typo.json")
    except vconf.ConfigError as e:
        assert "unknown configuration key: experimnt" in str(e)
    else:
        raise AssertionError("expected an unknown-key error")


def test_removed_rate_keys_are_ignored_with_a_warning(tmp_path: Path):
    """Configuration files of vafSOMA <= 0.2.3 with bootstrap / model.ploidy_factor still load (keys ignored)."""
    import json
    import warnings
    from vafsoma import config as vconf
    (tmp_path / "old.json").write_text(json.dumps({"experiment": "x", "bootstrap": {"B": 5},
                                                   "model": {"ploidy_factor": 1.0}}))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg, _ = vconf.load_config(tmp_path / "old.json")
    assert "bootstrap" not in cfg and "ploidy_factor" not in cfg["model"]
    assert sum("removed in vafSOMA 0.3.0" in str(x.message) for x in w) == 2

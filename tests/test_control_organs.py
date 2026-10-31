"""Validation for EstimateConfig.control_organs: the per-tree Mutect2-style
normal/reference organ that the upstream caller used as the baseline the
other organs' candidate SNPs were called against. Such an organ is expected
to show almost no alt-supporting reads at its own tree's candidate SNP set
BY CONSTRUCTION (see an empirical finding: leaf14/leaf19/leaf08 each had
~1-3% alt-positivity vs ~65-86% for their tree-mates, at identical sites and
comparable depth) -- not because it lacks somatic mutations. It is therefore
excluded from the depth-window metric and from the count sums of 03, while
its calls and counts are still reported -- flagged via is_control_organ /
is_control -- in 08, 16 and 17.

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_control_organs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vafsoma.estimate import EstimateConfig, ModelDesign, build_organ_variant_calls, fit_tree_depth_task
from vafsoma.model import MuTriple
from vafsoma.utils import variant_key_frame


def _cfg(**kwargs) -> EstimateConfig:
    defaults = dict(variants="v", effective_genome_size="g", root2tip="r", dataset_label="TestDS")
    defaults.update(kwargs)
    return EstimateConfig(**defaults)


def _fit_tree_depth_fixture():
    """orgA: 8 SNPs, clean L2-like signal (VAF~0.36). orgCTRL: the same 8
    positions, ref-only except two positions with a couple of background-noise
    alt reads (VAF=0.02) -- mirrors Mutect2's residual normal-artifact rate
    from the real data, small but nonzero, never L2-like."""
    rows = []
    for i in range(8):
        rows.append({"tree": "treeX", "chrom": "chr1", "pos": 100 + i, "ref": "A", "alt": "G", "sample": "orgA",
                      "depth_filter": 20, "ref_depth": 64, "alt_depth": 36, "DP": 100, "VAF": 0.36})
    for i in range(8):
        alt = 2 if i < 2 else 0
        rows.append({"tree": "treeX", "chrom": "chr1", "pos": 100 + i, "ref": "A", "alt": "G", "sample": "orgCTRL",
                      "depth_filter": 20, "ref_depth": 100 - alt, "alt_depth": alt, "DP": 100, "VAF": alt / 100})
    variants = pd.DataFrame(rows)
    variants["variant_id"] = variant_key_frame(variants)

    genome = pd.DataFrame([{"tree": "treeX", "depth_filter": 20, "effective_genome_size": 1.0e8}])
    root = pd.DataFrame([
        {"tree": "treeX", "sample": "orgA", "root2tip_length": 10.0},
        {"tree": "treeX", "sample": "orgCTRL", "root2tip_length": 10.0},
    ])
    design = ModelDesign(
        layer_specific_seq="no", vaf_phasing="unphased", vaf_scale=0.5,
        active_components=("E", "L1", "L2", "L3"),
        layer_proportions={"L1": 0.20, "L2": 0.72, "L3": 0.08},
        derived_mu=MuTriple(L1=0.10, L2=0.36, L3=0.04),
    )
    mu_grid = [MuTriple(L1=0.10, L2=0.36, L3=0.04)]
    rho_grid = [0.0]
    lambda_grid = np.array([20.0, 40.0, 60.0, 80.0])
    return variants, genome, root, design, mu_grid, rho_grid, lambda_grid


def test_control_organ_excluded_from_metric_but_still_classified():
    variants, genome, root, design, mu_grid, rho_grid, lambda_grid = _fit_tree_depth_fixture()
    cfg = _cfg(em_n_starts=4, lambda_refine=False, control_organs={"treeX": "orgCTRL"})

    fit_rows, variant_calls, tier_row, organ_calls, unit_rows = fit_tree_depth_task(
        ("treeX", 20), variants, genome, root, cfg, design, mu_grid, rho_grid, lambda_grid
    )

    assert tier_row["control_organ"] == "orgCTRL"
    assert tier_row["n_samples"] == 2  # all organs
    assert tier_row["n_units"] == 1  # orgCTRL excluded from the metric and the count sums

    # Both organs are written to the per-unit counts (16), the control flagged.
    units = {r["unit"]: r for r in unit_rows}
    assert set(units) == {"orgA", "orgCTRL"}
    assert units["orgCTRL"]["is_control"] and not units["orgA"]["is_control"]
    assert tier_row["sum_MAP_L2_count"] == units["orgA"]["MAP_L2_count"]

    # orgCTRL's own rows are still classified and present in organ_calls (not dropped),
    # just flagged -- this is the QC-transparency half of the design.
    assert set(organ_calls["sample"].unique()) == {"orgA", "orgCTRL"}
    flags = organ_calls.groupby("sample")["is_control_organ"].first()
    assert flags["orgA"] is np.False_ or flags["orgA"] == False  # noqa: E712
    assert flags["orgCTRL"] is np.True_ or flags["orgCTRL"] == True  # noqa: E712
    # and none of orgCTRL's background-noise reads should MAP to L2 (they're noise-level, not signal)
    ctrl_rows = organ_calls[organ_calls["sample"] == "orgCTRL"]
    assert not (ctrl_rows["obs_inferred_component_MAP"] == "L2").any()


def test_excluding_control_organ_changes_the_window_metric_the_way_the_math_predicts():
    """With orgCTRL in the average, an organ contributing ~0 fixed calls is averaged in on equal footing with orgA,
    halving the depth-window metric (mean over units of N_s / (2 G L_s)) relative to excluding it."""
    variants, genome, root, design, mu_grid, rho_grid, lambda_grid = _fit_tree_depth_fixture()
    cfg_ctrl = _cfg(em_n_starts=4, lambda_refine=False, control_organs={"treeX": "orgCTRL"})
    cfg_noctrl = _cfg(em_n_starts=4, lambda_refine=False)

    _, _, row_ctrl, _, _ = fit_tree_depth_task(
        ("treeX", 20), variants, genome, root, cfg_ctrl, design, mu_grid, rho_grid, lambda_grid
    )
    _, _, row_noctrl, _, _ = fit_tree_depth_task(
        ("treeX", 20), variants, genome, root, cfg_noctrl, design, mu_grid, rho_grid, lambda_grid
    )

    assert row_ctrl["fixed_call_density"] > row_noctrl["fixed_call_density"]
    ratio = row_ctrl["fixed_call_density"] / row_noctrl["fixed_call_density"]
    assert abs(ratio - 2.0) < 1e-9
    # the density is exactly the orgA count over 2 G L
    assert row_ctrl["fixed_call_density"] == row_ctrl["sum_MAP_fixed_count"] / (2.0 * 1.0e8 * 10.0)


def test_unconfigured_tree_is_unaffected():
    """A tree with no control_organs entry must behave exactly as before --
    this is an opt-in feature, not a behavior change for existing configs."""
    variants, genome, root, design, mu_grid, rho_grid, lambda_grid = _fit_tree_depth_fixture()
    cfg = _cfg(em_n_starts=4, lambda_refine=False, control_organs={"some_other_tree": "X"})

    _, _, tier_row, organ_calls, unit_rows = fit_tree_depth_task(
        ("treeX", 20), variants, genome, root, cfg, design, mu_grid, rho_grid, lambda_grid
    )
    assert tier_row["control_organ"] == ""
    assert tier_row["n_units"] == tier_row["n_samples"] == 2
    assert [r["unit"] for r in unit_rows] == ["orgA", "orgCTRL"]
    assert not organ_calls["is_control_organ"].any()


def test_bad_control_organ_id_raises():
    """A typo'd control-organ id (not among the tree's actual samples) must fail
    loudly rather than silently doing nothing -- see the real-data workflow,
    where guessing the wrong sample id would otherwise go unnoticed."""
    variants, genome, root, design, mu_grid, rho_grid, lambda_grid = _fit_tree_depth_fixture()
    cfg = _cfg(em_n_starts=4, lambda_refine=False, control_organs={"treeX": "typo_sample_id"})
    try:
        fit_tree_depth_task(("treeX", 20), variants, genome, root, cfg, design, mu_grid, rho_grid, lambda_grid)
        raise AssertionError("expected ValueError for an unknown control-organ id")
    except ValueError as e:
        assert "typo_sample_id" in str(e)


def _org_row(tree, pos, sample, dpf, alt_depth, dp=100):
    return {"tree": tree, "chrom": "chr1", "pos": pos, "ref": "A", "alt": "G", "sample": sample,
            "depth_filter": dpf, "ref_depth": dp - alt_depth, "alt_depth": alt_depth, "DP": dp, "VAF": alt_depth / dp}


def _fit_row(tree, dpf):
    return {"tree": tree, "depth_filter": dpf, "active_components": "E;L1;L2;L3", "mu_L1": 0.10, "mu_L2": 0.36,
            "mu_L3": 0.04, "layer_rho": 0.0, "lambda_hat": 40.0, "loglik": -1.0, "pi_E": 0.95, "pi_L1": 0.0,
            "pi_L2": 0.045, "pi_L3": 0.005, "em_n_iter": 5, "em_converged": True, "n_positive_rows": 10,
            "n_unique_positive_variants": 3}


def test_build_organ_variant_calls_flags_the_control_organ_column():
    """The 08_organ_variant_calls_window.csv audit table must carry the same
    is_control_organ flag as fit_tree_depth_task's organ_calls, so a reader can
    filter it out of any per-organ summary without needing a side lookup."""
    cfg = _cfg(control_organs={"treeX": "orgCTRL"})
    selected = pd.DataFrame([{"tree": "treeX", "selected_window_start": 20, "selected_window_end": 20}])
    fit_df = pd.DataFrame([_fit_row("treeX", 20)])
    rows = [
        _org_row("treeX", 100, "orgA", 20, alt_depth=50),
        _org_row("treeX", 100, "orgCTRL", 20, alt_depth=0),
        _org_row("treeX", 100, "orgA", 5, alt_depth=50),
        _org_row("treeX", 100, "orgCTRL", 5, alt_depth=0),
    ]
    variants = pd.DataFrame(rows)
    out = build_organ_variant_calls(variants, fit_df, selected, cfg)

    flags = out.set_index("sample")["is_control_organ"]
    assert flags["orgA"] == False  # noqa: E712
    assert flags["orgCTRL"] == True  # noqa: E712


def _fit_tree_depth_fixture_two_controls():
    """orgA: clean L2-like signal. orgCTRL1/orgCTRL2: two separate normal
    samples both used as Mutect2 baselines in one joint multi-sample call
    (e.g. a design with two normal organs per tree) --
    both should behave exactly like the single-control case, just excluded
    together."""
    rows = []
    for i in range(8):
        rows.append({"tree": "treeY", "chrom": "chr1", "pos": 100 + i, "ref": "A", "alt": "G", "sample": "orgA",
                      "depth_filter": 20, "ref_depth": 64, "alt_depth": 36, "DP": 100, "VAF": 0.36})
    for ctrl in ("orgCTRL1", "orgCTRL2"):
        for i in range(8):
            alt = 2 if i < 2 else 0
            rows.append({"tree": "treeY", "chrom": "chr1", "pos": 100 + i, "ref": "A", "alt": "G", "sample": ctrl,
                         "depth_filter": 20, "ref_depth": 100 - alt, "alt_depth": alt, "DP": 100, "VAF": alt / 100})
    variants = pd.DataFrame(rows)
    variants["variant_id"] = variant_key_frame(variants)

    genome = pd.DataFrame([{"tree": "treeY", "depth_filter": 20, "effective_genome_size": 1.0e8}])
    root = pd.DataFrame([
        {"tree": "treeY", "sample": "orgA", "root2tip_length": 10.0},
        {"tree": "treeY", "sample": "orgCTRL1", "root2tip_length": 10.0},
        {"tree": "treeY", "sample": "orgCTRL2", "root2tip_length": 10.0},
    ])
    design = ModelDesign(
        layer_specific_seq="no", vaf_phasing="unphased", vaf_scale=0.5,
        active_components=("E", "L1", "L2", "L3"),
        layer_proportions={"L1": 0.20, "L2": 0.72, "L3": 0.08},
        derived_mu=MuTriple(L1=0.10, L2=0.36, L3=0.04),
    )
    mu_grid = [MuTriple(L1=0.10, L2=0.36, L3=0.04)]
    rho_grid = [0.0]
    lambda_grid = np.array([20.0, 40.0, 60.0, 80.0])
    return variants, genome, root, design, mu_grid, rho_grid, lambda_grid


def test_multiple_control_organs_per_tree_are_all_excluded():
    """A tree can have MORE THAN ONE control organ (e.g. two Mutect2 normal
    samples used in one joint call) -- both must be excluded from the window
    metric, both flagged is_control_organ, both still classified."""
    variants, genome, root, design, mu_grid, rho_grid, lambda_grid = _fit_tree_depth_fixture_two_controls()
    cfg = _cfg(em_n_starts=4, lambda_refine=False, control_organs={"treeY": ["orgCTRL1", "orgCTRL2"]})

    _, _, tier_row, organ_calls, unit_rows = fit_tree_depth_task(
        ("treeY", 20), variants, genome, root, cfg, design, mu_grid, rho_grid, lambda_grid
    )

    assert tier_row["control_organ"] == "orgCTRL1;orgCTRL2"
    assert tier_row["n_samples"] == 3
    assert tier_row["n_units"] == 1  # only orgA
    assert [r["unit"] for r in unit_rows if not r["is_control"]] == ["orgA"]

    flags = organ_calls.groupby("sample")["is_control_organ"].first()
    assert flags["orgA"] == False  # noqa: E712
    assert flags["orgCTRL1"] == True  # noqa: E712
    assert flags["orgCTRL2"] == True  # noqa: E712
    # both control organs' rows are still classified, not dropped
    assert set(organ_calls["sample"].unique()) == {"orgA", "orgCTRL1", "orgCTRL2"}


def test_parse_control_organs_plus_syntax_for_multiple_organs():
    """CLI/config string syntax: '+' between organ ids for one tree."""
    from vafsoma.utils import parse_control_organs

    out = parse_control_organs("tree1=leaf14,tree2=leaf19+leaf17,tree3=leaf08+leaf19")
    assert out["tree1"] == "leaf14"
    assert out["tree2"] == ["leaf19", "leaf17"]
    assert out["tree3"] == ["leaf08", "leaf19"]


if __name__ == "__main__":
    test_control_organ_excluded_from_metric_but_still_classified()
    print("  PASS test_control_organ_excluded_from_metric_but_still_classified")
    test_excluding_control_organ_changes_the_window_metric_the_way_the_math_predicts()
    print("  PASS test_excluding_control_organ_changes_the_window_metric_the_way_the_math_predicts")
    test_unconfigured_tree_is_unaffected()
    print("  PASS test_unconfigured_tree_is_unaffected")
    test_bad_control_organ_id_raises()
    print("  PASS test_bad_control_organ_id_raises")
    test_build_organ_variant_calls_flags_the_control_organ_column()
    print("  PASS test_build_organ_variant_calls_flags_the_control_organ_column")
    test_multiple_control_organs_per_tree_are_all_excluded()
    print("  PASS test_multiple_control_organs_per_tree_are_all_excluded")
    test_parse_control_organs_plus_syntax_for_multiple_organs()
    print("  PASS test_parse_control_organs_plus_syntax_for_multiple_organs")
    print("All tests passed.")

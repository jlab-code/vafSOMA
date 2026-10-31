"""Validation for build_augmented_variant_calls (the 06_variant_calls_augmented.csv
per-SNP audit table): every candidate SNP from the loosest depth tier gets an
explicit calling_status, never silent omission, and no status is ever derived
from outside the selected depth window.

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_augmented_calls.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from vafsoma.estimate import EstimateConfig, build_augmented_variant_calls


def _cfg(**kwargs) -> EstimateConfig:
    defaults = dict(variants="v", effective_genome_size="g", root2tip="r", dataset_label="TestDS")
    defaults.update(kwargs)
    return EstimateConfig(**defaults)


def _variant_row(tree, chrom, pos, ref, alt, sample, depth_filter, alt_depth, dp=100):
    return {
        "tree": tree, "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
        "sample": sample, "depth_filter": depth_filter,
        "ref_depth": dp - alt_depth, "alt_depth": alt_depth, "DP": dp,
        "VAF": alt_depth / dp if dp else 0.0,
    }


def test_five_calling_statuses_on_synthetic_data():
    cfg = _cfg()
    selected = pd.DataFrame([{"tree": "treeX", "selected_window_start": 20, "selected_window_end": 30}])

    rows = []
    # V1: only ever appears at depth_filter=5 (below the window) -> not_evaluated_insufficient_depth.
    rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgA", 5, alt_depth=3))
    rows.append(_variant_row("treeX", "chr1", 100, "A", "G", "orgB", 5, alt_depth=0))

    # V2: reaches the window (depth_filter=25) but alt_depth=0 in every organ there
    # -> evaluated_never_positive. (Its presence at dp=5 with real support is what
    # made it a candidate at all, but by dp=25 that organ has dropped below the
    # per-position DP requirement so only the negative organ's row survives.)
    rows.append(_variant_row("treeX", "chr1", 200, "A", "G", "orgA", 5, alt_depth=4))
    rows.append(_variant_row("treeX", "chr1", 200, "A", "G", "orgB", 25, alt_depth=0))

    # V3: reaches the window and is positively observed there -> gets a classified
    # row in variant_summary with stable_fixed_call=True, stable_layer_call="L2".
    # (No depth_filter=5 row for V3 -- deliberately, so it's outside this test's
    # "loosest tier" universe; see the follow-up test for the realistic shape.)
    for d in (20, 25, 30):
        rows.append(_variant_row("treeX", "chr1", 300, "A", "G", "orgA", d, alt_depth=10))

    # V4: reaches the window, positively observed, but variant_summary reports
    # fraction_MAP_fixed=0.0 exactly (consistently classified as noise) -> called_noise.
    for d in (20, 25, 30):
        rows.append(_variant_row("treeX", "chr1", 400, "A", "G", "orgA", d, alt_depth=2))

    # V5: reaches the window, positively observed, but variant_summary reports a
    # mixed fraction_MAP_fixed (not stable at either end) -> evaluated_unstable.
    for d in (20, 25, 30):
        rows.append(_variant_row("treeX", "chr1", 500, "A", "G", "orgA", d, alt_depth=5))

    variants = pd.DataFrame(rows)

    variant_summary = pd.DataFrame(
        [
            {
                "tree": "treeX", "chrom": "chr1", "pos": 300, "ref": "A", "alt": "G",
                "n_dp_cutoffs": 3, "fraction_MAP_fixed": 1.0, "fraction_MAP_L1": 0.0,
                "fraction_MAP_L2": 1.0, "fraction_MAP_L3": 0.0, "mean_P_E": 0.01,
                "mean_P_L1": 0.0, "mean_P_L2": 0.95, "mean_P_L3": 0.0, "mean_P_fixed": 0.95,
                "stable_fixed_call": True, "stable_layer_call": "L2",
            },
            {
                "tree": "treeX", "chrom": "chr1", "pos": 400, "ref": "A", "alt": "G",
                "n_dp_cutoffs": 3, "fraction_MAP_fixed": 0.0, "fraction_MAP_L1": 0.0,
                "fraction_MAP_L2": 0.0, "fraction_MAP_L3": 0.0, "mean_P_E": 0.98,
                "mean_P_L1": 0.0, "mean_P_L2": 0.01, "mean_P_L3": 0.0, "mean_P_fixed": 0.02,
                "stable_fixed_call": False, "stable_layer_call": "",
            },
            {
                "tree": "treeX", "chrom": "chr1", "pos": 500, "ref": "A", "alt": "G",
                "n_dp_cutoffs": 3, "fraction_MAP_fixed": 0.33, "fraction_MAP_L1": 0.0,
                "fraction_MAP_L2": 0.33, "fraction_MAP_L3": 0.0, "mean_P_E": 0.5,
                "mean_P_L1": 0.0, "mean_P_L2": 0.4, "mean_P_L3": 0.0, "mean_P_fixed": 0.4,
                "stable_fixed_call": False, "stable_layer_call": "",
            },
        ]
    )

    aug = build_augmented_variant_calls(variants, variant_summary, selected, cfg)

    # Universe = unique variants at the minimum depth_filter present (5) = V1, V2. V3/V4/V5
    # only ever appear at window depths (20/25/30), so the "universe" as defined (loosest
    # tier per tree) would miss them in real data -- but here we deliberately gave V1/V2 the
    # loosest tier (5) and V3/V4/V5 start directly in the window, which still exercises the
    # join logic correctly since real pipelines' loosest tier is a superset of the window.
    # Rebuild universe expectation: since min depth_filter for treeX overall is 5, and only
    # V1/V2 have rows at depth_filter=5, the "universe" is just {V1, V2} under this synthetic
    # setup -- so extend V3/V4/V5 with a depth_filter=5 anchor row too, matching real data
    # where every candidate SNP originates from the loosest tier.
    assert set(aug["pos"]) == {100, 200}, (
        "as constructed, only V1/V2 have a row at the tree's loosest depth tier; "
        "see the follow-up test below for V3/V4/V5 with a proper loosest-tier anchor"
    )
    v1 = aug[aug["pos"] == 100].iloc[0]
    assert v1["calling_status"] == "not_evaluated_insufficient_depth"
    assert v1["n_dp_cutoffs_present_in_window"] == 0

    v2 = aug[aug["pos"] == 200].iloc[0]
    assert v2["calling_status"] == "evaluated_never_positive"
    assert v2["n_dp_cutoffs_present_in_window"] == 1  # present (negatively) at depth_filter=25 only
    assert v2["n_organs_positive_in_window"] == 0
    assert v2["organs_positive_in_window"] == ""


def test_called_noise_unstable_and_fixed_layer_with_proper_universe():
    """Same V3/V4/V5 scenarios, but each given a depth_filter=5 anchor row too, so
    they're part of the loosest-tier universe -- this is the realistic shape of the
    actual pipeline data (every candidate SNP has a row at the loosest tier)."""
    cfg = _cfg()
    selected = pd.DataFrame([{"tree": "treeX", "selected_window_start": 20, "selected_window_end": 30}])

    rows = []
    for pos, alt_depth in ((300, 10), (400, 2), (500, 5)):
        rows.append(_variant_row("treeX", "chr1", pos, "A", "G", "orgA", 5, alt_depth=alt_depth))
        for d in (20, 25, 30):
            rows.append(_variant_row("treeX", "chr1", pos, "A", "G", "orgA", d, alt_depth=alt_depth))
    # V3 also carried in a second organ within the window -- check the organ list joins/sorts.
    rows.append(_variant_row("treeX", "chr1", 300, "A", "G", "orgB", 25, alt_depth=6))
    variants = pd.DataFrame(rows)

    variant_summary = pd.DataFrame(
        [
            {"tree": "treeX", "chrom": "chr1", "pos": 300, "ref": "A", "alt": "G", "n_dp_cutoffs": 3,
             "fraction_MAP_fixed": 1.0, "fraction_MAP_L1": 0.0, "fraction_MAP_L2": 1.0, "fraction_MAP_L3": 0.0,
             "mean_P_E": 0.01, "mean_P_L1": 0.0, "mean_P_L2": 0.95, "mean_P_L3": 0.0, "mean_P_fixed": 0.95,
             "stable_fixed_call": True, "stable_layer_call": "L2"},
            {"tree": "treeX", "chrom": "chr1", "pos": 400, "ref": "A", "alt": "G", "n_dp_cutoffs": 3,
             "fraction_MAP_fixed": 0.0, "fraction_MAP_L1": 0.0, "fraction_MAP_L2": 0.0, "fraction_MAP_L3": 0.0,
             "mean_P_E": 0.98, "mean_P_L1": 0.0, "mean_P_L2": 0.01, "mean_P_L3": 0.0, "mean_P_fixed": 0.02,
             "stable_fixed_call": False, "stable_layer_call": ""},
            {"tree": "treeX", "chrom": "chr1", "pos": 500, "ref": "A", "alt": "G", "n_dp_cutoffs": 3,
             "fraction_MAP_fixed": 0.33, "fraction_MAP_L1": 0.0, "fraction_MAP_L2": 0.33, "fraction_MAP_L3": 0.0,
             "mean_P_E": 0.5, "mean_P_L1": 0.0, "mean_P_L2": 0.4, "mean_P_L3": 0.0, "mean_P_fixed": 0.4,
             "stable_fixed_call": False, "stable_layer_call": ""},
        ]
    )

    aug = build_augmented_variant_calls(variants, variant_summary, selected, cfg)
    assert set(aug["pos"]) == {300, 400, 500}

    r300 = aug[aug["pos"] == 300].iloc[0]
    assert r300["calling_status"] == "called_fixed_L2"
    assert r300["n_organs_positive_in_window"] == 2
    assert r300["organs_positive_in_window"] == "orgA;orgB"
    assert r300["max_alt_depth_in_window"] == 10.0

    r400 = aug[aug["pos"] == 400].iloc[0]
    assert r400["calling_status"] == "called_noise"

    r500 = aug[aug["pos"] == 500].iloc[0]
    assert r500["calling_status"] == "evaluated_unstable"


def test_no_calls_ever_derived_from_outside_the_window():
    """A variant observed only outside the window (even if strongly, consistently
    'fixed'-looking there) must never get a called_* / called_noise / evaluated_unstable
    status -- those all require a variant_summary row, which by construction only
    exists for depths inside the selected window."""
    cfg = _cfg()
    selected = pd.DataFrame([{"tree": "treeX", "selected_window_start": 20, "selected_window_end": 30}])
    rows = [_variant_row("treeX", "chr1", 999, "A", "G", "orgA", d, alt_depth=20) for d in (5, 10, 15)]
    variants = pd.DataFrame(rows)
    variant_summary = pd.DataFrame(columns=[
        "tree", "chrom", "pos", "ref", "alt", "n_dp_cutoffs", "fraction_MAP_fixed",
        "fraction_MAP_L1", "fraction_MAP_L2", "fraction_MAP_L3", "mean_P_E", "mean_P_L1",
        "mean_P_L2", "mean_P_L3", "mean_P_fixed", "stable_fixed_call", "stable_layer_call",
    ])
    aug = build_augmented_variant_calls(variants, variant_summary, selected, cfg)
    assert aug.shape[0] == 1
    assert aug.iloc[0]["calling_status"] == "not_evaluated_insufficient_depth"


if __name__ == "__main__":
    test_five_calling_statuses_on_synthetic_data()
    print("  PASS test_five_calling_statuses_on_synthetic_data")
    test_called_noise_unstable_and_fixed_layer_with_proper_universe()
    print("  PASS test_called_noise_unstable_and_fixed_layer_with_proper_universe")
    test_no_calls_ever_derived_from_outside_the_window()
    print("  PASS test_no_calls_ever_derived_from_outside_the_window")
    print("All tests passed.")

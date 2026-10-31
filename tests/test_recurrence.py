"""Tests for vafsoma.recurrence -- the organ-sharing recurrence test (see the
module docstring).

Runs with pytest, or directly:
    PYTHONPATH=. python3 tests/test_recurrence.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vafsoma.model import ExponentialComponentCache, FitResult, MuTriple
from vafsoma.recurrence import (
    ALREADY_CONFIDENT_STATUSES,
    RECURRENCE_VALIDATED_STATUS,
    apply_recurrence_boost,
    build_recurrence_report,
    enumerate_nontrivial_subsets,
    four_point_filter,
    organ_presence_patterns,
    recurrence_significance_table,
    simulate_null_organ_patterns,
    split_compatible,
)


def test_enumerate_nontrivial_subsets_count_and_exclusions():
    organs = ["O1", "O2", "O3", "O4"]
    subsets = enumerate_nontrivial_subsets(organs)
    assert len(subsets) == 2**4 - 2
    assert () not in subsets
    assert tuple(organs) not in subsets
    assert ("O1",) in subsets
    assert ("O1", "O2") in subsets
    assert ("O1", "O2", "O3") in subsets


def _organ_calls_row(tree, pos, sample, mean_P_fixed, status="called_fixed_L1"):
    return {
        "tree": tree, "chrom": "chr1", "pos": pos, "ref": "A", "alt": "T", "sample": sample,
        "organ_calling_status": status, "mean_P_fixed": mean_P_fixed,
    }


def test_organ_presence_patterns_threshold_and_missing_organ_handling():
    # Variant at pos=1: O1 strongly fixed (0.9), O2 weakly so (0.4, below threshold),
    # O3 never positive (NaN -> treated as absent), O4 not present at all in the table
    # (must still be treated as absent, not dropped).
    rows = [
        _organ_calls_row("t1", 1, "O1", 0.9),
        _organ_calls_row("t1", 1, "O2", 0.4),
        _organ_calls_row("t1", 1, "O3", np.nan, status="evaluated_never_positive"),
    ]
    oc = pd.DataFrame(rows)
    out = organ_presence_patterns(oc, ["O1", "O2", "O3", "O4"], threshold=0.5)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["organ_pattern"] == ("O1",)
    assert row["n_organs_present"] == 1


def test_organ_presence_patterns_empty_input():
    out = organ_presence_patterns(pd.DataFrame(), ["O1", "O2", "O3"], threshold=0.5)
    assert out.empty


def test_split_compatible_nested_and_disjoint_pass_crossing_fails():
    organs = ["O1", "O2", "O3", "O4"]
    # Nested (a real tree can have both as splits: {O1,O2} is a subset of {O1,O2,O3}).
    assert split_compatible(("O1", "O2"), ("O1", "O2", "O3"), organs)
    # Disjoint (also fine -- two unrelated branches).
    assert split_compatible(("O1",), ("O3", "O4"), organs)
    # Crossing: neither contains the other, they overlap, and neither's union covers
    # the full organ set -- no single tree can have both as splits.
    assert not split_compatible(("O1", "O2"), ("O2", "O3"), organs)


def test_four_point_filter_drops_incompatible_pair_keeps_compatible_ones():
    organs = ["O1", "O2", "O3", "O4"]
    significant = [("O1", "O2"), ("O2", "O3"), ("O1", "O3", "O4")]
    # (O1,O2) vs (O2,O3): crossing, incompatible.
    # (O1,O2) vs (O1,O3,O4): neither contains the other, overlap = {O1} nonempty,
    #   union = all 4 organs -> universe - A - B is empty -> compatible.
    # (O2,O3) vs (O1,O3,O4): overlap={O3}, union=all 4 -> compatible.
    kept, bad_pairs = four_point_filter(significant, organs)
    assert ("O1", "O2") not in kept
    assert ("O2", "O3") not in kept
    assert ("O1", "O3", "O4") in kept
    assert (("O1", "O2"), ("O2", "O3")) in bad_pairs


def _pattern_counts(*pattern_and_count: tuple[tuple[str, ...], int]) -> pd.Series:
    """Build a pattern -> count Series the same way the production code does
    (pd.Series(list_of_tuples).value_counts()), NOT via a dict literal with
    tuple keys -- pandas silently (and destructively) builds a padded
    MultiIndex from that instead of a plain Index of tuples, which corrupts
    variable-length tuple keys (a known pandas gotcha, caught here during
    test-writing rather than in the production code path, which never
    constructs a Series this way)."""
    flat: list[tuple[str, ...]] = []
    for pattern, count in pattern_and_count:
        flat.extend([pattern] * count)
    return pd.Series(flat, dtype=object).value_counts()


def test_recurrence_significance_table_flags_real_excess_not_noise():
    subsets = enumerate_nontrivial_subsets(["O1", "O2", "O3", "O4"])
    observed = _pattern_counts((("O1", "O2"), 13), (("O1",), 2), (("O2",), 1))
    null = _pattern_counts((("O1", "O2"), 1), (("O1",), 2), (("O2",), 2))
    table = recurrence_significance_table(observed, null, n_null_sites=20000, n_candidates=3000,
                                           subsets=subsets, alpha=0.05)
    row_o1o2 = table[table["subset"] == ("O1", "O2")].iloc[0]
    assert row_o1o2["significant"]
    row_o1 = table[table["subset"] == ("O1",)].iloc[0]
    assert not row_o1["significant"]
    # Never-seen-in-null subsets must not become spuriously "certain" (p=0) from a
    # single observed occurrence -- the continuity correction guards against this.
    row_unseen = table[table["subset"] == ("O3", "O4")].iloc[0]
    assert row_unseen["observed"] == 0
    assert row_unseen["pvalue"] == 1.0


def test_recurrence_significance_table_continuity_correction_avoids_false_certainty():
    subsets = [("O1", "O2", "O4")]
    observed = _pattern_counts((("O1", "O2", "O4"), 1))
    null = pd.Series(dtype=int)  # this subset was never drawn in the (finite) null sample
    table = recurrence_significance_table(observed, null, n_null_sites=20000, n_candidates=3000,
                                           subsets=subsets, alpha=0.05)
    row = table.iloc[0]
    # A single occurrence should NOT be treated as infinitely significant just
    # because the Monte-Carlo null sample happened not to draw this subset.
    assert row["pvalue"] > 0.01
    assert not row["significant"]


def test_apply_recurrence_boost_never_overrides_already_confident_calls():
    augmented = pd.DataFrame([
        {"chrom": "chr1", "pos": 1, "ref": "A", "alt": "T", "calling_status": "called_fixed_L1"},
        {"chrom": "chr1", "pos": 2, "ref": "A", "alt": "T", "calling_status": "evaluated_unstable"},
        {"chrom": "chr1", "pos": 3, "ref": "A", "alt": "T", "calling_status": "called_noise"},
    ])
    pattern_df = pd.DataFrame([
        {"chrom": "chr1", "pos": 1, "ref": "A", "alt": "T", "organ_pattern": ("O1", "O2")},
        {"chrom": "chr1", "pos": 2, "ref": "A", "alt": "T", "organ_pattern": ("O1", "O2")},
        {"chrom": "chr1", "pos": 3, "ref": "A", "alt": "T", "organ_pattern": ("O3",)},
    ])
    out = apply_recurrence_boost(augmented, pattern_df, validated_subsets=[("O1", "O2")])

    row1 = out[out["pos"] == 1].iloc[0]
    assert row1["calling_status"] == "called_fixed_L1"
    assert row1["recurrence_status"] == "called_fixed_L1"
    assert not row1["recurrence_validated"]  # already confident -- must not be "boosted" on top

    row2 = out[out["pos"] == 2].iloc[0]
    assert row2["recurrence_validated"]
    assert row2["recurrence_status"] == RECURRENCE_VALIDATED_STATUS

    row3 = out[out["pos"] == 3].iloc[0]
    assert not row3["recurrence_validated"]  # pattern not in validated_subsets
    assert row3["recurrence_status"] == "called_noise"

    assert "called_fixed_L1" in ALREADY_CONFIDENT_STATUSES


def test_build_recurrence_report_none_for_fewer_than_three_organs():
    fit = FitResult(mu=MuTriple(L1=0.5, L2=0.0, L3=0.0), lambda_hat=50.0, pi=np.array([0.9, 0.1, 0.0, 0.0]),
                     loglik=-1.0, n_positive_rows=1, n_unique_positive_variants=1, grouped=pd.DataFrame(),
                     active_components=("E", "L1"))
    augmented = pd.DataFrame([{"tree": "t1", "chrom": "chr1", "pos": 1, "ref": "A", "alt": "T",
                                "calling_status": "called_noise"}])
    organ_calls = pd.DataFrame([_organ_calls_row("t1", 1, "O1", 0.0, status="called_noise")])
    variants = pd.DataFrame([{"tree": "t1", "sample": "O1", "depth_filter": 50, "DP": 50, "chrom": "chr1",
                               "pos": 1, "ref": "A", "alt": "T", "alt_depth": 0}])
    report = build_recurrence_report(
        tree="t1", augmented=augmented, organ_calls_window=organ_calls, variants=variants, fit=fit,
        window_depths=[50], organs=["O1", "O2"], q_max=0.5, n_null_sites=100, presence_threshold=0.5,
        alpha=0.05, seed=0, exp_cache=ExponentialComponentCache(q_max=0.5, n_quad=32),
    )
    assert report is None


def test_simulate_null_organ_patterns_low_signal_fit_rarely_calls_fixed():
    """A fit with essentially zero prior mass on any layer component (pi_E~1)
    should classify almost nothing as fixed, so the null pattern distribution
    should be overwhelmingly the empty pattern."""
    fit = FitResult(mu=MuTriple(L1=0.5, L2=0.0, L3=0.0), lambda_hat=60.0,
                     pi=np.array([1.0 - 1e-6, 1e-6, 0.0, 0.0]), loglik=-1.0, n_positive_rows=1,
                     n_unique_positive_variants=1, grouped=pd.DataFrame(), active_components=("E", "L1"))
    exp_cache = ExponentialComponentCache(q_max=0.5, n_quad=64)
    rng = np.random.default_rng(0)
    dp_by_organ = {"O1": np.full(1000, 50), "O2": np.full(1000, 50), "O3": np.full(1000, 50)}
    patterns = simulate_null_organ_patterns(fit, ["O1", "O2", "O3"], dp_by_organ, q_max=0.5, n_null_sites=2000,
                                             threshold=0.5, exp_cache=exp_cache, rng=rng)
    total = patterns.sum()
    empty = patterns.get((), 0)
    assert total == 2000
    assert empty / total > 0.99


if __name__ == "__main__":
    test_enumerate_nontrivial_subsets_count_and_exclusions()
    print("  PASS test_enumerate_nontrivial_subsets_count_and_exclusions")
    test_organ_presence_patterns_threshold_and_missing_organ_handling()
    print("  PASS test_organ_presence_patterns_threshold_and_missing_organ_handling")
    test_organ_presence_patterns_empty_input()
    print("  PASS test_organ_presence_patterns_empty_input")
    test_split_compatible_nested_and_disjoint_pass_crossing_fails()
    print("  PASS test_split_compatible_nested_and_disjoint_pass_crossing_fails")
    test_four_point_filter_drops_incompatible_pair_keeps_compatible_ones()
    print("  PASS test_four_point_filter_drops_incompatible_pair_keeps_compatible_ones")
    test_recurrence_significance_table_flags_real_excess_not_noise()
    print("  PASS test_recurrence_significance_table_flags_real_excess_not_noise")
    test_recurrence_significance_table_continuity_correction_avoids_false_certainty()
    print("  PASS test_recurrence_significance_table_continuity_correction_avoids_false_certainty")
    test_apply_recurrence_boost_never_overrides_already_confident_calls()
    print("  PASS test_apply_recurrence_boost_never_overrides_already_confident_calls")
    test_build_recurrence_report_none_for_fewer_than_three_organs()
    print("  PASS test_build_recurrence_report_none_for_fewer_than_three_organs")
    test_simulate_null_organ_patterns_low_signal_fit_rarely_calls_fixed()
    print("  PASS test_simulate_null_organ_patterns_low_signal_fit_rarely_calls_fixed")
    print("All tests passed.")
